import json
import os
import traceback
from typing import Dict, List, Any, Set, Optional, Tuple
from pathlib import Path

class Multi3DReferConverter:
    def __init__(self, scenegraph_base_path: str = "data/scenegraph"):
        self.scenegraph_base_path = scenegraph_base_path
    
    def load_scene_objects(self, scene_id: str) -> Dict[str, str]:
        """从每个场景独立的objects.json文件加载物体信息"""
        scene_dir = os.path.join(self.scenegraph_base_path, scene_id)
        object_file_path = os.path.join(scene_dir, "objects.json")
        
        try:
            with open(object_file_path, 'r', encoding='utf-8') as f:
                object_data = json.load(f)
            
            if scene_id in object_data:
                return object_data[scene_id].get("inst_to_label", {})
            else:
                return object_data.get("inst_to_label", {})
            
        except FileNotFoundError:
            return {}
        except Exception as e:
            print(f"加载objects.json时出错: {str(e)}")
            return {}
    
    def get_object_name_by_id(self, inst_to_label: Dict[str, str], target_id: int) -> str:
        target_id_str = str(target_id)
        return inst_to_label.get(target_id_str, "object")
    
    def find_same_class_objects(self, inst_to_label: Dict[str, str], target_id: int) -> List[int]:
        target_object_name = self.get_object_name_by_id(inst_to_label, target_id)
        same_class_objects = []
        
        for obj_id_str, label in inst_to_label.items():
            try:
                obj_id = int(obj_id_str)
                if label == target_object_name and obj_id != target_id:
                    same_class_objects.append(obj_id)
            except ValueError:
                continue
        return same_class_objects

    def _create_ann_entry(self, scene_id: str, object_name: str, description: str, 
                          object_ids: List[int], eval_type: str, 
                          true_target_id: int = None) -> Dict:
        entry = {
            'scene_id': scene_id,
            'object_name': object_name,
            'description': description,
            'object_ids': object_ids,
            'eval_type': eval_type,
            'spatial': False,
            'color': False,
            'texture': False,
            'shape': False,
        }
        
        if true_target_id is not None:
            entry['true_target_id'] = true_target_id
            
        return entry
    
    def convert_single_item(self, input_data: Dict, filter_types: Set[str] = None) -> Tuple[List[Dict], List[Dict]]:
        main_entries = []
        restored_entries = []
        
        scene_id = input_data['scene_id']
        target_id = int(input_data['target_id'])
        
        inst_to_label = self.load_scene_objects(scene_id)
        if not inst_to_label:
            return [], []
            
        object_name = self.get_object_name_by_id(inst_to_label, target_id)
        same_class_objects = self.find_same_class_objects(inst_to_label, target_id)
        
        for dialogue in input_data['dialogues']:
            current_type = dialogue.get('ambiguity_type')
            
            if filter_types is not None:
                if current_type not in filter_types:
                    continue

            # 1. 处理原始 Query (Main Data)
            is_ambiguous = dialogue.get('is_ambiguous', False)
            
            # 使用 grounding_matches
            if 'grounding_matches' in dialogue and dialogue['grounding_matches']:
                orig_object_ids = dialogue['grounding_matches']
            else:
                if is_ambiguous:
                    orig_object_ids = [target_id] + same_class_objects
                else:
                    orig_object_ids = [target_id]
            
            if is_ambiguous:
                orig_eval_type = 'mt'
            else:
                orig_eval_type = 'st_w_d'
            
            entry_main = self._create_ann_entry(
                scene_id=scene_id,
                object_name=object_name,
                description=dialogue['input_query'],
                object_ids=orig_object_ids,
                eval_type=orig_eval_type,
                true_target_id=target_id
            )
            main_entries.append(entry_main)
            
            # 2. 处理 Restored Query
            restored_text = dialogue.get('restored_query')
            if restored_text:
                entry_restored = self._create_ann_entry(
                    scene_id=scene_id,
                    object_name=object_name,
                    description=restored_text,
                    object_ids=[target_id],
                    eval_type='st_w_d',
                    true_target_id=target_id
                )
                restored_entries.append(entry_restored)

        return main_entries, restored_entries
    
    # ### 修改处 1: 增加 set_zero_id 参数
    def assign_ids_and_save(self, data_list: List[Dict], file_path: str, set_zero_id: bool = False):
        if not data_list:
            print(f" -> 警告: 数据列表为空，跳过保存 {file_path}")
            return

        for i, entry in enumerate(data_list):
            # 如果是 Train 模式，强制设为 int 0；否则使用唯一 ID
            if set_zero_id:
                entry['ann_id'] = 0
            else:
                entry['ann_id'] = f"{i:06d}"
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
            
        print(f"  -> 已保存: {file_path}")
        print(f"     数量: {len(data_list)} 条")
        if set_zero_id:
            print(f"     ID设置: 全部为 int 0 (Training Mode)")
        else:
            print(f"     ID范围: 000000 - {len(data_list)-1:06d}")

    # ### 修改处 2: 传递 set_zero_id 参数
    def batch_convert(self, input_file_path: str, output_file_main: str, output_file_restored: str, 
                      target_scenes: List[str] = None, split_restored: bool = False,
                      filter_types: List[str] = None, set_zero_id: bool = False):
        
        target_scene_set = set(target_scenes) if target_scenes else None
        filter_type_set = set(filter_types) if filter_types else None

        try:
            print(f"读取输入文件: {input_file_path} ...")
            if filter_type_set:
                print(f">>> 正在根据 ambiguity_type 过滤: {filter_type_set}")
            
            with open(input_file_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            all_main_data = []
            all_restored_data = []
            
            skipped_scenes = set()
            filtered_count = 0 
            
            data_items = input_data if isinstance(input_data, list) else [input_data]
            
            print(f"源数据包含 {len(data_items)} 个 Item (Object Level)")
            
            for item in data_items:
                scene_id = item['scene_id']
                
                if target_scene_set is not None:
                    if scene_id not in target_scene_set:
                        filtered_count += 1
                        continue
                
                if not self.validate_scene_structure(scene_id):
                    skipped_scenes.add(scene_id)
                    continue
                
                m_list, r_list = self.convert_single_item(item, filter_types=filter_type_set)
                
                all_main_data.extend(m_list)
                if split_restored:
                    all_restored_data.extend(r_list)
            
            print("\n" + "="*30)
            print("正在保存文件...")
            
            print(f"[主文件]")
            # 传递 set_zero_id
            self.assign_ids_and_save(all_main_data, output_file_main, set_zero_id=set_zero_id)
            
            if split_restored:
                print(f"\n[Restored文件]")
                if all_restored_data:
                    # 也可以选择对 restored 数据也使用 zero id，这里假设同步
                    self.assign_ids_and_save(all_restored_data, output_file_restored, set_zero_id=set_zero_id)
                else:
                    print("  -> 警告：没有生成任何 restored 数据")

            if target_scene_set:
                print(f"\n过滤掉的非目标场景条目数: {filtered_count}")
            if skipped_scenes:
                print(f"因缺少场景文件而跳过的场景数: {len(skipped_scenes)}")
            print("="*30 + "\n")
            
        except Exception as e:
            traceback.print_exc()
            print(f"转换过程中出现错误: {str(e)}")
            
    def validate_scene_structure(self, scene_id: str) -> bool:
        scene_dir = os.path.join(self.scenegraph_base_path, scene_id)
        object_file_path = os.path.join(scene_dir, "objects.json")
        return os.path.exists(scene_dir) and os.path.exists(object_file_path)

def main():
    """主函数"""
    
    input_file = 'data/result/result_train.json' # 假设这是 train
    base_output_dir = 'data/result'
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return

    print(f"准备转换文件: {input_file}")
    
    # 1. 场景过滤
    target_scenes = None
    use_filter = input("Step 1: 是否只转换特定 Scene ID？(y/n): ").strip().lower()
    
    if use_filter == 'y':
        scenes_input = input("请输入场景ID (用逗号分隔): ")
        target_scenes = [s.strip() for s in scenes_input.split(',') if s.strip()]
        if not target_scenes:
            target_scenes = None

    # 2. Ambiguity Type 过滤
    target_types = None
    type_suffix = "" 
    
    print("\nStep 2: 选择 Ambiguity Type 过滤器")
    print("支持类型: no_visual, no_spatial, minimal")
    print("输入 'all' 或直接回车表示保留所有类型")
    types_input = input("请输入类型: ").strip().lower()
    
    if types_input and types_input != 'all':
        raw_types = [t.strip() for t in types_input.split(',') if t.strip()]
        valid_types = {'no_visual', 'no_spatial', 'minimal'}
        selected_types = [t for t in raw_types if t in valid_types]
        
        if selected_types:
            target_types = selected_types
            type_suffix = "_" + "_".join(selected_types)

    # 3. Restored Query
    split_restored = False
    print(f"\nStep 3: Restored Query 设置")
    use_restored = input("是否提取 restored_query 并存为独立文件？(y/n): ").strip().lower()

    if use_restored == 'y':
        split_restored = True
    
    # ### 修改处 3: 询问是否是 Train 模式 (set ann_id = 0)
    print(f"\nStep 4: Training Mode 设置")
    is_train_input = input("这是用于 Training 的数据吗？(ann_id 将被设为 int 0) (y/n): ").strip().lower()
    set_zero_id = (is_train_input == 'y')
    
    base_name = f"result_m3dref_train{type_suffix}.json" # 示例文件名
    output_file_main = os.path.join(base_output_dir, base_name)
    output_file_restored = ""

    if split_restored:
        path_obj = Path(output_file_main)
        output_file_restored = str(path_obj.parent / f"{path_obj.stem}_restored{path_obj.suffix}")
        print(f"Restored 文件将保存为: {output_file_restored}")
    
    print(f"主文件将保存为: {output_file_main}")
    if set_zero_id:
        print("注意: ann_id 将被统一设为整数 0")
    print("-" * 30)

    converter = Multi3DReferConverter(scenegraph_base_path="data/scenegraph")
    converter.batch_convert(input_file, output_file_main, output_file_restored,
                           target_scenes=target_scenes, 
                           split_restored=split_restored,
                           filter_types=target_types,
                           set_zero_id=set_zero_id) # 传入参数

if __name__ == "__main__":
    main()