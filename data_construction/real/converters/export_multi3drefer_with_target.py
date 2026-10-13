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

    # ### 修改处 1: 增加了 generated_question 参数 ###
    def _create_ann_entry(self, scene_id: str, object_name: str, description: str, 
                          object_ids: List[int], eval_type: str, 
                          true_target_id: int = None,
                          generated_question: str = None) -> Dict: # <--- 新增参数
        entry = {
            'scene_id': scene_id,
            'object_name': object_name,
            'description': description,
            'object_ids': object_ids,     # 这是一个列表，模糊查询时包含多个ID
            'eval_type': eval_type,
            'spatial': False,
            'color': False,
            'texture': False,
            'shape': False,
        }
        
        # 写入真实的单一目标ID
        if true_target_id is not None:
            entry['true_target_id'] = true_target_id
            
        # ### 新增：如果存在反问句，写入到转换后的数据中 ###
        if generated_question is not None:
            entry['generated_question'] = generated_question
            
        return entry
    
    def convert_single_item(self, input_data: Dict, filter_types: Set[str] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        返回两个列表: (主数据列表, Restored数据列表)
        """
        main_entries = []
        restored_entries = []
        
        scene_id = input_data['scene_id']
        # 获取真实的单一目标 ID
        target_id = int(input_data['target_id'])
        
        inst_to_label = self.load_scene_objects(scene_id)
        if not inst_to_label:
            return [], []
            
        object_name = self.get_object_name_by_id(inst_to_label, target_id)
        same_class_objects = self.find_same_class_objects(inst_to_label, target_id)
        
        for dialogue in input_data['dialogues']:
            current_type = dialogue.get('ambiguity_type')
            
            # 过滤器逻辑
            if filter_types is not None:
                if current_type not in filter_types:
                    continue

            # 1. 处理原始 Query (Main Data - 模糊/有歧义的数据)
            is_ambiguous = dialogue.get('is_ambiguous', False)
            
            # ### 修改处 2: 提取并传递 generated_question ###
            # 安全地从字典中获取反问句，如果没有则为 None
            gen_question = dialogue.get('generated_question') 

            # 确定物体列表
            if 'grounding_matches' in dialogue and dialogue['grounding_matches']:
                orig_object_ids = dialogue['grounding_matches']
            else:
                if is_ambiguous:
                    orig_object_ids = [target_id] + same_class_objects
                else:
                    orig_object_ids = [target_id]
            
            # 确定 eval_type
            if is_ambiguous:
                orig_eval_type = 'mt' # multiple targets (ambiguous)
            else:
                orig_eval_type = 'st_w_d' # single target with distractors
            
            # 创建 Main Entry
            entry_main = self._create_ann_entry(
                scene_id=scene_id,
                object_name=object_name,
                description=dialogue['input_query'],
                object_ids=orig_object_ids,
                eval_type=orig_eval_type,
                true_target_id=target_id,
                generated_question=gen_question # <--- 关键：将反问句传入
            )
            main_entries.append(entry_main)
            
            # 2. 处理 Restored Query (Restored Data - 消除歧义后的数据)
            restored_text = dialogue.get('restored_query')
            if restored_text:
                entry_restored = self._create_ann_entry(
                    scene_id=scene_id,
                    object_name=object_name,
                    description=restored_text,
                    object_ids=[target_id],       # 消除歧义后，目标明确，只有 [27]
                    eval_type='st_w_d',
                    true_target_id=target_id      # Restored 数据通常不需要反问句，所以没传
                )
                restored_entries.append(entry_restored)

        return main_entries, restored_entries
    
    def assign_ids_and_save(self, data_list: List[Dict], file_path: str):
        if not data_list:
            print(f" -> 警告: 数据列表为空，跳过保存 {file_path}")
            return

        for i, entry in enumerate(data_list):
            entry['ann_id'] = f"{i:06d}"
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
            
        print(f"  -> 已保存: {file_path}")
        print(f"     数量: {len(data_list)} 条")
        print(f"     ID范围: 000000 - {len(data_list)-1:06d}")

    def batch_convert(self, input_file_path: str, output_file_main: str, output_file_restored: str, 
                      target_scenes: List[str] = None, split_restored: bool = False,
                      filter_types: List[str] = None):
        
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
            self.assign_ids_and_save(all_main_data, output_file_main)
            
            if split_restored:
                print(f"\n[Restored文件]")
                if all_restored_data:
                    self.assign_ids_and_save(all_restored_data, output_file_restored)
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
    
    input_file = 'data/result/result_val.json'
    base_output_dir = 'data/result'
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return

    print(f"准备转换文件: {input_file}")
    
    # 1. 场景过滤交互
    target_scenes = None
    use_filter = input("Step 1: 是否只转换特定 Scene ID？(y/n): ").strip().lower()
    
    if use_filter == 'y':
        scenes_input = input("请输入场景ID (用逗号分隔): ")
        target_scenes = [s.strip() for s in scenes_input.split(',') if s.strip()]
        if not target_scenes:
            target_scenes = None

    # 2. Ambiguity Type 过滤交互
    target_types = None
    type_suffix = "" 
    
    print("\nStep 2: 选择 Ambiguity Type 过滤器")
    print("支持类型: no_visual, no_spatial, minimal")
    print("输入 'all' 或直接回车表示保留所有类型")
    print("输入多个类型请用逗号分隔 (例如: no_visual, minimal)")
    
    types_input = input("请输入类型: ").strip().lower()
    
    if types_input and types_input != 'all':
        raw_types = [t.strip() for t in types_input.split(',') if t.strip()]
        valid_types = {'no_visual', 'no_spatial', 'minimal'}
        selected_types = [t for t in raw_types if t in valid_types]
        
        if selected_types:
            target_types = selected_types
            type_suffix = "_" + "_".join(selected_types)
        else:
            print("警告: 未识别到有效类型，将使用默认值(全部转换)")

    # 3. Restored Query 分离交互
    split_restored = False
    
    print(f"\nStep 3: Restored Query 设置")
    use_restored = input("是否提取 restored_query 并存为独立文件？(y/n): ").strip().lower()
    
    base_name = f"result_m3dref_val{type_suffix}_t.json"
    output_file_main = os.path.join(base_output_dir, base_name)
    output_file_restored = ""

    if use_restored == 'y':
        split_restored = True
        path_obj = Path(output_file_main)
        output_file_restored = str(path_obj.parent / f"{path_obj.stem}_restored{path_obj.suffix}")
        print(f"Restored 文件将保存为: {output_file_restored}")
    
    print(f"主文件将保存为: {output_file_main}")
    print("-" * 30)

    converter = Multi3DReferConverter(scenegraph_base_path="data/scenegraph")
    converter.batch_convert(input_file, output_file_main, output_file_restored,
                           target_scenes=target_scenes, 
                           split_restored=split_restored,
                           filter_types=target_types)

if __name__ == "__main__":
    main()