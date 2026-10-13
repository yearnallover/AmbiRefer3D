import json
import os
from typing import Dict, List, Tuple

class MinimalQueryConverter:
    def __init__(self):
        # 因为输入数据里已经包含了 label 和 ids，
        # 所以这个转换器不需要加载 data/scenegraph 下的文件，非常快
        pass

    def _create_ann_entry(self, scene_id: str, object_name: str, description: str, 
                          object_ids: List[int], eval_type: str) -> Dict:
        """
        构建 Multi3DRefer 标准格式字典
        """
        return {
            'scene_id': scene_id,
            'object_name': object_name,
            'ann_id': "",  # 后续统一填充
            'description': description,
            'object_ids': object_ids,
            'eval_type': eval_type,
            # 以下为 Multi3DRefer 默认字段
            'spatial': False,
            'color': False, 
            'texture': False,
            'shape': False,
        }
    
    def convert_single_item(self, input_data: Dict) -> Dict:
        """
        处理单条 extracted_candidate 数据
        输入示例: 
        {
            "scene_id": "scene0000_00",
            "target_label": "table",
            "minimal_query": "a curtain and a table.",
            "candidate_ids": [2, 35]
        }
        """
        scene_id = input_data['scene_id']
        # 注意字段名映射：上一步生成的 key 是 'target_label' 和 'minimal_query'
        object_name = input_data['target_label']
        description = input_data['minimal_query']
        object_ids = input_data['candidate_ids']
        
        # === 核心逻辑：判定 eval_type ===
        # 如果有多个候选物，属于 Multi-Target (mt)
        # 如果只有一个，属于 Single-Target (st_w_d)
        if len(object_ids) > 1:
            eval_type = 'mt'
        else:
            eval_type = 'st_w_d'
            
        return self._create_ann_entry(
            scene_id=scene_id,
            object_name=object_name,
            description=description,
            object_ids=object_ids,
            eval_type=eval_type
        )

    def assign_ids_and_save(self, data_list: List[Dict], file_path: str):
        """统一分配 000000 格式的 ID 并保存"""
        if not data_list:
            print("数据为空，未保存。")
            return

        print(f"正在为 {len(data_list)} 条数据分配 ID...")
        for i, entry in enumerate(data_list):
            entry['ann_id'] = f"{i:06d}"
            
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
            
        print(f" -> 已保存: {file_path}")
        print(f" -> ID范围: 000000 - {len(data_list)-1:06d}")

    def process(self, input_file: str, output_file: str):
        if not os.path.exists(input_file):
            print(f"错误: 找不到输入文件 {input_file}")
            return

        print(f"读取输入: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        converted_list = []
        
        for item in raw_data:
            # 转换单条
            entry = self.convert_single_item(item)
            converted_list.append(entry)
            
        # 保存
        self.assign_ids_and_save(converted_list, output_file)

# ==========================================
# 主程序入口
# ==========================================
def main():
    # 1. 输入文件：上一步生成的候选物文件
    input_path = "data/stage_1_gt/extracted_candidates.json"
    
    # 2. 输出文件：符合 Multi3DRefer 格式的文件
    output_path = "data/stage_1_gt/minimal_multi3drefer.json"
    
    converter = MinimalQueryConverter()
    converter.process(input_path, output_path)

if __name__ == "__main__":
    main()