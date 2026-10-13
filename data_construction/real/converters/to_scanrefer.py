import json
import re
import os
import argparse
from typing import List, Dict, Any, Set, Tuple

def get_object_label(scene_id: str, object_id: int, objects_data_dir: str = "data/scenegraph") -> str:
    """从场景的objects.json文件中获取物体标签 (保持不变)"""
    objects_file_path = os.path.join(objects_data_dir, scene_id, "objects.json")
    
    if not os.path.exists(objects_file_path):
        # 降噪，仅在调试时取消注释
        # print(f"警告: 未找到objects.json文件: {objects_file_path}")
        return "object"
    
    try:
        with open(objects_file_path, 'r', encoding='utf-8') as f:
            objects_data = json.load(f)
        
        if scene_id in objects_data and "objects_info" in objects_data[scene_id]:
            for obj in objects_data[scene_id]["objects_info"]:
                obj_id = obj.get("id") or obj.get("object_id") or obj.get("objectId")
                if obj_id is not None and str(obj_id) == str(object_id):
                    label = obj.get("label") or obj.get("name") or obj.get("category")
                    if label:
                        return label
        return "object"
    except Exception:
        return "object"

def tokenize_text(text: str) -> List[str]:
    """将文本分词 (保持不变)"""
    if not text:
        return []
    text_lower = text.lower()
    tokens = re.findall(r"\b[\w'-]+\b|[.,!?;:()]", text_lower)
    return tokens

def convert_scene_data(input_data: List[Dict[str, Any]], 
                      query_type: str = "input", 
                      target_ambiguity_type: str = "all",
                      objects_data_dir: str = "data/scenegraph") -> List[Dict[str, Any]]:
    """
    转换场景数据格式，增加ambiguity_type过滤
    
    参数:
    input_data: 原始数据列表
    query_type: 使用哪种查询文本 ("input", "restored", "gt")
    target_ambiguity_type: 筛选哪种模糊类型 ("all", "no_spatial", "no_visual", "minimal")
    objects_data_dir: objects.json路径
    """
    result = []
    seen_entries = set() # 用于GT模式去重
    
    for scene_idx, scene_data in enumerate(input_data):
        scene_id = scene_data.get("scene_id", f"scene_{scene_idx:04d}")
        target_id = scene_data.get("target_id", "")
        
        # 1. 获取物体名称
        if target_id and objects_data_dir:
            object_name = get_object_label(scene_id, target_id, objects_data_dir)
        else:
            object_name = "object" # 简化处理，若需要正则提取可保留原逻辑
        
        # 2. 获取对话列表
        dialogues = scene_data.get("dialogues", [])
        
        # --- GT模式特殊处理 (GT是场景级的，不依赖dialogue loop) ---
        if query_type == "gt":
            query_text = scene_data.get("gt_description", "")
            if query_text:
                entry_key = (scene_id, str(target_id), query_text)
                if entry_key not in seen_entries:
                    tokens = tokenize_text(query_text)
                    entry = {
                        "scene_id": scene_id,
                        "object_id": str(target_id),
                        "object_name": object_name,
                        "ann_id": "1",
                        "description": query_text,
                        "token": tokens,
                        "ambiguity_type": "gt" # 标记为gt
                    }
                    result.append(entry)
                    seen_entries.add(entry_key)
            continue # GT模式处理完直接跳过后续dialogue循环
            
        # --- 普通模式 (Input/Restored) ---
        for dialogue_idx, dialogue in enumerate(dialogues, 1):
            
            # [关键修改] 过滤 Ambiguity Type
            current_ambiguity = dialogue.get("ambiguity_type", "unknown")
            
            if target_ambiguity_type != "all":
                # 如果指定了类型，且当前对话类型不匹配，则跳过
                if current_ambiguity != target_ambiguity_type:
                    continue
            
            # 确定查询文本
            if query_type == "restored":
                query_text = dialogue.get("restored_query", "")
                if not query_text:
                    query_text = dialogue.get("input_query", "")
            else: # default input
                query_text = dialogue.get("input_query", "")
            
            if not query_text:
                continue

            tokens = tokenize_text(query_text)
            
            entry = {
                "scene_id": scene_id,
                "object_id": str(target_id),
                "object_name": object_name,
                # ann_id 建议包含原始索引以方便回溯，或者重新编号
                "ann_id": str(dialogue_idx), 
                "description": query_text,
                "token": tokens,
                # 保留类型信息以便查看
                "ambiguity_type": current_ambiguity 
            }
            
            result.append(entry)
    
    return result

def process_json_file(input_file_path: str, output_file_path: str = None, 
                      query_type: str = "input", 
                      target_ambiguity_type: str = "all",
                      objects_data_dir: str = "data/scenegraph") -> None:
    """处理单个JSON文件"""
    if not os.path.exists(input_file_path):
        print(f"错误: 文件 {input_file_path} 不存在")
        return
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"错误: 读取文件失败 - {e}")
        return
    
    print(f"正在处理: {os.path.basename(input_file_path)}")
    print(f"配置: Query={query_type}, Ambiguity Filter={target_ambiguity_type}")
    
    # 转换数据
    converted_data = convert_scene_data(input_data, query_type, target_ambiguity_type, objects_data_dir)
    
    print(f"转换后条目数: {len(converted_data)}")
    
    # 自动生成输出路径
    if output_file_path is None:
        file_dir, file_name = os.path.split(input_file_path)
        file_base, file_ext = os.path.splitext(file_name)
        
        # 构建后缀: 例如 _no_spatial_input
        ambiguity_suffix = f"_{target_ambiguity_type}" if target_ambiguity_type != "all" else ""
        query_suffix = f"_{query_type}"
        
        new_filename = f"{file_base}{ambiguity_suffix}{query_suffix}{file_ext}"
        output_file_path = os.path.join(file_dir, new_filename)
    
    # 保存
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {output_file_path}\n")
    except Exception as e:
        print(f"错误: 保存失败 - {e}")

def batch_process_directory(input_dir: str, output_dir: str = None, 
                          query_type: str = "input", 
                          target_ambiguity_type: str = "all",
                          objects_data_dir: str = "data/scenegraph") -> None:
    """批量处理"""
    import glob
    if not os.path.exists(input_dir):
        print(f"错误: 目录不存在 {input_dir}")
        return
        
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not json_files:
        print("未找到JSON文件")
        return
        
    if output_dir is None:
        output_dir = input_dir
        
    for input_file in json_files:
        # 简单构建输出文件名，具体逻辑在process_json_file中也有，这里为了显式指定output_dir
        file_name = os.path.basename(input_file)
        file_base, file_ext = os.path.splitext(file_name)
        
        ambiguity_suffix = f"_{target_ambiguity_type}" if target_ambiguity_type != "all" else ""
        query_suffix = f"_{query_type}"
        
        new_filename = f"{file_base}{ambiguity_suffix}{query_suffix}{file_ext}"
        output_file = os.path.join(output_dir, new_filename)
        
        process_json_file(input_file, output_file, query_type, target_ambiguity_type, objects_data_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换场景数据格式 (支持Ambiguity过滤)")
    
    parser.add_argument("-input", default="data/result/result_val.json", help="输入文件或目录")
    parser.add_argument("-o", "--output", default=None, help="输出文件或目录 (可选)")
    parser.add_argument("-b", "--batch", action="store_true", help="是否批量处理目录")
    parser.add_argument("--objects_dir", default="data/scenegraph", help="objects.json目录")
    
    #原有参数
    parser.add_argument("-q", "--query_type", 
                        choices=["input", "restored", "gt"], 
                        default="input",
                        help="选择文本源: input(默认), restored, gt")
    
    # [新增参数] 
    parser.add_argument("-a", "--ambiguity_type", 
                        choices=["all", "no_spatial", "no_visual", "minimal"], 
                        default="all",
                        help="筛选Ambiguity类型: no_spatial, no_visual, minimal, 或 all (默认)")

    args = parser.parse_args()
    
    if args.batch:
        batch_process_directory(args.input, args.output, 
                              query_type=args.query_type, 
                              target_ambiguity_type=args.ambiguity_type,
                              objects_data_dir=args.objects_dir)
    else:
        process_json_file(args.input, args.output, 
                        query_type=args.query_type, 
                        target_ambiguity_type=args.ambiguity_type,
                        objects_data_dir=args.objects_dir)