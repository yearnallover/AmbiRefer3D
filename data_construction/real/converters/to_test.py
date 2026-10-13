import json
import re
import os
import argparse
import random
from typing import List, Dict, Any, Set

def load_scene_list(scene_list_file: str) -> Set[str]:
    """
    从文件中加载场景ID列表
    
    参数:
    scene_list_file: 包含场景ID列表的文件路径
    
    返回:
    场景ID的集合
    """
    scene_set = set()
    
    if not os.path.exists(scene_list_file):
        print(f"警告: 场景列表文件 {scene_list_file} 不存在")
        return scene_set
    
    try:
        with open(scene_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                scene_id = line.strip()
                if scene_id:  # 跳过空行
                    scene_set.add(scene_id)
        
        print(f"从 {scene_list_file} 加载了 {len(scene_set)} 个场景ID")
        return scene_set
    
    except Exception as e:
        print(f"错误: 读取场景列表文件 {scene_list_file} 失败: {e}")
        return scene_set

def get_object_label(scene_id: str, object_id: int, objects_data_dir: str = "data/scenegraph") -> str:
    """
    从场景的objects.json文件中获取物体标签
    
    参数:
    scene_id: 场景ID
    object_id: 物体ID
    objects_data_dir: objects.json文件的根目录
    
    返回:
    物体标签，如果未找到则返回"object"
    """
    # 构建objects.json文件路径
    objects_file_path = os.path.join(objects_data_dir, scene_id, "objects.json")
    
    if not os.path.exists(objects_file_path):
        print(f"警告: 未找到objects.json文件: {objects_file_path}")
        return "object"
    
    try:
        # 读取objects.json文件
        with open(objects_file_path, 'r', encoding='utf-8') as f:
            objects_data = json.load(f)
        
        # 根据新的数据结构获取场景对象
        if scene_id in objects_data and "objects_info" in objects_data[scene_id]:
            # 查找对应的物体
            for obj in objects_data[scene_id]["objects_info"]:
                # 尝试从多个可能的字段中获取物体ID
                obj_id = obj.get("id") or obj.get("object_id") or obj.get("objectId")
                if obj_id is not None and str(obj_id) == str(object_id):
                    # 尝试从多个可能的字段中获取标签
                    label = obj.get("label") or obj.get("name") or obj.get("category")
                    if label:
                        return label
        
        print(f"警告: 场景 {scene_id} 中未找到ID为 {object_id} 的物体")
        return "object"
    
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"错误: 读取objects.json文件 {objects_file_path} 失败: {e}")
        return "object"
    except Exception as e:
        print(f"错误: 处理objects.json文件 {objects_file_path} 时发生未知错误: {e}")
        return "object"

def tokenize_text(text: str) -> List[str]:
    """
    将文本分词
    
    参数:
    text: 需要分词的文本
    
    返回:
    分词后的token列表
    """
    if not text:
        return []
    
    # 转换为小写
    text_lower = text.lower()
    
    # 使用正则表达式分词，保留标点符号
    # 匹配单词（包括连字符、撇号）和标点符号
    tokens = re.findall(r"\b[\w'-]+\b|[.,!?;:()]", text_lower)
    
    return tokens

def filter_by_scene_list(data: List[Dict[str, Any]], allowed_scenes: Set[str]) -> List[Dict[str, Any]]:
    """
    根据允许的场景列表过滤数据
    
    参数:
    data: 原始数据列表
    allowed_scenes: 允许的场景ID集合
    
    返回:
    过滤后的数据列表
    """
    if not allowed_scenes:
        return data
    
    filtered_data = []
    filtered_out_count = 0
    
    for scene_data in data:
        scene_id = scene_data.get("scene_id", "")
        if scene_id in allowed_scenes:
            filtered_data.append(scene_data)
        else:
            filtered_out_count += 1
    
    print(f"场景过滤: 从 {len(data)} 个场景中过滤出 {len(filtered_data)} 个场景，过滤掉 {filtered_out_count} 个场景")
    return filtered_data

def convert_scene_data(input_data: List[Dict[str, Any]], query_type: str = "input", 
                      objects_data_dir: str = "data/scenegraph", 
                      keep_all_fields: bool = False) -> List[Dict[str, Any]]:
    """
    转换场景数据格式
    
    参数:
    input_data: 原始数据列表
    query_type: 使用哪种查询类型，可选值: "input", "restored", "gt"
    objects_data_dir: objects.json文件的根目录
    keep_all_fields: 是否保留所有字段（True）或只保留scene_id和query（False）
    
    返回:
    转换后的数据列表
    """
    result = []
    
    for scene_idx, scene_data in enumerate(input_data):
        scene_id = scene_data.get("scene_id", f"scene_{scene_idx:04d}")
        target_id = scene_data.get("target_id", "")
        
        # 从objects.json文件中获取物体标签
        if target_id and objects_data_dir:
            object_name = get_object_label(scene_id, target_id, objects_data_dir)
        else:
            # 如果没有target_id或objects_data_dir，尝试从描述中提取
            gt_description = scene_data.get("gt_description", "")
            # 简单的提取方法
            if gt_description:
                description_lower = gt_description.lower()
                patterns = [
                    r'a\s+([a-zA-Z\s]+?)\s+(?:in|with|near|on|at|\.|$)',
                    r'the\s+([a-zA-Z\s]+?)\s+(?:in|with|near|on|at|\.|$)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, description_lower)
                    if match:
                        object_name = match.group(1).strip()
                        object_name = re.sub(r'\s+', ' ', object_name)
                        if object_name:
                            break
                else:
                    object_name = "object"
            else:
                object_name = "object"
        
        # 处理每个对话
        dialogues = scene_data.get("dialogues", [])
        
        for dialogue_idx, dialogue in enumerate(dialogues, 1):
            # 根据参数选择使用哪种查询
            if query_type == "gt":
                # 使用gt_description
                query_text = scene_data.get("gt_description", "")
            elif query_type == "restored":
                # 尝试使用restored_query，如果不存在则回退到input_query
                query_text = dialogue.get("restored_query", "")
                if not query_text:
                    query_text = dialogue.get("input_query", "")
                    print(f"警告: 场景 {scene_id} 对话 {dialogue_idx} 的restored_query为空，使用input_query替代")
            else:  # 默认使用input_query
                query_text = dialogue.get("input_query", "")
            
            # 分词处理
            tokens = tokenize_text(query_text)
            
            if keep_all_fields:
                # 构建完整的结果条目
                entry = {
                    "scene_id": scene_id,
                    "object_id": str(target_id),
                    "object_name": object_name,
                    "ann_id": str(dialogue_idx),  # 从1开始计数
                    "description": query_text,
                    "token": tokens
                }
            else:
                # 构建简化的结果条目，只包含scene_id和query
                entry = {
                    "scene_id": scene_id,
                    "query": query_text
                }
            
            result.append(entry)
    
    return result

def sample_data(data: List[Dict[str, Any]], sample_size: int = 0, seed: int = 42) -> List[Dict[str, Any]]:
    """
    随机抽样数据
    
    参数:
    data: 原始数据列表
    sample_size: 抽样数量，0表示不抽样
    seed: 随机种子，用于可重复性
    
    返回:
    抽样后的数据列表
    """
    if sample_size <= 0:
        return data
    
    if sample_size >= len(data):
        print(f"警告: 抽样数量 {sample_size} 大于等于数据总数 {len(data)}，返回所有数据")
        return data
    
    # 设置随机种子以确保可重复性
    random.seed(seed)
    
    # 随机抽样
    sampled_data = random.sample(data, sample_size)
    
    print(f"从 {len(data)} 条数据中随机抽取了 {sample_size} 条数据 (随机种子: {seed})")
    
    return sampled_data

def process_json_file(input_file_path: str, output_file_path: str = None, 
                     query_type: str = "input", objects_data_dir: str = "data/scenegraph",
                     keep_all_fields: bool = False, sample_size: int = 0, seed: int = 42,
                     scene_list_file: str = None) -> None:
    """
    处理JSON文件：读取、转换、保存
    
    参数:
    input_file_path: 输入JSON文件路径
    output_file_path: 输出JSON文件路径（如果为None，则自动生成）
    query_type: 使用哪种查询类型，可选值: "input", "restored", "gt"
    objects_data_dir: objects.json文件的根目录
    keep_all_fields: 是否保留所有字段
    sample_size: 抽样数量，0表示不抽样
    seed: 随机种子，用于可重复性
    scene_list_file: 场景列表文件路径，包含允许的场景ID
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        print(f"错误: 文件 {input_file_path} 不存在")
        return
    
    # 加载场景列表
    allowed_scenes = set()
    if scene_list_file:
        allowed_scenes = load_scene_list(scene_list_file)
        if not allowed_scenes:
            print(f"警告: 场景列表为空，将处理所有场景")
    
    # 读取JSON文件
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件解析失败 - {e}")
        return
    except Exception as e:
        print(f"错误: 读取文件失败 - {e}")
        return
    
    print(f"成功读取文件: {input_file_path}")
    print(f"原始数据条数: {len(input_data)}")
    print(f"使用objects.json文件目录: {objects_data_dir}")
    print(f"输出格式: {'完整字段' if keep_all_fields else '简化字段(scene_id和query)'}")
    
    # 根据场景列表过滤数据
    if allowed_scenes:
        input_data = filter_by_scene_list(input_data, allowed_scenes)
        if not input_data:
            print(f"错误: 过滤后没有数据，请检查场景列表文件")
            return
    
    # 转换数据
    converted_data = convert_scene_data(input_data, query_type, objects_data_dir, keep_all_fields)
    
    print(f"转换后数据条数: {len(converted_data)}")
    
    # 随机抽样
    if sample_size > 0:
        converted_data = sample_data(converted_data, sample_size, seed)
    
    # 根据查询类型显示描述
    query_type_names = {
        "input": "input_query",
        "restored": "restored_query",
        "gt": "gt_description"
    }
    query_name = query_type_names.get(query_type, query_type)
    print(f"使用 {query_name} 作为查询文本")
    
    # 生成输出文件路径（如果未指定）
    if output_file_path is None:
        file_dir, file_name = os.path.split(input_file_path)
        file_base, file_ext = os.path.splitext(file_name)
        suffix_map = {
            "input": "_input",
            "restored": "_restored",
            "gt": "_gt"
        }
        suffix = suffix_map.get(query_type, "")
        format_suffix = "_full" if keep_all_fields else "_simple"
        sample_suffix = f"_sample{sample_size}" if sample_size > 0 else ""
        scene_suffix = "_filtered" if scene_list_file else ""
        output_file_path = os.path.join(file_dir, f"{file_base}{suffix}{format_suffix}{sample_suffix}{scene_suffix}{file_ext}")
    
    # 保存转换后的数据
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        print(f"转换完成！结果已保存到: {output_file_path}")
        
        # 打印前几条数据作为示例
        print(f"\n前5条转换后的数据示例:")
        for i, item in enumerate(converted_data[:5]):
            if keep_all_fields:
                print(f"{i+1}. 场景: {item['scene_id']}, 物体ID: {item['object_id']}, 物体名称: {item['object_name']}")
                print(f"   {query_name}: {item['description'][:80]}...")
            else:
                print(f"{i+1}. 场景: {item['scene_id']}")
                print(f"   查询: {item['query'][:80]}...")
        
    except Exception as e:
        print(f"错误: 保存文件失败 - {e}")

def batch_process_directory(input_dir: str, output_dir: str = None, 
                          file_pattern: str = "*.json", query_type: str = "input",
                          objects_data_dir: str = "data/scenegraph",
                          keep_all_fields: bool = False, sample_size: int = 0, seed: int = 42,
                          scene_list_file: str = None) -> None:
    """
    批量处理目录中的所有JSON文件
    
    参数:
    input_dir: 输入目录路径
    output_dir: 输出目录路径（如果为None，则使用输入目录）
    file_pattern: 文件匹配模式
    query_type: 使用哪种查询类型，可选值: "input", "restored", "gt"
    objects_data_dir: objects.json文件的根目录
    keep_all_fields: 是否保留所有字段
    sample_size: 抽样数量，0表示不抽样
    seed: 随机种子，用于可重复性
    scene_list_file: 场景列表文件路径，包含允许的场景ID
    """
    import glob
    
    if not os.path.exists(input_dir):
        print(f"错误: 目录 {input_dir} 不存在")
        return
    
    if output_dir is None:
        output_dir = input_dir
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载场景列表
    allowed_scenes = set()
    if scene_list_file:
        allowed_scenes = load_scene_list(scene_list_file)
        if not allowed_scenes:
            print(f"警告: 场景列表为空，将处理所有场景")
    
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not json_files:
        print(f"在目录 {input_dir} 中未找到JSON文件")
        return
    
    query_type_names = {
        "input": "input_query",
        "restored": "restored_query",
        "gt": "gt_description"
    }
    query_name = query_type_names.get(query_type, query_type)
    
    print(f"找到 {len(json_files)} 个JSON文件，开始批量处理...")
    print(f"使用 {query_name} 作为查询文本")
    print(f"输出格式: {'完整字段' if keep_all_fields else '简化字段(scene_id和query)'}")
    if sample_size > 0:
        print(f"随机抽样: 每个文件抽取 {sample_size} 条数据 (随机种子: {seed})")
    if scene_list_file:
        print(f"场景过滤: 只处理 {len(allowed_scenes)} 个指定场景")
    print(f"使用objects.json文件目录: {objects_data_dir}")
    
    for i, input_file in enumerate(json_files, 1):
        print(f"\n处理文件 {i}/{len(json_files)}: {os.path.basename(input_file)}")
        
        # 生成输出文件路径
        file_name = os.path.basename(input_file)
        file_base, file_ext = os.path.splitext(file_name)
        suffix_map = {
            "input": "_input",
            "restored": "_restored",
            "gt": "_gt"
        }
        suffix = suffix_map.get(query_type, "")
        format_suffix = "_full" if keep_all_fields else "_simple"
        sample_suffix = f"_sample{sample_size}" if sample_size > 0 else ""
        scene_suffix = "_filtered" if scene_list_file else ""
        output_file = os.path.join(output_dir, f"{file_base}{suffix}{format_suffix}{sample_suffix}{scene_suffix}{file_ext}")
        
        # 处理单个文件
        process_json_file(input_file, output_file, query_type, objects_data_dir, 
                         keep_all_fields, sample_size, seed, scene_list_file)
    
    print(f"\n批量处理完成！共处理 {len(json_files)} 个文件")

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换场景数据格式")
    parser.add_argument("input", help="输入JSON文件路径或目录路径")
    parser.add_argument("-o", "--output", help="输出文件/目录路径")
    parser.add_argument("-b", "--batch", action="store_true", help="批量处理目录")
    parser.add_argument("--objects_dir", default="data/scenegraph", help="objects.json文件的根目录（默认: data/scenegraph）")
    
    # 修改为使用查询类型参数
    parser.add_argument("-q", "--query_type", 
                       choices=["input", "restored", "gt"], 
                       default="input",
                       help="选择使用哪种查询类型: input (默认), restored, gt")
    
    # 添加新参数控制输出格式
    parser.add_argument("-f", "--full", action="store_true", 
                       help="保留所有字段（默认只保留scene_id和query）")
    
    # 添加随机抽样参数
    parser.add_argument("-s", "--sample_size", type=int, default=0,
                       help="随机抽样的数量，0表示不抽样（默认: 0）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子，用于可重复的随机抽样（默认: 42）")
    
    # 添加场景列表文件参数
    parser.add_argument("--scene_list", type=str, 
                       help="场景列表文件路径，包含允许的场景ID（每行一个）")
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量处理目录
        batch_process_directory(args.input, args.output, query_type=args.query_type, 
                              objects_data_dir=args.objects_dir, keep_all_fields=args.full,
                              sample_size=args.sample_size, seed=args.seed,
                              scene_list_file=args.scene_list)
    else:
        # 处理单个文件
        process_json_file(args.input, args.output, query_type=args.query_type, 
                         objects_data_dir=args.objects_dir, keep_all_fields=args.full,
                         sample_size=args.sample_size, seed=args.seed,
                         scene_list_file=args.scene_list)