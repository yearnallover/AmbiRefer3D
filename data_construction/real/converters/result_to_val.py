import json

def split_data_by_scene_ids(json_data, val_scene_ids_file, output_val_path, output_other_path):
    """
    根据验证集scene_id列表分离数据
    
    参数:
    - json_data: 原始JSON数据
    - val_scene_ids_file: 验证集scene_id列表文件路径
    - output_val_path: 验证集输出文件路径
    - output_other_path: 其他集输出文件路径
    """
    
    # 读取验证集scene_id列表
    with open(val_scene_ids_file, 'r') as f:
        val_scene_ids = {line.strip() for line in f if line.strip()}
    
    print(f"找到 {len(val_scene_ids)} 个验证集scene_id")
    
    # 分离数据
    val_data = []
    other_data = []
    
    for item in json_data:
        scene_id = item.get("scene_id")
        if scene_id in val_scene_ids:
            val_data.append(item)
        else:
            other_data.append(item)
    
    print(f"验证集: {len(val_data)} 条数据")
    print(f"其他集: {len(other_data)} 条数据")
    
    # 保存验证集
    with open(output_val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"验证集已保存到: {output_val_path}")
    
    # 保存其他集
    with open(output_other_path, 'w', encoding='utf-8') as f:
        json.dump(other_data, f, indent=2, ensure_ascii=False)
    print(f"其他集已保存到: {output_other_path}")
    
    return val_data, other_data


# 主程序
if __name__ == "__main__":
    # 1. 加载原始JSON数据
    input_json_file = "data/result/results_v2.json"  # 替换为你的文件路径
    
    with open(input_json_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"原始数据总数: {len(original_data)} 条")
    
    # 2. 分离数据
    val_scene_ids_file = "data/scannetv2_val.txt"
    output_val = "data/result/result_val.json"
    output_other = "data/result/result_train.json"
    
    val_data, other_data = split_data_by_scene_ids(
        original_data, 
        val_scene_ids_file, 
        output_val, 
        output_other
    )

    # 可选：查看一些统计信息
    print("\n=== 统计信息 ===")
    
    # 统计验证集中包含的scene_id
    val_scene_ids_set = set()
    for item in val_data:
        val_scene_ids_set.add(item.get("scene_id"))
    
    print(f"验证集包含 {len(val_scene_ids_set)} 个不同的scene_id:")
    for scene_id in sorted(val_scene_ids_set)[:10]:  # 只显示前10个
        print(f"  - {scene_id}")
    if len(val_scene_ids_set) > 10:
        print(f"  ... 和其他 {len(val_scene_ids_set) - 10} 个")