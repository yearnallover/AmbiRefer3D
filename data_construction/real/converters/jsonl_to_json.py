import json

def jsonl_to_json(jsonl_file_path, json_file_path):
    """
    将JSONL文件转换为JSON数组格式，并按scene_id排序
    """
    data = []
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                try:
                    # 解析JSON行
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析错误: {e}")
                    print(f"错误行内容: {line}")
                    continue
                    
    except FileNotFoundError:
        print(f"文件未找到: {jsonl_file_path}")
        return False
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return False
    
    # 按scene_id排序
    try:
        # 假设每个item都有'scene_id'字段
        data.sort(key=lambda x: x.get('scene_id', ''))
    except Exception as e:
        print(f"排序时出错: {e}")
        # 即使排序失败，也继续处理
    
    # 写入JSON文件
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"转换成功！共转换 {len(data)} 条记录")
        print(f"输出文件: {json_file_path}")
        return True
    except Exception as e:
        print(f"写入JSON文件时出错: {e}")
        return False

# 使用示例
if __name__ == "__main__":
    # 输入JSONL文件路径
    jsonl_file = "data/result/results_v2.jsonl"  # 替换为你的JSONL文件路径
    
    # 输出JSON文件路径
    json_file = "data/result/results_v2.json"   # 输出的JSON文件路径
    
    # 执行转换
    jsonl_to_json(jsonl_file, json_file)