import json
import os

# ================= 配置路径 =================
# 输入文件 (你的新格式数据)
INPUT_PATH = os.environ.get('AMBIREFER3D_REAL_AMBIGUOUS_JSON', 'data/real/ambiguous_queries.json') 

# 场景图数据根目录 (用于查物体数量)
SCENE_GRAPH_ROOT = os.environ.get('AMBIREFER3D_REAL_SCENEGRAPH_ROOT', 'data/real/scenegraph')

# 输出文件 (过滤后的数据)
OUTPUT_PATH = os.environ.get('AMBIREFER3D_REAL_FILTERED_JSON', 'data/real/filtered_queries.json')

def filter_data_new_format():
    # --- 1. 检查输入文件 ---
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return

    print(f"📖 正在读取原始文件: {INPUT_PATH} ...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        main_data = json.load(f)

    filtered_data = []
    dropped_count = 0
    missing_scene_count = 0

    print(f"🔍 开始扫描 {len(main_data)} 条数据，剔除同类干扰项 > 5 的条目...")

    for i, item in enumerate(main_data):
        # 打印进度
        if (i + 1) % 1000 == 0:
            print(f"   ...已处理 {i + 1} 条")

        # === 适配新数据结构 ===
        scene_id = item.get('scene_id')
        # 注意：新数据里是 object_id，且是字符串类型 "39"
        object_id = item.get('object_id') 
        
        if not scene_id or not object_id:
            continue

        # --- 2. 定位并读取场景文件 ---
        scene_obj_path = os.path.join(SCENE_GRAPH_ROOT, scene_id, "objects.json")
        
        if not os.path.exists(scene_obj_path):
            missing_scene_count += 1
            continue # 找不到场景文件，无法判断，跳过

        try:
            with open(scene_obj_path, 'r', encoding='utf-8') as f:
                scene_full = json.load(f)
        except:
            missing_scene_count += 1
            continue

        scene_data = scene_full.get(scene_id)
        if not scene_data: 
            continue

        inst_to_label = scene_data.get('inst_to_label', {})
        
        # 确保 object_id 是字符串格式 (JSON里看起来是字符串，但以防万一)
        target_id_str = str(object_id)
        
        # 如果目标ID不在场景里，跳过
        if target_id_str not in inst_to_label:
            continue
            
        target_label = inst_to_label[target_id_str]

        # --- 3. 统计同类物体数量 ---
        candidate_count = 0
        for label in inst_to_label.values():
            if label == target_label:
                candidate_count += 1

        # --- 4. 核心筛选逻辑 (>5 剔除) ---
        if candidate_count > 5:
            dropped_count += 1
            continue  # 🚫 剔除，不保存

        # --- 5. 保存 (原样追加) ---
        filtered_data.append(item)

    # --- 6. 保存结果 ---
    print(f"\n💾 正在保存到新文件: {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    # --- 最终报告 ---
    print("\n" + "="*30)
    print("✅ 过滤完成")
    print("="*30)
    print(f"原始总数: {len(main_data)}")
    print(f"保留总数: {len(filtered_data)} (<= 5)")
    print(f"剔除总数: {dropped_count} (> 5)")
    if missing_scene_count > 0:
        print(f"缺失场景: {missing_scene_count}")
    print("-" * 30)
    print(f"保留比例: {len(filtered_data)/len(main_data)*100:.2f}%")

if __name__ == "__main__":
    filter_data_new_format()