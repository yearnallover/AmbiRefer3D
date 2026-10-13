import csv
import os
import torch
import torch.multiprocessing as mp
import glob
import argparse  # [新增] 用于解析命令行参数
from data_construction.real.object_render.render_util import StrictViewRenderer, SceneRenderManager

# =========================================================
# 辅助函数：检查图片是否已存在
# =========================================================
def check_if_object_rendered(output_root, scene_id, obj_id):
    scene_dir = os.path.join(output_root, scene_id)
    if not os.path.exists(scene_dir): return False
    # 只要有3张以上特写，视为完成
    pattern = os.path.join(scene_dir, f"{scene_id}_obj{obj_id}_*.jpg")
    found_files = glob.glob(pattern)
    return len(found_files) >= 3

def check_if_overview_rendered(output_root, scene_id):
    scene_dir = os.path.join(output_root, scene_id)
    if not os.path.exists(scene_dir): return False
    # 只要有3张以上全景，视为完成
    pattern = os.path.join(scene_dir, f"{scene_id}_overview_*.jpg")
    found_files = glob.glob(pattern)
    return len(found_files) >= 3

# =========================================================
# 工作进程
# =========================================================
def worker_process(gpu_id, scene_tasks, dataset_root, output_root, bbox_root, render_mode):
    """
    render_mode: 'all', 'overview', or 'objects'
    """
    device = f"cuda:{gpu_id}"
    print(f"--- [GPU {gpu_id}] Worker started. Assigned {len(scene_tasks)} scenes. Mode: {render_mode} ---")

    try:
        manager = SceneRenderManager(dataset_root, output_root, bbox_root_dir=bbox_root)
        
        for scene_id, target_objects_map in scene_tasks.items():
            # 1. 路径检查
            ply_path, meta_path, bbox_path = manager._get_scene_files(scene_id)
            if not ply_path or not bbox_path: continue

            # 2. 初始化渲染器 (最耗时步骤，只做一次)
            try:
                renderer = StrictViewRenderer(
                    ply_path, bbox_path, meta_file_path=meta_path, scan_id=scene_id, device=device
                )
            except Exception as e:
                print(f"[GPU {gpu_id}] Error initializing {scene_id}: {e}")
                continue

            scene_output_dir = os.path.join(output_root, scene_id)
            os.makedirs(scene_output_dir, exist_ok=True)

            # =========================================================
            # === 模式 1: 渲染全场景概览图 (Overview) ===
            # =========================================================
            if render_mode in ["all", "overview"]:
                if not check_if_overview_rendered(output_root, scene_id):
                    try:
                        # elevation=30: 30度俯视角
                        # view_count=3: 环绕3张
                        renderer.render_scene_overview(
                            output_dir=scene_output_dir, 
                            view_count=3, 
                            elevation=30,
                            zoom=2.5
                        )
                    except Exception as e:
                        print(f"!!! [GPU {gpu_id}] Overview Error {scene_id}: {e}")
                else:
                    # print(f"[GPU {gpu_id}] Overview exists for {scene_id}.")
                    pass

            # =========================================================
            # === 模式 2: 渲染场景中的重复物体 (Objects) ===
            # =========================================================
            if render_mode in ["all", "objects"]:
                # A. 确定需要渲染的 Label 集合
                target_labels = set()
                for csv_obj_id in target_objects_map.keys():
                    obj_info = next((o for o in renderer.objects_list if o['id'] == csv_obj_id), None)
                    if obj_info:
                        target_labels.add(obj_info['label'])

                if target_labels:
                    # B. 找到场景中所有的同类物体 ID
                    final_render_list = set()
                    for obj in renderer.objects_list:
                        if obj['label'] in target_labels:
                            final_render_list.add(obj['id'])
                    
                    # C. 剔除已渲染的物体
                    objects_to_render = []
                    for obj_id in final_render_list:
                        if not check_if_object_rendered(output_root, scene_id, obj_id):
                            objects_to_render.append(obj_id)
                    
                    # D. 执行渲染
                    if objects_to_render:
                        for obj_id in objects_to_render:
                            renderer.render_object(obj_id, scene_output_dir)

    except Exception as e:
        print(f"!!! [GPU {gpu_id}] CRITICAL WORKER ERROR: {e}")

    print(f"--- [GPU {gpu_id}] Finished. ---")

# =========================================================
# 参数解析与主逻辑
# =========================================================
def parse_csv_tasks(csv_path):
    tasks = {}
    print(f"Parsing CSV: {csv_path} ...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3: continue
            scene_id = row[0].strip()
            try:
                obj_id = int(row[1])
            except ValueError: continue 
            label = row[2].strip()
            
            if scene_id not in tasks: tasks[scene_id] = {}
            tasks[scene_id][obj_id] = label
    return tasks

def main():
    # 1. 定义命令行参数
    parser = argparse.ArgumentParser(description="Multi-GPU Scene Renderer")
    
    # 参数: 选择场景 ID (可选，默认为空=所有)
    parser.add_argument(
        "--scenes", 
        nargs="+", 
        default=[], 
        help="List of scene IDs to render (e.g., scene0000_00 scene0001_00). Default: All in CSV."
    )
    
    # 参数: 选择渲染模式 (默认 all)
    parser.add_argument(
        "--mode", 
        choices=["all", "overview", "objects"], 
        default="all", 
        help="Render mode: 'overview' (whole scene), 'objects' (instances), or 'all' (both)."
    )
    
    args = parser.parse_args()

    # === 路径配置 ===
    CSV_FILE = os.environ.get("AMBIREFER3D_REAL_RENDER_CSV", "data/real/descriptions.csv") 
    # DATASET_ROOT = "data/scannet"
    # BBOX_ROOT = os.environ.get("AMBIREFER3D_REAL_SCENEGRAPH_ROOT", "data/real/scenegraph") 
    # OUTPUT_ROOT = "data/render"
    DATASET_ROOT = os.environ.get("SCANNET_RAW_ROOT", "dataset/scannetv2/scans")
    BBOX_ROOT = os.environ.get("AMBIREFER3D_REAL_SCENEGRAPH_ROOT", "data/real/scenegraph") 
    OUTPUT_ROOT = os.environ.get("AMBIREFER3D_REAL_RENDER_ROOT", "data/real/render")
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.")
    if num_gpus == 0: return

    # 2. 读取所有任务
    all_tasks = parse_csv_tasks(CSV_FILE)
    
    # 3. 筛选任务 (如果用户指定了 --scenes)
    final_tasks = {}
    if args.scenes:
        print(f"Filtering for specific scenes: {args.scenes}")
        for sid in args.scenes:
            if sid in all_tasks:
                final_tasks[sid] = all_tasks[sid]
            else:
                print(f" [Warning] Scene {sid} requested but not found in CSV.")
    else:
        final_tasks = all_tasks # 默认处理所有

    scene_ids = list(final_tasks.keys())
    total_scenes = len(scene_ids)
    
    print(f"Total scenes to process: {total_scenes}")
    print(f"Render Mode: [{args.mode.upper()}]")
    
    if total_scenes == 0:
        print("No scenes to process. Exiting.")
        return
    
    # 4. 分发任务
    chunk_size = (total_scenes + num_gpus - 1) // num_gpus
    processes = []
    
    for i in range(num_gpus):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_scenes)
        if start >= end: break
        
        batch_scene_ids = scene_ids[start:end]
        gpu_tasks = {sid: final_tasks[sid] for sid in batch_scene_ids}
        
        # 将 args.mode 传入 worker
        p = mp.Process(
            target=worker_process,
            args=(i, gpu_tasks, DATASET_ROOT, OUTPUT_ROOT, BBOX_ROOT, args.mode)
        )
        p.start()
        processes.append(p)
        print(f"Launched GPU {i}: {len(batch_scene_ids)} scenes")

    for p in processes:
        p.join()
        
    print("\nAll tasks finished.")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()