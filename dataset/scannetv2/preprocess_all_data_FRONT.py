# import os
# import numpy as np
# import torch
# import hydra
# from functools import partial
# from tqdm.contrib.concurrent import process_map
# from multiprocessing import cpu_count
# from typing import List, Dict, Any, Optional

# # --- 辅助函数：从 TSV 文件获取语义标签映射（必需） ---
# def get_semantic_mapping_file(mapping_file: str, mapping_name: str) -> Optional[Any]:
#     """
#     读取TSV标签映射文件，返回类别名到整数ID的映射字典。
#     mapping_name: 'nyu40'（第5列）、'eigen13'（第6列）、'mpcat40'（第17列）
#     """
#     label_mapping = {}
#     mapping_col_idx = {
#         "nyu40": 4,
#         "eigen13": 5,
#         "mpcat40": 16
#     }
#     if not os.path.exists(mapping_file):
#         print(f"[Error] Mapping file not found: {mapping_file}")
#         return {}
#     col_idx = mapping_col_idx.get(mapping_name, 4)
#     with open(mapping_file, "r") as f:
#         header = f.readline()
#         for line in f:
#             if not line.strip():
#                 continue
#             parts = line.strip().split("\t")
#             if len(parts) <= col_idx:
#                 continue
#             name = parts[1]
#             try:
#                 label = int(parts[col_idx])
#             except ValueError:
#                 continue
#             label_mapping[name] = label
#     return label_mapping

# # --- 实例边界框生成函数 ---
# def build_instance_bboxes(xyz_all: np.ndarray, inst_all: np.ndarray, sem_all: np.ndarray) -> np.ndarray:
#     """
#     计算场景中每个实例的 AABB 边界框。
#     返回的格式是 [cx, cy, cz, dx, dy, dz, sem_id, inst_id_minus_1]
#     """
#     instance_ids = np.unique(inst_all)
#     bboxes = []
#     for inst_id in instance_ids:
#         if inst_id == 0:
#             continue
#         mask = (inst_all == inst_id)
#         pts = xyz_all[mask]
#         if pts.shape[0] == 0:
#             continue
        
#         # 计算 AABB
#         xmin, ymin, zmin = pts.min(axis=0)
#         xmax, ymax, zmax = pts.max(axis=0)
#         cx = (xmin + xmax) / 2.0
#         cy = (ymin + ymax) / 2.0
#         cz = (zmin + zmax) / 2.0
#         dx = xmax - xmin
#         dy = ymax - ymin
#         dz = zmax - zmin
        
#         # 获取语义ID（原始 NYU40 ID）
#         sem_ids = sem_all[mask]
#         sem_id = int(sem_ids[0]) if sem_ids.size > 0 else 0
        
#         # 边界框格式：[中心点(3), 尺寸(3), NYU40 原始 ID, 实例 ID (减 1)]
#         bbox = np.array([cx, cy, cz, dx, dy, dz, sem_id, int(inst_id) - 1], dtype=np.float32)
#         bboxes.append(bbox)
        
#     if not bboxes:
#         return np.zeros((0, 8), dtype=np.float32)

#     return np.vstack(bboxes)


# # --- 核心处理函数：集成 NYU40 ID 到连续索引的映射逻辑 ---
# def process_one_scene_from_npy(scene: str, cfg: Any, split: str, label_map: Any, 
#                                invalid_ids: List[int], valid_semantic_mapping: List[int]):
    
#     # 1. 构造文件路径
#     # 确保 cfg 中有 raw_npy_path 配置项
#     if not hasattr(cfg, 'raw_npy_path'):
#         print(f"Error: cfg.raw_npy_path is not defined. Skipping scene {scene}.")
#         return

#     base_path = os.path.join(cfg.raw_npy_path, scene)
#     vert_file_path = base_path + '_vert.npy'
#     ins_label_path = base_path + '_ins_label.npy'
#     sem_label_path = base_path + '_sem_label.npy'
    
#     # 2. 检查文件是否存在
#     if not (os.path.exists(vert_file_path) and 
#             os.path.exists(ins_label_path) and 
#             os.path.exists(sem_label_path)):
#         print(f"Warning: Missing NPY files for scene {scene}. Skipping.")
#         return

#     # 3. 读取您的 .npy 文件
#     mesh_vertices = np.load(vert_file_path) # [N, 9] (xyz, rgb, nrm)
#     instance_ids = np.load(ins_label_path)  # [N]
#     sem_labels_raw_nyu40 = np.load(sem_label_path) # [N], 包含 NYU40 原始 ID (例如 5, 7, 6...)
    
#     # 4. 提取 XYZ, RGB, Normal
#     xyz = mesh_vertices[:, :3].astype(np.float32)
#     rgb = mesh_vertices[:, 3:6].astype(np.uint8) 
#     normal = mesh_vertices[:, 6:9].astype(np.float32)
    
#     # 5. 将 NYU40 原始ID 转换为连续索引 (0-19)
#     valid_semantic_mapping = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
#     nyu40_to_continuous_idx = {nyu_id: continuous_idx for continuous_idx, nyu_id in enumerate(valid_semantic_mapping)}

#     # 初始化新的语义标签数组。使用 -1 作为填充值（表示忽略或无效）
#     sem_labels_continuous = np.full_like(sem_labels_raw_nyu40, fill_value=-1, dtype=np.int8)

#     # 进行逐点映射
#     for nyu_id, continuous_idx in nyu40_to_continuous_idx.items():
#         # 找到所有匹配该 NYU ID 的点
#         mask = (sem_labels_raw_nyu40 == nyu_id)
#         # 赋值为新的连续索引 (0-19)
#         sem_labels_continuous[mask] = continuous_idx
        
#     sem_labels_final = sem_labels_continuous
    
#     # 6. 计算 AABB 并提取角点
#     # 注意：这里使用 sem_labels_raw_nyu40 来计算 bboxes，因为 build_instance_bboxes 需要它
#     # 但实际存储到 .pth 文件的 sem_labels 必须是连续索引。
    
#     # 这里的 sem_labels 必须是原始的 NYU40 ID，否则 build_instance_bboxes 中的 sem_id 不正确
#     # 6) 计算 AABB 角点（严格：只保留有足够点的实例，并保证全 finite）
#     inst = instance_ids.astype(np.int32)
#     xyz_f = xyz.astype(np.float32)

#     # 兼容两种背景编码：0 或 -1
#     unique_ids = np.unique(inst)
#     if unique_ids.min() == -1:
#         unique_ids = unique_ids[unique_ids >= 0]   # 背景=-1
#     else:
#         unique_ids = unique_ids[unique_ids > 0]    # 背景=0

#     min_inst_points = 1  # 可以先设 1 跑通，想“完美转化”建议 >=20

#     aabb_ids = []
#     aabb_corners = []

#     for inst_id in unique_ids:
#         mask = (inst == inst_id)
#         pts = xyz_f[mask]
#         if pts.shape[0] < min_inst_points:
#             continue
#         # 若 xyz 本身有 NaN/Inf，先过滤
#         pts = pts[np.isfinite(pts).all(axis=1)]
#         if pts.shape[0] < min_inst_points:
#             continue

#         mn = pts.min(axis=0)
#         mx = pts.max(axis=0)

#         # 8 corners: (xmin/xmax) × (ymin/ymax) × (zmin/zmax)
#         corners = np.array([
#             [mn[0], mn[1], mn[2]],
#             [mn[0], mn[1], mx[2]],
#             [mn[0], mx[1], mn[2]],
#             [mn[0], mx[1], mx[2]],
#             [mx[0], mn[1], mn[2]],
#             [mx[0], mn[1], mx[2]],
#             [mx[0], mx[1], mn[2]],
#             [mx[0], mx[1], mx[2]],
#         ], dtype=np.float32)

#         if not np.isfinite(corners).all():
#             continue

#         aabb_ids.append(inst_id)
#         aabb_corners.append(corners)

#     aabb_obj_ids = np.asarray(aabb_ids, dtype=np.int16)
#     aabb_corner_xyz = np.stack(aabb_corners, axis=0).astype(np.float32) if len(aabb_corners) > 0 else np.zeros((0,8,3), dtype=np.float32)

#     if aabb_corner_xyz is None:
#         aabb_corner_xyz = np.zeros((0,8,3), dtype=np.float32)
#     if aabb_obj_ids is None:
#         aabb_obj_ids = np.zeros((0,), dtype=np.int16)



#     # 7. 保存为 .pth 文件
#     output_dir = os.path.join(cfg.scene_dataset_path, split)
#     os.makedirs(output_dir, exist_ok=True) # 确保目录存在
    
#     torch.save(
#         {"xyz": xyz, 
#          "rgb": rgb, 
#          "normal": normal, 
#          "sem_labels": sem_labels_final.astype(np.int8), # <-- 使用转换后的连续索引
#          "instance_ids": instance_ids.astype(np.int16), 
#          "aabb_obj_ids": aabb_obj_ids, 
#          "aabb_corner_xyz": aabb_corner_xyz},
#         os.path.join(output_dir, f"{scene}.pth")
#     )
    
#     print(f"Scene {scene} processed and saved to {os.path.abspath(os.path.join(output_dir, f'{scene}.pth'))}")


# # --- 主函数 ---
# @hydra.main(version_base=None, config_path="../../config/data", config_name="scannetv2_FRONT")
# def main(cfg):
    
#     # 确保 data.scene_dataset_path 和 raw_npy_path 被正确定义
#     if not hasattr(cfg, 'scene_dataset_path') or not hasattr(cfg, 'raw_npy_path'):
#         print("Error: Configuration missing required fields (scene_dataset_path or raw_npy_path).")
#         return

#     max_workers = cpu_count() if "workers" not in cfg else cfg.workers
#     print(f"\nUsing {max_workers} CPU threads.")

#     # 读取语义标签映射文件（根据 YAML 配置，这一步是必需的）
#     label_map = get_semantic_mapping_file(cfg.scene_metadata.label_mapping_file,
#                                           cfg.scene_metadata.semantic_mapping_name)

#     # 准备并处理 train, val, test 划分
#     for split in ("train_FRONT", "val_FRONT", "test_FRONT"):
#         output_path = os.path.join(cfg.scene_dataset_path, split)
        
#         if os.path.exists(output_path):
#             print(f"\nWarning: output directory {os.path.abspath(output_path)} exists, "
#                   f"existing files will be overwritten.")
#         os.makedirs(output_path, exist_ok=True)
        
#         # 读取场景 ID 列表
#         scene_id_file = getattr(cfg.scene_metadata, f"{split}_scene_ids")
#         try:
#             with open(scene_id_file) as f:
#                 split_list = [line.strip() for line in f]
#         except FileNotFoundError:
#             print(f"Error: Scene ID list file not found: {scene_id_file}. Skipping {split} split.")
#             continue
            
#         print(f"==> Processing {split} split ({len(split_list)} scenes) ...")
        
#         # 获取 YAML 中定义的有效 NYU40 ID 列表
#         valid_semantic_mapping = cfg.scene_metadata.valid_semantic_mapping
        
#         # 使用多进程处理场景
#         process_map(
#             partial(
#                 process_one_scene_from_npy, cfg=cfg, split=split, label_map=label_map,
#                 invalid_ids=cfg.scene_metadata.invalid_semantic_labels,
#                 valid_semantic_mapping=valid_semantic_mapping # 传递 YAML 中的有效映射列表
#             ), split_list, chunksize=1, max_workers=max_workers
#         )

#         print(f"==> {split} split complete. Saved at: {os.path.abspath(output_path)}\n")


# if __name__ == '__main__':
#     main()




import os
import numpy as np
import torch
import hydra
from functools import partial
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count, Process
from typing import List, Dict, Any, Optional

# --- 辅助函数：从 TSV 文件获取语义标签映射（必需） ---
def get_semantic_mapping_file(mapping_file: str, mapping_name: str) -> Optional[Any]:
    """
    读取TSV标签映射文件，返回类别名到整数ID的映射字典。
    mapping_name: 'nyu40'（第5列）、'eigen13'（第6列）、'mpcat40'（第17列）
    """
    label_mapping = {}
    mapping_col_idx = {
        "nyu40": 4,
        "eigen13": 5,
        "mpcat40": 16
    }
    if not os.path.exists(mapping_file):
        print(f"[Error] Mapping file not found: {mapping_file}")
        return {}
    col_idx = mapping_col_idx.get(mapping_name, 4)
    with open(mapping_file, "r") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) <= col_idx:
                continue
            name = parts[1]
            try:
                label = int(parts[col_idx])
            except ValueError:
                continue
            label_mapping[name] = label
    return label_mapping

# --- 实例边界框生成函数 ---
def build_instance_bboxes(xyz_all: np.ndarray, inst_all: np.ndarray, sem_all: np.ndarray) -> np.ndarray:
    """
    计算场景中每个实例的 AABB 边界框。
    返回的格式是 [cx, cy, cz, dx, dy, dz, sem_id, inst_id_minus_1]
    """
    instance_ids = np.unique(inst_all)
    bboxes = []
    for inst_id in instance_ids:
        if inst_id == 0:
            continue
        mask = (inst_all == inst_id)
        pts = xyz_all[mask]
        if pts.shape[0] == 0:
            continue
        
        # 计算 AABB
        xmin, ymin, zmin = pts.min(axis=0)
        xmax, ymax, zmax = pts.max(axis=0)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cz = (zmin + zmax) / 2.0
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        
        # 获取语义ID（原始 NYU40 ID）
        sem_ids = sem_all[mask]
        sem_id = int(sem_ids[0]) if sem_ids.size > 0 else 0
        
        # 边界框格式：[中心点(3), 尺寸(3), NYU40 原始 ID, 实例 ID (减 1)]
        bbox = np.array([cx, cy, cz, dx, dy, dz, sem_id, int(inst_id) - 1], dtype=np.float32)
        bboxes.append(bbox)
        
    if not bboxes:
        return np.zeros((0, 8), dtype=np.float32)

    return np.vstack(bboxes)

# --- 单scene处理函数（原逻辑不变） ---
def process_one_scene_from_npy(scene: str, cfg: Any, split: str, label_map: Any, 
                               invalid_ids: List[int], valid_semantic_mapping: List[int]):
    # 1. 构造文件路径
    if not hasattr(cfg, 'raw_npy_path'):
        print(f"Error: cfg.raw_npy_path is not defined. Skipping scene {scene}.")
        return

    base_path = os.path.join(cfg.raw_npy_path, scene)
    vert_file_path = base_path + '_vert.npy'
    ins_label_path = base_path + '_ins_label.npy'
    sem_label_path = base_path + '_sem_label.npy'
    
    # 2. 检查文件是否存在
    if not (os.path.exists(vert_file_path) and 
            os.path.exists(ins_label_path) and 
            os.path.exists(sem_label_path)):
        print(f"Warning: Missing NPY files for scene {scene}. Skipping.")
        return

    # 3. 读取 .npy 文件
    mesh_vertices = np.load(vert_file_path) # [N, 9] (xyz, rgb, nrm)
    instance_ids = np.load(ins_label_path)  # [N]
    sem_labels_raw_nyu40 = np.load(sem_label_path) # [N], 包含 NYU40 原始 ID
    
    # 4. 提取 XYZ, RGB, Normal
    xyz = mesh_vertices[:, :3].astype(np.float32)
    rgb = mesh_vertices[:, 3:6].astype(np.uint8) 
    normal = mesh_vertices[:, 6:9].astype(np.float32)
    
    # 5. 将 NYU40 原始ID 转换为连续索引 (0-19)
    valid_semantic_mapping = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    nyu40_to_continuous_idx = {nyu_id: continuous_idx for continuous_idx, nyu_id in enumerate(valid_semantic_mapping)}
    sem_labels_continuous = np.full_like(sem_labels_raw_nyu40, fill_value=-1, dtype=np.int8)
    for nyu_id, continuous_idx in nyu40_to_continuous_idx.items():
        mask = (sem_labels_raw_nyu40 == nyu_id)
        sem_labels_continuous[mask] = continuous_idx
    sem_labels_final = sem_labels_continuous
    
    # 6. 计算 AABB 角点
    inst = instance_ids.astype(np.int32)
    xyz_f = xyz.astype(np.float32)
    unique_ids = np.unique(inst)
    if unique_ids.min() == -1:
        unique_ids = unique_ids[unique_ids >= 0]   # 背景=-1
    else:
        unique_ids = unique_ids[unique_ids > 0]    # 背景=0
    min_inst_points = 1
    aabb_ids = []
    aabb_corners = []
    for inst_id in unique_ids:
        mask = (inst == inst_id)
        pts = xyz_f[mask]
        if pts.shape[0] < min_inst_points:
            continue
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < min_inst_points:
            continue
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        corners = np.array([
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ], dtype=np.float32)
        if not np.isfinite(corners).all():
            continue
        aabb_ids.append(inst_id)
        aabb_corners.append(corners)
    aabb_obj_ids = np.asarray(aabb_ids, dtype=np.int16)
    aabb_corner_xyz = np.stack(aabb_corners, axis=0).astype(np.float32) if len(aabb_corners) > 0 else np.zeros((0,8,3), dtype=np.float32)
    if aabb_corner_xyz is None:
        aabb_corner_xyz = np.zeros((0,8,3), dtype=np.float32)
    if aabb_obj_ids is None:
        aabb_obj_ids = np.zeros((0,), dtype=np.int16)

    # 7. 保存为 .pth 文件
    output_dir = os.path.join(cfg.scene_dataset_path, split)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {"xyz": xyz, 
         "rgb": rgb, 
         "normal": normal, 
         "sem_labels": sem_labels_final.astype(np.int8),
         "instance_ids": instance_ids.astype(np.int16), 
         "aabb_obj_ids": aabb_obj_ids, 
         "aabb_corner_xyz": aabb_corner_xyz},
        os.path.join(output_dir, f"{scene}.pth")
    )
    print(f"Scene {scene} processed and saved to {os.path.abspath(os.path.join(output_dir, f'{scene}.pth'))}")

# --- 新增：单split并行处理函数（封装原split循环逻辑） ---
def process_single_split(split: str, cfg: Any, label_map: Any):
    """独立处理单个split（train/val/test），供多进程调用"""
    # 计算每个split的最大工作进程数（避免总进程数过高）
    total_cpu = cpu_count()
    split_workers = max(1, total_cpu // 3)  # 3个split均分CPU核心，至少1个
    if "workers" in cfg:
        split_workers = min(split_workers, cfg.workers)  # 不超过配置的workers
    
    # 创建split输出目录
    output_path = os.path.join(cfg.scene_dataset_path, split)
    if os.path.exists(output_path):
        print(f"\nWarning: output directory {os.path.abspath(output_path)} exists, existing files will be overwritten.")
    os.makedirs(output_path, exist_ok=True)
    
    # 读取当前split的scene列表
    try:
        scene_id_file = getattr(cfg.scene_metadata, f"{split}_scene_ids")
        with open(scene_id_file) as f:
            split_list = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: Scene ID list file not found: {scene_id_file}. Skipping {split} split.")
        return
    
    print(f"==> Starting parallel processing for {split} split ({len(split_list)} scenes) with {split_workers} workers ...")
    
    # 处理当前split的所有scene（内部多进程）
    process_map(
        partial(
            process_one_scene_from_npy, cfg=cfg, split=split, label_map=label_map,
            invalid_ids=cfg.scene_metadata.invalid_semantic_labels,
            valid_semantic_mapping=cfg.scene_metadata.valid_semantic_mapping
        ), split_list, chunksize=1, max_workers=split_workers
    )

    print(f"==> {split} split complete. Saved at: {os.path.abspath(output_path)}\n")

# --- 主函数（修改为并行处理split） ---
@hydra.main(version_base=None, config_path="../../config/data", config_name="scannetv2_FRONT")
def main(cfg):
    # 检查必要配置
    if not hasattr(cfg, 'scene_dataset_path') or not hasattr(cfg, 'raw_npy_path'):
        print("Error: Configuration missing required fields (scene_dataset_path or raw_npy_path).")
        return

    # 读取语义标签映射
    label_map = get_semantic_mapping_file(cfg.scene_metadata.label_mapping_file,
                                          cfg.scene_metadata.semantic_mapping_name)

    # 定义要并行处理的split列表
    splits = ["train_FRONT", "val_FRONT", "test_FRONT"]
    # 存储进程实例的列表
    processes = []

    print(f"\nStarting parallel processing for all splits (train/val/test) ...")
    print(f"Total CPU cores: {cpu_count()}, each split will use ~{max(1, cpu_count()//3)} cores.\n")

    # 为每个split创建独立进程
    for split in splits:
        p = Process(
            target=process_single_split,
            args=(split, cfg, label_map)
        )
        p.start()  # 启动进程
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()
        print(f"Split {p.name} finished (PID: {p.pid})")

    print("\nAll splits (train/val/test) processed successfully!")

if __name__ == '__main__':
    main()