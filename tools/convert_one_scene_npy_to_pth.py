import os, argparse
import numpy as np
import torch

VALID_NYU40 = [1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
NYU2IDX = {nyu:i for i, nyu in enumerate(VALID_NYU40)}

def corners_from_minmax(mn, mx):
    mn = np.asarray(mn, dtype=np.float32)
    mx = np.asarray(mx, dtype=np.float32)
    return np.array([
        [mn[0], mn[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mx[1], mx[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mn[2]],
        [mx[0], mx[1], mx[2]],
    ], dtype=np.float32)

def map_sem_to_contiguous(sem_raw: np.ndarray) -> np.ndarray:
    sem_raw = sem_raw.astype(np.int32, copy=False)
    sem_cont = np.full_like(sem_raw, fill_value=-1, dtype=np.int8)
    for nyu, idx in NYU2IDX.items():
        sem_cont[sem_raw == nyu] = idx
    return sem_cont

def safe_choice(rng, pool_idx: np.ndarray, k: int, replace_if_needed: bool):
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    n = len(pool_idx)
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    if k <= n:
        return rng.choice(pool_idx, size=k, replace=False)
    if replace_if_needed:
        return rng.choice(pool_idx, size=k, replace=True)
    return pool_idx.copy()

def allocate_foreground_sizes(
    counts: np.ndarray,
    fg_budget: int,
    per_inst_min: int,
    per_inst_max: int,
    rng: np.random.Generator,
):
    """
    counts: 每个实例的点数 (num_inst,)
    返回 sizes: 每个实例采样多少点 (num_inst,), sum = fg_budget (尽量)
    """
    num_inst = len(counts)
    if num_inst == 0 or fg_budget <= 0:
        return np.zeros((num_inst,), dtype=np.int64)

    desired = np.minimum(counts, per_inst_max).astype(np.int64)

    # 至少 1 个点的硬下限（只要有预算）
    if fg_budget < num_inst:
        sizes = np.zeros((num_inst,), dtype=np.int64)
        pick = rng.choice(num_inst, size=fg_budget, replace=False)
        sizes[pick] = 1
        return sizes

    mins = np.minimum(desired, max(1, per_inst_min)).astype(np.int64)

    # 如果连 mins 都不够，就退化为每个实例 1 点 + 剩余按容量分配
    if mins.sum() > fg_budget:
        mins = np.ones((num_inst,), dtype=np.int64)

    sizes = mins.copy()
    remaining = int(fg_budget - sizes.sum())
    if remaining <= 0:
        return sizes

    caps = (desired - sizes).clip(min=0)
    # 迭代分配，确保不超过 cap
    while remaining > 0 and caps.sum() > 0:
        probs = caps / caps.sum()
        draw = rng.multinomial(remaining, probs)
        draw = np.minimum(draw, caps)
        sizes += draw
        remaining -= int(draw.sum())
        caps = (desired - sizes).clip(min=0)

    return sizes

def foreground_quota_sampling(
    xyz, rgb, normal, sem_cont, inst,
    num_points: int,
    fg_ratio: float,
    per_inst_min: int,
    per_inst_max: int,
    seed: int,
    bg_replace: bool = True,
):
    """
    前景(inst>0)配额采样 + 少量背景(inst<=0)
    返回 selection indices
    """
    rng = np.random.default_rng(seed)

    n_all = xyz.shape[0]
    if n_all == 0:
        return np.empty((0,), dtype=np.int64)

    # 背景兼容：0 或 -1 都算背景（这里统一用 inst<=0）
    inst = inst.astype(np.int32, copy=False)
    fg_mask = inst > 0
    bg_mask = ~fg_mask

    fg_idx_all = np.where(fg_mask)[0]
    bg_idx_all = np.where(bg_mask)[0]

    # 如果没有前景，退化为全局随机采样
    if len(fg_idx_all) == 0:
        return safe_choice(rng, np.arange(n_all), num_points, replace_if_needed=True)

    fg_budget = int(round(num_points * fg_ratio))
    fg_budget = max(1, min(num_points, fg_budget))
    bg_budget = num_points - fg_budget

    # 前景：按实例分配
    inst_ids = np.unique(inst[fg_mask])
    inst_ids = inst_ids[inst_ids > 0]
    inst_ids = np.sort(inst_ids)

    counts = np.array([(inst == iid).sum() for iid in inst_ids], dtype=np.int64)
    sizes = allocate_foreground_sizes(counts, fg_budget, per_inst_min, per_inst_max, rng)

    chosen = []
    for iid, k in zip(inst_ids, sizes):
        pool = np.where(inst == iid)[0]
        # 这里通常不需要 replace，因为 pool 往往 >= k；若 k>pool 允许重复以满足固定点数
        sel = safe_choice(rng, pool, int(k), replace_if_needed=True)
        chosen.append(sel)

    fg_sel = np.concatenate(chosen, axis=0) if len(chosen) else np.empty((0,), dtype=np.int64)

    # 背景：随机取少量，尽量不喧宾夺主
    bg_sel = safe_choice(rng, bg_idx_all, bg_budget, replace_if_needed=bg_replace) if bg_budget > 0 else np.empty((0,), dtype=np.int64)

    sel = np.concatenate([fg_sel, bg_sel], axis=0)

    # 若因为各种原因不足 num_points，补齐（从已选里重复抽）
    if len(sel) < num_points:
        need = num_points - len(sel)
        pad = safe_choice(rng, sel, need, replace_if_needed=True)
        sel = np.concatenate([sel, pad], axis=0)

    # 若超了，截断
    if len(sel) > num_points:
        sel = rng.choice(sel, size=num_points, replace=False)

    rng.shuffle(sel)
    return sel.astype(np.int64)

def build_aabb_from_bbox_npy(bbox: np.ndarray):
    aabb_obj_ids = []
    aabb_corner_xyz = []
    # bbox row: [cx,cy,cz,dx,dy,dz,sem_id,inst_id_minus_1]
    for row in bbox:
        cx,cy,cz,dx,dy,dz,_, instm1 = row.tolist()
        inst_id = int(instm1) + 1
        mn = np.array([cx-dx/2, cy-dy/2, cz-dz/2], dtype=np.float32)
        mx = np.array([cx+dx/2, cy+dy/2, cz+dz/2], dtype=np.float32)
        aabb_obj_ids.append(inst_id)
        aabb_corner_xyz.append(corners_from_minmax(mn, mx))
    aabb_obj_ids = np.asarray(aabb_obj_ids, dtype=np.int16)
    aabb_corner_xyz = np.stack(aabb_corner_xyz, 0).astype(np.float32) if len(aabb_corner_xyz) else np.zeros((0,8,3), np.float32)
    return aabb_obj_ids, aabb_corner_xyz

def build_aabb_from_full_points(xyz_full: np.ndarray, inst_full: np.ndarray):
    aabb_obj_ids = []
    aabb_corner_xyz = []
    inst_full = inst_full.astype(np.int32, copy=False)
    for inst_id in np.unique(inst_full):
        if inst_id <= 0:
            continue
        pts = xyz_full[inst_full == inst_id]
        if pts.shape[0] == 0:
            continue
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] == 0:
            continue
        mn = pts.min(0); mx = pts.max(0)
        if not np.isfinite(mn).all() or not np.isfinite(mx).all():
            continue
        aabb_obj_ids.append(int(inst_id))
        aabb_corner_xyz.append(corners_from_minmax(mn, mx))
    aabb_obj_ids = np.asarray(aabb_obj_ids, dtype=np.int16)
    aabb_corner_xyz = np.stack(aabb_corner_xyz, 0).astype(np.float32) if len(aabb_corner_xyz) else np.zeros((0,8,3), np.float32)
    return aabb_obj_ids, aabb_corner_xyz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_dir", required=True)
    ap.add_argument("--scene_id", required=True)
    ap.add_argument("--out_pth", required=True)

    ap.add_argument("--num_points", type=int, default=40960)
    ap.add_argument("--seed", type=int, default=0)

    # -------- 前景配额采样参数 --------
    ap.add_argument("--fg_ratio", type=float, default=0.90,
                    help="前景点占比（inst>0），建议 0.85~0.95；想少点墙地就调高")
    ap.add_argument("--per_inst_min", type=int, default=256,
                    help="每个前景实例至少采样多少点（预算不足会自动退化）")
    ap.add_argument("--per_inst_max", type=int, default=4096,
                    help="每个前景实例最多采样多少点（避免大物体独占）")
    ap.add_argument("--bg_replace", action="store_true",
                    help="背景点不够时是否允许重复采样补齐（默认不启用；但固定点数时有时需要）")

    args = ap.parse_args()

    base = os.path.join(args.npy_dir, args.scene_id)
    vert = np.load(base + "_vert.npy")        # (N,9)
    inst = np.load(base + "_ins_label.npy")   # (N,)
    sem  = np.load(base + "_sem_label.npy")   # (N,)
    bbox = np.load(base + "_bbox.npy") if os.path.exists(base + "_bbox.npy") else None

    xyz_full = vert[:, :3].astype(np.float32)
    rgb_full = vert[:, 3:6]
    if rgb_full.dtype != np.uint8:
        rgb_full = np.clip(rgb_full, 0, 255).astype(np.uint8)
    normal_full = vert[:, 6:9].astype(np.float32)

    inst_full = inst.astype(np.int32, copy=False)
    sem_raw_full = sem.astype(np.int32, copy=False)
    sem_cont_full = map_sem_to_contiguous(sem_raw_full)  # (N,)

    # 1) 前景配额采样得到 sel
    sel = foreground_quota_sampling(
        xyz_full, rgb_full, normal_full, sem_cont_full, inst_full,
        num_points=args.num_points,
        fg_ratio=args.fg_ratio,
        per_inst_min=args.per_inst_min,
        per_inst_max=args.per_inst_max,
        seed=args.seed,
        bg_replace=args.bg_replace,
    )

    xyz = xyz_full[sel]
    rgb = rgb_full[sel]
    normal = normal_full[sel]
    sem_cont = sem_cont_full[sel]
    inst_out = inst_full[sel].astype(np.int16, copy=False)

    # 2) AABB：优先用 bbox.npy（稳定且不受采样影响）；否则用“全量点”算
    if bbox is not None and bbox.size > 0:
        aabb_obj_ids, aabb_corner_xyz = build_aabb_from_bbox_npy(bbox)
    else:
        aabb_obj_ids, aabb_corner_xyz = build_aabb_from_full_points(xyz_full, inst_full)

    out = {
        "xyz": xyz.astype(np.float32, copy=False),
        "rgb": rgb.astype(np.uint8, copy=False),
        "normal": normal.astype(np.float32, copy=False),
        "sem_labels": sem_cont.astype(np.int8, copy=False),
        "instance_ids": inst_out,
        "aabb_obj_ids": aabb_obj_ids,
        "aabb_corner_xyz": aabb_corner_xyz,
    }

    os.makedirs(os.path.dirname(args.out_pth), exist_ok=True)
    torch.save(out, args.out_pth)

    # 简单统计便于你确认“物体更密、墙地更少”
    fg = int((inst_out > 0).sum())
    bg = int((inst_out <= 0).sum())
    uniq_fg = int(len(np.unique(inst_out[inst_out > 0]))) if fg > 0 else 0
    print("[OK] wrote:", args.out_pth)
    print(f"sampled num_points={len(xyz)}  fg={fg}  bg={bg}  fg_ratio={fg/len(xyz):.3f}  uniq_fg_inst={uniq_fg}")
    print("xyz range min/max:", xyz.min(0), xyz.max(0))

if __name__ == "__main__":
    main()
