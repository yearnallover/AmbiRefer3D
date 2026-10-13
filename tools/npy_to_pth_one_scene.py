import os
import argparse
import numpy as np
import torch

def nyu40_to_continuous(sem_raw: np.ndarray):
    # 与你之前 preprocess 一致：20 类
    valid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    mp = {nyu: i for i, nyu in enumerate(valid)}
    out = np.full_like(sem_raw, fill_value=-1, dtype=np.int8)
    for nyu, idx in mp.items():
        out[sem_raw == nyu] = np.int8(idx)
    return out

def build_aabb(inst: np.ndarray, xyz: np.ndarray, min_inst_points: int = 20):
    inst = inst.astype(np.int32)
    xyz = xyz.astype(np.float32)

    # 兼容背景 =0 或 -1（你的 3D-FRONT 基本不会有背景点，但保守处理）
    u = np.unique(inst)
    if u.size == 0:
        return np.zeros((0,), np.int16), np.zeros((0, 8, 3), np.float32)

    if u.min() == -1:
        u = u[u >= 0]
    else:
        u = u[u > 0]

    ids, corners = [], []
    for inst_id in u:
        m = (inst == inst_id)
        pts = xyz[m]
        if pts.shape[0] < min_inst_points:
            continue
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < min_inst_points:
            continue

        mn = pts.min(axis=0)
        mx = pts.max(axis=0)

        c = np.array([
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ], dtype=np.float32)

        if not np.isfinite(c).all():
            continue

        ids.append(int(inst_id))
        corners.append(c)

    if len(ids) == 0:
        return np.zeros((0,), np.int16), np.zeros((0, 8, 3), np.float32)

    return np.asarray(ids, np.int16), np.stack(corners, axis=0).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", required=True)
    ap.add_argument("--npy_dir", required=True)
    ap.add_argument("--out_pth", required=True)
    ap.add_argument("--min_inst_points", type=int, default=20)
    args = ap.parse_args()

    base = os.path.join(args.npy_dir, args.scene_id)
    vert_p = base + "_vert.npy"
    ins_p  = base + "_ins_label.npy"
    sem_p  = base + "_sem_label.npy"

    for p in [vert_p, ins_p, sem_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    v = np.load(vert_p)               # (N,9) xyz rgb nrm
    inst = np.load(ins_p).astype(np.int32)   # (N,)
    sem_raw = np.load(sem_p).astype(np.int32)

    xyz = v[:, :3].astype(np.float32)

    rgb = v[:, 3:6]
    # 你的导出里 rgb 是 0；这里兼容 float/uint8 两种
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255)
        if rgb.max() <= 1.0:
            rgb = (rgb * 255.0)
        rgb = rgb.astype(np.uint8)

    nrm = v[:, 6:9].astype(np.float32)

    sem_cont = nyu40_to_continuous(sem_raw)  # (N,) int8, -1/0..19

    aabb_obj_ids, aabb_corner_xyz = build_aabb(inst, xyz, min_inst_points=args.min_inst_points)

    os.makedirs(os.path.dirname(args.out_pth), exist_ok=True)
    torch.save(
        {
            "xyz": xyz,
            "rgb": rgb,
            "normal": nrm,
            "sem_labels": sem_cont,
            "instance_ids": inst.astype(np.int16),
            "aabb_obj_ids": aabb_obj_ids,
            "aabb_corner_xyz": aabb_corner_xyz,
        },
        args.out_pth
    )

    print("[OK] wrote:", args.out_pth)
    print(" xyz:", xyz.shape, "rgb:", rgb.shape, "aabb:", aabb_corner_xyz.shape)

if __name__ == "__main__":
    main()
