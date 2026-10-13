import os
import argparse
import hashlib
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import cpu_count

VALID_NYU40 = [1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
NYU2IDX = {nyu:i for i, nyu in enumerate(VALID_NYU40)}

def stable_int_seed(scene_id: str, base_seed: int) -> int:
    # 跨进程/跨机器稳定：md5 -> 32bit int
    h = hashlib.md5(scene_id.encode("utf-8")).hexdigest()
    sid_seed = int(h[:8], 16)  # 32-bit
    return (base_seed ^ sid_seed) & 0xFFFFFFFF

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

def allocate_foreground_sizes(counts, fg_budget, per_inst_min, per_inst_max, rng):
    num_inst = len(counts)
    if num_inst == 0 or fg_budget <= 0:
        return np.zeros((num_inst,), dtype=np.int64)

    desired = np.minimum(counts, per_inst_max).astype(np.int64)

    if fg_budget < num_inst:
        sizes = np.zeros((num_inst,), dtype=np.int64)
        pick = rng.choice(num_inst, size=fg_budget, replace=False)
        sizes[pick] = 1
        return sizes

    mins = np.minimum(desired, max(1, per_inst_min)).astype(np.int64)
    if mins.sum() > fg_budget:
        mins = np.ones((num_inst,), dtype=np.int64)

    sizes = mins.copy()
    remaining = int(fg_budget - sizes.sum())
    if remaining <= 0:
        return sizes

    caps = (desired - sizes).clip(min=0)
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
    rng = np.random.default_rng(seed)
    n_all = xyz.shape[0]
    if n_all == 0:
        return np.empty((0,), dtype=np.int64)

    inst = inst.astype(np.int32, copy=False)
    fg_mask = inst > 0
    bg_mask = ~fg_mask
    fg_idx_all = np.where(fg_mask)[0]
    bg_idx_all = np.where(bg_mask)[0]

    if len(fg_idx_all) == 0:
        return safe_choice(rng, np.arange(n_all), num_points, replace_if_needed=True)

    fg_budget = int(round(num_points * fg_ratio))
    fg_budget = max(1, min(num_points, fg_budget))
    bg_budget = num_points - fg_budget

    inst_ids = np.unique(inst[fg_mask])
    inst_ids = inst_ids[inst_ids > 0]
    inst_ids = np.sort(inst_ids)

    counts = np.array([(inst == iid).sum() for iid in inst_ids], dtype=np.int64)
    sizes = allocate_foreground_sizes(counts, fg_budget, per_inst_min, per_inst_max, rng)

    chosen = []
    for iid, k in zip(inst_ids, sizes):
        pool = np.where(inst == iid)[0]
        sel = safe_choice(rng, pool, int(k), replace_if_needed=True)
        chosen.append(sel)
    fg_sel = np.concatenate(chosen, axis=0) if len(chosen) else np.empty((0,), dtype=np.int64)

    bg_sel = safe_choice(rng, bg_idx_all, bg_budget, replace_if_needed=bg_replace) if bg_budget > 0 else np.empty((0,), dtype=np.int64)

    sel = np.concatenate([fg_sel, bg_sel], axis=0)

    if len(sel) < num_points:
        need = num_points - len(sel)
        pad = safe_choice(rng, sel, need, replace_if_needed=True)
        sel = np.concatenate([sel, pad], axis=0)

    if len(sel) > num_points:
        sel = rng.choice(sel, size=num_points, replace=False)

    rng.shuffle(sel)
    return sel.astype(np.int64)

def build_aabb_from_bbox_npy(bbox: np.ndarray):
    aabb_obj_ids = []
    aabb_corner_xyz = []
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

def read_scene_ids(txt_path: str):
    ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids

def convert_one(scene_id: str, npy_dir: str, out_dir: str,
                num_points: int, fg_ratio: float, per_inst_min: int, per_inst_max: int,
                base_seed: int, bg_replace: bool, overwrite: bool):
    out_path = os.path.join(out_dir, f"{scene_id}.pth")
    if (not overwrite) and os.path.exists(out_path):
        return ("skip", scene_id, "")

    base = os.path.join(npy_dir, scene_id)
    vert_path = base + "_vert.npy"
    ins_path  = base + "_ins_label.npy"
    sem_path  = base + "_sem_label.npy"
    bbox_path = base + "_bbox.npy"

    if not (os.path.exists(vert_path) and os.path.exists(ins_path) and os.path.exists(sem_path)):
        return ("miss", scene_id, f"missing npy in {npy_dir}")

    try:
        # ====== 原有主体逻辑放到 try 里 ======
        vert = np.load(vert_path)  # (N,9)
        inst_full = np.load(ins_path).astype(np.int32, copy=False)
        sem_raw_full = np.load(sem_path).astype(np.int32, copy=False)
        bbox = np.load(bbox_path) if os.path.exists(bbox_path) else None

        xyz_full = vert[:, :3].astype(np.float32, copy=False)
        rgb_full = vert[:, 3:6]
        if rgb_full.dtype != np.uint8:
            rgb_full = np.clip(rgb_full, 0, 255).astype(np.uint8)
        normal_full = vert[:, 6:9].astype(np.float32, copy=False)

        sem_cont_full = map_sem_to_contiguous(sem_raw_full)

        seed = stable_int_seed(scene_id, base_seed)

        sel = foreground_quota_sampling(
            xyz_full, rgb_full, normal_full, sem_cont_full, inst_full,
            num_points=num_points,
            fg_ratio=fg_ratio,
            per_inst_min=per_inst_min,
            per_inst_max=per_inst_max,
            seed=seed,
            bg_replace=bg_replace,
        )

        xyz = xyz_full[sel]
        rgb = rgb_full[sel]
        normal = normal_full[sel]
        sem_cont = sem_cont_full[sel]
        inst_out = inst_full[sel].astype(np.int16, copy=False)

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

        os.makedirs(out_dir, exist_ok=True)
        torch.save(out, out_path)
        return ("ok", scene_id, "")
    except Exception as e:
        # 这里把错误字符串带回去
        return ("fail", scene_id, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_dir", required=True)
    ap.add_argument("--scene_list_txt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--num_points", type=int, default=40960)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fg_ratio", type=float, default=0.95)
    ap.add_argument("--per_inst_min", type=int, default=256)
    ap.add_argument("--per_inst_max", type=int, default=4096)
    ap.add_argument("--bg_replace", action="store_true")

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    scene_ids = read_scene_ids(args.scene_list_txt)
    if args.limit and args.limit > 0:
        scene_ids = scene_ids[:args.limit]

    workers = max(1, int(args.workers)) if args.workers else min(cpu_count(), 16)

    ok = miss = fail = skip = 0
    missing_samples = []

    pbar = tqdm(total=len(scene_ids), desc=f"NPY->PTH ({os.path.basename(args.out_dir)})")

    if workers <= 1:
        for sid in scene_ids:
            st, sid, msg = convert_one(
                sid, args.npy_dir, args.out_dir,
                args.num_points, args.fg_ratio, args.per_inst_min, args.per_inst_max,
                args.seed, args.bg_replace, args.overwrite
            )
            if st == "ok": ok += 1
            elif st == "miss":
                miss += 1
                if len(missing_samples) < 10:
                    missing_samples.append(sid)
            elif st == "skip": skip += 1
            else: fail += 1
            pbar.update(1)
        pbar.close()
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_to_sid = {
                ex.submit(
                    convert_one, sid, args.npy_dir, args.out_dir,
                    args.num_points, args.fg_ratio, args.per_inst_min, args.per_inst_max,
                    args.seed, args.bg_replace, args.overwrite
                ): sid
                for sid in scene_ids
            }

            for fut in as_completed(future_to_sid):
                sid = future_to_sid[fut]
                try:
                    st, sid_ret, msg = fut.result()
                    if st == "ok":
                        ok += 1
                    elif st == "miss":
                        miss += 1
                        if len(missing_samples) < 10:
                            missing_samples.append(sid_ret)
                        print(f"[MISS] scene={sid_ret} msg={msg}")
                    elif st == "skip":
                        skip += 1
                    elif st == "fail":
                        fail += 1
                        print(f"[FAIL] scene={sid_ret} msg={msg}")
                except Exception as e:
                    fail += 1
                    print(f"[FAIL-EXC] scene={sid} exc={repr(e)}")
                pbar.update(1)
            pbar.close()


    print(f"[DONE] ok={ok} skip={skip} miss_npy={miss} fail={fail}")
    if missing_samples:
        print("[MISS SAMPLE]", missing_samples)
    print(f"out_dir: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
