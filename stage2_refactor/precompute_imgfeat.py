import os, json, random
import numpy as np
import torch
from PIL import Image

# from .render import index_scene_renders, pick_views
from stage2_refactor.vision import VisualEncoder
# stage2/render.py
import os, re, random
from typing import Dict, List, Tuple, Optional

_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
PAT = re.compile(r"^(?P<scene>scene\d+_\d+)_obj(?P<obj>\d+)_.*_rank(?P<rank>\d+)_azim_(?P<azim>-?\d+)", re.IGNORECASE)

def index_scene_renders(scene_dir: str) -> Dict[int, Dict[int, List[Tuple[int, str]]]]:
    """
    Returns: obj_id -> rank -> list of (azim, filepath)
    """
    out: Dict[int, Dict[int, List[Tuple[int, str]]]] = {}
    if not os.path.isdir(scene_dir):
        return out
    for fn in os.listdir(scene_dir):
        if not fn.lower().endswith(_IMG_EXT):
            continue
        m = PAT.match(fn)
        if not m:
            continue
        obj_id = int(m.group("obj"))
        rank = int(m.group("rank"))
        azim = int(m.group("azim"))
        out.setdefault(obj_id, {}).setdefault(rank, []).append((azim, os.path.join(scene_dir, fn)))
    # sort by azim for deterministic choice
    for obj_id in out:
        for rank in out[obj_id]:
            out[obj_id][rank].sort(key=lambda x: x[0])
    return out

def pick_views(index: Dict[int, Dict[int, List[Tuple[int, str]]]],
               obj_id: int,
               ranks=(0,1,2),
               prefer_azim: int = 0,
               rng: Optional[random.Random] = None) -> List[str]:
    """
    Pick one image per rank for this obj_id.
    Strategy: pick prefer_azim if exists else first azim else random.
    """
    rng = rng or random.Random(0)
    out = []
    if obj_id not in index:
        return out
    for r in ranks:
        cand = index[obj_id].get(r, [])
        if not cand:
            continue
        # prefer azim
        best = None
        for az, fp in cand:
            if az == prefer_azim:
                best = fp
                break
        if best is None:
            best = cand[0][1]
        out.append(best)
    return out

import time  # ⭐ 新增

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", required=True)
    ap.add_argument("--render_root", required=True)
    ap.add_argument("--out_dir", default="outputs/imgfeat_cache")
    ap.add_argument("--prefer_azim", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = VisualEncoder(pretrained=True).to(device).eval()

    items = json.load(open(args.json_path, "r"))
    seen = sorted(set([x["scene_id"] for x in items]))

    total_images = 0  # ⭐ 图片计数
    start_time = time.time()  # ⭐ 开始时间

    for scene_id in seen:
        scene_dir = os.path.join(args.render_root, scene_id)
        idx = index_scene_renders(scene_dir)
        if not idx:
            print(f"[warn] no renders for {scene_id} under {scene_dir}")
            continue

        feats = {}
        for obj_id in idx.keys():
            views = pick_views(idx, obj_id, ranks=(0,1,2), prefer_azim=args.prefer_azim, rng=rng)
            if not views:
                continue

            total_images += len(views)  # ⭐ 累加图片数

            ims = []
            for fp in views:
                im = Image.open(fp).convert("RGB")
                x = enc.preprocess(im)
                ims.append(x)

            x = torch.stack(ims, dim=0).to(device)
            with torch.no_grad():
                f = enc(x).mean(dim=0).cpu().numpy().astype(np.float32)
            feats[int(obj_id)] = f

        out_path = os.path.join(args.out_dir, f"{scene_id}.npz")
        if len(feats) == 0:
            print(f"[warn] no feats for {scene_id}, skip saving")
            continue

        obj_ids = np.array(sorted(feats.keys()), dtype=np.int32)
        feat_arr = np.stack([feats[k] for k in obj_ids], axis=0).astype(np.float32)

        np.savez_compressed(out_path, obj_ids=obj_ids, feats=feat_arr)
        print("saved", out_path, "num_obj", len(feats), "dim", feat_arr.shape[1])

    end_time = time.time()  # ⭐ 结束时间

    total_time_ms = (end_time - start_time) * 1000  # ⭐ 转毫秒

    print("\n===== Summary =====")
    print(f"Total images processed: {total_images}")
    print(f"Total time: {total_time_ms:.2f} ms")
    if total_images > 0:
        print(f"Avg time per image: {total_time_ms / total_images:.2f} ms")
if __name__ == "__main__":
    main()
