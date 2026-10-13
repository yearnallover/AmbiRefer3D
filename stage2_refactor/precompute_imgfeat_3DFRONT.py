import os, json, random
import numpy as np
import torch
from PIL import Image

# from .render import index_scene_renders, pick_views
from stage2_refactor.vision import VisualEncoder
# stage2/render.py
import os, re, random
from typing import Dict, List, Tuple, Optional
# --- single-view naming: <scene_uuid>__obj<id>.png
import os, re
from typing import Dict, List, Tuple

_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
PAT = re.compile(r"^(?P<scene>[^_]+)__obj(?P<obj>\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)

def index_scene_renders_flat(render_root: str) -> Dict[str, Dict[int, str]]:
    """
    Scan render_root and build:
      scene_id -> { obj_id -> filepath }
    Assumes each object has exactly one view.
    """
    out: Dict[str, Dict[int, str]] = {}
    for fn in os.listdir(render_root):
        if not fn.lower().endswith(_IMG_EXT):
            continue
        m = PAT.match(fn)
        if not m:
            continue
        scene_id = m.group("scene")
        obj_id = int(m.group("obj"))
        out.setdefault(scene_id, {})[obj_id] = os.path.join(render_root, fn)
    return out

def pick_views_single(scene_map: Dict[int, str], obj_id: int) -> List[str]:
    """
    Return the single view for this obj.
    """
    fp = scene_map.get(obj_id, None)
    return [fp] if fp is not None else []


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", required=True)
    ap.add_argument("--render_root", required=True)
    ap.add_argument("--out_dir", default="outputs/imgfeat_cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = VisualEncoder(pretrained=True).to(device).eval()

    items = json.load(open(args.json_path, "r"))
    seen = sorted(set([x["scene_id"] for x in items]))

    # build global index once
    global_idx = index_scene_renders_flat(args.render_root)
    from tqdm import tqdm
    for scene_id in tqdm(seen, desc="Encoding scenes"):
        scene_map = global_idx.get(scene_id, {})
        if not scene_map:
            print(f"[warn] no renders for {scene_id} under {args.render_root}")
            continue

        feats = {}
        for obj_id, fp in scene_map.items():
            views = pick_views_single(scene_map, obj_id)  # one view
            if not views:
                continue

            ims = []
            for fp in views:
                im = Image.open(fp).convert("RGB")
                x = enc.preprocess(im)
                ims.append(x)

            x = torch.stack(ims, dim=0).to(device)  # [1,3,224,224]
            with torch.no_grad():
                f = enc(x).mean(dim=0).cpu().numpy().astype(np.float32)  # [768]
            feats[int(obj_id)] = f

        # if not idx:
        #     print(f"[warn] no renders for {scene_id} under {scene_dir}")
        #     continue

        # feats = {}  # obj_id -> feat(768,)
        # for obj_id in idx.keys():
        #     views = pick_views(idx, obj_id, ranks=(0,1,2), prefer_azim=args.prefer_azim, rng=rng)
        #     if not views:
        #         continue

        #     ims = []
        #     for fp in views:
        #         im = Image.open(fp).convert("RGB")
        #         if enc.preprocess is None:
        #             raise RuntimeError("Encoder preprocess is None; set pretrained=True or define preprocess.")
        #         x = enc.preprocess(im)  # tensor [3,224,224] (weights pipeline)
        #         ims.append(x)

        #     x = torch.stack(ims, dim=0).to(device)  # [V,3,224,224]
        #     with torch.no_grad():
        #         f = enc(x).mean(dim=0).cpu().numpy().astype(np.float32)  # [768]
        #     feats[int(obj_id)] = f

        out_path = os.path.join(args.out_dir, f"{scene_id}.npz")
        if len(feats) == 0:
            print(f"[warn] no feats for {scene_id}, skip saving")
            continue

        obj_ids = np.array(sorted(feats.keys()), dtype=np.int32)
        feat_arr = np.stack([feats[k] for k in obj_ids], axis=0).astype(np.float32)  # [N,768]

        np.savez_compressed(out_path, obj_ids=obj_ids, feats=feat_arr)
        print("saved", out_path, "num_obj", len(feats), "dim", feat_arr.shape[1])

if __name__ == "__main__":
    main()
