import os
import torch
import numpy as np
import argparse
import tempfile
import json
import math
from collections import defaultdict
from PIL import Image, ImageFilter

from data_construction.real.object_render.render_util import StrictViewRenderer  # 确保 render_util.py 在同级目录下


# -----------------------------
# Utils
# -----------------------------
def load_json_list(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON 顶层必须是 list，但得到的是: {type(data)}")
    return data


def aabb_from_6d(arr6):
    """
    你的 bbox/aabb 是 [xmin, ymin, zmin, xmax, ymax, zmax]
    """
    if arr6 is None:
        return None, None
    if len(arr6) != 6:
        raise ValueError(f"期望长度为 6 的 aabb，但得到 len={len(arr6)}: {arr6}")
    xyz_min = arr6[:3]
    xyz_max = arr6[3:]
    return xyz_min, xyz_max


def group_by_scene(items):
    grouped = defaultdict(list)
    for it in items:
        sid = it.get("scene_id", None)
        if sid is None:
            continue
        grouped[sid].append(it)
    return grouped


# -----------------------------
# Scene-level renderer cache
# -----------------------------
class SceneRendererCache:
    """
    一个 scene_id 只初始化一次 StrictViewRenderer，后续所有 bbox 复用它。
    可设置 max_cache_size 以避免缓存过多 scene 占显存（简单 LRU）。
    """
    def __init__(self, dataset_root: str, device: str, max_cache_size: int = 8):
        self.dataset_root = dataset_root
        self.device = device
        self.max_cache_size = max_cache_size
        self._cache = {}          # scene_id -> renderer
        self._lru = []            # scene_id list, newest at end

    def _touch(self, scene_id: str):
        if scene_id in self._lru:
            self._lru.remove(scene_id)
        self._lru.append(scene_id)

    def _evict_if_needed(self):
        while len(self._lru) > self.max_cache_size:
            old = self._lru.pop(0)
            r = self._cache.pop(old, None)
            try:
                del r
            except Exception:
                pass
            torch.cuda.empty_cache()
            print(f"🧹 Evicted renderer for scene {old}")

    def get(self, scene_id: str) -> StrictViewRenderer:
        if scene_id in self._cache:
            self._touch(scene_id)
            return self._cache[scene_id]

        scene_dir = os.path.join(self.dataset_root, scene_id)
        mesh_path = os.path.join(scene_dir, f"{scene_id}_vh_clean.ply")
        meta_path = os.path.join(scene_dir, f"{scene_id}.txt")

        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"网格文件不存在: {mesh_path}")

        # dummy json to satisfy renderer API
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"objects_info": []}, f)
            dummy_json = f.name

        renderer = StrictViewRenderer(
            mesh_path=mesh_path,
            bbox_json_path=dummy_json,
            meta_file_path=meta_path,
            scan_id=scene_id,
            device=self.device,
        )
        os.remove(dummy_json)

        self._cache[scene_id] = renderer
        self._touch(scene_id)
        self._evict_if_needed()

        print(f"✅ Renderer initialized & cached for scene {scene_id}")
        return renderer


# -----------------------------
# Rendering
# -----------------------------
def render_custom_bbox_with_renderer(
    renderer: StrictViewRenderer,
    scene_id: str,
    xyz_min,
    xyz_max,
    output_dir: str,
    elevation: float,
    zoom: float,
    view_count: int,
    tag: str = "bbox",        # "pred" or "gt"
    ann_id=None,
    bbox_id=None,
    extra_meta=None,          # dict: best_index/best_iou/pred_idx...
    thickness: int = 6,
    out_res: int = 672,
    crop_min_radius: float = 2.5,
    crop_scale: float = 3.0,
    crop_max_radius: float = 4.5,   # 关键：加上限避免 local_mesh 过大导致 overflow
):
    device = renderer.device if hasattr(renderer, "device") else "cuda"
    bbox_min = np.array(xyz_min, dtype=np.float32)
    bbox_max = np.array(xyz_max, dtype=np.float32)

    center_np = (bbox_min + bbox_max) / 2.0
    size_np = bbox_max - bbox_min
    max_dim = float(np.max(size_np))

    # crop radius（加上限）
    crop_r = max(max_dim * crop_scale, crop_min_radius)
    crop_r = min(crop_r, crop_max_radius)

    center_torch = torch.from_numpy(center_np).float().to(device).unsqueeze(0)
    local_mesh = renderer._crop_scene_mesh(center_torch, crop_r)
    if local_mesh is None:
        print(f"❌ [Error] crop 后 bbox 区域无网格: scene={scene_id}, tag={tag}")
        return

    # -------- view candidates --------
    fov_rad = np.deg2rad(renderer.fov) / 2.0
    base_dist = (max_dim / 2.0 / np.sin(fov_rad)) / 0.9
    hypotenuse_dist = max(base_dist, 0.8) / zoom

    elev_rad = math.radians(elevation)
    z_offset = hypotenuse_dist * math.sin(elev_rad)
    xy_radius = hypotenuse_dist * math.cos(elev_rad)

    candidates = []
    for azim in range(0, 360, 45):
        rad = math.radians(azim)
        eye_pos = np.array(
            [
                center_np[0] + xy_radius * math.cos(rad),
                center_np[1] + xy_radius * math.sin(rad),
                center_np[2] + z_offset,
            ],
            dtype=np.float32,
        )
        dist_to_center = np.linalg.norm(eye_pos - renderer.scene_center)
        candidates.append({"eye": eye_pos, "score": dist_to_center, "azim": azim})

    candidates.sort(key=lambda x: x["score"])
    top_views = candidates[:view_count]

    os.makedirs(output_dir, exist_ok=True)

    from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

    # -------- render loop --------
    for rank_i, view in enumerate(top_views):
        azim = view["azim"]
        eye_pos_torch = torch.from_numpy(view["eye"]).float().to(device).unsqueeze(0)
        R, T = look_at_view_transform(
            eye=eye_pos_torch, at=center_torch, up=((0, 0, 1),), device=device
        )
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=renderer.fov, znear=0.01)

        # try fast -> safe fallback
        try:
            images = renderer.renderer_fast(local_mesh, cameras=cameras)
        except RuntimeError as e:
            # 常见：overflow / bin error / cuda OOM
            print(f"⚠️ renderer_fast failed ({type(e).__name__}): {str(e)[:160]}")
            torch.cuda.empty_cache()
            try:
                images = renderer.renderer_safe(local_mesh, cameras=cameras)
            except RuntimeError as e2:
                print(f"❌ renderer_safe failed too ({type(e2).__name__}): {str(e2)[:160]}")
                torch.cuda.empty_cache()
                return

        img_np = images[0, ..., :3].cpu().detach().numpy()
        img_np = np.power(img_np, 1.0 / 2.2)  # gamma
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        pil_img = renderer._draw_bbox_on_image_thick(
            pil_img, bbox_min, bbox_max, cameras, thickness=thickness
        )
        pil_img = pil_img.resize((out_res, out_res), Image.Resampling.LANCZOS)
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=3))

        ann_str = f"ann{ann_id}" if ann_id is not None else "annNA"
        bbox_str = f"bbox{bbox_id}" if bbox_id is not None else "bboxNA"

        meta_str = ""
        if isinstance(extra_meta, dict) and len(extra_meta) > 0:
            parts = []
            # 只挑短字段
            for k in ["best_index", "best_iou", "pred_idx"]:
                if k in extra_meta and extra_meta[k] is not None:
                    v = extra_meta[k]
                    if isinstance(v, float):
                        parts.append(f"{k}{v:.3f}")
                    else:
                        parts.append(f"{k}{v}")
            if parts:
                meta_str = "_" + "_".join(parts)

        save_name = (
            f"{scene_id}_{ann_str}_{tag}_{bbox_str}"
            f"{meta_str}_rank{rank_i}_azim{azim}_elev{int(elevation)}.jpg"
        )
        save_path = os.path.join(output_dir, save_name)
        pil_img.save(save_path, quality=95)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Render predicted_bboxes and gt aabb from a JSON list (scene cached)")

    parser.add_argument("--json_path", type=str, required=True, help="结果 JSON 文件路径（顶层 list）")
    parser.add_argument("--dataset_root", type=str, default="data/scannet")
    parser.add_argument("--output_root", type=str, default="data/custom_renders_json")

    parser.add_argument("--elevation", type=float, default=45.0)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--view_count", type=int, default=3)

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=-1, help=">0 只处理前 N 条（调试用）")

    # 缓存最多保留多少个 scene 的 renderer（防显存爆）
    parser.add_argument("--max_cached_scenes", type=int, default=6)

    # crop 参数（建议保留默认）
    parser.add_argument("--crop_scale", type=float, default=3.0)
    parser.add_argument("--crop_min_radius", type=float, default=2.5)
    parser.add_argument("--crop_max_radius", type=float, default=4.5)

    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    items = load_json_list(args.json_path)
    if args.max_items > 0:
        items = items[: args.max_items]
    print(f"✅ Loaded {len(items)} items from {args.json_path}")

    # 建议：按 scene 分组，这样同一 scene 的 bbox 连续渲染，cache 命中更稳定
    grouped = group_by_scene(items)
    scene_ids = sorted(grouped.keys())
    print(f"✅ Found {len(scene_ids)} unique scenes")

    cache = SceneRendererCache(
        dataset_root=args.dataset_root,
        device=device,
        max_cache_size=args.max_cached_scenes,
    )

    total = 0
    from tqdm import tqdm
    for scene_id in tqdm(scene_ids):
        # 1) 获取（或初始化）该 scene 的 renderer（只发生一次）
        try:
            renderer = cache.get(scene_id)
        except Exception as e:
            print(f"❌ Skip scene {scene_id} due to init error: {e}")
            continue

        # 2) scene 输出目录
        scene_out_dir = os.path.join(args.output_root, scene_id)
        os.makedirs(scene_out_dir, exist_ok=True)

        # 3) 逐条 ann 渲染 pred + gt
        for it in grouped[scene_id]:
            ann_id = it.get("ann_id", None)
            predicted_bboxes = it.get("predicted_bboxes", []) or []
            true_target_aabb = it.get("true_target_aabb", None)

            best_index = it.get("best_index", None)
            best_iou = it.get("best_iou", None)

            # predicted
            for pred_i, bbox6 in enumerate(predicted_bboxes):
                xyz_min, xyz_max = aabb_from_6d(bbox6)
                render_custom_bbox_with_renderer(
                    renderer=renderer,
                    scene_id=scene_id,
                    xyz_min=xyz_min,
                    xyz_max=xyz_max,
                    output_dir=scene_out_dir,
                    elevation=args.elevation,
                    zoom=args.zoom,
                    view_count=args.view_count,
                    tag="pred",
                    ann_id=ann_id,
                    bbox_id=pred_i,
                    extra_meta={"best_index": best_index, "best_iou": best_iou, "pred_idx": pred_i},
                    crop_scale=args.crop_scale,
                    crop_min_radius=args.crop_min_radius,
                    crop_max_radius=args.crop_max_radius,
                )
                total += 1

            # gt
            if true_target_aabb is not None:
                xyz_min, xyz_max = aabb_from_6d(true_target_aabb)
                render_custom_bbox_with_renderer(
                    renderer=renderer,
                    scene_id=scene_id,
                    xyz_min=xyz_min,
                    xyz_max=xyz_max,
                    output_dir=scene_out_dir,
                    elevation=args.elevation,
                    zoom=args.zoom,
                    view_count=args.view_count,
                    tag="gt",
                    ann_id=ann_id,
                    bbox_id="target",
                    extra_meta={"best_index": best_index, "best_iou": best_iou},
                    crop_scale=args.crop_scale,
                    crop_min_radius=args.crop_min_radius,
                    crop_max_radius=args.crop_max_radius,
                )
                total += 1

        print(f"==== ✅ done scene: {scene_id} | anns={len(grouped[scene_id])} ====")

    print(f"🎉 All done. Render calls: {total}")
    print(f"Outputs saved under: {args.output_root}")


if __name__ == "__main__":
    main()
