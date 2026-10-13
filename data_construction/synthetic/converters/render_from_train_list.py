import blenderproc as bproc
import os
import argparse
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import traceback

# Blender 内置（BlenderProc 运行环境可用）
import bpy
import mathutils
import inspect

# ============================================================
# Patch 1) unknown/unknow label -> fallback id（最稳）
# ============================================================
def patch_label_mapping_unknown(label_mapping, fallback_label="other", fallback_id=0):
    """
    Monkey patch: label_mapping.id_from_label
    - 遇到 KeyError（unknown/unknow/任意缺失label）时，返回 fallback_label 的 id；若 fallback_label 不存在，返回 fallback_id
    """
    orig = label_mapping.id_from_label

    # 预先解析 fallback id
    resolved_fallback_id = fallback_id
    try:
        resolved_fallback_id = orig(fallback_label)
        print(f"[Patch] Label fallback: '{fallback_label}' -> id={resolved_fallback_id}")
    except Exception:
        print(f"[Patch] Label fallback: '{fallback_label}' not found. Use id={fallback_id}")
        resolved_fallback_id = fallback_id

    def safe_id_from_label(label: str):
        if label is None:
            return resolved_fallback_id
        key = str(label).strip().lower()
        # 常见拼写容错
        if key == "unknow":
            key = "unknown"
        try:
            return orig(key)
        except KeyError:
            # 兜底：所有未知 label 都归到 fallback
            return resolved_fallback_id
        except Exception:
            return resolved_fallback_id

    # 绑定到实例
    label_mapping.id_from_label = safe_id_from_label
    return label_mapping


# ============================================================
# Patch 2) texture 缺失 -> dummy image（允许无纹理也能加载）
# ============================================================
def patch_front3dloader_texture_fallback():
    """
    Monkey patch BlenderProc Front3DLoader:
    - 当纹理数量为 0 / 或原逻辑抛异常时，返回一个 1x1 白色 dummy 纹理，避免崩溃
    """
    try:
        from blenderproc.python.loader.Front3DLoader import _Front3DLoader
    except Exception as e:
        print(f"[Patch][Warn] Cannot import _Front3DLoader: {e}. Skip texture patch.")
        return

    if not hasattr(_Front3DLoader, "get_used_image"):
        print("[Patch][Warn] _Front3DLoader.get_used_image not found. Skip texture patch.")
        return

    orig_get_used_image = _Front3DLoader.get_used_image

    def _dummy_white_image(name="BPROC_DUMMY_WHITE_1x1"):
        # 若已经创建过，复用
        if name in bpy.data.images:
            return bpy.data.images[name]
        img = bpy.data.images.new(name=name, width=1, height=1, alpha=True, float_buffer=False)
        # RGBA = (1,1,1,1)
        img.pixels = [1.0, 1.0, 1.0, 1.0]
        img.pack()
        return img

    def safe_get_used_image(hash_folder, saved_images):
        try:
            img = orig_get_used_image(hash_folder, saved_images)
            return img
        except Exception as e:
            msg = str(e)
            # 典型报错：There is not just one texture: 0
            if "There is not just one texture" in msg:
                # 纹理为0：返回 dummy
                print(f"[Patch][Texture] Missing/ambiguous texture in '{hash_folder}'. Use dummy texture.")
                return _dummy_white_image()
            # 其它异常也给兜底，避免整个 scene 崩
            print(f"[Patch][Texture][Warn] get_used_image failed: {msg[:200]}... Use dummy texture.")
            return _dummy_white_image()

    _Front3DLoader.get_used_image = staticmethod(safe_get_used_image)
    print("[Patch] Front3DLoader texture fallback enabled.")


# -------------------------
# 新增：分片函数（稳定切分场景列表）
# -------------------------
def shard_scene_ids(scene_ids, num_splits: int, split: int):
    if num_splits <= 1:
        return scene_ids
    n = len(scene_ids)
    if n == 0:
        return []

    per_split = n // num_splits
    remainder = n % num_splits

    if split < remainder:
        start = split * (per_split + 1)
        end = start + (per_split + 1)
    else:
        start = remainder * (per_split + 1) + (split - remainder) * per_split
        end = start + per_split

    start = max(0, start)
    end = min(n, end)
    return scene_ids[start:end]


# -------------------------
# 兼容不同 BlenderProc 版本
# -------------------------
def _safe_set_render_devices(device: str):
    device = device.upper()
    if not hasattr(bproc.renderer, "set_render_devices"):
        print("[Warn] bproc.renderer.set_render_devices not found, use default device selection.")
        return
    fn = bproc.renderer.set_render_devices
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        if "use_only_cpu" in params:
            fn(use_only_cpu=(device == "CPU"))
            return
        if len(params) == 0:
            fn()
            return
        print("[Warn] set_render_devices signature not recognized, skip explicit device setting.")
    except Exception as e:
        print(f"[Warn] set_render_devices failed: {e}. Skip explicit device setting.")


def _safe_set_samples(samples: int):
    if hasattr(bproc.renderer, "set_max_amount_of_samples"):
        bproc.renderer.set_max_amount_of_samples(samples)
        return
    if hasattr(bproc.renderer, "set_render_samples"):
        bproc.renderer.set_render_samples(samples)
        return
    print("[Warn] No known sample-setting API found. Use default samples.")


# -------------------------
# 场景 bbox & 补光（增强兜底）
# -------------------------
def _compute_scene_bbox_world():
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects in scene after loading.")

    for obj in mesh_objs:
        mw = obj.matrix_world
        if not obj.data.vertices:
            print(f"[Warn] 跳过空mesh: {obj.name}")
            continue
        for v in obj.bound_box:
            vw = mw @ mathutils.Vector(v)
            p = np.array([vw.x, vw.y, vw.z], dtype=np.float64)
            if np.any(np.isnan(p)) or np.any(np.isinf(p)):
                print(f"[Warn] 跳过异常坐标: {obj.name} -> {p}")
                continue
            mins = np.minimum(mins, p)
            maxs = np.maximum(maxs, p)

    # 检测并交换异常的max/min
    for i in range(3):
        if maxs[i] < mins[i]:
            print(f"[Fix] 轴{i}(x=0/y=1/z=2)的max<min，交换值: {maxs[i]} < {mins[i]}")
            mins[i], maxs[i] = maxs[i], mins[i]

    extent = (maxs - mins)
    if np.any(extent <= 0) or np.any(np.isinf(mins)) or np.any(np.isinf(maxs)):
        print(f"[Warn] bbox范围异常: mins={mins}, maxs={maxs}，使用更大的默认范围")
        mins = np.array([-10.0, -10.0, 0.0], dtype=np.float64)
        maxs = np.array([10.0, 10.0, 10.0], dtype=np.float64)

    center = (mins + maxs) / 2.0
    extent = (maxs - mins)
    radius = float(np.linalg.norm(extent) / 2.0 + 1e-6)
    return center.astype(np.float32), float(radius), mins.astype(np.float32), maxs.astype(np.float32)


def _ensure_lights(scene_center, radius):
    has_light = any(o.type == "LIGHT" for o in bpy.context.scene.objects)
    if has_light:
        return

    def add_area_light(name, loc, energy=1500.0, size=2.0):
        light_data = bpy.data.lights.new(name=name, type="AREA")
        light_data.energy = energy
        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = loc
        light_obj.scale = (size, size, size)
        return light_obj

    c = scene_center
    r = float(radius)
    add_area_light("AreaKey",  (c[0] + 1.2*r, c[1] - 1.2*r, c[2] + 1.2*r), energy=2000.0, size=3.0)
    add_area_light("AreaFill", (c[0] - 1.2*r, c[1] - 0.8*r, c[2] + 0.8*r), energy=1200.0, size=3.0)
    add_area_light("AreaTop",  (c[0],        c[1],        c[2] + 1.8*r),     energy=1800.0, size=4.0)


# -------------------------
# 采样候选相机位姿（bbox 内）
# Patch: 强制排序 + 退化区间处理 + margin 截断
# -------------------------
def _sample_candidate_poses_inside_bbox(bb_min, bb_max, num_candidates, seed=0,
                                        cam_height=1.4,
                                        margin_ratio=0.08):
    rng = np.random.default_rng(seed)
    bb_min = np.asarray(bb_min, dtype=np.float32)
    bb_max = np.asarray(bb_max, dtype=np.float32)

    # 1) 强制排序，避免 min/max 反了
    bb_lo = np.minimum(bb_min, bb_max)
    bb_hi = np.maximum(bb_min, bb_max)

    center = (bb_lo + bb_hi) / 2.0
    extent = (bb_hi - bb_lo)

    # 2) margin 截断：margin 不允许超过半尺寸（否则 x_min > x_max）
    def _clamp_margin(axis_len, margin):
        # 留一点 epsilon，避免完全相等导致 uniform 报错
        half = float(axis_len) / 2.0
        return float(min(margin, max(0.0, half - 1e-4)))

    base_margin = float(max(extent[0], extent[1]) * margin_ratio)
    mx = _clamp_margin(extent[0], base_margin)
    my = _clamp_margin(extent[1], base_margin)

    x_min = float(bb_lo[0] + mx)
    x_max = float(bb_hi[0] - mx)
    y_min = float(bb_lo[1] + my)
    y_max = float(bb_hi[1] - my)

    # 3) 如果仍退化（例如 extent=0），则退化为固定值，不再 uniform
    def _safe_uniform(lo, hi):
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        if (hi - lo) < 1e-6:
            return float(lo)
        return float(rng.uniform(lo, hi))

    # 人眼高度：尽量在 bbox 内
    cam_z = float(bb_lo[2] + cam_height)
    if bb_hi[2] > bb_lo[2] + 0.2:
        cam_z = float(np.clip(cam_z, bb_lo[2] + 0.05, bb_hi[2] - 0.05))
    else:
        cam_z = float(bb_lo[2] + 0.3)

    poses = []
    for _ in range(int(num_candidates)):
        x = _safe_uniform(x_min, x_max)
        y = _safe_uniform(y_min, y_max)
        location = np.array([x, y, cam_z], dtype=np.float32)

        tgt = center.copy()
        tgt[2] = cam_z
        jitter = rng.normal(0.0, 0.12, size=(2,)).astype(np.float32) * max(
            0.2, float(min(extent[0], extent[1]) * 0.25)
        )
        tgt[0] += float(jitter[0])
        tgt[1] += float(jitter[1])

        forward = (tgt - location).astype(np.float32)
        rotation = bproc.camera.rotation_from_forward_vec(forward)
        cam2world = bproc.math.build_transformation_mat(location, rotation)
        poses.append(cam2world)
    return poses


# -------------------------
# 碰撞/贴墙过滤（raycast）
# -------------------------
def _scene_raycast(origin, direction, max_dist=1000.0):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    o = mathutils.Vector(origin)
    d = mathutils.Vector(direction).normalized()
    hit, loc, normal, face_index, obj, matrix = bpy.context.scene.ray_cast(depsgraph, o, d, distance=max_dist)
    return hit, loc, normal, obj


def _pose_is_valid(cam2world, target_point,
                   inside_eps=0.03,
                   min_forward_hit=0.25,
                   min_clearance=0.10):
    R = cam2world[:3, :3]
    t = cam2world[:3, 3]

    forward = -R[:, 2]
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    hit, loc, normal, obj = _scene_raycast(t, forward, max_dist=10.0)
    if hit:
        dist = float((np.array([loc.x, loc.y, loc.z], dtype=np.float32) - t).dot(forward))
        if dist < min_forward_hit:
            return False

    dirs = [
        np.array([1, 0, 0], np.float32), np.array([-1, 0, 0], np.float32),
        np.array([0, 1, 0], np.float32), np.array([0, -1, 0], np.float32),
        np.array([0, 0, 1], np.float32), np.array([0, 0, -1], np.float32),
    ]
    near_hits = 0
    for d in dirs:
        hit, loc, normal, obj = _scene_raycast(t, d, max_dist=2.0)
        if hit:
            dist = float(np.linalg.norm(np.array([loc.x, loc.y, loc.z], dtype=np.float32) - t))
            if dist < inside_eps:
                return False
            if dist < min_clearance:
                near_hits += 1
    if near_hits >= 4:
        return False

    to_target = (target_point - t).astype(np.float32)
    to_target = to_target / (np.linalg.norm(to_target) + 1e-8)
    if float(np.dot(forward, to_target)) < 0.2:
        return False

    return True


# -------------------------
# 选择位姿（修复 numpy in 判等）
# -------------------------
def _pose_in_list(pose, pose_list, atol=1e-6):
    for p in pose_list:
        if np.allclose(pose, p, atol=atol):
            return True
    return False


def _select_poses(candidates, num_views, target_point, seed=0,
                  relax_steps=5):
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    selected = []
    configs = [
        dict(inside_eps=0.03, min_forward_hit=0.35, min_clearance=0.12),
        dict(inside_eps=0.04, min_forward_hit=0.25, min_clearance=0.10),
        dict(inside_eps=0.05, min_forward_hit=0.18, min_clearance=0.08),
        dict(inside_eps=0.06, min_forward_hit=0.15, min_clearance=0.06),
        dict(inside_eps=0.08, min_forward_hit=0.10, min_clearance=0.04),
    ]
    configs = configs[:max(1, min(relax_steps, len(configs)))]

    for cfg in configs:
        for cam2world in candidates:
            if len(selected) >= num_views:
                break
            if cam2world is None:
                continue
            if _pose_is_valid(cam2world, target_point, **cfg):
                selected.append(cam2world)
        if len(selected) >= num_views:
            break

    # 还不够：直接补齐（最后兜底）
    if len(selected) < num_views:
        for cam2world in candidates:
            if len(selected) >= num_views:
                break
            if cam2world is None:
                continue
            if not _pose_in_list(cam2world, selected):
                selected.append(cam2world)

    return selected[:num_views]


def _save_pose_txt(path, mat4x4: np.ndarray):
    np.savetxt(path, mat4x4.astype(np.float32), fmt="%.8f")


# -------------------------
# 检查已有有效视图
# -------------------------
def _get_existing_valid_views(color_dir, depth_dir, pose_dir):
    existing_indices = set()

    if os.path.exists(color_dir):
        for f in os.listdir(color_dir):
            if f.endswith(".jpg"):
                try:
                    idx = int(f.split(".")[0])
                    existing_indices.add(idx)
                except ValueError:
                    continue

    valid_indices = []
    for idx in sorted(existing_indices):
        depth_file = os.path.join(depth_dir, f"{idx}.png")
        pose_file = os.path.join(pose_dir, f"{idx}.txt")
        if os.path.exists(depth_file) and os.path.exists(pose_file):
            valid_indices.append(idx)

    existing_num = len(valid_indices)
    next_start_idx = existing_num
    return existing_num, next_start_idx


def _already_done(scene_out: str, num_views: int):
    color_dir = os.path.join(scene_out, "color")
    depth_dir = os.path.join(scene_out, "depth")
    pose_dir = os.path.join(scene_out, "pose")
    existing_num, _ = _get_existing_valid_views(color_dir, depth_dir, pose_dir)
    return existing_num >= num_views


# -------------------------
# scene_id -> json path 索引
# -------------------------
def _index_front_jsons(front_dir: str):
    mapping = {}
    for p in Path(front_dir).rglob("*.json"):
        mapping[p.stem] = str(p)
    return mapping


def _read_id_list(list_path: str):
    ids = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            sid = s.split()[0]
            ids.append(sid)

    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def render_one_scene(front_json: str, args, mapping, max_retry=2):
    scene_id = Path(front_json).stem
    scene_out = os.path.join(args.out_root, scene_id)

    if args.skip_if_done and _already_done(scene_out, args.num_views):
        print(f"[Skip] {scene_id} already done (existing >= {args.num_views}).")
        return

    color_dir = os.path.join(scene_out, "color")
    depth_dir = os.path.join(scene_out, "depth")
    pose_dir = os.path.join(scene_out, "pose")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    existing_num, next_start_idx = _get_existing_valid_views(color_dir, depth_dir, pose_dir)
    need_generate = args.num_views - existing_num
    print(f"\n===== 场景 {scene_id} 渲染状态 =====")
    print(f"目标视图总数      : {args.num_views} 个")
    print(f"已有有效视图数    : {existing_num} 个")
    print(f"本次需补充渲染数  : {need_generate} 个")

    if need_generate <= 0:
        unrendered_num = 0
        print(f"未成功渲染视图数  : {unrendered_num} 个")
        print(f"=====================================")
        meta_file = os.path.join(scene_out, "meta.txt")
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_lines = f.readlines()
            with open(meta_file, "w", encoding="utf-8") as f:
                for line in meta_lines:
                    if line.startswith("final_views:"):
                        f.write(f"final_views: {existing_num}\n")
                    elif line.startswith("unrendered_views:"):
                        f.write(f"unrendered_views: {unrendered_num}\n")
                    elif line.startswith("num_views:"):
                        f.write(f"num_views: {args.num_views}\n")
                    else:
                        f.write(line)
        return

    for retry in range(max_retry + 1):
        try:
            bproc.clean_up()
            bproc.camera.set_resolution(args.W, args.H)

            fx_feat, fy_feat, cx_feat, cy_feat = 37.01983, 38.52470, 20.0, 15.5
            K = np.array([
                [fx_feat * 8.0, 0.0,           cx_feat * 8.0],
                [0.0,           fy_feat * 8.0, cy_feat * 8.0],
                [0.0,           0.0,           1.0],
            ], dtype=np.float32)
            try:
                bproc.camera.set_intrinsics_from_K_matrix(K, args.W, args.H)
            except Exception:
                from blenderproc.python.camera import CameraUtility
                CameraUtility.set_intrinsics_from_K_matrix(K, args.W, args.H)

            # load scene（mapping 已经被 patch：unknown->fallback）
            bproc.loader.load_front3d(
                json_path=front_json,
                future_model_path=args.future_folder,
                front_3D_texture_path=args.front_3D_texture_path,
                label_mapping=mapping,
            )

            center, radius, bb_min, bb_max = _compute_scene_bbox_world()
            _ensure_lights(center, radius)

            if bpy.context.scene.camera is not None:
                bpy.context.scene.camera.data.clip_start = args.clip_start
                bpy.context.scene.camera.data.clip_end = args.clip_end
            else:
                for cam in [o for o in bpy.context.scene.objects if o.type == "CAMERA"]:
                    cam.data.clip_start = args.clip_start
                    cam.data.clip_end = args.clip_end

            num_candidates = max(args.num_candidates, need_generate * 2)
            candidates = _sample_candidate_poses_inside_bbox(
                bb_min=bb_min,
                bb_max=bb_max,
                num_candidates=num_candidates,
                seed=args.seed + existing_num + retry,
                cam_height=args.cam_height,
                margin_ratio=args.margin_ratio,
            )

            selected = _select_poses(
                candidates=candidates,
                num_views=need_generate,
                target_point=center,
                seed=args.seed + existing_num + retry,
                relax_steps=args.relax_steps if hasattr(args, "relax_steps") else 5,
            )

            for cam2world in selected:
                bproc.camera.add_camera_pose(cam2world)

            num_render = len(selected)
            print(f"[Render] scene={scene_id} retry={retry} candidates={num_candidates} -> selected={num_render} (需补充{need_generate})")
            data = bproc.renderer.render()
            colors = data["colors"]
            depths = data["depth"]

            for i in range(num_render):
                current_idx = next_start_idx + i

                c = colors[i]
                if c.dtype != np.uint8:
                    c = np.clip(c, 0.0, 1.0)
                    c = (c * 255.0).astype(np.uint8)
                if c.shape[-1] == 4:
                    c = c[..., :3]
                imageio.imwrite(os.path.join(color_dir, f"{current_idx}.jpg"), c)

                d = depths[i]
                d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                d_mm = np.clip(d * 1000.0, 0.0, 65535.0).astype(np.uint16)
                imageio.imwrite(os.path.join(depth_dir, f"{current_idx}.png"), d_mm)

                _save_pose_txt(os.path.join(pose_dir, f"{current_idx}.txt"), selected[i])

            final_total = existing_num + num_render
            unrendered_num = args.num_views - final_total

            np.savetxt(os.path.join(scene_out, "K.txt"), K, fmt="%.8f")
            meta_file = os.path.join(scene_out, "meta.txt")

            meta_lines = []
            if os.path.exists(meta_file):
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_lines = f.readlines()

            with open(meta_file, "w", encoding="utf-8") as f:
                updated = False
                for line in meta_lines:
                    if line.startswith("final_views:"):
                        f.write(f"final_views: {final_total}\n")
                        updated = True
                    elif line.startswith("unrendered_views:"):
                        f.write(f"unrendered_views: {unrendered_num}\n")
                        updated = True
                    elif line.startswith("num_views:"):
                        f.write(f"num_views: {args.num_views}\n")
                    elif line.startswith("existing_views:"):
                        f.write(f"existing_views: {existing_num}\n")
                    else:
                        f.write(line)
                if not meta_lines or not updated:
                    f.write(f"scene_id: {scene_id}\n")
                    f.write(f"front_json: {front_json}\n")
                    f.write(f"resolution: {args.W}x{args.H}\n")
                    f.write(f"clip: {args.clip_start} ~ {args.clip_end}\n")
                    f.write(f"num_candidates: {num_candidates}\n")
                    f.write(f"num_views: {args.num_views}\n")
                    f.write(f"existing_views: {existing_num}\n")
                    f.write(f"added_views: {num_render}\n")
                    f.write(f"final_views: {final_total}\n")
                    f.write(f"unrendered_views: {unrendered_num}\n")
                    f.write(f"bbox_min: {bb_min.tolist()}\n")
                    f.write(f"bbox_max: {bb_max.tolist()}\n")

            print(f"===== 场景 {scene_id} 渲染完成 (retry={retry}) =====")
            print(f"目标视图总数      : {args.num_views} 个")
            print(f"已有有效视图数    : {existing_num} 个")
            print(f"本次成功补充数    : {num_render} 个")
            print(f"最终总视图数      : {final_total} 个")
            print(f"未成功渲染视图数  : {unrendered_num} 个")
            print(f"=====================================")
            if unrendered_num > 0:
                print(f"[Warn] 存在 {unrendered_num} 个视图未成功渲染，建议增大 --num_candidates")
            print(f"[OK] wrote frames_square: {scene_out}")
            break

        except Exception as e:
            if retry < max_retry:
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                print(f"[Retry] scene_id={scene_id} 第{retry+1}次失败，重试... (错误：{error_msg[:200]}...)")
                bproc.clean_up()
                continue
            else:
                raise e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_list", type=str, required=True)
    parser.add_argument("--front_dir", type=str, required=True)
    parser.add_argument("--future_folder", type=str, required=True)
    parser.add_argument("--front_3D_texture_path", type=str, required=True)
    parser.add_argument("--mapping_csv", type=str, default=None)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--skip_if_done", action="store_true")

    parser.add_argument("--num_views", type=int, default=150)
    parser.add_argument("--num_candidates", type=int, default=200)

    parser.add_argument("--W", type=int, default=328)
    parser.add_argument("--H", type=int, default=256)

    parser.add_argument("--render_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--cam_height", type=float, default=1.4)
    parser.add_argument("--margin_ratio", type=float, default=0.08)

    parser.add_argument("--clip_start", type=float, default=0.1)
    parser.add_argument("--clip_end", type=float, default=30.0)

    parser.add_argument("--num_splits", type=int, default=1, help="总分片数（如20）")
    parser.add_argument("--split", type=int, default=0, help="当前分片索引（0-based，0~num_splits-1）")

    parser.add_argument("--fail_log", type=str, default=None, help="失败场景日志保存路径")
    parser.add_argument("--relax_steps", type=int, default=5, help="位姿筛选的放宽步骤数")

    args = parser.parse_args()

    if args.num_splits < 1:
        raise ValueError(f"num_splits 必须 >=1，当前值：{args.num_splits}")
    if args.split < 0 or args.split >= args.num_splits:
        raise ValueError(f"split 必须在 0~{args.num_splits-1} 范围内，当前值：{args.split}")

    fail_log_path = args.fail_log if args.fail_log else os.path.join(args.out_root, f"failed_scenes_split_{args.split}.log")
    failed_scenes = []

    # ✅ 全局 init 一次
    bproc.init()
    _safe_set_render_devices(args.device)
    _safe_set_samples(args.render_samples)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    # ===== Patch texture fallback（允许无纹理）=====
    patch_front3dloader_texture_fallback()

    # 加载label映射
    if args.mapping_csv is None:
        mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    else:
        mapping_file = args.mapping_csv
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    # ===== Patch unknown label fallback（unknown/unknow -> other/0）=====
    mapping = patch_label_mapping_unknown(mapping, fallback_label="other", fallback_id=0)

    json_index = _index_front_jsons(args.front_dir)
    scene_ids = _read_id_list(args.id_list)

    total_scenes = len(scene_ids)
    scene_ids_sharded = shard_scene_ids(scene_ids, args.num_splits, args.split)
    sharded_num = len(scene_ids_sharded)

    print(f"[Info] 总分片数: {args.num_splits} | 当前分片索引: {args.split}")
    print(f"[Info] 原始场景总数: {total_scenes} | 本分片处理场景数: {sharded_num}")
    if sharded_num > 0:
        print(f"[Info] 本分片处理的场景范围: {scene_ids_sharded[0]} ~ {scene_ids_sharded[-1]}")

    missing = 0
    failed = 0
    total_unrendered = 0
    for sid in scene_ids_sharded:
        front_json = json_index.get(sid, None)
        if front_json is None:
            cand = os.path.join(args.front_dir, f"{sid}.json")
            if os.path.exists(cand):
                front_json = cand
        if front_json is None:
            print(f"[Missing] json not found for scene_id={sid}")
            missing += 1
            failed_scenes.append((sid, f"JSON文件缺失，候选路径：{cand}"))
            continue

        try:
            render_one_scene(front_json, args, mapping, max_retry=2)

            scene_out = os.path.join(args.out_root, sid)
            meta_file = os.path.join(scene_out, "meta.txt")
            if os.path.exists(meta_file):
                with open(meta_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("unrendered_views:"):
                            unrendered = int(line.strip().split(":")[1].strip())
                            total_unrendered += unrendered
                            break
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[Error] scene_id={sid} 最终失败: {error_msg[:500]}...")
            failed += 1
            failed_scenes.append((sid, error_msg))

    if failed_scenes:
        with open(fail_log_path, "w", encoding="utf-8") as f:
            f.write(f"分片 {args.split}/{args.num_splits} 失败场景列表（共{len(failed_scenes)}个）:\n")
            f.write("="*80 + "\n")
            for idx, (sid, err) in enumerate(failed_scenes, 1):
                f.write(f"\n[{idx}] scene_id: {sid}\n")
                f.write(f"错误信息: {err}\n")
                f.write("-"*80 + "\n")
        print(f"\n[Info] 失败场景日志已保存到：{fail_log_path}")

    print(f"\n==================== 分片 {args.split}/{args.num_splits} 处理总结 ====================")
    print(f"本分片总场景数    : {sharded_num}")
    print(f"缺失JSON文件数    : {missing}")
    print(f"渲染失败场景数    : {failed}")
    print(f"本分片未成功渲染视图数: {total_unrendered}")
    print(f"===========================================================================")


if __name__ == "__main__":
    main()
