import blenderproc as bproc
import os
import argparse
from pathlib import Path
import numpy as np
import imageio.v2 as imageio

import bpy
import mathutils
import inspect


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


def _compute_scene_bbox_world():
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects in scene after loading.")

    for obj in mesh_objs:
        mw = obj.matrix_world
        for v in obj.bound_box:
            vw = mw @ mathutils.Vector(v)
            p = np.array([vw.x, vw.y, vw.z], dtype=np.float64)
            mins = np.minimum(mins, p)
            maxs = np.maximum(maxs, p)

    center = (mins + maxs) / 2.0
    extent = (maxs - mins)
    radius = float(np.linalg.norm(extent) / 2.0 + 1e-6)
    return center.astype(np.float32), float(radius), mins.astype(np.float32), maxs.astype(np.float32)


def _sample_camera_poses_inside_bbox(bb_min, bb_max, num_views, seed=0,
                                     cam_height=1.4,
                                     radius_ratio=0.85,
                                     margin_ratio=0.08,
                                     jitter_xy=0.05,
                                     look_at_jitter=0.15):
    rng = np.random.default_rng(seed)

    bb_min = np.asarray(bb_min, dtype=np.float32)
    bb_max = np.asarray(bb_max, dtype=np.float32)
    center = (bb_min + bb_max) / 2.0
    extent = (bb_max - bb_min)

    margin_xy = float(max(extent[0], extent[1]) * margin_ratio)
    half_x = float(extent[0] * 0.5 - margin_xy)
    half_y = float(extent[1] * 0.5 - margin_xy)
    half_xy = max(1e-4, min(half_x, half_y))

    r = half_xy * float(radius_ratio)
    r = max(r, 0.2)

    cam_z = float(bb_min[2] + cam_height)
    cam_z = float(np.clip(
        cam_z,
        bb_min[2] + 0.05,
        bb_max[2] - 0.05 if bb_max[2] > bb_min[2] + 0.1 else bb_min[2] + 0.3
    ))

    cam2world_list = []
    for i in range(num_views):
        theta = 2.0 * np.pi * (i / max(1, num_views))

        x = float(center[0] + r * np.cos(theta))
        y = float(center[1] + r * np.sin(theta))

        x += float(rng.uniform(-jitter_xy, jitter_xy) * r)
        y += float(rng.uniform(-jitter_xy, jitter_xy) * r)

        x = float(np.clip(x, bb_min[0] + margin_xy, bb_max[0] - margin_xy))
        y = float(np.clip(y, bb_min[1] + margin_xy, bb_max[1] - margin_xy))

        location = np.array([x, y, cam_z], dtype=np.float32)

        tgt = center.copy()
        tgt[2] = cam_z
        tgt[:2] += rng.uniform(-look_at_jitter, look_at_jitter, size=(2,)).astype(np.float32) * r

        forward = (tgt - location).astype(np.float32)
        rotation = bproc.camera.rotation_from_forward_vec(forward)
        cam2world = bproc.math.build_transformation_mat(location, rotation)
        cam2world_list.append(cam2world)

    return cam2world_list


def _save_pose_txt(path, mat4x4: np.ndarray):
    np.savetxt(path, mat4x4.astype(np.float32), fmt="%.8f")


# =========================
# BVH + 检测/修复
# =========================

def _build_bvh_tree_from_scene():
    # 优先 bproc 的 multi objects BVH（若可用）
    try:
        objs = bproc.object.get_all_mesh_objects()
        if hasattr(bproc.object, "create_bvh_tree_multi_objects"):
            return bproc.object.create_bvh_tree_multi_objects(objs)
    except Exception:
        pass

    from mathutils.bvhtree import BVHTree

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    verts = []
    tris = []
    vid_offset = 0

    for obj in mesh_objs:
        me = obj.data
        me.calc_loop_triangles()
        mw = obj.matrix_world

        v_world = []
        for v in me.vertices:
            vw = mw @ v.co
            v_world.append((vw.x, vw.y, vw.z))
        verts.extend(v_world)

        for tri in me.loop_triangles:
            a, b, c = tri.vertices
            tris.append((a + vid_offset, b + vid_offset, c + vid_offset))

        vid_offset += len(me.vertices)

    return BVHTree.FromPolygons(verts, tris, all_triangles=True)


def _normalize(v):
    n = float(np.linalg.norm(v) + 1e-8)
    return v / n


def _ray_cast_dist(bvh_tree, origin, direction, max_dist):
    if not hasattr(bvh_tree, "ray_cast"):
        return None
    hit = bvh_tree.ray_cast(origin, direction, max_dist)
    if hit is None:
        return None
    loc = hit[0]
    if loc is None:
        return None
    return float(hit[-1])  # dist 通常在最后


def _is_pose_good(cam2world, bvh_tree, clip_end,
                  min_cam_clearance=0.25,
                  min_view_clearance=0.25,
                  wall_flat_std_min=0.20):
    origin = cam2world[:3, 3].astype(np.float32)

    # 1) 相机点离最近面太近：大概率在墙里/家具里
    if hasattr(bvh_tree, "find_nearest"):
        nearest = bvh_tree.find_nearest(origin)
        if nearest is not None and nearest[0] is not None:
            nearest_dist = float(nearest[-1])
            if nearest_dist < min_cam_clearance:
                return False

    # 2) 视野近障碍检查（若版本支持）
    try:
        if hasattr(bproc.camera, "perform_obstacle_in_view_check"):
            ok = bproc.camera.perform_obstacle_in_view_check(
                cam2world, {"min": float(min_view_clearance)}, bvh_tree
            )
            if not ok:
                return False
    except Exception:
        pass

    # 3) 多射线：避免“整面墙”
    R = cam2world[:3, :3].astype(np.float32)
    forward = _normalize(R @ np.array([0, 0, -1], np.float32))
    right   = _normalize(R @ np.array([1, 0,  0], np.float32))
    up      = _normalize(R @ np.array([0, 1,  0], np.float32))

    rays = [
        forward,
        _normalize(forward + 0.25 * right),
        _normalize(forward - 0.25 * right),
        _normalize(forward + 0.25 * up),
        _normalize(forward - 0.25 * up),
    ]

    dists = []
    for d in rays:
        dist = _ray_cast_dist(bvh_tree, origin, d, float(clip_end))
        if dist is None:
            return False
        dists.append(dist)

    dists = np.array(dists, dtype=np.float32)

    if dists.min() < min_view_clearance:
        return False
    if dists.std() < wall_flat_std_min:
        return False

    return True


def _repair_pose_small_steps(cam2world, bvh_tree, center, bb_min, bb_max, clip_end,
                            seed=0, tries=6,
                            min_cam_clearance=0.25,
                            min_view_clearance=0.25,
                            wall_flat_std_min=0.20):
    rng = np.random.default_rng(seed)
    bb_min = np.asarray(bb_min, np.float32)
    bb_max = np.asarray(bb_max, np.float32)

    def clip_loc(loc):
        margin = 0.10
        loc[0] = float(np.clip(loc[0], bb_min[0] + margin, bb_max[0] - margin))
        loc[1] = float(np.clip(loc[1], bb_min[1] + margin, bb_max[1] - margin))
        loc[2] = float(np.clip(loc[2], bb_min[2] + margin, bb_max[2] - margin))
        return loc

    if _is_pose_good(cam2world, bvh_tree, clip_end,
                     min_cam_clearance, min_view_clearance, wall_flat_std_min):
        return cam2world

    loc = cam2world[:3, 3].astype(np.float32).copy()

    for _ in range(tries):
        # 1) 太近就往中心推
        if hasattr(bvh_tree, "find_nearest"):
            nearest = bvh_tree.find_nearest(loc)
            if nearest is not None and nearest[0] is not None:
                nd = float(nearest[-1])
                if nd < min_cam_clearance:
                    push = (min_cam_clearance - nd + 0.10)
                    dir_to_center = _normalize((center - loc).astype(np.float32))
                    loc = loc + dir_to_center * push
                    loc = clip_loc(loc)

        # 2) 抖动 look-at
        tgt = center.copy().astype(np.float32)
        tgt[2] = loc[2]
        tgt[:2] += rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) * 0.5

        forward = (tgt - loc).astype(np.float32)
        rotation = bproc.camera.rotation_from_forward_vec(forward)
        new_pose = bproc.math.build_transformation_mat(loc, rotation)

        if _is_pose_good(new_pose, bvh_tree, clip_end,
                         min_cam_clearance, min_view_clearance, wall_flat_std_min):
            return new_pose

        # 3) 小幅移动位置（仍是“修复”，不是全局过采样）
        loc[:2] += rng.uniform(-0.3, 0.3, size=(2,)).astype(np.float32)
        loc = clip_loc(loc)

    return None


def _select_diverse_subset(cam2world_list, k, seed=0):
    """从合格 poses 中挑 k 个更分散的（最远点采样 / max-min）"""
    if len(cam2world_list) <= k:
        return cam2world_list

    rng = np.random.default_rng(seed)
    P = np.stack([c[:3, 3] for c in cam2world_list], axis=0).astype(np.float32)
    n = P.shape[0]

    # 起点：随机一个（也可以选离均值最远的）
    start = int(rng.integers(0, n))
    selected = [start]
    min_d = np.full((n,), np.inf, dtype=np.float32)

    for _ in range(1, k):
        last = P[selected[-1]]
        d = np.linalg.norm(P - last[None, :], axis=1).astype(np.float32)
        min_d = np.minimum(min_d, d)
        min_d[selected] = -1.0
        nxt = int(np.argmax(min_d))
        if min_d[nxt] < 0:
            break
        selected.append(nxt)

    return [cam2world_list[i] for i in selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front_json", type=str, required=True)
    parser.add_argument("--future_folder", type=str, required=True)
    parser.add_argument("--front_3D_texture_path", type=str, required=True)
    parser.add_argument("--mapping_csv", type=str, default=None)
    parser.add_argument("--out_root", type=str, required=True)

    parser.add_argument("--num_views", type=int, default=100)

    # 关键新增：先采多少候选
    parser.add_argument("--num_candidates", type=int, default=200)
    parser.add_argument("--resample_batch", type=int, default=120)
    parser.add_argument("--max_resample_rounds", type=int, default=5)

    parser.add_argument("--W", type=int, default=328)
    parser.add_argument("--H", type=int, default=256)

    parser.add_argument("--render_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--clip_start", type=float, default=0.1)
    parser.add_argument("--clip_end", type=float, default=10.0)

    # 检测/修复阈值
    parser.add_argument("--min_cam_clearance", type=float, default=0.25)
    parser.add_argument("--min_view_clearance", type=float, default=0.25)
    parser.add_argument("--wall_flat_std_min", type=float, default=0.20)

    args = parser.parse_args()

    scene_id = Path(args.front_json).stem
    scene_out = os.path.join(args.out_root, scene_id)
    color_dir = os.path.join(scene_out, "color")
    depth_dir = os.path.join(scene_out, "depth")
    pose_dir = os.path.join(scene_out, "pose")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    bproc.clean_up()
    bproc.init()

    _safe_set_render_devices(args.device)
    _safe_set_samples(args.render_samples)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    bproc.camera.set_resolution(args.W, args.H)

    fx_feat, fy_feat, cx_feat, cy_feat = 37.01983, 38.52470, 20.0, 15.5
    K = np.array([
        [fx_feat * 8.0, 0.0,           cx_feat * 8.0],
        [0.0,           fy_feat * 8.0, cy_feat * 8.0],
        [0.0,           0.0,           1.0]
    ], dtype=np.float32)

    try:
        bproc.camera.set_intrinsics_from_K_matrix(K, args.W, args.H)
    except Exception:
        from blenderproc.python.camera import CameraUtility
        CameraUtility.set_intrinsics_from_K_matrix(K, args.W, args.H)

    # load scene
    if args.mapping_csv is None:
        mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    else:
        mapping_file = args.mapping_csv
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    bproc.loader.load_front3d(
        json_path=args.front_json,
        future_model_path=args.future_folder,
        front_3D_texture_path=args.front_3D_texture_path,
        label_mapping=mapping,
    )

    center, radius, bb_min, bb_max = _compute_scene_bbox_world()
    _ensure_lights(center, radius)

    # clip（对所有 camera datablock）
    for cam in bpy.data.cameras:
        cam.clip_start = args.clip_start
        cam.clip_end = args.clip_end

    # build BVH once
    bvh_tree = _build_bvh_tree_from_scene()

    # ===============================
    # 先采很多候选 → 检测/修复 → 凑够 num_views
    # ===============================
    target = int(args.num_views)
    good_poses = []
    seen = set()  # 简单去重：按位置粗量化

    def add_if_good(pose, idx_seed):
        fixed = _repair_pose_small_steps(
            pose, bvh_tree, center, bb_min, bb_max, args.clip_end,
            seed=idx_seed, tries=6,
            min_cam_clearance=args.min_cam_clearance,
            min_view_clearance=args.min_view_clearance,
            wall_flat_std_min=args.wall_flat_std_min
        )
        if fixed is None:
            return False
        if not _is_pose_good(
            fixed, bvh_tree, args.clip_end,
            args.min_cam_clearance, args.min_view_clearance,
            args.wall_flat_std_min
        ):
            return False

        loc = fixed[:3, 3].astype(np.float32)
        key = tuple(np.round(loc, 2).tolist())  # 2cm 粗去重
        if key in seen:
            return False
        seen.add(key)
        good_poses.append(fixed)
        return True

    # Round 0：先采 num_candidates（默认 200）
    num_candidates = max(int(args.num_candidates), target)
    cand = _sample_camera_poses_inside_bbox(
        bb_min=bb_min, bb_max=bb_max, num_views=num_candidates,
        seed=args.seed,
        cam_height=1.4, radius_ratio=0.85, margin_ratio=0.08,
        jitter_xy=0.05, look_at_jitter=0.15
    )
    for i, pose in enumerate(cand):
        add_if_good(pose, idx_seed=args.seed + 10000 + i)
        if len(good_poses) >= target:
            break

    # 后续轮：如果不够，就按批次补采（不是一次性无限过采样）
    round_id = 0
    while len(good_poses) < target and round_id < int(args.max_resample_rounds):
        round_id += 1
        batch = int(args.resample_batch)
        cand2 = _sample_camera_poses_inside_bbox(
            bb_min=bb_min, bb_max=bb_max, num_views=batch,
            seed=args.seed + round_id * 99991,
            cam_height=1.4, radius_ratio=0.85, margin_ratio=0.08,
            jitter_xy=0.08, look_at_jitter=0.25  # 稍微大一点，增加多样性
        )
        for j, pose in enumerate(cand2):
            add_if_good(pose, idx_seed=args.seed + round_id * 200000 + j)
            if len(good_poses) >= target:
                break

    if len(good_poses) < target:
        print(f"[Warn] Not enough valid poses: got={len(good_poses)} target={target}. "
              f"Consider increasing --num_candidates / --resample_batch / relaxing thresholds.")

    # 若超过 target：挑更分散的一组
    cam2world_list = _select_diverse_subset(good_poses, target, seed=args.seed + 4242)

    # 注册相机位姿
    for pose in cam2world_list:
        bproc.camera.add_camera_pose(pose)

    # 再兜底设置一次 clip（有时 add_camera_pose 会创建/切换相机）
    for cam in bpy.data.cameras:
        cam.clip_start = args.clip_start
        cam.clip_end = args.clip_end

    # render
    num_render = len(cam2world_list)
    print(f"[Render] scene={scene_id} views={num_render}/{target} res={args.W}x{args.H} device={args.device} samples={args.render_samples}")
    data = bproc.renderer.render()
    colors = data["colors"]
    depths = data["depth"]

    # save
    for i in range(num_render):
        c = colors[i]
        if c.dtype != np.uint8:
            c = np.clip(c, 0.0, 1.0)
            c = (c * 255.0).astype(np.uint8)
        if c.shape[-1] == 4:
            c = c[..., :3]
        imageio.imwrite(os.path.join(color_dir, f"{i}.jpg"), c)

        d = depths[i]
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d_mm = np.clip(d * 1000.0, 0.0, 65535.0).astype(np.uint16)
        imageio.imwrite(os.path.join(depth_dir, f"{i}.png"), d_mm)

        _save_pose_txt(os.path.join(pose_dir, f"{i}.txt"), cam2world_list[i])

    np.savetxt(os.path.join(scene_out, "K.txt"), K, fmt="%.8f")
    with open(os.path.join(scene_out, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"scene_id: {scene_id}\n")
        f.write(f"resolution: {args.W}x{args.H}\n")
        f.write(f"clip: {args.clip_start} ~ {args.clip_end}\n")
        f.write(f"bbox_min: {bb_min.tolist()}\n")
        f.write(f"bbox_max: {bb_max.tolist()}\n")
        f.write(f"target_views: {target}\n")
        f.write(f"final_views: {num_render}\n")
        f.write(f"num_candidates_init: {num_candidates}\n")
        f.write(f"resample_rounds: {round_id}\n")

    print(f"[OK] wrote frames_square to: {scene_out}")


if __name__ == "__main__":
    main()
