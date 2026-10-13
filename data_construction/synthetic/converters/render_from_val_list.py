import blenderproc as bproc
import os
import argparse
from pathlib import Path
import numpy as np
import imageio.v2 as imageio

# Blender 内置（BlenderProc 运行环境可用）
import bpy
import mathutils
import inspect


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
# 场景 bbox & 补光
# -------------------------
def _compute_scene_bbox_world():
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects in scene after loading.")

    for obj in mesh_objs:
        mw = obj.matrix_world
        # 跳过空mesh或异常mesh
        if not obj.data.vertices:
            print(f"[Warn] 跳过空mesh: {obj.name}")
            continue
        # 遍历bound_box并计算世界坐标
        for v in obj.bound_box:  # 8 corners local
            vw = mw @ mathutils.Vector(v)
            p = np.array([vw.x, vw.y, vw.z], dtype=np.float64)
            # 过滤NaN/无穷大的异常坐标
            if np.any(np.isnan(p)) or np.any(np.isinf(p)):
                print(f"[Warn] 跳过异常坐标: {obj.name} -> {p}")
                continue
            mins = np.minimum(mins, p)
            maxs = np.maximum(maxs, p)

    # 核心修复：检测并交换异常的max/min
    for i in range(3):  # x/y/z轴分别检查
        if maxs[i] < mins[i]:
            print(f"[Fix] 轴{i}(x=0/y=1/z=2)的max<min，交换值: {maxs[i]} < {mins[i]}")
            mins[i], maxs[i] = maxs[i], mins[i]  # 交换最大值和最小值

    # 再次检测：若仍异常则使用默认范围（避免崩溃）
    extent = (maxs - mins)
    if np.any(extent <= 0):
        print(f"[Warn] bbox范围异常: mins={mins}, maxs={maxs}，使用默认范围")
        # 设置默认的有效范围（可根据场景调整）
        mins = np.array([-5.0, -5.0, 0.0], dtype=np.float64)
        maxs = np.array([5.0, 5.0, 5.0], dtype=np.float64)

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
# -------------------------
def _sample_candidate_poses_inside_bbox(bb_min, bb_max, num_candidates, seed=0,
                                        cam_height=1.4,
                                        margin_ratio=0.08):
    """
    在 bbox 内随机采样相机位置（更“走动式”），并 look_at bbox 中心附近。
    """
    rng = np.random.default_rng(seed)
    bb_min = np.asarray(bb_min, dtype=np.float32)
    bb_max = np.asarray(bb_max, dtype=np.float32)
    center = (bb_min + bb_max) / 2.0
    extent = (bb_max - bb_min)

    margin_xy = float(max(extent[0], extent[1]) * margin_ratio)
    x_min = float(bb_min[0] + margin_xy)
    x_max = float(bb_max[0] - margin_xy)
    y_min = float(bb_min[1] + margin_xy)
    y_max = float(bb_max[1] - margin_xy)

    # 人眼高度：尽量在 bbox 内
    cam_z = float(bb_min[2] + cam_height)
    if bb_max[2] > bb_min[2] + 0.2:
        cam_z = float(np.clip(cam_z, bb_min[2] + 0.05, bb_max[2] - 0.05))
    else:
        cam_z = float(bb_min[2] + 0.3)

    poses = []
    for _ in range(num_candidates):
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        location = np.array([x, y, cam_z], dtype=np.float32)

        # 看向中心（带一点抖动，避免全一样）
        tgt = center.copy()
        tgt[2] = cam_z
        jitter = rng.normal(0.0, 0.12, size=(2,)).astype(np.float32) * max(0.2, float(min(extent[0], extent[1]) * 0.25))
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
    """
    使用 Blender scene.ray_cast 做射线检测。
    返回: hit(bool), hit_location(vec), hit_normal(vec), hit_obj
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    o = mathutils.Vector(origin)
    d = mathutils.Vector(direction).normalized()
    hit, loc, normal, face_index, obj, matrix = bpy.context.scene.ray_cast(depsgraph, o, d, distance=max_dist)
    return hit, loc, normal, obj


def _pose_is_valid(cam2world, target_point,
                   inside_eps=0.03,
                   min_forward_hit=0.25,
                   min_clearance=0.10):
    """
    粗筛：
    1) 相机不能“几乎贴着墙/在墙里”（前向 ray hit 太近）
    2) 相机位置附近 6 个方向 ray hit 太近 => 可能在 mesh 内/穿模
    3) 视线方向尽量看向场景而不是直接撞墙
    """
    R = cam2world[:3, :3]
    t = cam2world[:3, 3]

    # 相机 forward（Blender 相机看 -Z，这里用 cam2world 的 -Z 作为前向）
    forward = -R[:, 2]
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    # (A) 前向碰撞：如果第一下就撞得很近，基本是在贴墙/墙里
    hit, loc, normal, obj = _scene_raycast(t, forward, max_dist=10.0)
    if hit:
        dist = float((np.array([loc.x, loc.y, loc.z], dtype=np.float32) - t).dot(forward))
        if dist < min_forward_hit:
            return False

    # (B) 原点附近“是否在几何体里面”的近似检测：6 个方向都很近容易出问题
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
                # 基本就是“穿在墙里”
                return False
            if dist < min_clearance:
                near_hits += 1
    if near_hits >= 4:
        return False

    # (C) 视线尽量指向 target_point（避免看向边界外）
    to_target = (target_point - t).astype(np.float32)
    to_target = to_target / (np.linalg.norm(to_target) + 1e-8)
    if float(np.dot(forward, to_target)) < 0.2:
        return False

    return True


def _select_poses(candidates, num_views, target_point, seed=0,
                  relax_steps=3):
    """
    先严格筛选，不够就逐步放宽（避免最终拿不到 num_views）。
    """
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    selected = []
    # 逐步放宽阈值
    configs = [
        dict(inside_eps=0.03, min_forward_hit=0.35, min_clearance=0.12),
        dict(inside_eps=0.04, min_forward_hit=0.25, min_clearance=0.10),
        dict(inside_eps=0.05, min_forward_hit=0.18, min_clearance=0.08),
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
            if cam2world not in selected:
                selected.append(cam2world)

    return selected[:num_views]


def _save_pose_txt(path, mat4x4: np.ndarray):
    np.savetxt(path, mat4x4.astype(np.float32), fmt="%.8f")

# -------------------------
# 新增：检查已有有效视图（补全核心逻辑）
# -------------------------
def _get_existing_valid_views(color_dir, depth_dir, pose_dir):
    """
    统计已有有效视图数（color/depth/pose三文件都存在）
    返回：已有有效视图数、下一个起始索引
    """
    existing_indices = set()
    
    # 遍历color目录获取所有jpg索引
    if os.path.exists(color_dir):
        for f in os.listdir(color_dir):
            if f.endswith(".jpg"):
                try:
                    idx = int(f.split(".")[0])
                    existing_indices.add(idx)
                except ValueError:
                    continue
    
    # 过滤：仅保留depth和pose都存在的索引
    valid_indices = []
    for idx in sorted(existing_indices):
        depth_file = os.path.join(depth_dir, f"{idx}.png")
        pose_file = os.path.join(pose_dir, f"{idx}.txt")
        if os.path.exists(depth_file) and os.path.exists(pose_file):
            valid_indices.append(idx)
    
    # 无断号场景，下一个索引=有效视图数
    existing_num = len(valid_indices)
    next_start_idx = existing_num
    
    return existing_num, next_start_idx

# -------------------------
# 修改：检查是否已完成（兼容补全逻辑）
# -------------------------
def _already_done(scene_out: str, num_views: int):
    """
    新逻辑：仅当已有有效视图数>=目标数时，才视为完成
    """
    color_dir = os.path.join(scene_out, "color")
    depth_dir = os.path.join(scene_out, "depth")
    pose_dir = os.path.join(scene_out, "pose")
    existing_num, _ = _get_existing_valid_views(color_dir, depth_dir, pose_dir)
    return existing_num >= num_views


# -------------------------
# scene_id -> json path 索引
# -------------------------
def _index_front_jsons(front_dir: str):
    """
    递归扫描 front_dir 下所有 *.json，建立 stem->path 映射
    """
    mapping = {}
    for p in Path(front_dir).rglob("*.json"):
        mapping[p.stem] = str(p)
    return mapping


def _read_id_list(list_path: str):
    """
    读取scene_id列表，去重并保持顺序
    """
    ids = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 兼容“scene_id ...”格式，只取第一列
            sid = s.split()[0]
            ids.append(sid)
    # 去重保持顺序
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def render_one_scene(front_json: str, args, mapping):
    """
    修改后的渲染逻辑：补全缺失视图，不覆写已有
    """
    scene_id = Path(front_json).stem
    scene_out = os.path.join(args.out_root, scene_id)

    # 检查是否已完成（已有>=目标数则跳过）
    if args.skip_if_done and _already_done(scene_out, args.num_views):
        print(f"[Skip] {scene_id} already done (existing >= {args.num_views}).")
        return

    # 初始化输出目录
    color_dir = os.path.join(scene_out, "color")
    depth_dir = os.path.join(scene_out, "depth")
    pose_dir = os.path.join(scene_out, "pose")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    # 核心：检查已有有效视图，计算需补充数量
    existing_num, next_start_idx = _get_existing_valid_views(color_dir, depth_dir, pose_dir)
    need_generate = args.num_views - existing_num
    print(f"\n===== 场景 {scene_id} 渲染状态 =====")
    print(f"目标视图总数      : {args.num_views} 个")
    print(f"已有有效视图数    : {existing_num} 个")
    print(f"本次需补充渲染数  : {need_generate} 个")

    # 无需补充则更新元数据并退出
    if need_generate <= 0:
        unrendered_num = 0
        print(f"未成功渲染视图数  : {unrendered_num} 个")
        print(f"=====================================")
        # 更新元数据文件
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

    # ✅ 每个 scene 只 clean_up，不要 init
    bproc.clean_up()

    bproc.camera.set_resolution(args.W, args.H)

    # ScanRefer projector 对齐：41x32 的内参 *8
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

    # load scene
    bproc.loader.load_front3d(
        json_path=front_json,
        future_model_path=args.future_folder,
        front_3D_texture_path=args.front_3D_texture_path,
        label_mapping=mapping,
    )

    center, radius, bb_min, bb_max = _compute_scene_bbox_world()
    _ensure_lights(center, radius)

    # clip
    if bpy.context.scene.camera is not None:
        bpy.context.scene.camera.data.clip_start = args.clip_start
        bpy.context.scene.camera.data.clip_end = args.clip_end
    else:
        for cam in [o for o in bpy.context.scene.objects if o.type == "CAMERA"]:
            cam.data.clip_start = args.clip_start
            cam.data.clip_end = args.clip_end

    # 仅采样需要补充的候选数（按比例调整）
    num_candidates = max(args.num_candidates, need_generate * 2)  # 保证候选数足够
    candidates = _sample_candidate_poses_inside_bbox(
        bb_min=bb_min,
        bb_max=bb_max,
        num_candidates=num_candidates,
        seed=args.seed + existing_num,  # 不同批次用不同seed
        cam_height=args.cam_height,
        margin_ratio=args.margin_ratio,
    )

    # 仅选择需要补充的视图数
    selected = _select_poses(
        candidates=candidates,
        num_views=need_generate,
        target_point=center,
        seed=args.seed + existing_num,
        relax_steps=3,
    )

    # 注册相机位姿
    for cam2world in selected:
        bproc.camera.add_camera_pose(cam2world)

    # 渲染缺失的视图
    num_render = len(selected)
    print(f"[Render] scene={scene_id} candidates={num_candidates} -> selected={num_render} (需补充{need_generate})")
    data = bproc.renderer.render()
    colors = data["colors"]
    depths = data["depth"]

    # 接续索引保存文件（不覆盖已有）
    for i in range(num_render):
        current_idx = next_start_idx + i
        # 保存color
        c = colors[i]
        if c.dtype != np.uint8:
            c = np.clip(c, 0.0, 1.0)
            c = (c * 255.0).astype(np.uint8)
        if c.shape[-1] == 4:
            c = c[..., :3]
        imageio.imwrite(os.path.join(color_dir, f"{current_idx}.jpg"), c)

        # 保存depth
        d = depths[i]
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d_mm = np.clip(d * 1000.0, 0.0, 65535.0).astype(np.uint16)
        imageio.imwrite(os.path.join(depth_dir, f"{current_idx}.png"), d_mm)

        # 保存pose
        _save_pose_txt(os.path.join(pose_dir, f"{current_idx}.txt"), selected[i])

    # 计算最终统计数据
    final_total = existing_num + num_render
    unrendered_num = args.num_views - final_total

    # 保存/更新内参和元数据
    np.savetxt(os.path.join(scene_out, "K.txt"), K, fmt="%.8f")
    meta_file = os.path.join(scene_out, "meta.txt")
    
    # 读取原有元数据（若存在）
    meta_lines = []
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_lines = f.readlines()
    
    # 写入元数据（更新关键字段）
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
        # 无原有元数据则新建
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

    # 输出最终状态
    print(f"===== 场景 {scene_id} 渲染完成 =====")
    print(f"目标视图总数      : {args.num_views} 个")
    print(f"已有有效视图数    : {existing_num} 个")
    print(f"本次成功补充数    : {num_render} 个")
    print(f"最终总视图数      : {final_total} 个")
    print(f"未成功渲染视图数  : {unrendered_num} 个")
    print(f"=====================================")
    if unrendered_num > 0:
        print(f"[Warn] 存在 {unrendered_num} 个视图未成功渲染，建议增大 --num_candidates")
    print(f"[OK] wrote frames_square: {scene_out}")


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

    args = parser.parse_args()

    # ✅ 只 init 一次（全局）
    bproc.init()
    _safe_set_render_devices(args.device)
    _safe_set_samples(args.render_samples)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    
    # 加载label映射
    if args.mapping_csv is None:
        mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    else:
        mapping_file = args.mapping_csv
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    # 索引json文件和scene_id列表
    json_index = _index_front_jsons(args.front_dir)
    scene_ids = _read_id_list(args.id_list)

    print(f"[Info] loaded {len(scene_ids)} scene_ids from {args.id_list}")
    print(f"[Info] indexed {len(json_index)} jsons from {args.front_dir}")

    # 批量处理每个scene
    missing = 0
    failed = 0
    total_unrendered = 0
    for sid in scene_ids:
        front_json = json_index.get(sid, None)
        if front_json is None:
            cand = os.path.join(args.front_dir, f"{sid}.json")
            if os.path.exists(cand):
                front_json = cand
        if front_json is None:
            print(f"[Missing] json not found for scene_id={sid}")
            missing += 1
            continue

        try:
            # 渲染单个场景（补全逻辑）
            render_one_scene(front_json, args, mapping)
            
            # 统计未成功渲染数
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
            print(f"[Error] scene_id={sid} failed: {e}")
            failed += 1

    # 输出批量处理总结
    print(f"\n==================== 批量处理总结 ====================")
    print(f"总场景数          : {len(scene_ids)}")
    print(f"缺失JSON文件数    : {missing}")
    print(f"渲染失败场景数    : {failed}")
    print(f"总未成功渲染视图数: {total_unrendered}")
    print(f"======================================================")


if __name__ == "__main__":
    main()