# generate_dataset.py
# 一条龙：Stage0(编辑+scene_graph) + Stage1_2(生成/退化) + Stage3(feedback, VLM 多视角)
#
# 用法示例：
# blenderproc run generate_dataset.py \
#   --front_dir /path/to/3D-FRONT-jsons \
#   --future_folder /path/to/3D-FUTURE-model \
#   --front_3D_texture_path /path/to/3D-FRONT-textures \
#   --output_root /path/to/OUTPUT \
#   --model_info_json /path/to/model_info.json \
#   --text_model qwen-llm \
#   --color_model qwen-vl \
#   --max_rounds 6

import blenderproc
import argparse
import os
import json
import random
import math
import base64
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

from tqdm import tqdm

import bpy
import numpy as np
import mathutils
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from bpy_extras.object_utils import world_to_camera_view

from openai import OpenAI
import sys
from contextlib import contextmanager

# ================== 全局常量 ==================

THREEDFRONT_CAT2NYU40 = {
    "cabinet": 3,   # cabinet
    "bed": 4,       # bed
    "chair": 5,     # chair
    "sofa": 6,      # sofa
    "table": 7,     # table
    "Lighting": 8, # desklamp / light
    "Pier/Stool": 9, # stool
    "others": 40    # otherfurniture / other
}

BACKGROUND_CATS = {
    "wall",
    "floor",
    "ceiling",
    "window",
    "unknown"
}
def aabb_intersects_ray(aabb_min, aabb_max, ray_origin, ray_dir):
    """
    Slab method for ray-AABB intersection.
    """
    t1 = (aabb_min - ray_origin) / ray_dir
    t2 = (aabb_max - ray_origin) / ray_dir
    tmin = np.maximum(np.minimum(t1, t2), 0).max()
    tmax = np.minimum(np.maximum(t1, t2), np.inf).min()
    return tmax >= tmin  # Has intersection


def compute_visibility_score(cam_loc, target_top_points, all_objects, target_id):
    """
    cam_loc: camera world position np.array(3,)
    target_top_points: list of world points on object top surface
    all_objects: all MeshObjects from BlenderProc
    target_id: the id of the target object (to skip itself)

    Return: float in [0,1], ratio of visible rays
    """
    visible = 0
    total = len(target_top_points)

    for obj in all_objects:
        if not isinstance(obj, blenderproc.types.MeshObject):
            continue
        if obj.get_cp("uid") == target_id:
            continue

        try:
            obj_min, obj_max = get_world_bbox(obj)
        except:
            continue

        aabb_min = obj_min
        aabb_max = obj_max

        # For each ray from camera → surface point
        blocked_any = False
        for p in target_top_points:
            ray_dir = p - cam_loc
            norm = np.linalg.norm(ray_dir)
            if norm < 1e-6:
                continue
            ray_dir = ray_dir / norm

            if aabb_intersects_ray(aabb_min, aabb_max, cam_loc, ray_dir):
                blocked_any = True
                break

        if blocked_any:
            # This object blocks at least one ray
            return 0.0

    # If no blocker detected:
    return 1.0

# 多视角渲染输出根目录（Stage3 用）
# 对每个 scene，会放在：RENDER_ROOT / scene_id / {scene_id}_obj{object_id}_viewname.jpg
RENDER_ROOT = os.environ.get('AMBIREFER3D_SYNTHETIC_RENDER_ROOT', 'data/synthetic/render_multiview')
# ================== 可视化 / 相机采样工具 ==================
def get_obj_bbox_corners_and_center(obj, grid_res: int = 5):
    """
    返回物体 AABB 顶面的稠密采样点（世界坐标）和顶面中心点

    参数：
        obj: blenderproc.types.MeshObject
        grid_res: 顶面网格分辨率，5 表示 5x5 共 25 个点

    返回：
        top_points_world: List[mathutils.Vector]，顶面稠密点（世界坐标）
        top_center_world: mathutils.Vector，顶面中心（世界坐标）
    """
    # 确保变换矩阵是最新的
    bpy.context.view_layer.update()

    # 物体的世界变换
    mat = obj.blender_obj.matrix_world

    # AABB 的 8 个局部角点
    corners_local = obj.blender_obj.bound_box  # tuple of 8 (x, y, z)

    # 顶面四个角点在 bound_box 中的索引：1,3,5,7（z 为最大）
    top_idx = [1, 3, 5, 7]
    top_local = [mathutils.Vector(corners_local[i]) for i in top_idx]

    # 转为世界坐标
    top_world = [mat @ p for p in top_local]
    p1, p2, p3, p4 = top_world  # 对应 1,3,5,7

    top_points_world = []

    # 在顶面做 grid_res x grid_res 的双线性插值
    for i in range(grid_res):
        for j in range(grid_res):
            u = i / (grid_res - 1)  # [0,1]
            v = j / (grid_res - 1)  # [0,1]

            # 双线性插值：p = (1-u)*(1-v)*p1 + (1-u)*v*p2 + u*(1-v)*p3 + u*v*p4
            p = (
                (1.0 - u) * (1.0 - v) * p1 +
                (1.0 - u) * v         * p2 +
                u         * (1.0 - v) * p3 +
                u         * v         * p4
            )
            top_points_world.append(p)

    # 顶面中心：四个角点的平均 / 或者网格点的平均都可以，这里用角点
    top_center_world = sum(top_world, mathutils.Vector((0.0, 0.0, 0.0))) / 4.0

    return top_points_world, top_center_world


def sample_candidate_cameras_around_object(
    obj,
    n_az: int = 12,
    elevations=(20.0, 35.0),
    dist_ratio: float = 1.8
):
    """
    在物体周围球面上采样一圈候选相机位置（只返回位置，不做遮挡判断）
    """
    corners, center = get_obj_bbox_corners_and_center(obj)
    # 用包围盒的对角线估一个尺寸
    diag = max((c - center).length for c in corners) * 2.0
    base_dist = max(diag * dist_ratio, 1.5)

    cam_positions = []
    for el in elevations:
        el_rad = math.radians(el)
        for k in range(n_az):
            az = 360.0 * k / n_az
            az_rad = math.radians(az)

            cx = center.x + base_dist * math.cos(el_rad) * math.cos(az_rad)
            cy = center.y + base_dist * math.cos(el_rad) * math.sin(az_rad)
            cz = center.z + base_dist * math.sin(el_rad)

            cam_positions.append(mathutils.Vector((cx, cy, cz)))
    return cam_positions


def is_view_unoccluded(target_obj, cam_location, max_dist: float = None, n_sample_rays: int = 5) -> bool:
    """
    判断从 cam_location 看向 target_obj 是否“基本无遮挡”。

    逻辑：从相机位置向 目标包围盒中心 + 若干角点 发射射线，
    如果第一个 hit 的 object 不是 target_obj，则视角认为被遮挡。
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    corners, center = get_obj_bbox_corners_and_center(target_obj)

    if max_dist is None:
        max_dist = (center - cam_location).length * 1.1

    # 要检测的若干目标点：中心 + 随机角点
    sample_pts = [center]
    if len(corners) <= n_sample_rays - 1:
        sample_pts += corners
    else:
        idxs = np.random.choice(len(corners), size=(n_sample_rays - 1), replace=False)
        sample_pts += [corners[i] for i in idxs]

    tgt_blender_obj = target_obj.blender_obj

    for p in sample_pts:
        direction = (p - cam_location).normalized()
        hit, loc, normal, face_index, hit_obj, _ = scene.ray_cast(
            depsgraph, cam_location, direction, distance=max_dist
        )

        if not hit:
            # 这条射线没撞到任何东西 -> 视线通畅
            continue

        # 如果第一个命中的不是目标物体，说明这条视线被别的物体挡住了
        if hit_obj != tgt_blender_obj:
            return False

    # 所有采样射线都没先撞到别的物体 -> 视角基本无遮挡
    return True


def pick_good_views_for_object(
    target_obj,
    max_views: int = 3,
    n_az: int = 12,
    elevations=(20.0, 35.0)
):
    """
    为目标物体挑选若干无遮挡的视角。
    返回: List[(cam_loc, cam_rot_euler)]
    """
    cand_positions = sample_candidate_cameras_around_object(
        target_obj,
        n_az=n_az,
        elevations=elevations
    )

    good_views = []
    corners, center = get_obj_bbox_corners_and_center(target_obj)

    for cam_loc in cand_positions:
        if is_view_unoccluded(target_obj, cam_loc):
            direction = (center - cam_loc)
            rot = direction.to_track_quat('-Z', 'Y').to_euler()
            good_views.append((cam_loc, rot))
        if len(good_views) >= max_views:
            break
    from ipdb import set_trace
    # set_trace()
    # 兜底：如果所有视角都被认为遮挡，就退而求其次，直接用前几个候选
    if not good_views:
        for cam_loc in cand_positions[:max_views]:
            direction = (center - cam_loc)
            rot = direction.to_track_quat('-Z', 'Y').to_euler()
            good_views.append((cam_loc, rot))

    return good_views

def get_bbox_corners_world(obj):
    """返回目标物体 AABB 的 8 个世界坐标角点 + 中心点"""
    mat = obj.blender_obj.matrix_world
    corners_local = obj.blender_obj.bound_box  # 8 corners in local coords
    corners_world = [mat @ mathutils.Vector(c) for c in corners_local]
    center = sum(corners_world, mathutils.Vector((0,0,0))) / 8.0
    return corners_world, center

def get_category_name_from_blender_obj(blender_obj, mapping):
    """根据 Blender Object 找回对应的 MeshObject 再用你现有的 get_category_name"""
    mesh_obj = blenderproc.types.MeshObject(blender_obj)
    return get_category_name(mesh_obj, mapping)
@contextmanager
def suppress_blender_output():
    """Temporarily suppress Blender console output (stdout + stderr)."""
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

# =============== LLM / VLM 配置 ===============

LIGHT_MODE = True
RETRY = 3
SLEEP = 1.2

# 文本 vLLM（Stage1/2）
TEXT_VLLM_BASE_URL = os.environ.get('AMBIREFER3D_TEXT_API_BASE', 'http://localhost:8200/v1')
text_llm_client = OpenAI(
    api_key=os.environ.get("AMBIREFER3D_API_KEY", "EMPTY"),
    base_url=TEXT_VLLM_BASE_URL,
)

# 视觉 VLM（颜色 & Stage3 多视角）
COLOR_VLM_BASE_URL = os.environ.get('AMBIREFER3D_VISION_API_BASE', 'http://localhost:8100/v1')
color_vlm_client = OpenAI(
    api_key=os.environ.get("AMBIREFER3D_API_KEY", "EMPTY"),
    base_url=COLOR_VLM_BASE_URL,
)

# 允许在 main() 里通过参数改 color_model
COLOR_MODEL_NAME = "qwen-vl"


# =============== 公共小工具 ===============

def call_vllm(model: str, system_prompt: str, user_prompt: str) -> str:
    """文本 LLM 调用（不带图像），用于 Stage1/2/3."""
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt + " /no_think"})

    for i in range(RETRY):
        try:
            resp = text_llm_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
            )
            text = resp.choices[0].message.content or ""
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()
            return text.strip()
        except Exception as e:
            if i == RETRY - 1:
                raise
            print(f"[LLM Error] {e}, retrying ({i+1}/{RETRY})...")
            time.sleep(SLEEP * (i + 1))
    return ""


def safe_json_loads(text: str) -> Any:
    """尽量从返回文本中抠出 JSON."""
    try:
        return json.loads(text)
    except Exception:
        text = text.strip()
        if "{" in text and "}" in text:
            try:
                text = text[text.find("{"): text.rfind("}") + 1]
                return json.loads(text)
            except Exception:
                pass
        raise


def call_vlm_color(image_path: str, model: str = None) -> str:
    """
    给 VLM 一张图片，让它回答：这个物体是什么颜色？
    返回一个粗粒度颜色名（lowercase 英文单词）。
    """
    if model is None:
        model = COLOR_MODEL_NAME

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    img_url = f"data:image/png;base64,{img_b64}"

    system_prompt = (
        "You are a color classifier. "
        "You see an object rendered on a plain background. "
        "Answer with ONE single coarse color word from this list:\n"
        "[white, black, red, green, blue, yellow, brown, gray, orange, purple, pink].\n"
        "If the color is between two, choose the dominant one.\n"
        "Return ONLY the color word, no extra text."
    )
    user_text = "What is the main color of the object in the image?"

    resp = color_vlm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            },
        ],
        max_tokens=1024,
        temperature=0.0,
    )

    text = resp.choices[0].message.content or ""
    text = text.strip().strip(".").strip().lower()
    return text


# =============== Stage0：编辑 + Scene Graph ===============

def get_category_name(obj, mapping, all_objects=None):
    """优先使用 cp('category') 或 category_id -> label。"""
    if obj.has_cp("category"):
        return str(obj.get_cp("category"))

    if obj.has_cp("category_id"):
        try:
            cid = int(obj.get_cp("category_id"))
            label = mapping.get_label_from_id(cid)
            if label is not None:
                return str(label)
        except Exception:
            pass

    name = obj.get_name().lower()
    if "wall" in name:
        return "wall"
    if "floor" in name:
        return "floor"
    if "ceiling" in name:
        return "ceiling"
    if "chair" in name:
        return "chair"
    if "table" in name or "desk" in name:
        return "table"
    if "sofa" in name or "couch" in name:
        return "sofa"
    if "bed" in name:
        return "bed"
    if "cabinet" in name:
        return "cabinet"
    return "unknown"


def sample_random_editable_object(
    objects, mapping,
    exclude=("wall", "floor", "ceiling", "cabinet", "unknown")
):
    cands = []
    for obj in objects:
        cat = get_category_name(obj, mapping, all_objects=objects)
        if cat in exclude:
            continue
        cands.append((obj, cat))
    if not cands:
        return None, None
    return random.choice(cands)


def compute_bbox_size_world(obj):
    bbox = np.array(obj.get_bound_box(local_coords=False))
    size = bbox.max(axis=0) - bbox.min(axis=0)
    return size


def get_world_bbox(obj):
    """获取物体在世界坐标系下的 AABB 包围盒 (min_point, max_point)"""
    bpy.context.view_layer.update()
    matrix = obj.blender_obj.matrix_world
    bbox_corners = [matrix @ mathutils.Vector(corner) for corner in obj.blender_obj.bound_box]
    bbox_array = np.array(bbox_corners)
    min_point = np.min(bbox_array, axis=0)
    max_point = np.max(bbox_array, axis=0)
    return min_point, max_point


def is_protected_structural(cat_name):
    """绝对不应该因尺寸过大或碰撞被删除的类别。"""
    protected = {"floor", "ceiling", "wall", "composite", "rug", "mat", "stairs", "window"}
    return cat_name in protected


def check_collision(target_obj, all_objects, mapping, ignore_names=None, buffer=0.01):
    """
    检测 target_obj 是否与 all_objects 中的其他物体发生碰撞（基于 AABB）。
    :param buffer: 碰撞缓冲距离，默认 1cm
    :return: True 表示发生碰撞
    """
    MAX_DIM_GENERIC = 10.0
    objects_after_size_filter = []

    for obj in all_objects:
        if not isinstance(obj, blenderproc.types.MeshObject):
            continue

        cat = get_category_name(obj, mapping)
        if is_protected_structural(cat):
            objects_after_size_filter.append(obj)
            continue

        size = compute_bbox_size_world(obj)
        max_dim = np.max(size)

        if max_dim > MAX_DIM_GENERIC or size[0] * size[1] * size[2] > 100.0:
            print(f"[Cleaning-Size] Removed Oversized Object: {obj.get_name()} ({cat}, MaxDim: {max_dim:.1f}m)")
            obj.delete()
            continue

        objects_after_size_filter.append(obj)

    if ignore_names is None:
        ignore_names = set()

    t_min, t_max = get_world_bbox(target_obj)
    t_min = t_min + buffer
    t_max = t_max - buffer
    target_name = target_obj.get_name()

    for obj in all_objects:
        if obj.get_name() == target_name or obj.get_name() in ignore_names:
            continue

        cat_name = get_category_name(obj, mapping, all_objects=all_objects)
        if cat_name in ["wall", "floor", "ceiling", "unknown"]:
            continue

        o_min, o_max = get_world_bbox(obj)
        overlap_x = (t_min[0] < o_max[0]) and (t_max[0] > o_min[0])
        overlap_y = (t_min[1] < o_max[1]) and (t_max[1] > o_min[1])
        overlap_z = (t_min[2] < o_max[2]) and (t_max[2] > o_min[2])

        if overlap_x and overlap_y and overlap_z:
            return True

    return False


def load_model_info(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    return data


def pick_model_id_for_category(cat_name, model_info, allow_any_if_empty=True):
    cat_l = (cat_name or "").lower()
    cands = []
    for rec in model_info:
        mid = rec.get("model_id")
        if not mid:
            continue
        super_cat = (rec.get("super-category") or "").lower()
        sub_cat = (rec.get("category") or "").lower()
        if cat_l and (cat_l in super_cat or cat_l in sub_cat):
            cands.append(mid)
    if not cands and allow_any_if_empty:
        cands = [rec["model_id"] for rec in model_info if rec.get("model_id")]
    if not cands:
        return None
    return random.choice(cands)


def random_move_one_object(objects, mapping, max_translate, edit_log, max_trials=10):
    """随机移动物体，带有碰撞检测和重力约束。"""
    obj, cat = sample_random_editable_object(objects, mapping)
    if obj is None:
        print("[Edit-Move] 没有可编辑对象，跳过移动。")
        return

    old_loc = np.array(obj.get_location())
    max_tx, max_ty, _ = max_translate

    success = False
    valid_new_loc = old_loc
    delta_final = [0, 0, 0]

    print(f"[Edit-Move] Trying to move {obj.get_name()} ({cat})...")

    for i in range(max_trials):
        dx = np.random.uniform(-max_tx, max_tx)
        dy = np.random.uniform(-max_ty, max_ty)
        dz = 0.0
        temp_loc = old_loc + np.array([dx, dy, dz])
        obj.set_location(temp_loc)

        if not check_collision(obj, objects, mapping):
            success = True
            valid_new_loc = temp_loc
            delta_final = [dx, dy, dz]
            break

    if success:
        rec = {
            "type": "move",
            "object_name": obj.get_name(),
            "category": cat,
            "old_location": old_loc.tolist(),
            "new_location": valid_new_loc.tolist(),
            "delta": delta_final,
            "ambiguity_type": "spatial_left_right"
        }
        edit_log.append(rec)
        print(f"[移动成功] Success: Moved {obj.get_name()} by {delta_final}")
    else:
        obj.set_location(old_loc)
        print(f"[移动失败] Failed: Could not find valid non-colliding position for {obj.get_name()}")


def random_replace_with_model_info(target_obj, target_cat, model_info, future_folder,
                                   edit_log, all_objects, mapping, size_match=True):
    """随机替换物体，带有底部对齐、朝向继承，并在碰撞时尝试缩小体积解决冲突。"""
    mid = pick_model_id_for_category(target_cat, model_info)
    if not mid:
        print("[替换] 未找到匹配 model_id，跳过。")
        return
    model_dir = os.path.join(future_folder, mid)
    obj_path = os.path.join(model_dir, "normalized_model.obj")
    if not os.path.exists(obj_path):
        print("[替换] 模型文件不存在:", obj_path)
        return

    target_loc = np.array(target_obj.get_location())
    target_rot = np.array(target_obj.get_rotation_euler())
    target_name = target_obj.get_name()

    t_min, t_max = get_world_bbox(target_obj)
    old_bottom_z = t_min[2]
    old_size = t_max - t_min
    old_diag = np.linalg.norm(old_size)

    loaded_objects = blenderproc.loader.load_obj(obj_path)
    new_obj = None
    other_parts = []
    for o in loaded_objects:
        if isinstance(o, blenderproc.types.MeshObject):
            if new_obj is None:
                new_obj = o
            else:
                other_parts.append(o)
    if new_obj is None:
        return
    for p in other_parts:
        p.delete()

    if size_match:
        n_min, n_max = get_world_bbox(new_obj)
        new_size = n_max - n_min
        new_diag = np.linalg.norm(new_size)
        if new_diag > 0:
            scale_factor = old_diag / new_diag
            scale_factor = np.clip(scale_factor, 0.5, 1.5)
            new_obj.set_scale([scale_factor] * 3)

    new_obj.set_location(target_loc)
    new_obj.set_rotation_euler(target_rot)

    n_min_curr, _ = get_world_bbox(new_obj)
    current_bottom_z = n_min_curr[2]
    z_offset = old_bottom_z - current_bottom_z
    corrected_loc = target_loc + np.array([0, 0, z_offset])
    new_obj.set_location(corrected_loc)

    is_colliding = check_collision(new_obj, all_objects, mapping, ignore_names={target_name})
    shrink_trials = 3
    success = False

    if not is_colliding:
        success = True
    else:
        print(f"[替换] Collision detected for new model {mid}. Attempting to shrink size...")
        current_scale = np.array(new_obj.get_scale())
        initial_bottom_z = old_bottom_z

        for i in range(shrink_trials):
            shrink_factor_step = 0.95
            current_scale *= shrink_factor_step
            if np.min(current_scale) < 0.1:
                break

            new_obj.set_scale(current_scale)
            n_min_scaled, _ = get_world_bbox(new_obj)
            current_bottom_z_scaled = n_min_scaled[2]
            z_offset_scaled = initial_bottom_z - current_bottom_z_scaled
            loc_after_scale = new_obj.get_location()
            corrected_loc_after_scale = np.array([
                loc_after_scale[0],
                loc_after_scale[1],
                loc_after_scale[2] + z_offset_scaled
            ])
            new_obj.set_location(corrected_loc_after_scale)
            is_colliding = check_collision(new_obj, all_objects, mapping, ignore_names={target_name})
            if not is_colliding:
                print(f"[替换] Collision resolved after {i+1} shrinkage trial.")
                success = True
                break

    if success:
        rec = {
            "type": "replace",
            "old_object_name": target_name,
            "new_object_name": new_obj.get_name(),
            "category": target_cat,
            "model_id": mid,
            "is_colliding_forced": False,
            "ambiguity_type": "attribute_or_shape"
        }
        target_obj.delete()
        edit_log.append(rec)
        print(f"[替换成功] Success: {target_name} -> {mid}")
    else:
        print(f"[替换失败] FAILED: Could not resolve collision. Cancelling.")
        new_obj.delete()

def estimate_color_name_from_materials(
    obj,
    mapping=None,
    debug_render=False,
    debug_output_dir=None,
    scene_id=None,
) -> str:
    """
    使用 BlenderProc 多视角渲染 + VLM 来估计物体颜色。

    - 先在物体周围采样一圈视角；
    - 用 ray_cast 过滤掉被其他物体遮挡的视角；
    - 对每个无遮挡视角渲染一张“只含当前物体”的图片；
    - 将所有视角的图片分别送入 VLM 得到颜色词；
    - 用多数票选出最终颜色。
    """

    # ---- 过滤非前景 / 不需要估色的类别 ----
    cat = None
    if mapping is not None:
        try:
            cat = get_category_name(obj, mapping, all_objects=None)
        except Exception:
            cat = None
    exclude_cats = ("wall", "floor", "ceiling", "cabinet", "unknown")
    if cat in exclude_cats:
        return None

    if not debug_render or debug_output_dir is None or scene_id is None:
        return None

    os.makedirs(debug_output_dir, exist_ok=True)

    scene = bpy.context.scene
    all_bpy_objs = list(bpy.data.objects)

    # 记录原始 hide_render 状态
    old_hide_render = {o: o.hide_render for o in all_bpy_objs}
    old_world_color = None
    if scene.world is not None:
        old_world_color = tuple(scene.world.color)

    # 选视角
    views = pick_good_views_for_object(obj, max_views=3)
    if not views:
        return None

    tmp_paths = []
    color_votes = []

    try:
        # 只渲染当前物体：其他 mesh 全部 hide_render=True
        for o in all_bpy_objs:
            if o.type == 'MESH':
                o.hide_render = (o.name != obj.get_name())
        
        # 背景尽量简单一点
        if scene.world is not None:
            scene.world.color = (0.0, 0.0, 0.0)

        for view_idx, (cam_loc, cam_rot) in enumerate(views):
            # 创建临时相机
            bpy.ops.object.camera_add()
            cam = bpy.context.active_object
            cam.name = f"ColorDebugCam_{scene_id}_{obj.get_name()}_{view_idx}"
            cam.location = cam_loc
            cam.rotation_euler = cam_rot
            scene.camera = cam

            # 加一个点光源，让物体亮一点
            bpy.ops.object.light_add(type='POINT')
            light = bpy.context.active_object
            light.location = cam_loc  # 灯光放在相机位置
            light.data.energy = 1500.0

            safe_name = obj.get_name().replace(" ", "_")
            raw_path = os.path.join(
                debug_output_dir,
                f"{scene_id}__{safe_name}_view{view_idx}.png"
            )
            scene.render.image_settings.file_format = "PNG"
            scene.render.filepath = raw_path

            with suppress_blender_output():
                bpy.ops.render.render(write_still=True)

            tmp_paths.append(raw_path)

            # 调 VLM 估颜色
            try:
                color_name = call_vlm_color(raw_path)
                if color_name:
                    color_votes.append(color_name)
            except Exception as e:
                # 单个视角失败就忽略，继续下一个
                print(f"[Color-VLM] Error on view {view_idx} for {safe_name}: {e}")

            # 清理刚刚创建的灯光和相机
            try:
                if light:
                    bpy.data.objects.remove(light, do_unlink=True)
            except Exception:
                pass
            try:
                if cam:
                    bpy.data.objects.remove(cam, do_unlink=True)
            except Exception:
                pass

    finally:
        # 恢复 hide_render & 世界背景
        for o, v in old_hide_render.items():
            o.hide_render = v
        if old_world_color is not None and scene.world is not None:
            scene.world.color = old_world_color
    exit()
    # 删除临时图片
    # for p in tmp_paths:
    #     try:
    #         if os.path.exists(p):
    #             os.remove(p)
    #     except Exception:
    #         pass

    if not color_votes:
        return None

    # 多数票投票
    from collections import Counter
    cnt = Counter(color_votes)
    final_color, _ = cnt.most_common(1)[0]
    return final_color


# =============== 多视角渲染工具（BlenderProc 版，高质量 VLM 输入） ===============

def resize_image_smart(image_path, max_edge=672):
    """渲染输出后的图片缩放（用于 VLM 输入时可选）。"""
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_edge:
            scale = max_edge / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"[Stage3-VLM][IMAGE_ERROR] {image_path}: {e}")
        return None


def encode_image_to_base64(pil_image):
    buf = BytesIO()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def render_multiview_for_object(
    scene_id: str,
    object_id: str,
    bbox_corners: List[List[float]],
    all_objects,
    output_dir: str,
    image_size: int = 672,
    top_k: int = 3,
):
    os.makedirs(output_dir, exist_ok=True)
    scene = bpy.context.scene

    # Prepare bbox & center
    bbox = np.array(bbox_corners)
    center = bbox.mean(axis=0)
    size = bbox.max(axis=0) - bbox.min(axis=0)
    max_dim = float(size.max())
    max_dim = max(max_dim, 1.0)

    # ✨ Generate top-surface dense points for visibility
    target_top_points = get_obj_bbox_corners_and_center(bbox_corners, grid_res=5)

    # ✨ Candidate angles (like your color-estimation sampling strategy)
    azims = np.linspace(0, 360, 24, endpoint=False)
    elevs = [10, 20, 30, 40]
    dist = max(max_dim * 1.8, 1.5)

    def cam_from_angles(az, el):
        az_r = math.radians(az)
        el_r = math.radians(el)
        cx = center[0] + dist * math.cos(el_r) * math.cos(az_r)
        cy = center[1] + dist * math.cos(el_r) * math.sin(az_r)
        cz = center[2] + dist * math.sin(el_r)
        return np.array([cx,cy,cz])

    # ✨ Compute visibility score for all candidate views
    candidates = []
    for az in azims:
        for el in elevs:
            cam_loc = cam_from_angles(az, el)
            score = compute_visibility_score(
                cam_loc,
                target_top_points,
                all_objects,
                target_id=object_id
            )
            candidates.append((score, az, el, cam_loc))

    # Sort by visibility (descending)
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Pick top-K best viewpoints
    best_views = candidates[:top_k]

    print(f"[INFO] Best view scores: {[v[0] for v in best_views]}")

    # ========= Now perform real rendering for the selected views ========= #
    for score, az, el, cam_loc in best_views:

        # camera setup
        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
        cam.location = cam_loc
        cam.rotation_euler = (0,0,0)

        # Aim camera at the object center
        target_vec = mathutils.Vector(center)
        cam_dir = target_vec - cam.location
        cam.rotation_euler = cam_dir.to_track_quat('-Z','Y').to_euler()

        # Add point light at camera
        bpy.ops.object.light_add(type='POINT')
        light = bpy.context.active_object
        light.location = cam.location + mathutils.Vector((0.5,0.5,0.5))
        light.data.energy = 1500

        # Rendering setup
        scene.render.resolution_x = image_size
        scene.render.resolution_y = image_size
        base = f"{scene_id}_obj{object_id}_az{int(az)}_el{int(el)}"
        tmp = os.path.join(output_dir, base + ".png")
        out = os.path.join(output_dir, base + ".jpg")
        scene.render.filepath = tmp

        with suppress_blender_output():
            bpy.ops.render.render(write_still=True)

        # Load image and draw bbox
        img = Image.open(tmp).convert("RGB")
        width, height = img.size

        corners = [mathutils.Vector(c) for c in bbox.tolist()]
        screen_pts = []
        for c in corners:
            ndc = world_to_camera_view(scene, cam, c)
            x = ndc.x * width
            y = (1 - ndc.y) * height
            screen_pts.append((x,y))

        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        draw = ImageDraw.Draw(img)
        for (a,b) in edges:
            draw.line([screen_pts[a], screen_pts[b]], fill=(0,255,0), width=3)

        img.save(out, quality=95)
        os.remove(tmp)

        bpy.data.objects.remove(light, do_unlink=True)
        bpy.data.objects.remove(cam, do_unlink=True)


# =============== 点云导出（Scene 点云 + ScanRefer 样式） ===============

def sample_object_surface_points(obj, n_points: int = 2048) -> np.ndarray:
    """
    对单个 MeshObject 做均匀表面采样（世界坐标系），返回 [N,3] 的点云.
    """
    bpy_obj = obj.blender_obj
    if bpy_obj is None or bpy_obj.type != 'MESH':
        return np.zeros((0, 3), dtype=np.float32)

    mesh = bpy_obj.data
    if mesh is None or len(mesh.polygons) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    mat_world = bpy_obj.matrix_world
    verts_world = [mat_world @ v.co for v in mesh.vertices]

    tri_verts = []
    areas = []
    for poly in mesh.polygons:
        idxs = list(poly.vertices)
        if len(idxs) < 3:
            continue
        v0 = verts_world[idxs[0]]
        for i in range(1, len(idxs) - 1):
            v1 = verts_world[idxs[i]]
            v2 = verts_world[idxs[i + 1]]
            tri_verts.append((v0, v1, v2))
            area = ((v1 - v0).cross(v2 - v0)).length / 2.0
            areas.append(area)

    if not tri_verts:
        return np.zeros((0, 3), dtype=np.float32)

    areas = np.asarray(areas, dtype=np.float64)
    total_area = areas.sum()
    if total_area <= 0:
        pts = np.array([[v.x, v.y, v.z] for v in verts_world], dtype=np.float32)
        return pts

    probs = areas / total_area
    tri_indices = np.random.choice(len(tri_verts), size=n_points, p=probs)
    pts = np.zeros((n_points, 3), dtype=np.float32)
    for i, tri_idx in enumerate(tri_indices):
        v0, v1, v2 = tri_verts[tri_idx]
        u = random.random()
        v = random.random()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        w = 1.0 - u - v
        p = v0 * w + v1 * u + v2 * w
        pts[i, :] = [p.x, p.y, p.z]
    return pts


def export_scene_point_cloud_npy(
    objects,
    mapping,
    scene_id: str,
    output_dir: str,
    points_per_object: int = 2048,
    background_cats=("wall", "floor", "ceiling", "unknown")
):
    os.makedirs(output_dir, exist_ok=True)

    pcs = []
    for obj in objects:
        if not isinstance(obj, blenderproc.types.MeshObject):
            continue

        cat = get_category_name(obj, mapping, all_objects=objects)
        if cat in background_cats:
            continue

        pts = sample_object_surface_points(obj, n_points=points_per_object)
        if pts.shape[0] == 0:
            continue
        pcs.append(pts)

    if not pcs:
        print(f"[PC] Scene {scene_id}: no foreground points, skip saving.")
        return

    pc = np.concatenate(pcs, axis=0)
    out_path = os.path.join(output_dir, f"{scene_id}_pc.npy")
    np.save(out_path, pc)
    print(f"[PC] Scene {scene_id}: point cloud saved to: {out_path}, shape = {pc.shape}")


def estimate_size_label(size_vec) -> str:
    diag = float(np.linalg.norm(size_vec))
    if diag < 0.5:
        return "small"
    elif diag < 1.5:
        return "medium"
    else:
        return "large"


def compute_overlap_ratio_2d(bbox_sub, bbox_container):
    x1 = max(bbox_sub[0], bbox_container[0])
    y1 = max(bbox_sub[1], bbox_container[1])
    x2 = min(bbox_sub[2], bbox_container[2])
    y2 = min(bbox_sub[3], bbox_container[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    sub_area = (bbox_sub[2] - bbox_sub[0]) * (bbox_sub[3] - bbox_sub[1])
    if sub_area <= 1e-6:
        return 0.0
    return inter_area / sub_area


def check_support(subj_node, obj_node, dist_thresh=0.15, overlap_thresh=0.2):
    s_min = np.array(subj_node["bbox_3d"]["corners"]).min(axis=0)
    s_max = np.array(subj_node["bbox_3d"]["corners"]).max(axis=0)
    o_min = np.array(obj_node["bbox_3d"]["corners"]).min(axis=0)
    o_max = np.array(obj_node["bbox_3d"]["corners"]).max(axis=0)

    vertical_gap = o_min[2] - s_max[2]
    if not (-0.05 <= vertical_gap <= dist_thresh):
        return False

    s_rect = [s_min[0], s_min[1], s_max[0], s_max[1]]
    o_rect = [o_min[0], o_min[1], o_max[0], o_max[1]]
    overlap_ratio = compute_overlap_ratio_2d(o_rect, s_rect)
    return overlap_ratio > overlap_thresh


def get_directional_relation(subj_center, obj_center):
    diff = subj_center - obj_center
    dx, dy, dz = diff
    if abs(dx) < 0.1 and abs(dy) < 0.1:
        return None
    if abs(dx) > abs(dy):
        return "right_of" if dx > 0 else "left_of"
    else:
        return "behind" if dy > 0 else "in_front_of"


def check_inside(container_node, object_node):
    c_min = np.array(container_node["bbox_3d"]["corners"]).min(axis=0)
    c_max = np.array(container_node["bbox_3d"]["corners"]).max(axis=0)
    o_min = np.array(object_node["bbox_3d"]["corners"]).min(axis=0)
    o_max = np.array(object_node["bbox_3d"]["corners"]).max(axis=0)
    buffer = 0.05
    return (o_min[0] >= c_min[0] - buffer and o_max[0] <= c_max[0] + buffer and
            o_min[1] >= c_min[1] - buffer and o_max[1] <= c_max[1] + buffer and
            o_min[2] >= c_min[2] - buffer and o_max[2] <= c_max[2] + buffer)


def build_scene_graph(objects, mapping, scene_id, dataset_name="3D-FRONT-edit"):
    nodes = []
    idx = 0

    background_cats = ("wall", "floor", "ceiling", "unknown")
    foreground_cats = [obj for obj in objects
                       if (get_category_name(obj, mapping, all_objects=objects) not in background_cats)]
    print(f"[SceneGraph] Scene {scene_id}: {len(foreground_cats)} foreground objects detected.")

    for obj in objects:
        if not isinstance(obj, blenderproc.types.MeshObject):
            continue

        obj_id = str(idx)
        idx += 1
        cat = get_category_name(obj, mapping, all_objects=objects)
        if cat in background_cats:
            continue

        bbox = np.array(obj.get_bound_box(local_coords=False))
        size = bbox.max(axis=0) - bbox.min(axis=0)
        center = bbox.mean(axis=0)

        super_cat = obj.get_cp("super-category") if obj.has_cp("super-category") else None
        model_id = obj.get_cp("model_id") if obj.has_cp("model_id") else None
        room_name = obj.get_cp("room_type") if obj.has_cp("room_type") else None
        room_id = obj.get_cp("room_id") if obj.has_cp("room_id") else None
        color_name = estimate_color_name_from_materials(
            obj,
            mapping=mapping,
            debug_render=True,
            debug_output_dir="/path/to/ambirefer3d_artifact",
            scene_id=scene_id,
        )
        size_label = estimate_size_label(size)

        node = {
            "object_id": obj_id,
            "instance_name": obj.get_name(),
            "category": cat,
            "super_category": super_cat,
            "model_id": model_id,
            "room_id": room_id,
            "room_name": room_name,
            "location": center.tolist(),
            "bbox_3d": {
                "center": center.tolist(),
                "size": size.tolist(),
                "corners": bbox.tolist()
            },
            "orientation": {
                "euler": list(obj.get_rotation_euler())
            },
            "attributes": {
                "color": color_name,
                "size": size_label,
                "material": None,
                "shape": None
            },
            "render": {}
        }
        nodes.append(node)

    relations = []
    node_cache = {}
    for n in nodes:
        node_cache[n["object_id"]] = {
            "center": np.array(n["bbox_3d"]["center"]),
            "node": n
        }

    for i, o1 in enumerate(nodes):
        for j, o2 in enumerate(nodes):
            if i == j:
                continue
            id1, id2 = o1["object_id"], o2["object_id"]
            c1 = node_cache[id1]["center"]
            c2 = node_cache[id2]["center"]
            dist = float(np.linalg.norm(c1 - c2))
            rel_types = set()

            if check_support(o2, o1):
                rel_types.add("on")
                rel_types.add("supporting")
            elif check_support(o1, o2):
                rel_types.add("under")

            container_cats = {"cabinet", "cupboard", "shelf", "refrigerator", "box", "table", "desk"}
            if o2["category"] in container_cats and check_inside(o2, o1):
                rel_types.add("inside")

            xy_dir = get_directional_relation(c1, c2)
            if xy_dir:
                rel_types.add(xy_dir)

            if "on" not in rel_types and "under" not in rel_types:
                if c1[2] > c2[2] + 0.5:
                    rel_types.add("above")
                elif c1[2] < c2[2] - 0.5:
                    rel_types.add("below")

            is_contact = any(r in rel_types for r in ["on", "under", "inside", "supporting"])
            if not is_contact and dist < 1.5:
                rel_types.add("near")

            for rel in rel_types:
                if rel == "supporting":
                    continue
                relations.append({
                    "subj": id1,
                    "obj": id2,
                    "predicate": rel,
                    "geom": {"dist": dist}
                })

    candidate_groups = []
    seen_groups = set()
    cat2objs = {}
    for o in nodes:
        cat = o["category"]
        if cat in ("wall", "floor", "ceiling", "unknown"):
            continue
        cat2objs.setdefault(cat, []).append(o)

    for cat, objs in cat2objs.items():
        if len(objs) < 2:
            continue
        for tgt in objs:
            tgt_center = np.array(tgt["bbox_3d"]["center"])
            tgt_size = np.array(tgt["bbox_3d"]["size"])
            tgt_diag = float(np.linalg.norm(tgt_size))
            cand_ids = []
            for other in objs:
                c2 = np.array(other["bbox_3d"]["center"])
                size2 = np.array(other["bbox_3d"]["size"])
                diag2 = float(np.linalg.norm(size2))
                dist = float(np.linalg.norm(c2 - tgt_center))
                if dist < 2.5 and (min(tgt_diag, diag2) / max(tgt_diag, diag2 + 1e-6)) > 0.6:
                    cand_ids.append(other["object_id"])

            if len(cand_ids) >= 2:
                if tgt["object_id"] not in cand_ids:
                    cand_ids.insert(0, tgt["object_id"])
                else:
                    cand_ids.remove(tgt["object_id"])
                    cand_ids.insert(0, tgt["object_id"])
                group_key = tuple(sorted(cand_ids))
                if group_key in seen_groups:
                    continue
                seen_groups.add(group_key)
                candidate_groups.append({
                    "group_id": len(candidate_groups),
                    "target_id": tgt["object_id"],
                    "candidate_ids": cand_ids,
                    "ambiguity_type": "spatial_same_category"
                })

    return {
        "scene_id": scene_id,
        "dataset": dataset_name,
        "objects": nodes,
        "relations": relations,
        "candidate_groups": candidate_groups
    }


# =============== Stage1+2：GT + 三种退化 ===============

SYSTEM_PARSE = (
    "You are an information extractor for 3D referring expressions.\n"
    "Extract ONLY information that is explicitly stated in the text.\n"
    "Return JSON with EXACT keys:\n"
    "- class: string or null\n"
    "- attributes: object with optional keys {color,size,material,shape}; values string or null\n"
    "- spatial_relations: ARRAY of objects with keys {rel, refs, surface}\n"
    "    rel ∈ {right_of,left_of,next_to,on,under,in_front_of,behind,near,between}\n"
    "    refs is an ARRAY of reference object phrases (nouns), e.g. ['table'] or ['chair','sofa']\n"
    "    surface is the EXACT phrase span from the text, e.g. 'next to the table near the window'\n"
    "- room_context: string or null\n"
    "- orientation: string or null\n\n"
    "Rules:\n"
    "* NO hallucination. Extract only what the text explicitly states.\n"
    "* Keep all surface phrases EXACTLY as they appear in the text.\n"
    "* 'refs' must be an array of object reference strings (usually nouns).\n"
    "* If a spatial phrase mentions another object, you MUST NOT leave 'refs' empty.\n"
    "* If something is not mentioned, set it to null or empty array.\n"
    "* JSON only. No explanations.\n\n"
)

USER_PARSE_TMPL = """GROUND-TRUTH QUERY:
"{gt_query}"

Return JSON only.
"""

SYSTEM_POLISH_DEGRADE = (
    "You are given a referring expression for a 3D object in an indoor scene.\n"
    "The sentence may sound templated or slightly unnatural.\n"
    "Rewrite it into ONE fluent, natural English sentence that preserves ALL\n"
    "the original information (category, attributes, spatial relations, room, etc.).\n"
    "Do NOT add new details. Do NOT remove any information.\n"
    "Return plain text only, no explanations."
)

POLISH_USER_TMPL_DEGRADE = """ORIGINAL EXPRESSION:
"{expr}"
"""


def build_instance_bboxes(xyz_all, inst_all, sem_all):
    instance_ids = np.unique(inst_all)
    bboxes = []
    for inst_id in instance_ids:
        if inst_id == 0:
            continue

        mask = (inst_all == inst_id)
        pts = xyz_all[mask]
        if pts.shape[0] == 0:
            continue
        xmin, ymin, zmin = pts.min(axis=0)
        xmax, ymax, zmax = pts.max(axis=0)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cz = (zmin + zmax) / 2.0
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        sem_ids = sem_all[mask]
        if sem_ids.size == 0:
            sem_id = 0
        else:
            sem_id = int(sem_ids[0])
        bbox = np.array([cx, cy, cz, dx, dy, dz, sem_id, int(inst_id) - 1], dtype=np.float32)
        bboxes.append(bbox)

    if not bboxes:
        return np.zeros((1, 8), dtype=np.float32)
    return np.vstack(bboxes)


def call_vllm_polish_degrade(model: str, expr: str) -> str:
    user_prompt = POLISH_USER_TMPL_DEGRADE.format(expr=expr.replace('"', '\\"'))
    text = call_vllm(model=model, system_prompt=SYSTEM_POLISH_DEGRADE, user_prompt=user_prompt)
    return text.strip().strip('"')


COMMON_OBJECT_WORDS = [
    "bed", "chair", "table", "desk", "sofa", "couch",
    "cabinet", "wardrobe", "door", "window", "lamp",
    "tv", "television", "bookshelf", "shelf"
]


def backfill_spatial_refs_from_surface(parsed: Dict[str, Any]) -> Dict[str, Any]:
    srs = parsed.get("spatial_relations") or []
    for sr in srs:
        refs = sr.get("refs") or []
        surface = (sr.get("surface") or "").lower()
        if refs or not surface:
            continue
        guessed = []
        for w in COMMON_OBJECT_WORDS:
            if w in surface:
                guessed.append(w)
        if guessed:
            sr["refs"] = list(dict.fromkeys(guessed))
        else:
            sr["refs"] = []
    parsed["spatial_relations"] = srs
    return parsed


def normalize_parsed(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        parsed = {}

    cls = parsed.get("class")
    if cls is not None:
        cls = str(cls).strip()
        if not cls:
            cls = None
    parsed["class"] = cls

    attrs = parsed.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    out_attrs = {}
    for k in ("color", "size", "material", "shape"):
        v = attrs.get(k)
        if v is not None:
            v = str(v).strip()
            if v:
                out_attrs[k] = v
    parsed["attributes"] = out_attrs

    srs_in = parsed.get("spatial_relations") or []
    srs_out = []
    if isinstance(srs_in, list):
        for sr in srs_in:
            if not isinstance(sr, dict):
                continue
            rel = str(sr.get("rel") or "").strip()
            refs = sr.get("refs") or []
            surface = str(sr.get("surface") or "").strip()
            if not rel or not surface:
                continue
            if not isinstance(refs, list):
                refs = [refs]
            refs = [str(r).strip() for r in refs if str(r).strip()]
            srs_out.append({"rel": rel, "refs": refs, "surface": surface})
    parsed["spatial_relations"] = srs_out

    rc = parsed.get("room_context")
    if rc is not None:
        rc = str(rc).strip()
        if not rc:
            rc = None
    parsed["room_context"] = rc

    ori = parsed.get("orientation")
    if ori is not None:
        ori = str(ori).strip()
        if not ori:
            ori = None
    parsed["orientation"] = ori

    return parsed


def atomize_components(parsed: Dict[str, Any]) -> Tuple[List[str], Dict[str, str]]:
    atoms: List[str] = []
    atom_types: Dict[str, str] = {}
    cls = parsed.get("class")
    if cls:
        a = f"class:{cls}"
        atoms.append(a)
        atom_types[a] = "class"

    attrs = parsed.get("attributes") or {}
    for k in ("color", "size", "material", "shape"):
        v = attrs.get(k)
        if not v:
            continue
        a = f"attribute_{k}:{v}"
        atoms.append(a)
        atom_types[a] = "visual"

    for sr in parsed.get("spatial_relations") or []:
        rel = sr.get("rel")
        refs = sr.get("refs") or []
        if not rel or not refs:
            continue
        if len(refs) == 1:
            atom = f"sprel:{rel}({refs[0]})"
        else:
            atom = f"sprel:{rel}({refs[0]}|{refs[1]})"
        atoms.append(atom)
        atom_types[atom] = "spatial"

    rc = parsed.get("room_context")
    if rc:
        a = f"room:{rc}"
        atoms.append(a)
        atom_types[a] = "spatial"

    ori = parsed.get("orientation")
    if ori:
        a = f"orientation:{ori}"
        atoms.append(a)
        atom_types[a] = "spatial"

    return atoms, atom_types


def deatomize_to_query(
    parsed: Dict[str, Any],
    keep_atoms: List[str],
    atom_types: Dict[str, str]
) -> str:
    keep_set = set(keep_atoms)

    class_name = None
    for a in keep_atoms:
        if atom_types.get(a) == "class":
            class_name = a.split(":", 1)[1]
            break
    if not class_name:
        class_name = parsed.get("class") or "object"

    attrs = parsed.get("attributes") or {}
    head_words: List[str] = []
    for k in ("color", "size", "material", "shape"):
        v = attrs.get(k)
        if not v:
            continue
        atom = f"attribute_{k}:{v}"
        if atom in keep_set:
            head_words.append(v)

    surfaces: List[str] = []
    for sr in parsed.get("spatial_relations") or []:
        rel = sr.get("rel")
        refs = sr.get("refs") or []
        surface = sr.get("surface") or ""
        if not surface:
            continue
        if len(refs) == 1:
            atom = f"sprel:{rel}({refs[0]})"
        else:
            atom = f"sprel:{rel}({refs[0]}|{refs[1]})"
        if atom in keep_set:
            surfaces.append(surface)

    rc = parsed.get("room_context")
    if rc:
        atom = f"room:{rc}"
        if atom in keep_set:
            surfaces.append(f"in the {rc}")

    ori = parsed.get("orientation")
    if ori:
        atom = f"orientation:{ori}"
        if atom in keep_set:
            surfaces.append(ori)

    det = "a"
    if class_name and class_name[0].lower() in "aeiou":
        det = "an"

    head = " ".join([w for w in head_words if w] + [class_name]).strip()
    if not head:
        head = class_name

    if not surfaces:
        return f"{det} {head}".strip()
    else:
        tail = ", ".join(s.strip().strip(",") for s in surfaces if s.strip())
        return f"{det} {head}, {tail}".strip()


SYSTEM_GEN_QUERY = (
    "You are generating a natural language 3D referring expression for a target object in a 3D indoor scene.\n"
    "Use the provided structured scene description to write ONE concise, natural description.\n"
    "The description SHOULD uniquely identify the target object among similar candidates.\n"
    "Prefer using: category, visual attributes (color/size/material/shape), spatial relations, and room.\n"
    "Avoid using IDs like 'object_12'. Return PLAIN TEXT only."
)

USER_GEN_QUERY_TMPL = """SCENE STRUCTURED INFO:
{scene_info}

TARGET OBJECT:
object_id: {object_id}
category: {category}
room: {room}
attributes (if any): {attributes}

CANDIDATE GROUP:
target_id: {target_id}
candidate_ids: {candidate_ids}

Write ONE natural referring expression in English.
"""


def generate_initial_query_from_scene_graph(item: Dict[str, Any], model: str) -> str:
    scene_info = json.dumps(
        {
            "objects_brief": item.get("objects_brief"),
            "relations_brief": item.get("relations_brief"),
        },
        ensure_ascii=False
    )
    user_prompt = USER_GEN_QUERY_TMPL.format(
        scene_info=scene_info,
        object_id=item.get("object_id"),
        category=item.get("category"),
        room=item.get("room_name"),
        attributes=json.dumps(item.get("attributes") or {}, ensure_ascii=False),
        target_id=item.get("object_id"),
        candidate_ids=json.dumps(item.get("candidate_ids") or [], ensure_ascii=False),
    )
    text = call_vllm(model, SYSTEM_GEN_QUERY, user_prompt)
    return text.strip('" \n')


def compress_relations(relations_brief: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pair2preds = {}
    for r in relations_brief:
        key = (r["subj"], r["obj"])
        pair2preds.setdefault(key, set()).add(r["predicate"])

    compressed = []
    for (subj, obj), preds in pair2preds.items():
        pred_combo = "+".join(sorted(preds))
        compressed.append({"subj": subj, "obj": obj, "predicate": pred_combo})
    return compressed


def load_items_from_scene_graph(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        sg = json.load(f)
    scene_id = sg["scene_id"]
    all_objects = sg["objects"]
    relations = sg.get("relations", [])
    groups = sg.get("candidate_groups", [])

    background_cats = {"wall", "floor", "cabinet", "ceiling", "unknown"}
    fg_objects = [o for o in all_objects if o.get("category") not in background_cats]
    objects_dict = {o["object_id"]: o for o in fg_objects}

    cat_count = {}
    for o in fg_objects:
        cat = o.get("category", "unknown")
        cat_count[cat] = cat_count.get(cat, 0) + 1

    relations_fg = []
    for r in relations:
        subj = r["subj"]
        obj = r["obj"]
        if subj in objects_dict and obj in objects_dict:
            relations_fg.append({
                "subj": subj,
                "obj": obj,
                "predicate": r["predicate"]
            })

    items = []
    for g in groups:
        target_id = g["target_id"]
        cand_ids_raw = g.get("candidate_ids", [])

        cand_ids = [cid for cid in cand_ids_raw if cid in objects_dict]
        if target_id not in cand_ids and target_id in objects_dict:
            cand_ids = [target_id] + cand_ids
        cand_ids = list(dict.fromkeys(cand_ids))
        if len(cand_ids) < 2:
            continue
        if target_id not in objects_dict:
            continue

        tgt = objects_dict[target_id]
        objects_brief = [{
            "object_id": oid,
            "category": objects_dict[oid].get("category"),
            "room_name": objects_dict[oid].get("room_name"),
        } for oid in cand_ids]
        cand_set = set(cand_ids)
        rel_brief = [
            r for r in relations_fg
            if (r["subj"] in cand_set) and (r["obj"] in cand_set)
        ]
        rel_brief = compress_relations(rel_brief)

        item = {
            "scene_id": scene_id,
            "object_id": target_id,
            "object_name": tgt.get("instance_name"),
            "category": tgt.get("category"),
            "room_name": tgt.get("room_name"),
            "attributes": tgt.get("attributes") or {},
            "candidate_ids": cand_ids,
            "group_id": g["group_id"],
            "ambiguity_type": g.get("ambiguity_type"),
            "objects_brief": objects_brief,
            "relations_brief": rel_brief,
            "description": None,
        }
        items.append(item)

    return items


def build_degradation_variants(
    parsed: Dict[str, Any],
    atoms: List[str],
    atom_types: Dict[str, str],
    model: str,
) -> Dict[str, Any]:
    def split_atoms(by_type: str) -> List[str]:
        return [a for a in atoms if atom_types.get(a) == by_type]

    def make_variant(name: str, keep_atoms: List[str]) -> Dict[str, Any]:
        keep_set = set(keep_atoms)
        removed = [a for a in atoms if a not in keep_set]
        removed_detail = [{"atom": a, "atom_type": atom_types.get(a, "unknown")}
                          for a in removed]
        degraded_struct = deatomize_to_query(parsed, keep_atoms, atom_types)
        degraded_query = call_vllm_polish_degrade(model, degraded_struct)
        return {
            "degraded_type": name,
            "degraded_query_struct": degraded_struct,
            "degraded_query": degraded_query,
            "kept_atoms": keep_atoms,
            "removed_atoms": removed,
            "removed_detail": removed_detail,
        }

    spatial_atoms = split_atoms("spatial")
    visual_atoms = split_atoms("visual")
    class_atoms = split_atoms("class")

    keep_no_spatial = [a for a in atoms if atom_types.get(a) != "spatial"]
    variant_no_spatial = make_variant("no_spatial", keep_no_spatial)

    keep_no_visual = [a for a in atoms if atom_types.get(a) != "visual"]
    variant_no_visual = make_variant("no_visual", keep_no_visual)

    keep_minimal = class_atoms[:] if class_atoms else []
    variant_minimal = make_variant("minimal", keep_minimal)

    return {
        "no_spatial": variant_no_spatial,
        "no_visual": variant_no_visual,
        "minimal": variant_minimal,
    }


def build_one_sample(item: Dict[str, Any], model: str) -> Dict[str, Any]:
    scene_id = item.get("scene_id") or "unknown_scene"
    object_id = item.get("object_id") or "unknown_obj"
    object_name = item.get("object_name")
    ann_id = f"group{item.get('group_id', 0)}"
    gt_query = (item.get("description") or "").strip()
    if not gt_query:
        gt_query = generate_initial_query_from_scene_graph(item, model)

    parsed_raw = call_vllm(
        model,
        SYSTEM_PARSE,
        USER_PARSE_TMPL.format(gt_query=gt_query)
    )
    parsed = safe_json_loads(parsed_raw)
    parsed = normalize_parsed(parsed)

    if not parsed.get("class") and item.get("category"):
        parsed["class"] = item["category"]

    parsed = backfill_spatial_refs_from_surface(parsed)
    atoms, atom_types = atomize_components(parsed)
    degradation = build_degradation_variants(parsed, atoms, atom_types, model=model)

    return {
        "scene_id": scene_id,
        "object_id": object_id,
        "object_name": object_name,
        "ann_id": ann_id,
        "group_id": item.get("group_id"),
        "candidate_ids": item.get("candidate_ids"),
        "ambiguity_type": item.get("ambiguity_type"),
        "gt": {"gt_query": gt_query},
        "parsed_gt": parsed,
        "atoms": {"all": atoms, "types": atom_types},
        "degradation": degradation,
        "tags": {
            "light_mode": LIGHT_MODE,
            "pipeline_stage": "stage1_2_three_way_degradation",
        },
    }


def run_stage1_2_for_scene(scene_graph_path: Path,
                           out_dir: Path,
                           text_model: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_items_from_scene_graph(scene_graph_path)
    if not items:
        return

    for item in items:
        try:
            out = build_one_sample(item, text_model)
        except Exception as e:
            print(f"[Stage1_2] Error building sample: {e}")
            continue

        scene_id = out["scene_id"]
        object_id = out["object_id"]
        ann_id = out["ann_id"]
        out_name = f"{scene_id}__obj{object_id}__ann{ann_id}.json"
        with (out_dir / out_name).open("w", encoding="utf-8") as fw:
            json.dump(out, fw, ensure_ascii=False, indent=2)


# =============== Stage3：文本 feedback（改为 VLM + 多视角） ===============

SYSTEM_POLISH = (
    "You are given a referring expression for a 3D object in an indoor scene.\n"
    "The sentence may sound a bit templated or awkward.\n"
    "Rewrite it into ONE fluent, natural English sentence that preserves ALL\n"
    "the original information (category, attributes, spatial relations, room, etc.).\n"
    "Do NOT add new details. Do NOT remove any information.\n"
    "Return plain text only, no explanations."
)

POLISH_USER_TMPL = """ORIGINAL EXPRESSION:
"{expr}"
"""


def call_vllm_polish(model: str, expr: str) -> str:
    user_prompt = POLISH_USER_TMPL.format(expr=expr.replace('"', '\\"'))
    text = call_vllm(model=model, system_prompt=SYSTEM_POLISH, user_prompt=user_prompt)
    return text.strip().strip('"')


def classify_atom_type(atom: str) -> str:
    if atom.startswith("class:"):
        return "class"
    if atom.startswith("attribute_color:"):
        return "visual_color"
    if atom.startswith("attribute_size:"):
        return "visual_size"
    if atom.startswith("attribute_material:"):
        return "visual_material"
    if atom.startswith("attribute_shape:"):
        return "visual_shape"
    if atom.startswith("sprel:"):
        return "spatial_relation"
    if atom.startswith("room:"):
        return "room"
    if atom.startswith("orientation:"):
        return "orientation"
    return "other"


def infer_removed_info_tags(removed_atoms: list, atom_types: dict) -> list:
    """从 removed_atoms 推断大类：visual_attributes / spatial"""
    types = set()
    for a in removed_atoms:
        t = atom_types.get(a) or classify_atom_type(a)
        types.add(t)
    tags = set()
    if any(t.startswith("visual") for t in types):
        tags.add("visual_attributes")
    if any(t in {"spatial_relation", "room", "orientation"} for t in types):
        tags.add("spatial")
    return sorted(tags)


def format_missing_hint_from_tags(tags: list) -> str:
    if not tags:
        return "None"
    parts = []
    if "visual_attributes" in tags:
        parts.append("VISUAL APPEARANCE")
    if "spatial" in tags:
        parts.append("SPATIAL LOCATION / RELATION")
    return " & ".join(parts)


def summarize_candidates_for_text(sample: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
    """
    把 Stage1 sample 中的候选对象 + 关系压缩成一段纯文本，给 VLM prompt 用。
    返回：
      - 多行字符串
      - object_id -> Candidate 序号 映射
    """
    objs_brief = sample.get("objects_brief") or []
    rels_brief = sample.get("relations_brief") or []
    obj_dict = {str(o["object_id"]): o for o in objs_brief}
    target_id = str(sample.get("object_id"))
    candidate_ids = [str(i) for i in (sample.get("candidate_ids") or [])]

    lines = []
    id2idx = {}
    for idx, oid in enumerate(candidate_ids, start=1):
        ob = obj_dict.get(oid, {})
        cat = ob.get("category", "unknown")
        room = ob.get("room_name") or "unknown room"
        role = "TARGET" if oid == target_id else "DISTRACTOR"
        line = (
            f"[Candidate {idx}] "
            f"Role: {role}; "
            f"ID: {oid}; "
            f"Category: {cat}; "
            f"Room: {room}"
        )
        lines.append(line)
        id2idx[oid] = idx

    relation_lines = []
    for r in rels_brief:
        s = str(r.get("subj"))
        o = str(r.get("obj"))
        if s not in id2idx or o not in id2idx:
            continue
        pred = r.get("predicate", "")
        cs = id2idx[s]
        co = id2idx[o]
        relation_lines.append(f"- Candidate {cs} is '{pred}' Candidate {co}")

    if relation_lines:
        lines.append("\n[Relations among candidates]")
        lines.extend(relation_lines[:10])

    return "\n".join(lines), id2idx


SYSTEM_STAGE3_VLM = """
You are an expert multimodal assistant for resolving ambiguous 3D referring expressions.

### TASK
The user gives a referring expression (query) for an object in an indoor 3D scene.
You are given:
1. The current (possibly ambiguous) query.
2. A list of candidate objects (one TARGET and several DISTRACTORS), with their:
   - Category
   - Room name
   - Pairwise relations (e.g., on, under, next_to, in_front_of, behind, inside, near, above, below).
3. Several images for each candidate object (multi-view crops).
4. A rough hint about what information may be missing (visual or spatial).

Your job:
- Decide whether the query is truly ambiguous among the candidates.
- If ambiguous: pick ONE best discriminative feature (color, size, room, or spatial relation).
- Ask ONE short clarification question about that feature.
- Propose ONE restored (completed) query that would uniquely identify the target.

If the query already uniquely identifies the target:
- Treat it as NON-ambiguous.
- Return "question": null and "restored_query": null.

### RULES
- Use images mainly for VISUAL differences (color, material, shape, style).
- Use the textual candidate list for ROOM and RELATIONAL differences.
- The clarification question must be:
  - Natural English.
  - Short (<= 20 words).
  - About ONE specific feature only.
- The restored_query = original query + that feature.
  - Example: "the chair" -> "the red chair"
  - Example: "the pillow" -> "the pillow on the bed"

### OUTPUT FORMAT
Return ONE JSON object:
{
  "reasoning": string,
  "question": string or null,
  "restored_query": string or null
}

If there is no ambiguity, explain why in reasoning and set question/restored_query to null.
Do NOT mention candidate indices, object IDs, or coordinates in your question or restored_query.
"""

USER_STAGE3_VLM_TMPL = """
### CURRENT QUERY
"{query}"

### LIKELY MISSING INFO
{missing_hint}

### CANDIDATES (TARGET + DISTRACTORS)
{candidates_text}

Now strictly follow the TASK and OUTPUT FORMAT and return JSON only.
"""


def get_multiview_images_for_object(scene_id: str, object_id: str, max_total_imgs: int = 8):
    """
    在 RENDER_ROOT/scene_id 下，按 {scene_id}_obj{object_id}_*.jpg 查找多视角图片。
    返回 PIL.Image 列表。
    """
    scene_dir = os.path.join(RENDER_ROOT, scene_id)
    if not os.path.isdir(scene_dir):
        print(f"[Stage3-VLM] Render dir not found: {scene_dir}")
        return []

    prefix = f"{scene_id}_obj{object_id}_"
    paths = []
    try:
        for fname in sorted(os.listdir(scene_dir)):
            if fname.startswith(prefix) and fname.lower().endswith(".jpg"):
                paths.append(os.path.join(scene_dir, fname))
    except Exception as e:
        print(f"[Stage3-VLM] List dir error: {scene_dir}, {e}")
        return []

    images = []
    for p in paths[:max_total_imgs]:
        img = resize_image_smart(p)
        if img is not None:
            images.append(img)
    if not images:
        print(f"[Stage3-VLM] No images for {scene_id}, obj {object_id}")
    return images


def run_text_clarification_for_variant(
    sample: Dict[str, Any],
    variant_name: str,
    variant_info: Dict[str, Any],
    model: str,
    max_rounds: int,
) -> Dict[str, Any]:
    """
    VLM 多视角版 Stage3：
    - 每个 variant 只调用一次 qwen-vl
    - 输入：当前歧义 query + missing info hint + 候选文本 + 候选多视角图像
    - 输出：reasoning + 最佳澄清问题 + restored_query
    """
    scene_id = sample["scene_id"]
    target_id = str(sample["object_id"])
    gt_query = sample["gt"]["gt_query"]

    ambiguous_query = (variant_info.get("degraded_query") or gt_query).strip()

    atoms_all: List[str] = sample.get("atoms", {}).get("all", [])
    atom_types_map: Dict[str, str] = sample.get("atoms", {}).get("types", {})

    removed_atoms = variant_info.get("removed_atoms", []) or []
    missing_tags = infer_removed_info_tags(removed_atoms, atom_types_map)
    missing_hint = format_missing_hint_from_tags(missing_tags)

    candidates_text, id2idx = summarize_candidates_for_text(sample)
    candidate_ids = [str(i) for i in (sample.get("candidate_ids") or [])]

    user_contents: List[Dict[str, Any]] = []
    MAX_TOTAL_IMGS = 20
    total_imgs = 0

    for oid in candidate_ids:
        imgs = get_multiview_images_for_object(scene_id, oid, max_total_imgs=4)
        for img in imgs:
            if total_imgs >= MAX_TOTAL_IMGS:
                break
            b64 = encode_image_to_base64(img)
            user_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
            total_imgs += 1
        if total_imgs >= MAX_TOTAL_IMGS:
            break

    user_prompt = USER_STAGE3_VLM_TMPL.format(
        query=ambiguous_query.replace('"', '\\"'),
        missing_hint=missing_hint,
        candidates_text=candidates_text,
    )
    user_contents.append({"type": "text", "text": user_prompt})
    # from ipdb import set_trace; set_trace()
    try:
        resp = color_vlm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_STAGE3_VLM},
                {"role": "user", "content": user_contents},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        raw_text = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[Stage3-VLM] API error on scene {scene_id}, obj {target_id}, variant {variant_name}: {e}")
        return {
            "variant_name": variant_name,
            "clarification_dialogue": [],
            "restored_query_struct": ambiguous_query,
            "restored_query": ambiguous_query,
        }

    try:
        result = safe_json_loads(raw_text)
    except Exception as e:
        print(f"[Stage3-VLM] JSON parse error: {e}, raw={raw_text[:200]}...")
        return {
            "variant_name": variant_name,
            "clarification_dialogue": [],
            "restored_query_struct": ambiguous_query,
            "restored_query": ambiguous_query,
        }

    reasoning = (result.get("reasoning") or "").strip()
    question = result.get("question", None)
    restored_query = result.get("restored_query", None)

    def is_empty(x):
        if x is None:
            return True
        s = str(x).strip().lower()
        return (not s) or (s in {"none", "null", "n/a"})

    if is_empty(question) and is_empty(restored_query):
        return {
            "variant_name": variant_name,
            "clarification_dialogue": [],
            "restored_query_struct": ambiguous_query,
            "restored_query": ambiguous_query,
        }

    if is_empty(restored_query):
        restored_query = ambiguous_query

    # 可选：再用文本 LLM 美化一下
    restored_query_polished = call_vllm_polish(model="qwen-llm", expr=restored_query)

    dialogue = [{
        "round": 0,
        "variant": variant_name,
        "atom": None,
        "atom_type": None,
        "query_before": ambiguous_query,
        "model_question": question,
        "oracle_question_template": None,
        "oracle_answer": None,
        "query_after": restored_query_polished,
        "reasoning": reasoning,
    }]

    return {
        "variant_name": variant_name,
        "clarification_dialogue": dialogue,
        "restored_query_struct": restored_query_polished,
        "restored_query": restored_query_polished,
    }


def run_stage3_for_scene(
    sample_dir: Path,
    out_dir: Path,
    text_model: str,
    max_rounds: int,
):
    """
    注意：这里 text_model 实际上传的是 VLM 模型名（比如 args.color_model = 'qwen-vl'）
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample_path in sample_dir.iterdir():
        if sample_path.suffix.lower() != ".json":
            continue

        with sample_path.open("r", encoding="utf-8") as f:
            sample = json.load(f)

        degradation = sample.get("degradation", {})
        if not degradation:
            continue

        stage3_results: Dict[str, Any] = {}
        for variant_name, variant_info in degradation.items():
            try:
                res = run_text_clarification_for_variant(
                    sample,
                    variant_name=variant_name,
                    variant_info=variant_info,
                    model=text_model,      # 这里用的是 VLM 模型名，比如 'qwen-vl'
                    max_rounds=max_rounds
                )
            except Exception as e:
                print(f"[Stage3] Error in variant {variant_name}: {e}")
                continue
            stage3_results[variant_name] = res

        sample["stage3_text"] = stage3_results
        out_path = out_dir / sample_path.name
        with out_path.open("w", encoding="utf-8") as fw:
            json.dump(sample, fw, ensure_ascii=False, indent=2)


# =============== ScanRefer 样式 npy 导出（可选） ===============

def get_semantic_id_from_cat(cat_name: str) -> int:
    return THREEDFRONT_CAT2NYU40.get(cat_name, THREEDFRONT_CAT2NYU40["others"])


def export_3dfront_scene_for_scanrefer(
    scene_id: str,
    objects,
    mapping,
    output_dir: str,
    points_per_object: int = 2048
):
    os.makedirs(output_dir, exist_ok=True)
    xyz_list = []
    rgb_list = []
    nrm_list = []
    inst_list = []
    sem_list = []
    inst_id_counter = 1

    for obj in objects:
        if not isinstance(obj, blenderproc.types.MeshObject):
            continue

        cat_name = get_category_name(obj, mapping, all_objects=objects)
        if cat_name in BACKGROUND_CATS:
            continue

        sem_id = get_semantic_id_from_cat(cat_name)
        pts = sample_object_surface_points(obj, n_points=points_per_object)
        if pts is None or pts.shape[0] == 0:
            continue

        if pts.shape[1] == 3:
            xyz = pts.astype(np.float32)
            rgb = np.zeros_like(xyz, dtype=np.float32)
            nrm = np.zeros_like(xyz, dtype=np.float32)
        elif pts.shape[1] == 9:
            xyz = pts[:, 0:3].astype(np.float32)
            rgb = pts[:, 3:6].astype(np.float32)
            nrm = pts[:, 6:9].astype(np.float32)
        else:
            print(f"[ScanReferExport] Unexpected points shape {pts.shape} for object {obj.get_name()}, skip.")
            continue

        Ni = xyz.shape[0]
        xyz_list.append(xyz)
        rgb_list.append(rgb)
        nrm_list.append(nrm)
        inst_list.append(np.full((Ni,), inst_id_counter, dtype=np.int32))
        sem_list.append(np.full((Ni,), sem_id, dtype=np.int32))
        inst_id_counter += 1

    if not xyz_list:
        print(f"[ScanReferExport] Scene {scene_id}: no foreground points, skip.")
        return

    xyz_all = np.concatenate(xyz_list, axis=0)
    rgb_all = np.concatenate(rgb_list, axis=0)
    nrm_all = np.concatenate(nrm_list, axis=0)
    inst_all = np.concatenate(inst_list, axis=0)
    sem_all = np.concatenate(sem_list, axis=0)

    mesh_vertices = np.concatenate([xyz_all, rgb_all, nrm_all], axis=1).astype(np.float32)
    aligned_vertices = mesh_vertices.copy()
    instance_bboxes = build_instance_bboxes(xyz_all, inst_all, sem_all).astype(np.float32)
    aligned_instance_bboxes = instance_bboxes.copy()

    base = os.path.join(output_dir, scene_id)
    if not os.path.exists(base + "_vert.npy"):
        np.save(base + "_vert.npy", mesh_vertices)
    if not os.path.exists(base + "_aligned_vert.npy"):
        np.save(base + "_aligned_vert.npy", aligned_vertices)
    if not os.path.exists(base + "_ins_label.npy"):
        np.save(base + "_ins_label.npy", inst_all.astype(np.int32))
    if not os.path.exists(base + "_sem_label.npy"):
        np.save(base + "_sem_label.npy", sem_all.astype(np.int32))
    if not os.path.exists(base + "_bbox.npy"):
        np.save(base + "_bbox.npy", instance_bboxes)
    if not os.path.exists(base + "_aligned_bbox.npy"):
        np.save(base + "_aligned_bbox.npy", aligned_instance_bboxes)


# =============== 顶层：批量跑所有场景 ===============

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front_dir", type=str, required=True,
                    help="包含多个 3D-FRONT 场景 json 的目录")
    ap.add_argument("--future_folder", type=str, required=True,
                    help="3D-FUTURE 模型根目录")
    ap.add_argument("--front_3D_texture_path", type=str, required=True,
                    help="3D-FRONT 纹理根目录")
    ap.add_argument("--output_root", type=str, required=True,
                    help="总输出目录（每个 scene 一个子目录）")
    ap.add_argument("--model_info_json", type=str, required=True,
                    help="3D-FUTURE 的 model_info.json")
    ap.add_argument("--num_splits", type=int, default=1)
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--split", type=int, default=0,
                    help="数据划分：0-9 划分数据集占比")
    ap.add_argument("--text_model", type=str, default="qwen-llm",
                    help="用于 Stage1/2 的文本 LLM 模型名（vLLM）")
    ap.add_argument("--color_model", type=str, default="qwen-vl",
                    help="用于颜色估计和 Stage3 的视觉 LLM 模型名（vLLM）")
    ap.add_argument("--max_rounds", type=int, default=6,
                    help="Stage3 每个 variant 最多澄清轮数（VLM 版目前只用一轮）")
    ap.add_argument("--cc_material_folder", type=str, default="path/to/cc_materials",
                    help="CCTextures 材质根目录")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def process_one_scene(front_path, future_folder, texture_root,
                      output_root, mapping, model_info, args):
    scene_id = os.path.splitext(os.path.basename(front_path))[0]
    scene_out_dir = Path(output_root) / scene_id
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    blenderproc.clean_up()

    cc_materials = blenderproc.loader.load_ccmaterials(
        args.cc_material_folder,
        ["Bricks", "Wood", "Carpet", "Tile", "Marble"]
    )
    wood_floor_materials = blenderproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
    marble_materials = blenderproc.filter.by_cp(cc_materials, "asset_name", "Marble.*", regex=True)

    objects = blenderproc.loader.load_front3d(
        json_path=front_path,
        future_model_path=future_folder,
        front_3D_texture_path=texture_root,
        label_mapping=mapping
    )

    floors = blenderproc.filter.by_attr(objects, "name", "Floor.*", regex=True)
    for floor in floors:
        for i in range(len(floor.get_materials())):
            floor.set_material(i, random.choice(cc_materials))

    baseboards_and_doors = blenderproc.filter.by_attr(
        objects, "name", "Baseboard.*|Door.*", regex=True
    )
    for obj in baseboards_and_doors:
        for i in range(len(obj.get_materials())):
            if len(wood_floor_materials) > 0:
                obj.set_material(i, random.choice(wood_floor_materials))

    walls = blenderproc.filter.by_attr(objects, "name", "Wall.*", regex=True)
    for wall in walls:
        for i in range(len(wall.get_materials())):
            if len(marble_materials) > 0:
                wall.set_material(i, random.choice(marble_materials))

    edit_log = []
    # 可以在这里做 random_move / random_replace 等编辑操作（暂时注释）

    objects_for_sg = blenderproc.object.get_all_mesh_objects()
    color_debug_dir = scene_out_dir / "_color_dbg"

    sg = build_scene_graph(
        objects_for_sg,
        mapping,
        scene_id,
        dataset_name="3D-FRONT-edit"
    )
    sg_path = scene_out_dir / f"{scene_id}_scene_graph.json"
    with sg_path.open("w", encoding="utf-8") as f:
        json.dump(sg, f, ensure_ascii=False, indent=2)

    export_scene_point_cloud_npy(
        objects=objects_for_sg,
        mapping=mapping,
        scene_id=scene_id,
        output_dir=scene_out_dir,
        points_per_object=4096,
    )
    shutil.rmtree(color_debug_dir, ignore_errors=True)

    # =========== 新增：为所有 candidate 对象渲染多视角图像（Stage3 VLM 用） ===========
    cand_obj_ids = set()
    for g in sg.get("candidate_groups", []):
        for cid in g.get("candidate_ids", []):
            cand_obj_ids.add(str(cid))

    obj_nodes = {o["object_id"]: o for o in sg["objects"]}
    scene_render_dir = os.path.join(RENDER_ROOT, scene_id)

    for oid in cand_obj_ids:
        node = obj_nodes.get(oid)
        if not node:
            continue
        bbox_corners = node["bbox_3d"]["corners"]
        render_multiview_for_object(
            scene_id=scene_id,
            object_id=oid,
            bbox_corners=bbox_corners,
            all_objects=objects_for_sg,
            output_dir=scene_render_dir,
        )

    # Stage1+2：文本生成 + 退化
    stage1_dir = scene_out_dir / "stage1_2"
    run_stage1_2_for_scene(sg_path, stage1_dir, text_model=args.text_model)

    # Stage3：文本 feedback（用 VLM + 多视角）
    stage3_dir = scene_out_dir / "stage3"
    run_stage3_for_scene(
        stage1_dir,
        stage3_dir,
        text_model=args.color_model,     # 这里传入 VLM 模型名（qwen-vl）
        max_rounds=args.max_rounds,
    )

    # 可选：ScanRefer npy 导出
    # scanrefer_npy_dir = os.path.join(output_root, "scanrefer_npy")
    # export_3dfront_scene_for_scanrefer(
    #     scene_id=scene_id,
    #     objects=objects,
    #     mapping=mapping,
    #     output_dir=scanrefer_npy_dir,
    #     points_per_object=50000
    # )


def main():
    args = parse_args()
    front_root = args.front_dir
    future_folder = args.future_folder
    texture_root = args.front_3D_texture_path
    output_root = args.output_root

    blenderproc.init()

    mapping_file = blenderproc.utility.resolve_resource(
        os.path.join("front_3D", "3D_front_mapping.csv")
    )
    mapping = blenderproc.utility.LabelIdMapping.from_csv(mapping_file)
    model_info = load_model_info(args.model_info_json)

    front_jsons = sorted([
        os.path.join(front_root, f)
        for f in os.listdir(front_root)
        if f.endswith(".json")
    ])
    if not 0 <= args.split < args.num_splits:
        raise ValueError("Require 0 <= split < num_splits")
    front_jsons = front_jsons[args.split::args.num_splits]
    if args.max_scenes > 0:
        front_jsons = front_jsons[:args.max_scenes]
    random.seed(args.seed)
    np.random.seed(args.seed)

    # num_splits = 10
    # total_scenes = len(front_jsons)
    # split_size = total_scenes // num_splits
    # start_idx = args.split * split_size
    # end_idx = (args.split + 1) * split_size if args.split < num_splits - 1 else total_scenes
    # front_jsons = front_jsons[start_idx:end_idx]

    # print(f"[Main] Total scenes: {total_scenes}, split={args.split}, "
    #       f"processing {len(front_jsons)} scenes [{start_idx}, {end_idx})")

    for front_path in tqdm(front_jsons, desc="Processing scenes"):
        process_one_scene(
            front_path=front_path,
            future_folder=future_folder,
            texture_root=texture_root,
            output_root=output_root,
            mapping=mapping,
            model_info=model_info,
            args=args,
        )


if __name__ == "__main__":
    main()
