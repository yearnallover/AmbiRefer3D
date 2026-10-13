# 标准库 (Standard Library)
import argparse  # [NEW] 新增参数解析库
import base64
import json
import logging
import math
import os
import re
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

# 第三方库 (Third-party)
from PIL import Image
from openai import OpenAI, APIConnectionError, APIStatusError
from tqdm import tqdm

# ================= 配置区域 =================
API_BASE_URL = os.environ.get('AMBIREFER3D_VISION_API_BASE', 'http://localhost:8100/v1')
API_KEY = os.environ.get('AMBIREFER3D_API_KEY', 'EMPTY')
MODEL_NAME = os.environ.get('AMBIREFER3D_VISION_MODEL', 'qwen-vl')
MAX_RETRIES = 3
MAX_WORKERS = 4  # [建议] 既然是多进程分片，这里的线程数建议设为 4-8 即可，不要太大

# 路径配置
SCENE_GRAPH_ROOT = os.environ.get('AMBIREFER3D_REAL_SCENEGRAPH_ROOT', 'data/real/scenegraph')
RENDER_ROOT = os.environ.get('AMBIREFER3D_REAL_RENDER_ROOT', 'data/real/render')
INPUT_JSON_FILE = os.environ.get('AMBIREFER3D_REAL_FILTERED_JSON', 'data/real/filtered_queries.json')
OUTPUT_JSON_FILE = os.environ.get('AMBIREFER3D_REAL_DIALOGUES_JSONL', 'data/real/dialogues.jsonl')
LOG_FILE = os.environ.get('AMBIREFER3D_REAL_ERROR_LOG', 'output/ambirefer3d_data/real_errors.txt')

# 参数配置
RELATION_CUTOFF_DIST = 2.5
BASIC_IGNORE = {"floor", "wall", "ceiling", "object", "unknown", "person", "doorframe"}

# 全局写入锁
WRITE_LOCK = threading.Lock()

# ================= 日志初始化 =================
# 注意：这里初始化的日志是默认的，main函数里我们会根据分片重新配置一下日志文件名
log_dir = os.path.dirname(LOG_FILE)

# [FIX] 2. 如果文件夹不存在，强制创建它 (包含所有父级目录)
if log_dir and not os.path.exists(log_dir):
    try:
        os.makedirs(log_dir, exist_ok=True)
        print(f"✅ Created missing directory: {log_dir}")
    except Exception as e:
        print(f"❌ Failed to create directory {log_dir}: {e}")

# [FIX] 3. 然后再配置日志，这就不会报错了
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

# ================= 初始化 API 客户端 =================
try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        timeout=600.0
    )
    print(f"✅ API Client Connected: {API_BASE_URL}")
except Exception as e:
    print(f"❌ Client Init Failed: {e}")
    exit(1)

# ================= 工具函数 (保持不变) =================

def get_camera_basis(cam_pos, cam_lookat):
    """预计算相机的基向量 (Right, Up, Forward)"""
    P_cam = np.array(cam_pos)
    P_lookat = np.array(cam_lookat)
    
    forward = P_lookat - P_cam
    norm_f = np.linalg.norm(forward)
    if norm_f < 1e-6: return None 
    forward = forward / norm_f
    
    world_up = np.array([0, 0, 1]) 
    right = np.cross(forward, world_up)
    
    norm_r = np.linalg.norm(right)
    if norm_r < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / norm_r
        
    true_up = np.cross(right, forward)
    return (right, true_up, forward)

def project_vector_to_camera(center_pos, other_pos, basis):
    if basis is None: return [0,0,0]
    right, true_up, forward = basis
    
    # 相对向量 (Target -> Neighbor)
    obj_vec = np.array(other_pos) - np.array(center_pos)
    
    x = np.dot(obj_vec, right)
    y = np.dot(obj_vec, true_up)
    z = np.dot(obj_vec, forward)
    return [x, y, z]

def fmt_list(x):
    return "[" + ", ".join(f"{v:.2f}" for v in x) + "]"

def encode_image_to_base64(pil_image):
    buffered = BytesIO()
    if pil_image.mode != "RGB": pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG", quality=75)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def clean_json_output(text: str) -> str:
    text = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return text
    text = json_match.group(0)
    text = re.sub(r",\s*([\]}])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def resize_image_smart(image_path, max_edge=336):
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_edge:
            scale = max_edge / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        logging.error(f"[IMAGE_ERROR] {image_path}: {e}")
        return None

def get_multiview_images_by_pattern(scene_id, object_id) -> List[Image.Image]:
    scene_dir = os.path.join(RENDER_ROOT, scene_id)
    if not os.path.exists(scene_dir):
        return []
    prefix = f"{scene_id}_obj{object_id}_"
    found_paths = []
    try:
        for fname in sorted(os.listdir(scene_dir)):
            if fname.startswith(prefix) and fname.endswith(".jpg"):
                found_paths.append(os.path.join(scene_dir, fname))
    except: return []
    images = []
    for p in found_paths:
        img = resize_image_smart(p)
        if img: images.append(img)
    return images

def calculate_distance(pos1, pos2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

def format_missing_hint(removed_info: List[str]) -> str:
    if not removed_info: return "None"
    hints = []
    if "visual_attributes" in removed_info: hints.append("VISUAL APPEARANCE")
    if "spatial" in removed_info: hints.append("SPATIAL LOCATION")
    return " & ".join(hints)

# [关键修改] 使用全局变量 OUTPUT_JSON_FILE，这样 main 函数修改后这里能生效
def append_to_jsonl_thread_safe(data: Dict, file_path: str):
    try:
        json_str = json.dumps(data, ensure_ascii=False)
        with WRITE_LOCK: # 获取锁
            # 注意：这里直接使用传入的 file_path，或者使用全局变量
            with open(file_path, "a", encoding='utf-8') as f:
                f.write(json_str + "\n")
    except Exception as e:
        logging.error(f"Failed to write JSONL line: {e}")

def get_theta(x, y):
    x = np.array(x)
    y = np.array(y)
    l_x = np.sqrt(x.dot(x))
    l_y = np.sqrt(y.dot(y))
    if l_x == 0 or l_y == 0: return 0
    dian = x.dot(y)
    cos_ = dian / (l_x * l_y)
    cos_ = np.clip(cos_, -1.0, 1.0)
    angle_hu = np.arccos(cos_)
    angle_d = angle_hu * 180 / np.pi
    return angle_d

def cw_rotate(point, ang):
    x, y, _ = point
    ang_rad = math.radians(ang)
    new_x = round(x * math.cos(ang_rad) - y * math.sin(ang_rad), 5)
    new_y = round(x * math.sin(ang_rad) + y * math.cos(ang_rad), 5)
    return new_x, new_y

def get_fixed_camera_angle():
    camera_view = [0, -1, 0]
    reference_view = [0, 1, 0]
    angle = get_theta(camera_view, reference_view)
    return angle

def find_unique_landmark(target_obj, distractors, all_objects):
    target_pos = target_obj['position']
    target_label = target_obj['label']
    best_landmark, best_score = None, -float('inf')
    
    for obj in all_objects:
        if obj['id'] == target_obj['id']: continue
        if obj['label'] in BASIC_IGNORE or obj['label'] == target_label: continue
        
        dist_target = calculate_distance(target_pos, obj['position'])
        if dist_target > 3.0: continue 
        
        min_dist_distractor = min([calculate_distance(d['position'], obj['position']) for d in distractors]) if distractors else float('inf')
        
        ratio = min_dist_distractor / (dist_target + 0.1)
        diff = min_dist_distractor - dist_target
        
        if ratio < 1.5 and diff < 1.5:
            continue 

        score = (min_dist_distractor - dist_target) * 2.0 + (1.0 / (dist_target + 0.5))
        if score > best_score:
            best_score = score
            best_landmark = obj
            
    if best_landmark:
        dist = calculate_distance(target_pos, best_landmark['position'])
        is_crowded = any(abs(calculate_distance(d['position'], best_landmark['position']) - dist) < 0.5 for d in distractors)
        hint = f"{best_landmark['label']} (dist: {dist:.1f}m)"
        if is_crowded: hint += " [WARN: Crowded!]"
        return hint
    return "None"

# ================= 数据管理类 (保持不变) =================
class DynamicSceneDataManager:
    def __init__(self):
        self.current_scene_id = None
        self.obj_data = None
        self.rel_data = None
        self.id_to_label = {}
        self.id_to_size = {}
        self.id_to_pos = {}
        self.camera_data = {}
        self.current_ann_id = None
    def load_scene(self, scene_id):
        if self.current_scene_id == scene_id: return True
        
        obj_path = os.path.join(SCENE_GRAPH_ROOT, scene_id, "objects.json")
        rel_path = os.path.join(SCENE_GRAPH_ROOT, scene_id, "relationships.json")
        cam_path = os.path.join(os.environ.get("AMBIREFER3D_REAL_CAMERA_ROOT", "data/real/annotated_cameras"), f"{scene_id}.anns.json")

        if os.path.exists(cam_path):
            try:
                with open(cam_path, 'r') as f:
                    raw_cams = json.load(f)
                    self.camera_data = {
                        f"{item['object_id']}_{item['ann_id']}": item['camera'] 
                        for item in raw_cams
                    }
            except Exception as e:
                logging.error(f"Error loading camera data: {e}")
                self.camera_data = {}
        else:
            self.camera_data = {}
            
        if not os.path.exists(obj_path) or not os.path.exists(rel_path): 
            return False
            
        try:
            with open(obj_path, 'r', encoding='utf-8') as f: raw_obj = json.load(f)
            with open(rel_path, 'r', encoding='utf-8') as f: self.rel_data = json.load(f)
            
            self.obj_data = raw_obj[scene_id] if scene_id in raw_obj else raw_obj
            self.id_to_label = {o["id"]: o["label"] for o in self.obj_data.get("objects_info", [])}
            self.id_to_size = {o["id"]: o.get("size", [0,0,0]) for o in self.obj_data.get("objects_info", [])}
            self.id_to_pos = {o["id"]: o.get("position", [0,0,0]) for o in self.obj_data.get("objects_info", [])}
            
            self.current_scene_id = scene_id
            return True
        except Exception as e:
            logging.error(f"Error loading scene {scene_id}: {e}")
            return False

    def get_scene_objects(self): return self.obj_data.get("objects_info", [])
    def get_label(self, obj_id): return self.id_to_label.get(obj_id, "object")
    def get_pos(self, obj_id): return self.id_to_pos.get(obj_id, [0,0,0])
    def get_camera_info(self, object_id, ann_id):
        key = f"{object_id}_{ann_id}"
        return self.camera_data.get(key, None)
    def get_relationships_filtered(self, object_id):
        KEEP_PREDICATES = {
            "support", "resting on", "placed on", "on", "supported by", "on the top of",
            "inside", "placed within the area of", "placed in", "embedded into",
            "hanging on", "hung on", "affixed to", "attached to", "mounted on", 
            "between", "aligned"
        }

        PREDICATE_MAPPING = {
            "placed within the area of": "inside",
            "placed in": "inside",
            "hung on": "hanging on",
            "affixed to": "attached to",
            "on the top of": "on top of"
        }

        raw_rels = self.rel_data.get(self.current_scene_id, {}).get("relationships", [])
        if not raw_rels and "relationships" in self.rel_data: raw_rels = self.rel_data["relationships"]
        
        rels = []
        my_pos = self.get_pos(object_id)

        for r in raw_rels:
            sub, obj, pred = r
            if sub != object_id and obj != object_id: continue
            
            other_id = obj if sub == object_id else sub
            other_label = self.get_label(other_id)

            if other_label in BASIC_IGNORE: continue
            if pred not in KEEP_PREDICATES and pred not in PREDICATE_MAPPING: continue
            if calculate_distance(my_pos, self.get_pos(other_id)) > RELATION_CUTOFF_DIST: continue

            pred_readable = PREDICATE_MAPPING.get(pred, pred)
            if sub == object_id: 
                rels.append(f"is {pred_readable} the {other_label}")
            elif obj == object_id:
                if pred == "support": rels.append(f"is supported by the {other_label}")
                else: rels.append(f"has a {other_label} {pred_readable} it")

        unique = list(set(rels))
        return "; ".join(unique[:5]) if unique else "None"

    def get_geometric_context(self, center_obj_id, ann_id, radius=2.5):
        center_pos = self.get_pos(center_obj_id)
        cam_info = self.get_camera_info(center_obj_id, ann_id)
        
        cam_basis = None
        if cam_info:
            cam_basis = get_camera_basis(cam_info['position'], cam_info['lookat'])
        
        nearby_objects = []
        LOCAL_IGNORE = BASIC_IGNORE.union({"floor", "wall", "ceiling"})
        
        for obj in self.get_scene_objects():
            if obj['id'] == center_obj_id: continue
            if obj['label'] in LOCAL_IGNORE: continue
            
            other_pos = obj['position']
            dist_3d = calculate_distance(center_pos, other_pos)
            if dist_3d > radius: continue

            raw_dz = other_pos[2] - center_pos[2]
            
            if cam_basis:
                aligned_vec = project_vector_to_camera(center_pos, other_pos, cam_basis)
                vec_str = f"[{aligned_vec[0]:.2f}, {aligned_vec[1]:.2f}, {aligned_vec[2]:.2f}]"
            else:
                dx, dy = other_pos[0]-center_pos[0], other_pos[1]-center_pos[1]
                vec_str = f"[{dx:.2f}, {dy:.2f}, {raw_dz:.2f}]"

            desc = (f"{obj['label']} ("
                    f"dist={dist_3d:.2f}m, "
                    f"aligned_vector={vec_str}, "
                    f"global_dz={raw_dz:.2f}m"
                    f")")
            
            nearby_objects.append(desc)
            
        nearby_objects.sort(key=lambda x: float(re.search(r"dist=([\d\.]+)", x).group(1)))
        return "; ".join(nearby_objects[:10])
    
# ================= 核心生成逻辑 (保持不变) =================

def generate_clarification(target, distractors, all_objects, data_mgr, scene_id, ambiguous_query, removed_info):
    temperatures = [0.01, 0.2, 0.5]
    for attempt in range(MAX_RETRIES):
        temp = temperatures[attempt]
        try:
            return _core_logic(target, distractors, all_objects, data_mgr, scene_id, ambiguous_query, removed_info, temp)
        except json.JSONDecodeError: pass
        # 增加一点重试保护
        except APIStatusError as e:
             if e.status_code in [500, 503]:
                 time.sleep((attempt+1)*5) # 简单避让
        except Exception as e:
            if "not a multimodal model" in str(e): return None
            logging.error(f"Gen Error (Attempt {attempt}): {e}")
    return None

def _core_logic(target, distractors, all_objects, data_mgr, scene_id, ambiguous_query, removed_info, temperature):
    # 1. 准备候选物并打乱顺序
    distractors.sort(key=lambda d: calculate_distance(target['position'], d['position']))
    candidates = [target] + distractors[:4]
    random.shuffle(candidates)
    
    active_distractor_ids = [c['id'] for c in candidates if c['id'] != target['id']]
    target_landmark_hint = find_unique_landmark(target, distractors, all_objects)
    missing_info_hint = format_missing_hint(removed_info)
    
    text_context_str_list = []
    input_content_list = []
    
    MAX_TOTAL_IMGS = 15
    current_img_count = 0
    ground_truth_index = -1

    for idx, obj in enumerate(candidates):
        is_target = (obj['id'] == target['id'])
        if is_target:
            ground_truth_index = idx
        
        candidate_label = f"Candidate_{idx}"
        
        raw_img_list = get_multiview_images_by_pattern(scene_id, obj['id'])
        img_list = raw_img_list
        remaining_quota = MAX_TOTAL_IMGS - current_img_count
        if remaining_quota > 0:
             img_list = img_list[:remaining_quota]
        else:
             img_list = []
        this_obj_img_count = len(img_list)
        current_img_count += this_obj_img_count

        if img_list:
            for img in img_list:
                base64_img = encode_image_to_base64(img)
                input_content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
        
        valid_count = idx + 1
        coords = fmt_list(obj['position'])
        sem_rels = data_mgr.get_relationships_filtered(obj['id']) 
        geo_rels = data_mgr.get_geometric_context(obj['id'], data_mgr.current_ann_id)      
        raw_size = data_mgr.id_to_size.get(obj['id'], [0,0,0])
        height = raw_size[2]
        horizontal_dims = sorted([raw_size[0], raw_size[1]])
        depth_thickness = horizontal_dims[0] # 短边 (厚度)
        width_span = horizontal_dims[1] # 长边 (宽度)
        volume = height * width_span * depth_thickness
        frontal_area = height * width_span
        dim_str = (
            f"H:{height:.2f}m, "
            f"MaxW:{width_span:.2f}m, "
            f"Thick:{depth_thickness:.2f}m, "
            f"Vol:{volume:.2f}m³"
        )

        desc_lines = []
        desc_lines.append(f"[{candidate_label}]") 
        desc_lines.append(f"- Class: {obj['label']}")
        
        visual_note = f"{this_obj_img_count} images provided." if this_obj_img_count > 0 else "No images available (Quota limit reached)."
        desc_lines.append(f"- Visual Evidence: {visual_note}")
        desc_lines.append(f"- Dimensions: {dim_str}")
        desc_lines.append(f"- Coordinates: {coords}")
        desc_lines.append(f"- Semantic Relations (Graph Reference): {sem_rels}")
        desc_lines.append(f"- Geometric Scan (<2.5m, Raw Data): {geo_rels}")
        
        desc_lines.append(f"-----------------------------------")    
        text_context_str_list.append("\n".join(desc_lines))

    text_context_str = "\n".join(text_context_str_list)

    if not input_content_list: return None

    # --- System Prompt (FULL RESTORED + BLIND LOGIC) ---
    system_prompt = """
    You are an expert AI assistant with advanced spatial awareness and visual reasoning capabilities.
    You possess a "Dual-Perspective" understanding of the scene:
    
    1. **Global Physical Awareness (The "God View"):** - You know absolute physical facts (e.g., gravity, semantic relations like "inside").
       - Used for: "Above", "Below", "Inside", "Supported by".
       
    2. **Egocentric Visual Awareness (The "User View"):**
       - You know exactly how objects appear in the user's camera frame.
       - Used for: "Left", "Right", "In Front", "Behind".

    ### INPUT DATA EXPLAINED
    You will receive:
    1. **User Query**: The initial description.
    2. **Candidates**: A list of objects (Anonymized as Candidate_0, Candidate_1...).
    3. **Images**: For visual attribute analysis (Color, Material, Shape).
    4. **Spatial Metadata (Crucial)**:
        - **Semantic Relations**: High-confidence relationships annotated by humans (e.g., "inside", "supported by"). Use these as primary references.
        - **Geometric Scan**: A raw list of neighbors.
            - Format: `Label (dist=Xm, aligned_vector=[x, y, z], global_dz=Zm)`.
            - **CRITICAL: DATA SOURCE MAPPING**
                * **1. "LEFT / RIGHT / FRONT / BACK" (Visual/Camera Relations)**
                    - Look at `aligned_vector` [x, y, z].
                    - **LEFT / RIGHT** (Screen Horizontal):
                        - `vector X > 0`: Neighbor is to the **RIGHT**.
                        - `vector X < 0`: Neighbor is to the **LEFT**.
                    - **FRONT / BEHIND** (Depth from Camera):
                        - `vector Z < 0`: Neighbor is **IN FRONT** (Closer to you/camera). 
                        - `vector Z > 0`: Neighbor is **BEHIND** (Further away from you).   
                * **2. "ABOVE / BELOW" (Physical Gravity Relations)**
                    - Look **ONLY** at `global_dz`. Do NOT use the aligned_vector for this.
                    - `global_dz > 0`: Neighbor is physically **ABOVE** target (Higher Z). 
                    - `global_dz < 0`: Neighbor is physically **BELOW** target (Lower Z). 
                    - **Constraint (The "Directly" Rule)**: 
                        - To call it "Directly Above/Below", the horizontal gap must be small.
                        - **Rule**: If `abs(global_dz)` is almost equal to the total `dist`, it is aligned. 
                        - *Example*: dist=2.0m, global_dz=1.9m -> Aligned (Good).
                        - *Example*: dist=2.0m, global_dz=0.1m -> NOT Aligned (It is to the side).
                * **3. "HIGHER / LOWER" (Loose Comparison)**
                    - Use these words if `global_dz` is large but the object is far away horizontally.            
            - **INSTRUCTION**: Use this to differentiate identical objects. (e.g., "Candidate_0 has a chair to its RIGHT (vector x=+1.2), while Candidate_1 does not.")
        - **Smart Dimensions**: 
            - `H`: Height (Vertical).
            - `MaxW`: The widest horizontal span (regardless of rotation).
            - `Vol`: Total Volume. **Use `Vol` or `H*MaxW` to judge "Large/Big".**
    5. **Landmark Hint**: HIDDEN GROUND TRUTH. This is a cheat sheet for YOU to locate the object. The user DOES NOT know this hint yet.

    ### IMPORTANT: REASONING RULES
    1. **Visual Pragmatism (Relative Comparison)**: 
        - Users describe objects loosely. If a user says "white", and Candidate_0 is "beige" but others are "brown", **Candidate_0 MATCHES "WHITE"**.
        - Do not be pedantic about exact color codes. Look for the *best fit* among candidates.
    2. **No Redundant Questions**: 
        - If the user says "white cabinet", DO NOT ask "Is it the white one?".
    3. **Synthesize Multiple Views**: Don’t judge based on a single image.
    4. **LIST & ASK STRATEGY**: If visual differences are obvious, list them first. (e.g., "I see a red one and a blue one. Do you mean the red one?").

    ### REASONING PIPELINE (Strict Step-by-Step)
    
    **STEP 1: BLIND GROUNDING (The "Turing Test")**
    - **Deconstruct Query**: Break down the description (e.g., "white", "cabinet", "large", "right of the table", "first one").
    - **Identify Matches**: Look at ALL Candidates. List *every* candidate that completely and strictly matches the description based on Visuals and Spatial context.
    - Output this list as `matched_candidate_indices`.

    **STEP 2: VERIFICATION & AMBIGUITY CHECK**
    - Reveal the **Hidden Ground Truth** provided at the end of the prompt.
    - Check: Is the Ground Truth in your matched list?
        - If NO: The query is poor/hallucinated.
    - Check: Are there OTHER candidates in your matched list?
        - If YES: **Ambiguity Exists**. Go to Step 3.
        - If NO (Only Ground Truth matches): **No Ambiguity**. Return `question: null`.

    **STEP 3: FEATURE EXTRACTION & GENERATION (If Ambiguous)**
    - Compare the Ground Truth against the other matched candidates found in Step 1.
    - Find the *most distinguishing feature* using this strict hierarchy:
        * **PRIORITY 1: VISUAL ATTRIBUTES** (Color, Shape, Material).
        * **PRIORITY 2: DIMENSIONS** (Is it significantly Taller/Wider/Smaller/Larger?).
        * **PRIORITY 3: PRECISE TOPOLOGY (Hybrid Reasoning)**
            - Check "Semantic Relations" first. Is it "inside" something?
            - Check "Geometric Scan". Use the **aligned_vectors** to find unique neighbors. 
            - Example: "The target has a trash can to its RIGHT (+X), while the distractor does not."
        * **PRIORITY 4: RELATIVE LANDMARKS** (Proximity to doors, beds, etc.).
        * **PRIORITY 5: EGOCENTRIC COORDINATES** (Far left, Middle, Corner).
    - Generate a clarification question based on this feature.

    ### STRICT CONSTRAINTS
    1. **NO METADATA LEAKAGE**: Do not output raw vector numbers in the question.
    2. **NATURAL LANGUAGE**: Use "it" or "the one".
    3. **NO REPETITION**: Do not ask about attributes already in the user query.
    4. `matched_candidate_indices` MUST include the Ground Truth Target index (unless the query is completely wrong).
    5. If `matched_candidate_indices` has length > 1, `question` MUST NOT be null.
    6. If `matched_candidate_indices` has length == 1, `question` SHOULD be null.

    ### FEW-SHOT EXAMPLES
    **Case 1 (Unique - Blind Grounding finds only one):**
    - Input: "the white cabinet"
    - Scene: Candidate_0 is white. Candidate_1 is brown. Ground Truth is Candidate_0.
    - Output:
    {
        "reasoning": "Step 1: 'white' matches Candidate_0. Candidate_1 is brown. Only one match. Step 2: Match is GT. No ambiguity.",
        "matched_candidate_indices": [0],
        "question": null,
        "restored_query": null
    }
    **Case 2 (Ambiguous - Blind Grounding finds two):**
    - Input: "the pillow"
    - Scene: Candidate_0 is on bed. Candidate_1 is on floor. Both are pillows. GT is Candidate_0.
    - Output:
    {
        "reasoning": "Step 1: Both are pillows. Matches: [0, 1]. Step 2: Multiple matches. Ambiguity exists. Step 3: Priority 3 (Topology) used. Candidate_0 has 'Semantic: on bed'. Candidate_1 has 'Scan: near floor'.",
        "matched_candidate_indices": [0, 1],
        "question": "Is it the one on the bed?",
        "restored_query": "the pillow on the bed"
    }
    **Case 3 (Visual Ambiguity & List Strategy):**
    - Input: "the chair"
    - Scene: Candidate_0 is Red. Candidate_1 is Brown. GT is Candidate_0.
    - Output:
    {
        "reasoning": "Step 1: Both are chairs. Matches: [0, 1]. Step 2: Multiple matches. Ambiguity. Step 3: Priority 1 (Visual) used. Colors differ. Using List & Ask strategy.",
        "matched_candidate_indices": [0, 1],
        "question": "I see a red chair and a brown one. Do you mean the red one?",
        "restored_query": "the red chair"
    }
    """
    
    user_text_prompt = f"""
    ### CURRENT TASK
    **User Query**: "{ambiguous_query}"
    **Likely Missing Info**: {missing_info_hint}
    **Landmark Hint**: {target_landmark_hint}

    ### CANDIDATE OBJECTS (Anonymized)
    (Analyze images and spatial data below)
    {text_context_str}

    ### HIDDEN GROUND TRUTH (For Phase 2 Verification)
    The actual target intended by the user is: **Candidate_{ground_truth_index}**
    (Do not let this knowledge bias your Phase 1 blind grounding!)

    ### INSTRUCTION
    Generate the JSON response following the REASONING PIPELINE.
    """
    
    input_content_list.append({"type": "text", "text": user_text_prompt})
    
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_content_list} 
        ],
        temperature=temperature, 
        max_tokens=1024 
    )
    
    response_json = json.loads(clean_json_output(completion.choices[0].message.content))
    
    matched_indices = response_json.get("matched_candidate_indices", [])
    matched_real_ids = []
    
    for idx_val in matched_indices:
        if isinstance(idx_val, int) and 0 <= idx_val < len(candidates):
            matched_real_ids.append(candidates[idx_val]['id'])
            
    response_json["matched_real_ids"] = matched_real_ids
    
    return response_json, active_distractor_ids

# ================= 线程处理函数 (无改动) =================
def process_single_item(item, processed_keys):
    """
    处理单个 query item 的工作函数。
    由于 DynamicSceneDataManager 是有状态的（存储了current_scene_id），
    在并发环境中，必须为每个线程/任务实例化一个独立的 Manager。
    """
    scene_id = item.get('scene_id')
    target_id = None
    ann_id = item.get('ann_id', '0')
    
    try: target_id = int(item.get('object_id'))
    except: return None # Invalid ID
    
    object_key = f"{scene_id}_{target_id}_{ann_id}"
    
    # 二次检查（防止重复提交，虽然main里已经检查了）
    if object_key in processed_keys: 
        return None

    # [IMPORTANT] 在线程内实例化 Manager，确保线程安全
    local_mgr = DynamicSceneDataManager()
    
    if not local_mgr.load_scene(scene_id):
        return None
    local_mgr.current_ann_id = ann_id
    original_gt = item.get("original_description", "N/A")
    variants = item.get("generated_ambiguity", {}).get("ambiguous_variants", [])
    all_objects = local_mgr.get_scene_objects()
    
    label_index = defaultdict(list)
    for obj in all_objects:
        label_index[obj['label']].append(obj)
        
    target = next((o for o in all_objects if o['id'] == target_id), None)
    if not target: return None
    
    distractors = [o for o in label_index[target['label']] if o['id'] != target_id]
    if not distractors: return None
    
    object_entry = {
        "scene_id": scene_id,
        "target_id": target_id,
        "ann_id": ann_id,
        "gt_description": original_gt,
        "distractors_count": len(distractors),
        "dialogues": []
    }
    
    has_valid_dialogue = False
    
    for variant in variants:
        input_query = variant["query"]
        removed_info = variant.get("removed_info", [])
        variant_type = variant.get("type", "unknown")
        
        # 调用核心生成函数
        try:
            response = generate_clarification(target, distractors, all_objects, local_mgr, scene_id, input_query, removed_info)
        except Exception as e:
            logging.error(f"Generate error: {e}")
            response = None

        if response:
            result, dist_ids = response
            q = result.get('question')
            r = result.get('restored_query')
            matched_ids = result.get("matched_real_ids", [])

            is_ambiguous = bool(q and str(q).lower() not in ["none", "null", "n/a", ""])
            
            q = None if not is_ambiguous else q
            if not r: r = input_query

            object_entry["dialogues"].append({
                "ambiguity_type": variant_type,
                "input_query": input_query,
                "removed_info": removed_info,
                "active_distractor_ids": dist_ids,
                "is_ambiguous": is_ambiguous,
                "generated_question": q,
                "restored_query": r,
                "reasoning": result.get('reasoning'),
                "grounding_matches": matched_ids 
            })
            has_valid_dialogue = True
    
    if has_valid_dialogue:
        # ⚠️ 注意：这里的 OUTPUT_JSON_FILE 会被 main 函数里的分片逻辑动态修改
        append_to_jsonl_thread_safe(object_entry, OUTPUT_JSON_FILE)
        return object_key
    
    return None

# ================= 主程序 (改造后) =================
def main():
    # [NEW] 1. 引入参数解析
    parser = argparse.ArgumentParser(description="Multi-process Sharding Runner")
    parser.add_argument("--split", type=int, default=0, help="当前分片的索引 (从0开始)")
    parser.add_argument("--num_splits", type=int, default=1, help="总分片数")
    args = parser.parse_args()

    # [NEW] 2. 动态修改全局配置，确保每个分片写不同的文件
    # 修改全局变量 OUTPUT_JSON_FILE 和 LOG_FILE
    global OUTPUT_JSON_FILE, LOG_FILE
    
    if args.num_splits > 1:
        # 如果启用分片，文件名加上 _partX 后缀
        # 例如: results_v2.jsonl -> results_v2_part0.jsonl
        if "." in OUTPUT_JSON_FILE:
            base, ext = os.path.splitext(OUTPUT_JSON_FILE)
            OUTPUT_JSON_FILE = f"{base}_part{args.split}{ext}"
        else:
            OUTPUT_JSON_FILE = f"{OUTPUT_JSON_FILE}_part{args.split}"
            
        LOG_FILE = f"{LOG_FILE.replace('.txt', '')}_part{args.split}.txt"

    # [NEW] 3. 重新配置日志 (为了让新文件名生效)
    # 先移除所有旧的 handler
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    
    output_dir = os.path.dirname(OUTPUT_JSON_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f: queries = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load input file: {e}")
        return

    # [NEW] 4. 数据切片 (Sharding Logic)
    # 根据 split 和 num_splits，只取属于当前进程的那一部分数据
    total_items = len(queries)
    chunk_size = math.ceil(total_items / args.num_splits)
    start_idx = args.split * chunk_size
    end_idx = min((args.split + 1) * chunk_size, total_items)
    
    # 切片
    my_queries = queries[start_idx:end_idx]
    
    print(f"🔹 [Split {args.split}/{args.num_splits}] Processing items {start_idx} to {end_idx} (Count: {len(my_queries)})")
    print(f"🔹 Output File: {OUTPUT_JSON_FILE}")
    print(f"🔹 Log File: {LOG_FILE}")

    processed_keys = set()
    if os.path.exists(OUTPUT_JSON_FILE):
        print(f"Reading existing progress from {OUTPUT_JSON_FILE}...")
        try:
            with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        item = json.loads(line)
                        key = f"{item['scene_id']}_{item['target_id']}_{item.get('ann_id', '0')}"
                        processed_keys.add(key)
                    except: pass
        except Exception as e:
            print(f"⚠️ Warning reading output file: {e}")

    print(f"Loaded {len(my_queries)} items for this split. Skipping {len(processed_keys)} objects already done.")
    
    # 筛选出需要处理的任务
    tasks = []
    for item in my_queries:
        scene_id = item.get('scene_id')
        target_id = item.get('object_id')
        ann_id = item.get('ann_id', '0')
        key = f"{scene_id}_{target_id}_{ann_id}"
        if key not in processed_keys:
            tasks.append(item)
    
    print(f"Starting ThreadPoolExecutor with {MAX_WORKERS} workers for {len(tasks)} tasks...")

    success_count = 0
    
    # 使用线程池并发处理 (这里的 MAX_WORKERS 是当前进程内的线程数)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_single_item, item, processed_keys): item for item in tasks}
        
        with tqdm(total=len(tasks), desc=f"Split {args.split}") as pbar:
            for future in as_completed(future_to_item):
                try:
                    result_key = future.result()
                    if result_key:
                        success_count += 1
                except Exception as exc:
                    logging.error(f"Task generated an exception: {exc}")
                finally:
                    pbar.update(1)

    print(f"\nAll Done! Split {args.split} Processed: {success_count}")

if __name__ == "__main__":
    main()