# import json
# import os
# import argparse
# from pathlib import Path
# from collections import defaultdict
# import hashlib
# import random
# from tqdm import tqdm

# # 二次 merge 校验时需要读取 .pth（可选）
# try:
#     import torch
# except ImportError:
#     torch = None


# # ================= 基础工具函数 =================

# def get_ann_id(scene_id, object_id: int) -> int:
#     """
#     生成稳定的 annotation id。
#     为了让不同退化类型在同一 (scene, object_id) 下也能有不同 ann_id，
#     我们在调用时会把 scene_id 拼上 deg_type 再传进来。
#     """
#     raw = f"{scene_id}_{int(object_id)}"
#     return int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


# def load_scene_graphs(output_root):
#     """
#     预扫描 scene_graph.json，统计每个场景中各类别物体数量（原先用于 unique/multiple，现在可选）。
#     counts[scene_id][category_name] = count
#     """
#     counts = defaultdict(lambda: defaultdict(int))
#     scene_dirs = [d for d in Path(output_root).iterdir() if d.is_dir()]

#     print("正在扫描 Scene Graphs 以构建类别直方图...")
#     for s_dir in tqdm(scene_dirs):
#         scene_id = s_dir.name
#         sg_path = s_dir / f"{scene_id}_scene_graph.json"
#         if not sg_path.exists():
#             continue

#         try:
#             with open(sg_path, "r", encoding="utf-8") as f:
#                 sg = json.load(f)

#             for obj in sg.get("objects", []):
#                 raw_cat = obj.get("category") or obj.get("instance_name")
#                 if raw_cat is None:
#                     continue
#                 counts[scene_id][raw_cat] += 1

#         except Exception as e:
#             print(f"[Warning] Error reading SG for {scene_id}: {e}")

#     return counts


# def read_split_txts(split_txt_dir: str):
#     """读取 train.txt/val.txt/test.txt，返回 scene_id -> split_name 的映射"""
#     mapping = {}
#     for split_name in ("train", "val", "test"):
#         p = os.path.join(split_txt_dir, f"{split_name}.txt")
#         if not os.path.exists(p):
#             raise FileNotFoundError(f"Split txt not found: {p}")
#         with open(p, "r", encoding="utf-8") as f:
#             for line in f:
#                 sid = line.strip()
#                 if sid:
#                     mapping[sid] = split_name
#     return mapping


# def ensure_dir(p):
#     os.makedirs(p, exist_ok=True)


# # ================= pth 缓存与校验（可选） =================

# class PthCache:
#     """避免重复读取同一个 scene 的 .pth，提高二次 merge 校验速度"""
#     def __init__(self, scene_dataset_path: str):
#         if torch is None:
#             raise RuntimeError("torch 未安装，无法开启 pth 校验模式。")
#         self.scene_dataset_path = scene_dataset_path
#         self.cache = {}  # (split_dir, scene_id) -> dict

#     def load(self, split_dir: str, scene_id: str):
#         key = (split_dir, scene_id)
#         if key in self.cache:
#             return self.cache[key]
#         pth_path = os.path.join(self.scene_dataset_path, split_dir, f"{scene_id}.pth")
#         if not os.path.exists(pth_path):
#             self.cache[key] = None
#             return None
#         try:
#             d = torch.load(pth_path, map_location="cpu")
#         except Exception:
#             d = None
#         self.cache[key] = d
#         return d


# def validate_and_fix_object_id(scene_pth: dict, object_id: int, min_inst_points: int = 1):
#     """
#     使用 scene_pth 中的 aabb_obj_ids / instance_ids 校验 object_id。
#     返回 (fixed_object_id or None, reason)
#     """
#     if scene_pth is None:
#         return None, "pth_missing_or_unreadable"

#     aabb_obj_ids = scene_pth.get("aabb_obj_ids", None)
#     instance_ids = scene_pth.get("instance_ids", None)

#     if aabb_obj_ids is None or instance_ids is None:
#         return None, "pth_missing_keys"

#     # aabb_obj_ids 可能是 numpy/torch/list，这里统一转为 python int set
#     try:
#         valid_ids = set(int(x) for x in list(aabb_obj_ids))
#     except Exception:
#         return None, "aabb_obj_ids_unparseable"

#     # instance_ids 用于点数检查
#     try:
#         inst_arr = instance_ids
#         # torch tensor -> numpy
#         if hasattr(inst_arr, "detach"):
#             inst_arr = inst_arr.detach().cpu().numpy()
#     except Exception:
#         inst_arr = None

#     candidates = [int(object_id), int(object_id) - 1, int(object_id) + 1]
#     for cand in candidates:
#         if cand <= 0:
#             continue
#         if cand not in valid_ids:
#             continue
#         if inst_arr is not None and min_inst_points > 1:
#             cnt = int((inst_arr == cand).sum())
#             if cnt < min_inst_points:
#                 continue
#         return cand, "ok"

#     return None, "not_in_aabb_or_too_few_points"


# # ================= 属性标记规则（spatial / color / texture / shape） =================

# def infer_attribute_flags(parsed_gt, atoms, kept_atoms):
#     """
#     根据 parsed_gt / atoms / kept_atoms 自动推断
#     spatial / color / texture / shape 四个布尔标记。

#     设计原则：
#     1）优先使用当前退化版本的 kept_atoms（更贴近 degraded_query 实际保留的语义）；
#     2）若 kept_atoms 为空，则回退到 atoms / parsed_gt 中的 atom 字符串；
#     3）通过简单的前缀或关键词匹配来判断属性类型：
#        - spatial:  "sprel:" 前缀，或 key 中带 "spatial"
#        - color:    "color:" / "colour:" / key 中带 "color"
#        - texture:  "texture:" / "material:" / "fabric:"
#        - shape:    "shape:" / "geom:" / key 中带 "shape"
#     """

#     spatial = False
#     color = False
#     texture = False
#     shape = False

#     atom_strings = []

#     # 1) 退化版本中显式给出的 kept_atoms
#     if kept_atoms:
#         for a in kept_atoms:
#             if isinstance(a, str):
#                 atom_strings.append(a)

#     # 2) 若 kept_atoms 不够，用 atoms（通常是 dict）做补充
#     #    atoms 的结构不固定，这里做一个尽量宽松的收集
#     if atoms and not atom_strings:
#         if isinstance(atoms, dict):
#             for v in atoms.values():
#                 if isinstance(v, list):
#                     for a in v:
#                         if isinstance(a, str):
#                             atom_strings.append(a)
#                 elif isinstance(v, str):
#                     atom_strings.append(v)
#         elif isinstance(atoms, list):
#             for a in atoms:
#                 if isinstance(a, str):
#                     atom_strings.append(a)

#     # 3) 再从 parsed_gt 中挖一些字符串（保险起见）
#     if parsed_gt and not atom_strings:
#         if isinstance(parsed_gt, dict):
#             for v in parsed_gt.values():
#                 if isinstance(v, list):
#                     for a in v:
#                         if isinstance(a, str):
#                             atom_strings.append(a)
#                 elif isinstance(v, str):
#                     atom_strings.append(v)

#     # 统一转小写，便于匹配
#     atom_strings_l = [a.lower() for a in atom_strings]

#     for s in atom_strings_l:
#         # 空间关系：sprel 前缀，或者 key 中包含 spatial
#         if s.startswith("sprel:") or "spatial" in s:
#             spatial = True

#         # 颜色：color / colour
#         if s.startswith("color:") or s.startswith("colour:") or "color" in s or "colour" in s:
#             color = True

#         # 纹理 / 材质：texture / material / fabric
#         if s.startswith("texture:") or "texture" in s or s.startswith("material:") or "material" in s \
#                 or "fabric" in s:
#             texture = True

#         # 形状：shape / geom
#         if s.startswith("shape:") or "shape" in s or s.startswith("geom:") or "geometric" in s:
#             shape = True

#     return spatial, color, texture, shape


# # ================= 主入口 =================

# def main():
#     parser = argparse.ArgumentParser()

#     # 输入：BlenderProc 输出目录（包含 scene_graph.json 与 stage3/*.json）
#     parser.add_argument(
#         "--output_root",
#         required=True,
#         help="main.py 的输出目录 (包含 scene_graph 和 stage3)"
#     )

#     # 输出：最终生成的 json（会自动加 _train_FRONT/_val_FRONT/_test_FRONT 后缀）
#     parser.add_argument(
#         "--save_path",
#         required=True,
#         help="最终生成的 json 路径前缀，例如 /path/to/scanrefer_front"
#     )

#     # 输出：划分结果保存目录（会写 train.txt/val.txt/test.txt）
#     parser.add_argument(
#         "--divide_path",
#         required=True,
#         help="划分结果保存目录（写 train.txt/val.txt/test.txt）"
#     )

#     parser.add_argument(
#         "--use_ambiguous",
#         action="store_true",
#         help="是否包含 ambiguous 样本（当前脚本未特殊处理，仅保留接口）"
#     )

#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="随机种子（第一次 merge 随机划分用）"
#     )

#     # 二次 merge 校验相关（可选）
#     parser.add_argument(
#         "--scene_dataset_path",
#         default=None,
#         help="preprocess 生成的 .pth 根目录（开启 object_id<->aabb_obj_ids 校验）"
#     )

#     parser.add_argument(
#         "--pth_split_suffix",
#         default="_FRONT",
#         help="pth 的 split 目录后缀，默认 train_FRONT/val_FRONT/test_FRONT"
#     )

#     parser.add_argument(
#         "--split_txt_dir",
#         default=None,
#         help="若提供，则按已有 train.txt/val.txt/test.txt 分配样本，不再随机划分（用于二次 merge）"
#     )

#     parser.add_argument(
#         "--min_inst_points",
#         type=int,
#         default=1,
#         help="校验时要求该 instance 至少包含的点数阈值（建议先用 1 跑通，再考虑 20/50）"
#     )

#     args = parser.parse_args()

#     random.seed(args.seed)

#     # 1) 扫 scene_graph，得到每场景的类别计数（这里目前只为兼容保留，eval_type 不再用它）
#     scene_cat_counts = load_scene_graphs(args.output_root)
#     _ = scene_cat_counts  # 当前脚本中未使用，仅保留以兼容后续可能的扩展

#     # 2) 读取 split（若提供 split_txt_dir，则固定 split；否则后面随机 split）
#     scene_to_split = None
#     if args.split_txt_dir is not None:
#         scene_to_split = read_split_txts(args.split_txt_dir)
#         print(f"使用已有 split_txt_dir={args.split_txt_dir}，将按 train/val/test.txt 固定划分。")

#     # 3) 收集所有语言条目（先不划分）
#     all_entries = []
#     scene_dirs = sorted([d for d in Path(args.output_root).iterdir() if d.is_dir()])
#     print(f"开始合并 {len(scene_dirs)} 个场景的数据（并按三种退化类型拆分）...")

#     for scene_dir in tqdm(scene_dirs):
#         scene_id = scene_dir.name
#         stage3_dir = scene_dir / "stage3"
#         if not stage3_dir.exists():
#             continue

#         for json_file in stage3_dir.glob("*.json"):
#             try:
#                 with open(json_file, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#             except Exception:
#                 continue

#             # ----------------- 基础字段 -----------------
#             oid_raw = data.get("object_id", None)
#             if oid_raw is None:
#                 continue
#             try:
#                 base_object_id = int(oid_raw)
#             except Exception:
#                 continue

#             parsed_gt = data.get("parsed_gt", {}) or {}
#             # object_name：优先用 parsed_gt["class"]，否则用原始 object_name 去掉 .xxx 后缀
#             obj_class = parsed_gt.get("class", None)
#             if obj_class is not None:
#                 object_name = obj_class
#             else:
#                 raw_obj_name = data.get("object_name", None)
#                 if raw_obj_name is not None:
#                     object_name = raw_obj_name.split(".")[0]
#                 else:
#                     object_name = None

#             gt_block = data.get("gt", {}) or {}
#             gt_query = gt_block.get("gt_query", "").strip()

#             ambiguity_type_global = data.get("ambiguity_type", None)
#             atoms = data.get("atoms", None)

#             # ----------------- 三种退化类型拆分 -----------------
#             degradation = data.get("degradation", {}) or {}
#             for deg_type in ("no_spatial", "no_visual", "minimal"):
#                 deg = degradation.get(deg_type, None)
#                 if not deg:
#                     continue

#                 description = (deg.get("degraded_query", "") or "").strip()
#                 if not description:
#                     continue

#                 # candidate / object_ids：variant_candidate_ids → int list
#                 raw_cands = deg.get("variant_candidate_ids", []) or []
#                 candidate_ids = []
#                 for x in raw_cands:
#                     try:
#                         candidate_ids.append(int(x))
#                     except Exception:
#                         continue
#                 if not candidate_ids:
#                     continue

#                 # object_ids：Multi3DRefer 风格，多目标集合
#                 object_ids = candidate_ids

#                 # eval_type：按 Multi3DRefer 风格设置
#                 # - 多目标：mt
#                 # - 单目标：st_wo_d
#                 if len(object_ids) > 1:
#                     eval_type = "mt"
#                 else:
#                     eval_type = "st_wo_d"

#                 # ann_id：为了避免和其他退化类型冲突，把 deg_type 拼进 scene_id 参与 hash
#                 ann_id = get_ann_id(scene_id=f"{scene_id}_{deg_type}", object_id=base_object_id)

#                 # 歧义相关标签
#                 ambiguity_type = deg.get("degraded_type", ambiguity_type_global or deg_type)
#                 is_ambiguous = bool(deg.get("is_ambiguous", False))

#                 # 四个属性标记：基于 kept_atoms / atoms / parsed_gt
#                 kept_atoms = deg.get("kept_atoms", []) or []
#                 spatial, color, texture, shape = infer_attribute_flags(
#                     parsed_gt=parsed_gt,
#                     atoms=atoms,
#                     kept_atoms=kept_atoms
#                 )

#                 entry = {
#                     "scene_id": scene_id,
#                     "object_id": base_object_id,          # 单个主目标 id（ScanRefer 风格，保持兼容）
#                     "object_ids": object_ids,             # Multi3DRefer 风格，多目标集合
#                     "candidate_ids": object_ids,          # 显式保留候选集合
#                     "object_name": object_name,
#                     "ann_id": ann_id,
#                     "description": description,
#                     "token": description.split(),
#                     "eval_type": eval_type,               # "mt" / "st_wo_d"
#                     "ambiguity_type": ambiguity_type,     # no_spatial / no_visual / minimal
#                     "is_ambiguous": is_ambiguous,
#                     "gt_query": gt_query,                 # 原始无歧义描述

#                     # 新增四个布尔属性标记（对当前 degraded_query 而言）
#                     "spatial": spatial,
#                     "color": color,
#                     "texture": texture,
#                     "shape": shape,
#                 }

#                 all_entries.append(entry)

#     print(f"合并完成: 有效样本 {len(all_entries)} 条。")

#     # 4) 分配到 train/val/test
#     if scene_to_split is not None:
#         # 4.1 若提供 split_txt_dir：按 scene_id 固定划分（场景级）
#         train_data, val_data, test_data = [], [], []
#         dropped = 0
#         for e in all_entries:
#             sid = e["scene_id"]
#             split = scene_to_split.get(sid, None)
#             if split is None:
#                 dropped += 1
#                 continue
#             if split == "train":
#                 train_data.append(e)
#             elif split == "val":
#                 val_data.append(e)
#             else:
#                 test_data.append(e)
#         if dropped > 0:
#             print(f"[Warning] 有 {dropped} 条样本的 scene_id 不在 split_txt_dir 中，已丢弃。")
#     else:
#         # 4.2 第一次划分：按「场景级」随机划分，保证同一 scene 的所有 query 进同一个 split
#         all_scene_ids = sorted({e["scene_id"] for e in all_entries})
#         random.shuffle(all_scene_ids)
#         m = len(all_scene_ids)
#         train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

#         m_train = int(m * train_ratio)
#         m_val = int(m * val_ratio)

#         train_scene_ids = set(all_scene_ids[:m_train])
#         val_scene_ids = set(all_scene_ids[m_train:m_train + m_val])
#         test_scene_ids = set(all_scene_ids[m_train + m_val:])

#         train_data, val_data, test_data = [], [], []
#         for e in all_entries:
#             sid = e["scene_id"]
#             if sid in train_scene_ids:
#                 train_data.append(e)
#             elif sid in val_scene_ids:
#                 val_data.append(e)
#             elif sid in test_scene_ids:
#                 test_data.append(e)
#             else:
#                 # 理论上不会发生（除非前面构造 all_scene_ids 有遗漏）
#                 continue

#     # 5) 可选：开启 pth 校验与 object_id 纠偏（仍基于 base_object_id，不影响 object_ids）
#     if args.scene_dataset_path is not None:
#         print(f"开启 pth 校验模式：scene_dataset_path={args.scene_dataset_path}")
#         cache = PthCache(args.scene_dataset_path)

#         def split_dir_name(split_short: str):
#             # split_short: train/val/test  ->  train_FRONT/val_FRONT/test_FRONT
#             return f"{split_short}{args.pth_split_suffix}"

#         def run_validate(data_list, split_short: str):
#             out = []
#             bad = defaultdict(int)
#             sdir = split_dir_name(split_short)
#             for e in tqdm(data_list, desc=f"Validating {split_short}"):
#                 sid = e["scene_id"]
#                 pth = cache.load(sdir, sid)
#                 fixed, reason = validate_and_fix_object_id(
#                     pth, e["object_id"], min_inst_points=args.min_inst_points
#                 )
#                 if fixed is None:
#                     bad[reason] += 1
#                     continue
#                 e2 = dict(e)
#                 e2["object_id"] = int(fixed)
#                 # ann_id 与新的 object_id 再同步一次（此处不再区分 deg_type，只按 scene_id 原样来 hash）
#                 e2["ann_id"] = get_ann_id(sid, int(fixed))
#                 out.append(e2)
#             return out, bad

#         train_data, bad_tr = run_validate(train_data, "train")
#         val_data, bad_va = run_validate(val_data, "val")
#         test_data, bad_te = run_validate(test_data, "test")

#         print("[Validate Summary]")
#         print("train bad:", dict(bad_tr))
#         print("val   bad:", dict(bad_va))
#         print("test  bad:", dict(bad_te))
#         print(f"validate后样本数：train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

#     # 6) 写出 json（Multi3DRefer 风格，支持多目标）
#     base_path = os.path.splitext(args.save_path)[0]
#     train_path = base_path + "_train_FRONT.json"
#     val_path = base_path + "_val_FRONT.json"
#     test_path = base_path + "_test_FRONT.json"

#     with open(train_path, "w", encoding="utf-8") as f:
#         json.dump(train_data, f, indent=2, ensure_ascii=False)
#     with open(val_path, "w", encoding="utf-8") as f:
#         json.dump(val_data, f, indent=2, ensure_ascii=False)
#     with open(test_path, "w", encoding="utf-8") as f:
#         json.dump(test_data, f, indent=2, ensure_ascii=False)

#     print(f"已保存: {train_path} ({len(train_data)}) | {val_path} ({len(val_data)}) | {test_path} ({len(test_data)})")

#     # 7) 写出 scene list（场景级划分）
#     ensure_dir(args.divide_path)

#     def save_scene_ids(data, path):
#         scene_ids = sorted(set(e["scene_id"] for e in data))
#         with open(path, "w", encoding="utf-8") as f:
#             for sid in scene_ids:
#                 f.write(sid + "\n")

#     save_scene_ids(train_data, os.path.join(args.divide_path, "train.txt"))
#     save_scene_ids(val_data, os.path.join(args.divide_path, "val.txt"))
#     save_scene_ids(test_data, os.path.join(args.divide_path, "test.txt"))

#     print(f"并保存 scene_id 列表到: {args.divide_path}/train.txt | val.txt | test.txt")


# if __name__ == "__main__":
#     main()

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# 针对query做了加强后的又一次json文件生成
# 没有按三种歧义类型进行数据merge的版本
# import json
# import os
# import argparse
# from pathlib import Path
# from collections import defaultdict
# import hashlib
# import random
# from tqdm import tqdm

# # 二次 merge 校验时需要读取 .pth（可选）
# try:
#     import torch
# except ImportError:
#     torch = None


# # ================= 基础工具函数 =================

# def get_ann_id(scene_id, object_id: int) -> int:
#     """
#     生成稳定的 annotation id。
#     为了让不同退化类型在同一 (scene, object_id) 下也能有不同 ann_id，
#     我们在调用时会把 scene_id 拼上 deg_type 再传进来。
#     """
#     raw = f"{scene_id}_{int(object_id)}"
#     return int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


# def load_scene_graphs(output_root):
#     """
#     预扫描 scene_graph.json，统计每个场景中各类别物体数量（原先用于 unique/multiple，现在可选）。
#     counts[scene_id][category_name] = count
#     """
#     counts = defaultdict(lambda: defaultdict(int))
#     scene_dirs = [d for d in Path(output_root).iterdir() if d.is_dir()]

#     print("正在扫描 Scene Graphs 以构建类别直方图...")
#     for s_dir in tqdm(scene_dirs):
#         scene_id = s_dir.name
#         sg_path = s_dir / f"{scene_id}_scene_graph.json"
#         if not sg_path.exists():
#             continue

#         try:
#             with open(sg_path, "r", encoding="utf-8") as f:
#                 sg = json.load(f)

#             for obj in sg.get("objects", []):
#                 raw_cat = obj.get("category") or obj.get("instance_name")
#                 if raw_cat is None:
#                     continue
#                 counts[scene_id][raw_cat] += 1

#         except Exception as e:
#             print(f"[Warning] Error reading SG for {scene_id}: {e}")

#     return counts


# def read_split_txts(split_txt_dir: str):
#     """读取 train.txt/val.txt/test.txt，返回 scene_id -> split_name 的映射"""
#     mapping = {}
#     for split_name in ("train", "val", "test"):
#         p = os.path.join(split_txt_dir, f"{split_name}.txt")
#         if not os.path.exists(p):
#             raise FileNotFoundError(f"Split txt not found: {p}")
#         with open(p, "r", encoding="utf-8") as f:
#             for line in f:
#                 sid = line.strip()
#                 if sid:
#                     mapping[sid] = split_name
#     return mapping


# def ensure_dir(p):
#     os.makedirs(p, exist_ok=True)


# # ================= pth 缓存与校验（可选） =================

# class PthCache:
#     """避免重复读取同一个 scene 的 .pth，提高二次 merge 校验速度"""
#     def __init__(self, scene_dataset_path: str):
#         if torch is None:
#             raise RuntimeError("torch 未安装，无法开启 pth 校验模式。")
#         self.scene_dataset_path = scene_dataset_path
#         self.cache = {}  # (split_dir, scene_id) -> dict

#     def load(self, split_dir: str, scene_id: str):
#         key = (split_dir, scene_id)
#         if key in self.cache:
#             return self.cache[key]
#         pth_path = os.path.join(self.scene_dataset_path, split_dir, f"{scene_id}.pth")
#         if not os.path.exists(pth_path):
#             self.cache[key] = None
#             return None
#         try:
#             d = torch.load(pth_path, map_location="cpu")
#         except Exception:
#             d = None
#         self.cache[key] = d
#         return d


# def validate_and_fix_object_id(scene_pth: dict, object_id: int, min_inst_points: int = 1):
#     """
#     使用 scene_pth 中的 aabb_obj_ids / instance_ids 校验 object_id。
#     返回 (fixed_object_id or None, reason)
#     """
#     if scene_pth is None:
#         return None, "pth_missing_or_unreadable"

#     aabb_obj_ids = scene_pth.get("aabb_obj_ids", None)
#     instance_ids = scene_pth.get("instance_ids", None)

#     if aabb_obj_ids is None or instance_ids is None:
#         return None, "pth_missing_keys"

#     # aabb_obj_ids 可能是 numpy/torch/list，这里统一转为 python int set
#     try:
#         valid_ids = set(int(x) for x in list(aabb_obj_ids))
#     except Exception:
#         return None, "aabb_obj_ids_unparseable"

#     # instance_ids 用于点数检查
#     try:
#         inst_arr = instance_ids
#         # torch tensor -> numpy
#         if hasattr(inst_arr, "detach"):
#             inst_arr = inst_arr.detach().cpu().numpy()
#     except Exception:
#         inst_arr = None

#     candidates = [int(object_id), int(object_id) - 1, int(object_id) + 1]
#     for cand in candidates:
#         if cand <= 0:
#             continue
#         if cand not in valid_ids:
#             continue
#         if inst_arr is not None and min_inst_points > 1:
#             cnt = int((inst_arr == cand).sum())
#             if cnt < min_inst_points:
#                 continue
#         return cand, "ok"

#     return None, "not_in_aabb_or_too_few_points"


# # ================= 属性标记规则（spatial / color / texture / shape） =================

# def infer_attribute_flags(parsed_gt, atoms, kept_atoms):
#     """
#     根据 parsed_gt / atoms / kept_atoms 自动推断
#     spatial / color / texture / shape 四个布尔标记。

#     设计原则：
#     1）优先使用当前退化版本的 kept_atoms（更贴近 degraded_query 实际保留的语义）；
#     2）若 kept_atoms 为空，则回退到 atoms / parsed_gt 中的 atom 字符串；
#     3）通过简单的前缀或关键词匹配来判断属性类型：
#        - spatial:  "sprel:" 前缀，或 key 中带 "spatial"
#        - color:    "color:" / "colour:" / key 中带 "color"
#        - texture:  "texture:" / "material:" / "fabric:"
#        - shape:    "shape:" / "geom:" / key 中带 "shape"
#     """

#     spatial = False
#     color = False
#     texture = False
#     shape = False

#     atom_strings = []

#     # 1) 退化版本中显式给出的 kept_atoms
#     if kept_atoms:
#         for a in kept_atoms:
#             if isinstance(a, str):
#                 atom_strings.append(a)

#     # 2) 若 kept_atoms 不够，用 atoms（通常是 dict）做补充
#     #    atoms 的结构不固定，这里做一个尽量宽松的收集
#     if atoms and not atom_strings:
#         if isinstance(atoms, dict):
#             for v in atoms.values():
#                 if isinstance(v, list):
#                     for a in v:
#                         if isinstance(a, str):
#                             atom_strings.append(a)
#                 elif isinstance(v, str):
#                     atom_strings.append(v)
#         elif isinstance(atoms, list):
#             for a in atoms:
#                 if isinstance(a, str):
#                     atom_strings.append(a)

#     # 3) 再从 parsed_gt 中挖一些字符串（保险起见）
#     if parsed_gt and not atom_strings:
#         if isinstance(parsed_gt, dict):
#             for v in parsed_gt.values():
#                 if isinstance(v, list):
#                     for a in v:
#                         if isinstance(a, str):
#                             atom_strings.append(a)
#                 elif isinstance(v, str):
#                     atom_strings.append(v)

#     # 统一转小写，便于匹配
#     atom_strings_l = [a.lower() for a in atom_strings]

#     for s in atom_strings_l:
#         # 空间关系：sprel 前缀，或者 key 中包含 spatial
#         if s.startswith("sprel:") or "spatial" in s:
#             spatial = True

#         # 颜色：color / colour
#         if s.startswith("color:") or s.startswith("colour:") or "color" in s or "colour" in s:
#             color = True

#         # 纹理 / 材质：texture / material / fabric
#         if s.startswith("texture:") or "texture" in s or s.startswith("material:") or "material" in s \
#                 or "fabric" in s:
#             texture = True

#         # 形状：shape / geom
#         if s.startswith("shape:") or "shape" in s or s.startswith("geom:") or "geometric" in s:
#             shape = True

#     return spatial, color, texture, shape


# # ================= 主入口 =================

# def main():
#     parser = argparse.ArgumentParser()

#     # 输入：BlenderProc 输出目录（包含 scene_graph.json 与 stage3/*.json）
#     parser.add_argument(
#         "--output_root",
#         required=True,
#         help="main.py 的输出目录 (包含 scene_graph 和 stage3)"
#     )

#     # 输出：最终生成的 json（会自动加 _train_FRONT/_val_FRONT/_test_FRONT 后缀）
#     parser.add_argument(
#         "--save_path",
#         required=True,
#         help="最终生成的 json 路径前缀，例如 /path/to/scanrefer_front"
#     )

#     # 输出：划分结果保存目录（会写 train.txt/val.txt/test.txt，此处主要用于兼容，实际读取的是 split_txt_dir）
#     parser.add_argument(
#         "--divide_path",
#         required=True,
#         help="划分结果保存目录（写 train.txt/val.txt/test.txt）"
#     )

#     parser.add_argument(
#         "--use_ambiguous",
#         action="store_true",
#         help="是否包含 ambiguous 样本（当前脚本未特殊处理，仅保留接口）"
#     )

#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="随机种子（当前脚本已强制使用已有划分，该参数仅保留兼容）"
#     )

#     # 二次 merge 校验相关（可选）
#     parser.add_argument(
#         "--scene_dataset_path",
#         default=None,
#         help="preprocess 生成的 .pth 根目录（开启 object_id<->aabb_obj_ids 校验）"
#     )

#     parser.add_argument(
#         "--pth_split_suffix",
#         default="_FRONT",
#         help="pth 的 split 目录后缀，默认 train_FRONT/val_FRONT/test_FRONT"
#     )

#     # 关键修改1：将 split_txt_dir 设为必选，且默认指向 divide_path（和用户原有命令兼容）
#     parser.add_argument(
#         "--split_txt_dir",
#         default=None,
#         help="已有划分文件（train/val/test.txt）的目录，默认等于 --divide_path"
#     )

#     parser.add_argument(
#         "--min_inst_points",
#         type=int,
#         default=1,
#         help="校验时要求该 instance 至少包含的点数阈值（建议先用 1 跑通，再考虑 20/50）"
#     )

#     args = parser.parse_args()

#     # 关键修改2：强制 split_txt_dir 等于 divide_path（用户无需额外传参）
#     if args.split_txt_dir is None:
#         args.split_txt_dir = args.divide_path
#         print(f"自动使用 divide_path 作为 split_txt_dir: {args.split_txt_dir}")

#     random.seed(args.seed)

#     # 1) 扫 scene_graph，得到每场景的类别计数（这里目前只为兼容保留，eval_type 不再用它）
#     scene_cat_counts = load_scene_graphs(args.output_root)
#     _ = scene_cat_counts  # 当前脚本中未使用，仅保留以兼容后续可能的扩展

#     # 2) 强制读取已有 split txt（核心：不再有随机划分的分支）
#     print(f"强制读取已有划分文件: {args.split_txt_dir}")
#     scene_to_split = read_split_txts(args.split_txt_dir)
#     print(f"读取到 {len(scene_to_split)} 个 scene_id 的划分信息")

#     # 3) 收集所有语言条目（先不划分）
#     all_entries = []
#     scene_dirs = sorted([d for d in Path(args.output_root).iterdir() if d.is_dir()])
#     print(f"开始重新合并 {len(scene_dirs)} 个场景的数据（基于已有划分，读取更新后的 stage3 json）...")

#     for scene_dir in tqdm(scene_dirs):
#         scene_id = scene_dir.name
#         # 跳过不在已有划分中的场景（避免处理无用数据）
#         if scene_id not in scene_to_split:
#             continue
        
#         stage3_dir = scene_dir / "stage3"
#         if not stage3_dir.exists():
#             continue

#         for json_file in stage3_dir.glob("*.json"):
#             try:
#                 with open(json_file, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#             except Exception as e:
#                 print(f"[Warning] 读取 {json_file} 失败: {e}")
#                 continue

#             # ----------------- 基础字段 -----------------
#             oid_raw = data.get("object_id", None)
#             if oid_raw is None:
#                 continue
#             try:
#                 base_object_id = int(oid_raw)
#             except Exception:
#                 continue

#             parsed_gt = data.get("parsed_gt", {}) or {}
#             # object_name：优先用 parsed_gt["class"]，否则用原始 object_name 去掉 .xxx 后缀
#             obj_class = parsed_gt.get("class", None)
#             if obj_class is not None:
#                 object_name = obj_class
#             else:
#                 raw_obj_name = data.get("object_name", None)
#                 if raw_obj_name is not None:
#                     object_name = raw_obj_name.split(".")[0]
#                 else:
#                     object_name = None

#             gt_block = data.get("gt", {}) or {}
#             gt_query = gt_block.get("gt_query", "").strip()

#             ambiguity_type_global = data.get("ambiguity_type", None)
#             atoms = data.get("atoms", None)

#             # ----------------- 三种退化类型拆分 -----------------
#             degradation = data.get("degradation", {}) or {}
#             for deg_type in ("no_spatial", "no_visual", "minimal"):
#                 deg = degradation.get(deg_type, None)
#                 if not deg:
#                     continue

#                 description = (deg.get("degraded_query", "") or "").strip()
#                 if not description:
#                     continue

#                 # candidate / object_ids：variant_candidate_ids → int list
#                 raw_cands = deg.get("variant_candidate_ids", []) or []
#                 candidate_ids = []
#                 for x in raw_cands:
#                     try:
#                         candidate_ids.append(int(x))
#                     except Exception:
#                         continue
#                 if not candidate_ids:
#                     continue

#                 # object_ids：Multi3DRefer 风格，多目标集合
#                 object_ids = candidate_ids

#                 # eval_type：按 Multi3DRefer 风格设置
#                 # - 多目标：mt
#                 # - 单目标：st_wo_d
#                 if len(object_ids) > 1:
#                     eval_type = "mt"
#                 else:
#                     eval_type = "st_wo_d"

#                 # ann_id：为了避免和其他退化类型冲突，把 deg_type 拼进 scene_id 参与 hash
#                 ann_id = get_ann_id(scene_id=f"{scene_id}_{deg_type}", object_id=base_object_id)

#                 # 歧义相关标签
#                 ambiguity_type = deg.get("degraded_type", ambiguity_type_global or deg_type)
#                 is_ambiguous = bool(deg.get("is_ambiguous", False))

#                 # 四个属性标记：基于 kept_atoms / atoms / parsed_gt
#                 kept_atoms = deg.get("kept_atoms", []) or []
#                 spatial, color, texture, shape = infer_attribute_flags(
#                     parsed_gt=parsed_gt,
#                     atoms=atoms,
#                     kept_atoms=kept_atoms
#                 )

#                 entry = {
#                     "scene_id": scene_id,
#                     "object_id": base_object_id,          # 单个主目标 id（ScanRefer 风格，保持兼容）
#                     "object_ids": object_ids,             # Multi3DRefer 风格，多目标集合
#                     "candidate_ids": object_ids,          # 显式保留候选集合
#                     "object_name": object_name,
#                     "ann_id": ann_id,
#                     "description": description,
#                     "token": description.split(),
#                     "eval_type": eval_type,               # "mt" / "st_wo_d"
#                     "ambiguity_type": ambiguity_type,     # no_spatial / no_visual / minimal
#                     "is_ambiguous": is_ambiguous,
#                     "gt_query": gt_query,                 # 原始无歧义描述

#                     # 新增四个布尔属性标记（对当前 degraded_query 而言）
#                     "spatial": spatial,
#                     "color": color,
#                     "texture": texture,
#                     "shape": shape,
#                 }

#                 all_entries.append(entry)

#     print(f"重新合并完成: 有效样本 {len(all_entries)} 条（仅包含已有划分中的 scene_id）。")

#     # 4) 强制按已有 split txt 划分（核心：移除随机划分分支）
#     train_data, val_data, test_data = [], [], []
#     dropped = 0
#     for e in all_entries:
#         sid = e["scene_id"]
#         split = scene_to_split.get(sid, None)
#         if split is None:
#             dropped += 1
#             continue
#         if split == "train":
#             train_data.append(e)
#         elif split == "val":
#             val_data.append(e)
#         else:
#             test_data.append(e)
#     if dropped > 0:
#         print(f"[Warning] 有 {dropped} 条样本的 scene_id 不在已有划分文件中，已丢弃。")

#     # 5) 可选：开启 pth 校验与 object_id 纠偏（仍基于 base_object_id，不影响 object_ids）
#     if args.scene_dataset_path is not None:
#         print(f"开启 pth 校验模式：scene_dataset_path={args.scene_dataset_path}")
#         cache = PthCache(args.scene_dataset_path)

#         def split_dir_name(split_short: str):
#             # split_short: train/val/test  ->  train_FRONT/val_FRONT/test_FRONT
#             return f"{split_short}{args.pth_split_suffix}"

#         def run_validate(data_list, split_short: str):
#             out = []
#             bad = defaultdict(int)
#             sdir = split_dir_name(split_short)
#             for e in tqdm(data_list, desc=f"Validating {split_short}"):
#                 sid = e["scene_id"]
#                 pth = cache.load(sdir, sid)
#                 fixed, reason = validate_and_fix_object_id(
#                     pth, e["object_id"], min_inst_points=args.min_inst_points
#                 )
#                 if fixed is None:
#                     bad[reason] += 1
#                     continue
#                 e2 = dict(e)
#                 e2["object_id"] = int(fixed)
#                 # ann_id 与新的 object_id 再同步一次（此处不再区分 deg_type，只按 scene_id 原样来 hash）
#                 e2["ann_id"] = get_ann_id(sid, int(fixed))
#                 out.append(e2)
#             return out, bad

#         train_data, bad_tr = run_validate(train_data, "train")
#         val_data, bad_va = run_validate(val_data, "val")
#         test_data, bad_te = run_validate(test_data, "test")

#         print("[Validate Summary]")
#         print("train bad:", dict(bad_tr))
#         print("val   bad:", dict(bad_va))
#         print("test  bad:", dict(bad_te))
#         print(f"validate后样本数：train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

#     # 6) 写出 json（Multi3DRefer 风格，支持多目标）
#     base_path = os.path.splitext(args.save_path)[0]
#     train_path = base_path + "_train_FRONT.json"
#     val_path = base_path + "_val_FRONT.json"
#     test_path = base_path + "_test_FRONT.json"

#     with open(train_path, "w", encoding="utf-8") as f:
#         json.dump(train_data, f, indent=2, ensure_ascii=False)
#     with open(val_path, "w", encoding="utf-8") as f:
#         json.dump(val_data, f, indent=2, ensure_ascii=False)
#     with open(test_path, "w", encoding="utf-8") as f:
#         json.dump(test_data, f, indent=2, ensure_ascii=False)

#     print(f"已更新保存: {train_path} ({len(train_data)}) | {val_path} ({len(val_data)}) | {test_path} ({len(test_data)})")

#     # 7) 写出 scene list（兼容原有逻辑，保持 divide_path 下的 txt 文件不变）
#     ensure_dir(args.divide_path)

#     def save_scene_ids(data, path):
#         scene_ids = sorted(set(e["scene_id"] for e in data))
#         with open(path, "w", encoding="utf-8") as f:
#             for sid in scene_ids:
#                 f.write(sid + "\n")

#     # 可选：若需要同步更新 divide_path 下的 txt，取消下面注释；否则保持原有 txt 不变
#     # save_scene_ids(train_data, os.path.join(args.divide_path, "train.txt"))
#     # save_scene_ids(val_data, os.path.join(args.divide_path, "val.txt"))
#     # save_scene_ids(test_data, os.path.join(args.divide_path, "test.txt"))

#     print(f"scene_id 列表已保留在: {args.divide_path}/train.txt | val.txt | test.txt（未修改原有文件）")


# if __name__ == "__main__":
#     main()


import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
import hashlib
import random
from tqdm import tqdm

# ================= 基础工具函数 =================

def get_ann_id(scene_id, object_id: int) -> int:
    """
    生成稳定的 annotation id。
    这里不再区分退化类型，只按 (scene_id, object_id) 生成 hash。
    """
    raw = f"{scene_id}_{int(object_id)}"
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def load_scene_graphs(output_root):
    """
    预扫描 scene_graph.json，统计每个场景中各类别物体数量（当前版本不实际使用，只保留接口）。
    counts[scene_id][category_name] = count
    """
    counts = defaultdict(lambda: defaultdict(int))
    scene_dirs = [d for d in Path(output_root).iterdir() if d.is_dir()]

    print("正在扫描 Scene Graphs 以构建类别直方图（当前脚本中不实际使用，仅保留以兼容）...")
    for s_dir in tqdm(scene_dirs):
        scene_id = s_dir.name
        sg_path = s_dir / f"{scene_id}_scene_graph.json"
        if not sg_path.exists():
            continue

        try:
            with open(sg_path, "r", encoding="utf-8") as f:
                sg = json.load(f)

            for obj in sg.get("objects", []):
                raw_cat = obj.get("category") or obj.get("instance_name")
                if raw_cat is None:
                    continue
                counts[scene_id][raw_cat] += 1

        except Exception as e:
            print(f"[Warning] 读取 Scene Graph {scene_id} 失败: {e}")

    return counts


def read_split_txts(split_txt_dir: str):
    """读取 train.txt/val.txt/test.txt，返回 scene_id -> split_name 的映射"""
    mapping = {}
    for split_name in ("train", "val", "test"):
        p = os.path.join(split_txt_dir, f"{split_name}.txt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Split txt not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                sid = line.strip()
                if sid:
                    mapping[sid] = split_name
    return mapping


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# ================= 属性标记规则（spatial / color / texture / shape） =================

def infer_attribute_flags(parsed_gt, atoms, kept_atoms):
    """
    根据 parsed_gt / atoms / kept_atoms 自动推断
    spatial / color / texture / shape 四个布尔标记。

    规则：
      - 优先用当前退化版本的 kept_atoms；
      - 再回退到 atoms / parsed_gt；
      - 通过前缀或关键词匹配判断属性类型：
        * spatial:  "sprel:" 或 "spatial"
        * color:    "color:" / "colour:" / "color"/"colour"
        * texture:  "texture:" / "material:" / "fabric:"
        * shape:    "shape:" / "geom:" / "geometric"
    """

    spatial = False
    color = False
    texture = False
    shape = False

    atom_strings = []

    # 1) 退化版本中显式给出的 kept_atoms
    if kept_atoms:
        for a in kept_atoms:
            if isinstance(a, str):
                atom_strings.append(a)

    # 2) 回退到 atoms
    if atoms and not atom_strings:
        if isinstance(atoms, dict):
            for v in atoms.values():
                if isinstance(v, list):
                    for a in v:
                        if isinstance(a, str):
                            atom_strings.append(a)
                elif isinstance(v, str):
                    atom_strings.append(v)
        elif isinstance(atoms, list):
            for a in atoms:
                if isinstance(a, str):
                    atom_strings.append(a)

    # 3) 再从 parsed_gt 中挖
    if parsed_gt and not atom_strings:
        if isinstance(parsed_gt, dict):
            for v in parsed_gt.values():
                if isinstance(v, list):
                    for a in v:
                        if isinstance(a, str):
                            atom_strings.append(a)
                elif isinstance(v, str):
                    atom_strings.append(v)

    atom_strings_l = [a.lower() for a in atom_strings]

    for s in atom_strings_l:
        # 空间
        if s.startswith("sprel:") or "spatial" in s:
            spatial = True
        # 颜色
        if s.startswith("color:") or s.startswith("colour:") or "color" in s or "colour" in s:
            color = True
        # 纹理 / 材质
        if s.startswith("texture:") or "texture" in s or s.startswith("material:") or "material" in s \
                or "fabric" in s:
            texture = True
        # 形状
        if s.startswith("shape:") or "shape" in s or s.startswith("geom:") or "geometric" in s:
            shape = True

    return spatial, color, texture, shape


# ================= 主入口 =================

def main():
    parser = argparse.ArgumentParser()

    # 输入：BlenderProc 输出目录（包含 scene_graph.json 与 stage3/*.json）
    parser.add_argument(
        "--output_root",
        required=True,
        help="main.py 的输出目录 (包含 scene_graph 和 stage3)"
    )

    # 用来提供“文件名前缀”的路径，例如 /.../ScanRefer_FRONT_ambi.json
    # 实际不会写这个文件，只取 basename 做前缀
    parser.add_argument(
        "--save_path",
        required=True,
        help="仅用于决定输出文件名前缀，如 ambi 或 ScanRefer_FRONT_ambi"
    )

    # 已有的 train.txt/val.txt/test.txt 所在目录
    parser.add_argument(
        "--divide_path",
        required=True,
        help="已有划分结果目录（包含 train.txt/val.txt/test.txt）"
    )

    parser.add_argument(
        "--use_ambiguous",
        action="store_true",
        help="是否保留 is_ambiguous=True 的样本（当前脚本不额外过滤，仅保留开关）"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（当前主要用于 shuffle 等，实际没用到随机划分）"
    )

    # 三类退化的输出目录（你已经建好三套目录）
    parser.add_argument(
        "--no_visual_json_dir",
        type=str,
        required=True,
        help="no_visual 三个 json 输出目录"
    )
    parser.add_argument(
        "--no_spatial_json_dir",
        type=str,
        required=True,
        help="no_spatial 三个 json 输出目录"
    )
    parser.add_argument(
        "--minimal_json_dir",
        type=str,
        required=True,
        help="minimal 三个 json 输出目录"
    )

    args = parser.parse_args()

    random.seed(args.seed)

    # 1) 扫 scene_graph（当前仅兼容保留，不实际用）
    scene_cat_counts = load_scene_graphs(args.output_root)
    _ = scene_cat_counts

    # 2) 读取已有划分文件
    print(f"使用已有划分 txt: {args.divide_path}")
    scene_to_split = read_split_txts(args.divide_path)
    print(f"读取到 {len(scene_to_split)} 个 scene_id 的划分信息（train/val/test）")

    # 3) 遍历所有场景，收集三种退化类型的样本
    scene_dirs = sorted([d for d in Path(args.output_root).iterdir() if d.is_dir()])
    print(f"开始遍历 {len(scene_dirs)} 个场景的 stage3 json，按退化类型拆分...")

    # data_by_deg[deg_type]["train"/"val"/"test"] = list of entries
    data_by_deg = {
        "no_visual": {"train": [], "val": [], "test": []},
        "no_spatial": {"train": [], "val": [], "test": []},
        "minimal": {"train": [], "val": [], "test": []},
    }

    dropped_not_in_split = 0

    for scene_dir in tqdm(scene_dirs):
        scene_id = scene_dir.name

        # 不在划分里的场景直接跳过
        if scene_id not in scene_to_split:
            dropped_not_in_split += 1
            continue

        split_name = scene_to_split[scene_id]  # train/val/test
        stage3_dir = scene_dir / "stage3"
        if not stage3_dir.exists():
            continue

        for json_file in stage3_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[Warning] 读取 {json_file} 失败: {e}")
                continue

            # ----------------- 基础字段 -----------------
            oid_raw = data.get("object_id", None)
            if oid_raw is None:
                continue
            try:
                base_object_id = int(oid_raw)
            except Exception:
                continue

            parsed_gt = data.get("parsed_gt", {}) or {}
            # object_name：优先 parsed_gt["class"]，否则用原始 object_name 去掉 ".xxx"
            obj_class = parsed_gt.get("class", None)
            if obj_class is not None:
                object_name = obj_class
            else:
                raw_obj_name = data.get("object_name", None)
                if raw_obj_name is not None:
                    object_name = raw_obj_name.split(".")[0]
                else:
                    object_name = None

            gt_block = data.get("gt", {}) or {}
            gt_query = gt_block.get("gt_query", "").strip()

            ambiguity_type_global = data.get("ambiguity_type", None)
            atoms = data.get("atoms", None)

            # ----------------- 三种退化类型拆分 -----------------
            degradation = data.get("degradation", {}) or {}

            for deg_type in ("no_visual", "no_spatial", "minimal"):
                deg = degradation.get(deg_type, None)
                if not deg:
                    continue

                description = (deg.get("degraded_query", "") or "").strip()
                if not description:
                    continue

                # candidate_ids：variant_candidate_ids -> int list
                raw_cands = deg.get("variant_candidate_ids", []) or []
                candidate_ids = []
                for x in raw_cands:
                    try:
                        candidate_ids.append(int(x))
                    except Exception:
                        continue
                if not candidate_ids:
                    continue

                # Multi3DRefer 风格：object_ids = candidate_ids
                object_ids = candidate_ids

                # eval_type：由目标数量决定
                if len(object_ids) > 1:
                    eval_type = "mt"
                else:
                    eval_type = "st_wo_d"

                # ann_id：只按 scene_id + object_id 生成（不区分退化类型）
                ann_id = get_ann_id(scene_id, base_object_id)

                # 歧义相关
                ambiguity_type = deg.get("degraded_type", ambiguity_type_global or deg_type)
                is_ambiguous = bool(deg.get("is_ambiguous", False))
                if (not args.use_ambiguous) and is_ambiguous:
                    # 如果你希望只保留非 ambiguous 的，这里可以 continue
                    # 当前逻辑：无论 ambiguous 与否都保留
                    pass

                # 四个属性标记
                kept_atoms = deg.get("kept_atoms", []) or []
                spatial, color, texture, shape = infer_attribute_flags(
                    parsed_gt=parsed_gt,
                    atoms=atoms,
                    kept_atoms=kept_atoms
                )

                entry = {
                    "scene_id": scene_id,
                    "object_id": base_object_id,      # 主目标 id（ScanRefer 风格字段）
                    "object_ids": object_ids,         # Multi3DRefer 风格多目标
                    "candidate_ids": object_ids,      # 显式保留候选集合
                    "object_name": object_name,
                    "ann_id": ann_id,
                    "description": description,
                    "token": description.split(),
                    "eval_type": eval_type,
                    "ambiguity_type": ambiguity_type,
                    "is_ambiguous": is_ambiguous,
                    "gt_query": gt_query,

                    # 四个属性标记
                    "spatial": spatial,
                    "color": color,
                    "texture": texture,
                    "shape": shape,
                }

                if deg_type in data_by_deg:
                    data_by_deg[deg_type][split_name].append(entry)

    print(f"重新合并完成。丢弃（不在 split txt 中的场景）: {dropped_not_in_split} 个。")
    for deg_type in ("no_visual", "no_spatial", "minimal"):
        print(
            f"{deg_type}: "
            f"train={len(data_by_deg[deg_type]['train'])}, "
            f"val={len(data_by_deg[deg_type]['val'])}, "
            f"test={len(data_by_deg[deg_type]['test'])}"
        )

    # 4) 写出 json：文件名格式为
    #    {base_name}_{deg_type}_{split}.json
    #    例如: ambi_no_visual_train.json
    base_name = os.path.splitext(os.path.basename(args.save_path))[0]

    deg_out_root = {
        "no_visual": args.no_visual_json_dir,
        "no_spatial": args.no_spatial_json_dir,
        "minimal": args.minimal_json_dir,
    }

    for deg_type, split_dict in data_by_deg.items():
        out_root = deg_out_root[deg_type]
        ensure_dir(out_root)

        for split_name, entries in split_dict.items():  # split_name: train/val/test
            file_name = f"{base_name}_{deg_type}_{split_name}.json"
            save_path = os.path.join(out_root, file_name)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
            print(f"[Save] {deg_type} {split_name}: {save_path}  ({len(entries)} 条样本)")

    print("全部完成。")


if __name__ == "__main__":
    main()

