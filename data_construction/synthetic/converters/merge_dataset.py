import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
import hashlib
import random
from tqdm import tqdm

# 二次 merge 校验时需要读取 .pth
try:
    import torch
except ImportError:
    torch = None


def get_ann_id(scene_id, object_id: int) -> int:
    """生成稳定的 annotation id（与 object_id 的类型无关，最终用 int 参与 hash 更稳）"""
    raw = f"{scene_id}_{int(object_id)}"
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16) % (10**8)


def load_scene_graphs(output_root):
    """
    预扫描 scene_graph.json，统计每个场景中各类别物体数量，用于 eval_type: unique/multiple
    counts[scene_id][category_name] = count
    """
    counts = defaultdict(lambda: defaultdict(int))
    scene_dirs = [d for d in Path(output_root).iterdir() if d.is_dir()]

    print("正在扫描 Scene Graphs 以构建类别直方图...")
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
            print(f"[Warning] Error reading SG for {scene_id}: {e}")

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


class PthCache:
    """避免重复读取同一个 scene 的 .pth，提高二次 merge 校验速度"""
    def __init__(self, scene_dataset_path: str):
        if torch is None:
            raise RuntimeError("torch 未安装，无法开启 pth 校验模式。")
        self.scene_dataset_path = scene_dataset_path
        self.cache = {}  # (split_dir, scene_id) -> dict

    def load(self, split_dir: str, scene_id: str):
        key = (split_dir, scene_id)
        if key in self.cache:
            return self.cache[key]
        pth_path = os.path.join(self.scene_dataset_path, split_dir, f"{scene_id}.pth")
        if not os.path.exists(pth_path):
            self.cache[key] = None
            return None
        try:
            d = torch.load(pth_path, map_location="cpu")
        except Exception:
            d = None
        self.cache[key] = d
        return d


def validate_and_fix_object_id(scene_pth: dict, object_id: int, min_inst_points: int = 1):
    """
    使用 scene_pth 中的 aabb_obj_ids / instance_ids 校验 object_id。
    返回 (fixed_object_id or None, reason)
    """
    if scene_pth is None:
        return None, "pth_missing_or_unreadable"

    aabb_obj_ids = scene_pth.get("aabb_obj_ids", None)
    instance_ids = scene_pth.get("instance_ids", None)

    if aabb_obj_ids is None or instance_ids is None:
        return None, "pth_missing_keys"

    # aabb_obj_ids 可能是 numpy/torch/list，这里统一转为 python int set
    try:
        valid_ids = set(int(x) for x in list(aabb_obj_ids))
    except Exception:
        return None, "aabb_obj_ids_unparseable"

    # instance_ids 用于点数检查
    try:
        inst_arr = instance_ids
        # torch tensor -> numpy
        if hasattr(inst_arr, "detach"):
            inst_arr = inst_arr.detach().cpu().numpy()
    except Exception:
        inst_arr = None

    candidates = [int(object_id), int(object_id) - 1, int(object_id) + 1]
    for cand in candidates:
        if cand <= 0:
            continue
        if cand not in valid_ids:
            continue
        if inst_arr is not None and min_inst_points > 1:
            cnt = int((inst_arr == cand).sum())
            if cnt < min_inst_points:
                continue
        return cand, "ok"

    return None, "not_in_aabb_or_too_few_points"


def main():
    parser = argparse.ArgumentParser()

    # 输入：BlenderProc 输出目录（包含 scene_graph.json 与 stage3/*.json）
    parser.add_argument("--output_root", required=True,
                        help="main.py 的输出目录 (包含 scene_graph 和 stage3)")

    # 输出：最终生成的 json（会自动加 _train_FRONT/_val_FRONT/_test_FRONT 后缀）
    parser.add_argument("--save_path", required=True,
                        help="最终生成的 json 路径前缀，例如 /path/to/scanrefer_front")

    # 输出：划分结果保存目录（会写 train.txt/val.txt/test.txt）
    parser.add_argument("--divide_path", required=True,
                        help="划分结果保存目录（写 train.txt/val.txt/test.txt）")

    parser.add_argument("--use_ambiguous", action="store_true",
                        help="是否包含 ambiguous 样本（当前脚本未特殊处理，仅保留接口）")

    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（第一次 merge 随机划分用）")

    # 二次 merge 校验相关（可选）
    parser.add_argument("--scene_dataset_path", default=None,
                        help="preprocess 生成的 .pth 根目录（开启 object_id<->aabb_obj_ids 校验）")

    parser.add_argument("--pth_split_suffix", default="_FRONT",
                        help="pth 的 split 目录后缀，默认 train_FRONT/val_FRONT/test_FRONT")

    parser.add_argument("--split_txt_dir", default=None,
                        help="若提供，则按已有 train.txt/val.txt/test.txt 分配样本，不再随机划分（用于二次 merge）")

    parser.add_argument("--min_inst_points", type=int, default=1,
                        help="校验时要求该 instance 至少包含的点数阈值（建议先用 1 跑通，再考虑 20/50）")

    args = parser.parse_args()

    random.seed(args.seed)

    # 1) 扫 scene_graph，得到每场景的类别计数，用于 eval_type
    scene_cat_counts = load_scene_graphs(args.output_root)

    # 2) 读取 split（若提供 split_txt_dir，则固定 split；否则后面随机 split）
    scene_to_split = None
    if args.split_txt_dir is not None:
        scene_to_split = read_split_txts(args.split_txt_dir)
        print(f"使用已有 split_txt_dir={args.split_txt_dir}，将按 train/val/test.txt 固定划分。")

    # 3) 收集所有语言条目（先不划分）
    all_entries = []
    scene_dirs = sorted([d for d in Path(args.output_root).iterdir() if d.is_dir()])
    print(f"开始合并 {len(scene_dirs)} 个场景的数据...")

    for scene_dir in tqdm(scene_dirs):
        scene_id = scene_dir.name
        stage3_dir = scene_dir / "stage3"
        if not stage3_dir.exists():
            continue

        for json_file in stage3_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            # object_id：强制转 int（这是你要求的关键修改）
            oid_raw = data.get("object_id", None)
            if oid_raw is None:
                continue
            try:
                object_id = int(oid_raw)
            except Exception:
                continue

            parsed_gt = data.get("parsed_gt", {}) or {}
            raw_obj_name = parsed_gt.get("class", None)

            # description
            gt = data.get("gt", {}) or {}
            description = gt.get("gt_query") if isinstance(gt, dict) else None
            if not description:
                continue

            total_count = scene_cat_counts[scene_id].get(raw_obj_name, 1)
            eval_type = "unique" if total_count == 1 else "multiple"

            entry = {
                "scene_id": scene_id,
                "object_id": object_id,                     # int
                "object_name": raw_obj_name,
                "ann_id": get_ann_id(scene_id, object_id), # int
                "description": description,
                "token": description.split(),
                "eval_type": eval_type,
            }
            all_entries.append(entry)

    print(f"合并完成: 有效样本 {len(all_entries)} 条。")

    # 4) 分配到 train/val/test
    # 4.1 若提供 split_txt_dir：按 scene_id 固定划分
    if scene_to_split is not None:
        train_data, val_data, test_data = [], [], []
        dropped = 0
        for e in all_entries:
            sid = e["scene_id"]
            split = scene_to_split.get(sid, None)
            if split is None:
                dropped += 1
                continue
            if split == "train":
                train_data.append(e)
            elif split == "val":
                val_data.append(e)
            else:
                test_data.append(e)
        if dropped > 0:
            print(f"[Warning] 有 {dropped} 条样本的 scene_id 不在 split_txt_dir 中，已丢弃。")
    else:
        # 4.2 否则随机按“条目级别”划分（与原脚本一致）
        random.shuffle(all_entries)
        n = len(all_entries)
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_data = all_entries[:n_train]
        val_data = all_entries[n_train:n_train + n_val]
        test_data = all_entries[n_train + n_val:]

    # 5) 可选：开启 pth 校验与纠偏（建议用于二次 merge）
    if args.scene_dataset_path is not None:
        print(f"开启 pth 校验模式：scene_dataset_path={args.scene_dataset_path}")
        cache = PthCache(args.scene_dataset_path)

        def split_dir_name(split_short: str):
            # split_short: train/val/test  ->  train_FRONT/val_FRONT/test_FRONT
            return f"{split_short}{args.pth_split_suffix}"

        def run_validate(data_list, split_short: str):
            out = []
            bad = defaultdict(int)
            sdir = split_dir_name(split_short)
            for e in tqdm(data_list, desc=f"Validating {split_short}"):
                sid = e["scene_id"]
                pth = cache.load(sdir, sid)
                fixed, reason = validate_and_fix_object_id(
                    pth, e["object_id"], min_inst_points=args.min_inst_points
                )
                if fixed is None:
                    bad[reason] += 1
                    continue
                # 写回纠偏后的 object_id，并保持 ann_id 与之对应（强烈建议同步更新 ann_id）
                e2 = dict(e)
                e2["object_id"] = int(fixed)
                e2["ann_id"] = get_ann_id(sid, int(fixed))
                out.append(e2)
            return out, bad

        train_data, bad_tr = run_validate(train_data, "train")
        val_data, bad_va = run_validate(val_data, "val")
        test_data, bad_te = run_validate(test_data, "test")

        print("[Validate Summary]")
        print("train bad:", dict(bad_tr))
        print("val   bad:", dict(bad_va))
        print("test  bad:", dict(bad_te))
        print(f"validate后样本数：train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # 6) 写出 json
    base_path = os.path.splitext(args.save_path)[0]
    train_path = base_path + "_train_FRONT.json"
    val_path = base_path + "_val_FRONT.json"
    test_path = base_path + "_test_FRONT.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"已保存: {train_path} ({len(train_data)}) | {val_path} ({len(val_data)}) | {test_path} ({len(test_data)})")

    # 7) 写出 scene list（第一次 merge 必须写；二次 merge 是否重写都可以，这里保持写出，便于追溯）
    ensure_dir(args.divide_path)

    def save_scene_ids(data, path):
        scene_ids = sorted(set(e["scene_id"] for e in data))
        with open(path, "w", encoding="utf-8") as f:
            for sid in scene_ids:
                f.write(sid + "\n")

    save_scene_ids(train_data, os.path.join(args.divide_path, "train.txt"))
    save_scene_ids(val_data, os.path.join(args.divide_path, "val.txt"))
    save_scene_ids(test_data, os.path.join(args.divide_path, "test.txt"))

    print(f"并保存 scene_id 列表到: {args.divide_path}/train.txt | val.txt | test.txt")


if __name__ == "__main__":
    main()
