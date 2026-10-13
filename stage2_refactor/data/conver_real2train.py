# convert_results_back_to_dialogue_format.py
from __future__ import annotations
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

def load_json(p: str):
    with open(p, "r") as f:
        return json.load(f)

def save_json(p: str, obj):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _norm_text(s: Optional[str]) -> str:
    # match 用：小写、去首尾空白、压缩空格
    if not s:
        return ""
    s = s.strip().lower()
    s = " ".join(s.split())
    return s

def infer_mode_and_round(result_json_path: str) -> Tuple[str, int]:
    """
    你给的规则：
      if "all" in args.result_json: cand_mode="all"
      if "spatial" in args.result_json: round_id=0
      if "visual" in args.result_json: round_id=1
      if "minimal" in args.result_json: round_id=2
    """
    p = result_json_path.lower()
    cand_mode = "one"   # 默认只查一个 round
    round_id = 0        # 默认 0

    if "all" in p:
        cand_mode = "all"
    if "spatial" in p:
        round_id = 0
    if "visual" in p:
        round_id = 1
    if "minimal" in p:
        round_id = 2
    return cand_mode, round_id

def build_scene_index(orig_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    scene_id -> [orig_item...]
    因为同 scene 可能多个 ann
    """
    mp: Dict[str, List[Dict[str, Any]]] = {}
    for it in orig_items:
        sid = it.get("scene_id")
        if sid is None:
            continue
        mp.setdefault(sid, []).append(it)
    return mp

def pick_dialogue_by_generated_question(
    orig_item: Dict[str, Any],
    gen_q: str,
    cand_mode: str,
    round_id: int
) -> Optional[Tuple[int, Dict[str, Any]]]:
    """
    返回 (matched_round_id, matched_dialogue_dict)
    - cand_mode == "all": 在 0/1/2 都找
    - cand_mode != "all": 只在 round_id 找
    匹配策略：优先 exact norm match；找不到就返回 None
    """
    dialogues = orig_item.get("dialogues") or []
    qn = _norm_text(gen_q)

    if cand_mode == "all":
        round_candidates = [0, 1, 2]
    else:
        round_candidates = [0, 1, 2]

    for rid in round_candidates:
        if rid < 0 or rid >= len(dialogues):
            continue
        d = dialogues[rid] or {}
        d_q = _norm_text(d.get("input_query"))
        if d_q and d_q == qn:
            return rid, d

    return None

def convert(
    orig_json_path: str,
    result_json_path: str,
    out_json_path: str,
) -> None:
    orig_items = load_json(orig_json_path)
    res_items = load_json(result_json_path)

    cand_mode, round_id = infer_mode_and_round(result_json_path)
    scene_index = build_scene_index(orig_items)

    out: List[Dict[str, Any]] = []

    # 逐条结果去找原始条目
    for r in res_items:
        scene_id = r.get("scene_id")
        if not scene_id:
            continue

        gen_q = r.get("query")
        if not gen_q:
            # 你说要用 generated_question 定位；这里缺失就无法可靠匹配
            continue

        L = len(r.get("predicted_bboxes") or [])
        if L <= 0:
            continue

        # 在同 scene 下所有 ann 里找 dialogue 匹配
        candidates = scene_index.get(scene_id, [])
        matched = None  # (orig_item, rid, dialogue)
        for it in candidates:
            got = pick_dialogue_by_generated_question(it, gen_q, cand_mode=cand_mode, round_id=round_id)
            if got is None:
                continue
            rid, d = got
            matched = (it, rid, d)
            break

        if matched is None:
            continue

        it, rid, d = matched

        candidate_ids = r.get("candidate_ids", r.get("cand_ids"))
        if candidate_ids is None:
            raise ValueError("Predicted boxes require explicit candidate_ids; box-count truncation is invalid")
        candidate_ids = [int(value) for value in candidate_ids]
        if len(candidate_ids) != L:
            raise ValueError("candidate_ids and predicted_bboxes must have matching lengths")
        d2 = dict(d)
        d2["candidate_ids"] = candidate_ids
        d2["active_distractor_ids"] = [value for value in candidate_ids if value != int(it["target_id"])]
        d2["target_survived_stage1"] = int(it["target_id"]) in candidate_ids

        # 其余字段（ambiguity_type/input_query/removed_info/.../grounding_matches 等）保持不动
        # scene-level 字段从 orig_item 取
        out.append({
            "scene_id": it["scene_id"],
            "target_id": int(it["target_id"]),
            "ann_id": str(it.get("ann_id", "0")),
            "gt_description": it.get("gt_description", it.get("description", "")),  # 兼容字段名不同
            "distractors_count": int(it.get("distractors_count", len(d2["active_distractor_ids"]))),
            "dialogues": [d2],
        })

    save_json(out_json_path, out)
    print(f"saved: {out_json_path}, n={len(out)}")
    print(f"mode={cand_mode}, round_id={round_id}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_json", required=True, help="原始 dialogues json（含 target_id/dialogues/...）")
    ap.add_argument("--result_json", required=True, help="待处理结果 json（含 predicted_bboxes + generated_question）")
    ap.add_argument("--out_json", required=True, help="输出为你给的新格式")
    args = ap.parse_args()
    convert(args.orig_json, args.result_json, args.out_json)
