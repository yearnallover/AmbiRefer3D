from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _to_int(x: Any) -> Any:
    """Best-effort convert numeric string to int; otherwise keep original."""
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return x


def removed_info_from_variant(variant: str) -> List[str]:
    # 你示例里旧格式用的是 ["spatial", "visual_attributes"]
    if variant == "no_spatial":
        return ["spatial"]
    if variant == "no_visual":
        return ["visual_attributes"]
    if variant == "minimal":
        return ["spatial", "visual_attributes"]
    # fallback：未知类型不强行猜
    return []


def build_dialogue_from_variant(it: Dict[str, Any], variant: str) -> Optional[Dict[str, Any]]:
    degradation = (it.get("degradation") or {}).get(variant)
    stage3 = (it.get("stage3_text") or {}).get(variant)

    if degradation is None and stage3 is None:
        return None

    # input_query：优先用 degraded_query（你新格式里就是退化后的 query）
    input_query = None
    if isinstance(degradation, dict):
        input_query = degradation.get("degraded_query") or degradation.get("degraded_query_struct")
    if input_query is None:
        # fallback：如果没有 degradation，就用 stage3 的 query_before
        if isinstance(stage3, dict):
            dial = stage3.get("clarification_dialogue") or []
            if isinstance(dial, list) and len(dial) > 0:
                input_query = dial[0].get("query_before")

    # generated_question / reasoning：默认取 round0
    generated_question = None
    reasoning = None
    restored_query = None

    if isinstance(stage3, dict):
        dial = stage3.get("clarification_dialogue") or []
        if isinstance(dial, list) and len(dial) > 0:
            generated_question = dial[0].get("model_question")
            # 如果你希望把多轮 reasoning 合起来：
            if len(dial) == 1:
                reasoning = dial[0].get("reasoning")
            else:
                parts = []
                for r in dial:
                    rr = r.get("round")
                    rs = r.get("reasoning")
                    if rs:
                        parts.append(f"[round {rr}] {rs}")
                reasoning = "\n".join(parts) if parts else dial[0].get("reasoning")

        # restored_query：优先用 stage3_text[variant].restored_query（你新格式里有）
        restored_query = stage3.get("restored_query") or stage3.get("restored_query_raw")
        # 如果没有 restored_query，就退回用最后一轮 query_after
        if restored_query is None and isinstance(dial, list) and len(dial) > 0:
            restored_query = dial[-1].get("query_after")

    # active_distractor_ids：候选里去掉 target（object_id）
    target_id = _to_int(it.get("object_id"))
    cand_ids_raw = it.get("candidate_ids") or []
    cand_ids = [_to_int(x) for x in cand_ids_raw]
    active_distractor_ids = [x for x in cand_ids if x != target_id]

    # grounding_matches：用 variant_candidate_ids（如果有），否则用 candidate_ids
    grounding_matches = None
    if isinstance(degradation, dict) and degradation.get("variant_candidate_ids") is not None:
        grounding_matches = [_to_int(x) for x in degradation.get("variant_candidate_ids")]
    else:
        grounding_matches = cand_ids

    # is_ambiguous：优先 degradation.is_ambiguous
    is_ambiguous = None
    if isinstance(degradation, dict):
        is_ambiguous = degradation.get("is_ambiguous")
    if is_ambiguous is None:
        # fallback：如果没提供就按候选数>1
        is_ambiguous = len(cand_ids) > 1

    dialogue = {
        "ambiguity_type": variant,
        "input_query": input_query,
        "removed_info": removed_info_from_variant(variant),
        "active_distractor_ids": active_distractor_ids,
        "is_ambiguous": bool(is_ambiguous),
        "generated_question": generated_question,
        "restored_query": restored_query,
        "reasoning": reasoning,
        "grounding_matches": grounding_matches,
    }

    return dialogue


def convert_item(it: Dict[str, Any], keep_variants: List[str]) -> Dict[str, Any]:
    scene_id = it.get("scene_id")
    target_id = _to_int(it.get("object_id"))
    ann_id = it.get("ann_id")

    gt_description = None
    gt = it.get("gt") or {}
    if isinstance(gt, dict):
        gt_description = gt.get("gt_query")

    cand_ids_raw = it.get("candidate_ids") or []
    cand_ids = [_to_int(x) for x in cand_ids_raw]
    distractors_count = max(0, len(cand_ids) - 1)

    dialogues: List[Dict[str, Any]] = []
    for v in keep_variants:
        d = build_dialogue_from_variant(it, v)
        if d is not None:
            dialogues.append(d)

    out = {
        "scene_id": scene_id,
        "target_id": target_id,
        "ann_id": ann_id,
        "gt_description": gt_description,
        "distractors_count": distractors_count,
        "dialogues": dialogues,
    }
    return out


def split_by_type(converted_items: List[Dict[str, Any]], keep_types: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = {t: [] for t in keep_types}
    for it in converted_items:
        ds = it.get("dialogues") or []
        for t in keep_types:
            sub = [d for d in ds if d.get("ambiguity_type") == t]
            if not sub:
                continue
            new_it = deepcopy(it)
            new_it["dialogues"] = sub
            buckets[t].append(new_it)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="input new-format json (list)")
    ap.add_argument("--out_json", default="", help="output merged json (all variants in dialogues). If empty, skip.")
    ap.add_argument("--out_dir", default="", help="if set, also save split jsons into this dir")
    ap.add_argument(
        "--types",
        nargs="+",
        default=["no_spatial", "no_visual", "minimal"],
        help="variants/types to keep (default: no_spatial no_visual minimal)",
    )
    ap.add_argument("--name_prefix", default="split", help="prefix for split outputs (default: split)")
    args = ap.parse_args()

    items = load_json(args.in_json)
    if not isinstance(items, list):
        raise ValueError("Expected top-level JSON to be a list.")

    converted = [convert_item(it, args.types) for it in items]

    if args.out_json:
        save_json(args.out_json, converted)
        print(f"[OK] merged: {len(converted)} items -> {args.out_json}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        buckets = split_by_type(converted, args.types)
        for t, out_items in buckets.items():
            out_path = out_dir / f"{args.name_prefix}_{t}.json"
            save_json(str(out_path), out_items)
            print(f"[OK] {t}: {len(out_items)} items -> {out_path}")


if __name__ == "__main__":
    main()
