import json
import argparse
from typing import Any, Dict, List, Optional


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return x


def _get_variant(sample: Dict[str, Any], prefer_variant: Optional[str] = None) -> str:
    """
    Decide which variant to export.
    Priority:
      1) user-specified prefer_variant
      2) sample["variant"]
      3) first key in sample["degradation"] (if exists)
      4) "minimal" as fallback
    """
    if prefer_variant:
        return prefer_variant
    if isinstance(sample.get("variant"), str) and sample["variant"]:
        return sample["variant"]
    degr = sample.get("degradation", {})
    if isinstance(degr, dict) and len(degr) > 0:
        return next(iter(degr.keys()))
    return "minimal"


def _map_removed_info(degr_block: Dict[str, Any]) -> List[str]:
    """
    Map removed atom types -> target schema removed_info.
    Your target schema expects something like:
      ["spatial", "visual_attributes"]
    """
    removed_detail = degr_block.get("removed_detail", [])
    types = set()
    for rd in removed_detail:
        t = rd.get("atom_type")
        if not t:
            continue
        types.add(t)

    removed_info = []
    # map your types -> desired names
    if "spatial" in types:
        removed_info.append("spatial")
    if "visual" in types:
        removed_info.append("visual_attributes")
    # sometimes class is removed in some settings (rare in your sample)
    if "class" in types:
        removed_info.append("class")

    return removed_info


def _get_gt_description(sample: Dict[str, Any]) -> str:
    # prefer gt.gt_query, else fallback to restored_query_raw/restored_query
    gt = sample.get("gt", {})
    if isinstance(gt, dict) and isinstance(gt.get("gt_query"), str) and gt["gt_query"]:
        return gt["gt_query"]
    if isinstance(sample.get("restored_query_raw"), str) and sample["restored_query_raw"]:
        return sample["restored_query_raw"]
    if isinstance(sample.get("restored_query"), str) and sample["restored_query"]:
        return sample["restored_query"]
    return ""


def convert_one(sample: Dict[str, Any], prefer_variant: Optional[str] = None) -> Dict[str, Any]:
    variant = _get_variant(sample, prefer_variant)

    scene_id = sample.get("scene_id", "")
    target_id = _safe_int(sample.get("object_id", sample.get("target_id", "")))
    ann_id = sample.get("ann_id", "")

    # candidate set (alive pool)
    cand_ids = sample.get("candidate_ids", sample.get("variant_candidate_ids", [])) or []
    cand_ids_norm = [_safe_int(x) for x in cand_ids]

    # distractors_count: candidates - 1 (exclude target if present)
    distractors_count = max(len(set(cand_ids_norm)) - 1, 0)

    # pick degraded block for removed_info + input query
    degr_block = {}
    if isinstance(sample.get("degradation"), dict):
        degr_block = sample["degradation"].get(variant, {}) or {}

    input_query = sample.get("degraded_query")
    if not isinstance(input_query, str) or not input_query:
        input_query = degr_block.get("degraded_query", "")
    if not isinstance(input_query, str):
        input_query = ""

    removed_info = _map_removed_info(degr_block)

    # dialogue list:
    # Priority: sample["clarification_dialogue"] (flattened) else sample["stage3_text"][variant]["clarification_dialogue"]
    clar_list = sample.get("clarification_dialogue", None)
    if not isinstance(clar_list, list):
        stage3 = sample.get("stage3_text", {})
        if isinstance(stage3, dict):
            vblock = stage3.get(variant, {})
            if isinstance(vblock, dict) and isinstance(vblock.get("clarification_dialogue"), list):
                clar_list = vblock["clarification_dialogue"]
    if not isinstance(clar_list, list):
        clar_list = []

    # track "alive" candidate ids across rounds (optional)
    alive = set(cand_ids_norm)

    dialogues_out = []
    for turn in clar_list:
        if not isinstance(turn, dict):
            continue

        # update alive if keep_candidate_ids provided and non-empty
        keep = turn.get("keep_candidate_ids", [])
        if isinstance(keep, list) and len(keep) > 0:
            keep_norm = set(_safe_int(x) for x in keep)
            # many pipelines mean "keep these candidates"; ensure target kept if it was missing
            keep_norm.add(target_id)
            alive = keep_norm

        active_distractors = sorted([x for x in alive if x != target_id], key=lambda z: str(z))

        dialogues_out.append({
            "ambiguity_type": variant,
            "input_query": input_query,
            "removed_info": removed_info,
            "active_distractor_ids": active_distractors,
            "is_ambiguous": bool(sample.get("is_ambiguous", degr_block.get("is_ambiguous", True))),
            "generated_question": turn.get("model_question", ""),
            "restored_query": turn.get("query_after", sample.get("restored_query", "")),
            "reasoning": turn.get("reasoning", ""),
        })

    # if no dialogue turns, still output a single stub turn (optional but often convenient)
    if len(dialogues_out) == 0:
        active_distractors = sorted([x for x in alive if x != target_id], key=lambda z: str(z))
        dialogues_out = [{
            "ambiguity_type": variant,
            "input_query": input_query,
            "removed_info": removed_info,
            "active_distractor_ids": active_distractors,
            "is_ambiguous": bool(sample.get("is_ambiguous", degr_block.get("is_ambiguous", True))),
            "generated_question": "",
            "restored_query": sample.get("restored_query", ""),
            "reasoning": "",
        }]

    out = {
        "scene_id": scene_id,
        "target_id": target_id,
        "ann_id": ann_id,
        "gt_description": _get_gt_description(sample),
        "distractors_count": distractors_count,
        "dialogues": dialogues_out,
    }
    return out


def convert_file(in_path: str, out_path: str, prefer_variant: Optional[str] = None):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples.")

    out_data = [convert_one(s, prefer_variant=prefer_variant) for s in data]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="path to original json")
    ap.add_argument("--out_json", required=True, help="path to converted json")
    ap.add_argument("--variant", default=None, help="force export variant, e.g., minimal / w_o_visual / w_o_spatial")
    args = ap.parse_args()

    convert_file(args.in_json, args.out_json, prefer_variant=args.variant)


if __name__ == "__main__":
    main()
