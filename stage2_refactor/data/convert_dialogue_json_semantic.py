# stage2/data/convert_dialogue_json_semantic.py
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
try:
    from stage2_refactor.utils.io import load_json, save_json
except ImportError:
    from stage2_refactor.utils.io import load_json, save_json
import random

REL_KEYWORDS = {
    # proximity
    "near": [
        "near", "next to", "close to", "close", "beside", "by", "adjacent to"
    ],

    # front / back
    "behind": [
        "behind", "in back of", "back of"
    ],
    "in_front_of": [
        "in front of", "front of", "in the front of"
    ],

    # left / right
    "left_of": [
        "left of", "on the left of", "left", "leftmost"
    ],
    "right_of": [
        "right of", "on the right of", "right", "rightmost"
    ],

    # vertical
    "above": [
        "above", "over", "on top of", "top"
    ],
    "below": [
        "below", "under", "underneath", "beneath"
    ],

    # between / middle
    "between": [
        "between", "middle", "center", "in the middle of"
    ],

    # distance comparison (可选，常用于 disambiguation)
    "closest": [
        "closest", "nearest"
    ],
    "farthest": [
        "furthest", "farthest", "far"
    ],

    # relative / layout
    "opposite": [
        "opposite", "across from", "across"
    ],
    "along": [
        "along", "against"
    ],
    "inside": [
        "inside", "within"
    ],
}
ATTR_KEYS = {
    # color
    "color": [
        "white", "black", "brown", "blue", "red", "green", "yellow",
        "pink", "orange", "purple", "gray", "grey",
        "beige", "tan", "silver",
        "dark", "light"
    ],

    # size / scale
    "size": [
        "big", "large", "small", "tiny",
        "long", "short",
        "wide", "narrow"
    ],

    # height / vertical extent
    "height": [
        "tall", "low", "high"
    ],

    # shape
    "shape": [
        "round", "square", "rectangular", "rectangle",
        "circular", "curved", "shaped"
    ],

    # material
    "material": [
        "wood", "wooden",
        "metal", "glass", "plastic", "leather", "marble"
    ],

    # state / openness（可选，但在家具中有用）
    "state": [
        "open", "closed"
    ],
    "position": [
        "leftmost", "rightmost", "middle", "center"
    ],
}


def _norm(s: str) -> str:
    return (s or "").strip().lower()
# ---- add near the top (after _norm) ----
STOPWORDS = {
    "the","a","an","this","that","these","those",
    "of","to","in","on","at","from","into","by","with","for","as",
    "and","or","but","than","then","it","one","another",
}

# canonicalize attr surface forms -> normalized token
ATTR_NORM = {
    "grey": "gray",
    "rectangular": "rectangle",
    "circular": "circle",
    "wooden": "wood",
    "larger": "large",
    "largest": "large",
    "bigger": "large",
    "biggest": "large",
    "smaller": "small",
    "smallest": "small",
    "taller": "tall",
    "tallest": "tall",
    "shorter": "short",
    "shortest": "short",
}

# prefer longer rel phrases first to avoid "front of" stealing "in front of"
def _build_rel_phrase_map():
    pairs = []
    for rel, kws in REL_KEYWORDS.items():
        for kw in kws:
            pairs.append((kw, rel))
    # sort by number of tokens then length
    pairs.sort(key=lambda x: (len(x[0].split()), len(x[0])), reverse=True)
    return pairs

REL_PHRASES = _build_rel_phrase_map()

# regex boundary match for multi-word phrases
def _phrase_in_text(phrase: str, text: str) -> Optional[re.Match]:
    # allow flexible spaces
    p = re.escape(phrase).replace(r"\ ", r"\s+")
    # word boundaries at ends
    pat = rf"(?<![a-z0-9']){p}(?![a-z0-9'])"
    return re.search(pat, text)

def _clean_landmark(after: str) -> str:
    # cut at punctuation
    after = re.split(r"[?.!,;:]", after, maxsplit=1)[0]
    # remove leading determiners
    after = re.sub(r"^\s*(the|a|an)\s+", "", after).strip()
    toks = re.findall(r"[a-z0-9']+", after.lower())

    # stop at conjunctions / prepositions that usually start another clause
    out = []
    for tok in toks:
        if tok in STOPWORDS:
            if out:
                break
            else:
                continue
        # stop if we accidentally hit another relation cue token
        if tok in {"left","right","behind","front","near","between","above","below","under","over","closest","furthest","farthest","across","opposite","along","against"}:
            if out:
                break
            # if landmark starts with relation token, skip it
            continue
        out.append(tok)
        if len(out) >= 3:
            break
    return " ".join(out)

def infer_semantic_constraint(text: str) -> Dict[str, Any]:
    """
    Output format (unchanged):
      REL:  {"ctype":"REL","rel":"near","landmark_text":"cabinet"}
      ATTR: {"ctype":"ATTR","attr_key":"color","attr_val_text":"brown"}
    """
    t = _norm(text)
    if not t:
        return {"ctype": "REL", "rel": "near", "landmark_text": ""}

    # Handle inverse containment before generic relation matching. In phrases
    # such as "cabinet with a TV inside it", the landmark precedes "inside".
    contains_patterns = (
        r"\bwith\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+inside(?:\s+it)?(?:[?.!,;:]|$)",
        r"\b(?:has|have|having|contains)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+inside(?:\s+it)?|[?.!,;:]|$)",
    )
    for pattern in contains_patterns:
        match = re.search(pattern, t)
        if match:
            landmark = _clean_landmark(match.group(1))
            if landmark:
                return {
                    "ctype": "REL",
                    "rel": "contains",
                    "landmark_text": landmark,
                }

    inverse_relation_patterns = (
        (r"\bwith\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:directly\s+)?to\s+its\s+right\b", "left_of"),
        (r"\bwith\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:directly\s+)?to\s+its\s+left\b", "right_of"),
        (r"\bwith\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:directly\s+)?in\s+front\s+of\s+it\b", "behind"),
        (r"\bwith\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:directly\s+)?behind\s+it\b", "in_front_of"),
        (r"\bwith\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:directly\s+)?above\s+it\b", "below"),
        (r"\bwith\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:directly\s+)?(?:below|under)\s+it\b", "above"),
        (r"\bwith\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:on\s+top|next\s+to|nearby|near\s+it)\b", "near"),
    )
    for pattern, rel in inverse_relation_patterns:
        match = re.search(pattern, t)
        if match:
            landmark = _clean_landmark(match.group(1))
            if landmark:
                return {"ctype": "REL", "rel": rel, "landmark_text": landmark}

    direct_relation_patterns = (
        (r"\b(?:on|upon)\s+(?:top\s+of\s+)?(?:the\s+|a\s+|an\s+)?(.+?)(?:[?.!,;:]|$)", "above"),
        (r"\bcloser\s+to\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:[?.!,;:]|$)", "closest"),
        (r"\b(?:farther|further)\s+from\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:[?.!,;:]|$)", "farthest"),
    )
    for pattern, rel in direct_relation_patterns:
        match = re.search(pattern, t)
        if match:
            landmark = _clean_landmark(match.group(1))
            if landmark:
                return {"ctype": "REL", "rel": rel, "landmark_text": landmark}

    # -------- REL detection (robust, longest phrase first) --------
    for phrase, rel in REL_PHRASES:
        m = _phrase_in_text(phrase, t)
        if not m:
            continue

        after = t[m.end():].strip()

        # special handling: "between X and Y" -> keep first landmark chunk
        if rel == "between":
            # take chunk before "and"
            after = re.split(r"\band\b", after, maxsplit=1)[0]

        landmark = _clean_landmark(after)
        if landmark:
            return {"ctype": "REL", "rel": rel, "landmark_text": landmark}

    # -------- ATTR detection (token based + normalization) --------
    toks = re.findall(r"[a-z0-9']+", t)
    # normalize tokens
    toks_norm = [ATTR_NORM.get(x, x) for x in toks]

    for attr_key, vals in ATTR_KEYS.items():
        # normalize vals too
        vals_norm = [ATTR_NORM.get(v, v) for v in vals]
        hit = next((v for v in vals_norm if v in toks_norm), None)
        if hit is not None:
            return {"ctype": "ATTR", "attr_key": attr_key, "attr_val_text": hit}

    # fallback (unchanged structure)
    return {"ctype": "REL", "rel": "near", "landmark_text": ""}


def _unique_ints(values: List[Any]) -> List[int]:
    out: List[int] = []
    seen = set()
    for x in values or []:
        try:
            v = int(x)
        except (TypeError, ValueError):
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _candidate_ids_from_dialogue(d: Dict[str, Any], target_id: int) -> List[int]:
    for key in ("candidate_ids", "variant_candidate_ids", "keep_candidate_ids", "grounding_matches"):
        cand_ids = _unique_ints(d.get(key) or [])
        if cand_ids:
            return cand_ids

    distractors = _unique_ints(d.get("active_distractor_ids") or [])
    return _unique_ints([target_id] + distractors)


def convert(scene_json_path: str, out_round_path: str, seed: int = 0) -> None:
    items = load_json(scene_json_path)
    out: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    for it in items:
        scene_id = it["scene_id"]
        ann_id = str(it.get("ann_id", "0"))
        target_id = int(it["target_id"])
        dialogues = it.get("dialogues", [])

        for di, d in enumerate(dialogues):
            query = (d.get("input_query") or "").strip()

            # Preserve the stage1 candidate set. Do not force-add the target;
            # if stage1 missed it, stage2 should count that as not survived.
            cand_ids = _candidate_ids_from_dialogue(d, target_id)
            if not cand_ids:
                continue

            # --- shuffle candidates & track target index ---
            rng.shuffle(cand_ids)
            target_index = cand_ids.index(target_id) if target_id in cand_ids else -1

            is_amb = bool(d.get("is_ambiguous", True))
            stop_y = 0 if is_amb else 1

            teacher = d.get("generated_question") or d.get("restored_query") or query
            constraint_gt = infer_semantic_constraint(teacher)

            out.append({
                "scene_id": scene_id,
                "ann_id": ann_id,
                "round": di,
                "query": query,
                "cand_ids": cand_ids,
                "target_id": target_id,
                "target_survived_stage1": target_index >= 0,
                "constraint_gt": constraint_gt,
                "stop_y": stop_y,
                "target_index": target_index,   
            })
            row = out[-1]
            source_ids = _candidate_ids_from_dialogue(d, target_id)
            order = [source_ids.index(candidate_id) for candidate_id in cand_ids]
            for field in ("cand_bboxes", "cand_stage1_features", "cand_stage1_scores",
                          "cand_point_counts", "cand_alive_probs", "cand_semantic_ids", "cand_belief"):
                if field in d:
                    if len(d[field]) != len(source_ids):
                        raise ValueError(f"{scene_id}: {field} length differs from candidate IDs")
                    row[field] = [d[field][i] for i in order]


    save_json(out_round_path, out)
    print(f"saved round samples: {out_round_path}, n={len(out)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_json", required=True)
    ap.add_argument("--out_round_json", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    convert(args.scene_json, args.out_round_json, seed=args.seed)
