# split_by_variant.py
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def split_items_by_variant(
    items: List[Dict[str, Any]],
    keep_types: List[str],
    *,
    drop_empty: bool = True,
    keep_only_variant_subtree: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    New json format:
      - it["degradation"] is a dict: { "no_spatial": {...}, "no_visual": {...}, "minimal": {...} }
      - it["stage3_text"] is a dict: { "no_spatial": {...}, "no_visual": {...}, "minimal": {...} }

    Output:
      buckets[t] is a list of items. Each output item keeps only the subtree for variant t
      (similar to your old "keep same item but filter dialogues").
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {t: [] for t in keep_types}

    for it in items:
        degradation = it.get("degradation") or {}
        stage3_text = it.get("stage3_text") or {}

        # 这条样本有哪些 variant 可用（出现在 degradation 或 stage3_text 中都算）
        available = set()
        if isinstance(degradation, dict):
            available |= set(degradation.keys())
        if isinstance(stage3_text, dict):
            available |= set(stage3_text.keys())

        # 为每个要保留的 t 生成一个“只包含 t 子树”的新样本
        for t in keep_types:
            if t not in available:
                if not drop_empty:
                    # 如果你希望缺失的也输出（一般不需要），可以走这里
                    new_it = deepcopy(it)
                    if keep_only_variant_subtree:
                        if "degradation" in new_it:
                            new_it["degradation"] = {}
                        if "stage3_text" in new_it:
                            new_it["stage3_text"] = {}
                    new_it["variant"] = t
                    buckets[t].append(new_it)
                continue

            new_it = deepcopy(it)

            # 可选：只保留当前 variant 的 degradation / stage3_text
            if keep_only_variant_subtree:
                if isinstance(new_it.get("degradation"), dict):
                    new_it["degradation"] = {t: new_it["degradation"][t]} if t in new_it["degradation"] else {}
                if isinstance(new_it.get("stage3_text"), dict):
                    new_it["stage3_text"] = {t: new_it["stage3_text"][t]} if t in new_it["stage3_text"] else {}

            # 可选：加一个顶层字段方便下游识别当前属于哪个 split
            new_it["variant"] = t

            # （可选增强）把常用字段“抬平”到顶层，方便后续训练/统计
            # 你如果不需要可以删除这段
            dg = (new_it.get("degradation") or {}).get(t, {})
            s3 = (new_it.get("stage3_text") or {}).get(t, {})
            if isinstance(dg, dict):
                new_it.setdefault("degraded_query", dg.get("degraded_query"))
                new_it.setdefault("degraded_query_struct", dg.get("degraded_query_struct"))
                new_it.setdefault("variant_candidate_ids", dg.get("variant_candidate_ids"))
                new_it.setdefault("is_ambiguous", dg.get("is_ambiguous"))
                new_it.setdefault("need_feedback", dg.get("need_feedback"))
            if isinstance(s3, dict):
                new_it.setdefault("clarification_dialogue", s3.get("clarification_dialogue"))
                new_it.setdefault("restored_query", s3.get("restored_query"))
                new_it.setdefault("restored_query_raw", s3.get("restored_query_raw"))

            buckets[t].append(new_it)

    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="input json (top-level list)")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument(
        "--types",
        nargs="+",
        default=["no_spatial", "no_visual", "minimal"],
        help="variants to split (default: no_spatial no_visual minimal)",
    )
    ap.add_argument("--name_prefix", default="split", help="output filename prefix (default: split)")
    ap.add_argument(
        "--keep_only_variant_subtree",
        type=int,
        default=1,
        help="1: keep only degradation/stage3_text of that variant; 0: keep full item (default: 1)",
    )
    ap.add_argument(
        "--drop_empty",
        type=int,
        default=1,
        help="1: only output items that have this variant; 0: also output empty placeholders (default: 1)",
    )
    args = ap.parse_args()

    items = load_json(args.in_json)
    if not isinstance(items, list):
        raise ValueError("Expected top-level JSON to be a list of items.")

    buckets = split_items_by_variant(
        items,
        args.types,
        drop_empty=bool(args.drop_empty),
        keep_only_variant_subtree=bool(args.keep_only_variant_subtree),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for t, out_items in buckets.items():
        out_path = out_dir / f"{args.name_prefix}_{t}_3DFRONT.json"
        save_json(str(out_path), out_items)
        print(f"[OK] {t}: {len(out_items)} items -> {out_path}")


if __name__ == "__main__":
    main()
