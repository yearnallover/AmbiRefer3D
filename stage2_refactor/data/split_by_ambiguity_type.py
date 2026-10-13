# split_by_ambiguity_type.py
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

def split_scene_dialogues_by_type(
    items: List[Dict[str, Any]],
    keep_types: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {t: [] for t in keep_types}

    for it in items:
        dialogues = it.get("dialogues", []) or []
        # 按 type 收集该 item 里属于某 type 的 dialogues
        per_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for d in dialogues:
            t = d.get("ambiguity_type")
            if t in buckets:
                per_type[t].append(d)

        # 对每个 type：如果该 item 有对应 dialogues，则输出一个“同样的 item，但 dialogues 过滤后”
        for t, ds in per_type.items():
            new_it = deepcopy(it)
            new_it["dialogues"] = ds
            buckets[t].append(new_it)

    return buckets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="input scene-level json (list of items)")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument(
        "--types",
        nargs="+",
        default=["no_spatial", "no_visual", "minimal"],
        help="ambiguity types to split (default: no_spatial no_visual minimal)",
    )
    ap.add_argument(
        "--name_prefix",
        default="split",
        help="output filename prefix (default: split)",
    )
    args = ap.parse_args()

    items = load_json(args.in_json)
    if not isinstance(items, list):
        raise ValueError("Expected top-level JSON to be a list of items.")

    buckets = split_scene_dialogues_by_type(items, args.types)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存
    for t, out_items in buckets.items():
        out_path = out_dir / f"{args.name_prefix}_{t}.json"
        save_json(str(out_path), out_items)
        print(f"[OK] {t}: {len(out_items)} items -> {out_path}")

if __name__ == "__main__":
    main()
