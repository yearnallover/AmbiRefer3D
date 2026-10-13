#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def find_list_container(obj: Any) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    尝试从 obj 中找到“样本列表”容器。
    返回 (list_ref, key_path)
    - list_ref: 找到的样本 list（元素应为 dict）
    - key_path: list 在原 json 中的大致路径（便于打印提示）
    """
    if isinstance(obj, list):
        # 顶层就是 list
        if len(obj) == 0 or isinstance(obj[0], dict):
            return obj, "<root>"
        return None, None

    if isinstance(obj, dict):
        # 常见键名候选
        candidates = ["data", "annotations", "items", "samples", "entries"]
        for k in candidates:
            if k in obj:
                v = obj[k]
                if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                    return v, k
                # 如果是 dict，再往下一层找
                if isinstance(v, dict):
                    for kk in candidates:
                        if kk in v:
                            vv = v[kk]
                            if isinstance(vv, list) and (len(vv) == 0 or isinstance(vv[0], dict)):
                                return vv, f"{k}.{kk}"

        # 兜底：在所有键里找第一个“list of dict”
        for k, v in obj.items():
            if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                return v, k

    return None, None


def filter_mt(samples: List[Dict[str, Any]], mt_value: str = "mt") -> List[Dict[str, Any]]:
    out = []
    missing = 0
    for s in samples:
        if not isinstance(s, dict):
            continue
        if "eval_type" not in s:
            missing += 1
            continue
        if s.get("eval_type") == mt_value:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="path to multi3drefer_val.json")
    ap.add_argument("--output", "-o", required=True, help="path to write multi3drefer_val_mt.json")
    ap.add_argument("--mt_value", default="mt", help='value of sval_type for MT (default: "mt")')
    ap.add_argument("--indent", type=int, default=2, help="json indent (default: 2)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples, key_path = find_list_container(data)
    if samples is None:
        raise RuntimeError(
            "无法在输入 JSON 中定位到“样本列表”。请打开 multi3drefer_val.json 看一下样本存放的键名，"
            "然后告诉我它的结构（顶层是 list 还是 dict，以及样本列表在哪个 key 下）。"
        )

    total = len(samples)
    mt_samples = filter_mt(samples, mt_value=args.mt_value)
    mt_total = len(mt_samples)

    # 写出：保持原结构不变，只替换样本列表为 MT 子集
    if isinstance(data, list) and key_path == "<root>":
        out_data = mt_samples
    elif isinstance(data, dict):
        out_data = data
        # 根据 key_path 替换对应 list
        if "." in (key_path or ""):
            k1, k2 = key_path.split(".", 1)
            out_data[k1][k2] = mt_samples
        else:
            out_data[key_path] = mt_samples
    else:
        out_data = mt_samples

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=args.indent)

    print(f"[OK] Input : {in_path}  (total samples: {total})")
    print(f"[OK] MT    : eval_type == '{args.mt_value}'  -> {mt_total} samples")
    print(f"[OK] Saved : {out_path}")
    print(f"[INFO] List container path: {key_path}")


if __name__ == "__main__":
    main()
