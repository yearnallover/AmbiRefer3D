from __future__ import annotations
from typing import Dict, Any, List, Tuple

from .constraint_table import constraint_to_key

def build_inventory(samples: List[Dict[str, Any]]) -> Tuple[Dict[str,int], List[str]]:
    keys = []
    for s in samples:
        keys.append(constraint_to_key(s.get("constraint_gt") or {}))
    uniq = sorted(set(keys))
    stoi = {k:i for i,k in enumerate(uniq)}
    return stoi, uniq
