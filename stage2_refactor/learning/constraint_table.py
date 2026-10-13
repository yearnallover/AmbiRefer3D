from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import re
import torch

def norm(s: str) -> str:
    return (s or "").strip().lower()

def constraint_to_key(c: Dict[str, Any]) -> str:
    c = c or {}
    ctype = (c.get("ctype") or "REL").upper()
    if ctype == "ATTR":
        ak = norm(c.get("attr_key") or "other")
        av = norm(c.get("attr_val_text") or "")
        return f"ATTR|{ak}|{av}"
    rel = norm(c.get("rel") or "near")
    lm = norm(c.get("landmark_text") or "")
    return f"REL|{rel}|{lm}"

@dataclass
class ConstraintRow:
    key: str
    ctype_id: int
    rel_id: int
    attr_id: int
    field_text: str  # landmark_text or attr_val_text (already normalized)

class ConstraintTable:
    """
    Discrete inventory of constraints, with ability to:
      - map constraint_gt dict -> constraint_id
      - map constraint_id -> slot ids + field_text
      - provide tensor pack for batch update during eval rollout
    """
    def __init__(self, keys: List[str], rel2id: Dict[str,int], attr2id: Dict[str,int]) -> None:
        self.keys = keys
        self.stoi = {k:i for i,k in enumerate(keys)}
        self.rel2id = rel2id
        self.attr2id = attr2id
        self.rows: List[ConstraintRow] = [self._parse_key(k) for k in keys]

    @staticmethod
    def build_from_samples(samples: List[Dict[str, Any]], rel2id: Dict[str,int], attr2id: Dict[str,int]) -> "ConstraintTable":
        uniq = sorted(set(constraint_to_key(s.get("constraint_gt") or {}) for s in samples))
        return ConstraintTable(uniq, rel2id=rel2id, attr2id=attr2id)

    def _parse_key(self, key: str) -> ConstraintRow:
        # key format: REL|near|cabinet  or ATTR|color|brown
        parts = key.split("|")
        if len(parts) < 3:
            # fallback
            parts = ["REL", "near", ""]
        ctype = parts[0].upper()
        a = norm(parts[1])
        b = norm("|".join(parts[2:]))  # allow landmark/value to contain pipes (rare)

        if ctype == "ATTR":
            ctype_id = 1
            attr_id = self.attr2id.get(a, self.attr2id.get("other", 5))
            rel_id = 0
            field_text = b
        else:
            ctype_id = 0
            rel_id = self.rel2id.get(a, 0)
            attr_id = self.attr2id.get("other", 5)
            field_text = b

        return ConstraintRow(key=key, ctype_id=ctype_id, rel_id=rel_id, attr_id=attr_id, field_text=field_text)

    def id_from_constraint_gt(self, c: Dict[str, Any]) -> int:
        key = constraint_to_key(c)
        return self.stoi.get(key, 0)

    def row(self, cid: int) -> ConstraintRow:
        return self.rows[int(cid)]

    def num_constraints(self) -> int:
        return len(self.keys)

    def apply_id_to_batch(self, batch: Dict[str, torch.Tensor], cid: torch.Tensor, vocab, max_len: int) -> Dict[str, torch.Tensor]:
        """
        Update batch fields in-place for given constraint id(s).
        batch: contains c_ctype_id, c_rel_id, c_attr_id, field_ids
        cid: [B] long
        """
        device = cid.device
        B = cid.shape[0]
        c_ctype = torch.zeros((B,), dtype=torch.long, device=device)
        c_rel = torch.zeros((B,), dtype=torch.long, device=device)
        c_attr = torch.zeros((B,), dtype=torch.long, device=device)
        field_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)

        # Python loop is fine for eval B=1 or small B
        for i in range(B):
            r = self.row(int(cid[i].item()))
            c_ctype[i] = r.ctype_id
            c_rel[i] = r.rel_id
            c_attr[i] = r.attr_id
            ids = vocab.encode(r.field_text, max_len)
            field_ids[i] = torch.tensor(ids, dtype=torch.long, device=device)

        batch["c_ctype_id"] = c_ctype
        batch["c_rel_id"] = c_rel
        batch["c_attr_id"] = c_attr
        batch["field_ids"] = field_ids
        return batch

