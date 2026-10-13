from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.,!?;]")

def simple_tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    return _TOKEN_RE.findall(text)

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int = 0
    unk_id: int = 1

    @classmethod
    def build(cls, texts: List[str], min_freq: int = 1, max_size: int = 20000) -> "Vocab":
        from collections import Counter
        ctr = Counter()
        for t in texts:
            ctr.update(simple_tokenize(t))
        itos = ["<pad>", "<unk>"]
        for w, c in ctr.most_common():
            if c < min_freq:
                break
            if w in ("<pad>", "<unk>"):
                continue
            itos.append(w)
            if len(itos) >= max_size:
                break
        stoi = {w: i for i, w in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)
    @classmethod
    def from_itos(cls, itos: List[str]) -> "Vocab":
        """
        Rebuild vocab from a stored index-to-string list.
        Assumes itos[0] is <pad>, itos[1] is <unk> (or your existing convention).
        """
        v = cls.__new__(cls)  # bypass __init__ if it requires args
        v.itos = list(itos)
        v.stoi = {s: i for i, s in enumerate(v.itos)}
        # if you have special tokens fields, set them safely
        v.pad_token = getattr(v, "pad_token", "<pad>")
        v.unk_token = getattr(v, "unk_token", "<unk>")
        v.pad_id = v.stoi.get(v.pad_token, 0)
        v.unk_id = v.stoi.get(v.unk_token, 1 if len(v.itos) > 1 else 0)
        return v
    def encode(self, text: str, max_len: int) -> List[int]:
        toks = simple_tokenize(text)
        ids = [self.stoi.get(t, self.unk_id) for t in toks[:max_len]]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids
