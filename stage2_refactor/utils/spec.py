# utils/spec.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import json
import os

@dataclass
class ExperimentSpec:
    seed: int
    train_indices: List[int]
    test_indices: List[int]
    vocab_itos: List[str]
    constraint_keys: List[str]
    rel2id: Dict[str, int]
    attr2id: Dict[str, int]
    train_keys: List[Any] = None
    test_keys: List[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "train_indices": self.train_indices,
            "test_indices": self.test_indices,
            "vocab_itos": self.vocab_itos,
            "constraint_keys": self.constraint_keys,
            "rel2id": self.rel2id,
            "attr2id": self.attr2id,
            "train_keys": self.train_keys or [],
            "test_keys": self.test_keys or [],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentSpec":
        return ExperimentSpec(
            seed=int(d["seed"]),
            train_indices=[int(x) for x in d["train_indices"]],
            test_indices=[int(x) for x in d["test_indices"]],
            vocab_itos=list(d["vocab_itos"]),
            constraint_keys=list(d["constraint_keys"]),
            rel2id={k: int(v) for k, v in d["rel2id"].items()},
            attr2id={k: int(v) for k, v in d["attr2id"].items()},
            train_keys=list(d.get("train_keys", [])),
            test_keys=list(d.get("test_keys", [])),
        )

def save_spec(path: str, spec: ExperimentSpec) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(spec.to_dict(), f, indent=2)

def load_spec(path: str) -> ExperimentSpec:
    with open(path, "r") as f:
        return ExperimentSpec.from_dict(json.load(f))
