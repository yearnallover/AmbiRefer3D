from __future__ import annotations
from typing import List, Dict, Any, Tuple
import random

def split_samples_7_3(samples: List[Dict[str, Any]], seed: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministic 7:3 split.
    Default: shuffle all samples then split.
    If you prefer scene-disjoint split, tell me and I'll change it.
    """
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_train = int(round(len(samples) * 0.7))
    train = [samples[i] for i in idx[:n_train]]
    test  = [samples[i] for i in idx[n_train:]]
    return train, test

def split_samples_0_10(samples, seed=0):
    return [], samples
