from __future__ import annotations
from typing import Tuple
import numpy as np

def parse_bbox(b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accept bbox formats:
    - [cx,cy,cz, dx,dy,dz]
    - [cx,cy,cz, dx,dy,dz, sem]
    Returns:
      center: (C,3)
      size:   (C,3)
      sem:    (C,1) float (0 if missing)
    """
    b = b.astype(np.float32)
    if b.ndim != 2 or b.shape[1] < 6:
        raise ValueError(f"bbox must be [C,>=6], got {b.shape}")
    center = b[:, 0:3]
    size = b[:, 3:6]
    if b.shape[1] >= 7:
        sem = b[:, 6:7]
    else:
        sem = np.zeros((b.shape[0], 1), dtype=np.float32)
    return center, size, sem

def cand_geom_features(cand_bboxes: np.ndarray) -> np.ndarray:
    """
    Output: [C, 11]
      center(3), size(3), volume(1), size_norm(1), center_norm(1), aspect(1), sem(1)
    """
    center, size, sem = parse_bbox(cand_bboxes)
    vol = np.prod(size, axis=-1, keepdims=True)
    size_norm = np.linalg.norm(size, axis=-1, keepdims=True) + 1e-6
    center_norm = np.linalg.norm(center, axis=-1, keepdims=True) + 1e-6
    aspect = (np.max(size, axis=-1, keepdims=True) / (np.min(size, axis=-1, keepdims=True) + 1e-6))
    feat = np.concatenate([center, size, vol, size_norm, center_norm, aspect, sem], axis=-1).astype(np.float32)
    return feat

def pairwise_rel_features(cand_bboxes: np.ndarray) -> np.ndarray:
    """
    Output: [C,C,7] = delta_center(3), dist(1), unit_dir(3)
    """
    center, _, _ = parse_bbox(cand_bboxes)
    dc = center[None, :, :] - center[:, None, :]  # (j - i)
    dist = np.linalg.norm(dc, axis=-1, keepdims=True) + 1e-6
    unit = dc / dist
    rel = np.concatenate([dc, dist, unit], axis=-1).astype(np.float32)
    return rel

def normalize_candidates(cand_bboxes: np.ndarray) -> np.ndarray:
    """
    Normalize center/size only. Keep sem unchanged.
    """
    cand_bboxes = cand_bboxes.astype(np.float32)
    center = cand_bboxes[:, 0:3]
    size = cand_bboxes[:, 3:6]

    c_mu = center.mean(axis=0, keepdims=True)
    c_std = center.std(axis=0, keepdims=True) + 1e-6
    s_mu = size.mean(axis=0, keepdims=True)
    s_std = size.std(axis=0, keepdims=True) + 1e-6

    center_n = (center - c_mu) / c_std
    size_n = (size - s_mu) / s_std

    out = cand_bboxes.copy()
    out[:, 0:3] = center_n
    out[:, 3:6] = size_n
    return out
