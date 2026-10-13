# stage2/data/bbox_provider.py
from __future__ import annotations
from typing import Dict, List, Protocol, Optional
import os
import numpy as np
import torch

class BBoxProvider(Protocol):
    def get_bboxes(self, scene_id: str, cand_ids: List[int], sample: Optional[Dict] = None) -> np.ndarray:
        """
        Return cand_bboxes: [C,7] float32
        """
        ...

class PassthroughBBoxProvider:
    """
    For real stage1: sample already contains cand_bboxes [C,7].
    """
    def get_bboxes(self, scene_id: str, cand_ids: List[int], sample: Optional[Dict] = None) -> np.ndarray:
        if sample is None or ("cand_bboxes" not in sample):
            raise KeyError("sample['cand_bboxes'] missing for PassthroughBBoxProvider")
        b = np.asarray(sample["cand_bboxes"], dtype=np.float32)
        if b.ndim != 2 or b.shape[0] != len(cand_ids) or b.shape[1] < 7:
            raise ValueError(f"cand_bboxes must be [C,7], got {b.shape}, C={len(cand_ids)}")
        return b[:, :7].astype(np.float32)

class ScanNetBBoxProvider:
    """
    For simulation: load bbox from ScanNet files (you already do this).
    You need to implement _load_scene_bbox_dict according to your ScanNet folder format.
    """
    def __init__(self, scannet_root: str, split: str = "auto") -> None:
        self.scannet_root = scannet_root
        self.split = split
        self._cache = {}  # scene_id -> dict(inst_id -> bbox7)

    def _scene_path(self, scene_id: str) -> str:
        candidates = []
        if self.split != "auto":
            candidates.append(os.path.join(self.scannet_root, self.split, f"{scene_id}.pth"))
        candidates.append(os.path.join(self.scannet_root, f"{scene_id}.pth"))
        for split in ("train", "val", "test"):
            candidates.append(os.path.join(self.scannet_root, split, f"{scene_id}.pth"))
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Could not find {scene_id}.pth under {self.scannet_root}; checked split={self.split}."
        )

    def _load_scene_bbox_dict(self, scene_id: str) -> Dict[int, np.ndarray]:
        scene = torch.load(self._scene_path(scene_id), map_location="cpu", weights_only=False)
        corners = np.asarray(scene["aabb_corner_xyz"], dtype=np.float32)
        obj_ids = np.asarray(scene["aabb_obj_ids"], dtype=np.int64).reshape(-1)
        if corners.ndim != 3 or corners.shape[0] != obj_ids.shape[0] or corners.shape[-1] != 3:
            raise ValueError(
                f"Invalid aabb arrays for {scene_id}: corners={corners.shape}, obj_ids={obj_ids.shape}"
            )

        instance_ids = np.asarray(scene.get("instance_ids", []), dtype=np.int64).reshape(-1)
        sem_labels = np.asarray(scene.get("sem_labels", []), dtype=np.int64).reshape(-1)
        out: Dict[int, np.ndarray] = {}
        for row, obj_id in enumerate(obj_ids.tolist()):
            lo = corners[row].min(axis=0)
            hi = corners[row].max(axis=0)
            center = (lo + hi) * 0.5
            size = np.maximum(hi - lo, 1e-4)
            semantic_id = 0
            if instance_ids.shape == sem_labels.shape and instance_ids.size > 0:
                labels = sem_labels[(instance_ids == obj_id) & (sem_labels >= 0)]
                if labels.size > 0:
                    semantic_id = int(np.bincount(labels).argmax())
            out[int(obj_id)] = np.concatenate(
                (center, size, np.asarray([semantic_id], dtype=np.float32)), axis=0
            ).astype(np.float32)
        return out

    def get_bboxes(self, scene_id: str, cand_ids: List[int], sample: Optional[Dict] = None) -> np.ndarray:
        if scene_id not in self._cache:
            self._cache[scene_id] = self._load_scene_bbox_dict(scene_id)
        m = self._cache[scene_id]
        C = len(cand_ids)
        out = np.zeros((C, 7), dtype=np.float32)
        for i, cid in enumerate(cand_ids):
            if int(cid) in m:
                out[i] = m[int(cid)]
            else:
                # keep zeros; caller may drop or keep
                out[i] = 0
        return out
