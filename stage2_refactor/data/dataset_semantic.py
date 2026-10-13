from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .vocab import Vocab
from .bbox_provider import *
from stage2_refactor.learning.constraint_table import ConstraintTable

def load_imgfeat_cache_npz(img_cache_dir: str, scene_id: str) -> Optional[Dict[int, np.ndarray]]:
    path = os.path.join(img_cache_dir, f"{scene_id}.npz")
    if not os.path.exists(path):
        return None
    npz = np.load(path)
    obj_ids = npz["obj_ids"].astype(np.int32)
    feats = npz["feats"].astype(np.float32)  # [N,D]
    return {int(obj_ids[i]): feats[i] for i in range(obj_ids.shape[0])}


def load_stage1_cache_npz(stage1_cache_dir: str, scene_id: str) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
    if not stage1_cache_dir:
        return None
    path = os.path.join(stage1_cache_dir, f"{scene_id}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as npz:
        obj_ids = np.asarray(npz["obj_ids"], dtype=np.int64).reshape(-1)
        arrays = {
            key: np.asarray(npz[key]).copy()
            for key in npz.files
            if key != "obj_ids"
        }
    for key, array in arrays.items():
        if array.shape[0] != obj_ids.shape[0]:
            raise ValueError(
                f"Invalid Stage1 cache {path}: {key} has {array.shape[0]} rows, "
                f"expected {obj_ids.shape[0]}."
            )
    return {
        int(obj_id): {key: array[row] for key, array in arrays.items()}
        for row, obj_id in enumerate(obj_ids.tolist())
    }

def parse_constraint(c: Dict[str, Any]) -> Dict[str, str]:
    c = c or {}
    ctype = (c.get("ctype") or "REL").upper()
    if ctype == "ATTR":
        return {
            "ctype": "ATTR",
            "attr_key": (c.get("attr_key") or "other").lower(),
            "field_text": (c.get("attr_val_text") or "").lower(),
            "rel": "",
        }
    return {
        "ctype": "REL",
        "rel": (c.get("rel") or "near").lower(),
        "field_text": (c.get("landmark_text") or "").lower(),
        "attr_key": "",
    }

class RoundDatasetSemantic(Dataset):
    """
    Each sample:
      - query_ids [L]
      - field_ids [L] (landmark_text or attr_val_text)
      - constraint slot ids: ctype_id, rel_id, attr_id
      - img [C,D] from cache
      - mask_y [C] one-hot at target_index (weak supervision)
      - stop_y [1]
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        vocab: Vocab,
        max_len: int,
        # constraint_stoi: Dict[str,int],
        bbox_provider: PassthroughBBoxProvider,
        img_cache_dir: str,
        imgfeat_dim: int = 768,
        constraint_table: Optional[ConstraintTable] = None,
        use_imgfeat: bool = True,
        use_bbox_geom: bool = False,
        use_relative_geom: bool = False,
        use_semantic_label: bool = False,
        use_stage1_features: bool = False,
        stage1_feat_dim: int = 128,
        use_stage1_quality: bool = False,
        stage1_cache_dir: str = "",
        stage1_feature_key: str = "features_3d",
        use_stage1_cache_bboxes: bool = False,
        require_stage1_cache: bool = False,
    ) -> None:
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len
        self.bbox_provider = bbox_provider
        self.img_cache_dir = img_cache_dir
        self.imgfeat_dim = imgfeat_dim
        self.constraint_table = constraint_table
        # self.constraint_stoi = constraint_stoi
        self.use_imgfeat = use_imgfeat
        self.use_bbox_geom = use_bbox_geom
        self.use_relative_geom = use_relative_geom
        self.use_semantic_label = use_semantic_label
        self.use_stage1_features = use_stage1_features
        self.stage1_feat_dim = stage1_feat_dim
        self.use_stage1_quality = use_stage1_quality
        self.stage1_cache_dir = stage1_cache_dir
        self.stage1_feature_key = stage1_feature_key
        self.use_stage1_cache_bboxes = use_stage1_cache_bboxes
        self.require_stage1_cache = require_stage1_cache
        self._img_cache_mem: Dict[str, Optional[Dict[int, np.ndarray]]] = {}
        self._stage1_cache_mem: Dict[str, Optional[Dict[int, Dict[str, np.ndarray]]]] = {}

        # fixed id maps (keep small; you can expand later)
        self.rel2id = {
            "near": 0, "behind": 1, "in_front_of": 2, "left_of": 3,
            "right_of": 4, "next_to": 5, "above": 6, "below": 7,
            "between": 8, "closest": 9, "farthest": 10, "opposite": 11,
            "along": 12, "inside": 13, "contains": 14,
        }
        self.attr2id = {
            "color": 0, "material": 1, "shape": 2, "size": 3,
            "height": 4, "other": 5, "state": 6, "position": 7,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def _get_scene_featmap(self, scene_id: str) -> Optional[Dict[int, np.ndarray]]:
        if scene_id not in self._img_cache_mem:
            self._img_cache_mem[scene_id] = load_imgfeat_cache_npz(self.img_cache_dir, scene_id)
        return self._img_cache_mem[scene_id]

    def _get_imgfeat(self, scene_id: str, cand_ids: List[int]) -> np.ndarray:
        C = len(cand_ids)
        if not self.use_imgfeat:
            return np.zeros((C, self.imgfeat_dim), dtype=np.float32)

        m = self._get_scene_featmap(scene_id)
        out = np.zeros((C, self.imgfeat_dim), dtype=np.float32)
        if m is None:
            return out

        # allow feature dim mismatch
        any_feat = next(iter(m.values()))
        D = int(any_feat.shape[0])

        for i, cid in enumerate(cand_ids):
            v = m.get(int(cid))
            if v is None:
                continue
            if D >= self.imgfeat_dim:
                out[i] = v[: self.imgfeat_dim]
            else:
                out[i, :D] = v
        return out

    def _get_stage1_cache(self, scene_id: str) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        if scene_id not in self._stage1_cache_mem:
            self._stage1_cache_mem[scene_id] = load_stage1_cache_npz(self.stage1_cache_dir, scene_id)
        cache = self._stage1_cache_mem[scene_id]
        if cache is None and self.require_stage1_cache:
            raise FileNotFoundError(
                f"Stage1 candidate cache missing for {scene_id} under {self.stage1_cache_dir}. "
                "Run tools/export_stage1_candidate_cache.py for the checkpoint used by Stage1."
            )
        return cache

    @staticmethod
    def _from_stage1_cache(
        cache: Optional[Dict[int, Dict[str, np.ndarray]]],
        cand_ids: List[int],
        key: str,
        shape: tuple,
        dtype,
    ) -> np.ndarray:
        output = np.zeros((len(cand_ids), *shape), dtype=dtype)
        if cache is None:
            return output
        for row, candidate_id in enumerate(cand_ids):
            candidate = cache.get(int(candidate_id))
            if candidate is None or key not in candidate:
                continue
            value = np.asarray(candidate[key], dtype=dtype).reshape(-1)
            if not shape:
                if value.size:
                    output[row] = value[0]
                continue
            flat_output = output[row].reshape(-1)
            copy_count = min(flat_output.shape[0], value.shape[0])
            flat_output[:copy_count] = value[:copy_count]
        return output

    @staticmethod
    def _candidate_matrix(sample: Dict[str, Any], key: str, count: int, width: int) -> np.ndarray:
        value = sample.get(key)
        if value is None:
            return np.zeros((count, width), dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != count:
            raise ValueError(f"sample['{key}'] must be [C,D], got {array.shape}, C={count}")
        out = np.zeros((count, width), dtype=np.float32)
        copy_width = min(width, array.shape[1])
        out[:, :copy_width] = array[:, :copy_width]
        return out

    @staticmethod
    def _candidate_vector(sample: Dict[str, Any], key: str, count: int) -> np.ndarray:
        value = sample.get(key)
        if value is None:
            return np.zeros((count,), dtype=np.float32)
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape[0] != count:
            raise ValueError(f"sample['{key}'] must be [C], got {array.shape}, C={count}")
        return array

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        scene_id = s["scene_id"]
        query = (s.get("query") or "").strip()
        cand_ids = [int(x) for x in s["cand_ids"]]
        C = len(cand_ids)

        c = parse_constraint(s.get("constraint_gt") or {})
        ctype_id = 0 if c["ctype"] == "REL" else 1
        rel_id = self.rel2id.get(c["rel"], 0)
        attr_id = self.attr2id.get(c.get("attr_key", "other"), 5)
        constraint_id = self.constraint_table.id_from_constraint_gt(s.get("constraint_gt") or {})
        target_index = int(s.get("target_index", 0))

        field_text = self.constraint_table.row(constraint_id).field_text
        q_ids = self.vocab.encode(query, self.max_len)
        f_ids = self.vocab.encode(field_text, self.max_len)

        img = self._get_imgfeat(scene_id, cand_ids)  # [C,D]
        cand_mask = np.ones((C,), dtype=np.float32)
        stage1_cache = self._get_stage1_cache(scene_id) if self.stage1_cache_dir else None
        if self.require_stage1_cache and stage1_cache is not None:
            missing_ids = [candidate_id for candidate_id in cand_ids if candidate_id not in stage1_cache]
            if missing_ids:
                raise KeyError(f"Stage1 cache for {scene_id} is missing candidate ids {missing_ids[:10]}.")
            required_keys = []
            if self.use_stage1_cache_bboxes:
                required_keys.append("bboxes")
            if self.use_stage1_features:
                required_keys.append(self.stage1_feature_key)
            if self.use_stage1_quality:
                required_keys.extend(("scores", "point_counts"))
            if self.use_semantic_label:
                required_keys.append("semantic_ids")
            for candidate_id in cand_ids:
                missing_keys = [
                    key for key in required_keys
                    if key not in stage1_cache[int(candidate_id)]
                ]
                if missing_keys:
                    raise KeyError(
                        f"Stage1 cache for {scene_id}/{candidate_id} is missing fields {missing_keys}."
                    )

        # optional bbox
        # from ipdb import set_trace;set_trace()
        needs_bbox = self.use_bbox_geom or self.use_relative_geom or (
            self.use_semantic_label and s.get("cand_semantic_ids") is None
        )
        if needs_bbox:
            if self.use_stage1_cache_bboxes:
                bboxes = self._from_stage1_cache(
                    stage1_cache, cand_ids, "bboxes", (7,), np.float32
                )
            else:
                bboxes = self.bbox_provider.get_bboxes(scene_id, cand_ids, sample=s).astype(np.float32)  # [C,7]
        else:
            bboxes = np.zeros((C, 7), dtype=np.float32)
        if s.get("cand_semantic_ids") is not None:
            semantic_ids = np.asarray(s["cand_semantic_ids"], dtype=np.int64).reshape(-1)
        elif stage1_cache is not None:
            semantic_ids = self._from_stage1_cache(
                stage1_cache, cand_ids, "semantic_ids", (), np.int64
            ).reshape(-1)
        else:
            semantic_ids = np.asarray(
                bboxes[:, 6] if bboxes.shape[1] >= 7 else np.zeros(C), dtype=np.int64
            ).reshape(-1)
        if semantic_ids.shape[0] != C:
            raise ValueError(f"sample['cand_semantic_ids'] must be [C], got {semantic_ids.shape}, C={C}")
        if s.get("cand_stage1_features") is not None:
            stage1_features = self._candidate_matrix(s, "cand_stage1_features", C, self.stage1_feat_dim)
        else:
            stage1_features = self._from_stage1_cache(
                stage1_cache, cand_ids, self.stage1_feature_key, (self.stage1_feat_dim,), np.float32
            )
        if s.get("cand_stage1_scores") is not None:
            stage1_scores = self._candidate_vector(s, "cand_stage1_scores", C)
        else:
            stage1_scores = self._from_stage1_cache(
                stage1_cache, cand_ids, "scores", (), np.float32
            ).reshape(-1)
        if s.get("cand_point_counts") is not None:
            point_counts = self._candidate_vector(s, "cand_point_counts", C)
        else:
            point_counts = self._from_stage1_cache(
                stage1_cache, cand_ids, "point_counts", (), np.float32
            ).reshape(-1)
        # one-hot mask supervision
        target_index = int(s.get("target_index", 0))
        mask_y = np.zeros((C,), dtype=np.float32)
        if 0 <= target_index < C:
            mask_y[target_index] = 1.0

        out = {
            "query_ids": torch.tensor(q_ids, dtype=torch.long),
            "field_ids": torch.tensor(f_ids, dtype=torch.long),
            "c_ctype_id": torch.tensor(ctype_id, dtype=torch.long),
            "c_rel_id": torch.tensor(rel_id, dtype=torch.long),
            "c_attr_id": torch.tensor(attr_id, dtype=torch.long),
            "constraint_id": torch.tensor(constraint_id, dtype=torch.long),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "img": torch.tensor(img, dtype=torch.float32),
            "cand_mask": torch.tensor(cand_mask, dtype=torch.float32),
            "mask_y": torch.tensor(mask_y, dtype=torch.float32),
            "stop_y": torch.tensor(float(s.get("stop_y", 0)), dtype=torch.float32),

            "cand_bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "cand_semantic_ids": torch.tensor(semantic_ids, dtype=torch.long),
            "cand_stage1_features": torch.tensor(stage1_features, dtype=torch.float32),
            "cand_stage1_scores": torch.tensor(stage1_scores, dtype=torch.float32),
            "cand_point_counts": torch.tensor(point_counts, dtype=torch.float32),
        }
        return out
