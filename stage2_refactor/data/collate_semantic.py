from __future__ import annotations
from typing import Dict, List
import torch

def collate_semantic(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    C_list = [b["img"].shape[0] for b in batch]
    maxC = max(C_list)
    D = batch[0]["img"].shape[1]

    def pad_2d(x: torch.Tensor, T: int, value: float = 0.0) -> torch.Tensor:
        C, DD = x.shape
        out = x.new_full((T, DD), value)
        out[:C] = x
        return out

    def pad_1d(x: torch.Tensor, T: int, value: float = 0.0) -> torch.Tensor:
        C = x.shape[0]
        out = x.new_full((T,), value)
        out[:C] = x
        return out

    out = {
        "query_ids": torch.stack([b["query_ids"] for b in batch], dim=0),   # [B,L]
        "field_ids": torch.stack([b["field_ids"] for b in batch], dim=0),   # [B,L]
        "c_ctype_id": torch.stack([b["c_ctype_id"] for b in batch], dim=0), # [B]
        "c_rel_id": torch.stack([b["c_rel_id"] for b in batch], dim=0),
        "c_attr_id": torch.stack([b["c_attr_id"] for b in batch], dim=0),
        "stop_y": torch.stack([b["stop_y"] for b in batch], dim=0),         # [B]
        "constraint_id": torch.stack([b["constraint_id"] for b in batch], dim=0),  # [B]
        "target_index": torch.stack([b["target_index"] for b in batch], dim=0),    # [B]

        "img": torch.stack([pad_2d(b["img"], maxC) for b in batch], dim=0),  # [B,maxC,D]
        "cand_mask": torch.stack([pad_1d(b["cand_mask"], maxC) for b in batch], dim=0), # [B,maxC]
        "mask_y": torch.stack([pad_1d(b["mask_y"], maxC) for b in batch], dim=0),       # [B,maxC]
        "cand_bboxes": torch.stack([pad_2d(b["cand_bboxes"], maxC) for b in batch], dim=0), # [B,maxC,7]
        "cand_semantic_ids": torch.stack(
            [pad_1d(b["cand_semantic_ids"], maxC) for b in batch], dim=0
        ),
        "cand_stage1_features": torch.stack(
            [pad_2d(b["cand_stage1_features"], maxC) for b in batch], dim=0
        ),
        "cand_stage1_scores": torch.stack(
            [pad_1d(b["cand_stage1_scores"], maxC) for b in batch], dim=0
        ),
        "cand_point_counts": torch.stack(
            [pad_1d(b["cand_point_counts"], maxC) for b in batch], dim=0
        ),
    }
    return out
