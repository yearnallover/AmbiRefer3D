from __future__ import annotations
from typing import Dict, List
import torch

def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # query
    query_ids = torch.stack([b["query_ids"] for b in batch], dim=0)  # [B,L]

    # candidates: pad to maxC
    C_list = [b["geom"].shape[0] for b in batch]
    maxC = max(C_list)
    B = len(batch)

    def pad_2d(x: torch.Tensor, T: int, value: float = 0.0) -> torch.Tensor:
        # x [C,D]
        C, D = x.shape
        if C == T:
            return x
        out = x.new_full((T, D), value)
        out[:C, :] = x
        return out

    def pad_3d(x: torch.Tensor, T: int, value: float = 0.0) -> torch.Tensor:
        # x [C,C,Dr] -> pad to [T,T,Dr]
        C, _, Dr = x.shape
        if C == T:
            return x
        out = x.new_full((T, T, Dr), value)
        out[:C, :C, :] = x
        return out

    geom = torch.stack([pad_2d(b["geom"], maxC) for b in batch], dim=0)         # [B,maxC,Dg]
    img = torch.stack([pad_2d(b["img"], maxC) for b in batch], dim=0)           # [B,maxC,512]
    rel = torch.stack([pad_3d(b["rel"], maxC) for b in batch], dim=0)           # [B,maxC,maxC,Dr]
    cand_mask = torch.stack([pad_2d(b["cand_mask"].unsqueeze(-1), maxC).squeeze(-1) for b in batch], dim=0)  # [B,maxC]
    mask_y = torch.stack([pad_2d(b["mask_y"].unsqueeze(-1), maxC).squeeze(-1) for b in batch], dim=0)        # [B,maxC]

    # constraint params
    keys_long = ["c_ctype","c_axis","c_side","c_metric","c_extreme","c_prox_mode","c_nearfar","c_ref_index"]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys_long}

    # round state
    out["query_ids"] = query_ids
    out["geom"] = geom
    out["img"] = img
    out["rel"] = rel
    out["cand_mask"] = cand_mask
    out["mask_y"] = mask_y
    out["stop_y"] = torch.stack([b["stop_y"] for b in batch], dim=0)  # [B]
    out["used_type_mask"] = torch.stack([b["used_type_mask"] for b in batch], dim=0)  # [B,3]
    return out
