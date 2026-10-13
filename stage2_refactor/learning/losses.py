from __future__ import annotations
import torch
import torch.nn.functional as F

def candidate_ce_loss(mask_logits: torch.Tensor, target_index: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    # mask_logits [B,C], target_index [B]
    logits = mask_logits.masked_fill(cand_mask <= 0, -1e9)
    valid = (target_index >= 0) & (target_index < logits.shape[1])
    if valid.any():
        safe_target = target_index[valid].long()
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        valid = cand_mask[valid_indices, safe_target] > 0
        valid_indices = valid_indices[valid]
        if valid_indices.numel() > 0:
            return F.cross_entropy(logits[valid_indices], target_index[valid_indices].long())
    return logits.sum() * 0.0

def selector_ce_loss(selector_logits: torch.Tensor, constraint_id: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(selector_logits, constraint_id)

def stop_bce_loss(stop_logit: torch.Tensor, stop_y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(stop_logit, stop_y)
