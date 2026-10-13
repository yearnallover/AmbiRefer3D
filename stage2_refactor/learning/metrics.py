from __future__ import annotations
import torch

def should_stop_rule(prob: torch.Tensor, cand_mask: torch.Tensor, top1_thr: float, gap_thr: float) -> torch.Tensor:
    """
    prob: [B,C] after sigmoid+masked
    returns stop_pred [B] bool
    """
    prob = prob.masked_fill(cand_mask <= 0, -1.0)
    topv, topi = prob.max(dim=-1)
    # second best
    prob2 = prob.clone()
    prob2.scatter_(1, topi.unsqueeze(1), -1.0)
    second, _ = prob2.max(dim=-1)
    gap = topv - second
    return (topv >= top1_thr) & (gap >= gap_thr)

def mask_accuracy_top1(mask_logits: torch.Tensor, mask_y: torch.Tensor, cand_mask: torch.Tensor) -> float:
    """
    Pick argmax candidate; success if it's a positive satisfy-candidate in mask_y.
    """
    logits = mask_logits.masked_fill(cand_mask <= 0, -1e9)
    pred = logits.argmax(dim=-1)  # [B]
    B = logits.size(0)
    hit = 0
    for b in range(B):
        if mask_y[b, pred[b]] > 0.5:
            hit += 1
    return hit / max(B, 1)
