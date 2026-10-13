from __future__ import annotations
import torch
import torch.nn.functional as F

def candidate_ce_loss(mask_logits: torch.Tensor, target_index: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    # mask_logits [B,C], target_index [B]
    logits = mask_logits.masked_fill(cand_mask <= 0, -1e9)
    valid = (target_index >= 0) & (target_index < logits.shape[1])
    if valid.any():
        safe_target = target_index[valid].long()
        valid = valid.clone()
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        valid = cand_mask[valid_indices, safe_target] > 0
        valid_indices = valid_indices[valid]
        if valid_indices.numel() > 0:
            return F.cross_entropy(logits[valid_indices], target_index[valid_indices].long())
    return logits.sum() * 0.0


def candidate_filter_loss(
    mask_logits: torch.Tensor,
    target_index: torch.Tensor,
    cand_mask: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    ce_weight: float = 0.5,
    bce_weight: float = 1.0,
    positive_weight: float = 2.0,
    negative_weight: float = 0.25,
    decision_threshold: float = 0.5,
) -> torch.Tensor:
    """Train both candidate ranking and the absolute rollout threshold."""
    if not 0.0 < float(decision_threshold) < 1.0:
        raise ValueError("decision_threshold must be in (0, 1).")
    valid_mask = cand_mask.bool()
    if target_mask is None:
        target_mask = torch.zeros_like(valid_mask)
        valid_target = (target_index >= 0) & (target_index < mask_logits.shape[1])
        if valid_target.any():
            rows = torch.nonzero(valid_target, as_tuple=False).squeeze(1)
            target_mask[rows, target_index[rows].long()] = True
    else:
        target_mask = target_mask.bool()
    target_mask = target_mask & valid_mask
    valid_rows = valid_mask.any(dim=1) & target_mask.any(dim=1)
    if not valid_rows.any():
        return mask_logits.sum() * 0.0

    logits = mask_logits[valid_rows]
    valid = valid_mask[valid_rows]
    targets = target_mask[valid_rows]
    negatives = valid & ~targets

    boundary = torch.logit(
        logits.new_tensor(float(decision_threshold)).clamp(1e-6, 1.0 - 1e-6)
    )
    decision_logits = logits - boundary
    positive_bce = (F.softplus(-decision_logits) * targets.float()).sum(dim=1)
    positive_bce = positive_bce / targets.sum(dim=1).clamp_min(1)
    negative_bce = (F.softplus(decision_logits) * negatives.float()).sum(dim=1)
    negative_bce = negative_bce / negatives.sum(dim=1).clamp_min(1)
    calibrated = (
        float(positive_weight) * positive_bce
        + float(negative_weight) * negative_bce
    ).mean()

    min_value = torch.finfo(logits.dtype).min
    denominator = torch.logsumexp(logits.masked_fill(~valid, min_value), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~targets, min_value), dim=1)
    ranking = (denominator - numerator).mean()
    return float(bce_weight) * calibrated + float(ce_weight) * ranking

def selector_ce_loss(
    selector_logits: torch.Tensor,
    constraint_id: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    target_distribution: torch.Tensor | None = None,
) -> torch.Tensor:
    if target_distribution is not None:
        targets = target_distribution.float().clamp_min(0.0)
        normalizer = targets.sum(dim=1, keepdim=True)
        valid_rows = normalizer.squeeze(1) > 0
        if not valid_rows.any():
            return selector_logits.sum() * 0.0
        targets = targets[valid_rows] / normalizer[valid_rows].clamp_min(1e-12)
        log_probability = F.log_softmax(selector_logits[valid_rows], dim=1)
        return -(targets * log_probability).sum(dim=1).mean()

    if target_mask is None:
        return F.cross_entropy(selector_logits, constraint_id)

    targets = target_mask.bool()
    valid_rows = targets.any(dim=1)
    if not valid_rows.any():
        return selector_logits.sum() * 0.0
    logits = selector_logits[valid_rows]
    targets = targets[valid_rows]
    min_value = torch.finfo(logits.dtype).min
    denominator = torch.logsumexp(logits, dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~targets, min_value), dim=1)
    return (denominator - numerator).mean()

def stop_bce_loss(stop_logit: torch.Tensor, stop_y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(stop_logit, stop_y)
