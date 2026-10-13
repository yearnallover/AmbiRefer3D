import torch
import torch.nn.functional as F


def candidate_ce_loss(mask_logits, target_index, cand_mask):
    valid = (target_index >= 0) & (target_index < mask_logits.shape[1])
    rows = torch.nonzero(valid, as_tuple=False).flatten()
    if rows.numel():
        rows = rows[cand_mask[rows, target_index[rows]] > 0]
    if not rows.numel():
        return mask_logits.sum() * 0
    logits = mask_logits.masked_fill(cand_mask <= 0, -1e9)
    return F.cross_entropy(logits[rows], target_index[rows])


def selector_ce_loss(selector_logits, constraint_id):
    return F.cross_entropy(selector_logits, constraint_id)


def stop_bce_loss(stop_logit, stop_y):
    return F.binary_cross_entropy_with_logits(stop_logit, stop_y)
