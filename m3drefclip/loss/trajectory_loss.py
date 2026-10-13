import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot_from_index(target_index, num_candidates, cand_mask):
    labels = torch.zeros(
        size=(target_index.shape[0], num_candidates),
        dtype=torch.float32,
        device=target_index.device,
    )
    valid_target = (target_index >= 0) & (target_index < num_candidates)
    if valid_target.any():
        labels[valid_target].scatter_(1, target_index[valid_target].unsqueeze(1), 1.0)
    return labels * cand_mask.float()


def _prepare_logits_labels(logits, labels=None, target_index=None, valid_mask=None):
    if logits.dim() == 3:
        logits = logits.flatten(0, 1)
        if labels is not None:
            labels = labels.flatten(0, 1)
        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(1).expand(-1, logits.shape[0] // valid_mask.shape[0], -1).flatten(0, 1)

    if valid_mask is None:
        valid_mask = torch.ones_like(logits, dtype=torch.bool)
    else:
        valid_mask = valid_mask.bool()

    if labels is None:
        labels = _one_hot_from_index(target_index.long(), logits.shape[1], valid_mask)
    else:
        labels = labels.float()

    labels = labels * valid_mask.float()
    valid_rows = valid_mask.any(dim=1) & labels.any(dim=1)
    return logits, labels, valid_mask, valid_rows


class PreferenceRankingLoss(pl.LightningModule):
    def __init__(self, margin=0.2, temperature=1.0, hard_negative_topk=0, **kwargs):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.hard_negative_topk = hard_negative_topk

    def forward(self, logits, labels=None, target_index=None, valid_mask=None):
        logits, labels, valid_mask, valid_rows = _prepare_logits_labels(logits, labels, target_index, valid_mask)
        if not valid_rows.any():
            return logits.sum() * 0.0

        logits = logits[valid_rows]
        labels = labels[valid_rows].bool()
        valid_mask = valid_mask[valid_rows]
        losses = []
        for row_logits, row_labels, row_valid in zip(logits, labels, valid_mask):
            pos_logits = row_logits[row_labels & row_valid]
            neg_logits = row_logits[(~row_labels) & row_valid]
            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue
            if self.hard_negative_topk and neg_logits.numel() > self.hard_negative_topk:
                neg_logits = neg_logits.topk(self.hard_negative_topk).values
            pairwise = (self.margin - pos_logits.unsqueeze(1) + neg_logits.unsqueeze(0)) / self.temperature
            losses.append(F.softplus(pairwise).mean())

        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class MetricDrivenLoss(pl.LightningModule):
    def __init__(self, rank_temperature=0.2, f1_weight=1.0, rank_weight=1.0, eps=1e-6, **kwargs):
        super().__init__()
        self.rank_temperature = rank_temperature
        self.f1_weight = f1_weight
        self.rank_weight = rank_weight
        self.eps = eps

    def forward(self, logits, labels=None, target_index=None, valid_mask=None):
        logits, labels, valid_mask, valid_rows = _prepare_logits_labels(logits, labels, target_index, valid_mask)
        if not valid_rows.any():
            return logits.sum() * 0.0

        logits = logits[valid_rows]
        labels = labels[valid_rows]
        valid_mask = valid_mask[valid_rows].float()
        probs = torch.sigmoid(logits) * valid_mask

        soft_tp = (probs * labels).sum(dim=1)
        soft_pred = probs.sum(dim=1)
        soft_gt = labels.sum(dim=1)
        soft_f1 = (2.0 * soft_tp + self.eps) / (soft_pred + soft_gt + self.eps)
        f1_loss = 1.0 - soft_f1.mean()

        rank_losses = []
        bool_labels = labels.bool()
        bool_valid = valid_mask.bool()
        for row_logits, row_labels, row_valid in zip(logits, bool_labels, bool_valid):
            pos_logits = row_logits[row_labels & row_valid]
            neg_logits = row_logits[(~row_labels) & row_valid]
            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue
            soft_violations = torch.sigmoid(
                (neg_logits.unsqueeze(0) - pos_logits.unsqueeze(1)) / self.rank_temperature
            )
            rank_losses.append(soft_violations.mean())
        if rank_losses:
            rank_loss = torch.stack(rank_losses).mean()
        else:
            rank_loss = logits.sum() * 0.0

        return self.f1_weight * f1_loss + self.rank_weight * rank_loss
