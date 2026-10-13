"""Query-conditioned preference reversal loss for atomic diagnostic pairs."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualPairPreferenceLoss(nn.Module):
    def __init__(
        self, margin: float = 0.2, strict_pair_presence: bool = True,
        enabled: bool = True, weight: float = 1.0,
    ):
        super().__init__()
        self.margin = float(margin)
        self.strict_pair_presence = bool(strict_pair_presence)
        # ``enabled`` and ``weight`` live in the Hydra node and are consumed by
        # M3DRefCLIP; accepting them here keeps direct Hydra instantiation safe.
        self.config_enabled = bool(enabled)
        self.config_weight = float(weight)

    @staticmethod
    def _masked_max(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return scores.masked_fill(~mask, torch.finfo(scores.dtype).min).max()

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        pair_ids: list[list[str]],
        sample_ids: list[list[str]],
        valid_mask: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scores.ndim != 3 or labels.shape != scores.shape:
            raise ValueError("scores and labels must have matching [B,C,P] shape")
        batch_size, chunk_size, proposal_count = scores.shape
        if len(pair_ids) != batch_size or len(sample_ids) != batch_size:
            raise ValueError("pair/sample IDs must have one row per batch scene")
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch_size, proposal_count), dtype=torch.bool, device=scores.device
            )
        if valid_mask.shape != (batch_size, proposal_count):
            raise ValueError("valid_mask must have [B,P] shape")
        if sample_weights is None:
            sample_weights = torch.ones(
                (batch_size, chunk_size), dtype=scores.dtype, device=scores.device
            )
        if sample_weights.shape != (batch_size, chunk_size):
            raise ValueError("sample_weights must have [B,C] shape")

        losses = []
        weights = []
        for batch_index in range(batch_size):
            positions_by_pair: dict[str, dict[str, int]] = defaultdict(dict)
            for chunk_index, raw_pair_id in enumerate(pair_ids[batch_index]):
                current_pair_id = str(raw_pair_id or "").strip()
                if not current_pair_id:
                    continue
                current_sample_id = str(sample_ids[batch_index][chunk_index] or "").strip()
                if not current_sample_id:
                    raise ValueError("paired rows require diagnostic_sample_id")
                positions_by_pair[current_pair_id].setdefault(current_sample_id, chunk_index)
            for current_pair_id, positions_by_sample in positions_by_pair.items():
                positions = list(positions_by_sample.values())
                if len(positions) != 2:
                    if self.strict_pair_presence:
                        raise ValueError(
                            f"pair {current_pair_id!r} has {len(positions)} unique members in one scene chunk"
                        )
                    continue
                left, right = positions
                scene_valid = valid_mask[batch_index].bool()
                for positive_index, negative_index in ((left, right), (right, left)):
                    positive_mask = labels[batch_index, positive_index].bool() & scene_valid
                    negative_mask = labels[batch_index, negative_index].bool() & scene_valid
                    positive_only = positive_mask & ~negative_mask
                    negative_only = negative_mask & ~positive_mask
                    if not positive_only.any() or not negative_only.any():
                        continue
                    positive_score = self._masked_max(scores[batch_index, positive_index], positive_only)
                    negative_score = self._masked_max(scores[batch_index, positive_index], negative_only)
                    losses.append(F.softplus(self.margin - positive_score + negative_score))
                    weights.append(sample_weights[batch_index, positive_index].to(scores.dtype).clamp_min(0))
        if not losses:
            return scores.sum() * 0.0
        stacked_losses = torch.stack(losses)
        stacked_weights = torch.stack(weights)
        if not (stacked_weights > 0).any():
            return scores.sum() * 0.0
        return (stacked_losses * stacked_weights).sum() / stacked_weights.sum().clamp_min(1e-12)
