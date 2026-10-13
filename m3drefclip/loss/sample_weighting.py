"""Shared reduction for per-language supervision weights."""

from __future__ import annotations

import torch


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Return a finite non-negative weighted mean without changing legacy behavior."""
    if values.ndim != 1:
        raise ValueError("weighted_mean expects a one-dimensional loss tensor")
    if weights is None:
        return values.mean() if values.numel() else values.sum() * 0.0
    weights = weights.reshape(-1).to(device=values.device, dtype=values.dtype)
    if weights.shape != values.shape:
        raise ValueError("sample weights must match the number of per-query losses")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("sample weights must be finite and non-negative")
    positive = weights > 0
    if not positive.any():
        return values.sum() * 0.0
    return (values[positive] * weights[positive]).sum() / weights[positive].sum()
