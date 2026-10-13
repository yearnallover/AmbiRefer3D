"""Lightweight pair-aware packing helpers for diagnostic language metadata."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def pack_pair_aware_indices(
    language_rows: list[dict[str, Any]], chunk_size: int, rng: random.Random | None = None
) -> list[list[int]]:
    """Pack rows without splitting a non-empty counterfactual pair across chunks.

    Padding repeats indices from within the same chunk.  Downstream pair loss
    deduplicates by ``diagnostic_sample_id``, so repeated padding cannot create
    an artificial third pair member.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not language_rows:
        return []
    groups: dict[str, list[int]] = defaultdict(list)
    singletons: list[list[int]] = []
    for index, row in enumerate(language_rows):
        pair_id = str(row.get("counterfactual_pair_id") or "").strip()
        if pair_id:
            groups[pair_id].append(index)
        else:
            singletons.append([index])
    malformed = {pair_id: len(indices) for pair_id, indices in groups.items() if len(indices) != 2}
    if malformed:
        raise ValueError(f"counterfactual metadata contains non-atomic pairs; first={list(sorted(malformed.items()))[:5]}")
    units = list(groups.values()) + singletons
    if any(len(unit) > chunk_size for unit in units):
        raise ValueError("counterfactual unit is larger than chunk_size")
    (rng or random).shuffle(units)

    bins: list[list[int]] = []
    for unit in units:
        destination = next((bucket for bucket in bins if len(bucket) + len(unit) <= chunk_size), None)
        if destination is None:
            destination = []
            bins.append(destination)
        destination.extend(unit)
    for bucket in bins:
        original = list(bucket)
        while len(bucket) < chunk_size:
            bucket.append(original[(len(bucket) - len(original)) % len(original)])
    return bins
