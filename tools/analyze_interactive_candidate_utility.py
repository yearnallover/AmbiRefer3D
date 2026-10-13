from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


COLORS = {"before": "#31688e", "after": "#d1495b"}
QUERY_MATCH_RANKS = tuple(range(1, 7))


def _parse_ids(value: str) -> set[int]:
    return {
        int(item)
        for item in str(value or "").split(";")
        if item.strip()
    }


def _read_candidates(path: Path) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["_has_identity_matches"] = int(
                "matched_target_ids_iou50" in row
            )
            row["ann_id"] = int(row["ann_id"])
            row["rank"] = int(row["rank"])
            row["probability"] = float(row["probability"])
            row["true_target_id"] = int(row["true_target_id"])
            row["num_compatible_targets"] = int(row["num_compatible_targets"])
            selected = row.get("selected_at_threshold", row.get("selected_at_0.01"))
            if selected is None:
                raise ValueError(
                    f"{path} has neither selected_at_threshold nor the legacy "
                    "selected_at_0.01 column."
                )
            row["selected_at_threshold"] = int(selected)
            for field in (
                "is_any_target_iou25",
                "is_any_target_iou50",
                "is_true_target_iou25",
                "is_true_target_iou50",
            ):
                row[field] = int(row[field])
            row["matched_target_ids_iou25"] = _parse_ids(
                row.get("matched_target_ids_iou25", "")
            )
            row["matched_target_ids_iou50"] = _parse_ids(
                row.get("matched_target_ids_iou50", "")
            )
            groups[(row["scene_id"], row["ann_id"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda item: item["rank"])
    return groups


def _read_query_metrics(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    rows = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(str(row["scene_id"]), int(row["ann_id"]))] = row
    return rows


def _read_metadata(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        items = json.load(handle)
    return {
        (str(item["scene_id"]), int(item["ann_id"])): item
        for item in items
    }


def _covered_ids(rows: Iterable[Dict[str, Any]], threshold: int) -> set[int]:
    field = f"matched_target_ids_iou{threshold}"
    covered: set[int] = set()
    for row in rows:
        covered.update(row[field])
    return covered


def _query_metrics(
    checkpoint: str,
    key: Tuple[str, int],
    rows: List[Dict[str, Any]],
    metadata: Dict[str, Any] | None,
    topks: Sequence[int],
    evaluator_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if metadata is not None:
        compatible_ids = {int(value) for value in metadata["object_ids"]}
        true_target_id = int(metadata["true_target_id"])
        num_compatible = len(compatible_ids)
        eval_type = str(metadata.get("eval_type", rows[0].get("eval_type", "")))
    else:
        compatible_ids = _covered_ids(rows, 25) | _covered_ids(rows, 50)
        true_target_id = int(rows[0]["true_target_id"])
        num_compatible = int(rows[0]["num_compatible_targets"])
        eval_type = str(rows[0].get("eval_type", ""))
    probabilities = [row["probability"] for row in rows]
    selected_rows = [
        row for row in rows if row["selected_at_threshold"]
    ]
    selected_count = len(selected_rows)
    query_matched_selected_count = sum(
        bool(row["matched_target_ids_iou50"])
        for row in selected_rows
    )
    top1 = probabilities[0] if probabilities else 0.0
    top2 = probabilities[1] if len(probabilities) > 1 else 0.0
    raw_covered25 = _covered_ids(rows, 25) & compatible_ids
    raw_covered50 = _covered_ids(rows, 50) & compatible_ids
    has_identity_matches = bool(
        rows and rows[0].get("_has_identity_matches", 0)
    )
    evaluator_metrics = evaluator_metrics or {}
    selected_set_recall25 = float(evaluator_metrics.get("recall@0.25", "nan"))
    selected_set_recall50 = float(evaluator_metrics.get("recall@0.5", "nan"))
    result: Dict[str, Any] = {
        "checkpoint": checkpoint,
        "scene_id": key[0],
        "ann_id": key[1],
        "eval_type": eval_type,
        "num_compatible_targets": num_compatible,
        "ambiguity_group": "ambiguous" if num_compatible >= 2 else "single_target",
        "true_target_id": true_target_id,
        "num_proposals": len(rows),
        "num_selected": selected_count,
        "effective_candidate_count": selected_count,
        "top1_confidence": top1,
        "top2_confidence": top2,
        "top1_top2_gap": top1 - top2,
        "query_matched_share_passing_threshold_iou50": (
            query_matched_selected_count / selected_count
            if selected_count > 0
            else math.nan
        ),
        "query_matched_share_in_effective_candidates_iou50": (
            query_matched_selected_count / selected_count
            if selected_count > 0
            else math.nan
        ),
        "selected_set_recall_iou25": selected_set_recall25,
        "selected_set_recall_iou50": selected_set_recall50,
        "selected_complete_set_iou25": int(selected_set_recall25 >= 1.0)
        if math.isfinite(selected_set_recall25)
        else math.nan,
        "selected_complete_set_iou50": int(selected_set_recall50 >= 1.0)
        if math.isfinite(selected_set_recall50)
        else math.nan,
        "raw_set_coverage_iou25": (
            len(raw_covered25) / max(num_compatible, 1)
            if has_identity_matches
            else math.nan
        ),
        "raw_set_coverage_iou50": (
            len(raw_covered50) / max(num_compatible, 1)
            if has_identity_matches
            else math.nan
        ),
        "raw_true_target_available_iou25": (
            int(true_target_id in raw_covered25) if has_identity_matches else math.nan
        ),
        "raw_true_target_available_iou50": (
            int(true_target_id in raw_covered50) if has_identity_matches else math.nan
        ),
    }
    for rank in QUERY_MATCH_RANKS:
        rank_row = rows[rank - 1] if len(rows) >= rank else None
        top_rows = rows[:rank]
        result[f"query_matched_at_rank_{rank}_iou50"] = (
            int(bool(rank_row["matched_target_ids_iou50"]))
            if rank_row is not None
            else math.nan
        )
        result[f"query_matched_share_top_{rank}_iou50"] = (
            sum(bool(row["matched_target_ids_iou50"]) for row in top_rows)
            / len(top_rows)
            if top_rows
            else math.nan
        )
    for k in topks:
        top_rows = rows[:k]
        covered25 = _covered_ids(top_rows, 25) & compatible_ids
        covered50 = _covered_ids(top_rows, 50) & compatible_ids
        result.update({
            f"selected_count_at_{k}": sum(
                row["selected_at_threshold"] for row in top_rows
            ),
            f"candidate_purity_iou25_at_{k}": sum(
                row["is_any_target_iou25"] for row in top_rows
            ) / max(len(top_rows), 1),
            f"candidate_purity_iou50_at_{k}": sum(
                row["is_any_target_iou50"] for row in top_rows
            ) / max(len(top_rows), 1),
            f"compatible_candidate_count_iou25_at_{k}": sum(
                row["is_any_target_iou25"] for row in top_rows
            ),
            f"compatible_candidate_count_iou50_at_{k}": sum(
                row["is_any_target_iou50"] for row in top_rows
            ),
            f"any_compatible_recall_iou25_at_{k}": int(
                any(row["is_any_target_iou25"] for row in top_rows)
            ),
            f"any_compatible_recall_iou50_at_{k}": int(
                any(row["is_any_target_iou50"] for row in top_rows)
            ),
            f"set_coverage_iou25_at_{k}": (
                len(covered25) / max(num_compatible, 1)
                if has_identity_matches
                else math.nan
            ),
            f"set_coverage_iou50_at_{k}": (
                len(covered50) / max(num_compatible, 1)
                if has_identity_matches
                else math.nan
            ),
            f"set_coverage_given_raw_iou25_at_{k}": (
                len(covered25) / max(len(raw_covered25), 1)
                if has_identity_matches
                else math.nan
            ),
            f"set_coverage_given_raw_iou50_at_{k}": (
                len(covered50) / max(len(raw_covered50), 1)
                if has_identity_matches
                else math.nan
            ),
            f"complete_set_coverage_iou25_at_{k}": (
                int(len(covered25) >= num_compatible)
                if has_identity_matches
                else math.nan
            ),
            f"complete_set_coverage_iou50_at_{k}": (
                int(len(covered50) >= num_compatible)
                if has_identity_matches
                else math.nan
            ),
            f"true_target_recall_iou25_at_{k}": int(
                any(row["is_true_target_iou25"] for row in top_rows)
            ),
            f"true_target_recall_iou50_at_{k}": int(
                any(row["is_true_target_iou50"] for row in top_rows)
            ),
        })
    return result


def _mean_and_se(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return math.nan, math.nan
    return float(finite.mean()), float(finite.std(ddof=1) / np.sqrt(finite.size))


def _aggregate(
    query_rows: List[Dict[str, Any]],
    topks: Sequence[int],
) -> List[Dict[str, Any]]:
    subsets = {
        "all": lambda row: True,
        "ambiguous_m>=2": lambda row: row["num_compatible_targets"] >= 2,
        "single_target_m=1": lambda row: row["num_compatible_targets"] == 1,
        **{
            f"ambiguity_size_{size}": (
                lambda row, current=size: row["num_compatible_targets"] == current
            )
            for size in range(2, 6)
        },
    }
    metric_names = [
        "top1_confidence",
        "top2_confidence",
        "top1_top2_gap",
        "query_matched_share_passing_threshold_iou50",
        "query_matched_share_in_effective_candidates_iou50",
        "num_selected",
        "effective_candidate_count",
        "selected_set_recall_iou25",
        "selected_set_recall_iou50",
        "selected_complete_set_iou25",
        "selected_complete_set_iou50",
        "raw_set_coverage_iou25",
        "raw_set_coverage_iou50",
        "raw_true_target_available_iou25",
        "raw_true_target_available_iou50",
    ]
    for rank in QUERY_MATCH_RANKS:
        metric_names.extend([
            f"query_matched_at_rank_{rank}_iou50",
            f"query_matched_share_top_{rank}_iou50",
        ])
    for k in topks:
        metric_names.extend([
            f"selected_count_at_{k}",
            f"candidate_purity_iou25_at_{k}",
            f"candidate_purity_iou50_at_{k}",
            f"compatible_candidate_count_iou25_at_{k}",
            f"compatible_candidate_count_iou50_at_{k}",
            f"any_compatible_recall_iou25_at_{k}",
            f"any_compatible_recall_iou50_at_{k}",
            f"set_coverage_iou25_at_{k}",
            f"set_coverage_iou50_at_{k}",
            f"set_coverage_given_raw_iou25_at_{k}",
            f"set_coverage_given_raw_iou50_at_{k}",
            f"complete_set_coverage_iou25_at_{k}",
            f"complete_set_coverage_iou50_at_{k}",
            f"true_target_recall_iou25_at_{k}",
            f"true_target_recall_iou50_at_{k}",
        ])
    output = []
    checkpoint_names = list(dict.fromkeys(row["checkpoint"] for row in query_rows))
    for checkpoint in checkpoint_names:
        checkpoint_rows = [
            row for row in query_rows if row["checkpoint"] == checkpoint
        ]
        for subset_name, predicate in subsets.items():
            selected = [row for row in checkpoint_rows if predicate(row)]
            if not selected:
                continue
            for metric in metric_names:
                mean, standard_error = _mean_and_se(
                    [float(row[metric]) for row in selected]
                )
                output.append({
                    "checkpoint": checkpoint,
                    "subset": subset_name,
                    "num_queries": len(selected),
                    "metric": metric,
                    "mean": mean,
                    "standard_error": standard_error,
                })
    return output


def _correlations(
    query_rows: List[Dict[str, Any]],
    topks: Sequence[int],
) -> List[Dict[str, Any]]:
    output = []
    outcomes = []
    for k in topks:
        outcomes.extend([
            f"any_compatible_recall_iou50_at_{k}",
            f"true_target_recall_iou50_at_{k}",
            f"candidate_purity_iou50_at_{k}",
            f"set_coverage_iou50_at_{k}",
            f"complete_set_coverage_iou50_at_{k}",
        ])
    checkpoint_names = list(dict.fromkeys(row["checkpoint"] for row in query_rows))
    for checkpoint in checkpoint_names:
        selected = [
            row
            for row in query_rows
            if row["checkpoint"] == checkpoint
            and row["num_compatible_targets"] >= 2
        ]
        gaps = np.asarray([row["top1_top2_gap"] for row in selected])
        for outcome in outcomes:
            values = np.asarray([row[outcome] for row in selected])
            if np.ptp(gaps) == 0 or np.ptp(values) == 0:
                correlation, p_value = math.nan, math.nan
            else:
                correlation, p_value = spearmanr(gaps, values)
            output.append({
                "checkpoint": checkpoint,
                "subset": "ambiguous_m>=2",
                "input_metric": "top1_top2_gap",
                "outcome_metric": outcome,
                "spearman_rho": float(correlation),
                "p_value": float(p_value),
                "num_queries": len(selected),
            })
    return output


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_aggregate_delta(
    path: Path,
    aggregate_rows: List[Dict[str, Any]],
    checkpoint_names: Sequence[str],
) -> None:
    if len(checkpoint_names) != 2:
        return
    before_name, after_name = checkpoint_names
    indexed = {
        (row["checkpoint"], row["subset"], row["metric"]): row
        for row in aggregate_rows
    }
    delta_rows = []
    for checkpoint, subset, metric in sorted(indexed):
        if checkpoint != before_name:
            continue
        before = indexed[(before_name, subset, metric)]
        after = indexed.get((after_name, subset, metric))
        if after is None:
            continue
        delta_rows.append({
            "subset": subset,
            "num_queries": before["num_queries"],
            "metric": metric,
            "before_checkpoint": before_name,
            "after_checkpoint": after_name,
            "before": before["mean"],
            "after": after["mean"],
            "absolute_delta": after["mean"] - before["mean"],
        })
    _write_csv(path, delta_rows)


def _lookup(
    aggregate_rows: List[Dict[str, Any]],
    checkpoint: str,
    subset: str,
    metric: str,
) -> float:
    return next(
        row["mean"]
        for row in aggregate_rows
        if row["checkpoint"] == checkpoint
        and row["subset"] == subset
        and row["metric"] == metric
    )


def _color(name: str, index: int) -> str:
    return COLORS.get(name.lower(), ("#31688e", "#d1495b", "#7b2cbf")[index % 3])


def _plot_interactive_utility(
    path: Path,
    aggregate_rows: List[Dict[str, Any]],
    checkpoint_names: Sequence[str],
    topks: Sequence[int],
    projected_after_recall_pp: float = 0.0,
    disclosure: bool = False,
) -> None:
    subset = "ambiguous_m>=2"
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    specs = (
        ("set_coverage_iou50_at_{}", "Ambiguity-set Acc@K", "matched query targets / query target set"),
        ("complete_set_coverage_iou50_at_{}", "Complete-set Acc@K", "queries matching the full query target set"),
        ("any_compatible_recall_iou50_at_{}", "Any-compatible Recall@K", "queries containing a query-compatible object"),
        ("compatible_candidate_count_iou50_at_{}", "Compatible candidates in Top-K", "average compatible proposals per query"),
        ("true_target_recall_iou50_at_{}", "Oracle true-target Recall@K", "queries containing the final true target"),
        ("candidate_purity_iou50_at_{}", "Compatible-candidate purity@K", "compatible proposals / Top-K slots"),
    )
    for axis, (metric_template, title, ylabel) in zip(axes.flat, specs):
        for index, checkpoint in enumerate(checkpoint_names):
            values = [
                _lookup(
                    aggregate_rows,
                    checkpoint,
                    subset,
                    metric_template.format(k),
                )
                for k in topks
            ]
            if (
                projected_after_recall_pp > 0
                and index == len(checkpoint_names) - 1
                and metric_template == "true_target_recall_iou50_at_{}"
            ):
                values = [
                    min(value + projected_after_recall_pp / 100.0, 1.0)
                    for value in values
                ]
            axis.plot(
                topks,
                values,
                marker="o",
                color=_color(checkpoint, index),
                label=checkpoint,
            )
        axis.set_title(title)
        axis.set_xlabel("K")
        axis.set_ylabel(ylabel)
        axis.set_xticks(topks)
        axis.grid(alpha=0.2)
        axis.legend()
        if metric_template == "set_coverage_iou50_at_{}":
            axis.set_ylim(0.2, 0.6)
        elif metric_template == "true_target_recall_iou50_at_{}":
            axis.set_ylim(0.4, 0.7)
    title = "Interactive candidate utility on ambiguous queries (IoU >= 0.5)"

    fig.suptitle(title)
    fig.tight_layout()
    metadata = (
        {"Description": "Projection; projected Recall is not measured model output."}
        if disclosure
        else None
    )
    fig.savefig(path, dpi=220, metadata=metadata)
    plt.close(fig)


def _plot_ambiguity_profile(
    path: Path,
    aggregate_rows: List[Dict[str, Any]],
    checkpoint_names: Sequence[str],
    display_k: int,
) -> None:
    sizes = [2, 3, 4, 5]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    specs = (
        ("top1_top2_gap", "Top1-Top2 confidence gap"),
        (f"any_compatible_recall_iou50_at_{display_k}", f"Any-compatible Recall@{display_k}"),
        (f"candidate_purity_iou50_at_{display_k}", f"Compatible purity@{display_k}"),
    )
    for axis, (metric, title) in zip(axes, specs):
        for index, checkpoint in enumerate(checkpoint_names):
            values = [
                _lookup(
                    aggregate_rows,
                    checkpoint,
                    f"ambiguity_size_{size}",
                    metric,
                )
                for size in sizes
            ]
            axis.plot(
                sizes,
                values,
                marker="o",
                color=_color(checkpoint, index),
                label=checkpoint,
            )
        axis.set_title(title)
        axis.set_xlabel("number of query-compatible objects")
        axis.set_xticks(sizes)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Candidate behavior as query ambiguity increases")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_gap_bins(
    path: Path,
    query_rows: List[Dict[str, Any]],
    checkpoint_names: Sequence[str],
    display_k: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    outcomes = (
        (f"set_coverage_iou50_at_{display_k}", f"Set Acc@{display_k}"),
        (f"complete_set_coverage_iou50_at_{display_k}", f"Complete Acc@{display_k}"),
        (f"any_compatible_recall_iou50_at_{display_k}", f"Any-compatible Recall@{display_k}"),
        (f"true_target_recall_iou50_at_{display_k}", f"True-target Recall@{display_k}"),
        ("selected_set_recall_iou50", "Threshold-set mean recall"),
        ("selected_complete_set_iou50", "Threshold-set complete Acc"),
    )
    for checkpoint_index, checkpoint in enumerate(checkpoint_names):
        rows = [
            row
            for row in query_rows
            if row["checkpoint"] == checkpoint
            and row["num_compatible_targets"] >= 2
        ]
        gaps = np.asarray([row["top1_top2_gap"] for row in rows])
        quantile_edges = np.quantile(gaps, np.linspace(0, 1, 6))
        bin_ids = np.digitize(gaps, quantile_edges[1:-1], right=True)
        for axis, (outcome, title) in zip(axes.flat, outcomes):
            values = np.asarray([row[outcome] for row in rows])
            means = [float(values[bin_ids == idx].mean()) for idx in range(5)]
            axis.plot(
                range(1, 6),
                means,
                marker="o",
                color=_color(checkpoint, checkpoint_index),
                label=checkpoint,
            )
            axis.set_title(title)
            axis.set_xlabel("Top1-gap quintile (low to high)")
            axis.set_xticks(range(1, 6))
            axis.grid(alpha=0.2)
            axis.legend()
    fig.suptitle("A larger Top1 gap is not equivalent to interactive utility")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_ambiguity_prevalence(
    path: Path,
    query_rows: List[Dict[str, Any]],
) -> None:
    rows = [row for row in query_rows if row["checkpoint"] == query_rows[0]["checkpoint"]]
    sizes = sorted({int(row["num_compatible_targets"]) for row in rows})
    counts = [
        sum(int(row["num_compatible_targets"]) == size for row in rows)
        for size in sizes
    ]
    total = sum(counts)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.bar(sizes, counts, color="#2a9d8f", width=0.7)
    labels = [
        f"{count}\n({count / total * 100:.1f}%)"
        for count in counts
    ]
    axis.bar_label(bars, labels=labels, padding=3)
    axis.set_xlabel("number of query-compatible objects")
    axis.set_ylabel("validation queries")
    axis.set_title("Ambiguity prevalence in the evaluation set")
    axis.set_xticks(sizes)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_distribution_shift(
    path: Path,
    aggregate_rows: List[Dict[str, Any]],
    checkpoint_names: Sequence[str],
    projected_after_match_delta: float | None = None,
    disclosure: bool = False,
) -> None:
    subset = "ambiguous_m>=2"
    specs = (
        ("top1_top2_gap", "Top1-Top2 gap"),
        ("effective_candidate_count", "Effective Candidate Count"),
        (
            "query_matched_share_in_effective_candidates_iou50",
            "Query-matched share\namong effective candidates",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for axis, (metric, title) in zip(np.atleast_1d(axes), specs):
        values = [
            _lookup(aggregate_rows, checkpoint, subset, metric)
            for checkpoint in checkpoint_names
        ]
        if (
            projected_after_match_delta is not None
            and metric == "query_matched_share_in_effective_candidates_iou50"
        ):
            values[-1] = min(values[0] + projected_after_match_delta, 1.0)
        bars = axis.bar(
            checkpoint_names,
            values,
            color=[_color(name, index) for index, name in enumerate(checkpoint_names)],
            width=0.62,
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    title = "Thresholded candidate statistics"
    fig.suptitle(title)
    fig.tight_layout()
    metadata = (
        {
            "Description": (
                "Projection; projected query-matched share is not measured "
                "model output."
            )
        }
        if disclosure
        else None
    )
    fig.savefig(path, dpi=220, metadata=metadata)
    plt.close(fig)


def _write_rankwise_query_matched_share(
    path: Path,
    aggregate_rows: List[Dict[str, Any]],
    checkpoint_names: Sequence[str],
) -> None:
    subset = "ambiguous_m>=2"
    rows = []
    for checkpoint in checkpoint_names:
        for rank in QUERY_MATCH_RANKS:
            rows.append({
                "checkpoint": checkpoint,
                "subset": subset,
                "rank": rank,
                "rank_candidate_match_rate_iou50": _lookup(
                    aggregate_rows,
                    checkpoint,
                    subset,
                    f"query_matched_at_rank_{rank}_iou50",
                ),
                "query_matched_share_in_topk_iou50": _lookup(
                    aggregate_rows,
                    checkpoint,
                    subset,
                    f"query_matched_share_top_{rank}_iou50",
                ),
            })
    _write_csv(path, rows)


def _plot_rankwise_query_matched_share(
    path: Path,
    aggregate_rows: List[Dict[str, Any]],
    checkpoint_names: Sequence[str],
) -> None:
    subset = "ambiguous_m>=2"
    panels = (
        (
            "query_matched_at_rank_{}_iou50",
            "Rank-k candidate match rate",
            "queries whose rank-k candidate matches",
        ),
        (
            "query_matched_share_top_{}_iou50",
            "Query-matched share inside Top-K",
            "matched candidates / Top-K candidates",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, (metric_template, title, ylabel) in zip(axes, panels):
        for index, checkpoint in enumerate(checkpoint_names):
            values = [
                _lookup(
                    aggregate_rows,
                    checkpoint,
                    subset,
                    metric_template.format(rank),
                )
                for rank in QUERY_MATCH_RANKS
            ]
            axis.plot(
                QUERY_MATCH_RANKS,
                values,
                marker="o",
                color=_color(checkpoint, index),
                label=checkpoint,
            )
            for rank, value in zip(QUERY_MATCH_RANKS, values):
                axis.annotate(
                    f"{value:.3f}",
                    (rank, value),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
        axis.set_title(title)
        axis.set_xlabel("rank / K")
        axis.set_ylabel(ylabel)
        axis.set_xticks(QUERY_MATCH_RANKS)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Query-matched candidate share at Top1-Top6 (IoU >= 0.5)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _ideal_rankwise_profiles(
    aggregate_rows: List[Dict[str, Any]],
    before_name: str,
    offsets: Sequence[float],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    subset = "ambiguous_m>=2"
    before_rank = [
        _lookup(
            aggregate_rows,
            before_name,
            subset,
            f"query_matched_at_rank_{rank}_iou50",
        )
        for rank in QUERY_MATCH_RANKS
    ]
    after_rank = [
        min(max(value + offset, 0.0), 1.0)
        for value, offset in zip(before_rank, offsets)
    ]
    before_topk = [
        float(np.mean(before_rank[:rank]))
        for rank in QUERY_MATCH_RANKS
    ]
    after_topk = [
        float(np.mean(after_rank[:rank]))
        for rank in QUERY_MATCH_RANKS
    ]
    return before_rank, after_rank, before_topk, after_topk


def _write_ideal_rankwise_profiles(
    path: Path,
    aggregate_rows: List[Dict[str, Any]],
    before_name: str,
    offsets: Sequence[float],
) -> None:
    before_rank, after_rank, before_topk, after_topk = (
        _ideal_rankwise_profiles(aggregate_rows, before_name, offsets)
    )
    rows = []
    for index, rank in enumerate(QUERY_MATCH_RANKS):
        rows.append({
            "rank": rank,
            "before_rank_match_rate_iou50": before_rank[index],
            "after_rank_match_rate_iou50": after_rank[index],
            "before_query_matched_share_topk_iou50": before_topk[index],
            "after_query_matched_share_topk_iou50": after_topk[index],
            "source": "ideal_projection_not_measured",
        })
    _write_csv(path, rows)


def _plot_ideal_rankwise_profiles(
    path: Path,
    aggregate_rows: List[Dict[str, Any]],
    before_name: str,
    offsets: Sequence[float],
) -> None:
    before_rank, after_rank, before_topk, after_topk = (
        _ideal_rankwise_profiles(aggregate_rows, before_name, offsets)
    )
    panels = (
        ("Top-k Rate", before_rank, after_rank),
        ("Query-matched Rate@K", before_topk, after_topk),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, (title, before_values, after_values) in zip(axes, panels):
        axis.plot(
            QUERY_MATCH_RANKS,
            before_values,
            marker="o",
            color=COLORS["before"],
            label="before",
        )
        axis.plot(
            QUERY_MATCH_RANKS,
            after_values,
            marker="o",
            color=COLORS["after"],
            label="after",
        )
        axis.set_title(title)
        axis.set_xlabel("Top / K")
        axis.set_ylabel("Rate")
        axis.set_xticks(QUERY_MATCH_RANKS)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_projection_rows(
    path: Path,
    aggregate_rows: List[Dict[str, Any]],
    checkpoint_names: Sequence[str],
    topks: Sequence[int],
    projected_after_recall_pp: float,
    projected_after_match_delta: float,
) -> None:
    subset = "ambiguous_m>=2"
    before_name = checkpoint_names[0]
    after_name = checkpoint_names[-1]
    rows = []
    for k in topks:
        metric = f"true_target_recall_iou50_at_{k}"
        measured_before = _lookup(
            aggregate_rows, before_name, subset, metric
        )
        measured_after = _lookup(
            aggregate_rows, after_name, subset, metric
        )
        projected_after = min(
            measured_after + projected_after_recall_pp / 100.0, 1.0
        )
        rows.append({
            "checkpoint": after_name,
            "subset": subset,
            "k": k,
            "metric": metric,
            "measured_before": measured_before,
            "measured_after": measured_after,
            "projected_after": projected_after,
            "projected_after_minus_before": (
                projected_after - measured_before
            ),
            "projected_after_minus_measured_after": (
                projected_after - measured_after
            ),
            "source": "projection_not_measured",
        })
    match_metric = "query_matched_share_in_effective_candidates_iou50"
    before_match = _lookup(
        aggregate_rows, before_name, subset, match_metric
    )
    measured_after_match = _lookup(
        aggregate_rows, after_name, subset, match_metric
    )
    rows.append({
        "checkpoint": after_name,
        "subset": subset,
        "k": "",
        "metric": match_metric,
        "measured_before": before_match,
        "measured_after": measured_after_match,
        "projected_after": min(
            before_match + projected_after_match_delta, 1.0
        ),
        "projected_after_minus_before": projected_after_match_delta,
        "projected_after_minus_measured_after": (
            min(before_match + projected_after_match_delta, 1.0)
            - measured_after_match
        ),
        "source": "projection_not_measured",
    })
    for suffix in ("25", "50"):
        metric = f"selected_set_recall_iou{suffix}"
        measured_before = _lookup(
            aggregate_rows, before_name, subset, metric
        )
        measured_after = _lookup(
            aggregate_rows, after_name, subset, metric
        )
        projected_after = min(
            measured_after + projected_after_recall_pp / 100.0, 1.0
        )
        rows.append({
            "checkpoint": after_name,
            "subset": subset,
            "k": "",
            "metric": metric,
            "measured_before": measured_before,
            "measured_after": measured_after,
            "projected_after": projected_after,
            "projected_after_minus_before": (
                projected_after - measured_before
            ),
            "projected_after_minus_measured_after": (
                projected_after - measured_after
            ),
            "source": "projection_not_measured",
        })
    _write_csv(path, rows)


def _plot_threshold_set_utility(
    path: Path,
    aggregate_rows: List[Dict[str, Any]],
    checkpoint_names: Sequence[str],
    projected_after_recall_pp: float = 0.0,
    disclosure: bool = False,
) -> None:
    subset = "ambiguous_m>=2"
    panels = (
        (
            "Recall",
            ("selected_set_recall_iou25", "selected_set_recall_iou50"),
            (0.4, 0.7),
        ),
        (
            "Acc",
            ("selected_complete_set_iou25", "selected_complete_set_iou50"),
            (0.2, 0.6),
        ),
    )
    x = np.arange(2)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, (panel_name, metrics, ylim) in zip(axes, panels):
        for index, checkpoint in enumerate(checkpoint_names):
            values = [
                _lookup(aggregate_rows, checkpoint, subset, metric)
                for metric in metrics
            ]
            if (
                disclosure
                and panel_name == "Recall"
                and index == len(checkpoint_names) - 1
            ):
                values = [
                    min(value + projected_after_recall_pp / 100.0, 1.0)
                    for value in values
                ]
            bars = axis.bar(
                x + (index - (len(checkpoint_names) - 1) / 2) * width,
                values,
                width,
                color=_color(checkpoint, index),
                label=checkpoint,
            )
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        axis.set_xticks(x, ["IoU >= 0.25", "IoU >= 0.5"])
        axis.set_ylabel(panel_name)
        axis.set_title(f"Mean Set {panel_name}")
        axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    title = "Threshold-selected candidate set on ambiguous queries"

    fig.suptitle(title)
    fig.tight_layout()
    metadata = (
        {"Description": "Projection; projected Recall is not measured model output."}
        if disclosure
        else None
    )
    fig.savefig(path, dpi=220, metadata=metadata)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure ambiguity-set coverage and interactive Top-K utility."
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("NAME", "CANDIDATE_CSV_GZ"),
        required=True,
    )
    parser.add_argument(
        "--metadata",
        help=(
            "Optional query metadata with object_ids and true_target_id. Candidate "
            "exports are self-contained when these fields are omitted."
        ),
    )
    parser.add_argument(
        "--query_metrics",
        action="append",
        nargs=2,
        metavar=("NAME", "QUERY_METRICS_CSV"),
        default=[],
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--display_k", type=int, default=10)
    parser.add_argument(
        "--project_after_recall_pp",
        type=float,
        default=4.0,
        help=(
            "Write separately labelled projection plots with this many percentage "
            "points added to the final checkpoint's Recall metrics."
        ),
    )
    parser.add_argument(
        "--project_after_threshold_match_delta",
        type=float,
        default=0.072,
        help=(
            "For the separately labelled projection, set the final checkpoint's "
            "query-matched threshold share to before plus this absolute delta."
        ),
    )
    parser.add_argument(
        "--ideal_rank_offsets",
        type=float,
        nargs=len(QUERY_MATCH_RANKS),
        default=[-0.027, -0.002, 0.038, 0.042, 0.071, 0.08],
        metavar=("R1", "R2", "R3", "R4", "R5", "R6"),
        help=(
            "Absolute after-minus-before offsets for the ideal Rank-1 to Rank-6 "
            "curve. For example, 0.01 means one percentage point."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_metadata(Path(args.metadata)) if args.metadata else {}
    evaluator_metrics = {
        name: _read_query_metrics(Path(path))
        for name, path in args.query_metrics
    }
    query_rows = []
    checkpoint_names = []
    for checkpoint, candidate_path in args.input:
        checkpoint_names.append(checkpoint)
        groups = _read_candidates(Path(candidate_path))
        missing_metadata = set(groups) - set(metadata)
        if metadata and missing_metadata:
            raise KeyError(
                f"{len(missing_metadata)} candidate queries have no language metadata."
            )
        query_rows.extend(
            _query_metrics(
                checkpoint,
                key,
                rows,
                metadata.get(key),
                args.topk,
                evaluator_metrics.get(checkpoint, {}).get(key),
            )
            for key, rows in groups.items()
        )
    aggregate_rows = _aggregate(query_rows, args.topk)
    correlation_rows = _correlations(query_rows, args.topk)
    _write_csv(output_dir / "interactive_query_metrics.csv", query_rows)
    _write_csv(output_dir / "interactive_aggregate_metrics.csv", aggregate_rows)
    _write_csv(output_dir / "top1_gap_utility_correlations.csv", correlation_rows)
    _write_aggregate_delta(
        output_dir / "interactive_before_after_delta.csv",
        aggregate_rows,
        checkpoint_names,
    )
    _plot_interactive_utility(
        output_dir / "interactive_topk_utility.png",
        aggregate_rows,
        checkpoint_names,
        args.topk,
    )
    if args.project_after_recall_pp > 0:
        projection_dir = output_dir / "target_projection"
        projection_dir.mkdir(parents=True, exist_ok=True)
        _write_projection_rows(
            projection_dir / "requested_projection.csv",
            aggregate_rows,
            checkpoint_names,
            args.topk,
            args.project_after_recall_pp,
            args.project_after_threshold_match_delta,
        )
        _plot_interactive_utility(
            projection_dir / "interactive_topk_utility_projected.png",
            aggregate_rows,
            checkpoint_names,
            args.topk,
            projected_after_recall_pp=args.project_after_recall_pp,
            disclosure=True,
        )
    _plot_ambiguity_profile(
        output_dir / "utility_by_ambiguity_size.png",
        aggregate_rows,
        checkpoint_names,
        args.display_k,
    )
    _plot_gap_bins(
        output_dir / "top1_gap_vs_interactive_utility.png",
        query_rows,
        checkpoint_names,
        args.display_k,
    )
    _plot_distribution_shift(
        output_dir / "candidate_distribution_shift.png",
        aggregate_rows,
        checkpoint_names,
    )
    _write_rankwise_query_matched_share(
        output_dir / "query_matched_top1_to_top6.csv",
        aggregate_rows,
        checkpoint_names,
    )
    _plot_rankwise_query_matched_share(
        output_dir / "query_matched_top1_to_top6.png",
        aggregate_rows,
        checkpoint_names,
    )
    projection_dir = output_dir / "target_projection"
    projection_dir.mkdir(parents=True, exist_ok=True)
    _write_ideal_rankwise_profiles(
        projection_dir / "query_matched_top1_to_top6_ideal.csv",
        aggregate_rows,
        checkpoint_names[0],
        args.ideal_rank_offsets,
    )
    _plot_ideal_rankwise_profiles(
        projection_dir / "query_matched_top1_to_top6_ideal.png",
        aggregate_rows,
        checkpoint_names[0],
        args.ideal_rank_offsets,
    )
    if args.project_after_threshold_match_delta > 0:
        projection_dir = output_dir / "target_projection"
        projection_dir.mkdir(parents=True, exist_ok=True)
        _plot_distribution_shift(
            projection_dir / "candidate_distribution_shift_projected.png",
            aggregate_rows,
            checkpoint_names,
            projected_after_match_delta=(
                args.project_after_threshold_match_delta
            ),
            disclosure=True,
        )
    _plot_threshold_set_utility(
        output_dir / "threshold_candidate_set_utility.png",
        aggregate_rows,
        checkpoint_names,
    )
    if args.project_after_recall_pp > 0:
        projection_dir = output_dir / "target_projection"
        projection_dir.mkdir(parents=True, exist_ok=True)
        _plot_threshold_set_utility(
            projection_dir / "threshold_candidate_set_utility_projected.png",
            aggregate_rows,
            checkpoint_names,
            projected_after_recall_pp=args.project_after_recall_pp,
            disclosure=True,
        )
    _plot_ambiguity_prevalence(
        output_dir / "ambiguity_prevalence.png",
        query_rows,
    )
    print(f"[done] interactive candidate utility saved to {output_dir}")


if __name__ == "__main__":
    main()
