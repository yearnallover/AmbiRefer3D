#!/usr/bin/env python3
"""Plot measured Stage1 Top-K metrics from paired per-query trajectory exports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    ("Direct", "direct.json", "#3975AD", "o"),
    ("Shuffled pairs", "shuffled.json", "#CB792F", "s"),
    ("Ordered trajectory", "trajectory.json", "#19816A", "^"),
)


def query_key(row):
    return str(row["scene_id"]), int(row["ann_id"])


def load_traces(directory):
    models, sources = {}, {}
    for name, filename, _, _ in MODELS:
        path = directory / filename
        payload = path.read_bytes()
        rows = json.loads(payload)
        grouped = {query_key(row): row for row in rows}
        if not rows or len(grouped) != len(rows):
            raise ValueError(f"{path}: empty input or duplicate query keys")
        checkpoints = sorted({row["checkpoint"] for row in rows})
        if len(checkpoints) != 1:
            raise ValueError(f"{path}: mixed checkpoints")
        models[name] = grouped
        sources[name] = {
            "file": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "checkpoint": checkpoints[0],
        }
    reference = models["Direct"]
    for name, grouped in models.items():
        if grouped.keys() != reference.keys():
            raise ValueError(f"{name}: query set differs from Direct; no silent intersection")
        for key, row in grouped.items():
            other = reference[key]
            if (row["true_target_id"] != other["true_target_id"]
                    or set(row["annotated_candidate_ids"]) != set(other["annotated_candidate_ids"])
                    or row["description"] != other["description"]):
                raise ValueError(f"{name}, {key}: inconsistent query/target annotations")
    return models, sources


def ranked_predictions(row):
    valid = np.asarray(row["proposal_valid"], dtype=bool)
    scores = np.asarray(row["proposal_scores"], dtype=float)
    boxes = np.asarray(row["proposal_boxes"], dtype=float).reshape(-1, 2, 3)
    object_ids = np.asarray(row["proposal_object_ids"], dtype=int)
    object_iou = np.asarray(row["proposal_object_iou"], dtype=float)
    if any(array.shape[0] != len(valid) for array in (scores, boxes, object_ids, object_iou)):
        raise ValueError(f"{query_key(row)}: inconsistent proposal arrays")
    if not np.isfinite(scores[valid]).all() or np.any((scores[valid] < 0) | (scores[valid] > 1)):
        raise ValueError(f"{query_key(row)}: expected finite sigmoid scores")
    if row["target_gt_box"] is None:
        raise ValueError(f"{query_key(row)}: missing target ground truth")
    gt = np.asarray(row["target_gt_box"], dtype=float)
    if gt.shape != (2, 3) or not np.isfinite(gt).all():
        raise ValueError(f"{query_key(row)}: invalid target box")
    if not np.isfinite(boxes[valid]).all() or not np.isfinite(object_iou[valid]).all():
        raise ValueError(f"{query_key(row)}: invalid proposal geometry")
    # Recompute true-target IoU independently of any Stage2 alignment labels.
    intersection = np.maximum(np.minimum(boxes[:, 1], gt[1])
                              - np.maximum(boxes[:, 0], gt[0]), 0).prod(axis=-1)
    volume = np.maximum(boxes[:, 1] - boxes[:, 0], 0).prod(axis=-1)
    gt_volume = np.maximum(gt[1] - gt[0], 0).prod()
    union = volume + gt_volume - intersection
    true_target = intersection / np.maximum(union, 1e-12) >= 0.5
    query_matching = np.isin(object_ids, row["annotated_candidate_ids"]) & (object_iou >= 0.5)
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(-scores[indices], kind="stable")]
    return scores[indices], true_target[indices], query_matching[indices]


def summarize(rows, topks, threshold):
    ranked = [ranked_predictions(row) for row in rows]
    metrics = []
    for k in topks:
        count = [min(k, len(scores)) for scores, _, _ in ranked]
        target_hit = [bool(target[:k].any()) for _, target, _ in ranked]
        share = [float(matching[:k].mean()) if matching[:k].size else 0.0
                 for _, _, matching in ranked]
        metrics.append({
            "k": k,
            "num_queries": len(rows),
            "true_target_at_k_percent": 100 * float(np.mean(target_hit)),
            "query_matching_share_at_k_percent": 100 * float(np.mean(share)),
            "mean_available_proposals": float(np.mean(count)),
            "queries_with_fewer_than_k_proposals": sum(n < k for n in count),
        })
    audit = {
        "num_queries": len(rows),
        "num_empty_proposal_queries": sum(len(scores) == 0 for scores, _, _ in ranked),
        "true_target_ts_at_threshold_percent": 100 * float(np.mean([
            bool(target[scores >= threshold].any()) for scores, target, _ in ranked
        ])),
        "raw_true_target_ts_percent": 100 * float(np.mean([
            bool(target.any()) for _, target, _ in ranked
        ])),
    }
    return metrics, audit


def plot(metrics, output, dataset_label, status):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    fields = ("true_target_at_k_percent", "query_matching_share_at_k_percent")
    titles = ("(a) True Target@K", "(b) Query-Matching Share@K")
    for name, _, color, marker in MODELS:
        rows = [row for row in metrics if row["method"] == name]
        for ax, field in zip(axes, fields):
            ax.plot([row["k"] for row in rows], [row[field] for row in rows],
                    color=color, marker=marker, linewidth=2, markersize=5, label=name)
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("Proposal budget K")
        ax.set_ylabel("Queries with target (%)" if ax is axes[0] else "Matching proposals (%)")
        ax.set_xticks(sorted({row["k"] for row in metrics}))
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E4E7EB", linewidth=0.7)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=3, frameon=False)
    fig.suptitle(dataset_label, fontsize=11, y=0.995)
    fig.text(0.5, 0.025, status, ha="center", fontsize=9, color="#5C6269")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.17, wspace=0.30)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"stage1_topk_preservation.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing three trace JSON files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--status", default="Preview: checkpoint alignment with the main table is unverified")
    parser.add_argument("--topks", type=int, nargs="+", default=[1, 3, 5, 10, 15, 20])
    parser.add_argument("--stage1-threshold", type=float, default=0.01, help="Used for TS audit, not Top-K selection")
    args = parser.parse_args()
    if min(args.topks) < 1 or not 0 < args.stage1_threshold < 1:
        parser.error("K must be positive and the Stage1 threshold must be in (0, 1)")
    models, sources = load_traces(args.input)
    metrics, audits = [], {}
    for name, grouped in models.items():
        rows, audit = summarize([grouped[key] for key in sorted(grouped)],
                                sorted(set(args.topks)), args.stage1_threshold)
        metrics.extend({"method": name, **row} for row in rows)
        audits[name] = audit
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "stage1_topk_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    manifest = {
        "dataset": args.dataset_label, "status": args.status, "sources": sources,
        "stage1_audit_threshold": args.stage1_threshold, "audit": audits,
        "protocol": {
            "selection": "Top-K valid Stage1 proposals by sigmoid score before thresholding; no GT deduplication",
            "true_target": "At least one selected proposal has bbox IoU >= 0.5 with the intended target",
            "query_matching": "Best-IoU scene GT object belongs to initial annotated_candidate_ids, with IoU >= 0.5",
            "share": "Macro-average of per-query matching share among min(K, available) proposals; empty queries contribute zero",
            "denominator": "All paired exported queries; no filtering by target presence or rollout status",
            "note": "Fixed-K coverage is not threshold-based TS or final interaction Acc; no curve fitting to table values",
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot(metrics, args.output, args.dataset_label, args.status)
    print(json.dumps(audits, indent=2))
    print(f"Saved figures, measured metrics, and provenance to {args.output.resolve()}")


if __name__ == "__main__":
    main()
