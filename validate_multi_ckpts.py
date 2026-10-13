from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import importlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from m3drefclip.data.data_module import DataModule
from train import load_initial_weights


def _load_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _config_for_checkpoint(base_cfg: Any, ckpt_path: str, enabled: bool) -> Any:
    if not enabled:
        return base_cfg
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint_cfg = (
        checkpoint.get("hyper_parameters", {}).get("cfg")
        if isinstance(checkpoint, dict)
        else None
    )
    checkpoint_model_cfg = (
        OmegaConf.select(checkpoint_cfg, "model")
        if OmegaConf.is_config(checkpoint_cfg)
        else (checkpoint_cfg or {}).get("model")
    )
    if checkpoint_model_cfg is None:
        print(f"[config] no saved model config in {ckpt_path}; using composed config")
        return base_cfg

    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    runtime_paths = {
        key: OmegaConf.select(cfg, f"model.stage2.{key}")
        for key in (
            "enabled",
            "root_path",
            "round_json_path",
            "bbox_source",
            "scene_data_root",
            "scene_split",
            "validation_enabled",
        )
    }
    runtime_stage1_paths = {
        key: OmegaConf.select(cfg, f"model.stage1_backbone.{key}")
        for key in ("detector_root", "scene_root", "cache_scenes")
    }
    runtime_end_to_end = OmegaConf.select(cfg, "model.stage2.end_to_end.enabled")
    cfg.model = OmegaConf.merge(cfg.model, checkpoint_model_cfg)
    for key, value in runtime_paths.items():
        if value is not None:
            OmegaConf.update(cfg, f"model.stage2.{key}", value, merge=False)
    for key, value in runtime_stage1_paths.items():
        if value is not None:
            OmegaConf.update(
                cfg, f"model.stage1_backbone.{key}", value, merge=False
            )
    if runtime_end_to_end is not None:
        OmegaConf.update(
            cfg, "model.stage2.end_to_end.enabled", runtime_end_to_end, merge=False
        )
    print(f"[config] inherited model architecture from {ckpt_path}")
    return cfg


def _safe_name(path: str) -> str:
    ckpt = Path(path)
    parts = ckpt.parts
    if "output" in parts:
        idx = parts.index("output")
        return "__".join(parts[idx + 1 :]).replace(".ckpt", "")
    return str(ckpt.with_suffix("")).strip(os.sep).replace(os.sep, "__")


def _as_float(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return float(value.item())
    return value


def _collect_ckpts(patterns: List[str], explicit: List[str]) -> List[str]:
    paths: List[str] = []
    for item in explicit:
        paths.append(item)
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    deduped = []
    seen = set()
    for path in paths:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return sorted(deduped)


def _has_hydra_override(overrides: List[str], key: str) -> bool:
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=", f"~{key}")
    return any(item.startswith(prefixes) for item in overrides)


def _write_outputs(rows: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "val_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    if not rows:
        return

    metric_keys = sorted({k for row in rows for k in row.keys() if k not in {"ckpt", "name"}})
    csv_path = os.path.join(out_dir, "val_metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "ckpt", *metric_keys])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _count_eval_types(cfg: Any) -> Dict[str, int]:
    split = str(cfg.data.evaluation.split)
    lang_path = getattr(cfg.data.lang_metadata, f"{split}_language_data", None)
    if not lang_path:
        return {}
    lang_path = str(lang_path)
    if not os.path.exists(lang_path):
        return {}
    with open(lang_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    counts = Counter(item.get("eval_type", "unknown") for item in data)
    counts["total"] = len(data)
    return {f"num_eval_{key}": int(value) for key, value in sorted(counts.items())}


def _load_language_index(cfg: Any) -> Dict[tuple, Dict[str, Any]]:
    split = str(cfg.data.evaluation.split)
    lang_path = str(getattr(cfg.data.lang_metadata, f"{split}_language_data", "") or "")
    if not lang_path or not os.path.exists(lang_path):
        return {}
    with open(lang_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    index = {}
    for row in rows:
        try:
            ann_id = int(row.get("ann_id", -1))
        except (TypeError, ValueError):
            continue
        index[(str(row.get("scene_id", "")), ann_id)] = row
    return index


def _write_query_metrics(
    records: List[Dict[str, Any]],
    language_index: Dict[tuple, Dict[str, Any]],
    out_dir: str,
    checkpoint_name: str,
) -> None:
    query_dir = os.path.join(out_dir, "query_metrics")
    os.makedirs(query_dir, exist_ok=True)
    enriched = []
    for record in records:
        language = language_index.get((record["scene_id"], int(record["ann_id"])), {})
        row = {
            "checkpoint": checkpoint_name,
            **record,
            "object_name": str(language.get("object_name", "")),
            "description": str(language.get("description", "")),
            "generated_question": str(language.get("generated_question", "") or ""),
            "num_candidate_objects": int(len(language.get("object_ids", []) or [])),
            "true_target_id": int(language.get("true_target_id", -1)),
            "has_spatial": bool(language.get("spatial", False)),
            "has_color": bool(language.get("color", False)),
            "has_texture": bool(language.get("texture", False)),
            "has_shape": bool(language.get("shape", False)),
        }
        enriched.append(row)

    json_path = os.path.join(query_dir, f"{checkpoint_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    if enriched:
        fieldnames = list(enriched[0].keys())
        csv_path = os.path.join(query_dir, f"{checkpoint_name}.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enriched)


def _parse_optional_limit(value: Any, total_batches: int) -> int:
    if value is None:
        return total_batches
    try:
        count = int(value)
    except ValueError:
        fraction = float(value)
        if not 0 < fraction <= 1:
            raise ValueError("--stage2_limit_batches as a float must be in (0, 1].")
        count = math.ceil(total_batches * fraction)
    return max(0, min(count, total_batches))


def _flatten_stage2_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                separator = "_" if key == "trajectory" else "@"
                row[f"stage2_{key}{separator}{inner_key}"] = inner_value
        elif key != "results":
            row[f"stage2_{key}"] = value
    return row


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _count_queries_without_target_proposal(
    query_mask: torch.Tensor,
    target_labels: torch.Tensor,
) -> int:
    """Count active queries for which Stage1 produced no target proposal."""
    if target_labels.ndim != query_mask.ndim + 1:
        raise ValueError("target_labels must add one proposal dimension to query_mask.")
    if target_labels.shape[:-1] != query_mask.shape:
        raise ValueError("target_labels and query_mask batch/query dimensions must match.")
    has_target_proposal = target_labels.bool().any(dim=-1)
    return int((query_mask.bool() & ~has_target_proposal).sum().item())


def _standalone_eval_device(model: torch.nn.Module, args: argparse.Namespace) -> torch.device:
    configured_accelerator = getattr(
        getattr(getattr(model, "hparams", None), "cfg", None), "trainer", None
    )
    configured_accelerator = getattr(configured_accelerator, "accelerator", "")
    accelerator = str(args.accelerator or configured_accelerator or "").lower()
    if accelerator in ("gpu", "cuda") and torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.to(device)
        return device
    return next(model.parameters()).device


def _trajectory_grounding_sums(
    trajectory_output: Dict[str, torch.Tensor],
    target_labels: torch.Tensor,
    compatible_labels: torch.Tensor,
    query_mask: torch.Tensor,
    threshold: float,
    topk=(1, 3, 6),
) -> Dict[str, float]:
    """Compute raw fixed-C0 and filtered metrics for one model batch."""
    refined = trajectory_output["refined_score_history"]
    round_mask = trajectory_output["round_mask"].bool()
    proposal_mask = trajectory_output["proposal_mask"].bool()
    sums = Counter()

    for batch_idx, lang_idx in torch.nonzero(query_mask, as_tuple=False).tolist():
        valid = proposal_mask[batch_idx]
        rounds = int(round_mask[batch_idx, lang_idx].sum().item())
        if rounds <= 0 or not valid.any():
            continue
        scores0 = refined[batch_idx, lang_idx, 0]
        scores_t = refined[batch_idx, lang_idx, rounds - 1]
        target = target_labels[batch_idx, lang_idx].bool() & valid
        compatible = compatible_labels[batch_idx, lang_idx].bool() & valid
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        order0 = valid_indices[torch.argsort(scores0[valid_indices], descending=True)]
        order_t = valid_indices[torch.argsort(scores_t[valid_indices], descending=True)]

        sums["raw_num"] += 1
        sums["raw_acc_r0"] += float(target[order0[0]].item())
        sums["raw_acc_rt"] += float(target[order_t[0]].item())

        if target.any():
            sums["raw_target_num"] += 1
            best_target0 = scores0[target].max()
            best_target_t = scores_t[target].max()
            rank0 = 1 + int((scores0[valid & ~target] > best_target0).sum().item())
            rank_t = 1 + int((scores_t[valid & ~target] > best_target_t).sum().item())
            sums["raw_target_rank_r0"] += rank0
            sums["raw_target_rank_rt"] += rank_t
            sums["raw_mrr_r0"] += 1.0 / rank0
            sums["raw_mrr_rt"] += 1.0 / rank_t
            if (valid & ~target).any():
                sums["raw_margin_r0"] += float(
                    (best_target0 - scores0[valid & ~target].max()).item()
                )
                sums["raw_margin_rt"] += float(
                    (best_target_t - scores_t[valid & ~target].max()).item()
                )
                sums["raw_margin_num"] += 1

        compatible_count = int(compatible.sum().item())
        if compatible_count > 0:
            sums["compatible_num"] += 1
            for k in topk:
                selected = order0[: min(k, order0.numel())]
                hits = int(compatible[selected].sum().item())
                sums[f"compatible_recall@{k}"] += hits / compatible_count
                sums[f"compatible_precision@{k}"] += hits / max(int(selected.numel()), 1)
                discounts = 1.0 / torch.log2(
                    torch.arange(
                        2, 2 + selected.numel(), device=selected.device, dtype=torch.float32
                    )
                )
                dcg = float((compatible[selected].float() * discounts).sum().item())
                ideal_count = min(compatible_count, int(selected.numel()))
                idcg = float(discounts[:ideal_count].sum().item())
                sums[f"compatible_ndcg@{k}"] += dcg / max(idcg, 1e-6)

        alive = valid.clone()
        initial_alive = alive & (torch.sigmoid(scores0) >= threshold)
        sums["initial_target_retention"] += float((target & initial_alive).any().item())
        for round_idx in range(rounds):
            alive &= torch.sigmoid(refined[batch_idx, lang_idx, round_idx]) >= threshold
        sums["target_retention"] += float((target & alive).any().item())
        sums["filtered_candidate_count"] += int(alive.sum().item())
        if alive.any():
            filtered_scores = scores_t.masked_fill(~alive, torch.finfo(scores_t.dtype).min)
            sums["filtered_acc"] += float(target[filtered_scores.argmax()].item())

    return dict(sums)


def _merge_sums(destination: Counter, source: Dict[str, float]) -> None:
    for key, value in source.items():
        destination[key] += value


def _accumulate_alive_counts(
    destination: Dict[int, int],
    alive_count_history: List[torch.Tensor],
    max_round: int,
) -> None:
    """Carry a stopped rollout's final candidate count through later rounds."""
    if not alive_count_history:
        return
    for round_number in range(1, int(max_round) + 1):
        history_index = min(round_number, len(alive_count_history)) - 1
        destination[round_number] += int(
            alive_count_history[history_index].sum().item()
        )


@torch.no_grad()
def _evaluate_stage2(model: torch.nn.Module, args: argparse.Namespace, stage1_num: int | None = None) -> Dict[str, Any]:
    stage2 = getattr(model, "stage2", None)
    if stage2 is None:
        raise RuntimeError(
            "Stage2 evaluation requested, but model.stage2 is not initialized. "
            "Pass --eval_stage2 and the required Hydra overrides, especially "
            "model.stage2.round_json_path=... and model.stage2.img_cache_dir=..."
        )

    eval_rollout_batch = importlib.import_module("eval_rollout").eval_rollout_batch

    device = _standalone_eval_device(model, args)
    stage2.to(device)
    stage2.net.eval()

    dl = DataLoader(
        stage2.loader.dataset,
        batch_size=args.stage2_eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=stage2.collate_semantic,
        drop_last=False,
    )
    max_batches = _parse_optional_limit(args.stage2_limit_batches, len(dl))

    total = 0
    correct = 0
    conditional_total = 0
    conditional_correct = 0
    sum_rounds = 0
    initial_target_survival = 0
    candidate_count_sum = 0
    success_at = {k: 0 for k in range(1, args.stage2_max_round + 1)}
    accuracy_at = {k: 0 for k in range(1, args.stage2_max_round + 1)}
    target_survival_at = {k: 0 for k in range(1, args.stage2_max_round + 1)}
    sum_alive_cnt = {k: 0 for k in range(1, args.stage2_max_round + 1)}

    for batch_idx, batch in enumerate(dl):
        if batch_idx >= max_batches:
            break

        pred_idx, _, rounds, alive_hist, alive_cnt_hist, trace = eval_rollout_batch(
            model=stage2.net,
            batch=batch,
            constraint_table=stage2.constraint_table,
            vocab=stage2.vocab,
            max_len=stage2.max_len,
            max_round=args.stage2_max_round,
            thr=args.stage2_thr,
            stop_thr=args.stage2_stop_thr,
            eval_mode=args.stage2_eval_mode,
            v_target_strategy=args.stage2_v_target_strategy,
            selector_strategy=args.stage2_selector_strategy,
            return_trace=True,
        )

        gt = batch["target_index"].detach().cpu().long()
        cand_mask = batch["cand_mask"].detach().cpu()
        batch_size = int(gt.shape[0])
        valid_target = (gt >= 0) & (gt < cand_mask.shape[1])
        safe_gt = gt.clamp(0, max(cand_mask.shape[1] - 1, 0))
        initial_alive = valid_target & (cand_mask.gather(1, safe_gt.view(batch_size, 1)).squeeze(1) > 0.5)
        ok = valid_target & (pred_idx.long() == gt)

        _accumulate_alive_counts(
            sum_alive_cnt, alive_cnt_hist, args.stage2_max_round
        )

        def alive_at_k(k: int):
            if not alive_hist:
                return torch.zeros((batch_size,), dtype=torch.bool)
            idx = min(k, len(alive_hist)) - 1
            return alive_hist[idx].bool()

        total += batch_size
        correct += int(ok.sum().item())
        conditional_total += int(initial_alive.sum().item())
        conditional_correct += int((ok & initial_alive).sum().item())
        sum_rounds += int(rounds.sum().item())
        initial_target_survival += int(initial_alive.sum().item())
        candidate_count_sum += int(cand_mask.sum(dim=1).sum().item())

        for k in range(1, args.stage2_max_round + 1):
            success_at[k] += int((ok & (rounds <= k)).sum().item())
            pred_history = trace["pred_idx_history"]
            if pred_history:
                pred_at_k = pred_history[min(k, len(pred_history)) - 1].long()
                accuracy_at[k] += int(
                    (valid_target & (pred_at_k == gt)).sum().item()
                )
            target_survival_at[k] += int(alive_at_k(k).sum().item())

    denom = max(total, 1)
    conditional_denom = max(conditional_total, 1)
    stage1_denom = max(int(stage1_num or 0), 1)
    metrics = {
        "acc": float(correct / denom),
        "acc_stage1_den": float(correct / stage1_denom) if stage1_num else None,
        "conditional_acc": float(conditional_correct / conditional_denom),
        "num": int(total),
        "stage1_num": int(stage1_num) if stage1_num is not None else None,
        "conditional_num": int(conditional_total),
        "coverage_vs_stage1": float(total / stage1_denom) if stage1_num else None,
        "avg_rounds": float(sum_rounds / denom),
        "initial_target_survival": float(initial_target_survival / denom),
        "final_target_survival": float(target_survival_at[args.stage2_max_round] / denom),
        "avg_candidate_count": float(candidate_count_sum / denom),
        "max_round": int(args.stage2_max_round),
        "thr": float(args.stage2_thr),
        "stop_thr": float(args.stage2_stop_thr),
        "eval_mode": args.stage2_eval_mode,
        "selector_strategy": args.stage2_selector_strategy,
        "v_target_strategy": args.stage2_v_target_strategy,
        "success": {str(k): float(v / denom) for k, v in success_at.items()},
        "accuracy_after": {
            str(k): float(v / denom) for k, v in accuracy_at.items()
        },
        "target_survival": {str(k): float(v / denom) for k, v in target_survival_at.items()},
        "avg_alive_cnt": {str(k): float(v / denom) for k, v in sum_alive_cnt.items()},
    }
    for k, value in metrics["success"].items():
        metrics[f"success@{k}"] = value
    for k, value in metrics["accuracy_after"].items():
        metrics[f"acc_after@{k}"] = value
    for k, value in metrics["target_survival"].items():
        metrics[f"target_survival@{k}"] = value
        metrics[f"target_alive@{k}"] = value
    return metrics


@torch.no_grad()
def _evaluate_stage2_end_to_end(
    model: torch.nn.Module,
    val_loader: DataLoader,
    args: argparse.Namespace,
    stage1_num: int | None = None,
    collect_stage1_metrics: bool = False,
) -> Dict[str, Any]:
    stage2 = getattr(model, "stage2", None)
    if stage2 is None:
        raise RuntimeError("End-to-end Stage2 evaluation requires model.stage2.enabled=True.")

    eval_rollout_batch = importlib.import_module("eval_rollout").eval_rollout_batch
    device = _standalone_eval_device(model, args)
    model.eval()
    stage2.net.eval()

    total = int(stage1_num or 0)
    observed_queries = 0
    aligned = 0
    missing_metadata = 0
    missing_target_proposal = 0
    empty_initial_candidates = 0
    correct = 0
    conditional_total = 0
    conditional_correct = 0
    sum_rounds = 0
    rollout_count = 0
    required_interactions_sum = 0
    scheduled_rollout_count = 0
    premature_stop_count = 0
    premature_stop_signal_count = 0
    premature_one_left_signal_count = 0
    executed_round_histogram = Counter()
    required_round_histogram = Counter()
    minimum_round_histogram = Counter()
    initial_target_survival = 0
    candidate_count_sum = 0
    success_at = {k: 0 for k in range(1, args.stage2_max_round + 1)}
    accuracy_at = {k: 0 for k in range(1, args.stage2_max_round + 1)}
    protocol_eligible_at = {
        k: 0 for k in range(1, args.stage2_max_round + 1)
    }
    target_survival_at = {k: 0 for k in range(1, args.stage2_max_round + 1)}
    sum_alive_cnt = {k: 0 for k in range(1, args.stage2_max_round + 1)}
    trajectory_sums = Counter()
    stage1_predictions = {}
    stage1_ground_truths = {}

    threshold = (
        float(args.stage2_stage1_thr)
        if args.stage2_stage1_thr is not None
        else float(model.hparams.cfg.model.inference.output_threshold)
    )
    max_batches = _parse_optional_limit(args.stage2_limit_batches, len(val_loader))
    seen_queries = set()

    for batch_idx, cpu_batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        if collect_stage1_metrics and (
            batch_idx == 0 or (batch_idx + 1) % 100 == 0
        ):
            print(
                f"[single-pass] batch {batch_idx + 1}/{max_batches}",
                flush=True,
            )
        batch = _move_to_device(cpu_batch, device)
        new_query_mask = torch.zeros_like(batch["ann_id"], dtype=torch.bool)
        for scene_idx, scene_id in enumerate(batch["scene_id"]):
            for lang_idx, ann_id in enumerate(batch["ann_id"][scene_idx].tolist()):
                query_key = (str(scene_id), int(ann_id))
                if query_key in seen_queries:
                    continue
                seen_queries.add(query_key)
                if (
                    args.stage2_eval_type != "all"
                    and str(batch["eval_type"][scene_idx][lang_idx])
                    != args.stage2_eval_type
                ):
                    continue
                new_query_mask[scene_idx, lang_idx] = True
        new_query_count = int(new_query_mask.sum().item())
        observed_queries += new_query_count
        if not new_query_mask.any():
            continue

        trajectory_query_mask = batch["trajectory_round_mask"].any(dim=-1)
        missing_metadata += int(
            (new_query_mask & ~trajectory_query_mask).sum().item()
        )
        batch["trajectory_round_mask"] = (
            batch["trajectory_round_mask"] & new_query_mask.unsqueeze(-1)
        )

        stage1_output = model(batch)
        model.ref_loss(
            stage1_output,
            stage1_output["pred_aabb_min_max_bounds"],
            stage1_output["pred_aabb_scores"],
            batch["gt_aabb_min_max_bounds"],
            batch["gt_target_obj_id_mask"].permute(dims=(1, 0)),
            batch["aabb_count_offsets"],
        )
        if collect_stage1_metrics:
            stage1_predictions.update(model._parse_pred_results(batch, stage1_output))
            stage1_ground_truths.update(model._parse_gt(batch))
        trajectory_context = model._online_trajectory_context(batch, stage1_output)
        if trajectory_context is None:
            continue
        trajectory_output, target_labels = trajectory_context
        _merge_sums(
            trajectory_sums,
            _trajectory_grounding_sums(
                trajectory_output=trajectory_output,
                target_labels=target_labels,
                compatible_labels=stage1_output["gt_labels"],
                query_mask=new_query_mask & trajectory_query_mask,
                threshold=threshold,
            ),
        )
        end_to_end = stage2.build_end_to_end_batch(
            data_dict=batch,
            stage1_output=stage1_output,
            trajectory_output=trajectory_output,
            target_labels=target_labels,
        )
        if end_to_end is None:
            # With batch_size=1, a missing target proposal leaves no encoded
            # Stage2 rows, so build_end_to_end_batch cannot return its local
            # diagnostics. Recover that count here instead of misreporting the
            # query as missing trajectory metadata in the final reconciliation.
            missing_target_proposal += _count_queries_without_target_proposal(
                new_query_mask & trajectory_query_mask,
                target_labels,
            )
            continue

        missing_metadata += int(end_to_end["missing_metadata"])
        missing_target_proposal += int(end_to_end["missing_target_proposal"])
        flat_batch = end_to_end["batch"]
        lengths = end_to_end["trajectory_batch"]["lengths"].long()
        starts = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=device),
                torch.cumsum(lengths, dim=0)[:-1],
            )
        )
        initial_batch = {
            key: value.index_select(0, starts)
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == flat_batch["cand_mask"].shape[0]
            else value
            for key, value in flat_batch.items()
        }

        proposal_mask = initial_batch["cand_mask"].bool()
        stage1_keep = torch.sigmoid(initial_batch["cand_stage1_scores"]) >= threshold
        initial_mask = proposal_mask & stage1_keep
        if stage2.end_to_end_preserve_scene_context:
            initial_batch["context_mask"] = proposal_mask.float()
        else:
            initial_batch["cand_mask"] = initial_mask.float()
        initial_batch["alive_mask"] = initial_mask.float()

        gt = initial_batch["target_index"].long()
        target_candidate_mask = initial_batch["target_candidate_mask"].bool()
        valid_target = target_candidate_mask.any(dim=1)
        initial_alive = valid_target & (target_candidate_mask & initial_mask).any(dim=1)
        nonempty = initial_mask.any(dim=1)

        batch_aligned = int(gt.shape[0])
        aligned += batch_aligned
        initial_target_survival += int(initial_alive.sum().item())
        conditional_total += int(initial_alive.sum().item())
        candidate_count_sum += int(initial_mask.sum().item())
        empty_initial_candidates += int((~nonempty).sum().item())

        if not nonempty.any():
            continue
        active_indices = torch.nonzero(nonempty, as_tuple=False).squeeze(1)
        rollout_batch = {
            key: value.index_select(0, active_indices)
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_aligned
            else value
            for key, value in initial_batch.items()
        }
        rollout_initial_alive = initial_alive.index_select(0, active_indices).detach().cpu()
        rollout_gt = gt.index_select(0, active_indices).detach().cpu()
        rollout_target_mask = target_candidate_mask.index_select(
            0, active_indices
        ).detach().cpu()

        pred_idx, _, rounds, alive_hist, alive_cnt_hist, trace = eval_rollout_batch(
            model=stage2.net,
            batch=rollout_batch,
            constraint_table=stage2.constraint_table,
            vocab=stage2.vocab,
            max_len=stage2.max_len,
            max_round=args.stage2_max_round,
            thr=args.stage2_thr,
            stop_thr=args.stage2_stop_thr,
            eval_mode=args.stage2_eval_mode,
            v_target_strategy=args.stage2_v_target_strategy,
            selector_strategy=args.stage2_selector_strategy,
            return_trace=True,
        )
        safe_pred = pred_idx.long().clamp(0, max(rollout_target_mask.shape[1] - 1, 0))
        ok = rollout_target_mask.gather(1, safe_pred.unsqueeze(1)).squeeze(1)
        correct += int(ok.sum().item())
        conditional_correct += int((ok & rollout_initial_alive).sum().item())
        sum_rounds += int(rounds.sum().item())
        rollout_count += int(rounds.numel())
        executed_round_histogram.update(int(value) for value in rounds.tolist())
        required_rounds = rollout_batch.get("required_rollout_rounds")
        minimum_rounds = rollout_batch.get("minimum_rollout_rounds")
        if minimum_rounds is None:
            minimum_rounds = torch.zeros_like(rounds)
        else:
            minimum_rounds = minimum_rounds.detach().cpu().long()
        minimum_round_histogram.update(
            int(value) for value in minimum_rounds.tolist()
        )
        for k in range(1, args.stage2_max_round + 1):
            protocol_eligible_at[k] += int((minimum_rounds <= k).sum().item())
        if required_rounds is not None:
            required_rounds = required_rounds.detach().cpu().long()
            required_round_histogram.update(
                int(value) for value in required_rounds.tolist()
            )
            scheduled = required_rounds > 0
            scheduled_rollout_count += int(scheduled.sum().item())
            required_interactions_sum += int(required_rounds[scheduled].sum().item())
            premature_stop_count += int(
                ((rounds.long() < required_rounds) & scheduled).sum().item()
            )
            stop_history = trace.get("stop_probability_history", [])
            for row_idx in torch.nonzero(scheduled, as_tuple=False).squeeze(1).tolist():
                early_steps = min(
                    max(int(required_rounds[row_idx].item()) - 1, 0),
                    len(stop_history),
                )
                if any(
                    float(stop_history[step][row_idx]) >= args.stage2_stop_thr
                    for step in range(early_steps)
                ):
                    premature_stop_signal_count += 1
                if any(
                    int(alive_cnt_hist[step][row_idx]) <= 1
                    for step in range(min(early_steps, len(alive_cnt_hist)))
                ):
                    premature_one_left_signal_count += 1

        _accumulate_alive_counts(
            sum_alive_cnt, alive_cnt_hist, args.stage2_max_round
        )

        for k in range(1, args.stage2_max_round + 1):
            success_at[k] += int((ok & (rounds <= k)).sum().item())
            pred_history = trace["pred_idx_history"]
            if pred_history:
                pred_at_k = pred_history[
                    min(k, len(pred_history)) - 1
                ].long()
                accuracy_at[k] += int(
                    rollout_target_mask.gather(
                        1, pred_at_k.unsqueeze(1)
                    ).squeeze(1).sum().item()
                )
            if alive_hist:
                history_idx = min(k, len(alive_hist)) - 1
                target_survival_at[k] += int(alive_hist[history_idx].bool().sum().item())

    if total <= 0 or args.stage2_limit_batches is not None:
        total = observed_queries
    missing_metadata = max(total - aligned - missing_target_proposal, 0)
    denom = max(total, 1)
    conditional_denom = max(conditional_total, 1)
    rollout_denom = max(rollout_count, 1)
    scheduled_rollout_denom = max(scheduled_rollout_count, 1)
    raw_denom = max(int(trajectory_sums["raw_num"]), 1)
    raw_target_denom = max(int(trajectory_sums["raw_target_num"]), 1)
    raw_margin_denom = max(int(trajectory_sums["raw_margin_num"]), 1)
    compatible_denom = max(int(trajectory_sums["compatible_num"]), 1)
    trajectory_metrics = {
        "raw_num": int(trajectory_sums["raw_num"]),
        "raw_target_num": int(trajectory_sums["raw_target_num"]),
        "raw_acc_r0": float(trajectory_sums["raw_acc_r0"] / raw_denom),
        "raw_acc_rt": float(trajectory_sums["raw_acc_rt"] / raw_denom),
        "raw_target_rank_r0": float(
            trajectory_sums["raw_target_rank_r0"] / raw_target_denom
        ),
        "raw_target_rank_rt": float(
            trajectory_sums["raw_target_rank_rt"] / raw_target_denom
        ),
        "raw_mrr_r0": float(trajectory_sums["raw_mrr_r0"] / raw_target_denom),
        "raw_mrr_rt": float(trajectory_sums["raw_mrr_rt"] / raw_target_denom),
        "target_hard_negative_margin_r0": float(
            trajectory_sums["raw_margin_r0"] / raw_margin_denom
        ),
        "target_hard_negative_margin_rt": float(
            trajectory_sums["raw_margin_rt"] / raw_margin_denom
        ),
        "initial_target_retention": float(
            trajectory_sums["initial_target_retention"] / raw_denom
        ),
        "target_retention": float(trajectory_sums["target_retention"] / raw_denom),
        "filtered_acc": float(trajectory_sums["filtered_acc"] / raw_denom),
        "filtered_candidate_count": float(
            trajectory_sums["filtered_candidate_count"] / raw_denom
        ),
    }
    for k in (1, 3, 6):
        trajectory_metrics[f"query_compatible_recall@{k}"] = float(
            trajectory_sums[f"compatible_recall@{k}"] / compatible_denom
        )
        trajectory_metrics[f"query_compatible_precision@{k}"] = float(
            trajectory_sums[f"compatible_precision@{k}"] / compatible_denom
        )
        trajectory_metrics[f"query_compatible_ndcg@{k}"] = float(
            trajectory_sums[f"compatible_ndcg@{k}"] / compatible_denom
        )
    metrics = {
        "acc": float(correct / denom),
        "acc_stage1_den": float(correct / denom),
        "conditional_acc": float(conditional_correct / conditional_denom),
        "num": int(total),
        "stage1_num": int(total),
        "conditional_num": int(conditional_total),
        "coverage_vs_stage1": float(aligned / denom),
        "aligned_num": int(aligned),
        "missing_metadata": int(missing_metadata),
        "missing_target_proposal": int(missing_target_proposal),
        "empty_initial_candidates": int(empty_initial_candidates),
        "avg_rounds": float(sum_rounds / denom),
        "avg_rounds_dataset_den": float(sum_rounds / denom),
        "avg_rounds_rollout_den": float(sum_rounds / rollout_denom),
        "avg_dialogue_rounds_rollout_den": float(sum_rounds / rollout_denom + 1.0),
        "rollout_num": int(rollout_count),
        "scheduled_rollout_num": int(scheduled_rollout_count),
        "required_interactions_avg": float(
            required_interactions_sum / scheduled_rollout_denom
        ),
        "premature_stop_num": int(premature_stop_count),
        "premature_stop_rate": float(
            premature_stop_count / scheduled_rollout_denom
        ),
        "premature_stop_signal_num": int(premature_stop_signal_count),
        "premature_stop_signal_rate": float(
            premature_stop_signal_count / scheduled_rollout_denom
        ),
        "premature_one_left_signal_num": int(premature_one_left_signal_count),
        "premature_one_left_signal_rate": float(
            premature_one_left_signal_count / scheduled_rollout_denom
        ),
        "executed_round_histogram": {
            str(key): int(value)
            for key, value in sorted(executed_round_histogram.items())
        },
        "required_round_histogram": {
            str(key): int(value)
            for key, value in sorted(required_round_histogram.items())
        },
        "minimum_round_histogram": {
            str(key): int(value)
            for key, value in sorted(minimum_round_histogram.items())
        },
        "initial_target_survival": float(initial_target_survival / denom),
        "final_target_survival": float(target_survival_at[args.stage2_max_round] / denom),
        "avg_candidate_count": float(candidate_count_sum / denom),
        "max_round": int(args.stage2_max_round),
        "thr": float(args.stage2_thr),
        "stop_thr": float(args.stage2_stop_thr),
        "stage1_thr": threshold,
        "eval_mode": args.stage2_eval_mode,
        "selector_strategy": args.stage2_selector_strategy,
        "v_target_strategy": args.stage2_v_target_strategy,
        "eval_type": args.stage2_eval_type,
        "success": {str(k): float(v / denom) for k, v in success_at.items()},
        "success_count": {str(k): int(v) for k, v in success_at.items()},
        "accuracy_after": {
            str(k): float(v / denom) for k, v in accuracy_at.items()
        },
        "accuracy_after_count": {
            str(k): int(v) for k, v in accuracy_at.items()
        },
        "protocol_eligible_num": {
            str(k): int(v) for k, v in protocol_eligible_at.items()
        },
        "protocol_eligible_rate": {
            str(k): float(v / denom) for k, v in protocol_eligible_at.items()
        },
        "target_survival": {
            str(k): float(v / denom) for k, v in target_survival_at.items()
        },
        "avg_alive_cnt": {str(k): float(v / denom) for k, v in sum_alive_cnt.items()},
        "trajectory": trajectory_metrics,
    }
    if collect_stage1_metrics:
        model.evaluator.set_ground_truths(stage1_ground_truths)
        stage1_results = model.evaluator.evaluate(stage1_predictions)
        metrics["_stage1_metrics"] = {
            f"val_eval/{metric_name}_{breakdown}": _as_float(value)
            for metric_name, breakdowns in stage1_results.items()
            for breakdown, value in breakdowns.items()
        }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one or more checkpoints and save val metrics as JSON/CSV."
    )
    parser.add_argument("--ckpt", action="append", default=[], help="Checkpoint path. Can be passed multiple times.")
    parser.add_argument(
        "--ckpt_glob",
        action="append",
        default=[],
        help="Glob pattern for checkpoints, e.g. 'output/Multi3DRefer/joint_stage12_minimal*/training/last.ckpt'.",
    )
    parser.add_argument("--out_dir", default="output/Multi3DRefer/val_metrics", help="Directory for metrics files.")
    parser.add_argument("--config_name", default="global_config", help="Hydra config name.")
    parser.add_argument("--config_path", default="config", help="Hydra config path.")
    parser.add_argument("--devices", default="1", help="Lightning devices value, e.g. 1 or auto.")
    parser.add_argument("--accelerator", default="", help="Override accelerator. Defaults to cfg.trainer.accelerator.")
    parser.add_argument(
        "--limit_val_batches",
        default=None,
        help="Optional Lightning limit_val_batches for smoke tests, e.g. 1 or 0.1.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict Lightning checkpoint loading. Default is non-strict manual loading.",
    )
    parser.add_argument(
        "--use_checkpoint_config",
        action="store_true",
        help=(
            "Instantiate each model with the architecture saved in its checkpoint while "
            "retaining the current dataset and Stage2 runtime paths."
        ),
    )
    parser.add_argument(
        "--keep_stage2",
        action="store_true",
        help="Do not force model.stage2.enabled=False during validation.",
    )
    parser.add_argument(
        "--eval_stage2",
        action="store_true",
        help="Run stage2 rollout evaluation and append stage2 metrics to the same CSV row.",
    )
    parser.add_argument(
        "--eval_stage2_end_to_end",
        action="store_true",
        help=(
            "Run Stage1 on the validation loader and feed its current proposals, scores, "
            "features, geometry, and survival mask into Stage2 rollout."
        ),
    )
    parser.add_argument(
        "--single_pass_end_to_end",
        action="store_true",
        help=(
            "Collect Stage1 validation metrics during the end-to-end Stage2 pass "
            "instead of running the detector twice. Requires --eval_stage2_end_to_end."
        ),
    )
    parser.add_argument("--stage2_eval_batch_size", type=int, default=64)
    parser.add_argument("--stage2_limit_batches", default=None, help="Optional stage2 limit, e.g. 10 or 0.1.")
    parser.add_argument("--stage2_max_round", type=int, default=8)
    parser.add_argument(
        "--stage2_stage1_thr",
        type=float,
        default=None,
        help=(
            "Override the Stage1 confidence threshold used to initialize the "
            "Stage2 candidate set. Defaults to model.inference.output_threshold."
        ),
    )
    parser.add_argument("--stage2_thr", type=float, default=0.4)
    parser.add_argument("--stage2_stop_thr", type=float, default=0.5)
    parser.add_argument("--stage2_eval_mode", choices=["oracle", "agent"], default="agent")
    parser.add_argument(
        "--stage2_eval_type",
        default="all",
        help="Evaluate Stage2 only on this eval_type (for example, mt), or all.",
    )
    parser.add_argument(
        "--stage2_selector_strategy",
        choices=["ours", "random", "heuristic", "all_at_once"],
        default="ours",
    )
    parser.add_argument(
        "--stage2_v_target_strategy",
        choices=["top1", "random", "mean", "candidate_belief"],
        default="top1",
    )
    parser.add_argument(
        "--export_query_metrics",
        action="store_true",
        help="Export per-query Stage1 metrics enriched with language metadata.",
    )
    parser.add_argument(
        "--reuse_stage1_metrics_json",
        default=None,
        help=(
            "Reuse Stage1 metrics from a prior per-checkpoint JSON and skip Lightning "
            "validation; useful when rerunning only end-to-end Stage2 rollout."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra overrides, e.g. data=multi3drefer data.dataloader.batch_size=2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.single_pass_end_to_end and not args.eval_stage2_end_to_end:
        raise SystemExit(
            "--single_pass_end_to_end requires --eval_stage2_end_to_end."
        )
    ckpt_paths = _collect_ckpts(args.ckpt_glob, args.ckpt)
    if not ckpt_paths:
        raise SystemExit("No checkpoints found. Pass --ckpt or --ckpt_glob.")

    overrides = list(args.overrides)
    if not _has_hydra_override(overrides, "project_root_path"):
        overrides.append(f"project_root_path={os.path.abspath(os.getcwd())}")
    if args.eval_stage2 or args.eval_stage2_end_to_end:
        overrides.append("model.stage2.enabled=True")
    if not args.keep_stage2 and not args.eval_stage2 and not args.eval_stage2_end_to_end:
        overrides.append("model.stage2.enabled=False")
    overrides.append("logger=null")

    with hydra.initialize(version_base=None, config_path=args.config_path):
        cfg = hydra.compose(config_name=args.config_name, overrides=overrides)

    pl.seed_everything(cfg.test_seed, workers=True)
    cfg.data.evaluator.verbose = False
    eval_type_counts = _count_eval_types(cfg)
    language_index = _load_language_index(cfg) if args.export_query_metrics else {}

    data_module = DataModule(cfg.data)
    data_module.val_set = data_module.dataset(cfg.data, cfg.data.evaluation.split)
    val_loader = data_module.val_dataloader()

    accelerator = args.accelerator or cfg.trainer.accelerator
    devices: Any = args.devices
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)

    trainer_kwargs = {}
    if args.limit_val_batches is not None:
        try:
            limit_val_batches: Any = int(args.limit_val_batches)
        except ValueError:
            limit_val_batches = float(args.limit_val_batches)
        trainer_kwargs["limit_val_batches"] = limit_val_batches

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        max_epochs=1,
        **trainer_kwargs,
    )

    rows: List[Dict[str, Any]] = []
    for ckpt_path in ckpt_paths:
        if not os.path.exists(ckpt_path):
            print(f"[skip] missing checkpoint: {ckpt_path}")
            continue

        print(f"[validate] {ckpt_path}")
        model_cfg = _config_for_checkpoint(
            cfg, ckpt_path, enabled=args.use_checkpoint_config
        )
        model = hydra.utils.instantiate(model_cfg.model.model_name, model_cfg)

        ckpt_for_trainer = ckpt_path
        if not args.strict:
            load_initial_weights(model, ckpt_path, strict=False)
            ckpt_for_trainer = None

        reused_row = {}
        if args.reuse_stage1_metrics_json:
            with open(args.reuse_stage1_metrics_json, "r", encoding="utf-8") as f:
                reused_data = json.load(f)
            if isinstance(reused_data, list):
                reused_row = next(
                    (
                        item for item in reused_data
                        if os.path.abspath(str(item.get("ckpt", ""))) == os.path.abspath(ckpt_path)
                    ),
                    reused_data[0] if len(reused_data) == 1 else {},
                )
            else:
                reused_row = reused_data
            result = {
                key: value for key, value in reused_row.items()
                if key.startswith("val_")
            }
            print(f"[reuse] Stage1 metrics from {args.reuse_stage1_metrics_json}")
        elif args.single_pass_end_to_end:
            result = {}
            print("[single-pass] collecting Stage1 metrics during Stage2 rollout")
        else:
            result_list = trainer.validate(
                model=model, dataloaders=val_loader, ckpt_path=ckpt_for_trainer
            )
            result = result_list[0] if result_list else {}
        row = {
            key: value for key, value in reused_row.items()
            if not key.startswith("stage2_")
        }
        row.update({
            "name": _safe_name(ckpt_path),
            "ckpt": os.path.abspath(ckpt_path),
        })
        row.update(eval_type_counts)
        row.update({k: _as_float(v) for k, v in result.items()})
        if args.eval_stage2_end_to_end:
            stage1_count_key = (
                "num_eval_total"
                if args.stage2_eval_type == "all"
                else f"num_eval_{args.stage2_eval_type}"
            )
            stage1_num = eval_type_counts.get(stage1_count_key)
            stage2_metrics = _evaluate_stage2_end_to_end(
                model,
                val_loader,
                args,
                stage1_num=stage1_num,
                collect_stage1_metrics=args.single_pass_end_to_end,
            )
            result.update(stage2_metrics.pop("_stage1_metrics", {}))
            row.update({k: _as_float(v) for k, v in result.items()})
            row.update(_flatten_stage2_metrics(stage2_metrics))
        elif args.eval_stage2:
            stage1_num = eval_type_counts.get("num_eval_total")
            stage2_metrics = _evaluate_stage2(model, args, stage1_num=stage1_num)
            row.update(_flatten_stage2_metrics(stage2_metrics))
        rows.append(row)

        if args.export_query_metrics:
            _write_query_metrics(
                getattr(model.evaluator, "last_query_results", []),
                language_index,
                args.out_dir,
                row["name"],
            )

        per_ckpt_path = os.path.join(args.out_dir, f"{row['name']}.json")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(per_ckpt_path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)

    _write_outputs(rows, args.out_dir)
    print(f"[done] saved metrics to {args.out_dir}")


if __name__ == "__main__":
    main()
