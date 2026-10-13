from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from m3drefclip.common_ops.functions import common_ops
from m3drefclip.data.data_module import DataModule, _collate_fn
from m3drefclip.util.utils import get_batch_aabb_pair_ious


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


def _first_chunk_per_scene(dataset) -> List[int]:
    first_indices: Dict[int, int] = {}
    for chunk_index, scene_index in enumerate(dataset.chunk_scene_indices.tolist()):
        first_indices.setdefault(int(scene_index), chunk_index)
    return [first_indices[index] for index in sorted(first_indices)]


def _dense(values: torch.Tensor, offsets: torch.Tensor, max_proposals: int) -> torch.Tensor:
    return common_ops.convert_sparse_tensor_to_dense(values, offsets, max_proposals)


def _proposal_semantics(output: Dict[str, torch.Tensor], proposal_indices: torch.Tensor) -> torch.Tensor:
    _, cluster_point_indices, proposal_offsets = output["proposal_scores"]
    point_semantics = output["semantic_scores"].argmax(dim=1)
    semantic_ids = torch.zeros_like(proposal_indices)
    for row, proposal_index in enumerate(proposal_indices.tolist()):
        if proposal_index < 0:
            continue
        start = int(proposal_offsets[proposal_index].item())
        end = int(proposal_offsets[proposal_index + 1].item())
        labels = point_semantics[cluster_point_indices[start:end].long()]
        if labels.numel():
            semantic_ids[row] = torch.bincount(labels.long()).argmax()
    return semantic_ids


def _match_gt_to_proposals(gt_boxes: torch.Tensor, proposal_boxes: torch.Tensor):
    if proposal_boxes.shape[0] == 0:
        return (
            torch.full((gt_boxes.shape[0],), -1, dtype=torch.long, device=gt_boxes.device),
            gt_boxes.new_zeros(gt_boxes.shape[0]),
        )
    best_indices = torch.zeros(gt_boxes.shape[0], dtype=torch.long, device=gt_boxes.device)
    best_ious = gt_boxes.new_zeros(gt_boxes.shape[0])
    for gt_index, gt_box in enumerate(gt_boxes):
        ious = get_batch_aabb_pair_ious(
            proposal_boxes,
            gt_box.unsqueeze(0).expand(proposal_boxes.shape[0], -1, -1),
        )
        best_ious[gt_index], best_indices[gt_index] = ious.max(dim=0)
    return best_indices, best_ious


def _load_checkpoint(model: torch.nn.Module, path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    result = model.load_state_dict(state_dict, strict=False)
    print(
        f"loaded {path}: missing={len(result.missing_keys)}, "
        f"unexpected={len(result.unexpected_keys)}"
    )


def _export_scene(model, batch: Dict[str, Any], output_dir: Path, max_proposals: int) -> Dict[str, float]:
    output = model.detector(batch)
    valid_mask = _dense(
        output["aabb_features"].new_ones((output["aabb_features"].shape[0], 1)),
        output["proposal_batch_offsets"],
        max_proposals,
    )[0, :, 0].bool()
    features_3d = _dense(
        output["aabb_features"], output["proposal_batch_offsets"], max_proposals
    )[0, valid_mask]
    proposal_boxes = _dense(
        output["pred_aabb_min_max_bounds"].reshape(-1, 6),
        output["proposal_batch_offsets"],
        max_proposals,
    )[0, valid_mask].reshape(-1, 2, 3)
    proposal_indices = output["proposal_indices_dense"][0, valid_mask].long()

    gt_start = int(batch["aabb_count_offsets"][0].item())
    gt_end = int(batch["aabb_count_offsets"][1].item())
    gt_boxes = batch["gt_aabb_min_max_bounds"][gt_start:gt_end]
    obj_ids = batch["gt_aabb_obj_ids"][gt_start:gt_end].long()
    best_indices, best_ious = _match_gt_to_proposals(gt_boxes, proposal_boxes)

    feature_dim = int(output["aabb_features"].shape[-1])
    matched_features = features_3d.new_zeros((gt_boxes.shape[0], feature_dim))
    matched_bboxes = gt_boxes.new_zeros((gt_boxes.shape[0], 7))
    matched_scores = gt_boxes.new_zeros(gt_boxes.shape[0])
    matched_counts = gt_boxes.new_zeros(gt_boxes.shape[0])
    matched_semantics = torch.zeros(gt_boxes.shape[0], dtype=torch.long, device=gt_boxes.device)
    proposal_valid = best_indices >= 0

    if proposal_valid.any():
        selected = best_indices[proposal_valid]
        selected_boxes = proposal_boxes[selected]
        centers = selected_boxes.mean(dim=1) + batch["scene_center_xyz"][0]
        sizes = (selected_boxes[:, 1] - selected_boxes[:, 0]).clamp_min(1e-4)
        semantic_ids = _proposal_semantics(output, proposal_indices)
        proposal_logits = output["proposal_scores"][0]
        proposal_offsets = output["proposal_scores"][2]
        original_indices = proposal_indices[selected]

        matched_features[proposal_valid] = features_3d[selected]
        matched_semantics[proposal_valid] = semantic_ids[selected]
        matched_bboxes[proposal_valid, :3] = centers
        matched_bboxes[proposal_valid, 3:6] = sizes
        matched_bboxes[proposal_valid, 6] = matched_semantics[proposal_valid].float()
        matched_scores[proposal_valid] = proposal_logits[original_indices]
        matched_counts[proposal_valid] = (
            proposal_offsets[original_indices + 1] - proposal_offsets[original_indices]
        ).float()

    scene_id = str(batch["scene_id"][0])
    output_path = output_dir / f"{scene_id}.npz"
    np.savez_compressed(
        output_path,
        obj_ids=obj_ids.detach().cpu().numpy().astype(np.int32),
        bboxes=matched_bboxes.detach().cpu().numpy().astype(np.float32),
        features_3d=matched_features.detach().cpu().numpy().astype(np.float32),
        scores=matched_scores.detach().cpu().numpy().astype(np.float32),
        point_counts=matched_counts.detach().cpu().numpy().astype(np.float32),
        semantic_ids=matched_semantics.detach().cpu().numpy().astype(np.int32),
        matched_ious=best_ious.detach().cpu().numpy().astype(np.float32),
        proposal_valid=proposal_valid.detach().cpu().numpy().astype(np.bool_),
    )
    return {
        "objects": int(obj_ids.numel()),
        "mean_matched_iou": float(best_ious.mean().item()) if best_ious.numel() else 0.0,
        "ts@0.5": float((best_ious >= 0.5).float().mean().item()) if best_ious.numel() else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage1 3D proposal information for Stage2 candidates.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = [
        "data=multi3drefer",
        "data.scene_augmentation.jitter_xyz=False",
        "data.scene_augmentation.flip_x=False",
        "data.scene_augmentation.rotate_z=False",
        "data.dataloader.num_workers=0",
        "model.stage2.enabled=False",
        "model.online_trajectory.enabled=False",
        "logger=null",
        f"project_root_path={PROJECT_ROOT}",
        *args.overrides,
    ]
    with hydra.initialize(version_base=None, config_path="../config"):
        cfg = hydra.compose(config_name="global_config", overrides=overrides)

    data_module = DataModule(cfg.data)
    dataset = data_module.dataset(cfg.data, args.split)
    indices = _first_chunk_per_scene(dataset)
    if args.max_scenes > 0:
        indices = indices[: args.max_scenes]
    loader = DataLoader(Subset(dataset, indices), batch_size=1, collate_fn=_collate_fn, num_workers=0)

    model = hydra.utils.instantiate(cfg.model.model_name, cfg)
    _load_checkpoint(model, args.ckpt)
    device = torch.device(args.device)
    model.to(device).eval()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_stats = []
    with torch.no_grad():
        for scene_index, batch in enumerate(loader, start=1):
            batch = _move_to_device(batch, device)
            stats = _export_scene(
                model, batch, output_dir, int(cfg.model.network.max_num_proposals)
            )
            scene_stats.append(stats)
            print(
                f"[{scene_index}/{len(indices)}] {batch['scene_id'][0]}: "
                f"objects={stats['objects']} mean_iou={stats['mean_matched_iou']:.4f} "
                f"ts@0.5={stats['ts@0.5']:.4f}"
            )

    total_objects = sum(item["objects"] for item in scene_stats)
    manifest = {
        "checkpoint": os.path.abspath(args.ckpt),
        "split": args.split,
        "num_scenes": len(scene_stats),
        "num_objects": total_objects,
        "mean_scene_matched_iou": float(np.mean([x["mean_matched_iou"] for x in scene_stats]))
        if scene_stats else 0.0,
        "mean_scene_ts@0.5": float(np.mean([x["ts@0.5"] for x in scene_stats]))
        if scene_stats else 0.0,
    }
    with open(output_dir / f"manifest_{args.split}.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
