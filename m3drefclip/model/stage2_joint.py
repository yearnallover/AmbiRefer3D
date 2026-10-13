from __future__ import annotations

import os
import sys
import copy
import random
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from m3drefclip.loss.trajectory_loss import MetricDrivenLoss, PreferenceRankingLoss


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "y")
    return bool(value)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return getattr(cfg, key, default) if cfg is not None else default


def _move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = _move_to_device(value, device)
        else:
            moved[key] = value
    return moved


class TrajectoryPairDataset(torch.utils.data.Dataset):
    def __init__(self, round_dataset, trajectories: List[List[int]]):
        self.round_dataset = round_dataset
        self.trajectories = trajectories

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int):
        return [self.round_dataset[i] for i in self.trajectories[idx]]


def _resolve_path(path: Any, project_root: Any = None) -> str:
    if path is None:
        return ""
    path = str(path)
    if not path or os.path.isabs(path):
        return path
    candidates = [os.path.abspath(path)]
    if project_root:
        candidates.append(os.path.abspath(os.path.join(str(project_root), path)))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def _looks_like_stage2_root(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "model", "stage2_net_semantic.py"))


def _resolve_stage2_root(stage2_cfg: Any, project_root: Any = None) -> str:
    configured = _resolve_path(_cfg_get(stage2_cfg, "root_path", ""), project_root)
    if configured and os.path.isdir(configured) and _looks_like_stage2_root(configured):
        return configured

    search_roots = []
    if configured and os.path.isdir(configured):
        search_roots.append(configured)
    if project_root:
        search_roots.append(os.path.join(str(project_root), "stage2_refactor"))
    search_roots.append(os.path.abspath("stage2_refactor"))

    for search_root in search_roots:
        if not os.path.isdir(search_root):
            continue
        for root, _, files in os.walk(search_root):
            if "stage2_net_semantic.py" in files and os.path.basename(root) == "model":
                candidate = os.path.dirname(root)
                if _looks_like_stage2_root(candidate):
                    return candidate

    if configured:
        raise FileNotFoundError(
            f"Could not find a valid stage2 root. Configured path: {configured}. "
            "A valid root must contain model/stage2_net_semantic.py."
        )
    raise ValueError("model.stage2.root_path must be set when stage2 is enabled.")


class Stage2JointModule(nn.Module):
    def __init__(self, stage2_cfg: Any, project_root_path: Any = None):
        super().__init__()
        self.stage2_cfg = stage2_cfg
        project_root = project_root_path
        stage2_root = _resolve_stage2_root(stage2_cfg, project_root)
        if stage2_root not in sys.path:
            sys.path.insert(0, stage2_root)

        from stage2_refactor.config import TrainConfig
        from stage2_refactor.data.bbox_provider import (
            PassthroughBBoxProvider,
            ScanNetBBoxProvider,
        )
        from stage2_refactor.data.collate_semantic import collate_semantic
        from stage2_refactor.data.dataset_semantic import RoundDatasetSemantic
        from stage2_refactor.data.vocab import Vocab
        from stage2_refactor.learning.constraint_table import ConstraintTable
        from stage2_refactor.learning.losses_semantic import (
            candidate_ce_loss,
            candidate_filter_loss,
            selector_ce_loss,
            stop_bce_loss,
        )
        from stage2_refactor.model.stage2_net_semantic import Stage2NetSemantic

        self.candidate_ce_loss = candidate_ce_loss
        self.candidate_filter_loss = candidate_filter_loss
        self.selector_ce_loss = selector_ce_loss
        self.stop_bce_loss = stop_bce_loss
        self.collate_semantic = collate_semantic

        round_json_path = _resolve_path(_cfg_get(stage2_cfg, "round_json_path"), project_root)
        if not round_json_path:
            raise ValueError("model.stage2.round_json_path must be set when stage2 is enabled.")
        self.samples = self._load_json(round_json_path)
        if not self.samples:
            raise ValueError(f"No stage2 samples found in {round_json_path}")
        self.supervision_mode = str(
            _cfg_get(stage2_cfg, "supervision_mode", "ordered")
        ).lower()
        if self.supervision_mode not in {
            "ordered", "independent_pairs", "shuffled_pairs"
        }:
            raise ValueError(
                "model.stage2.supervision_mode must be ordered, "
                f"independent_pairs, or shuffled_pairs; got {self.supervision_mode!r}."
            )
        self.independent_pair_seed = int(
            _cfg_get(stage2_cfg, "independent_pair_seed", 123)
        )
        end_to_end_cfg = _cfg_get(stage2_cfg, "end_to_end", None)
        self.end_to_end_enabled = _as_bool(_cfg_get(end_to_end_cfg, "enabled", False))
        self.end_to_end_weight = float(_cfg_get(end_to_end_cfg, "weight", 1.0))
        self.end_to_end_feature_key = str(
            _cfg_get(end_to_end_cfg, "feature_key", "aabb_features_3d")
        )
        self.end_to_end_require_aligned_batch = _as_bool(
            _cfg_get(end_to_end_cfg, "require_aligned_batch", True)
        )
        self.end_to_end_use_hard_alive_mask = _as_bool(
            _cfg_get(end_to_end_cfg, "use_hard_alive_mask", False)
        )
        self.end_to_end_teacher_force_candidate_state = _as_bool(
            _cfg_get(end_to_end_cfg, "teacher_force_candidate_state", True)
        )
        default_state_mode = (
            "teacher" if self.end_to_end_teacher_force_candidate_state else "model"
        )
        configured_state_mode = str(
            _cfg_get(end_to_end_cfg, "candidate_state_mode", "auto")
        ).lower()
        self.end_to_end_candidate_state_mode = (
            default_state_mode
            if configured_state_mode in {"", "auto", "none"}
            else configured_state_mode
        )
        if self.end_to_end_candidate_state_mode not in {
            "teacher",
            "model",
            "union",
            "initial_union_then_teacher",
        }:
            raise ValueError(
                "model.stage2.end_to_end.candidate_state_mode must be one of "
                "teacher, model, union, or initial_union_then_teacher."
            )
        self.end_to_end_preserve_scene_context = _as_bool(
            _cfg_get(end_to_end_cfg, "preserve_scene_context", False)
        )
        self.end_to_end_use_initial_stage1_state = _as_bool(
            _cfg_get(end_to_end_cfg, "use_initial_stage1_state", False)
        )
        self.end_to_end_stage1_gradient_scale = float(
            _cfg_get(end_to_end_cfg, "stage1_gradient_scale", 1.0)
        )
        if not 0.0 <= self.end_to_end_stage1_gradient_scale <= 1.0:
            raise ValueError(
                "model.stage2.end_to_end.stage1_gradient_scale must be in [0, 1]."
            )
        self.end_to_end_autoregressive_state_training = _as_bool(
            _cfg_get(end_to_end_cfg, "autoregressive_state_training", False)
        )
        self.end_to_end_autoregressive_teacher_mix = float(
            _cfg_get(end_to_end_cfg, "autoregressive_teacher_mix", 0.5)
        )
        if not 0.0 <= self.end_to_end_autoregressive_teacher_mix <= 1.0:
            raise ValueError(
                "model.stage2.end_to_end.autoregressive_teacher_mix must be in [0, 1]."
            )

        rel2id = dict(_cfg_get(stage2_cfg, "rel2id", None) or {
            "near": 0,
            "behind": 1,
            "in_front_of": 2,
            "left_of": 3,
            "right_of": 4,
            "next_to": 5,
        })
        attr2id = dict(_cfg_get(stage2_cfg, "attr2id", None) or {
            "color": 0,
            "material": 1,
            "shape": 2,
            "size": 3,
            "height": 4,
            "other": 5,
        })
        max_len = int(_cfg_get(stage2_cfg, "max_len", 16))

        spec_path = _resolve_path(_cfg_get(stage2_cfg, "spec_path"), project_root)
        if spec_path and os.path.exists(spec_path):
            from utils.spec import load_spec

            spec = load_spec(spec_path)
            self.vocab = Vocab.from_itos(spec.vocab_itos)
            self.constraint_table = ConstraintTable(keys=spec.constraint_keys, rel2id=spec.rel2id, attr2id=spec.attr2id)
        else:
            self.vocab = self._build_vocab(Vocab, self.samples)
            self.constraint_table = ConstraintTable.build_from_samples(self.samples, rel2id=rel2id, attr2id=attr2id)
        self.max_len = max_len
        self.high_frequency_constraint_ids = self._build_high_frequency_constraint_ids(self.samples)

        train_cfg = TrainConfig(
            max_len=max_len,
            img_cache_dir=_resolve_path(_cfg_get(stage2_cfg, "img_cache_dir", ""), project_root),
            imgfeat_dim=int(_cfg_get(stage2_cfg, "imgfeat_dim", 768)),
            use_imgfeat=_as_bool(_cfg_get(stage2_cfg, "use_imgfeat", True)),
            use_bbox_geom=_as_bool(_cfg_get(stage2_cfg, "use_bbox_geom", False)),
            use_relative_geom=_as_bool(_cfg_get(stage2_cfg, "use_relative_geom", False)),
            use_semantic_label=_as_bool(_cfg_get(stage2_cfg, "use_semantic_label", False)),
            num_semantic_classes=int(_cfg_get(stage2_cfg, "num_semantic_classes", 256)),
            use_stage1_features=_as_bool(_cfg_get(stage2_cfg, "use_stage1_features", False)),
            stage1_feat_dim=int(_cfg_get(stage2_cfg, "stage1_feat_dim", 128)),
            use_stage1_quality=_as_bool(_cfg_get(stage2_cfg, "use_stage1_quality", False)),
            use_scene_context_attention=_as_bool(
                _cfg_get(stage2_cfg, "use_scene_context_attention", False)
            ),
            use_constraint_conditioned_context=_as_bool(
                _cfg_get(stage2_cfg, "use_constraint_conditioned_context", False)
            ),
            selector_pre_action_state=_as_bool(
                _cfg_get(stage2_cfg, "selector_pre_action_state", False)
            ),
            stage1_cache_dir=_resolve_path(_cfg_get(stage2_cfg, "stage1_cache_dir", ""), project_root),
            stage1_feature_key=str(_cfg_get(stage2_cfg, "stage1_feature_key", "features_3d")),
            use_stage1_cache_bboxes=_as_bool(
                _cfg_get(stage2_cfg, "use_stage1_cache_bboxes", False)
            ),
            require_stage1_cache=_as_bool(_cfg_get(stage2_cfg, "require_stage1_cache", False)),
            batch_size=int(_cfg_get(stage2_cfg, "batch_size", 64)),
            lr=float(_cfg_get(stage2_cfg, "lr", 3e-4)),
            weight_decay=float(_cfg_get(stage2_cfg, "weight_decay", 1e-2)),
            seed=int(_cfg_get(stage2_cfg, "seed", 42)),
        )
        self.train_cfg = train_cfg

        bbox_source = str(_cfg_get(stage2_cfg, "bbox_source", "passthrough")).lower()
        if bbox_source == "passthrough":
            bbox_provider = PassthroughBBoxProvider()
        elif bbox_source == "scannet":
            scene_data_root = _resolve_path(_cfg_get(stage2_cfg, "scene_data_root", ""), project_root)
            if not scene_data_root:
                raise ValueError("model.stage2.scene_data_root must be set when bbox_source=scannet.")
            bbox_provider = ScanNetBBoxProvider(
                scene_data_root,
                split=str(_cfg_get(stage2_cfg, "scene_split", "auto")),
            )
        else:
            raise ValueError(f"Unknown model.stage2.bbox_source={bbox_source!r}")

        dataset = RoundDatasetSemantic(
            samples=self.samples,
            vocab=self.vocab,
            max_len=max_len,
            bbox_provider=bbox_provider,
            img_cache_dir=train_cfg.img_cache_dir,
            imgfeat_dim=train_cfg.imgfeat_dim,
            use_imgfeat=train_cfg.use_imgfeat,
            use_bbox_geom=train_cfg.use_bbox_geom,
            use_relative_geom=train_cfg.use_relative_geom,
            use_semantic_label=train_cfg.use_semantic_label,
            use_stage1_features=train_cfg.use_stage1_features,
            stage1_feat_dim=train_cfg.stage1_feat_dim,
            use_stage1_quality=train_cfg.use_stage1_quality,
            stage1_cache_dir=train_cfg.stage1_cache_dir,
            stage1_feature_key=train_cfg.stage1_feature_key,
            use_stage1_cache_bboxes=train_cfg.use_stage1_cache_bboxes,
            require_stage1_cache=train_cfg.require_stage1_cache,
            constraint_table=self.constraint_table,
        )
        self.round_dataset = dataset
        trajectories = self._build_trajectories(
            self.samples,
            require_distinct_rounds=_as_bool(
                _cfg_get(stage2_cfg, "trajectory_require_distinct_rounds", True)
            ),
        )
        self.end_to_end_trajectory_lookup = {}
        self.end_to_end_fallback_lookup = {}
        for trajectory in trajectories:
            if not trajectory:
                continue
            key = self._trajectory_key(self.samples[trajectory[0]])
            self.end_to_end_trajectory_lookup[key] = trajectory
            self.end_to_end_fallback_lookup.setdefault(key[:3], []).append(trajectory)
        self.trajectory_loader = DataLoader(
            TrajectoryPairDataset(dataset, trajectories),
            batch_size=int(_cfg_get(stage2_cfg, "trajectory_batch_size", _cfg_get(stage2_cfg, "batch_size", 64))),
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_trajectory_batch,
            drop_last=False,
        ) if trajectories else None
        self.trajectory_loader_iter = None
        self.loader = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=int(_cfg_get(stage2_cfg, "num_workers", 0)),
            collate_fn=self.collate_semantic,
            drop_last=_as_bool(_cfg_get(stage2_cfg, "drop_last", False)),
        )
        self.loader_iter = None
        self.num_constraints = self.constraint_table.num_constraints()
        self.end_to_end_available_constraint_ids = {}
        self.end_to_end_selector_target_ids = {}
        for sample in self.samples:
            key = self._trajectory_key(sample)[:3]
            constraint_id = self.constraint_table.id_from_constraint_gt(
                sample.get("constraint_gt") or {}
            )
            self.end_to_end_available_constraint_ids.setdefault(key, set()).add(
                int(constraint_id)
            )
            state_key = key + (
                tuple(sorted(int(value) for value in sample.get("cand_ids") or [])),
            )
            self.end_to_end_selector_target_ids.setdefault(state_key, set()).add(
                int(constraint_id)
            )
        self.selector_return_discount = float(
            _cfg_get(stage2_cfg, "selector_return_discount", 0.9)
        )
        self.selector_step_cost = float(
            _cfg_get(stage2_cfg, "selector_step_cost", 0.1)
        )
        self.selector_information_gain_weight = float(
            _cfg_get(stage2_cfg, "selector_information_gain_weight", 0.25)
        )
        self.selector_utility_temperature = max(
            float(_cfg_get(stage2_cfg, "selector_utility_temperature", 0.1)),
            1e-6,
        )
        self.end_to_end_selector_action_utilities = (
            self._build_selector_action_utilities(trajectories)
        )
        constraint_rows = self.constraint_table.rows
        constraint_bank = {
            "ctype_ids": torch.tensor(
                [row.ctype_id for row in constraint_rows], dtype=torch.long
            ),
            "rel_ids": torch.tensor(
                [row.rel_id for row in constraint_rows], dtype=torch.long
            ),
            "attr_ids": torch.tensor(
                [row.attr_id for row in constraint_rows], dtype=torch.long
            ),
            "field_ids": torch.tensor(
                [
                    self.vocab.encode(row.field_text, max_len)
                    for row in constraint_rows
                ],
                dtype=torch.long,
            ),
        }
        self.net = Stage2NetSemantic(
            vocab_size=len(self.vocab.itos),
            cfg=train_cfg,
            num_constraints=self.num_constraints,
            num_rel=max(rel2id.values()) + 1,
            num_attr=max(attr2id.values()) + 1,
            constraint_bank=constraint_bank,
            use_semantic_constraint_scoring=_as_bool(
                _cfg_get(stage2_cfg, "use_semantic_constraint_scoring", True)
            ),
        )

        pretrained = _resolve_path(_cfg_get(stage2_cfg, "pretrained_path", ""), project_root)
        if pretrained:
            self._load_pretrained(pretrained)

        self.w_selector = float(_cfg_get(stage2_cfg, "w_selector", 1.0))
        self.w_mask = float(_cfg_get(stage2_cfg, "w_mask", 1.0))
        self.mask_ce_weight = float(_cfg_get(stage2_cfg, "mask_ce_weight", 0.5))
        self.mask_bce_weight = float(_cfg_get(stage2_cfg, "mask_bce_weight", 1.0))
        self.mask_positive_weight = float(
            _cfg_get(stage2_cfg, "mask_positive_weight", 2.0)
        )
        self.mask_negative_weight = float(
            _cfg_get(stage2_cfg, "mask_negative_weight", 0.25)
        )
        self.mask_decision_threshold = float(
            _cfg_get(
                stage2_cfg,
                "mask_decision_threshold",
                _cfg_get(stage2_cfg, "validation_thr", 0.4),
            )
        )
        if not 0.0 < self.mask_decision_threshold < 1.0:
            raise ValueError("model.stage2.mask_decision_threshold must be in (0, 1).")
        self.w_stop = float(_cfg_get(stage2_cfg, "w_stop", 0.2))
        self.w_preference = float(_cfg_get(stage2_cfg, "w_preference", 1.0))
        self.w_metric = float(_cfg_get(stage2_cfg, "w_metric", 1.0))
        self.use_selector_loss = _as_bool(_cfg_get(stage2_cfg, "use_selector_loss", True))
        self.use_mask_loss = _as_bool(_cfg_get(stage2_cfg, "use_mask_loss", True))
        self.use_stop_loss = _as_bool(_cfg_get(stage2_cfg, "use_stop_loss", True))
        self.use_preference_loss = _as_bool(_cfg_get(stage2_cfg, "use_preference_loss", False))
        self.use_metric_loss = _as_bool(_cfg_get(stage2_cfg, "use_metric_loss", False))
        self.use_counterfactual_trajectory_loss = _as_bool(
            _cfg_get(stage2_cfg, "use_counterfactual_trajectory_loss", False)
        )
        self.use_trajectory_dpo_loss = _as_bool(_cfg_get(stage2_cfg, "use_trajectory_dpo_loss", False))
        self.trajectory_dpo_use_reference = _as_bool(_cfg_get(stage2_cfg, "trajectory_dpo_use_reference", False))
        self.trajectory_require_explicit_negatives = _as_bool(
            _cfg_get(stage2_cfg, "trajectory_require_explicit_negatives", False)
        )
        if (
            self.supervision_mode != "ordered"
            and self.use_trajectory_dpo_loss
        ):
            raise ValueError(
                "Trajectory DPO is incompatible with independent-pair supervision. "
                "Set model.stage2.use_trajectory_dpo_loss=False."
            )
        if (
            self.supervision_mode != "ordered"
            and self.end_to_end_autoregressive_state_training
        ):
            raise ValueError(
                "Autoregressive state training is incompatible with independent-pair "
                "supervision. Set autoregressive_state_training=False."
            )
        self.preference_loss = PreferenceRankingLoss(
            margin=float(_cfg_get(stage2_cfg, "preference_margin", 0.2)),
            temperature=float(_cfg_get(stage2_cfg, "preference_temperature", 1.0)),
            hard_negative_topk=int(_cfg_get(stage2_cfg, "preference_hard_negative_topk", 16)),
        )
        self.metric_loss = MetricDrivenLoss(
            rank_temperature=float(_cfg_get(stage2_cfg, "metric_rank_temperature", 0.2)),
            f1_weight=float(_cfg_get(stage2_cfg, "metric_f1_weight", 1.0)),
            rank_weight=float(_cfg_get(stage2_cfg, "metric_rank_weight", 1.0)),
        )
        self.w_counterfactual = float(_cfg_get(stage2_cfg, "w_counterfactual", 1.0))
        self.w_trajectory_dpo = float(_cfg_get(stage2_cfg, "w_trajectory_dpo", 1.0))
        self.counterfactual_margin = float(_cfg_get(stage2_cfg, "counterfactual_margin", 0.2))
        self.trajectory_dpo_beta = float(_cfg_get(stage2_cfg, "trajectory_dpo_beta", 0.1))
        self.trajectory_grounding_logprob_weight = float(
            _cfg_get(stage2_cfg, "trajectory_grounding_logprob_weight", 1.0)
        )
        self.trajectory_action_min_utility_gap = max(
            float(
                _cfg_get(
                    stage2_cfg, "trajectory_action_min_utility_gap", 0.01
                )
            ),
            0.0,
        )
        self.validation_enabled = _as_bool(
            _cfg_get(stage2_cfg, "validation_enabled", False)
        )
        self.validation_max_round = int(
            _cfg_get(stage2_cfg, "validation_max_round", 8)
        )
        self.validation_thr = float(_cfg_get(stage2_cfg, "validation_thr", 0.4))
        self.validation_stop_thr = float(
            _cfg_get(stage2_cfg, "validation_stop_thr", 0.5)
        )
        self.validation_eval_mode = str(
            _cfg_get(stage2_cfg, "validation_eval_mode", "agent")
        )
        self.validation_v_target_strategy = str(
            _cfg_get(stage2_cfg, "validation_v_target_strategy", "top1")
        )
        self.validation_selector_strategy = str(
            _cfg_get(stage2_cfg, "validation_selector_strategy", "ours")
        )
        self.validation_risk_sensitive_pruning = _as_bool(
            _cfg_get(stage2_cfg, "validation_risk_sensitive_pruning", True)
        )
        self.validation_min_keep_ratio = float(
            _cfg_get(stage2_cfg, "validation_min_keep_ratio", 0.5)
        )
        self.validation_min_keep_count = int(
            _cfg_get(stage2_cfg, "validation_min_keep_count", 1)
        )
        self.lr = float(_cfg_get(stage2_cfg, "lr", 3e-4))
        self.weight_decay = float(_cfg_get(stage2_cfg, "weight_decay", 1e-2))
        if self.use_trajectory_dpo_loss and self.trajectory_loader is None:
            raise ValueError("No valid stage2 trajectories were built, but use_trajectory_dpo_loss=True.")
        self.reference_net = None
        if self.use_trajectory_dpo_loss and self.trajectory_dpo_use_reference:
            self.refresh_reference_model()

    def refresh_reference_model(self):
        if not self.use_trajectory_dpo_loss or not self.trajectory_dpo_use_reference:
            return
        self.reference_net = copy.deepcopy(self.net)
        self.reference_net.eval()
        for param in self.reference_net.parameters():
            param.requires_grad = False

    def next_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if self.loader_iter is None:
            self.loader_iter = iter(self.loader)
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.loader)
            batch = next(self.loader_iter)
        return _move_to_device(batch, device)

    def next_trajectory_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if self.trajectory_loader is None:
            raise ValueError("Trajectory loader is not available.")
        if self.trajectory_loader_iter is None:
            self.trajectory_loader_iter = iter(self.trajectory_loader)
        try:
            batch = next(self.trajectory_loader_iter)
        except StopIteration:
            self.trajectory_loader_iter = iter(self.trajectory_loader)
            batch = next(self.trajectory_loader_iter)
        return _move_to_device(batch, device)

    def _uses_independent_pair_training(self) -> bool:
        return self.training and self.supervision_mode != "ordered"

    @staticmethod
    def _shuffle_independent_rows(rows: List[Any], seed: int) -> List[Any]:
        rows = list(rows)
        if len(rows) < 2:
            return rows
        indices = list(range(len(rows)))
        rng = random.Random(int(seed))
        rng.shuffle(indices)
        if indices == list(range(len(rows))):
            indices = indices[1:] + indices[:1]
        return [rows[index] for index in indices]

    def _prepare_runtime_masks(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = batch["cand_mask"].shape[0]
        if "alive_mask" not in batch:
            batch["alive_mask"] = batch["cand_mask"]
        if "used_constraint_mask" not in batch:
            batch["used_constraint_mask"] = torch.zeros(
                (batch_size, self.num_constraints),
                dtype=batch["cand_mask"].dtype,
                device=batch["cand_mask"].device,
            )
        available = batch.get("available_constraint_mask")
        if available is not None:
            batch["used_constraint_mask"] = torch.maximum(
                batch["used_constraint_mask"].float(),
                (~available.bool()).float(),
            )
        return batch

    @staticmethod
    def _decision_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Candidates that the policy may retain or select in the current state."""
        return (
            batch.get("alive_mask", batch["cand_mask"]).gt(0.5)
            & batch["cand_mask"].bool()
        )

    @staticmethod
    def _scale_stage1_gradient(
        value: torch.Tensor, scale: float
    ) -> torch.Tensor:
        """Preserve Stage1 values while controlling downstream credit magnitude."""
        scale = float(scale)
        if not value.is_floating_point() or not value.requires_grad or scale == 1.0:
            return value
        if scale == 0.0:
            return value.detach()
        detached = value.detach()
        return detached + scale * (value - detached)

    @staticmethod
    def _gather_dynamic_trajectory_state(
        trajectory_output: Dict[str, torch.Tensor],
        scene_indices: torch.Tensor,
        language_indices: torch.Tensor,
        round_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the post-query belief state that precedes each Stage2 action."""
        alive = trajectory_output["alive_history"][
            scene_indices, language_indices, round_indices
        ].clamp(0.0, 1.0)
        belief = trajectory_output["belief_history"][
            scene_indices, language_indices, round_indices
        ]
        return alive, belief

    @staticmethod
    def _merge_candidate_state_masks(
        teacher_mask: torch.Tensor,
        model_alive: torch.Tensor,
        proposal_mask: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Build an exposure-resistant state without ever inventing proposals."""
        teacher = teacher_mask.bool() & proposal_mask.bool()
        predicted = model_alive.ge(0.5) & proposal_mask.bool()
        if mode == "teacher":
            return teacher
        if mode == "model":
            return predicted
        if mode == "union":
            # Teacher candidates protect the target; predicted false positives
            # expose Stage2 to the states it will actually see during rollout.
            return teacher | predicted
        raise ValueError(f"Unknown candidate state mode: {mode}")

    @classmethod
    def _trajectory_candidate_state_masks(
        cls,
        teacher_mask: torch.Tensor,
        model_alive: torch.Tensor,
        proposal_mask: torch.Tensor,
        round_indices: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if mode != "initial_union_then_teacher":
            return cls._merge_candidate_state_masks(
                teacher_mask, model_alive, proposal_mask, mode
            )
        initial_union = cls._merge_candidate_state_masks(
            teacher_mask, model_alive, proposal_mask, "union"
        )
        teacher = teacher_mask.bool() & proposal_mask.bool()
        return torch.where(
            round_indices.eq(0).unsqueeze(1), initial_union, teacher
        )

    @staticmethod
    def _return_conditioned_rejected_action(
        action_utilities: Dict[int, float],
        current_action: int,
        valid_actions: Iterable[int],
        min_utility_gap: float,
    ) -> tuple[int, bool]:
        """Choose a lower-return action observed from the same candidate state."""
        current_action = int(current_action)
        current_utility = action_utilities.get(current_action)
        if current_utility is None:
            return current_action, False
        valid_actions = {int(value) for value in valid_actions}
        lower_return = [
            (float(action_utilities[action_id]), int(action_id))
            for action_id in action_utilities
            if int(action_id) in valid_actions
            and int(action_id) != current_action
            and float(current_utility) - float(action_utilities[action_id])
            >= float(min_utility_gap)
        ]
        if not lower_return:
            return current_action, False
        _, rejected_action = min(lower_return)
        return rejected_action, True

    @staticmethod
    def _post_action_filter_targets(
        candidate_labels: torch.Tensor | None,
        proposal_targets: torch.Tensor,
        batch_idx: int,
        lang_idx: int,
        round_idx: int,
        final_round_idx: int,
        terminal: bool,
    ) -> torch.Tensor:
        """Return the candidate set after the current clarification action."""
        if terminal or candidate_labels is None:
            return proposal_targets
        next_round_idx = min(int(round_idx) + 1, int(final_round_idx))
        targets = candidate_labels[batch_idx, lang_idx, next_round_idx].bool()
        return targets if targets.any() else proposal_targets

    def _autoregressive_trajectory_forward(
        self,
        net: nn.Module,
        batch: Dict[str, torch.Tensor],
        lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Roll predicted filtering evidence into the next training state."""
        lengths = lengths.long()
        num_rows = int(batch["cand_mask"].shape[0])
        if int(lengths.sum().item()) != num_rows:
            raise ValueError("Trajectory lengths do not cover the flattened Stage2 batch.")
        starts = torch.cat(
            (
                lengths.new_zeros(1),
                torch.cumsum(lengths, dim=0)[:-1],
            )
        )
        teacher_mix = float(
            getattr(self, "end_to_end_autoregressive_teacher_mix", 0.5)
        )
        collected_indices = []
        collected_outputs: Dict[str, List[torch.Tensor]] = {}
        previous_active_ids = None
        previous_belief = None
        previous_alive_probability = None
        previous_mask_logits = None

        for position in range(int(lengths.max().item())):
            active_ids = torch.nonzero(
                lengths > position, as_tuple=False
            ).squeeze(1)
            row_indices = starts.index_select(0, active_ids) + position
            row_batch = {
                key: value.index_select(0, row_indices)
                if torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] == num_rows
                else value
                for key, value in batch.items()
            }
            if "cand_alive_probs" not in row_batch:
                row_batch["cand_alive_probs"] = row_batch.get(
                    "alive_mask", row_batch["cand_mask"]
                ).float()
            if "cand_belief" not in row_batch:
                prior_scores = row_batch.get("cand_stage1_scores")
                if prior_scores is None:
                    prior_scores = torch.zeros_like(row_batch["cand_mask"])
                prior_scores = prior_scores.float().masked_fill(
                    row_batch["cand_alive_probs"].le(0), -1e9
                )
                row_batch["cand_belief"] = torch.softmax(prior_scores, dim=1)
            if position > 0:
                previous_positions = torch.searchsorted(
                    previous_active_ids, active_ids
                )
                prior_belief = previous_belief.index_select(
                    0, previous_positions
                )
                prior_alive = previous_alive_probability.index_select(
                    0, previous_positions
                )
                prior_probability = torch.sigmoid(
                    previous_mask_logits.index_select(0, previous_positions)
                )
                predicted_alive = (prior_alive * prior_probability).clamp(0.0, 1.0)
                predicted_belief = prior_belief * prior_probability
                predicted_belief = predicted_belief / predicted_belief.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-6)
                teacher_belief = row_batch["cand_belief"].float()
                teacher_alive = row_batch["cand_alive_probs"].float()
                mixed_belief = (
                    teacher_mix * teacher_belief
                    + (1.0 - teacher_mix) * predicted_belief
                )
                row_batch["cand_belief"] = mixed_belief / mixed_belief.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-6)
                row_batch["cand_alive_probs"] = (
                    teacher_mix * teacher_alive
                    + (1.0 - teacher_mix) * predicted_alive
                ).clamp(0.0, 1.0)

            round_output = net(row_batch)
            collected_indices.append(row_indices)
            for key, value in round_output.items():
                collected_outputs.setdefault(key, []).append(value)
            previous_active_ids = active_ids
            previous_belief = row_batch["cand_belief"]
            previous_alive_probability = row_batch["cand_alive_probs"]
            previous_mask_logits = round_output["mask_logits"]

        flat_indices = torch.cat(collected_indices, dim=0)
        restore_order = torch.argsort(flat_indices)
        return {
            key: torch.cat(values, dim=0).index_select(0, restore_order)
            for key, values in collected_outputs.items()
        }

    def loss(
        self,
        batch: Dict[str, torch.Tensor],
        trajectory_batch: Dict[str, torch.Tensor] = None,
        use_offline_trajectory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch = self._prepare_runtime_masks(batch)
        trajectory_lengths = (
            trajectory_batch.get("lengths")
            if trajectory_batch is not None
            else None
        )
        if (
            getattr(self, "end_to_end_autoregressive_state_training", False)
            and trajectory_lengths is not None
        ):
            out = self._autoregressive_trajectory_forward(
                self.net, batch, trajectory_lengths
            )
        else:
            out = self.net(batch)
        decision_mask = self._decision_mask(batch)
        selector_rows = decision_mask.any(dim=1)
        true_target_candidate_mask = batch.get(
            "target_candidate_mask", batch.get("mask_y")
        )
        filter_candidate_mask = batch.get(
            "filter_candidate_mask", true_target_candidate_mask
        )
        grounding_rows = (
            filter_candidate_mask.bool() & decision_mask
        ).any(dim=1)
        if selector_rows.any():
            selector_target_mask = batch.get("selector_target_mask")
            selector_loss = self.selector_ce_loss(
                out["selector_logits"][selector_rows],
                batch["constraint_id"][selector_rows],
                target_mask=(
                    selector_target_mask[selector_rows]
                    if selector_target_mask is not None
                    else None
                ),
                target_distribution=(
                    batch["selector_target_distribution"][selector_rows]
                    if "selector_target_distribution" in batch
                    else None
                ),
            )
        else:
            selector_loss = out["selector_logits"].sum() * 0.0
        if grounding_rows.any():
            mask_loss = self.candidate_filter_loss(
                out["mask_logits"][grounding_rows],
                batch["target_index"][grounding_rows],
                decision_mask[grounding_rows],
                target_mask=filter_candidate_mask[grounding_rows],
                ce_weight=self.mask_ce_weight,
                bce_weight=self.mask_bce_weight,
                positive_weight=self.mask_positive_weight,
                negative_weight=self.mask_negative_weight,
                decision_threshold=getattr(
                    self,
                    "mask_decision_threshold",
                    getattr(self, "validation_thr", 0.5),
                ),
            )
        else:
            mask_loss = out["mask_logits"].sum() * 0.0
        losses = {
            "stage2_selector_loss": selector_loss,
            "stage2_mask_loss": mask_loss,
            "stage2_stop_loss": self.stop_bce_loss(out["stop_logit"], batch["stop_y"]),
        }
        if self.use_preference_loss:
            losses["stage2_preference_loss"] = (
                self.preference_loss(
                    out["mask_logits"][grounding_rows],
                    labels=filter_candidate_mask[grounding_rows],
                    valid_mask=decision_mask[grounding_rows],
                )
                if grounding_rows.any()
                else out["mask_logits"].sum() * 0.0
            )
        else:
            losses["stage2_preference_loss"] = out["mask_logits"].sum() * 0.0

        if self.use_metric_loss:
            losses["stage2_metric_loss"] = (
                self.metric_loss(
                    out["mask_logits"][grounding_rows],
                    labels=filter_candidate_mask[grounding_rows],
                    valid_mask=decision_mask[grounding_rows],
                )
                if grounding_rows.any()
                else out["mask_logits"].sum() * 0.0
            )
        else:
            losses["stage2_metric_loss"] = out["mask_logits"].sum() * 0.0

        if self.use_counterfactual_trajectory_loss:
            losses.update(self._counterfactual_trajectory_losses(batch, out))
        else:
            zero = out["mask_logits"].sum() * 0.0
            losses["stage2_wrong_constraint_loss"] = zero
            losses["stage2_low_info_constraint_loss"] = zero
            losses["stage2_early_stop_loss"] = zero

        if self.use_trajectory_dpo_loss:
            if trajectory_batch is not None:
                losses.update(self._trajectory_dpo_losses(trajectory_batch))
            elif use_offline_trajectory:
                offline_trajectory_batch = self.next_trajectory_batch(batch["cand_mask"].device)
                losses.update(self._trajectory_dpo_losses(offline_trajectory_batch))
            else:
                zero = out["mask_logits"].sum() * 0.0
                losses["stage2_trajectory_dpo_loss"] = zero
                losses["stage2_dpo_wrong_loss"] = zero
                losses["stage2_dpo_low_info_loss"] = zero
                losses["stage2_dpo_early_stop_loss"] = zero
        else:
            zero = out["mask_logits"].sum() * 0.0
            losses["stage2_trajectory_dpo_loss"] = zero
            losses["stage2_dpo_wrong_loss"] = zero
            losses["stage2_dpo_low_info_loss"] = zero
            losses["stage2_dpo_early_stop_loss"] = zero

        losses["stage2_loss"] = (
            self.w_selector * losses["stage2_selector_loss"] * float(self.use_selector_loss)
            + self.w_mask * losses["stage2_mask_loss"] * float(self.use_mask_loss)
            + self.w_stop * losses["stage2_stop_loss"] * float(self.use_stop_loss)
            + self.w_preference * losses["stage2_preference_loss"]
            + self.w_metric * losses["stage2_metric_loss"]
            + self.w_counterfactual * (
                losses["stage2_wrong_constraint_loss"]
                + losses["stage2_low_info_constraint_loss"]
                + losses["stage2_early_stop_loss"]
            )
            + self.w_trajectory_dpo * losses["stage2_trajectory_dpo_loss"]
        )
        return losses

    def _trajectory_dpo_losses(self, trajectory_batch: Dict[str, torch.Tensor]):
        positive_batch = self._prepare_runtime_masks(trajectory_batch["flat_batch"])
        lengths = trajectory_batch["lengths"].long()
        positive_out = self._deterministic_trajectory_dpo_forward(
            self.net, positive_batch, lengths
        )

        explicit_negative_batch = trajectory_batch.get("negative_flat_batch")
        if explicit_negative_batch is not None:
            negative_batch = self._prepare_runtime_masks(explicit_negative_batch)
            negative_out = self._deterministic_trajectory_dpo_forward(
                self.net, negative_batch, lengths
            )
            preference_step_mask = trajectory_batch.get(
                "preference_step_mask"
            )
            if preference_step_mask is None:
                preference_step_mask = torch.ones_like(
                    positive_batch["stop_y"], dtype=torch.bool
                )
            preference_trajectory_mask = self._sum_by_lengths(
                preference_step_mask.float(), lengths
            ).gt(0)
            positive_logp = self._trajectory_logprob(
                self.net,
                positive_batch,
                positive_out,
                lengths,
                step_mask=preference_step_mask,
            )
            negative_logp = self._trajectory_logprob(
                self.net,
                negative_batch,
                negative_out,
                lengths,
                step_mask=preference_step_mask,
            )
            if self.reference_net is not None:
                with torch.no_grad():
                    ref_positive_out = self._deterministic_trajectory_dpo_forward(
                        self.reference_net, positive_batch, lengths
                    )
                    ref_negative_out = self._deterministic_trajectory_dpo_forward(
                        self.reference_net, negative_batch, lengths
                    )
                    ref_positive_logp = self._trajectory_logprob(
                        self.reference_net,
                        positive_batch,
                        ref_positive_out,
                        lengths,
                        step_mask=preference_step_mask,
                    )
                    ref_negative_logp = self._trajectory_logprob(
                        self.reference_net,
                        negative_batch,
                        ref_negative_out,
                        lengths,
                        step_mask=preference_step_mask,
                    )
            else:
                ref_positive_logp = torch.zeros_like(positive_logp)
                ref_negative_logp = torch.zeros_like(negative_logp)
            rejected_loss = self._dpo_pair_loss(
                positive_logp,
                negative_logp,
                ref_positive_logp,
                ref_negative_logp,
                valid_mask=preference_trajectory_mask,
            )
            (
                positive_continue_logp,
                early_stop_logp,
                continue_trajectory_mask,
            ) = self._continue_vs_early_stop_logprobs(
                positive_out, positive_batch, lengths
            )
            if self.reference_net is not None:
                with torch.no_grad():
                    (
                        ref_positive_continue_logp,
                        ref_early_stop_logp,
                        _,
                    ) = self._continue_vs_early_stop_logprobs(
                        ref_positive_out, positive_batch, lengths
                    )
            else:
                ref_positive_continue_logp = torch.zeros_like(
                    positive_continue_logp
                )
                ref_early_stop_logp = torch.zeros_like(early_stop_logp)
            early_stop_loss = self._dpo_pair_loss(
                positive_continue_logp,
                early_stop_logp,
                ref_positive_continue_logp,
                ref_early_stop_logp,
                valid_mask=continue_trajectory_mask,
            )
            rejected_active = preference_trajectory_mask.any().float()
            early_stop_active = continue_trajectory_mask.any().float()
            active_count = (rejected_active + early_stop_active).clamp_min(1.0)
            total = (
                rejected_active * rejected_loss
                + early_stop_active * early_stop_loss
            ) / active_count
            zero = total * 0.0
            return {
                "stage2_trajectory_dpo_loss": total,
                "stage2_dpo_wrong_loss": rejected_loss,
                "stage2_dpo_low_info_loss": zero,
                "stage2_dpo_early_stop_loss": early_stop_loss,
            }

        if self.trajectory_require_explicit_negatives:
            zero = positive_out["mask_logits"].sum() * 0.0
            return {
                "stage2_trajectory_dpo_loss": zero,
                "stage2_dpo_wrong_loss": zero,
                "stage2_dpo_low_info_loss": zero,
                "stage2_dpo_early_stop_loss": zero,
            }

        wrong_ids = self._sample_wrong_constraint_ids(positive_batch["constraint_id"])
        low_info_ids = self._sample_low_info_constraint_ids(positive_batch["constraint_id"])
        wrong_batch = self._make_counterfactual_batch(positive_batch, wrong_ids)
        low_info_batch = self._make_counterfactual_batch(positive_batch, low_info_ids)
        wrong_out = self._deterministic_trajectory_dpo_forward(
            self.net, wrong_batch, lengths
        )
        low_info_out = self._deterministic_trajectory_dpo_forward(
            self.net, low_info_batch, lengths
        )

        positive_logp = self._trajectory_logprob(self.net, positive_batch, positive_out, lengths)
        wrong_logp = self._trajectory_logprob(self.net, wrong_batch, wrong_out, lengths, constraint_ids=wrong_ids)
        low_info_logp = self._trajectory_logprob(
            self.net, low_info_batch, low_info_out, lengths, constraint_ids=low_info_ids
        )
        (
            positive_continue_logp,
            early_stop_logp,
            continue_trajectory_mask,
        ) = self._continue_vs_early_stop_logprobs(
            positive_out, positive_batch, lengths
        )

        if self.reference_net is not None:
            with torch.no_grad():
                ref_positive_out = self._deterministic_trajectory_dpo_forward(
                    self.reference_net, positive_batch, lengths
                )
                ref_wrong_out = self._deterministic_trajectory_dpo_forward(
                    self.reference_net, wrong_batch, lengths
                )
                ref_low_info_out = self._deterministic_trajectory_dpo_forward(
                    self.reference_net, low_info_batch, lengths
                )
                ref_positive_logp = self._trajectory_logprob(
                    self.reference_net, positive_batch, ref_positive_out, lengths
                )
                ref_wrong_logp = self._trajectory_logprob(
                    self.reference_net, wrong_batch, ref_wrong_out, lengths, constraint_ids=wrong_ids
                )
                ref_low_info_logp = self._trajectory_logprob(
                    self.reference_net, low_info_batch, ref_low_info_out, lengths, constraint_ids=low_info_ids
                )
                (
                    ref_positive_continue_logp,
                    ref_early_stop_logp,
                    _,
                ) = self._continue_vs_early_stop_logprobs(
                    ref_positive_out, positive_batch, lengths
                )
        else:
            ref_positive_logp = torch.zeros_like(positive_logp)
            ref_wrong_logp = torch.zeros_like(wrong_logp)
            ref_low_info_logp = torch.zeros_like(low_info_logp)
            ref_positive_continue_logp = torch.zeros_like(
                positive_continue_logp
            )
            ref_early_stop_logp = torch.zeros_like(early_stop_logp)

        wrong_loss = self._dpo_pair_loss(positive_logp, wrong_logp, ref_positive_logp, ref_wrong_logp)
        low_info_loss = self._dpo_pair_loss(positive_logp, low_info_logp, ref_positive_logp, ref_low_info_logp)
        early_stop_loss = self._dpo_pair_loss(
            positive_continue_logp,
            early_stop_logp,
            ref_positive_continue_logp,
            ref_early_stop_logp,
            valid_mask=continue_trajectory_mask,
        )
        early_stop_active = continue_trajectory_mask.any().float()
        total = (
            wrong_loss + low_info_loss + early_stop_active * early_stop_loss
        ) / (2.0 + early_stop_active)
        return {
            "stage2_trajectory_dpo_loss": total,
            "stage2_dpo_wrong_loss": wrong_loss,
            "stage2_dpo_low_info_loss": low_info_loss,
            "stage2_dpo_early_stop_loss": early_stop_loss,
        }

    def _deterministic_trajectory_dpo_forward(self, net, batch, lengths):
        forward_fn = None
        if getattr(self, "end_to_end_autoregressive_state_training", False):
            forward_fn = lambda current_net, current_batch: (
                self._autoregressive_trajectory_forward(
                    current_net, current_batch, lengths
                )
            )
        return self._deterministic_dpo_forward(
            net, batch, forward_fn=forward_fn
        )

    @staticmethod
    def _deterministic_dpo_forward(net, batch, forward_fn=None):
        """Disable stochastic layers while keeping cuDNN RNNs trainable."""
        stochastic_modules = [
            module
            for module in net.modules()
            if isinstance(module, (nn.Dropout, nn.MultiheadAttention))
        ]
        training_states = [module.training for module in stochastic_modules]
        for module in stochastic_modules:
            module.training = False
        try:
            return net(batch) if forward_fn is None else forward_fn(net, batch)
        finally:
            for module, was_training in zip(
                stochastic_modules, training_states
            ):
                module.training = was_training

    def _dpo_pair_loss(
        self,
        positive_logp,
        negative_logp,
        ref_positive_logp,
        ref_negative_logp,
        valid_mask=None,
    ):
        preference_logit = self.trajectory_dpo_beta * (
            positive_logp - negative_logp - ref_positive_logp + ref_negative_logp
        )
        losses = -F.logsigmoid(preference_logit)
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            if not valid_mask.any():
                return preference_logit.sum() * 0.0
            losses = losses[valid_mask]
        return losses.mean()

    def _trajectory_logprob(
        self,
        net,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        lengths: torch.Tensor,
        constraint_ids: torch.Tensor = None,
        step_mask: torch.Tensor = None,
    ):
        if constraint_ids is None:
            constraint_ids = batch["constraint_id"]
        selector_logprob = F.log_softmax(out["selector_logits"], dim=1).gather(
            1, constraint_ids.long().unsqueeze(1)
        ).squeeze(1)
        target_index = batch["target_index"].long()
        valid_target = (target_index >= 0) & (target_index < out["mask_logits"].shape[1])
        safe_target = target_index.clamp(0, max(out["mask_logits"].shape[1] - 1, 0))
        grounding_mask = self._decision_mask(batch)
        true_targets = batch.get("target_candidate_mask", batch.get("mask_y")).bool()
        filter_targets = batch.get("filter_candidate_mask", true_targets).bool()
        step_filter_targets = filter_targets & grounding_mask
        min_value = torch.finfo(out["mask_logits"].dtype).min
        mask_denominator = torch.logsumexp(
            out["mask_logits"].masked_fill(~grounding_mask, min_value), dim=1
        )
        mask_numerator = torch.logsumexp(
            out["mask_logits"].masked_fill(~step_filter_targets, min_value), dim=1
        )
        valid_step_targets = step_filter_targets.any(dim=1)
        mask_logprob = torch.where(
            valid_step_targets,
            mask_numerator - mask_denominator,
            torch.zeros_like(mask_denominator),
        )
        grounding_logprob = torch.zeros_like(mask_logprob)
        if "cand_stage1_scores" in batch:
            grounding_scores = batch["cand_stage1_scores"]
            min_value = torch.finfo(grounding_scores.dtype).min
            grounding_denominator = torch.logsumexp(
                grounding_scores.masked_fill(~grounding_mask, min_value), dim=1
            )
            state_targets = batch.get("state_candidate_mask", true_targets).bool()
            grounding_targets = state_targets & grounding_mask
            grounding_numerator = torch.logsumexp(
                grounding_scores.masked_fill(~grounding_targets, min_value), dim=1
            )
            grounding_valid = grounding_targets.any(dim=1)
            grounding_logprob = torch.where(
                grounding_valid,
                grounding_numerator - grounding_denominator,
                torch.zeros_like(grounding_denominator),
            )
        continue_logprob = F.logsigmoid(-out["stop_logit"])
        stop_logprob = F.logsigmoid(out["stop_logit"])
        action_logprob = (
            selector_logprob
            + mask_logprob
            + self.trajectory_grounding_logprob_weight * grounding_logprob
            + continue_logprob
        )
        # A terminal action is only good when it stops on the correct target.
        # Omitting final grounding lets DPO reward premature but confident stops.
        terminal_logprob = (
            selector_logprob
            + stop_logprob
            + mask_logprob
            + self.trajectory_grounding_logprob_weight * grounding_logprob
        )
        step_logprob = torch.where(
            batch["stop_y"] >= 0.5, terminal_logprob, action_logprob
        )
        if step_mask is not None:
            step_logprob = step_logprob * step_mask.float()
        return self._sum_by_lengths(step_logprob, lengths)

    def _continue_vs_early_stop_logprobs(
        self,
        out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        lengths: torch.Tensor,
    ):
        starts = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=lengths.device),
                torch.cumsum(lengths, dim=0)[:-1],
            )
        )
        start_stop_logits = out["stop_logit"][starts]
        valid = batch["stop_y"][starts] < 0.5
        return (
            F.logsigmoid(-start_stop_logits),
            F.logsigmoid(start_stop_logits),
            valid,
        )

    def _counterfactual_trajectory_losses(self, batch: Dict[str, torch.Tensor], positive_out: Dict[str, torch.Tensor]):
        positive_score = self._trajectory_action_score(positive_out, batch["constraint_id"], batch["target_index"])
        wrong_ids = self._sample_wrong_constraint_ids(batch["constraint_id"])
        low_info_ids = self._sample_low_info_constraint_ids(batch["constraint_id"])

        wrong_out = self.net(self._make_counterfactual_batch(batch, wrong_ids))
        low_info_out = self.net(self._make_counterfactual_batch(batch, low_info_ids))

        wrong_score = self._trajectory_action_score(wrong_out, wrong_ids, batch["target_index"])
        low_info_score = self._trajectory_action_score(low_info_out, low_info_ids, batch["target_index"])

        not_stop_mask = batch["stop_y"] < 0.5
        if not_stop_mask.any():
            early_stop_loss = F.softplus(self.counterfactual_margin + positive_out["stop_logit"][not_stop_mask]).mean()
        else:
            early_stop_loss = positive_out["stop_logit"].sum() * 0.0

        return {
            "stage2_wrong_constraint_loss": self._pairwise_preference_loss(positive_score, wrong_score),
            "stage2_low_info_constraint_loss": self._pairwise_preference_loss(positive_score, low_info_score),
            "stage2_early_stop_loss": early_stop_loss,
        }

    def _trajectory_action_score(self, out: Dict[str, torch.Tensor], constraint_ids: torch.Tensor, target_index: torch.Tensor):
        selector_logprob = F.log_softmax(out["selector_logits"], dim=1).gather(
            1, constraint_ids.long().unsqueeze(1)
        ).squeeze(1)
        target_index = target_index.long()
        valid_target = (target_index >= 0) & (target_index < out["mask_logits"].shape[1])
        safe_target = target_index.clamp(0, max(out["mask_logits"].shape[1] - 1, 0))
        mask_logprob = F.log_softmax(out["mask_logits"], dim=1).gather(
            1, safe_target.unsqueeze(1)
        ).squeeze(1)
        mask_logprob = torch.where(valid_target, mask_logprob, torch.zeros_like(mask_logprob))
        continue_logprob = F.logsigmoid(-out["stop_logit"])
        return selector_logprob + mask_logprob + continue_logprob

    def _pairwise_preference_loss(self, positive_score: torch.Tensor, negative_score: torch.Tensor):
        return F.softplus(self.counterfactual_margin - positive_score + negative_score).mean()

    def _make_counterfactual_batch(self, batch: Dict[str, torch.Tensor], constraint_ids: torch.Tensor):
        counterfactual = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        self.constraint_table.apply_id_to_batch(counterfactual, constraint_ids.long(), vocab=self.vocab, max_len=self.max_len)
        counterfactual["constraint_id"] = constraint_ids.long().clone()
        counterfactual["alive_mask"] = batch["alive_mask"]
        counterfactual["used_constraint_mask"] = batch["used_constraint_mask"]
        return counterfactual

    def _sample_wrong_constraint_ids(self, positive_ids: torch.Tensor):
        if self.num_constraints <= 1:
            return positive_ids
        wrong_ids = torch.randint(
            low=0,
            high=self.num_constraints - 1,
            size=positive_ids.shape,
            device=positive_ids.device,
        )
        wrong_ids = wrong_ids + (wrong_ids >= positive_ids.long()).long()
        return wrong_ids

    def _sample_low_info_constraint_ids(self, positive_ids: torch.Tensor):
        if not self.high_frequency_constraint_ids:
            return self._sample_wrong_constraint_ids(positive_ids)
        candidates = torch.tensor(self.high_frequency_constraint_ids, dtype=torch.long, device=positive_ids.device)
        low_info = candidates[0].expand_as(positive_ids.long()).clone()
        for candidate_id in candidates:
            replace_mask = low_info == positive_ids.long()
            if not replace_mask.any():
                break
            low_info[replace_mask] = candidate_id
        same_mask = low_info == positive_ids.long()
        if same_mask.any():
            low_info[same_mask] = self._sample_wrong_constraint_ids(positive_ids[same_mask])
        return low_info

    @staticmethod
    def _sum_by_lengths(values: torch.Tensor, lengths: torch.Tensor):
        sums = []
        start = 0
        for length in lengths.tolist():
            end = start + int(length)
            sums.append(values[start:end].sum())
            start = end
        return torch.stack(sums)

    def _encode_end_to_end_sample(
        self, sample: Dict[str, Any], query_override: str = None
    ) -> Dict[str, torch.Tensor]:
        constraint_id = self.constraint_table.id_from_constraint_gt(sample.get("constraint_gt") or {})
        constraint_row = self.constraint_table.row(constraint_id)
        query_text = sample.get("query") if query_override is None else query_override
        query_ids = self.vocab.encode((query_text or "").strip(), self.max_len)
        field_ids = self.vocab.encode(constraint_row.field_text, self.max_len)
        return {
            "query_ids": torch.tensor(query_ids, dtype=torch.long),
            "field_ids": torch.tensor(field_ids, dtype=torch.long),
            "c_ctype_id": torch.tensor(constraint_row.ctype_id, dtype=torch.long),
            "c_rel_id": torch.tensor(constraint_row.rel_id, dtype=torch.long),
            "c_attr_id": torch.tensor(constraint_row.attr_id, dtype=torch.long),
            "constraint_id": torch.tensor(constraint_id, dtype=torch.long),
            "target_index": torch.tensor(0, dtype=torch.long),
            "img": torch.zeros((1, self.train_cfg.imgfeat_dim), dtype=torch.float32),
            "cand_mask": torch.ones(1, dtype=torch.float32),
            "mask_y": torch.ones(1, dtype=torch.float32),
            "stop_y": torch.tensor(float(sample.get("stop_y", 0)), dtype=torch.float32),
            "cand_bboxes": torch.zeros((1, 7), dtype=torch.float32),
            "cand_semantic_ids": torch.zeros(1, dtype=torch.long),
            "cand_stage1_features": torch.zeros(
                (1, self.train_cfg.stage1_feat_dim), dtype=torch.float32
            ),
            "cand_stage1_scores": torch.zeros(1, dtype=torch.float32),
            "cand_point_counts": torch.zeros(1, dtype=torch.float32),
        }

    def _find_end_to_end_trajectory(
        self,
        scene_id: str,
        ann_id: str,
        target_id: int,
        trajectory_id: str,
    ) -> List[int]:
        exact_key = (scene_id, ann_id, target_id, trajectory_id)
        if trajectory_id and exact_key in self.end_to_end_trajectory_lookup:
            return self.end_to_end_trajectory_lookup[exact_key]
        candidates = self.end_to_end_fallback_lookup.get((scene_id, ann_id, target_id), [])
        return candidates[0] if len(candidates) == 1 else []

    @staticmethod
    def _dense_proposal_point_counts(stage1_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        proposal_indices = stage1_output.get("proposal_indices_dense")
        proposal_scores = stage1_output.get("proposal_scores")
        if proposal_indices is None or proposal_scores is None:
            proposal_mask = stage1_output["proposal_masks_dense"]
            return proposal_mask.new_zeros(proposal_mask.shape, dtype=torch.float32)
        proposal_offsets = proposal_scores[2]
        all_counts = (proposal_offsets[1:] - proposal_offsets[:-1]).float()
        safe_indices = proposal_indices.long().clamp(0, max(all_counts.numel() - 1, 0))
        dense_counts = all_counts[safe_indices]
        return torch.where(proposal_indices >= 0, dense_counts, torch.zeros_like(dense_counts))

    def build_end_to_end_batch(
        self,
        data_dict: Dict[str, Any],
        stage1_output: Dict[str, torch.Tensor],
        trajectory_output: Dict[str, torch.Tensor],
        target_labels: torch.Tensor,
    ) -> Dict[str, Any]:
        if not self.end_to_end_enabled:
            return None
        if trajectory_output is None or target_labels is None:
            if self.end_to_end_require_aligned_batch:
                raise ValueError(
                    "End-to-end Stage2 requires online trajectory queries and target labels "
                    "from the current Stage1 batch."
                )
            return None

        independent_pair_training = self._uses_independent_pair_training()

        round_mask = trajectory_output["round_mask"].bool()
        raw_score_history = trajectory_output.get("raw_score_history")
        if raw_score_history is None:
            raw_score_history = trajectory_output["score_history"]
        stage1_initial_scores = trajectory_output.get("stage1_initial_scores")
        if stage1_initial_scores is None:
            stage1_initial_scores = raw_score_history[:, :, 0]
        stage1_initial_candidate_mask = trajectory_output.get(
            "stage1_initial_candidate_mask"
        )
        if stage1_initial_candidate_mask is None:
            stage1_initial_candidate_mask = trajectory_output["alive_history"][
                :, :, 0
            ].ge(0.5)
        batch_size, chunk_size, num_rounds = round_mask.shape
        trajectory_ids = data_dict.get("trajectory_id")
        trajectory_source_ann_ids = data_dict.get("trajectory_source_ann_id")
        target_ids = data_dict.get(
            "trajectory_target_obj_id", data_dict.get("true_target_id")
        )
        if target_ids is None:
            raise ValueError("End-to-end Stage2 requires trajectory_target_obj_id or true_target_id.")

        encoded_rows = []
        alignments = []
        target_indices = []
        target_candidate_masks = []
        filter_candidate_masks = []
        state_candidate_masks = []
        available_constraint_masks = []
        used_constraint_masks = []
        selector_target_masks = []
        selector_target_distributions = []
        rejected_constraint_ids = []
        preference_step_masks = []
        state_constraint_sequences = []
        minimum_rollout_rounds = []
        required_rollout_rounds = []
        lengths = []
        trajectory_keys = []
        missing_metadata = 0
        missing_target_proposal = 0
        missing_initial_target = 0

        for batch_idx in range(batch_size):
            scene_id = str(data_dict["scene_id"][batch_idx])
            for lang_idx in range(chunk_size):
                if not round_mask[batch_idx, lang_idx].any():
                    continue
                target_id = int(target_ids[batch_idx, lang_idx].item())
                if target_id < 0:
                    continue
                ann_id_tensor = (
                    trajectory_source_ann_ids
                    if trajectory_source_ann_ids is not None
                    else data_dict["ann_id"]
                )
                ann_id = str(int(ann_id_tensor[batch_idx, lang_idx].item()))
                trajectory_id = ""
                if trajectory_ids is not None:
                    trajectory_id = str(trajectory_ids[batch_idx][lang_idx])
                trajectory = self._find_end_to_end_trajectory(
                    scene_id, ann_id, target_id, trajectory_id
                )
                if not trajectory:
                    missing_metadata += 1
                    continue
                available_ids = self.end_to_end_available_constraint_ids.get(
                    (scene_id, ann_id, target_id), set()
                )

                proposal_targets = target_labels[batch_idx, lang_idx].bool()
                if not proposal_targets.any():
                    missing_target_proposal += 1
                    continue
                initial_alive = stage1_initial_candidate_mask[
                    batch_idx, lang_idx
                ].bool()
                if not (proposal_targets & initial_alive).any():
                    # Stage2 cannot recover a proposal removed before interaction.
                    # Stage1 trajectory-survival losses remain responsible for
                    # these rows; including them here creates an impossible
                    # teacher-forced state that never occurs during rollout.
                    missing_initial_target += 1
                    if self.training:
                        continue
                with torch.no_grad():
                    target_scores = stage1_initial_scores[
                        batch_idx, lang_idx
                    ].masked_fill(
                        ~proposal_targets,
                        torch.finfo(stage1_initial_scores.dtype).min,
                    )
                    target_index = int(target_scores.argmax().item())

                current_rows = []
                used_constraint_ids = set()
                for sample_idx in trajectory:
                    sample = self.samples[sample_idx]
                    round_idx = int(sample.get("round", 0))
                    if round_idx < 0 or round_idx >= num_rounds:
                        continue
                    if not bool(round_mask[batch_idx, lang_idx, round_idx].item()):
                        continue
                    current_rows.append(
                        (
                            self._encode_end_to_end_sample(sample),
                            sample,
                            (batch_idx, lang_idx, round_idx),
                            target_index,
                            set() if independent_pair_training else set(used_constraint_ids),
                            set(
                                self.end_to_end_selector_target_ids.get(
                                    (scene_id, ann_id, target_id) + (
                                        tuple(
                                            sorted(
                                                int(value)
                                                for value in sample.get("cand_ids") or []
                                            )
                                        ),
                                    ),
                                    (),
                                )
                            ),
                        )
                    )
                    if (
                        not independent_pair_training
                        and float(sample.get("stop_y", 0)) < 0.5
                    ):
                        used_constraint_ids.add(
                            int(
                                self.constraint_table.id_from_constraint_gt(
                                    sample.get("constraint_gt") or {}
                                )
                            )
                        )
                if not current_rows:
                    missing_metadata += 1
                    continue
                has_explicit_schedule = all(
                    "action_step" in row_sample
                    for _, row_sample, *_ in current_rows
                )
                if has_explicit_schedule:
                    current_rows.sort(
                        key=lambda item: int(
                            item[1].get("action_step", item[1].get("round", 0))
                        )
                    )
                scheduled_constraint_ids = (
                    [int(row["constraint_id"].item()) for row, *_ in current_rows]
                    if has_explicit_schedule
                    else []
                )
                required_interactions = (
                    max(
                        int(row_sample.get("required_interactions", len(current_rows)))
                        for _, row_sample, *_ in current_rows
                    )
                    if has_explicit_schedule
                    else 0
                )
                minimum_interactions = (
                    max(
                        int(
                            row_sample.get(
                                "minimum_interactions_before_stop",
                                required_interactions,
                            )
                        )
                        for _, row_sample, *_ in current_rows
                    )
                    if has_explicit_schedule
                    else 0
                )
                minimum_interactions = min(
                    minimum_interactions, len(scheduled_constraint_ids)
                )
                if (
                    self.training
                    and self.supervision_mode == "shuffled_pairs"
                    and len(current_rows) > 1
                ):
                    rng = random.Random(
                        self.independent_pair_seed
                        + batch_idx * 1_000_003
                        + lang_idx * 10_007
                        + sum(ord(char) for char in f"{scene_id}:{ann_id}:{trajectory_id}")
                    )
                    current_rows = self._shuffle_independent_rows(
                        current_rows, rng.randint(0, 2**31 - 1)
                    )
                if independent_pair_training:
                    lengths.extend([1] * len(current_rows))
                    trajectory_keys.extend([
                        (
                            scene_id,
                            int(data_dict["ann_id"][batch_idx, lang_idx].item()),
                        )
                        for _ in current_rows
                    ])
                else:
                    lengths.append(len(current_rows))
                    trajectory_keys.append(
                        (
                            scene_id,
                            int(data_dict["ann_id"][batch_idx, lang_idx].item()),
                        )
                    )
                for (
                    row,
                    row_sample,
                    alignment,
                    row_target_index,
                    row_used_constraint_ids,
                    row_selector_target_ids,
                ) in current_rows:
                    encoded_rows.append(row)
                    alignments.append(alignment)
                    target_indices.append(row_target_index)
                    state_constraint_sequences.append(
                        [] if independent_pair_training else scheduled_constraint_ids
                    )
                    minimum_rollout_rounds.append(
                        0 if independent_pair_training else minimum_interactions
                    )
                    required_rollout_rounds.append(
                        0 if independent_pair_training else required_interactions
                    )
                    target_candidate_masks.append(proposal_targets)
                    candidate_labels = trajectory_output.get("candidate_labels")
                    final_round_idx = max(
                        int(
                            trajectory_output["round_mask"][
                                alignment[0], alignment[1]
                            ].long().sum().item()
                        ) - 1,
                        0,
                    )
                    filter_targets = self._post_action_filter_targets(
                        candidate_labels,
                        proposal_targets,
                        alignment[0],
                        alignment[1],
                        alignment[2],
                        final_round_idx,
                        terminal=float(row_sample.get("stop_y", 0.0)) >= 0.5,
                    )
                    if candidate_labels is not None:
                        state_targets = candidate_labels[
                            alignment[0], alignment[1], alignment[2]
                        ].bool()
                    else:
                        state_targets = proposal_targets
                    if not state_targets.any():
                        state_targets = proposal_targets
                    filter_candidate_masks.append(filter_targets)
                    state_candidate_masks.append(state_targets)
                    available_mask = torch.zeros(
                        self.num_constraints, dtype=torch.bool
                    )
                    if has_explicit_schedule:
                        available_mask[int(row["constraint_id"].item())] = True
                    elif available_ids:
                        available_mask[list(available_ids)] = True
                    else:
                        available_mask[int(row["constraint_id"].item())] = True
                    available_constraint_masks.append(available_mask)
                    used_mask = torch.zeros(
                        self.num_constraints, dtype=torch.float32
                    )
                    if row_used_constraint_ids:
                        used_mask[list(row_used_constraint_ids)] = 1.0
                    current_constraint_id = int(row["constraint_id"].item())
                    used_mask[current_constraint_id] = 0.0
                    used_constraint_masks.append(used_mask)
                    selector_target_mask = torch.zeros(
                        self.num_constraints, dtype=torch.bool
                    )
                    if row_selector_target_ids and not independent_pair_training:
                        selector_target_mask[list(row_selector_target_ids)] = True
                    selector_target_mask &= available_mask
                    selector_target_mask &= ~used_mask.bool()
                    selector_target_mask[current_constraint_id] = True
                    selector_target_masks.append(selector_target_mask)
                    state_key = (scene_id, ann_id, target_id) + (
                        tuple(
                            sorted(
                                int(value)
                                for value in row_sample.get("cand_ids") or []
                            )
                        ),
                    )
                    action_utilities = (
                        {}
                        if independent_pair_training
                        else dict(
                            self.end_to_end_selector_action_utilities.get(
                                state_key, {}
                            )
                        )
                    )
                    action_utilities.setdefault(current_constraint_id, 0.0)
                    valid_action_ids = sorted(
                        action_id
                        for action_id in action_utilities
                        if bool(available_mask[action_id])
                        and not bool(used_mask[action_id])
                    )
                    if not valid_action_ids:
                        valid_action_ids = [current_constraint_id]
                    utility_values = torch.tensor(
                        [action_utilities[action_id] for action_id in valid_action_ids],
                        dtype=torch.float32,
                    )
                    action_probabilities = torch.softmax(
                        utility_values / self.selector_utility_temperature,
                        dim=0,
                    )
                    target_distribution = torch.zeros(
                        self.num_constraints, dtype=torch.float32
                    )
                    target_distribution[valid_action_ids] = action_probabilities
                    selector_target_distributions.append(target_distribution)
                    if independent_pair_training:
                        rejected_action_id, has_preference = (
                            current_constraint_id, False
                        )
                    else:
                        rejected_action_id, has_preference = (
                            self._return_conditioned_rejected_action(
                                action_utilities,
                                current_constraint_id,
                                valid_action_ids,
                                self.trajectory_action_min_utility_gap,
                            )
                        )
                    rejected_constraint_ids.append(rejected_action_id)
                    preference_step_masks.append(has_preference)

        if not encoded_rows:
            if self.end_to_end_require_aligned_batch:
                raise ValueError(
                    "No current-batch trajectories could be aligned for end-to-end Stage2. "
                    f"missing_metadata={missing_metadata}, "
                    f"missing_target_proposal={missing_target_proposal}, "
                    f"missing_initial_target={missing_initial_target}."
                )
            return None

        device = stage1_output["proposal_masks_dense"].device
        batch = _move_to_device(self.collate_semantic(encoded_rows), device)
        scene_indices = torch.tensor(
            [row[0] for row in alignments], dtype=torch.long, device=device
        )
        lang_indices = torch.tensor(
            [row[1] for row in alignments], dtype=torch.long, device=device
        )
        round_indices = torch.tensor(
            [row[2] for row in alignments], dtype=torch.long, device=device
        )
        row_target_indices = torch.tensor(target_indices, dtype=torch.long, device=device)
        stage1_round_indices = (
            round_indices
            if independent_pair_training
            else torch.zeros_like(round_indices)
            if self.end_to_end_use_initial_stage1_state
            else round_indices
        )

        feature_key = self.end_to_end_feature_key
        if feature_key not in stage1_output:
            raise KeyError(
                f"Stage1 output does not contain end-to-end feature key {feature_key!r}."
            )
        adapter_feature_history = trajectory_output.get("candidate_feature_history")
        initial_candidate_features = trajectory_output.get(
            "stage1_initial_candidate_features"
        )
        if (
            self.end_to_end_use_initial_stage1_state
            and initial_candidate_features is not None
            and not independent_pair_training
        ):
            candidate_features = initial_candidate_features[
                scene_indices, lang_indices
            ]
        elif adapter_feature_history is not None:
            candidate_features = adapter_feature_history[
                scene_indices, lang_indices, stage1_round_indices
            ]
        else:
            candidate_features = stage1_output[feature_key].index_select(0, scene_indices)
        if candidate_features.shape[-1] != self.train_cfg.stage1_feat_dim:
            raise ValueError(
                f"Stage1 {feature_key} width is {candidate_features.shape[-1]}, but "
                f"model.stage2.stage1_feat_dim={self.train_cfg.stage1_feat_dim}."
            )

        proposal_mask = stage1_output["proposal_masks_dense"].index_select(
            0, scene_indices
        ).float()
        if independent_pair_training:
            candidate_scores = trajectory_output["refined_score_history"][
                scene_indices, lang_indices, round_indices
            ]
        else:
            candidate_scores = stage1_initial_scores[
                scene_indices, lang_indices
            ]
        # Visual features and raw Stage1 scores stay anchored to the initial
        # query, but the decision state must advance with the trajectory. A
        # static initial belief makes later selector decisions blind to prior
        # clarification outcomes and disagrees with autoregressive rollout.
        candidate_alive, candidate_belief = self._gather_dynamic_trajectory_state(
            trajectory_output,
            scene_indices,
            lang_indices,
            round_indices,
        )
        valid_candidates = proposal_mask.bool()
        candidate_features = candidate_features.masked_fill(
            ~valid_candidates.unsqueeze(-1), 0.0
        )
        candidate_scores = candidate_scores.masked_fill(
            ~valid_candidates, 0.0
        )
        candidate_alive = candidate_alive.masked_fill(
            ~valid_candidates, 0.0
        )
        candidate_belief = candidate_belief.masked_fill(
            ~valid_candidates, 0.0
        )

        bounds = stage1_output["pred_aabb_min_max_bounds"].index_select(0, scene_indices)
        centers = bounds.mean(dim=2)
        if "scene_center_xyz" in data_dict:
            centers = centers + data_dict["scene_center_xyz"].index_select(
                0, scene_indices
            ).unsqueeze(1)
        sizes = (bounds[:, :, 1] - bounds[:, :, 0]).abs().clamp_min(1e-4)
        semantic_ids = torch.zeros(
            proposal_mask.shape, dtype=torch.long, device=device
        )
        candidate_bboxes = torch.cat(
            (centers, sizes, semantic_ids.unsqueeze(-1).float()), dim=-1
        )
        point_counts = self._dense_proposal_point_counts(stage1_output).index_select(
            0, scene_indices
        )
        stage1_gradient_scale = self.end_to_end_stage1_gradient_scale
        candidate_features = self._scale_stage1_gradient(
            candidate_features, stage1_gradient_scale
        )
        candidate_scores = self._scale_stage1_gradient(
            candidate_scores, stage1_gradient_scale
        )
        candidate_alive = self._scale_stage1_gradient(
            candidate_alive, stage1_gradient_scale
        )
        candidate_belief = self._scale_stage1_gradient(
            candidate_belief, stage1_gradient_scale
        )
        candidate_bboxes = self._scale_stage1_gradient(
            candidate_bboxes, stage1_gradient_scale
        )
        mask_y = proposal_mask.new_zeros(proposal_mask.shape)
        mask_y.scatter_(1, row_target_indices.unsqueeze(1), 1.0)
        teacher_state_mask = torch.stack(state_candidate_masks, dim=0).to(
            device=device
        )
        decision_mask = self._trajectory_candidate_state_masks(
            teacher_state_mask,
            candidate_alive,
            proposal_mask,
            round_indices,
            self.end_to_end_candidate_state_mode,
        )
        max_scheduled_actions = max(
            (len(sequence) for sequence in state_constraint_sequences), default=0
        )
        state_constraint_ids = torch.full(
            (len(encoded_rows), max_scheduled_actions),
            -1,
            dtype=torch.long,
            device=device,
        )
        for row_idx, sequence in enumerate(state_constraint_sequences):
            if sequence:
                state_constraint_ids[row_idx, :len(sequence)] = torch.tensor(
                    sequence, dtype=torch.long, device=device
                )

        batch.update({
            "img": candidate_features.new_zeros(
                (*proposal_mask.shape, self.train_cfg.imgfeat_dim)
            ),
            "cand_mask": proposal_mask,
            "mask_y": mask_y,
            "target_index": row_target_indices,
            "target_candidate_mask": torch.stack(target_candidate_masks, dim=0).bool(),
            "filter_candidate_mask": torch.stack(filter_candidate_masks, dim=0).bool(),
            "state_candidate_mask": torch.stack(state_candidate_masks, dim=0).bool(),
            "available_constraint_mask": torch.stack(
                available_constraint_masks, dim=0
            ).to(device=device),
            "used_constraint_mask": torch.stack(
                used_constraint_masks, dim=0
            ).to(device=device),
            "selector_target_mask": torch.stack(
                selector_target_masks, dim=0
            ).to(device=device),
            "selector_target_distribution": torch.stack(
                selector_target_distributions, dim=0
            ).to(device=device),
            "cand_bboxes": candidate_bboxes,
            "cand_semantic_ids": semantic_ids,
            "cand_stage1_features": candidate_features,
            "cand_stage1_scores": candidate_scores,
            "cand_point_counts": point_counts,
            "cand_alive_probs": candidate_alive,
            "cand_belief": candidate_belief,
            "alive_mask": decision_mask.float(),
            "state_constraint_ids": state_constraint_ids,
            "minimum_rollout_rounds": torch.tensor(
                minimum_rollout_rounds, dtype=torch.long, device=device
            ),
            "required_rollout_rounds": torch.tensor(
                required_rollout_rounds, dtype=torch.long, device=device
            ),
        })
        if self.end_to_end_preserve_scene_context:
            batch["context_mask"] = proposal_mask
        rejected_ids = torch.tensor(
            rejected_constraint_ids, dtype=torch.long, device=device
        )
        preference_step_mask = torch.tensor(
            preference_step_masks, dtype=torch.bool, device=device
        )
        # The rejected policy sees exactly the same pre-action state. Only the
        # lower-return action is changed, so DPO cannot exploit post-outcome
        # language or candidate-state shortcuts.
        negative_batch = self._make_counterfactual_batch(batch, rejected_ids)
        trajectory_batch = {
            "flat_batch": batch,
            "negative_flat_batch": negative_batch,
            "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
            "preference_step_mask": preference_step_mask,
        }
        return {
            "batch": batch,
            "trajectory_batch": trajectory_batch,
            "num_trajectories": len(lengths),
            "num_rounds": len(encoded_rows),
            "num_preference_steps": int(sum(preference_step_masks)),
            "missing_metadata": missing_metadata,
            "missing_target_proposal": missing_target_proposal,
            "missing_initial_target": missing_initial_target,
            "trajectory_keys": trajectory_keys,
            "alignments": alignments,
        }

    @torch.no_grad()
    def evaluate_end_to_end_batch(
        self,
        end_to_end: Dict[str, Any],
        stage1_threshold: float,
        return_details: bool = False,
    ):
        """Run the same Stage2 agent rollout used by standalone validation."""
        from eval_rollout import eval_rollout_batch

        flat_batch = end_to_end["batch"]
        lengths = end_to_end["trajectory_batch"]["lengths"].long()
        starts = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=lengths.device),
                torch.cumsum(lengths, dim=0)[:-1],
            )
        )
        num_round_rows = flat_batch["cand_mask"].shape[0]
        initial_batch = {
            key: value.index_select(0, starts)
            if torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == num_round_rows
            else value
            for key, value in flat_batch.items()
        }
        proposal_mask = initial_batch["cand_mask"].bool()
        stage1_keep = (
            torch.sigmoid(initial_batch["cand_stage1_scores"])
            >= float(stage1_threshold)
        )
        initial_mask = proposal_mask & stage1_keep
        if self.end_to_end_preserve_scene_context:
            initial_batch["context_mask"] = proposal_mask.float()
        else:
            initial_batch["cand_mask"] = initial_mask.float()
        initial_batch["alive_mask"] = initial_mask.float()

        keys = list(end_to_end["trajectory_keys"])
        records = {key: False for key in keys}
        full_target_mask = initial_batch.get(
            "target_candidate_mask", initial_batch["mask_y"]
        ).bool()
        initial_target_survival = (
            full_target_mask & initial_mask
        ).any(dim=1).detach().cpu()
        details = {
            "aligned": {key: True for key in keys},
            "initial_target_survival": {
                key: bool(value)
                for key, value in zip(keys, initial_target_survival.tolist())
            },
            "final_target_survival": {key: False for key in keys},
        }
        nonempty = initial_mask.any(dim=1)
        if not nonempty.any():
            return (records, details) if return_details else records

        active_indices = torch.nonzero(nonempty, as_tuple=False).squeeze(1)
        aligned_count = int(initial_mask.shape[0])
        rollout_batch = {
            key: value.index_select(0, active_indices)
            if torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == aligned_count
            else value
            for key, value in initial_batch.items()
        }
        target_mask = initial_batch.get(
            "target_candidate_mask", initial_batch["mask_y"]
        ).bool().index_select(
            0, active_indices
        )
        pred_idx, _, _, alive_history, _ = eval_rollout_batch(
            model=self.net,
            batch=rollout_batch,
            constraint_table=self.constraint_table,
            vocab=self.vocab,
            max_len=self.max_len,
            max_round=self.validation_max_round,
            thr=self.validation_thr,
            stop_thr=self.validation_stop_thr,
            eval_mode=self.validation_eval_mode,
            v_target_strategy=self.validation_v_target_strategy,
            selector_strategy=self.validation_selector_strategy,
            risk_sensitive_pruning=self.validation_risk_sensitive_pruning,
            min_keep_ratio=self.validation_min_keep_ratio,
            min_keep_count=self.validation_min_keep_count,
        )
        pred_idx = pred_idx.long().clamp(0, max(target_mask.shape[1] - 1, 0))
        correct = target_mask.detach().cpu().gather(
            1, pred_idx.unsqueeze(1)
        ).squeeze(1)
        for row_idx, is_correct in zip(active_indices.tolist(), correct.tolist()):
            records[keys[row_idx]] = bool(is_correct)
        if alive_history:
            final_survival = alive_history[-1].bool().tolist()
            for row_idx, survived in zip(active_indices.tolist(), final_survival):
                details["final_target_survival"][keys[row_idx]] = bool(survived)
        return (records, details) if return_details else records

    @staticmethod
    def _load_json(path: str) -> List[Dict[str, Any]]:
        import json

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _build_vocab(vocab_cls: Any, samples: Iterable[Dict[str, Any]]) -> Any:
        from stage2_refactor.learning.constraint_table import constraint_field_text

        texts = []
        for sample in samples:
            texts.append(sample.get("query", "") or "")
            constraint = sample.get("constraint_gt") or {}
            texts.append(constraint.get("landmark_text", "") or "")
            texts.append(constraint.get("attr_val_text", "") or "")
            texts.append(constraint_field_text(constraint))
        return vocab_cls.build(texts, min_freq=1, max_size=20000)

    def _build_high_frequency_constraint_ids(self, samples: Iterable[Dict[str, Any]]) -> List[int]:
        from collections import Counter

        counter = Counter()
        for sample in samples:
            constraint = sample.get("constraint_gt") or {}
            cid = self.constraint_table.id_from_constraint_gt(constraint)
            counter[int(cid)] += 1
        return [cid for cid, _ in counter.most_common()]

    def _build_selector_action_utilities(
        self, trajectories: List[List[int]]
    ) -> Dict[tuple, Dict[int, float]]:
        """Estimate action value from future candidate dynamics in each trajectory."""
        from collections import defaultdict

        utility_samples = defaultdict(lambda: defaultdict(list))
        discount = self.selector_return_discount
        for trajectory in trajectories:
            if not trajectory:
                continue
            samples = [self.samples[index] for index in trajectory]
            target_id = self._trajectory_key(samples[0])[2]
            transitions = []
            for position, sample in enumerate(samples):
                current = {
                    int(value) for value in sample.get("cand_ids") or []
                }
                if position + 1 < len(samples):
                    next_candidates = {
                        int(value)
                        for value in samples[position + 1].get("cand_ids") or []
                    }
                elif float(sample.get("stop_y", 0)) >= 0.5:
                    next_candidates = {target_id}
                else:
                    next_candidates = set(current)
                reduction = max(
                    0.0,
                    min(
                        1.0,
                        (len(current) - len(next_candidates))
                        / max(len(current), 1),
                    ),
                )
                transitions.append(
                    (sample, current, next_candidates, reduction)
                )

            final_candidates = transitions[-1][2]
            terminal_success = float(target_id in final_candidates)
            for position, (sample, current, _, _) in enumerate(transitions):
                remaining = len(transitions) - position
                target_survives = all(
                    target_id in transition[2]
                    for transition in transitions[position:]
                )
                success_value = (
                    discount ** max(remaining - 1, 0)
                ) * terminal_success
                if not target_survives:
                    success_value = -1.0
                future_gain = sum(
                    (discount ** offset) * transition[3]
                    for offset, transition in enumerate(
                        transitions[position:]
                    )
                )
                utility = (
                    success_value
                    - self.selector_step_cost * max(remaining - 1, 0)
                    + self.selector_information_gain_weight * future_gain
                )
                state_key = self._trajectory_key(sample)[:3] + (
                    tuple(sorted(current)),
                )
                constraint_id = int(
                    self.constraint_table.id_from_constraint_gt(
                        sample.get("constraint_gt") or {}
                    )
                )
                utility_samples[state_key][constraint_id].append(utility)

        return {
            state_key: {
                constraint_id: sum(values) / len(values)
                for constraint_id, values in action_values.items()
            }
            for state_key, action_values in utility_samples.items()
        }

    @staticmethod
    def _trajectory_key(sample: Dict[str, Any]):
        cand_ids = [int(x) for x in sample.get("cand_ids", [])]
        target_index = int(sample.get("target_index", 0))
        target_id = cand_ids[target_index] if 0 <= target_index < len(cand_ids) else -1
        return (
            sample.get("scene_id", ""),
            str(sample.get("ann_id", "")),
            int(target_id),
            str(sample.get("trajectory_id", sample.get("ambiguity_type", "default"))),
        )

    def _build_trajectories(
        self,
        samples: List[Dict[str, Any]],
        require_distinct_rounds: bool = True,
    ) -> List[List[int]]:
        grouped = {}
        for idx, sample in enumerate(samples):
            grouped.setdefault(self._trajectory_key(sample), []).append(idx)
        trajectories = []
        for indices in grouped.values():
            by_round = {}
            for idx in indices:
                round_id = int(samples[idx].get("round", 0))
                by_round.setdefault(round_id, idx)
            round_ids = sorted(by_round)
            if require_distinct_rounds and len(round_ids) < 2:
                continue
            trajectory = [by_round[round_id] for round_id in round_ids]
            if trajectory:
                trajectories.append(trajectory)
        return trajectories

    def _collate_trajectory_batch(self, trajectories):
        flat = []
        lengths = []
        for trajectory in trajectories:
            lengths.append(len(trajectory))
            flat.extend(trajectory)
        return {
            "flat_batch": self.collate_semantic(flat),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }

    def _load_pretrained(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.net.load_state_dict(state, strict=False)
