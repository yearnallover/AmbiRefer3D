import os
import json
import clip
import torch
import torch.distributed as dist
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import lightning.pytorch as pl
from m3drefclip.common_ops.functions import common_ops
from m3drefclip.model.vision_module.pointgroup import PointGroup
from m3drefclip.model.cross_modal_module.match_module import MatchModule
from m3drefclip.model.vision_module.object_renderer import ObjectRenderer
from m3drefclip.model.vision_module.clip_image_encoder import CLIPImageEncoder
from m3drefclip.model.stage2_joint import Stage2JointModule
from m3drefclip.model.trajectory_grounding_adapter import TrajectoryConditionedGroundingAdapter
from m3drefclip.model.stage1_backbones import (
    BUTDDETRStage1Backbone,
    TSP3DStage1Backbone,
    VistaStage1Backbone,
)
from m3drefclip.util.utils import get_batch_aabb_pair_ious


def freeze_batch_norm_running_stats(modules):
    for root in modules:
        if root is None:
            continue
        for module in root.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def enforce_detector_only_recovery_parameters(module):
    """Freeze every non-detector parameter and return the trainable names."""
    trainable = []
    for name, parameter in module.named_parameters():
        parameter.requires_grad = name.startswith("detector.")
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise ValueError("detector-only recovery found no detector parameters")
    return trainable


class M3DRefCLIP(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = cfg.data.lang_dataset
        stage1_backbone_cfg = getattr(cfg.model, "stage1_backbone", None)
        self.stage1_backbone_name = str(
            getattr(stage1_backbone_cfg, "name", "m3drefclip")
        ).lower()

        # vision modules
        if self.stage1_backbone_name == "m3drefclip":
            input_channel = 3 + 3 * cfg.data.point_features.use_rgb + 3 * cfg.data.point_features.use_normal + \
                            128 * cfg.data.point_features.use_multiview
            self.detector = PointGroup(
                input_channel=input_channel, output_channel=cfg.model.network.detector.output_channel,
                max_proposals=cfg.model.network.max_num_proposals, semantic_class=cfg.data.semantic_class,
                use_gt=cfg.model.network.detector.use_gt_proposal
            )
        elif self.stage1_backbone_name == "butd_detr":
            self.detector = BUTDDETRStage1Backbone(
                project_root=cfg.project_root_path,
                checkpoint=stage1_backbone_cfg.checkpoint,
                span_checkpoint=stage1_backbone_cfg.span_checkpoint,
                detector_root=stage1_backbone_cfg.detector_root,
                proposal_dim=cfg.model.network.detector.output_channel,
                common_dim=cfg.model.network.matching_module.feat_channel,
                freeze_point_backbone=stage1_backbone_cfg.freeze_point_backbone,
                freeze_native_backbone=stage1_backbone_cfg.freeze_native_backbone,
                cache_scenes=stage1_backbone_cfg.cache_scenes,
            )
        elif self.stage1_backbone_name == "3d_vista":
            self.detector = VistaStage1Backbone(
                project_root=cfg.project_root_path,
                checkpoint=stage1_backbone_cfg.checkpoint,
                bert_root=stage1_backbone_cfg.bert_root,
                scene_root=stage1_backbone_cfg.scene_root,
                proposal_dim=cfg.model.network.detector.output_channel,
                common_dim=cfg.model.network.matching_module.feat_channel,
                freeze_point_features=stage1_backbone_cfg.freeze_point_features,
                cache_scenes=stage1_backbone_cfg.cache_scenes,
            )
        elif self.stage1_backbone_name == "tsp3d":
            self.detector = TSP3DStage1Backbone(
                project_root=cfg.project_root_path,
                checkpoint=stage1_backbone_cfg.checkpoint,
                detector_root=stage1_backbone_cfg.detector_root,
                roberta_root=stage1_backbone_cfg.roberta_root,
                proposal_dim=cfg.model.network.detector.output_channel,
                common_dim=cfg.model.network.matching_module.feat_channel,
                max_proposals=cfg.model.network.max_num_proposals,
                voxel_size=stage1_backbone_cfg.voxel_size,
                nms_iou_threshold=stage1_backbone_cfg.nms_iou_threshold,
                freeze_sparse_backbone=stage1_backbone_cfg.freeze_sparse_backbone,
                cache_scenes=stage1_backbone_cfg.cache_scenes,
            )
        else:
            raise ValueError(
                "model.stage1_backbone.name must be one of "
                "m3drefclip, butd_detr, 3d_vista, or tsp3d; got "
                f"{self.stage1_backbone_name!r}."
            )

        # loss
        if self.dataset_name in ("ScanRefer", "Nr3D"):
            self.ref_loss = hydra.utils.instantiate(
                cfg.model.loss.reference_ce_loss, chunk_size=cfg.data.chunk_size,
                max_num_proposals=cfg.model.network.max_num_proposals
            )
        elif self.dataset_name == "Multi3DRefer":
            self.ref_loss = hydra.utils.instantiate(
                cfg.model.loss.reference_bce_loss, chunk_size=cfg.data.chunk_size,
                max_num_proposals=cfg.model.network.max_num_proposals
            )
        else:
            raise NotImplementedError

        self.clip_model = None
        self.text_encoder = None
        self.match_module = None
        if self.stage1_backbone_name == "m3drefclip":
            self.clip_model = clip.load(cfg.model.network.clip_model, device=self.device)[0]

            # freeze CLIP
            for param in self.clip_model.parameters():
                param.requires_grad = False

            if self.hparams.cfg.model.network.use_2d_feature:
                self.object_renderer = ObjectRenderer(**cfg.model.network.object_renderer)
                self.clip_image = CLIPImageEncoder(clip_model=self.clip_model, **cfg.model.network.clip_img_encoder)

            self.text_encoder = hydra.utils.instantiate(cfg.model.network.clip_word_encoder, clip_model=self.clip_model)

            self.match_module = MatchModule(
                **cfg.model.network.matching_module,
                input_channel=cfg.model.network.detector.output_channel *
                              self.hparams.cfg.model.network.use_3d_features +
                              self.hparams.cfg.model.network.use_2d_feature *
                              self.hparams.cfg.model.network.clip_img_encoder.output_channel
            )

        self.contrastive_loss = hydra.utils.instantiate(cfg.model.loss.contrastive_loss)
        self.preference_loss = None
        if getattr(cfg.model.loss.preference_loss, "enabled", False):
            self.preference_loss = hydra.utils.instantiate(cfg.model.loss.preference_loss)
        self.metric_loss = None
        if getattr(cfg.model.loss.metric_loss, "enabled", False):
            self.metric_loss = hydra.utils.instantiate(cfg.model.loss.metric_loss)
        self.counterfactual_pair_loss = None
        pair_loss_cfg = getattr(cfg.model.loss, "counterfactual_pair_loss", None)
        if pair_loss_cfg is not None and getattr(pair_loss_cfg, "enabled", False):
            self.counterfactual_pair_loss = hydra.utils.instantiate(pair_loss_cfg)
        self.stage2 = None
        stage2_cfg = getattr(cfg.model, "stage2", None)
        if stage2_cfg is not None and getattr(stage2_cfg, "enabled", False):
            self.stage2 = Stage2JointModule(stage2_cfg, project_root_path=cfg.project_root_path)
        self.online_trajectory_loss = None
        online_trajectory_cfg = getattr(cfg.model, "online_trajectory", None)
        if online_trajectory_cfg is not None and getattr(online_trajectory_cfg, "enabled", False):
            self.online_trajectory_loss = hydra.utils.instantiate(online_trajectory_cfg.loss)
        self.trajectory_adapter = None
        adapter_cfg = getattr(cfg.model, "trajectory_adapter", None)
        if adapter_cfg is not None and getattr(adapter_cfg, "enabled", False):
            self.trajectory_adapter = TrajectoryConditionedGroundingAdapter(
                proposal_dim=int(adapter_cfg.proposal_dim),
                query_dim=int(adapter_cfg.query_dim),
                hidden_dim=int(adapter_cfg.hidden_dim),
                relation_dim=int(adapter_cfg.relation_dim),
                dropout=float(adapter_cfg.dropout),
                initial_residual_scale=float(adapter_cfg.initial_residual_scale),
                max_residual_scale=float(getattr(adapter_cfg, "max_residual_scale", 1.0)),
                max_logit_residual=float(getattr(adapter_cfg, "max_logit_residual", 4.0)),
                max_feature_residual=float(
                    getattr(adapter_cfg, "max_feature_residual", 2.0)
                ),
                initial_prune_gate=float(getattr(adapter_cfg, "initial_prune_gate", 0.05)),
                risk_gate_enabled=bool(getattr(adapter_cfg, "risk_gate_enabled", True)),
            )

        detector_recovery_cfg = getattr(
            cfg.model, "detector_only_recovery", None
        )
        self.detector_only_recovery_enabled = bool(
            detector_recovery_cfg is not None
            and getattr(detector_recovery_cfg, "enabled", False)
        )
        if self.detector_only_recovery_enabled:
            invalid_recovery_settings = []
            if self.stage1_backbone_name != "m3drefclip":
                invalid_recovery_settings.append("stage1_backbone.name")
            if bool(getattr(cfg.model, "freeze_stage1", False)):
                invalid_recovery_settings.append("freeze_stage1")
            if float(getattr(cfg.model.loss, "detector_weight", 0.0)) <= 0:
                invalid_recovery_settings.append("loss.detector_weight")
            if float(getattr(cfg.model.loss, "reference_weight", 1.0)) != 0:
                invalid_recovery_settings.append("loss.reference_weight")
            if float(getattr(cfg.model.loss, "contrastive_weight", 1.0)) != 0:
                invalid_recovery_settings.append("loss.contrastive_weight")
            if bool(getattr(cfg.model.network, "use_contrastive_loss", True)):
                invalid_recovery_settings.append("network.use_contrastive_loss")
            for name, component in (
                ("loss.preference_loss", self.preference_loss),
                ("loss.metric_loss", self.metric_loss),
                ("loss.counterfactual_pair_loss", self.counterfactual_pair_loss),
                ("online_trajectory", self.online_trajectory_loss),
                ("trajectory_adapter", self.trajectory_adapter),
                ("stage2", self.stage2),
            ):
                if component is not None:
                    invalid_recovery_settings.append(name)
            if invalid_recovery_settings:
                raise ValueError(
                    "detector-only recovery has active grounding settings: "
                    + ", ".join(invalid_recovery_settings)
                )
            self.detector_only_trainable_parameter_names = (
                enforce_detector_only_recovery_parameters(self)
            )
        else:
            self.detector_only_trainable_parameter_names = []

        # evaluator
        self.evaluator = hydra.utils.instantiate(cfg.data.evaluator)
        self.val_test_step_outputs = []
        self.freeze_stage1 = bool(getattr(cfg.model, "freeze_stage1", False))
        if self.freeze_stage1:
            for name, parameter in self.named_parameters():
                if not name.startswith("stage2."):
                    parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        freeze_stage1_bn = bool(
            getattr(
                getattr(self.hparams.cfg.model, "optimizer_groups", None),
                "freeze_stage1_batch_norm",
                False,
            )
        )
        if mode and freeze_stage1_bn:
            freeze_batch_norm_running_stats((
                self.detector,
                self.text_encoder,
                self.match_module,
                getattr(self, "clip_image", None),
            ))
        if mode and self.clip_model is not None:
            self.clip_model.eval()
        if mode and self.freeze_stage1:
            for name, module in self.named_children():
                if name != "stage2":
                    module.eval()
        if mode and self.detector_only_recovery_enabled:
            for name, module in self.named_children():
                if name != "detector":
                    module.eval()
        return self

    def on_after_backward(self):
        nonfinite_count = 0
        for parameter in self.parameters():
            if parameter.grad is None:
                continue
            finite = torch.isfinite(parameter.grad)
            if finite.all():
                continue
            nonfinite_count += int((~finite).sum().item())
            torch.nan_to_num_(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)
        self.log(
            "train_numerics/nonfinite_gradient_values",
            float(nonfinite_count),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def encode_scene(self, data_dict):
        if self.stage1_backbone_name != "m3drefclip":
            return self.detector.encode_scene(data_dict)
        output_dict = self.detector(data_dict)
        batch_size = len(data_dict["scene_id"])
        output_dict["aabb_features_3d"] = common_ops.convert_sparse_tensor_to_dense(
            output_dict["aabb_features"], output_dict["proposal_batch_offsets"],
            self.hparams.cfg.model.network.max_num_proposals
        )
        if self.hparams.cfg.model.network.use_3d_features:
            aabb_features = output_dict["aabb_features"]
        else:
            aabb_features = torch.empty(
                size=(output_dict["aabb_features"].shape[0], 0),
                dtype=output_dict["aabb_features"].dtype, device=self.device
            )
        data_dict["lang_attention_mask"] = None
        if self.hparams.cfg.model.network.use_2d_feature:
            rendered_imgs = self.object_renderer(data_dict, output_dict)
            img_features = self.clip_image(rendered_imgs.permute(dims=(0, 3, 1, 2)))
            views = len(self.hparams.cfg.model.network.object_renderer.eye)
            aabb_img_features = torch.nn.functional.avg_pool1d(
                img_features.permute(1, 0), kernel_size=views, stride=views
            ).permute(1, 0)
            # TODO: adjust mask
            # data_dict["lang_attention_mask"] = data_dict["lang_attention_mask"][:, :77]  # CLIP context length
            # concatenate 2D and 3D features
            aabb_features = torch.nn.functional.normalize(torch.cat((aabb_features, aabb_img_features), dim=1), dim=1)

        output_dict["aabb_features"] = common_ops.convert_sparse_tensor_to_dense(
            aabb_features, output_dict["proposal_batch_offsets"],
            self.hparams.cfg.model.network.max_num_proposals
        )

        output_dict["pred_aabb_min_max_bounds"] = common_ops.convert_sparse_tensor_to_dense(
            output_dict["pred_aabb_min_max_bounds"].reshape(-1, 6), output_dict["proposal_batch_offsets"],
            self.hparams.cfg.model.network.max_num_proposals
        ).reshape(batch_size, self.hparams.cfg.model.network.max_num_proposals, 2, 3)

        return output_dict

    def _apply_trajectory_adapter_step(
        self,
        data_dict,
        output_dict,
        alive_probs=None,
        previous_state=None,
        previous_belief=None,
        feedback=None,
    ):
        batch_size, chunk_size = data_dict["ann_id"].shape[:2]
        num_proposals = output_dict["pred_aabb_min_max_bounds"].shape[1]
        proposal_features = output_dict["aabb_features_3d"].unsqueeze(1).expand(
            -1, chunk_size, -1, -1
        ).reshape(batch_size * chunk_size, num_proposals, -1)
        bboxes = output_dict["pred_aabb_min_max_bounds"].unsqueeze(1).expand(
            -1, chunk_size, -1, -1, -1
        ).reshape(batch_size * chunk_size, num_proposals, 2, 3)
        proposal_mask = output_dict["proposal_masks_dense"].unsqueeze(1).expand(
            -1, chunk_size, -1
        ).reshape(batch_size * chunk_size, num_proposals)
        raw_logits = output_dict["pred_aabb_scores"].reshape(
            batch_size * chunk_size, num_proposals
        )
        adapter_output = self.trajectory_adapter(
            proposal_features=proposal_features,
            bboxes=bboxes,
            raw_logits=raw_logits,
            query_features=output_dict["sentence_features"],
            proposal_mask=proposal_mask,
            alive_probs=None if alive_probs is None else alive_probs.reshape_as(raw_logits),
            previous_state=previous_state,
            previous_belief=previous_belief,
            feedback=feedback,
        )
        output_dict["pred_aabb_scores_raw"] = raw_logits
        output_dict["pred_aabb_scores"] = adapter_output["logits"]
        output_dict["trajectory_adapter_features"] = adapter_output[
            "proposal_features"
        ].reshape(batch_size, chunk_size, num_proposals, -1)
        output_dict["trajectory_adapter_belief"] = adapter_output["belief"].reshape(
            batch_size, chunk_size, num_proposals
        )
        output_dict["trajectory_adapter_state"] = adapter_output["state"].reshape(
            batch_size, chunk_size, -1
        )
        output_dict["trajectory_target_summary"] = adapter_output["target_summary"].reshape(
            batch_size, chunk_size, -1
        )
        return output_dict, adapter_output

    def ground_queries(
        self, data_dict, scene_output, allowed_candidate_mask=None, apply_trajectory_adapter=True
    ):
        if self.stage1_backbone_name == "m3drefclip":
            output_dict = dict(scene_output)
            self.text_encoder(data_dict, output_dict)
            self.match_module(data_dict, output_dict)
        else:
            output_dict = self.detector.ground_queries(data_dict, scene_output)
        if self.trajectory_adapter is not None and apply_trajectory_adapter:
            output_dict, _ = self._apply_trajectory_adapter_step(data_dict, output_dict)
        if allowed_candidate_mask is not None:
            batch_size, chunk_size = data_dict["ann_id"].shape[:2]
            scores = output_dict["pred_aabb_scores"].reshape(batch_size, chunk_size, -1)
            if allowed_candidate_mask.shape != scores.shape:
                raise ValueError(
                    f"allowed_candidate_mask has shape {tuple(allowed_candidate_mask.shape)}, "
                    f"expected {tuple(scores.shape)}"
                )
            output_dict["pred_aabb_scores_raw"] = output_dict["pred_aabb_scores"]
            output_dict["pred_aabb_scores"] = scores.masked_fill(
                ~allowed_candidate_mask.bool(), torch.finfo(scores.dtype).min
            ).flatten(0, 1)
        return output_dict

    def forward(self, data_dict):
        apply_adapter = bool(
            getattr(
                self.hparams.cfg.model.trajectory_adapter,
                "apply_to_stage1_output",
                False,
            )
        ) if self.trajectory_adapter is not None else False
        return self.ground_queries(
            data_dict,
            self.encode_scene(data_dict),
            apply_trajectory_adapter=apply_adapter,
        )

    def ground_query_trajectory(
        self,
        data_dict,
        scene_output=None,
        hard=False,
        token_key="trajectory_clip_tokens",
        round_mask_key="trajectory_round_mask",
        feedback_key="trajectory_feedback",
    ):
        tokens = data_dict[token_key]
        round_mask = data_dict[round_mask_key].bool()
        if tokens.dim() != 4:
            raise ValueError("trajectory_clip_tokens must have shape [B, L, T, 77].")
        if tokens.shape[:3] != round_mask.shape:
            raise ValueError("trajectory token and round-mask shapes do not match.")
        if scene_output is None:
            scene_output = self.encode_scene(data_dict)

        cfg = self.hparams.cfg.model.online_trajectory
        sequence_mode = str(getattr(cfg, "sequence_mode", "ordered")).lower()
        if sequence_mode not in {"ordered", "independent_pairs"}:
            raise ValueError(
                "model.online_trajectory.sequence_mode must be ordered or "
                f"independent_pairs, got {sequence_mode!r}."
            )
        independent_pairs = sequence_mode == "independent_pairs"
        threshold = float(getattr(cfg, "candidate_threshold", self.hparams.cfg.model.inference.output_threshold))
        temperature = float(getattr(cfg, "temperature", 0.5))
        threshold = min(max(threshold, 1e-6), 1.0 - 1e-6)
        threshold_logit = torch.logit(tokens.new_tensor(threshold, dtype=torch.float32))

        batch_size, chunk_size, num_rounds = tokens.shape[:3]
        proposal_mask = scene_output["proposal_masks_dense"].bool()
        num_proposals = proposal_mask.shape[-1]
        alive = proposal_mask.unsqueeze(1).expand(-1, chunk_size, -1)
        alive = alive if hard else alive.float()
        alive_history = []
        score_history = []
        raw_score_history = []
        refined_score_history = []
        candidate_feature_history = []
        belief_history = []
        state_history = []
        target_summary_history = []
        include_cross_modal_features = bool(
            getattr(
                self.hparams.cfg.model.trajectory_adapter,
                "include_cross_modal_stage2_features",
                False,
            )
        )
        previous_state = None
        previous_belief = None

        for round_idx in range(num_rounds):
            step_alive = (
                proposal_mask.unsqueeze(1).expand(-1, chunk_size, -1)
                if independent_pairs
                else alive
            )
            step_alive = step_alive if hard else step_alive.float()
            round_data = dict(data_dict)
            round_data["clip_tokens"] = tokens[:, :, round_idx]
            text_key = (
                "negative_trajectory_query_texts"
                if token_key == "negative_trajectory_clip_tokens"
                else "trajectory_query_texts"
            )
            if text_key in data_dict:
                round_data["stage1_query_text"] = [
                    [
                        data_dict[text_key][batch_idx][lang_idx][round_idx]
                        or data_dict["query_text"][batch_idx][lang_idx]
                        for lang_idx in range(chunk_size)
                    ]
                    for batch_idx in range(batch_size)
                ]
            round_output = self.ground_queries(
                round_data, scene_output, apply_trajectory_adapter=False
            )
            raw_scores = round_output["pred_aabb_scores"].reshape(
                batch_size, chunk_size, -1
            )
            active_round = round_mask[:, :, round_idx].unsqueeze(-1)
            if self.trajectory_adapter is not None:
                feedback = data_dict.get(feedback_key)
                round_feedback = (
                    feedback[:, :, round_idx].reshape(-1)
                    if feedback is not None
                    else raw_scores.new_zeros(batch_size * chunk_size)
                )
                round_output, adapter_output = self._apply_trajectory_adapter_step(
                    round_data,
                    round_output,
                    alive_probs=step_alive,
                    previous_state=None if independent_pairs else previous_state,
                    previous_belief=None if independent_pairs else previous_belief,
                    feedback=round_feedback,
                )
                scores = round_output["pred_aabb_scores"].reshape(
                    batch_size, chunk_size, -1
                )
                next_state = adapter_output["state"]
                next_belief = adapter_output["belief"]
                if independent_pairs:
                    current_state = next_state
                    current_belief = next_belief
                else:
                    if previous_state is None:
                        previous_state = torch.zeros_like(next_state)
                        previous_belief = next_belief
                    active_flat = active_round.reshape(-1, 1)
                    previous_state = torch.where(active_flat, next_state, previous_state)
                    previous_belief = torch.where(active_flat, next_belief, previous_belief)
                    current_state = previous_state
                    current_belief = previous_belief
                candidate_features = round_output["trajectory_adapter_features"]
                target_summary = round_output["trajectory_target_summary"]
            else:
                scores = raw_scores
                base_features = scene_output["aabb_features_3d"]
                candidate_features = base_features.unsqueeze(1).expand(
                    -1, chunk_size, -1, -1
                )
                valid_probability = torch.sigmoid(scores) * step_alive.float()
                current_belief = (
                    valid_probability
                    / valid_probability.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                ).reshape(batch_size * chunk_size, -1)
                if not independent_pairs:
                    previous_belief = current_belief
                current_state = None
                target_summary = scores.new_zeros(
                    (batch_size, chunk_size, 0)
                )
            if include_cross_modal_features:
                cross_modal_features = round_output.get(
                    "aabb_features_cross_modal"
                )
                if cross_modal_features is None:
                    raise KeyError(
                        "Cross-modal Stage2 features were requested, but MatchModule "
                        "did not return aabb_features_cross_modal."
                    )
                cross_modal_features = cross_modal_features.reshape(
                    batch_size, chunk_size, num_proposals, -1
                )
                candidate_features = torch.cat(
                    (candidate_features, cross_modal_features), dim=-1
                )
            if hard:
                keep = torch.sigmoid(scores) >= threshold
                next_alive = step_alive & keep
                current_alive = torch.where(active_round, next_alive, step_alive)
                if not independent_pairs:
                    alive = current_alive
                effective_scores = scores.masked_fill(
                    ~current_alive, torch.finfo(scores.dtype).min
                )
            else:
                keep = torch.sigmoid((scores - threshold_logit) / temperature)
                next_alive = step_alive * keep
                current_alive = torch.where(active_round, next_alive, step_alive)
                if not independent_pairs:
                    alive = current_alive
                effective_scores = scores + torch.log(current_alive.clamp_min(1e-6))
            alive_history.append(current_alive)
            score_history.append(effective_scores)
            raw_score_history.append(raw_scores)
            refined_score_history.append(scores)
            candidate_feature_history.append(candidate_features)
            belief_history.append(current_belief.reshape(batch_size, chunk_size, -1))
            if current_state is not None:
                state_history.append(current_state.reshape(batch_size, chunk_size, -1))
            target_summary_history.append(target_summary)

        return {
            "alive_history": torch.stack(alive_history, dim=2),
            "score_history": torch.stack(score_history, dim=2),
            "raw_score_history": torch.stack(raw_score_history, dim=2),
            "refined_score_history": torch.stack(refined_score_history, dim=2),
            "candidate_feature_history": torch.stack(candidate_feature_history, dim=2),
            "belief_history": torch.stack(belief_history, dim=2),
            "state_history": torch.stack(state_history, dim=2) if state_history else None,
            "target_summary_history": torch.stack(target_summary_history, dim=2),
            "final_alive_mask": current_alive,
            "proposal_mask": proposal_mask,
            "round_mask": round_mask,
        }

    def _build_online_target_labels(self, data_dict, output_dict):
        target_ids = data_dict["trajectory_target_obj_id"].long()
        true_target_ids = data_dict.get("true_target_id")
        if true_target_ids is not None:
            target_ids = torch.where(
                target_ids.ge(0), target_ids, true_target_ids.long()
            )
        batch_size, chunk_size = target_ids.shape
        num_proposals = output_dict["pred_aabb_min_max_bounds"].shape[1]
        labels = torch.zeros(
            (batch_size, chunk_size, num_proposals), dtype=torch.bool, device=target_ids.device
        )
        threshold = float(self.hparams.cfg.model.online_trajectory.target_iou_threshold)
        with torch.no_grad():
            for batch_idx in range(batch_size):
                gt_start = int(data_dict["aabb_count_offsets"][batch_idx].item())
                gt_end = int(data_dict["aabb_count_offsets"][batch_idx + 1].item())
                scene_obj_ids = data_dict["gt_aabb_obj_ids"][gt_start:gt_end].long()
                scene_gt_boxes = data_dict["gt_aabb_min_max_bounds"][gt_start:gt_end]
                pred_boxes = output_dict["pred_aabb_min_max_bounds"][batch_idx]
                valid_proposals = output_dict["proposal_masks_dense"][batch_idx].bool()
                for lang_idx in range(chunk_size):
                    target_id = target_ids[batch_idx, lang_idx]
                    target_box_mask = scene_obj_ids == target_id
                    if target_id < 0 or not target_box_mask.any():
                        continue
                    target_boxes = scene_gt_boxes[target_box_mask]
                    best_iou = pred_boxes.new_zeros(num_proposals)
                    for target_box in target_boxes:
                        ious = get_batch_aabb_pair_ious(
                            pred_boxes,
                            target_box.unsqueeze(0).expand(num_proposals, -1, -1),
                        )
                        best_iou = torch.maximum(best_iou, ious)
                    labels[batch_idx, lang_idx] = (best_iou >= threshold) & valid_proposals
        return labels

    def _build_online_candidate_labels(self, data_dict, output_dict):
        candidate_gt_mask = data_dict.get("trajectory_candidate_gt_mask")
        if candidate_gt_mask is None:
            return None
        batch_size = len(data_dict["scene_id"])
        chunk_size = data_dict["ann_id"].shape[1]
        num_rounds = candidate_gt_mask.shape[-1]
        num_proposals = output_dict["pred_aabb_min_max_bounds"].shape[1]
        labels = torch.zeros(
            (batch_size, chunk_size, num_rounds, num_proposals),
            dtype=torch.bool,
            device=candidate_gt_mask.device,
        )
        threshold = float(
            getattr(
                self.hparams.cfg.model.online_trajectory,
                "candidate_iou_threshold",
                self.hparams.cfg.model.online_trajectory.target_iou_threshold,
            )
        )
        with torch.no_grad():
            for batch_idx in range(batch_size):
                gt_start = int(data_dict["aabb_count_offsets"][batch_idx].item())
                gt_end = int(data_dict["aabb_count_offsets"][batch_idx + 1].item())
                scene_gt_boxes = data_dict["gt_aabb_min_max_bounds"][gt_start:gt_end]
                if scene_gt_boxes.numel() == 0:
                    continue
                pred_boxes = output_dict["pred_aabb_min_max_bounds"][batch_idx]
                valid_proposals = output_dict["proposal_masks_dense"][batch_idx].bool()
                proposal_gt_iou = pred_boxes.new_zeros(
                    (num_proposals, scene_gt_boxes.shape[0])
                )
                for gt_idx, gt_box in enumerate(scene_gt_boxes):
                    proposal_gt_iou[:, gt_idx] = get_batch_aabb_pair_ious(
                        pred_boxes,
                        gt_box.unsqueeze(0).expand(num_proposals, -1, -1),
                    )
                scene_candidate_mask = candidate_gt_mask[gt_start:gt_end]
                for lang_idx in range(chunk_size):
                    for round_idx in range(num_rounds):
                        selected_gt = scene_candidate_mask[:, lang_idx, round_idx].bool()
                        if not selected_gt.any():
                            continue
                        best_iou = proposal_gt_iou[:, selected_gt].max(dim=1).values
                        labels[batch_idx, lang_idx, round_idx] = (
                            (best_iou >= threshold) & valid_proposals
                        )
        return labels

    @staticmethod
    def _restrict_candidate_labels_to_one_to_one(
        candidate_labels, compatible_labels
    ):
        if candidate_labels is None or compatible_labels is None:
            return candidate_labels
        if candidate_labels.shape[:2] != compatible_labels.shape[:2] or (
            candidate_labels.shape[-1] != compatible_labels.shape[-1]
        ):
            raise ValueError(
                "candidate labels and one-to-one compatible labels do not align."
            )
        return candidate_labels & compatible_labels.bool().unsqueeze(2)

    def _online_trajectory_context(self, data_dict, output_dict):
        if "trajectory_clip_tokens" not in data_dict:
            return None
        if not data_dict["trajectory_round_mask"].any():
            return None
        trajectory_output = self.ground_query_trajectory(data_dict, scene_output=output_dict, hard=False)
        batch_size, chunk_size = data_dict["ann_id"].shape
        stage1_initial_scores = output_dict["pred_aabb_scores"].reshape(
            batch_size, chunk_size, -1
        )
        stage1_threshold = float(self.hparams.cfg.model.inference.output_threshold)
        trajectory_output["stage1_initial_scores"] = stage1_initial_scores
        trajectory_output["stage1_initial_candidate_mask"] = (
            output_dict["proposal_masks_dense"].bool().unsqueeze(1)
            & (torch.sigmoid(stage1_initial_scores) >= stage1_threshold)
        )
        trajectory_temperature = float(
            self.hparams.cfg.model.online_trajectory.temperature
        )
        threshold_logit = torch.logit(
            stage1_initial_scores.new_tensor(stage1_threshold).clamp(
                1e-6, 1.0 - 1e-6
            )
        )
        trajectory_output["stage1_initial_alive_probs"] = (
            torch.sigmoid(
                (stage1_initial_scores - threshold_logit)
                / max(trajectory_temperature, 1e-6)
            )
            * output_dict["proposal_masks_dense"].float().unsqueeze(1)
        )
        initial_belief = output_dict.get("trajectory_adapter_belief")
        if initial_belief is None:
            initial_probability = (
                torch.sigmoid(stage1_initial_scores)
                * output_dict["proposal_masks_dense"].float().unsqueeze(1)
            )
            initial_belief = initial_probability / initial_probability.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
        trajectory_output["stage1_initial_belief"] = initial_belief

        initial_features = output_dict.get("trajectory_adapter_features")
        if initial_features is not None:
            if bool(
                self.hparams.cfg.model.trajectory_adapter.include_cross_modal_stage2_features
            ):
                cross_modal = output_dict.get("aabb_features_cross_modal")
                if cross_modal is None:
                    raise KeyError(
                        "Stage1 forward did not expose cross-modal candidate features."
                    )
                cross_modal = cross_modal.reshape(
                    batch_size, chunk_size, initial_features.shape[2], -1
                )
                initial_features = torch.cat(
                    (initial_features, cross_modal), dim=-1
                )
            trajectory_output[
                "stage1_initial_candidate_features"
            ] = initial_features
        negative_mask = data_dict.get("negative_trajectory_round_mask")
        if negative_mask is not None and negative_mask.any():
            negative_output = self.ground_query_trajectory(
                data_dict,
                scene_output=output_dict,
                hard=False,
                token_key="negative_trajectory_clip_tokens",
                round_mask_key="negative_trajectory_round_mask",
                feedback_key="negative_trajectory_feedback",
            )
            trajectory_output["negative_score_history"] = negative_output[
                "refined_score_history"
            ]
            trajectory_output["negative_trajectory_output"] = negative_output
            trajectory_output["negative_round_mask"] = negative_output["round_mask"]
        target_labels = self._build_online_target_labels(data_dict, output_dict)
        trajectory_output["candidate_labels"] = (
            self._restrict_candidate_labels_to_one_to_one(
                self._build_online_candidate_labels(data_dict, output_dict),
                output_dict.get("gt_labels"),
            )
        )
        return trajectory_output, target_labels

    def _online_trajectory_losses(self, data_dict, output_dict, trajectory_context=None):
        if self.online_trajectory_loss is None:
            return None
        if trajectory_context is None:
            trajectory_context = self._online_trajectory_context(data_dict, output_dict)
        if trajectory_context is None:
            return None
        trajectory_output, target_labels = trajectory_context
        return self.online_trajectory_loss(
            alive_history=trajectory_output["alive_history"],
            score_history=trajectory_output["refined_score_history"],
            target_labels=target_labels,
            proposal_mask=trajectory_output["proposal_mask"],
            round_mask=trajectory_output["round_mask"],
            raw_score_history=trajectory_output.get("raw_score_history"),
            compatible_labels=output_dict.get("gt_labels"),
            candidate_labels=trajectory_output.get("candidate_labels"),
            feedback=data_dict.get("trajectory_feedback"),
            negative_score_history=trajectory_output.get("negative_score_history"),
            negative_round_mask=trajectory_output.get("negative_round_mask"),
        )

    def _loss(self, data_dict, output_dict):
        loss_dict = self.detector.loss(data_dict, output_dict)
        detector_weight = float(getattr(self.hparams.cfg.model.loss, "detector_weight", 1.0))
        if detector_weight != 1.0:
            loss_dict = {loss_name: loss_value * detector_weight for loss_name, loss_value in loss_dict.items()}
        if self.detector_only_recovery_enabled:
            return loss_dict
        # reference loss
        loss_dict["reference_loss"] = self.ref_loss(
            output_dict,
            output_dict["pred_aabb_min_max_bounds"],
            output_dict["pred_aabb_scores"],
            data_dict["gt_aabb_min_max_bounds"],
            data_dict["gt_target_obj_id_mask"].permute(dims=(1, 0)),
            data_dict["aabb_count_offsets"],
            sample_weights=data_dict.get("sample_weight"),
        )
        loss_dict["reference_loss"] *= float(getattr(self.hparams.cfg.model.loss, "reference_weight", 1.0))
        if self.hparams.cfg.model.network.use_contrastive_loss:
            # contrastive loss
            loss_dict["contrastive_loss"] = self.contrastive_loss(
                output_dict["aabb_features_inter"],
                output_dict["sentence_features"],
                output_dict["gt_labels"],
                output_dict.get("proposal_masks_dense"),
                sample_weights=data_dict.get("sample_weight"),
            )
            loss_dict["contrastive_loss"] *= float(getattr(self.hparams.cfg.model.loss, "contrastive_weight", 1.0))
        if self.preference_loss is not None:
            loss_dict["preference_loss"] = self.preference_loss(
                output_dict["pred_aabb_scores"].reshape_as(output_dict["gt_labels"]),
                labels=output_dict["gt_labels"],
                valid_mask=output_dict.get("proposal_masks_dense"),
            ) * float(getattr(self.hparams.cfg.model.loss.preference_loss, "weight", 1.0))
        if self.metric_loss is not None:
            loss_dict["metric_loss"] = self.metric_loss(
                output_dict["pred_aabb_scores"].reshape_as(output_dict["gt_labels"]),
                labels=output_dict["gt_labels"],
                valid_mask=output_dict.get("proposal_masks_dense"),
            ) * float(getattr(self.hparams.cfg.model.loss.metric_loss, "weight", 1.0))
        if self.counterfactual_pair_loss is not None:
            if "counterfactual_pair_id" not in data_dict or "diagnostic_sample_id" not in data_dict:
                raise ValueError(
                    "counterfactual pair loss requires pair-aware diagnostic metadata in the data batch"
                )
            loss_dict["counterfactual_pair_loss"] = self.counterfactual_pair_loss(
                output_dict["pred_aabb_scores"].reshape_as(output_dict["gt_labels"]),
                labels=output_dict["gt_labels"],
                pair_ids=data_dict["counterfactual_pair_id"],
                sample_ids=data_dict["diagnostic_sample_id"],
                valid_mask=output_dict.get("proposal_masks_dense"),
                sample_weights=data_dict.get("sample_weight"),
            ) * float(getattr(self.hparams.cfg.model.loss.counterfactual_pair_loss, "weight", 1.0))
        return loss_dict

    def _build_optimizer(self, params):
        optimizer_cfg = self.hparams.cfg.model.optimizer
        optimizer_cls = hydra.utils.get_class(optimizer_cfg._target_)
        optimizer_kwargs = OmegaConf.to_container(optimizer_cfg, resolve=True)
        optimizer_kwargs.pop("_target_", None)
        return optimizer_cls(params, **optimizer_kwargs)

    def configure_optimizers(self):
        optimizer_groups_cfg = getattr(
            self.hparams.cfg.model, "optimizer_groups", None
        )
        use_discriminative_lr = bool(
            optimizer_groups_cfg is not None
            and getattr(optimizer_groups_cfg, "enabled", False)
        )
        stage2_param_ids = (
            {id(param) for param in self.stage2.parameters() if param.requires_grad}
            if self.stage2 is not None
            else set()
        )

        if use_discriminative_lr:
            grouped_params = {
                "detector_backbone": [],
                "detector_heads": [],
                "grounding": [],
                "trajectory_adapter": [],
                "stage2": [],
            }
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if id(param) in stage2_param_ids:
                    group_name = "stage2"
                elif (
                    self.stage1_backbone_name == "tsp3d"
                    and name.startswith("detector.backbone.vision_backbone.")
                ):
                    group_name = "detector_backbone"
                elif (
                    self.stage1_backbone_name == "tsp3d"
                    and name.startswith("detector.backbone.head.")
                ):
                    group_name = "detector_heads"
                elif (
                    self.stage1_backbone_name == "tsp3d"
                    and name.startswith("detector.backbone.text_projector.")
                ):
                    group_name = "grounding"
                elif name.startswith((
                    "detector.backbone.", "detector.point_encoder."
                )):
                    group_name = "detector_backbone"
                elif name.startswith("detector."):
                    group_name = "detector_heads"
                elif name.startswith("trajectory_adapter."):
                    group_name = "trajectory_adapter"
                else:
                    group_name = "grounding"
                grouped_params[group_name].append(param)

            group_lrs = {
                "detector_backbone": float(
                    optimizer_groups_cfg.detector_backbone_lr
                ),
                "detector_heads": float(optimizer_groups_cfg.detector_heads_lr),
                "grounding": float(optimizer_groups_cfg.grounding_lr),
                "trajectory_adapter": float(
                    optimizer_groups_cfg.trajectory_adapter_lr
                ),
                "stage2": self.stage2.lr if self.stage2 is not None else 0.0,
            }
            param_groups = []
            for group_name, params in grouped_params.items():
                if not params:
                    continue
                group = {
                    "name": group_name,
                    "params": params,
                    "lr": group_lrs[group_name],
                }
                if group_name == "stage2":
                    group["weight_decay"] = self.stage2.weight_decay
                param_groups.append(group)
            return self._build_optimizer(param_groups)

        trainable_params = [param for param in self.parameters() if param.requires_grad]
        if (
            self.stage2 is not None
            and getattr(self.hparams.cfg.model.stage2, "use_separate_lr", True)
        ):
            base_params = [
                param
                for param in trainable_params
                if id(param) not in stage2_param_ids
            ]
            stage2_params = [
                param for param in trainable_params if id(param) in stage2_param_ids
            ]
            return self._build_optimizer([
                {"name": "stage1", "params": base_params},
                {
                    "name": "stage2",
                    "params": stage2_params,
                    "lr": self.stage2.lr,
                    "weight_decay": self.stage2.weight_decay,
                },
            ])
        return self._build_optimizer(trainable_params)

    def training_step(self, data_dict, idx):
        output_dict = (
            self.encode_scene(data_dict)
            if self.detector_only_recovery_enabled
            else self(data_dict)
        )
        fallback_mask = output_dict.get("fallback_proposal_masks_dense")
        fallback_rate = output_dict["aabb_features_3d"].new_zeros(())
        if fallback_mask is not None:
            fallback_rate = fallback_mask.any(dim=1).float().mean()
            self.log(
                "train_detector/fallback_scene_rate",
                fallback_rate,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        loss_dict = self._loss(data_dict, output_dict)
        stage2_end_to_end = bool(
            self.stage2 is not None and self.stage2.end_to_end_enabled
        )
        trajectory_context = None
        if self.online_trajectory_loss is not None or stage2_end_to_end:
            trajectory_context = self._online_trajectory_context(data_dict, output_dict)

        # calculate the total loss and log
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        stage1_loss = total_loss
        detector_loss_names = {
            "semantic_loss", "offset_norm_loss", "offset_dir_loss", "score_loss"
        }
        detector_loss = sum(
            (
                loss_value
                for loss_name, loss_value in loss_dict.items()
                if loss_name in detector_loss_names
            ),
            total_loss.new_zeros(()),
        )
        grounding_loss = (
            total_loss.new_zeros(())
            if self.detector_only_recovery_enabled
            else stage1_loss - detector_loss
        )
        weighted_trajectory_loss = total_loss.new_zeros(())
        trajectory_loss_dict = self._online_trajectory_losses(
            data_dict, output_dict, trajectory_context=trajectory_context
        )
        if trajectory_loss_dict is not None:
            weighted_trajectory_loss = trajectory_loss_dict["online_trajectory_loss"] * float(
                self.hparams.cfg.model.online_trajectory.weight
            )
            total_loss = total_loss + weighted_trajectory_loss
            for loss_name, loss_value in trajectory_loss_dict.items():
                self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        weighted_stage2_loss = total_loss.new_zeros(())
        stage2_loss_dict = None
        if self.stage2 is not None:
            if stage2_end_to_end:
                trajectory_output, target_labels = trajectory_context or (None, None)
                end_to_end_batch = self.stage2.build_end_to_end_batch(
                    data_dict=data_dict,
                    stage1_output=output_dict,
                    trajectory_output=trajectory_output,
                    target_labels=target_labels,
                )
                stage2_loss_dict = None
                if end_to_end_batch is not None:
                    stage2_loss_dict = self.stage2.loss(
                        end_to_end_batch["batch"],
                        trajectory_batch=end_to_end_batch["trajectory_batch"],
                        use_offline_trajectory=False,
                    )
                    self.log(
                        "train_stage2/end_to_end_trajectories",
                        float(end_to_end_batch["num_trajectories"]),
                        on_step=True,
                        on_epoch=False,
                    )
                    self.log(
                        "train_stage2/end_to_end_rounds",
                        float(end_to_end_batch["num_rounds"]),
                        on_step=True,
                        on_epoch=False,
                    )
                    self.log(
                        "train_stage2/return_preference_steps",
                        float(end_to_end_batch["num_preference_steps"]),
                        on_step=True,
                        on_epoch=False,
                    )
                    self.log(
                        "train_stage2/missing_metadata",
                        float(end_to_end_batch["missing_metadata"]),
                        on_step=True,
                        on_epoch=False,
                    )
                    self.log(
                        "train_stage2/missing_target_proposal",
                        float(end_to_end_batch["missing_target_proposal"]),
                        on_step=True,
                        on_epoch=False,
                    )
                    self.log(
                        "train_stage2/missing_initial_target",
                        float(end_to_end_batch["missing_initial_target"]),
                        on_step=True,
                        on_epoch=False,
                    )
                else:
                    self.log(
                        "train_stage2/end_to_end_trajectories",
                        0.0,
                        on_step=True,
                        on_epoch=False,
                    )
            else:
                stage2_batch = self.stage2.next_batch(self.device)
                stage2_loss_dict = self.stage2.loss(stage2_batch)
            if stage2_loss_dict is None:
                stage2_loss_dict = {}
            for loss_name, loss_value in stage2_loss_dict.items():
                weighted_loss_value = (
                    loss_value * self.stage2.end_to_end_weight
                    if loss_name == "stage2_loss" and stage2_end_to_end
                    else loss_value if loss_name == "stage2_loss"
                    else 0
                )
                total_loss = total_loss + weighted_loss_value
                if loss_name == "stage2_loss":
                    weighted_stage2_loss = weighted_loss_value
                self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        if not total_loss.requires_grad and self.stage2 is not None:
            # A frozen Stage1 can produce a batch with no recoverable target and
            # therefore no Stage2 supervision. Keep DDP/backward valid while
            # making this batch an explicit zero update.
            stage2_anchor = next(
                (parameter for parameter in self.stage2.parameters() if parameter.requires_grad),
                None,
            )
            if stage2_anchor is not None:
                total_loss = total_loss + stage2_anchor.reshape(-1)[0] * 0.0
        self._terminal_train_components = {
            "stage1": stage1_loss.detach(),
            "detector": detector_loss.detach(),
            "grounding": grounding_loss.detach(),
            "trajectory": weighted_trajectory_loss.detach(),
            "stage2": weighted_stage2_loss.detach(),
            "fallback": fallback_rate.detach(),
            "traj_survival": (
                trajectory_loss_dict.get("online_trajectory_survival_loss", total_loss.new_zeros(()))
                if trajectory_loss_dict is not None else total_loss.new_zeros(())
            ).detach(),
            "traj_initial": (
                trajectory_loss_dict.get("online_trajectory_initial_survival_loss", total_loss.new_zeros(()))
                if trajectory_loss_dict is not None else total_loss.new_zeros(())
            ).detach(),
            "traj_coverage": (
                trajectory_loss_dict.get("online_trajectory_coverage_loss", total_loss.new_zeros(()))
                if trajectory_loss_dict is not None else total_loss.new_zeros(())
            ).detach(),
            "traj_dpo": (
                trajectory_loss_dict.get("online_trajectory_dpo_loss", total_loss.new_zeros(()))
                if trajectory_loss_dict is not None else total_loss.new_zeros(())
            ).detach(),
            "traj_set": (
                trajectory_loss_dict.get(
                    "online_trajectory_candidate_set_loss", total_loss.new_zeros(())
                )
                if trajectory_loss_dict is not None else total_loss.new_zeros(())
            ).detach(),
            "traj_raw_set": (
                trajectory_loss_dict.get(
                    "online_trajectory_raw_candidate_set_loss", total_loss.new_zeros(())
                )
                if trajectory_loss_dict is not None else total_loss.new_zeros(())
            ).detach(),
            "stage2_selector": (
                stage2_loss_dict.get("stage2_selector_loss", total_loss.new_zeros(()))
                if stage2_loss_dict is not None else total_loss.new_zeros(())
            ).detach(),
            "stage2_mask": (
                stage2_loss_dict.get("stage2_mask_loss", total_loss.new_zeros(()))
                if stage2_loss_dict is not None else total_loss.new_zeros(())
            ).detach(),
            "stage2_stop": (
                stage2_loss_dict.get("stage2_stop_loss", total_loss.new_zeros(()))
                if stage2_loss_dict is not None else total_loss.new_zeros(())
            ).detach(),
            "stage2_preference": (
                stage2_loss_dict.get("stage2_preference_loss", total_loss.new_zeros(()))
                if stage2_loss_dict is not None else total_loss.new_zeros(())
            ).detach(),
            "stage2_metric": (
                stage2_loss_dict.get("stage2_metric_loss", total_loss.new_zeros(()))
                if stage2_loss_dict is not None else total_loss.new_zeros(())
            ).detach(),
            "stage2_dpo": (
                stage2_loss_dict.get("stage2_trajectory_dpo_loss", total_loss.new_zeros(()))
                if stage2_loss_dict is not None else total_loss.new_zeros(())
            ).detach(),
        }
        self.log(f"train_loss/total_loss", total_loss, on_step=True, on_epoch=False)
        return total_loss

    def validation_step(self, data_dict, idx):
        output_dict = self(data_dict)
        fallback_mask = output_dict.get("fallback_proposal_masks_dense")
        if fallback_mask is not None:
            self.log(
                "val_detector/fallback_scene_rate",
                fallback_mask.any(dim=1).float().mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(data_dict["scene_id"]),
            )
        loss_dict = self._loss(data_dict, output_dict)
        # calculate the total loss and log
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(
                f"val_loss/{loss_name}",
                loss_value,
                on_step=True,
                on_epoch=False,
                batch_size=len(data_dict["scene_id"]),
            )
        stage2_validation = bool(
            self.stage2 is not None
            and self.stage2.end_to_end_enabled
            and self.stage2.validation_enabled
        )
        if stage2_validation:
            for batch_idx, scene_id in enumerate(data_dict["scene_id"]):
                for lang_idx, ann_id in enumerate(
                    data_dict["ann_id"][batch_idx].tolist()
                ):
                    key = (str(scene_id), int(ann_id))
                    self._stage2_validation_records.setdefault(key, False)
                    self._stage2_validation_initial_survival.setdefault(key, False)
                    self._stage2_validation_final_survival.setdefault(key, False)
                    self._stage2_validation_aligned.setdefault(key, False)
                    self._stage2_validation_eval_types.setdefault(
                        key, str(data_dict["eval_type"][batch_idx][lang_idx])
                    )
        trajectory_context = None
        if self.online_trajectory_loss is not None or stage2_validation:
            trajectory_context = self._online_trajectory_context(data_dict, output_dict)
        trajectory_loss_dict = self._online_trajectory_losses(
            data_dict,
            output_dict,
            trajectory_context=trajectory_context,
        )
        if trajectory_loss_dict is not None:
            total_loss += trajectory_loss_dict["online_trajectory_loss"] * float(
                self.hparams.cfg.model.online_trajectory.weight
            )
            for loss_name, loss_value in trajectory_loss_dict.items():
                self.log(
                    f"val_loss/{loss_name}",
                    loss_value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=len(data_dict["scene_id"]),
                )
        if stage2_validation and trajectory_context is not None:
            trajectory_output, target_labels = trajectory_context
            batch_size, chunk_size = data_dict["ann_id"].shape
            proposal_mask = output_dict["proposal_masks_dense"].bool().unsqueeze(1)
            stage1_scores = output_dict["pred_aabb_scores"].reshape(
                batch_size, chunk_size, -1
            )
            stage1_keep = proposal_mask & (
                torch.sigmoid(stage1_scores)
                >= float(self.hparams.cfg.model.inference.output_threshold)
            )
            initial_survival = (target_labels.bool() & stage1_keep).any(dim=2)
            for batch_idx, scene_id in enumerate(data_dict["scene_id"]):
                for lang_idx, ann_id in enumerate(
                    data_dict["ann_id"][batch_idx].tolist()
                ):
                    self._stage2_validation_initial_survival[
                        (str(scene_id), int(ann_id))
                    ] = bool(initial_survival[batch_idx, lang_idx].item())
            end_to_end = self.stage2.build_end_to_end_batch(
                data_dict=data_dict,
                stage1_output=output_dict,
                trajectory_output=trajectory_output,
                target_labels=target_labels,
            )
            if end_to_end is not None:
                records, details = self.stage2.evaluate_end_to_end_batch(
                    end_to_end,
                    stage1_threshold=float(
                        self.hparams.cfg.model.inference.output_threshold
                    ),
                    return_details=True,
                )
                self._stage2_validation_records.update(records)
                self._stage2_validation_aligned.update(details["aligned"])
                self._stage2_validation_final_survival.update(
                    details["final_target_survival"]
                )
        self.log(
            "val_loss/total_loss",
            total_loss,
            on_step=True,
            on_epoch=False,
            batch_size=len(data_dict["scene_id"]),
        )

        # get predictions and gts
        self.val_test_step_outputs.append((self._parse_pred_results(data_dict, output_dict), self._parse_gt(data_dict)))

    def test_step(self, data_dict, idx):
        output_dict = self(data_dict)
        self.val_test_step_outputs.append(
            (self._parse_pred_results(data_dict, output_dict), self._parse_gt(data_dict))
        )

    def on_validation_epoch_start(self):
        self._stage2_validation_records = {}
        self._stage2_validation_initial_survival = {}
        self._stage2_validation_final_survival = {}
        self._stage2_validation_aligned = {}
        self._stage2_validation_eval_types = {}

    def on_validation_epoch_end(self):
        total_pred_results = {}
        total_gt_results = {}
        for pred_results, gt_results in self.val_test_step_outputs:
            total_pred_results.update(pred_results)
            total_gt_results.update(gt_results)
        self.val_test_step_outputs.clear()
        self.evaluator.set_ground_truths(total_gt_results)
        results = self.evaluator.evaluate(total_pred_results)

        # log
        for metric_name, result in results.items():
            for breakdown, value in result.items():
                self.log(
                    f"val_eval/{metric_name}_{breakdown}",
                    value,
                    sync_dist=True,
                    batch_size=1,
                )

        stage2_acc = None
        if (
            self.stage2 is not None
            and self.stage2.end_to_end_enabled
            and self.stage2.validation_enabled
        ):
            record_groups = {
                "correct": self._stage2_validation_records,
                "initial": self._stage2_validation_initial_survival,
                "final": self._stage2_validation_final_survival,
                "aligned": self._stage2_validation_aligned,
                "eval_type": self._stage2_validation_eval_types,
            }
            if dist.is_available() and dist.is_initialized():
                for name, group in tuple(record_groups.items()):
                    gathered = [None] * dist.get_world_size()
                    dist.all_gather_object(gathered, group)
                    merged = {}
                    for rank_records in gathered:
                        merged.update(rank_records)
                    record_groups[name] = merged
            records = record_groups["correct"]
            total = len(records)
            correct = sum(bool(value) for value in records.values())
            conditional_correct = sum(
                bool(records.get(key, False)) and bool(value)
                for key, value in record_groups["initial"].items()
            )
            aligned_correct = sum(
                bool(records.get(key, False)) and bool(value)
                for key, value in record_groups["aligned"].items()
            )
            initial_count = sum(
                bool(value) for value in record_groups["initial"].values()
            )
            final_count = sum(
                bool(value) for value in record_groups["final"].values()
            )
            aligned_count = sum(
                bool(value) for value in record_groups["aligned"].values()
            )
            acc = float(correct / max(total, 1))
            stage2_acc = acc
            self.log("val_stage2/acc", acc, sync_dist=True, batch_size=1)
            self.log(
                "val_stage2/num", float(total), sync_dist=True, batch_size=1
            )
            self.log(
                "val_stage2/initial_target_survival",
                float(initial_count / max(total, 1)),
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "val_stage2/final_target_survival",
                float(final_count / max(total, 1)),
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "val_stage2/conditional_acc",
                float(conditional_correct / max(initial_count, 1)),
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "val_stage2/conditional_num",
                float(initial_count),
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "val_stage2/aligned_num",
                float(aligned_count),
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "val_stage2/aligned_acc",
                float(aligned_correct / max(aligned_count, 1)),
                sync_dist=True,
                batch_size=1,
            )
            mt_keys = {
                key
                for key, eval_type in record_groups["eval_type"].items()
                if str(eval_type) == "mt"
            }
            mt_total = len(mt_keys)
            mt_correct = sum(bool(records.get(key, False)) for key in mt_keys)
            mt_initial_count = sum(
                bool(record_groups["initial"].get(key, False)) for key in mt_keys
            )
            mt_final_count = sum(
                bool(record_groups["final"].get(key, False)) for key in mt_keys
            )
            mt_conditional_correct = sum(
                bool(records.get(key, False))
                and bool(record_groups["initial"].get(key, False))
                for key in mt_keys
            )
            mt_acc = float(mt_correct / max(mt_total, 1))
            self.log("val_stage2/acc_mt", mt_acc, sync_dist=True, batch_size=1)
            self.log(
                "val_stage2/num_mt", float(mt_total), sync_dist=True, batch_size=1
            )
            self.log(
                "val_stage2/initial_target_survival_mt",
                float(mt_initial_count / max(mt_total, 1)),
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "val_stage2/final_target_survival_mt",
                float(mt_final_count / max(mt_total, 1)),
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "val_stage2/conditional_acc_mt",
                float(mt_conditional_correct / max(mt_initial_count, 1)),
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "val_stage2/conditional_num_mt",
                float(mt_initial_count),
                sync_dist=True,
                batch_size=1,
            )
        if stage2_acc is not None:
            ts50 = float(results.get("target_survival@0.5", {}).get("overall", 0.0))
            f1_50 = float(results.get("f1@0.5", {}).get("overall", 0.0))
            stage2_cfg = self.hparams.cfg.model.stage2
            joint_score = (
                float(getattr(stage2_cfg, "validation_joint_ts_weight", 0.5)) * ts50
                + float(getattr(stage2_cfg, "validation_joint_f1_weight", 0.25)) * f1_50
                + float(getattr(stage2_cfg, "validation_joint_acc_weight", 0.25))
                * stage2_acc
            )
            self.log(
                "val_eval/joint_score_overall",
                joint_score,
                sync_dist=True,
                batch_size=1,
            )
            mt_ts50 = float(results.get("target_survival@0.5", {}).get("mt", 0.0))
            mt_f1_50 = float(results.get("f1@0.5", {}).get("mt", 0.0))
            mt_joint_score = (
                float(getattr(stage2_cfg, "validation_joint_ts_weight", 0.5))
                * mt_ts50
                + float(getattr(stage2_cfg, "validation_joint_f1_weight", 0.25))
                * mt_f1_50
                + float(getattr(stage2_cfg, "validation_joint_acc_weight", 0.25))
                * mt_acc
            )
            self.log(
                "val_eval/joint_score_mt",
                mt_joint_score,
                sync_dist=True,
                batch_size=1,
            )

    def on_test_epoch_end(self):
        total_pred_results = {}
        total_gt_results = {}
        for pred_results, gt_results in self.val_test_step_outputs:
            total_pred_results.update(pred_results)
            total_gt_results.update(gt_results)
        self.val_test_step_outputs.clear()
        self._save_predictions(total_pred_results)

    def _parse_pred_results(self, data_dict, output_dict):
        batch_size, lang_chunk_size = data_dict["ann_id"].shape
        # Optional experiment mode: decode ScanRefer/Nr3D with Multi3DRefer-style
        # thresholding to keep multiple proposals per query.
        force_multi_object_decode = getattr(self.hparams.cfg.model.inference, "force_multi_object_decode", False)
        pred_scores = output_dict["pred_aabb_scores"].reshape(batch_size, lang_chunk_size, -1)

        if self.dataset_name in ("ScanRefer", "Nr3D") and not force_multi_object_decode:
            pred_aabb_score_masks = torch.zeros_like(pred_scores, dtype=torch.bool)
            pred_aabb_score_masks.scatter_(2, pred_scores.argmax(dim=2, keepdim=True), True)
        elif self.dataset_name == "Multi3DRefer" or force_multi_object_decode:
            pred_aabb_score_masks = (
                    torch.sigmoid(pred_scores) >= self.hparams.cfg.model.inference.output_threshold
            )
        else:
            raise NotImplementedError
        pred_aabb_score_masks &= output_dict["proposal_masks_dense"].bool().unsqueeze(1)

        pred_results = {}
        for i in range(batch_size):
            for j in range(lang_chunk_size):
                selected_mask = pred_aabb_score_masks[i, j]
                raw_proposal_mask = output_dict["proposal_masks_dense"][i].bool()
                pred_aabbs = output_dict["pred_aabb_min_max_bounds"][i][selected_mask]
                raw_iou_survival = self._raw_proposal_target_survival_iou(
                    data_dict, output_dict, batch_idx=i, lang_idx=j
                )
                pred_results[
                    (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                     data_dict["ann_id"][i][j].item())
                ] = {
                    "aabb_bound": (pred_aabbs + data_dict["scene_center_xyz"][i]).cpu().numpy(),
                    "target_survival_no_iou": self._target_survival_no_iou(
                        data_dict, output_dict, batch_idx=i, lang_idx=j, selected_mask=selected_mask
                    ),
                    "raw_proposal_target_survival_no_iou": self._target_survival_no_iou(
                        data_dict, output_dict, batch_idx=i, lang_idx=j, selected_mask=raw_proposal_mask
                    ),
                    "raw_proposal_target_survival@0.25": raw_iou_survival[0.25],
                    "raw_proposal_target_survival@0.5": raw_iou_survival[0.5],
                }
        return pred_results

    @staticmethod
    def _raw_proposal_target_survival_iou(data_dict, output_dict, batch_idx, lang_idx):
        aabb_start_idx = int(data_dict["aabb_count_offsets"][batch_idx].item())
        aabb_end_idx = int(data_dict["aabb_count_offsets"][batch_idx + 1].item())
        target_mask = data_dict["gt_target_obj_id_mask"][aabb_start_idx:aabb_end_idx, lang_idx].bool()
        target_aabbs = data_dict["gt_aabb_min_max_bounds"][aabb_start_idx:aabb_end_idx][target_mask]
        valid_mask = output_dict["proposal_masks_dense"][batch_idx].bool()
        pred_aabbs = output_dict["pred_aabb_min_max_bounds"][batch_idx][valid_mask]
        if target_aabbs.numel() == 0:
            return {0.25: np.nan, 0.5: np.nan}
        if pred_aabbs.numel() == 0:
            return {0.25: 0.0, 0.5: 0.0}
        best_iou = pred_aabbs.new_zeros(target_aabbs.shape[0])
        for target_idx, target_aabb in enumerate(target_aabbs):
            ious = get_batch_aabb_pair_ious(
                pred_aabbs,
                target_aabb.unsqueeze(0).expand(pred_aabbs.shape[0], -1, -1),
            )
            best_iou[target_idx] = ious.max()
        return {
            threshold: float((best_iou >= threshold).any().item())
            for threshold in (0.25, 0.5)
        }

    @staticmethod
    def _target_survival_no_iou(data_dict, output_dict, batch_idx, lang_idx, selected_mask):
        proposal_obj_ids = output_dict.get("proposal_gt_obj_ids_dense")
        if proposal_obj_ids is not None:
            aabb_start_idx = int(data_dict["aabb_count_offsets"][batch_idx].item())
            aabb_end_idx = int(data_dict["aabb_count_offsets"][batch_idx + 1].item())
            target_mask = data_dict["gt_target_obj_id_mask"][
                aabb_start_idx:aabb_end_idx, lang_idx
            ].bool()
            target_obj_ids = data_dict["gt_aabb_obj_ids"][
                aabb_start_idx:aabb_end_idx
            ][target_mask].long()
            selected_obj_ids = proposal_obj_ids[batch_idx][selected_mask].long()
            selected_obj_ids = selected_obj_ids[selected_obj_ids >= 0]
            return float(
                target_obj_ids.numel() > 0
                and selected_obj_ids.numel() > 0
                and torch.isin(selected_obj_ids, target_obj_ids).any().item()
            )

        proposal_indices_dense = output_dict.get("proposal_indices_dense")
        if proposal_indices_dense is None:
            return np.nan

        proposal_scores = output_dict.get("proposal_scores")
        if proposal_scores is None:
            return np.nan
        _, cluster_point_idxs, proposals_offset = proposal_scores

        aabb_start_idx = int(data_dict["aabb_count_offsets"][batch_idx].item())
        aabb_end_idx = int(data_dict["aabb_count_offsets"][batch_idx + 1].item())
        target_mask = data_dict["gt_target_obj_id_mask"][aabb_start_idx:aabb_end_idx, lang_idx].bool()
        target_obj_ids = data_dict["gt_aabb_obj_ids"][aabb_start_idx:aabb_end_idx][target_mask].long()
        if target_obj_ids.numel() == 0:
            return np.nan

        point_start_idx = int(data_dict["point_count_offsets"][batch_idx].item())
        point_end_idx = int(data_dict["point_count_offsets"][batch_idx + 1].item())
        scene_instance_ids = data_dict["instance_ids"][point_start_idx:point_end_idx].long()
        valid_scene_instance_ids = scene_instance_ids[scene_instance_ids >= 0]
        instance_offset = valid_scene_instance_ids.min() if valid_scene_instance_ids.numel() > 0 else target_obj_ids.new_tensor(0)
        target_instance_ids = target_obj_ids + instance_offset

        selected_proposal_indices = proposal_indices_dense[batch_idx][selected_mask]
        for proposal_idx in selected_proposal_indices.tolist():
            if proposal_idx < 0:
                continue
            start = int(proposals_offset[proposal_idx].item())
            end = int(proposals_offset[proposal_idx + 1].item())
            proposal_point_idxs = cluster_point_idxs[start:end].long()
            proposal_instance_ids = data_dict["instance_ids"][proposal_point_idxs].long()
            proposal_instance_ids = proposal_instance_ids[proposal_instance_ids >= 0]
            if proposal_instance_ids.numel() > 0 and torch.isin(proposal_instance_ids, target_instance_ids).any():
                return 1.0
        return 0.0

    def _parse_gt(self, data_dict):
        batch_size, lang_chunk_size = data_dict["ann_id"].shape
        gts = {}
        gt_target_obj_id_masks = data_dict["gt_target_obj_id_mask"].permute(1, 0)
        for i in range(batch_size):
            aabb_start_idx = data_dict["aabb_count_offsets"][i]
            aabb_end_idx = data_dict["aabb_count_offsets"][i + 1]
            for j in range(lang_chunk_size):
                gts[
                    (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                     data_dict["ann_id"][i][j].item())
                ] = {
                    "aabb_bound":
                        (data_dict["gt_aabb_min_max_bounds"][aabb_start_idx:aabb_end_idx][gt_target_obj_id_masks[j]
                    [aabb_start_idx:aabb_end_idx]] + data_dict["scene_center_xyz"][i]).cpu().numpy(),
                    "eval_type": data_dict["eval_type"][i][j]
                }
        return gts

    def _save_predictions(self, predictions):
        export_multi3d_style = getattr(self.hparams.cfg.model.inference, "export_multi3drefer_style", False)
        scene_pred = {}
        for key, value in predictions.items():
            scene_id = key[0]
            if key[0] not in scene_pred:
                scene_pred[scene_id] = []
            corners = np.empty(shape=(value["aabb_bound"].shape[0], 8, 3), dtype=np.float32)
            for i, aabb in enumerate(value["aabb_bound"]):
                min_point = aabb[0]
                max_point = aabb[1]
                corners[i] = np.array([
                    [x, y, z]
                    for x in [min_point[0], max_point[0]]
                    for y in [min_point[1], max_point[1]]
                    for z in [min_point[2], max_point[2]]
                ], dtype=np.float32)

            if self.dataset_name in ("ScanRefer", "Nr3D"):
                # ScanRefer/Nr3D entries must keep object_id, because
                # (scene_id, object_id, ann_id) is the unique query key.
                scene_pred[scene_id].append({
                    "object_id": key[1],
                    "ann_id": key[2],
                    "aabb": corners.tolist()
                })
            elif self.dataset_name == "Multi3DRefer":
                scene_pred[scene_id].append({
                    "ann_id": key[2],
                    "aabb": corners.tolist()
                })
        prediction_output_root_path = os.path.join(
            self.hparams.cfg.pred_path, self.hparams.cfg.data.inference.split
        )
        os.makedirs(prediction_output_root_path, exist_ok=True)
        for scene_id in tqdm(scene_pred.keys(), desc="Saving predictions"):
            with open(os.path.join(prediction_output_root_path, f"{scene_id}.json"), "w") as f:
                json.dump(scene_pred[scene_id], f, indent=2)
        self.print(f"==> Complete. Saved at: {os.path.abspath(prediction_output_root_path)}")
