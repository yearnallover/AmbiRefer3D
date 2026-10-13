import os
import json
import clip
import torch
import hydra
import numpy as np
from tqdm import tqdm
import lightning.pytorch as pl
from m3drefclip.common_ops.functions import common_ops
from m3drefclip.model.vision_module.pointgroup import PointGroup
from m3drefclip.model.cross_modal_module.match_module import MatchModule
from m3drefclip.model.vision_module.object_renderer import ObjectRenderer
from m3drefclip.model.vision_module.clip_image_encoder import CLIPImageEncoder


import torch.distributed as dist
from m3drefclip.util.utils import get_batch_aabb_pair_ious
from m3drefclip.model.stage1_backbones import BUTDDETRStage1Backbone, VistaStage1Backbone, TSP3DStage1Backbone

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

        self.evaluator = hydra.utils.instantiate(cfg.data.evaluator)
        self.val_test_step_outputs = []


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


    def train(self, mode=True):
        super().train(mode)
        if self.clip_model is not None:
            self.clip_model.eval()
        return self

    def forward(self, data_dict):
        output_dict = self.encode_scene(data_dict)
        if self.stage1_backbone_name == 'm3drefclip':
            self.text_encoder(data_dict, output_dict)
            self.match_module(data_dict, output_dict)
        else:
            output_dict = self.detector.ground_queries(data_dict, output_dict)
        return output_dict

    def on_validation_epoch_end(self):
        outputs = self.val_test_step_outputs
        if dist.is_available() and dist.is_initialized():
            collected = [None] * dist.get_world_size()
            dist.all_gather_object(collected, outputs)
            outputs = [row for group in collected for row in group]
        predictions, ground_truths = {}, {}
        for predicted, ground_truth in outputs:
            predictions.update(predicted)
            ground_truths.update(ground_truth)
        self.val_test_step_outputs.clear()
        self.evaluator.set_ground_truths(ground_truths)
        results = self.evaluator.evaluate(predictions)
        self.last_validation_results = results
        for metric, groups in results.items():
            for group, value in groups.items():
                self.log(f'val_eval/{metric}_{group}', value, sync_dist=False)



    def _loss(self, data_dict, output_dict):
        loss_dict = self.detector.loss(data_dict, output_dict)
        # reference loss
        loss_dict["reference_loss"] = self.ref_loss(
            output_dict,
            output_dict["pred_aabb_min_max_bounds"],
            output_dict["pred_aabb_scores"],
            data_dict["gt_aabb_min_max_bounds"],
            data_dict["gt_target_obj_id_mask"].permute(dims=(1, 0)),
            data_dict["aabb_count_offsets"],
        )
        if self.hparams.cfg.model.network.use_contrastive_loss:
            # contrastive loss
            loss_dict["contrastive_loss"] = self.contrastive_loss(
                output_dict["aabb_features_inter"],
                output_dict["sentence_features"],
                output_dict["gt_labels"]
            )
        return loss_dict


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.cfg.model.optimizer, params=self.parameters())
        return optimizer


    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)

        # calculate the total loss and log
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=True)
        self.log(f"train_loss/total_loss", total_loss, on_step=True, on_epoch=True)
        return total_loss


    def validation_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        # calculate the total loss and log
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"val_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        self.log(f"val_loss/total_loss", total_loss, on_step=True, on_epoch=False)

        # get predictions and gts
        self.val_test_step_outputs.append((self._parse_pred_results(data_dict, output_dict), self._parse_gt(data_dict)))


    def test_step(self, data_dict, idx):
        output_dict = self(data_dict)
        self.val_test_step_outputs.append(
            (self._parse_pred_results(data_dict, output_dict), self._parse_gt(data_dict))
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
        pred_scores = output_dict["pred_aabb_scores"].reshape(batch_size, lang_chunk_size, -1)

        if self.dataset_name in ("ScanRefer", "Nr3D"):
            pred_aabb_score_masks = torch.zeros_like(pred_scores, dtype=torch.bool)
            pred_aabb_score_masks.scatter_(2, pred_scores.argmax(dim=2, keepdim=True), True)
        elif self.dataset_name == "Multi3DRefer":
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

