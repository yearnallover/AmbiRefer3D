from __future__ import annotations

import hashlib
import os
import random
import sys
from collections import Counter, OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel, BertTokenizer


@contextmanager
def _external_import_context(root: Path, extra_paths=()):
    old_cwd = Path.cwd()
    paths = [str(root), *(str(path) for path in extra_paths)]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def _flatten_texts(data_dict):
    texts = data_dict.get("stage1_query_text", data_dict.get("query_text"))
    if texts is None:
        raise KeyError(
            "External Stage1 backbones require query_text from the dataset."
        )
    return [str(text) for scene_texts in texts for text in scene_texts]


def _stable_seed(*values):
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


class _SceneCache:
    def __init__(self, max_items=8):
        self.max_items = int(max_items)
        self.items = OrderedDict()

    def get(self, key, loader):
        if key in self.items:
            value = self.items.pop(key)
            self.items[key] = value
            return value
        value = loader()
        self.items[key] = value
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return value


class _ExternalStage1Base(nn.Module):
    def __init__(self, proposal_dim=32, common_dim=128):
        super().__init__()
        self.proposal_dim = int(proposal_dim)
        self.common_dim = int(common_dim)

    def loss(self, data_dict, output_dict):
        # The official checkpoints do not expose detector supervision in the

        # still update every trainable module used by the forward graph.
        return {}


class BUTDDETRStage1Backbone(_ExternalStage1Base):
    """Official BUTD-DETR as a differentiable Stage1 provider.

    BUTD-DETR predicts language-conditioned DETR queries. The framework uses
    one language item per scene for this backbone so query slots remain the
    candidate identity across grounding queries.
    """

    def __init__(
        self,
        project_root,
        checkpoint,
        span_checkpoint,
        detector_root,
        proposal_dim=32,
        common_dim=128,
        freeze_point_backbone=False,
        freeze_native_backbone=False,
        cache_scenes=4,
    ):
        super().__init__(proposal_dim=proposal_dim, common_dim=common_dim)
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / "comparison/BUTD-DETR"
        self.detector_root = Path(detector_root).resolve()
        os.environ.setdefault(
            "HF_HOME", str(self.project_root / "comparison/cache/huggingface")
        )
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        with _external_import_context(
            self.root, (self.root / "pointnet2",)
        ):
            from data.model_util_scannet import ScannetDatasetConfig
            from data.scannet_utils import read_label_mapping
            from models.bdetr import BeaUTyDETR
            from src.text_cls import TextClassifier

            state = _torch_load(checkpoint)
            config = state["config"]
            self.backbone = BeaUTyDETR(
                num_class=256,
                num_obj_class=485,
                input_feature_dim=3,
                num_queries=config.num_target,
                num_decoder_layers=config.num_decoder_layers,
                self_position_embedding=config.self_position_embedding,
                contrastive_align_loss=config.use_contrastive_align,
                butd=config.butd or config.butd_gt or config.butd_cls,
                pointnet_ckpt=None,
                self_attend=config.self_attend,
            )
            state_dict = {
                (key[7:] if key.startswith("module.") else key): value
                for key, value in state["model"].items()
            }
            self.backbone.load_state_dict(state_dict, strict=True)

            self.span_model = TextClassifier()
            span_state = _torch_load(span_checkpoint)
            self.span_model.load_state_dict(
                span_state["model_state_dict"], strict=True
            )

            label_map = read_label_mapping(
                str(self.root / "data/meta_data/scannetv2-labels.combined.tsv"),
                label_from="raw_category",
                label_to="id",
            )
            dataset_config = ScannetDatasetConfig(485)
            self.class_name_to_id = {
                name: dataset_config.nyu40id2class[label_id]
                for name, label_id in label_map.items()
                if label_id in dataset_config.nyu40id2class
            }
            self.unknown_class_id = dataset_config.type2class["object"]
            # Synthetic ScanNet edits use the 18-class canonical label, while
            # BUTD-DETR's 485-class inventory names the equivalent class furniture.
            self.class_name_to_id.setdefault(
                "other furniture",
                self.class_name_to_id.get("furniture", self.unknown_class_id),
            )

        self.freeze_native_backbone = bool(freeze_native_backbone)
        _freeze(self.span_model)
        if self.freeze_native_backbone:
            _freeze(self.backbone)
        else:
            _freeze(self.backbone.text_encoder)
        if freeze_point_backbone and not self.freeze_native_backbone:
            _freeze(self.backbone.backbone_net)
        self.visual_projection = nn.Linear(64, self.proposal_dim)
        self.object_projection = nn.Linear(64, self.common_dim)
        self.sentence_projection = nn.Linear(64, self.common_dim)
        self.logit_scale = nn.Parameter(torch.ones(()))
        self.logit_bias = nn.Parameter(torch.zeros(()))
        self.num_proposals = int(self.backbone.num_queries)
        self.scene_cache = _SceneCache(cache_scenes)

    def train(self, mode=True):
        super().train(mode)
        self.span_model.eval()
        if self.freeze_native_backbone:
            self.backbone.eval()
        else:
            self.backbone.text_encoder.eval()
        return self

    def _load_scene(self, scene_id):
        candidates = [self.detector_root / f"{scene_id}.npy"]
        search_root = (
            self.detector_root.parent
            if self.detector_root.name.startswith("group_free_pred_bboxes_")
            else self.detector_root
        )
        candidates.extend(
            search_root / f"group_free_pred_bboxes_{split}" / f"{scene_id}.npy"
            for split in ("train", "val", "test")
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"BUTD-DETR detector input is missing for {scene_id} under "
                f"{self.detector_root}. "
                "Run scripts/prepare_external_stage1.sh first."
            )

        def loader():
            detected = np.load(path, allow_pickle=True).item()
            points = np.asarray(detected["pc"], dtype=np.float32)
            boxes = np.asarray(detected["box"], dtype=np.float32)
            class_ids = np.asarray(
                [
                    self.class_name_to_id.get(str(name), self.unknown_class_id)
                    for name in detected["class"]
                ],
                dtype=np.int64,
            )
            count = len(boxes)
            padded_boxes = np.zeros((132, 6), dtype=np.float32)
            padded_boxes[:count] = np.concatenate(
                ((boxes[:, :3] + boxes[:, 3:]) * 0.5, boxes[:, 3:] - boxes[:, :3]),
                axis=1,
            )
            padded_classes = np.zeros(132, dtype=np.int64)
            padded_classes[:count] = class_ids
            padded_mask = np.zeros(132, dtype=np.bool_)
            padded_mask[:count] = True
            return points, padded_boxes, padded_classes, padded_mask

        return self.scene_cache.get(str(path), loader)

    def encode_scene(self, data_dict):

        return {"scene_id": tuple(data_dict["scene_id"])}

    def _span_positive_map(self, texts, token_count, device):
        tokenized = self.span_model.tokenizer.batch_encode_plus(
            texts, padding="longest", return_tensors="pt"
        ).to(device)
        encoded = self.span_model.text_encoder(**tokenized)
        logits = self.span_model.text_projector(
            encoded.last_hidden_state
        ).squeeze(-1)
        positive = (logits > 0).float()
        positive *= tokenized.attention_mask.float()
        no_positive = positive.sum(dim=1) == 0
        if no_positive.any():
            fallback = logits.masked_fill(
                ~tokenized.attention_mask.bool(), -torch.inf
            ).argmax(dim=1)
            positive[no_positive] = 0
            positive[no_positive, fallback[no_positive]] = 1
        positive = positive[:, :token_count]
        return positive / positive.sum(dim=1, keepdim=True).clamp_min(1.0)

    def ground_queries(self, data_dict, scene_output):
        batch_size, chunk_size = data_dict["ann_id"].shape
        if chunk_size != 1:
            raise ValueError(
                "BUTD-DETR integration requires data.chunk_size=1 because its "
                "proposal boxes are language-conditioned."
            )
        texts = [
            " ".join(text.replace(",", " ,").split()) + " . not mentioned"
            for text in _flatten_texts(data_dict)
        ]
        device = data_dict["scene_center_xyz"].device
        scenes = [self._load_scene(scene_id) for scene_id in data_dict["scene_id"]]
        inputs = {
            "point_clouds": torch.stack(
                [torch.from_numpy(scene[0]) for scene in scenes]
            ).to(device),
            "text": texts,
            "det_boxes": torch.stack(
                [torch.from_numpy(scene[1]) for scene in scenes]
            ).to(device),
            "det_class_ids": torch.stack(
                [torch.from_numpy(scene[2]) for scene in scenes]
            ).to(device),
            "det_bbox_label_mask": torch.stack(
                [torch.from_numpy(scene[3]) for scene in scenes]
            ).to(device),
        }
        end_points = self.backbone(inputs)
        query_features = end_points["last_proj_queries"]
        token_features = end_points["proj_tokens"]
        positive_map = self._span_positive_map(
            texts, token_features.shape[1], device
        )
        token_scores = torch.matmul(
            query_features, token_features.transpose(-1, -2)
        )
        token_scores = (token_scores / 0.07).softmax(dim=-1)
        probabilities = (
            token_scores * positive_map.unsqueeze(1)
        ).sum(dim=-1)
        logits = torch.logit(probabilities.clamp(1e-5, 1.0 - 1e-5))
        logits = logits * self.logit_scale + self.logit_bias

        centers = end_points["last_center"]
        sizes = end_points["last_pred_size"]
        finite_candidates = (
            torch.isfinite(centers).all(dim=-1)
            & torch.isfinite(sizes).all(dim=-1)
            & torch.isfinite(query_features).all(dim=-1)
            & torch.isfinite(logits)
        )
        if not finite_candidates.any(dim=1).all():
            invalid_scenes = [
                str(scene_id)
                for scene_id, has_valid in zip(
                    data_dict["scene_id"], finite_candidates.any(dim=1).tolist()
                )
                if not has_valid
            ]
            raise FloatingPointError(
                "BUTD-DETR produced no finite candidates for scenes: "
                + ", ".join(invalid_scenes)
            )
        centers = torch.nan_to_num(
            centers, nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp(-1e4, 1e4)
        sizes = torch.nan_to_num(
            sizes, nan=1e-4, posinf=1e4, neginf=1e-4
        ).clamp(1e-4, 1e4)
        query_features = torch.nan_to_num(
            query_features, nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp(-1e4, 1e4)
        token_features = torch.nan_to_num(
            token_features, nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp(-1e4, 1e4)
        logits = torch.nan_to_num(
            logits, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)
        logits = logits.masked_fill(~finite_candidates, -20.0)
        bounds = torch.stack((centers - sizes * 0.5, centers + sizes * 0.5), dim=2)
        bounds = bounds - data_dict["scene_center_xyz"][:, None, None, :]
        proposal_features = self.visual_projection(query_features)
        common_features = self.object_projection(query_features)
        sentence64 = (
            token_features * positive_map.unsqueeze(-1)
        ).sum(dim=1)
        sentence_features = self.sentence_projection(sentence64)
        proposal_mask = finite_candidates
        return {
            "pred_aabb_min_max_bounds": bounds,
            "proposal_masks_dense": proposal_mask,
            "aabb_features": proposal_features,
            "aabb_features_3d": proposal_features,
            "aabb_features_inter": common_features,
            "aabb_features_cross_modal": common_features,
            "sentence_features": sentence_features,
            "pred_aabb_scores": logits,
            "external_stage1_native": end_points,
        }


class VistaStage1Backbone(_ExternalStage1Base):
    """Official 3D-VisTA object encoder and grounding stack as Stage1."""

    max_objects = 79
    padded_objects = 80
    points_per_object = 1024

    def __init__(
        self,
        project_root,
        checkpoint,
        bert_root,
        scene_root,
        proposal_dim=32,
        common_dim=128,
        freeze_point_features=True,
        cache_scenes=8,
    ):
        super().__init__(proposal_dim=proposal_dim, common_dim=common_dim)
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / "comparison/3D-VisTA"
        self.scene_root = Path(scene_root).resolve()
        with _external_import_context(self.root):
            from model.vision.grounding_head import GroundHeadV1
            from model.vision.point_encoder import PointTokenizeEncoder
            from model.vision.unified_encoder import UnifiedSpatialCrossEncoderV2

            self.tokenizer = BertTokenizer.from_pretrained(
                str(bert_root), do_lower_case=True, local_files_only=True
            )
            bert_config = BertConfig(
                hidden_size=768,
                num_hidden_layers=4,
                num_attention_heads=12,
                type_vocab_size=2,
            )
            self.language_encoder = BertModel.from_pretrained(
                str(bert_root), config=bert_config, local_files_only=True
            )
            self.point_encoder = PointTokenizeEncoder(
                backbone="pointnet++",
                hidden_size=768,
                path=None,
                freeze_feature=bool(freeze_point_features),
                num_attention_heads=12,
                spatial_dim=5,
                num_layers=4,
                dim_loc=6,
                pairwise_rel_type="center",
                mixup_strategy="linear_decay",
                mixup_stage1=0.2,
                mixup_stage2=0.4,
            )
            self.unified_encoder = UnifiedSpatialCrossEncoderV2(
                hidden_size=768, num_attention_heads=12, num_layers=4, dim_loc=6
            )
            self.ground_head = GroundHeadV1(
                input_size=768, hidden_size=768, sem_cls_size=607, dropout=0.3
            )

        state = _torch_load(checkpoint)
        self.language_encoder.load_state_dict(state["lang_encoder"], strict=True)
        self.point_encoder.load_state_dict(state["point_encoder"], strict=True)
        self.unified_encoder.load_state_dict(state["unified_encoder"], strict=True)
        self.ground_head.load_state_dict(state["ground_head"], strict=True)
        if freeze_point_features:
            # The official flag is applied before the spatial layers are
            # constructed, so those layers accidentally remain trainable.
            # Freeze the complete pretrained object tokenizer for stable

            _freeze(self.point_encoder)
        self.visual_projection = nn.Linear(768, self.proposal_dim)
        self.object_projection = nn.Linear(768, self.common_dim)
        self.sentence_projection = nn.Linear(768, self.common_dim)
        self.logit_scale = nn.Parameter(torch.ones(()))
        self.logit_bias = nn.Parameter(torch.zeros(()))
        self.num_proposals = self.padded_objects
        self.scene_cache = _SceneCache(cache_scenes)

    @staticmethod
    def _dominant_semantics(instance_ids, semantic_labels, object_ids):
        return {
            int(object_id): int(
                Counter(semantic_labels[instance_ids == object_id].tolist())
                .most_common(1)[0][0]
            )
            for object_id in object_ids
        }

    def _scene_path(self, scene_id):
        for split in ("train", "val", "test"):
            path = self.scene_root / split / f"{scene_id}.pth"
            if path.exists():
                return path
        raise FileNotFoundError(f"3D-VisTA scene not found: {scene_id}")

    def _load_scene(self, scene_id):
        path = self._scene_path(scene_id)

        def loader():
            scene = _torch_load(path)
            xyz = np.asarray(scene["xyz"], dtype=np.float32)
            rgb = np.asarray(scene["rgb"], dtype=np.float32) / 127.5 - 1.0
            instance_ids = np.asarray(scene["instance_ids"], dtype=np.int64)
            semantic_labels = np.asarray(scene["sem_labels"], dtype=np.int64)
            object_ids = [
                int(value)
                for value in np.asarray(scene["aabb_obj_ids"]).tolist()
            ][: self.max_objects]
            self._dominant_semantics(
                instance_ids, semantic_labels, object_ids
            )
            features = np.ones(
                (self.padded_objects, self.points_per_object, 6),
                dtype=np.float32,
            )
            locations = np.zeros((self.padded_objects, 6), dtype=np.float32)
            boxes = np.zeros((self.padded_objects, 2, 3), dtype=np.float32)
            object_id_array = np.full(self.padded_objects, -1, dtype=np.int64)
            for index, object_id in enumerate(object_ids):
                mask = instance_ids == object_id
                object_xyz = xyz[mask]
                object_rgb = rgb[mask]
                center = object_xyz.mean(axis=0)
                minimum = object_xyz.min(axis=0)
                maximum = object_xyz.max(axis=0)
                locations[index] = np.concatenate((center, maximum - minimum))
                boxes[index] = np.stack((minimum, maximum))
                object_id_array[index] = object_id
                rng = np.random.RandomState(
                    _stable_seed(scene_id, object_id, "points")
                )
                sampled = rng.choice(
                    len(object_xyz),
                    size=self.points_per_object,
                    replace=len(object_xyz) < self.points_per_object,
                )
                sampled_xyz = object_xyz[sampled].copy()
                sampled_xyz -= sampled_xyz.mean(axis=0, keepdims=True)
                radius = np.sqrt(np.square(sampled_xyz).sum(axis=1)).max()
                if radius >= 1e-6:
                    sampled_xyz /= radius
                features[index] = np.concatenate(
                    (sampled_xyz, object_rgb[sampled]), axis=1
                )
            mask = np.arange(self.padded_objects) < len(object_ids)
            return features, locations, boxes, mask, object_id_array

        return self.scene_cache.get(str(path), loader)

    def encode_scene(self, data_dict):
        device = data_dict["scene_center_xyz"].device
        scenes = [self._load_scene(scene_id) for scene_id in data_dict["scene_id"]]
        point_inputs = {
            "obj_fts": torch.from_numpy(
                np.stack([scene[0] for scene in scenes])
            ).to(device),
            "obj_locs": torch.from_numpy(
                np.stack([scene[1] for scene in scenes])
            ).to(device),
            "obj_masks": torch.from_numpy(
                np.stack([scene[3] for scene in scenes])
            ).to(device),
        }
        point_inputs["obj_sem_masks"] = torch.ones_like(
            point_inputs["obj_masks"]
        )
        object_features, object_features_pre, _ = self.point_encoder(
            point_inputs["obj_fts"],
            point_inputs["obj_locs"],
            point_inputs["obj_masks"],
            point_inputs["obj_sem_masks"],
            None,
            1,
            1,
        )
        bounds = torch.from_numpy(
            np.stack([scene[2] for scene in scenes])
        ).to(device)
        bounds = bounds - data_dict["scene_center_xyz"][:, None, None, :]
        object_ids = torch.from_numpy(
            np.stack([scene[4] for scene in scenes])
        ).to(device)
        proposal_features = self.visual_projection(object_features)
        return {
            "pred_aabb_min_max_bounds": bounds,
            "proposal_masks_dense": point_inputs["obj_masks"],
            "proposal_gt_obj_ids_dense": object_ids,
            "aabb_features": proposal_features,
            "aabb_features_3d": proposal_features,
            "vista_object_features": object_features,
            "vista_object_features_pre": object_features_pre,
            "vista_object_locations": point_inputs["obj_locs"],
        }

    def ground_queries(self, data_dict, scene_output):
        batch_size, chunk_size = data_dict["ann_id"].shape
        texts = _flatten_texts(data_dict)
        device = scene_output["aabb_features_3d"].device
        encoded = self.tokenizer(
            texts,
            max_length=50,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        text_ids = encoded["input_ids"].to(device)
        text_masks = encoded["attention_mask"].to(device)
        object_features = scene_output["vista_object_features"].repeat_interleave(
            chunk_size, dim=0
        )
        object_features_pre = scene_output[
            "vista_object_features_pre"
        ].repeat_interleave(chunk_size, dim=0)
        object_locations = scene_output[
            "vista_object_locations"
        ].repeat_interleave(chunk_size, dim=0)
        object_masks = scene_output["proposal_masks_dense"].repeat_interleave(
            chunk_size, dim=0
        )
        language_features = self.language_encoder(
            input_ids=text_ids, attention_mask=text_masks
        ).last_hidden_state
        language_fused, object_fused = self.unified_encoder(
            language_features,
            text_masks,
            object_features,
            object_locations,
            object_masks,
        )
        logits = self.ground_head.og3d_head(object_fused).squeeze(2)
        logits = logits * self.logit_scale + self.logit_bias
        logits = logits.masked_fill(~object_masks, torch.finfo(logits.dtype).min)
        output = dict(scene_output)
        output["aabb_features_inter"] = self.object_projection(
            scene_output["vista_object_features"]
        )
        output["aabb_features_cross_modal"] = self.object_projection(object_fused)
        output["sentence_features"] = self.sentence_projection(
            language_fused[:, 0]
        )
        output["pred_aabb_scores"] = logits
        output["external_stage1_native"] = {
            "language_fused": language_fused,
            "object_fused": object_fused,
            "object_features_pre": object_features_pre,
        }
        return output


class TSP3DStage1Backbone(_ExternalStage1Base):
    """CVPR 2025 TSP3D adapted to stable multi-round Stage1 candidates.

    TSP3D's native predictions are text-guided sparse voxels rather than fixed
    DETR slots. The first query therefore selects the geometric top-K anchors.
    Additional queries rerun the official text-guided head, but rescore
    those fixed anchors so candidate identity is preserved across rounds.
    """

    def __init__(
        self,
        project_root,
        checkpoint,
        detector_root,
        roberta_root,
        proposal_dim=32,
        common_dim=128,
        max_proposals=256,
        voxel_size=0.01,
        nms_iou_threshold=0.5,
        freeze_sparse_backbone=False,
        cache_scenes=4,
    ):
        super().__init__(proposal_dim=proposal_dim, common_dim=common_dim)
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / "comparison/TSP3D"
        self.detector_root = Path(detector_root).resolve()
        self.max_proposals = int(max_proposals)
        self.nms_iou_threshold = float(nms_iou_threshold)

        roberta_root = Path(roberta_root)
        if not (roberta_root.resolve() / "config.json").is_file():
            raise FileNotFoundError(
                f"TSP3D RoBERTa files are missing under {roberta_root}. "
                "Run scripts/prepare_tsp3d_stage1.sh first."
            )
        if roberta_root.name != "roberta-base":
            raise ValueError(
                "TSP3D roberta_root must end in 'roberta-base' because the "
                "official model appends that directory name."
            )
        model_data_root = roberta_root.absolute().parent

        with _external_import_context(self.root):
            import MinkowskiEngine as ME
            from models.bdetr import BeaUTyDETR

            self.ME = ME
            self.backbone = BeaUTyDETR(
                num_class=256,
                num_obj_class=485,
                input_feature_dim=3,
                num_queries=self.max_proposals,
                num_decoder_layers=0,
                contrastive_align_loss=False,
                d_model=128,
                butd=False,
                pointnet_ckpt=None,
                data_path=str(model_data_root) + os.sep,
                self_attend=True,
                voxel_size=float(voxel_size),
            )

        state = _torch_load(checkpoint)
        state_dict = state.get("model", state.get("state_dict", state))
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        incompatible = self.backbone.load_state_dict(state_dict, strict=False)
        # Transformers 4.29 registers this deterministic buffer while the
        # official checkpoint was saved with a version that did not.
        allowed_missing = {"text_encoder.embeddings.position_ids"}
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "TSP3D checkpoint does not match the official architecture: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        _freeze(self.backbone.text_encoder)
        if freeze_sparse_backbone:
            _freeze(self.backbone.vision_backbone)

        self.visual_projection = nn.Linear(128, self.proposal_dim)
        self.object_projection = nn.Linear(128, self.common_dim)
        self.sentence_projection = nn.Linear(128, self.common_dim)
        self.logit_scale = nn.Parameter(torch.ones(()))
        self.logit_bias = nn.Parameter(torch.zeros(()))
        self.num_proposals = self.max_proposals
        self.scene_cache = _SceneCache(cache_scenes)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.text_encoder.eval()
        return self

    def _load_scene(self, scene_id):
        candidates = [self.detector_root / f"{scene_id}.npy"]
        search_root = (
            self.detector_root.parent
            if self.detector_root.name.startswith("group_free_pred_bboxes_")
            else self.detector_root
        )
        candidates.extend(
            search_root / f"group_free_pred_bboxes_{split}" / f"{scene_id}.npy"
            for split in ("train", "val", "test")
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"TSP3D point-cloud input is missing for {scene_id} under "
                f"{self.detector_root}. Run scripts/prepare_tsp3d_stage1.sh first."
            )

        def loader():
            detected = np.load(path, allow_pickle=True).item()
            points = np.asarray(detected["pc"], dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 6:
                raise ValueError(
                    f"Expected {path}['pc'] to have shape [N, 6], got {points.shape}."
                )
            return points

        return self.scene_cache.get(str(path), loader)

    def encode_scene(self, data_dict):
        device = data_dict["scene_center_xyz"].device
        points = [
            torch.from_numpy(self._load_scene(scene_id)).to(device)
            for scene_id in data_dict["scene_id"]
        ]
        coordinates, features = self.ME.utils.batch_sparse_collate(
            [
                (point[:, :3] / self.backbone.voxel_size, point)
                for point in points
            ],
            device=device,
        )
        sparse = self.ME.SparseTensor(
            coordinates=coordinates, features=features
        )
        return {
            "tsp3d_vision_features": self.backbone.vision_backbone(sparse),
        }

    def _run_raw_query(self, data_dict, scene_output):
        texts = _flatten_texts(data_dict)
        device = data_dict["scene_center_xyz"].device
        tokenized = self.backbone.tokenizer.batch_encode_plus(
            texts, padding="longest", return_tensors="pt"
        ).to(device)
        encoded = self.backbone.text_encoder(**tokenized)
        text_features = self.backbone.text_projector(encoded.last_hidden_state)
        text_padding_mask = tokenized.attention_mask.ne(1).bool()
        raw, _ = self.backbone.head.forward_test(
            scene_output["tsp3d_vision_features"],
            text_features,
            text_padding_mask,
            return_raw=True,
        )
        return raw

    @staticmethod
    def _center_size_to_bounds(boxes):
        half_size = boxes[..., 3:6].clamp_min(1e-4) * 0.5
        return torch.stack((boxes[..., :3] - half_size, boxes[..., :3] + half_size), dim=-2)

    @staticmethod
    def _axis_aligned_nms(boxes, scores, threshold, max_keep):
        if boxes.numel() == 0:
            return scores.new_zeros((0,), dtype=torch.long)
        pre_nms = min(scores.numel(), max(max_keep * 8, max_keep))
        order = scores.topk(pre_nms, sorted=True).indices
        bounds = TSP3DStage1Backbone._center_size_to_bounds(boxes[order])
        volumes = (bounds[:, 1] - bounds[:, 0]).clamp_min(0).prod(dim=-1)
        keep = []
        while order.numel() and len(keep) < max_keep:
            keep.append(order[0])
            if order.numel() == 1:
                break
            first = bounds[0]
            rest = bounds[1:]
            intersection = (
                torch.minimum(first[1], rest[:, 1])
                - torch.maximum(first[0], rest[:, 0])
            ).clamp_min(0).prod(dim=-1)
            union = volumes[0] + volumes[1:] - intersection
            iou = intersection / union.clamp_min(1e-6)
            remaining = iou <= threshold
            order = order[1:][remaining]
            bounds = bounds[1:][remaining]
            volumes = volumes[1:][remaining]
        return torch.stack(keep) if keep else scores.new_zeros((0,), dtype=torch.long)

    def _select_anchors(self, raw, scene_centers):
        batch_size = len(raw["boxes"])
        device = scene_centers.device
        dtype = raw["boxes"][0].dtype
        boxes = torch.zeros(batch_size, self.max_proposals, 6, device=device, dtype=dtype)
        logits = torch.full(
            (batch_size, self.max_proposals), -20.0, device=device, dtype=dtype
        )
        base_features = torch.zeros(
            batch_size, self.max_proposals, 128, device=device, dtype=dtype
        )
        fused_features = torch.zeros_like(base_features)
        proposal_mask = torch.zeros(
            batch_size, self.max_proposals, device=device, dtype=torch.bool
        )

        for batch_idx in range(batch_size):
            scene_boxes = raw["boxes"][batch_idx]
            scene_logits = raw["logits"][batch_idx].squeeze(-1)
            scene_base = raw["base_features"][batch_idx]
            scene_fused = raw["fused_features"][batch_idx]
            finite = (
                torch.isfinite(scene_boxes).all(dim=-1)
                & torch.isfinite(scene_logits)
                & torch.isfinite(scene_base).all(dim=-1)
                & torch.isfinite(scene_fused).all(dim=-1)
                & scene_boxes[:, 3:6].gt(0).all(dim=-1)
            )
            valid_indices = finite.nonzero(as_tuple=False).flatten()
            if not valid_indices.numel():
                continue
            selected_local = self._axis_aligned_nms(
                scene_boxes[valid_indices],
                scene_logits[valid_indices],
                self.nms_iou_threshold,
                self.max_proposals,
            )
            selected = valid_indices[selected_local]
            count = selected.numel()
            boxes[batch_idx, :count] = scene_boxes[selected]
            logits[batch_idx, :count] = scene_logits[selected]
            base_features[batch_idx, :count] = scene_base[selected]
            fused_features[batch_idx, :count] = scene_fused[selected]
            proposal_mask[batch_idx, :count] = True

        bounds = self._center_size_to_bounds(boxes)
        bounds = bounds - scene_centers[:, None, None, :]
        return bounds, logits, base_features, fused_features, proposal_mask

    @staticmethod
    def _dense_minkowski_linear(module, features):
        kernel = module.kernel
        if kernel.dim() == 3:
            if kernel.shape[0] != 1:
                raise ValueError("Only pointwise Minkowski convolutions can be densified.")
            kernel = kernel[0]
        output = features.matmul(kernel)
        if module.bias is not None:
            output = output + module.bias
        return output

    def _rescore_anchors(self, base_features, proposal_mask, query_features):
        flat_mask = proposal_mask.flatten()
        flat_base = base_features.flatten(0, 1)
        if not flat_mask.any():
            return torch.zeros_like(base_features), flat_base.new_full(
                proposal_mask.shape, -20.0
            )
        repeated_query = query_features[:, None, :].expand_as(base_features).flatten(0, 1)
        valid_input = torch.cat((flat_base[flat_mask], repeated_query[flat_mask]), dim=-1)
        valid_fused = self._dense_minkowski_linear(
            self.backbone.head.fuse.conv, valid_input
        )
        valid_fused = self.backbone.head.fuse.norm.bn(valid_fused)
        valid_fused = F.relu(valid_fused)
        valid_logits = self._dense_minkowski_linear(
            self.backbone.head.cls_conv, valid_fused
        ).squeeze(-1)

        fused = torch.zeros_like(flat_base)
        logits = flat_base.new_full((flat_base.shape[0],), -20.0)
        fused[flat_mask] = valid_fused
        logits[flat_mask] = valid_logits
        return fused.reshape_as(base_features), logits.reshape(proposal_mask.shape)

    def ground_queries(self, data_dict, scene_output):
        batch_size, chunk_size = data_dict["ann_id"].shape
        if chunk_size != 1:
            raise ValueError(
                "TSP3D integration requires data.chunk_size=1 because its first "
                "query defines the stable proposal candidates."
            )
        raw = self._run_raw_query(data_dict, scene_output)
        query_features = raw["text_features"][:, 0]

        if "tsp3d_anchor_base_features" in scene_output:
            bounds = scene_output["pred_aabb_min_max_bounds"]
            base_features = scene_output["tsp3d_anchor_base_features"]
            proposal_mask = scene_output["proposal_masks_dense"]
            fused_features, logits = self._rescore_anchors(
                base_features, proposal_mask, query_features
            )
        else:
            bounds, logits, base_features, fused_features, proposal_mask = (
                self._select_anchors(raw, data_dict["scene_center_xyz"])
            )

        logits = logits * self.logit_scale + self.logit_bias
        logits = logits.masked_fill(~proposal_mask, -20.0)
        proposal_features = self.visual_projection(base_features)
        common_features = self.object_projection(base_features)
        cross_modal_features = self.object_projection(fused_features)
        output = dict(scene_output)
        output.update({
            "pred_aabb_min_max_bounds": bounds,
            "proposal_masks_dense": proposal_mask,
            "aabb_features": proposal_features,
            "aabb_features_3d": proposal_features,
            "aabb_features_inter": common_features,
            "aabb_features_cross_modal": cross_modal_features,
            "sentence_features": self.sentence_projection(query_features),
            "pred_aabb_scores": logits,
            "tsp3d_anchor_base_features": base_features,
            "external_stage1_native": raw,
        })
        return output
