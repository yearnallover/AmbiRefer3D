import lightning.pytorch as pl
import torch.nn as nn
import torch
from m3drefclip.loss.sample_weighting import weighted_mean


class SinglePairContrastiveLoss(pl.LightningModule):
    def __init__(self, temperature, split_batch):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(temperature, device=self.device, dtype=torch.float32))
        self.split_batch = split_batch

    def forward(
        self, aabb_features, sentence_features, gt_labels,
        proposal_masks=None, multiple_gt=False, sample_weights=None,
    ):
        aabb_features_filtered = torch.einsum("abc,adb->adc", aabb_features, gt_labels).flatten(0, 1)

        # for multiple gt labels, it takes the average of their features as the final feature
        gt_count = torch.count_nonzero(gt_labels, dim=2).flatten(0, 1)
        gt_mask = gt_count != 0

        if not gt_mask.any():
            return aabb_features.sum() * 0.0

        aabb_features_filtered = aabb_features_filtered[gt_mask] / gt_count[gt_mask].unsqueeze(-1)

        # normalized features
        normalized_aabb_features = nn.functional.normalize(aabb_features_filtered, dim=1)
        normalized_sentence_features = nn.functional.normalize(sentence_features[gt_mask], dim=1)

        logit_scale = self.logit_scale.exp()
        logits_1 = logit_scale * normalized_aabb_features @ normalized_sentence_features.t()
        logits_2 = logit_scale * normalized_sentence_features @ normalized_aabb_features.t()

        labels = torch.arange(normalized_aabb_features.shape[0], device=aabb_features.device, dtype=torch.long)
        selected_weights = None
        if sample_weights is not None:
            selected_weights = sample_weights.reshape(-1)[gt_mask]
        loss_a = weighted_mean(
            nn.functional.cross_entropy(logits_1, labels, reduction="none"),
            selected_weights,
        )
        loss_b = weighted_mean(
            nn.functional.cross_entropy(logits_2, labels, reduction="none"),
            selected_weights,
        )
        return (loss_a + loss_b) / 2


class MultiPositiveContrastiveLoss(pl.LightningModule):
    def __init__(self, temperature, direction="bidirectional", split_batch=False, use_all_batch_negatives=True):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        self.direction = direction
        self.split_batch = split_batch
        self.use_all_batch_negatives = use_all_batch_negatives

    def forward(
        self, aabb_features, sentence_features, gt_labels,
        proposal_masks=None, sample_weights=None,
    ):
        batch_size, chunk_size, num_proposals = gt_labels.shape
        object_features = nn.functional.normalize(aabb_features.reshape(batch_size * num_proposals, -1), dim=1)
        text_features = nn.functional.normalize(sentence_features, dim=1)

        positive_mask = torch.block_diag(*gt_labels.bool().unbind(dim=0))

        if proposal_masks is None:
            valid_object_mask = torch.ones(batch_size, num_proposals, dtype=torch.bool, device=aabb_features.device)
        else:
            valid_object_mask = proposal_masks.bool()

        valid_object_mask = valid_object_mask.reshape(batch_size * num_proposals)
        valid_text_mask = positive_mask.any(dim=1)
        text_weights = None
        object_weights = None
        if sample_weights is not None:
            text_weights = sample_weights.reshape(-1).to(
                device=aabb_features.device, dtype=aabb_features.dtype
            )
            if text_weights.shape[0] != batch_size * chunk_size:
                raise ValueError("sample_weights must have [B,C] shape")
            positive_count = positive_mask.sum(dim=0)
            object_weights = (
                positive_mask.to(text_weights.dtype).t() @ text_weights
            ) / positive_count.clamp_min(1).to(text_weights.dtype)
        same_scene_object_mask = torch.block_diag(
            *[
                valid_object_mask.reshape(batch_size, num_proposals)[i].unsqueeze(0).expand(chunk_size, -1)
                for i in range(batch_size)
            ]
        )
        same_scene_text_mask = torch.block_diag(
            *[
                valid_text_mask.reshape(batch_size, chunk_size)[i].unsqueeze(0).expand(num_proposals, -1)
                for i in range(batch_size)
            ]
        )
        losses = []

        if self.direction in ("text_to_object", "bidirectional"):
            logits = self.logit_scale.exp() * text_features @ object_features.t()
            if self.use_all_batch_negatives:
                valid_logits_mask = valid_object_mask.unsqueeze(0).expand_as(logits)
            else:
                valid_logits_mask = same_scene_object_mask
            losses.append(self._multi_positive_nce(
                logits, positive_mask, valid_logits_mask, valid_text_mask,
                row_weights=text_weights,
            ))

        if self.direction in ("object_to_text", "bidirectional"):
            object_positive_mask = positive_mask.t()
            valid_object_rows = valid_object_mask & object_positive_mask.any(dim=1)
            logits = self.logit_scale.exp() * object_features @ text_features.t()
            if self.use_all_batch_negatives:
                valid_logits_mask = valid_text_mask.unsqueeze(0).expand_as(logits)
            else:
                valid_logits_mask = same_scene_text_mask
            losses.append(self._multi_positive_nce(
                logits, object_positive_mask, valid_logits_mask,
                valid_object_rows, row_weights=object_weights,
            ))

        if not losses:
            raise ValueError(f"Unsupported contrastive direction: {self.direction}")
        return torch.stack(losses).mean()

    @staticmethod
    def _multi_positive_nce(
        logits, positive_mask, valid_logits_mask, valid_row_mask,
        row_weights=None,
    ):
        valid_positive_mask = positive_mask & valid_logits_mask
        valid_row_mask = valid_row_mask & valid_positive_mask.any(dim=1)
        if not valid_row_mask.any():
            return logits.sum() * 0.0

        selected_weights = row_weights[valid_row_mask] if row_weights is not None else None
        logits = logits[valid_row_mask]
        valid_logits_mask = valid_logits_mask[valid_row_mask]
        valid_positive_mask = valid_positive_mask[valid_row_mask]

        min_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~valid_logits_mask, min_value)
        positive_logits = logits.masked_fill(~valid_positive_mask, min_value)
        per_row_loss = -(
            torch.logsumexp(positive_logits, dim=1)
            - torch.logsumexp(logits, dim=1)
        )
        return weighted_mean(per_row_loss, selected_weights)
