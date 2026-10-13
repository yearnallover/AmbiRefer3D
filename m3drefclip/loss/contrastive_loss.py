import lightning.pytorch as pl
import torch
from torch import nn


class SinglePairContrastiveLoss(pl.LightningModule):
    def __init__(self, temperature, split_batch):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(float(temperature)))
        self.split_batch = split_batch

    def forward(self, aabb_features, sentence_features, gt_labels):
        features = torch.einsum('abc,adb->adc', aabb_features, gt_labels).flatten(0, 1)
        counts = gt_labels.count_nonzero(dim=2).flatten()
        valid = counts > 0
        if not valid.any():
            return aabb_features.sum() * 0
        objects = nn.functional.normalize(features[valid] / counts[valid].unsqueeze(1), dim=1)
        sentences = nn.functional.normalize(sentence_features[valid], dim=1)
        logits = self.logit_scale.exp() * objects @ sentences.t()
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (nn.functional.cross_entropy(logits, labels)
                + nn.functional.cross_entropy(logits.t(), labels)) / 2
