from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class QueryEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_text: int) -> None:
        super().__init__()
        self.wemb = nn.Embedding(vocab_size, d_text, padding_idx=0)
        self.rnn = nn.GRU(d_text, d_text, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d_text, d_text)

    def forward(self, query_ids: torch.Tensor) -> torch.Tensor:
        x = self.wemb(query_ids)
        lengths = query_ids.ne(0).sum(dim=1).clamp_min(1)
        packed = pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        last = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return self.proj(last)  # [B,d_text]

class ImageCandidateEncoder(nn.Module):
    """
    imgfeat [B,C,D] -> cand_feat [B,C,d_cand]
    Optionally fuse bbox geom.
    """
    def __init__(
        self,
        imgfeat_dim: int,
        d_cand: int,
        use_bbox_geom: bool = False,
        use_relative_geom: bool = False,
        use_semantic_label: bool = False,
        num_semantic_classes: int = 256,
        use_stage1_features: bool = False,
        stage1_feat_dim: int = 128,
        use_stage1_quality: bool = False,
    ) -> None:
        super().__init__()
        self.use_bbox_geom = use_bbox_geom
        self.use_relative_geom = use_relative_geom
        self.use_semantic_label = use_semantic_label
        self.use_stage1_features = use_stage1_features
        self.use_stage1_quality = use_stage1_quality
        self.img_mlp = nn.Sequential(
            nn.Linear(imgfeat_dim, d_cand),
            nn.ReLU(),
            nn.Linear(d_cand, d_cand),
        )
        if use_bbox_geom:
            self.bbox_mlp = nn.Sequential(
                nn.Linear(18, d_cand),
                nn.ReLU(),
                nn.Linear(d_cand, d_cand),
            )
        if use_relative_geom:
            self.rel_mlp = nn.Sequential(
                nn.Linear(11, d_cand),
                nn.ReLU(),
                nn.Linear(d_cand, d_cand),
            )
        if use_semantic_label:
            self.semantic_embedding = nn.Embedding(num_semantic_classes + 1, d_cand, padding_idx=0)
        if use_stage1_features:
            self.stage1_mlp = nn.Sequential(
                nn.Linear(stage1_feat_dim, d_cand),
                nn.ReLU(),
                nn.Linear(d_cand, d_cand),
            )
        if use_stage1_quality:
            self.quality_mlp = nn.Sequential(
                nn.Linear(3, d_cand),
                nn.ReLU(),
                nn.Linear(d_cand, d_cand),
            )

        branch_count = 1 + sum((
            use_bbox_geom,
            use_relative_geom,
            use_semantic_label,
            use_stage1_features,
            use_stage1_quality,
        ))
        self.fuse = nn.Identity() if branch_count == 1 else nn.Sequential(
            nn.Linear(d_cand * branch_count, d_cand),
            nn.ReLU(),
            nn.Linear(d_cand, d_cand),
        )

    @staticmethod
    def _bbox_features(cand_bboxes: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        center = cand_bboxes[..., :3]
        size = cand_bboxes[..., 3:6].abs().clamp_min(1e-4)
        mask = cand_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        center_mean = (center * mask).sum(dim=1, keepdim=True) / denom
        center_var = ((center - center_mean).square() * mask).sum(dim=1, keepdim=True) / denom
        # Clamp before sqrt: sqrt(0) has an infinite derivative even if clamped later.
        center_std = center_var.clamp_min(1e-6).sqrt()
        mean_size = (size * mask).sum(dim=1, keepdim=True) / denom

        center_norm = (center - center_mean) / center_std
        log_size = torch.log(size)
        size_rel = torch.log(size / mean_size.clamp_min(1e-4))
        log_volume = log_size.sum(dim=-1, keepdim=True)
        log_aspect = torch.stack((
            log_size[..., 0] - log_size[..., 1],
            log_size[..., 1] - log_size[..., 2],
            log_size[..., 0] - log_size[..., 2],
        ), dim=-1)
        bottom = center[..., 2:3] - size[..., 2:3] * 0.5
        top = center[..., 2:3] + size[..., 2:3] * 0.5
        return torch.cat(
            (center, center_norm, log_size, size_rel, log_volume, log_aspect, bottom, top), dim=-1
        ) * mask

    def _relative_features(
        self, cand_bboxes: torch.Tensor, cand_mask: torch.Tensor
    ) -> torch.Tensor:
        center = cand_bboxes[..., :3]
        size = cand_bboxes[..., 3:6].abs().clamp_min(1e-4)
        delta = center.unsqueeze(1) - center.unsqueeze(2)  # j - i
        dist = delta.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        horizontal_dist = delta[..., :2].norm(dim=-1, keepdim=True)
        unit = delta / dist
        size_ratio = torch.log(size.unsqueeze(1) / size.unsqueeze(2))
        rel = torch.cat((delta, dist, horizontal_dist, unit, size_ratio), dim=-1)
        rel_encoded = self.rel_mlp(rel)

        pair_mask = cand_mask.unsqueeze(1).bool() & cand_mask.unsqueeze(2).bool()
        eye = torch.eye(pair_mask.shape[1], dtype=torch.bool, device=pair_mask.device).unsqueeze(0)
        pair_mask = pair_mask & ~eye
        pair_mask_f = pair_mask.unsqueeze(-1).float()
        mean = (rel_encoded * pair_mask_f).sum(dim=2) / pair_mask_f.sum(dim=2).clamp_min(1.0)
        max_values = rel_encoded.masked_fill(~pair_mask.unsqueeze(-1), -1e9).max(dim=2).values
        max_values = torch.where(pair_mask.any(dim=2, keepdim=True), max_values, torch.zeros_like(max_values))
        return 0.5 * (mean + max_values)

    def forward(
        self,
        img: torch.Tensor,
        cand_bboxes: torch.Tensor,
        cand_mask: torch.Tensor,
        cand_semantic_ids: torch.Tensor | None = None,
        cand_stage1_features: torch.Tensor | None = None,
        cand_stage1_scores: torch.Tensor | None = None,
        cand_point_counts: torch.Tensor | None = None,
        cand_alive_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        im = self.img_mlp(img)  # [B,C,d]
        branches = [im]
        if self.use_bbox_geom:
            branches.append(self.bbox_mlp(self._bbox_features(cand_bboxes, cand_mask)))
        if self.use_relative_geom:
            branches.append(self._relative_features(cand_bboxes, cand_mask))
        if self.use_semantic_label:
            if cand_semantic_ids is None:
                cand_semantic_ids = torch.zeros_like(cand_mask, dtype=torch.long)
            semantic_ids = cand_semantic_ids.long().clamp(0, self.semantic_embedding.num_embeddings - 2) + 1
            semantic_ids = torch.where(cand_mask > 0, semantic_ids, torch.zeros_like(semantic_ids))
            branches.append(self.semantic_embedding(semantic_ids))
        if self.use_stage1_features:
            if cand_stage1_features is None:
                cand_stage1_features = im.new_zeros((*im.shape[:2], self.stage1_mlp[0].in_features))
            branches.append(self.stage1_mlp(cand_stage1_features))
        if self.use_stage1_quality:
            if cand_stage1_scores is None:
                cand_stage1_scores = cand_mask.new_zeros(cand_mask.shape)
            if cand_point_counts is None:
                cand_point_counts = cand_mask.new_zeros(cand_mask.shape)
            score_or_alive_prob = (
                cand_alive_probs if cand_alive_probs is not None
                else torch.sigmoid(cand_stage1_scores)
            )
            quality = torch.stack((
                cand_stage1_scores,
                score_or_alive_prob,
                torch.log1p(cand_point_counts.clamp_min(0.0)),
            ), dim=-1)
            branches.append(self.quality_mlp(quality))
        return self.fuse(torch.cat(branches, dim=-1) if len(branches) > 1 else branches[0]) * cand_mask.unsqueeze(-1)

class SetPool(nn.Module):
    def __init__(self, d_text: int, d_cand: int, d_state: int) -> None:
        super().__init__()
        self.q = nn.Linear(d_text, d_state)
        self.k = nn.Linear(d_cand, d_state)
        self.v = nn.Linear(d_cand, d_state)
        self.out = nn.Sequential(nn.Linear(d_state, d_state), nn.ReLU(), nn.Linear(d_state, d_state))

    def forward(self, txt: torch.Tensor, cand_feat: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        q = self.q(txt).unsqueeze(1)
        k = self.k(cand_feat)
        v = self.v(cand_feat)
        att = (q * k).sum(dim=-1)
        att = att.masked_fill(cand_mask <= 0, -1e9)
        w = torch.softmax(att, dim=-1).unsqueeze(-1)
        state = (v * w).sum(dim=1)
        return self.out(state)
