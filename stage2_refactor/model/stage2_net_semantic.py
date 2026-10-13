from .heads_selector import ConstraintSelectorHead
try:
    from stage2_refactor.config import TrainConfig
except ImportError:
    from stage2_refactor.config import TrainConfig
import torch
from torch import nn
from .constraint_embed_semantic import SemanticConstraintEmbedder
from .encoders_semantic import QueryEncoder, ImageCandidateEncoder, SetPool
from .heads_semantic import MaskHead, StopHead


class Stage2NetSemantic(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        cfg: TrainConfig,
        num_constraints: int,
        num_rel: int = 6,
        num_attr: int = 6,
        constraint_bank: dict | None = None,
        use_semantic_constraint_scoring: bool = False,
    ):
        super().__init__()
        self.cfg = cfg

        self.qenc = QueryEncoder(vocab_size, cfg.d_text)
        self.cenc = SemanticConstraintEmbedder(
            cfg.d_constraint, vocab_size, num_rel=num_rel, num_attr=num_attr
        )
        self.cand = ImageCandidateEncoder(
            cfg.imgfeat_dim,
            cfg.d_cand,
            use_bbox_geom=cfg.use_bbox_geom,
            use_relative_geom=cfg.use_relative_geom,
            use_semantic_label=cfg.use_semantic_label,
            num_semantic_classes=cfg.num_semantic_classes,
            use_stage1_features=cfg.use_stage1_features,
            stage1_feat_dim=cfg.stage1_feat_dim,
            use_stage1_quality=cfg.use_stage1_quality,
        )
        self.pool = SetPool(cfg.d_text, cfg.d_cand, cfg.d_state)
        self.use_scene_context_attention = bool(
            getattr(cfg, "use_scene_context_attention", False)
        )
        if self.use_scene_context_attention:
            self.scene_context_attention = nn.MultiheadAttention(
                cfg.d_cand, num_heads=4, dropout=0.1, batch_first=True
            )
            self.scene_context_norm = nn.LayerNorm(cfg.d_cand)
            self.scene_context_scale = nn.Parameter(torch.zeros(()))
        self.use_constraint_conditioned_context = bool(
            getattr(cfg, "use_constraint_conditioned_context", False)
        )
        if self.use_constraint_conditioned_context:
            self.constraint_context_condition = nn.Linear(
                cfg.d_text + cfg.d_constraint, cfg.d_cand
            )
            self.constraint_context_attention = nn.MultiheadAttention(
                cfg.d_cand, num_heads=4, dropout=0.1, batch_first=True
            )
            self.constraint_relation_geom = nn.Sequential(
                nn.Linear(10, cfg.d_cand),
                nn.GELU(),
                nn.Linear(cfg.d_cand, cfg.d_cand),
            )
            self.constraint_context_fuse = nn.Sequential(
                nn.LayerNorm(6 * cfg.d_cand),
                nn.Linear(6 * cfg.d_cand, 2 * cfg.d_cand),
                nn.GELU(),
                nn.Linear(2 * cfg.d_cand, cfg.d_cand),
            )
            # Start as a small residual so legacy checkpoints retain their policy.
            self.constraint_context_gate = nn.Parameter(torch.tensor(-3.0))

        self.mask = MaskHead(cfg.d_cand, cfg.d_text, cfg.d_constraint, d_model=256)
        self.stop = StopHead(
            cfg.d_state + cfg.d_text + cfg.d_constraint + 4
        )
        self.selector = ConstraintSelectorHead(
            d_state=cfg.d_state,
            d_text=cfg.d_text,
            d_cand=cfg.d_cand,
            num_constraints=num_constraints,
            d_model=256,
            nhead=4,
            num_layers=4,
            dim_ff=512,
            dropout=0.1,
            vocab_size=vocab_size,
            num_rel=num_rel,
            num_attr=num_attr,
            use_semantic_constraint_scoring=use_semantic_constraint_scoring,
        )
        if constraint_bank is not None:
            for name, value in constraint_bank.items():
                self.register_buffer(f"constraint_bank_{name}", value.long())

    def _constraint_bank(self):
        names = ("ctype_ids", "rel_ids", "attr_ids", "field_ids")
        if not all(hasattr(self, f"constraint_bank_{name}") for name in names):
            return None
        return {
            name: getattr(self, f"constraint_bank_{name}") for name in names
        }

    def forward(self, batch: dict) -> dict:
        txt = self.qenc(batch["query_ids"])   # [B, d_text]
        c_emb = self.cenc(
            batch["c_ctype_id"],
            batch["c_rel_id"],
            batch["c_attr_id"],
            batch["field_ids"]
        )  # [B, d_constraint]

        alive_mask = batch.get("alive_mask", batch["cand_mask"])   # [B, C]
        alive_mask = alive_mask * batch["cand_mask"]
        has_alive = alive_mask.gt(0).any(dim=1, keepdim=True)
        alive_mask = torch.where(has_alive, alive_mask, batch["cand_mask"])
        # Context proposals include landmarks and other scene objects needed to
        # interpret relational constraints. They are encoded but can never be
        # selected unless they are also present in alive_mask.
        context_mask = batch.get("context_mask", alive_mask)
        context_mask = context_mask * batch["cand_mask"]
        has_context = context_mask.gt(0).any(dim=1, keepdim=True)
        context_mask = torch.where(has_context, context_mask, alive_mask)

        cand_feat = self.cand(
            batch["img"],
            batch["cand_bboxes"],
            alive_mask,
            cand_semantic_ids=batch.get("cand_semantic_ids"),
            cand_stage1_features=batch.get("cand_stage1_features"),
            cand_stage1_scores=batch.get("cand_stage1_scores"),
            cand_point_counts=batch.get("cand_point_counts"),
            cand_alive_probs=batch.get("cand_alive_probs"),
        )
        context_feat = None
        if self.use_scene_context_attention or self.use_constraint_conditioned_context:
            context_feat = self.cand(
                batch["img"],
                batch["cand_bboxes"],
                context_mask,
                cand_semantic_ids=batch.get("cand_semantic_ids"),
                cand_stage1_features=batch.get("cand_stage1_features"),
                cand_stage1_scores=batch.get("cand_stage1_scores"),
                cand_point_counts=batch.get("cand_point_counts"),
                cand_alive_probs=batch.get("cand_alive_probs"),
            )
        if self.use_scene_context_attention:
            context_update, _ = self.scene_context_attention(
                cand_feat,
                context_feat,
                context_feat,
                key_padding_mask=context_mask.le(0.5),
                need_weights=False,
            )
            cand_feat = cand_feat + torch.tanh(
                self.scene_context_scale
            ) * self.scene_context_norm(context_update)
            cand_feat = cand_feat * alive_mask.unsqueeze(-1).float()
        # Select the next action without conditioning on that action's label.
        pre_action_cand_feat = cand_feat
        if self.use_constraint_conditioned_context:
            condition = self.constraint_context_condition(
                torch.cat((txt, c_emb), dim=-1)
            )
            conditioned_query = cand_feat + condition.unsqueeze(1)
            context_update, landmark_weights = self.constraint_context_attention(
                conditioned_query,
                context_feat,
                context_feat,
                key_padding_mask=context_mask.le(0.5),
                need_weights=True,
                average_attn_weights=True,
            )
            candidate_bbox = batch["cand_bboxes"][..., :6].float()
            landmark_weights = landmark_weights * context_mask.unsqueeze(1).float()
            landmark_weights = landmark_weights / landmark_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            landmark_bbox = torch.bmm(
                landmark_weights.to(candidate_bbox.dtype), candidate_bbox
            )
            center_delta = candidate_bbox[..., :3] - landmark_bbox[..., :3]
            size_ratio = torch.log(
                candidate_bbox[..., 3:6].abs().clamp_min(1e-4)
                / landmark_bbox[..., 3:6].abs().clamp_min(1e-4)
            )
            relation_geometry = self.constraint_relation_geom(
                torch.cat(
                    (
                        center_delta,
                        center_delta.abs(),
                        center_delta.norm(dim=-1, keepdim=True),
                        size_ratio,
                    ),
                    dim=-1,
                )
            )
            context_residual = self.constraint_context_fuse(
                torch.cat(
                    (
                        cand_feat,
                        context_update,
                        cand_feat - context_update,
                        cand_feat * context_update,
                        relation_geometry,
                        condition.unsqueeze(1).expand_as(cand_feat),
                    ),
                    dim=-1,
                )
            )
            cand_feat = cand_feat + torch.sigmoid(
                self.constraint_context_gate
            ) * context_residual
            cand_feat = cand_feat * alive_mask.unsqueeze(-1).float()
        used_mask = batch.get("used_constraint_mask", None)        # [B, Nc] or None
        state = self.pool(txt, cand_feat, alive_mask)       # [B, d_state]
        selector_cand_feat = cand_feat
        selector_state = state
        if getattr(self.cfg, "selector_pre_action_state", False):
            selector_cand_feat = pre_action_cand_feat
            selector_state = self.pool(txt, selector_cand_feat, alive_mask)

        mask_logits = self.mask(cand_feat, txt, c_emb, alive_mask)   # [B, C]
        alive = alive_mask.float()
        alive_count = alive.sum(dim=1).clamp_min(1.0)
        total_count = batch["cand_mask"].float().sum(dim=1).clamp_min(1.0)
        mask_probability = torch.sigmoid(mask_logits) * alive
        expected_keep = mask_probability.sum(dim=1) / alive_count
        masked_logits = mask_logits.masked_fill(alive <= 0, -1e9)
        belief = torch.softmax(masked_logits, dim=1) * alive
        belief = belief / belief.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(belief * torch.log(belief.clamp_min(1e-6))).sum(dim=1)
        entropy = entropy / torch.log(alive_count.clamp_min(2.0))
        top_probability = mask_probability.max(dim=1).values
        stop_stats = torch.stack(
            (
                alive_count / total_count,
                torch.log1p(alive_count) / torch.log1p(total_count),
                expected_keep,
                0.5 * (entropy + top_probability),
            ),
            dim=1,
        )
        stop_logit = self.stop(
            torch.cat([state, txt, c_emb, stop_stats], dim=-1)
        )

        # A target summary is only valid when it comes from a previous model
        # prediction during rollout. Reading target_index here would leak the
        # answer during training and is unavailable in first-round inference.
        target_summary = batch.get("target_summary", None)
        if target_summary is None and batch.get("cand_belief") is not None:
            # A removed candidate must not contribute to the next-round target
            # summary. Using cand_mask here retained probability mass from stale
            # candidates and made the selector state disagree with rollout.
            belief = batch["cand_belief"].float() * alive_mask.float()
            belief = belief / belief.sum(dim=1, keepdim=True).clamp_min(1e-6)
            target_summary = torch.sum(selector_cand_feat * belief.unsqueeze(-1), dim=1)
        selector_logits = self.selector(
            state=selector_state,
            txt=txt,
            cand_feat=selector_cand_feat,
            alive_mask=alive_mask,
            target_summary=target_summary,
            used_mask=used_mask,
            constraint_bank=self._constraint_bank(),
        )

        return {
            "txt": txt,
            "c_emb": c_emb,
            "cand_feat": cand_feat,
            "state": state,
            "mask_logits": mask_logits,
            "stop_logit": stop_logit,
            "selector_logits": selector_logits,
            "selector_cand_feat": selector_cand_feat,
        }
