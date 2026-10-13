from .heads_selector import ConstraintSelectorHead
from stage2_refactor.config import TrainConfig
import torch
from torch import nn
from .constraint_embed_semantic import SemanticConstraintEmbedder
from .encoders_semantic import QueryEncoder, ImageCandidateEncoder, SetPool
from .heads_semantic import MaskHead, StopHead


class Stage2NetSemantic(nn.Module):
    def __init__(self, vocab_size: int, cfg: TrainConfig, num_constraints: int, num_rel: int = 6, num_attr: int = 6):
        super().__init__()
        self.cfg = cfg

        self.qenc = QueryEncoder(vocab_size, cfg.d_text)
        self.cenc = SemanticConstraintEmbedder(
            cfg.d_constraint, vocab_size, num_rel=num_rel, num_attr=num_attr
        )
        self.cand = ImageCandidateEncoder(
            cfg.imgfeat_dim, cfg.d_cand, use_bbox_geom=cfg.use_bbox_geom
        )
        self.pool = SetPool(cfg.d_text, cfg.d_cand, cfg.d_state)

        self.mask = MaskHead(cfg.d_cand, cfg.d_text, cfg.d_constraint, d_model=256)
        self.stop = StopHead(cfg.d_state + cfg.d_text)
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
        )

    def forward(self, batch: dict) -> dict:
        txt = self.qenc(batch["query_ids"])   # [B, d_text]
        c_emb = self.cenc(
            batch["c_ctype_id"],
            batch["c_rel_id"],
            batch["c_attr_id"],
            batch["field_ids"]
        )  # [B, d_constraint]

        cand_feat = self.cand(batch["img"], batch["cand_bboxes"])   # [B, C, d_cand]
        state = self.pool(txt, cand_feat, batch["cand_mask"])       # [B, d_state]

        mask_logits = self.mask(cand_feat, txt, c_emb, batch["cand_mask"])   # [B, C]
        stop_logit = self.stop(torch.cat([state, txt], dim=-1))               # [B]

        alive_mask = batch.get("alive_mask", batch["cand_mask"])   # [B, C]
        used_mask = batch.get("used_constraint_mask", None)        # [B, Nc] or None

        # -------- target summary logic --------
        # 优先使用外部传入的 target_summary（eval/rollout 第二轮及以后）
        target_summary = batch.get("target_summary", None)
        # print(self.training/)
        # 训练时如果没有传 target_summary，则可回退到 GT
        if target_summary is None and self.training:
            gt = batch["target_index"].long()
            valid_gt = (gt >= 0) & (gt < cand_feat.shape[1])
            gt = gt.clamp(0, cand_feat.shape[1] - 1)  # [B]
            target_summary = cand_feat.gather(
                1, gt.view(-1, 1, 1).expand(-1, 1, cand_feat.size(-1))
            ).squeeze(1) * valid_gt.unsqueeze(1)  # [B, d_cand]

        # eval 第一轮时 target_summary=None，是允许的
        selector_logits = self.selector(
            state=state,
            txt=txt,
            cand_feat=cand_feat,
            alive_mask=alive_mask,
            target_summary=target_summary,   # None for round-1 eval
            used_mask=used_mask,
        )

        return {
            "txt": txt,
            "c_emb": c_emb,
            "cand_feat": cand_feat,
            "state": state,
            "mask_logits": mask_logits,
            "stop_logit": stop_logit,
            "selector_logits": selector_logits,
        }
