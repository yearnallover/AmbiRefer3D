# model/heads_selector_transformer.py
from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class ConstraintSelectorHead(nn.Module):
    """
    2-layer Transformer selector.

    Tokens:
      [CLS] + [TGT] + [CAND_1..CAND_C]

    If target_summary is None:
      - use zero vector as target token / target reference
      - add a binary flag (has_target_summary) into conditioning
    """
    def __init__(
        self,
        d_state: int,
        d_text: int,
        d_cand: int,
        num_constraints: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_constraints = num_constraints
        self.d_cand = d_cand

        # candidate triple -> d_model
        self.cand_proj = nn.Sequential(
            nn.Linear(3 * d_cand, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # target feature -> d_model
        self.tgt_proj = nn.Sequential(
            nn.Linear(d_cand, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # condition from (state + txt + has_target_flag) -> d_model
        self.cond_proj = nn.Sequential(
            nn.Linear(d_state + d_text + 1, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out = nn.Linear(d_model, num_constraints)

    def forward(
        self,
        state: Tensor,                    # [B, d_state]
        txt: Tensor,                      # [B, d_text]
        cand_feat: Tensor,                # [B, C, d_cand]
        alive_mask: Tensor,               # [B, C] float 0/1
        target_summary: Optional[Tensor] = None,   # [B, d_cand] or None
        used_mask: Optional[Tensor] = None,        # [B, Nc] or None
    ) -> Tensor:
        B, C, D = cand_feat.shape
        device = cand_feat.device
        dtype = cand_feat.dtype

        # -----------------------------
        # handle optional target_summary
        # -----------------------------
        if target_summary is None:
            target_summary = torch.zeros(B, D, device=device, dtype=dtype)
            has_target = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            has_target = torch.ones(B, 1, device=device, dtype=dtype)

        # -----------------------------
        # build tokens
        # -----------------------------
        cond_in = torch.cat([state, txt, has_target], dim=-1)   # [B, d_state+d_text+1]
        cond = self.cond_proj(cond_in)                          # [B, d_model]

        cls = self.cls.expand(B, 1, self.d_model)              # [B,1,d_model]

        # TGT token
        tgt = self.tgt_proj(target_summary).unsqueeze(1)        # [B,1,d_model]

        # CAND tokens
        t = target_summary.unsqueeze(1).expand(B, C, D)         # [B,C,D]
        triple = torch.cat([cand_feat, cand_feat - t, cand_feat * t], dim=-1)
        cand_tok = self.cand_proj(triple)                       # [B,C,d_model]

        x = torch.cat([cls, tgt, cand_tok], dim=1)              # [B, 2+C, d_model]
        x = x + cond.unsqueeze(1)

        # -----------------------------
        # attention mask
        # -----------------------------
        alive_bool = (alive_mask > 0.5)                         # [B,C]
        pad_cand = ~alive_bool                                  # [B,C]
        pad_prefix = torch.zeros((B, 2), dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([pad_prefix, pad_cand], dim=1)  # [B,2+C]

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)    # [B,2+C,d_model]
        cls_h = h[:, 0, :]                                            # [B,d_model]

        logits = self.out(cls_h)                                      # [B,Nc]
        if used_mask is not None:
            logits = logits.masked_fill(used_mask > 0, -1e9)

        return logits
