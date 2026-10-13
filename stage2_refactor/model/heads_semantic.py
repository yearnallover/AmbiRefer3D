from __future__ import annotations
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MaskHead(nn.Module):
    """
    Candidate tokens attend to condition tokens (query + constraint).
    Then score each candidate.
    """
    def __init__(self, d_cand: int, d_text: int, d_constraint: int,
                 d_model: int = 256, nhead: int = 4, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.cand_proj = nn.Linear(d_cand, d_model)
        self.txt_proj = nn.Linear(d_text, d_model)
        self.c_proj = nn.Linear(d_constraint, d_model)

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, cand_feat: torch.Tensor, txt: torch.Tensor, c_emb: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        # cand_feat [B,C,Dc]
        B, C, _ = cand_feat.shape

        q = self.cand_proj(cand_feat)                       # [B,C,d]
        mem_txt = self.txt_proj(txt).unsqueeze(1)           # [B,1,d]
        mem_c   = self.c_proj(c_emb).unsqueeze(1)           # [B,1,d]
        mem = torch.cat([mem_txt, mem_c], dim=1)            # [B,2,d]

        # key_padding for candidates: True means ignore
        cand_kpm = (cand_mask <= 0.5)                       # [B,C] bool

        # Cross-attn: candidates query, memory key/value
        h, _ = self.attn(query=q, key=mem, value=mem, key_padding_mask=None, need_weights=False)  # [B,C,d]
        h = h + q
        h = h + self.ffn(h)

        logits = self.out(h).squeeze(-1)                    # [B,C]
        logits = logits.masked_fill(cand_kpm, -1e9)
        return logits


class StopHead(nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, d_in), nn.ReLU(), nn.Linear(d_in, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)
