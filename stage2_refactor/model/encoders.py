from __future__ import annotations
import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_text: int = 256) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_text, padding_idx=0)
        self.rnn = nn.GRU(d_text, d_text, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_text * 2, d_text)

    def forward(self, query_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(query_ids)
        h, _ = self.rnn(x)
        last = h[:, -1, :]
        return self.proj(last)

class CandidateEncoder(nn.Module):
    def __init__(
        self,
        d_text: int,
        d_cand: int = 256,
        use_img: bool = True,
        geom_dim: int = 11,
        rel_dim: int = 7,
        img_dim: int = 768,
    ) -> None:
        super().__init__()
        self.use_img = use_img
        self.img_dim = img_dim

        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, d_cand),
            nn.ReLU(),
            nn.Linear(d_cand, d_cand),
        )
        self.rel_mlp = nn.Sequential(
            nn.Linear(rel_dim, d_cand),
            nn.ReLU(),
            nn.Linear(d_cand, d_cand),
        )

        if use_img:
            self.img_mlp = nn.Sequential(
                nn.Linear(img_dim, d_cand),
                nn.ReLU(),
                nn.Linear(d_cand, d_cand),
            )
            self.fuse = nn.Sequential(
                nn.Linear(d_cand * 3, d_cand),
                nn.ReLU(),
                nn.Linear(d_cand, d_cand),
            )
        else:
            self.fuse = nn.Sequential(
                nn.Linear(d_cand * 2, d_cand),
                nn.ReLU(),
                nn.Linear(d_cand, d_cand),
            )

    def forward(
        self,
        geom: torch.Tensor,      # [B,C,geom_dim]
        rel: torch.Tensor,       # [B,C,C,rel_dim]
        img: torch.Tensor,       # [B,C,img_dim]
        cand_mask: torch.Tensor, # [B,C]
    ) -> torch.Tensor:
        g = self.geom_mlp(geom)

        rr = self.rel_mlp(rel)  # [B,C,C,d]
        m = cand_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,C,1]
        rr = rr * m
        denom = m.sum(dim=2).clamp(min=1.0)       # [B,1,1]
        r_pool = rr.sum(dim=2) / denom            # [B,C,d]

        if self.use_img:
            if img.size(-1) != self.img_dim:
                # soft fallback: if mismatch, project by slicing/padding (keeps it robust)
                B, C, D = img.shape
                if D > self.img_dim:
                    img2 = img[..., : self.img_dim]
                else:
                    pad = img.new_zeros((B, C, self.img_dim - D))
                    img2 = torch.cat([img, pad], dim=-1)
                img = img2
            im = self.img_mlp(img)
            feat = torch.cat([g, r_pool, im], dim=-1)
        else:
            feat = torch.cat([g, r_pool], dim=-1)

        return self.fuse(feat)

class SetPool(nn.Module):
    def __init__(self, d_text: int, d_cand: int, d_state: int) -> None:
        super().__init__()
        self.q = nn.Linear(d_text, d_state)
        self.k = nn.Linear(d_cand, d_state)
        self.v = nn.Linear(d_cand, d_state)
        self.out = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.ReLU(),
            nn.Linear(d_state, d_state),
        )

    def forward(self, txt: torch.Tensor, cand_feat: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        q = self.q(txt).unsqueeze(1)
        k = self.k(cand_feat)
        v = self.v(cand_feat)
        att = (q * k).sum(dim=-1)
        att = att.masked_fill(cand_mask <= 0, -1e9)
        w = torch.softmax(att, dim=-1).unsqueeze(-1)
        state = (v * w).sum(dim=1)
        return self.out(state)
