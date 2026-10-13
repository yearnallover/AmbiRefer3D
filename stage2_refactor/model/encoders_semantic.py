from __future__ import annotations
import torch
import torch.nn as nn

class QueryEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_text: int) -> None:
        super().__init__()
        self.wemb = nn.Embedding(vocab_size, d_text, padding_idx=0)
        self.rnn = nn.GRU(d_text, d_text, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d_text, d_text)

    def forward(self, query_ids: torch.Tensor) -> torch.Tensor:
        x = self.wemb(query_ids)
        h, _ = self.rnn(x)
        last = h[:, -1, :]
        return self.proj(last)  # [B,d_text]

class ImageCandidateEncoder(nn.Module):
    """
    imgfeat [B,C,D] -> cand_feat [B,C,d_cand]
    Optionally fuse bbox geom.
    """
    def __init__(self, imgfeat_dim: int, d_cand: int, use_bbox_geom: bool = False) -> None:
        super().__init__()
        self.use_bbox_geom = use_bbox_geom
        self.img_mlp = nn.Sequential(
            nn.Linear(imgfeat_dim, d_cand),
            nn.ReLU(),
            nn.Linear(d_cand, d_cand),
        )
        if use_bbox_geom:
            self.bbox_mlp = nn.Sequential(
                nn.Linear(7, d_cand),
                nn.ReLU(),
                nn.Linear(d_cand, d_cand),
            )
            self.fuse = nn.Sequential(
                nn.Linear(d_cand * 2, d_cand),
                nn.ReLU(),
                nn.Linear(d_cand, d_cand),
            )
        else:
            self.fuse = nn.Identity()

    def forward(self, img: torch.Tensor, cand_bboxes: torch.Tensor) -> torch.Tensor:
        im = self.img_mlp(img)  # [B,C,d]
        if self.use_bbox_geom:
            bb = self.bbox_mlp(cand_bboxes)  # [B,C,d]
            return self.fuse(torch.cat([im, bb], dim=-1))
        return im

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

