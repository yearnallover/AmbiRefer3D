from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class SemanticConstraintEmbedder(nn.Module):
    """
    c_emb = fuse( slot_embs + field_text_emb )
    Slots:
      - ctype: REL/ATTR
      - rel_id / attr_id
      - field_text_ids: landmark_text or attr_val_text
    """
    def __init__(self, d: int, vocab_size: int, num_rel: int, num_attr: int) -> None:
        super().__init__()
        self.d = d
        self.e_ctype = nn.Embedding(2, d)
        self.e_rel = nn.Embedding(num_rel, d)
        self.e_attr = nn.Embedding(num_attr, d)

        # text encoder for field text
        self.wemb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.rnn = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.field_proj = nn.Linear(2 * d, d)

        self.fuse = nn.Sequential(
            nn.Linear(d * 4, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

    def encode_field(self, field_ids: torch.Tensor) -> torch.Tensor:
        x = self.wemb(field_ids)     # [B,L,d]
        lengths = field_ids.ne(0).sum(dim=1).clamp_min(1)
        packed = pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        last = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return self.field_proj(last) # [B,d]

    def forward(self, ctype_id: torch.Tensor, rel_id: torch.Tensor, attr_id: torch.Tensor, field_ids: torch.Tensor) -> torch.Tensor:
        et = self.e_ctype(ctype_id)  # [B,d]
        er = self.e_rel(rel_id.clamp(min=0, max=self.e_rel.num_embeddings-1))
        ea = self.e_attr(attr_id.clamp(min=0, max=self.e_attr.num_embeddings-1))
        ef = self.encode_field(field_ids)  # [B,d]
        x = torch.cat([et, er, ea, ef], dim=-1)
        return self.fuse(x)  # [B,d]
