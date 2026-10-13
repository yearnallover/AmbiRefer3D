from __future__ import annotations
import torch
import torch.nn as nn

class SemanticConstraintEmbedder(nn.Module):
    """
    Parameterized semantic constraint, still embedded.
    Slots:
      - ctype: REL/ATTR
      - rel type / attr key
      - text field: landmark_text or attr_val_text (encoded by a tiny text encoder)
    """
    def __init__(self, d: int, vocab_size: int) -> None:
        super().__init__()
        self.d = d

        # slot enums
        self.ctype2id = {"REL": 0, "ATTR": 1}
        self.rel2id = {"near": 0, "behind": 1, "in_front_of": 2, "next_to": 3}
        self.attr2id = {"color": 0, "material": 1, "shape": 2, "size": 3, "height": 4, "other": 5}

        self.e_ctype = nn.Embedding(2, d)
        self.e_rel = nn.Embedding(len(self.rel2id), d)
        self.e_attr = nn.Embedding(len(self.attr2id), d)

        # text-field encoder (small)
        self.wemb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.rnn = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.txt_proj = nn.Linear(2 * d, d)

        self.fuse = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        # text_ids [B,L]
        x = self.wemb(text_ids)        # [B,L,d]
        h, _ = self.rnn(x)             # [B,L,2d]
        last = h[:, -1, :]
        return self.txt_proj(last)     # [B,d]

    def forward(
        self,
        ctype_id: torch.Tensor,   # [B] 0/1
        rel_id: torch.Tensor,     # [B] (valid if REL)
        attr_id: torch.Tensor,    # [B] (valid if ATTR)
        field_text_ids: torch.Tensor,  # [B,L] landmark_text or attr_val_text
    ) -> torch.Tensor:
        et = self.e_ctype(ctype_id)
        er = self.e_rel(rel_id.clamp(min=0, max=self.e_rel.num_embeddings-1))
        ea = self.e_attr(attr_id.clamp(min=0, max=self.e_attr.num_embeddings-1))
        ef = self.encode_text(field_text_ids)
        x = torch.cat([et, er + ea, ef], dim=-1)
        return self.fuse(x)  # [B,d]
