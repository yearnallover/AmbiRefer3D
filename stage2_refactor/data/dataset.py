from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from .vocab import Vocab
from .bbox_provider import BBoxProvider

def _as_np(x: Any, dtype=np.float32) -> np.ndarray:
    return np.asarray(x, dtype=dtype)

def parse_semantic_constraint(d: Dict[str, Any]) -> Dict[str, Any]:
    # created by your converter; keep permissive
    c = d.get("constraint_gt") or {}
    ctype = (c.get("ctype") or "REL").upper()
    rel = (c.get("rel") or "near").lower()
    landmark = (c.get("landmark_text") or "")
    attr_key = (c.get("attr_key") or "other").lower()
    attr_val = (c.get("attr_val_text") or "")
    return {"ctype": ctype, "rel": rel, "landmark_text": landmark, "attr_key": attr_key, "attr_val_text": attr_val}

class RoundDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        vocab: Vocab,
        max_len: int,
        bbox_provider: BBoxProvider,
        img_cache_dir: Optional[str] = None,
        imgfeat_dim: int = 768,
        use_imgfeat: bool = True,
    ) -> None:
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len
        self.bbox_provider = bbox_provider
        self.img_cache_dir = img_cache_dir
        self.imgfeat_dim = imgfeat_dim
        self.use_imgfeat = use_imgfeat
        self._img_cache_mem = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        scene_id = s["scene_id"]
        query = (s.get("query") or "").strip()
        cand_ids = [int(x) for x in s["cand_ids"]]
        C = len(cand_ids)

        # bbox (optional for now; still pass through)
        cand_bboxes = self.bbox_provider.get_bboxes(scene_id, cand_ids, sample=s)  # [C,7]

        # imgfeat should be loaded similarly as之前（略），你可以复用我上一条给你的 cache 读取函数
        # 这里先置零，确保代码结构完整
        img = np.zeros((C, self.imgfeat_dim), dtype=np.float32)

        cand_mask = np.ones((C,), dtype=np.float32)

        # target supervision (weak but works)
        target_index = int(s.get("target_index", 0))
        mask_y = np.zeros((C,), dtype=np.float32)
        if 0 <= target_index < C:
            mask_y[target_index] = 1.0

        c = parse_semantic_constraint(s)
        # map to ids (REL/ATTR etc.) ——在 collate 前就转成 int，避免 Python dict 在多进程 dataloader 里麻烦
        ctype_id = 0 if c["ctype"] == "REL" else 1

        # simple fixed maps
        rel2id = {"near": 0, "behind": 1, "in_front_of": 2, "next_to": 3}
        attr2id = {"color": 0, "material": 1, "shape": 2, "size": 3, "height": 4, "other": 5}
        rel_id = rel2id.get(c["rel"], 0)
        attr_id = attr2id.get(c["attr_key"], 5)

        # field text: landmark_text if REL else attr_val_text
        field_text = c["landmark_text"] if c["ctype"] == "REL" else c["attr_val_text"]
        field_ids = self.vocab.encode(field_text, self.max_len)

        out = {
            "query_ids": torch.tensor(self.vocab.encode(query, self.max_len), dtype=torch.long),
            "field_ids": torch.tensor(field_ids, dtype=torch.long),

            "cand_bboxes": torch.tensor(_as_np(cand_bboxes, np.float32), dtype=torch.float32),  # optional
            "img": torch.tensor(img, dtype=torch.float32),
            "cand_mask": torch.tensor(cand_mask, dtype=torch.float32),
            "mask_y": torch.tensor(mask_y, dtype=torch.float32),

            "c_ctype_id": torch.tensor(ctype_id, dtype=torch.long),
            "c_rel_id": torch.tensor(rel_id, dtype=torch.long),
            "c_attr_id": torch.tensor(attr_id, dtype=torch.long),

            "stop_y": torch.tensor(float(s.get("stop_y", 0)), dtype=torch.float32),
        }
        return out
