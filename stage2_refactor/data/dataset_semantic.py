from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RoundDatasetSemantic(Dataset):
    def __init__(self, samples, vocab, max_len, bbox_provider, img_cache_dir,
                 imgfeat_dim=768, constraint_table=None, use_imgfeat=True, use_bbox_geom=False):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len
        self.bbox_provider = bbox_provider
        self.img_cache_dir = Path(img_cache_dir)
        self.imgfeat_dim = imgfeat_dim
        self.constraint_table = constraint_table
        self.use_imgfeat = use_imgfeat
        self.use_bbox_geom = use_bbox_geom
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def _images(self, scene, candidate_ids):
        if not self.use_imgfeat:
            return np.zeros((len(candidate_ids), self.imgfeat_dim), dtype=np.float32)
        if scene not in self._cache:
            with np.load(self.img_cache_dir / f'{scene}.npz') as data:
                ids = np.asarray(data['obj_ids']).reshape(-1)
                features = np.asarray(data['feats'], dtype=np.float32)
                if features.shape != (len(ids), self.imgfeat_dim) or not np.isfinite(features).all():
                    raise ValueError(f'Invalid image features for {scene}')
                self._cache[scene] = dict(zip(ids.tolist(), features.copy()))
        mapping = self._cache[scene]
        if any(candidate not in mapping for candidate in candidate_ids):
            raise KeyError(f'Missing candidate image features for {scene}')
        return np.stack([mapping[candidate] for candidate in candidate_ids])

    def __getitem__(self, index):
        sample = self.samples[index]
        candidates = [int(value) for value in sample['cand_ids']]
        count = len(candidates)
        if count == 0:
            raise ValueError('Empty candidate set')
        target = int(sample['target_index'])
        if not -1 <= target < count:
            raise ValueError('Invalid target_index')
        constraint_id = self.constraint_table.id_from_constraint_gt(sample.get('constraint_gt') or {})
        fields = self.constraint_table.row(constraint_id)
        labels = np.zeros(count, dtype=np.float32)
        if target >= 0:
            labels[target] = 1
        boxes = self.bbox_provider.get_bboxes(sample['scene_id'], candidates, sample=sample) if self.use_bbox_geom else np.zeros((count, 7), dtype=np.float32)
        return {
            'query_ids': torch.tensor(self.vocab.encode(sample['query'], self.max_len), dtype=torch.long),
            'field_ids': torch.tensor(self.vocab.encode(fields.field_text, self.max_len), dtype=torch.long),
            'c_ctype_id': torch.tensor(fields.ctype_id, dtype=torch.long),
            'c_rel_id': torch.tensor(fields.rel_id, dtype=torch.long),
            'c_attr_id': torch.tensor(fields.attr_id, dtype=torch.long),
            'constraint_id': torch.tensor(constraint_id, dtype=torch.long),
            'target_index': torch.tensor(target, dtype=torch.long),
            'stop_y': torch.tensor(float(sample['stop_y'])),
            'img': torch.tensor(self._images(sample['scene_id'], candidates)),
            'cand_mask': torch.ones(count),
            'mask_y': torch.tensor(labels),
            'cand_bboxes': torch.tensor(np.asarray(boxes, dtype=np.float32)),
        }
