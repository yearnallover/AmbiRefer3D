import torch
from torch.nn.utils.rnn import pad_sequence


def collate_semantic(batch):
    padded = {'img', 'cand_mask', 'mask_y', 'cand_bboxes'}
    return {key: pad_sequence([sample[key] for sample in batch], batch_first=True)
            if key in padded else torch.stack([sample[key] for sample in batch])
            for key in batch[0]}
