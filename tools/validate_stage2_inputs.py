import argparse
import json
from pathlib import Path

import numpy as np


def validate_rows(rows, img_cache):
    if not rows:
        raise ValueError('Empty Stage 2 round dataset')
    cache = {}
    missing_target = 0
    for row in rows:
        ids = row.get('cand_ids', [])
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{row.get('scene_id')}: empty or duplicate candidate IDs")
        if not row.get('query', '').strip():
            raise ValueError('Empty Stage 2 query')
        target = int(row['target_index'])
        if target < -1 or target >= len(ids):
            raise ValueError('target_index must be -1 (missed) or a valid candidate index')
        missing_target += target == -1
        if target >= 0 and row.get('target_id') is not None and ids[target] != row['target_id']:
            raise ValueError('target_index and target_id are inconsistent')
        scene = row['scene_id']
        if scene not in cache:
            with np.load(Path(img_cache) / f'{scene}.npz') as data:
                features = np.asarray(data['feats'])
                object_ids = np.asarray(data['obj_ids']).reshape(-1)
                if features.shape != (len(object_ids), 768) or not np.isfinite(features).all():
                    raise ValueError(f'{scene}: image cache must contain finite [N,768] features')
                if len(set(object_ids.tolist())) != len(object_ids):
                    raise ValueError(f'{scene}: duplicate image-cache object IDs')
                cache[scene] = set(object_ids.tolist())
        if not set(ids).issubset(cache[scene]):
            raise ValueError(f'{scene}: candidate images are missing; refusing zero-feature fallback')
        if 'cand_bboxes' in row:
            boxes = np.asarray(row['cand_bboxes'])
            if boxes.shape != (len(ids), 7) or not np.isfinite(boxes).all():
                raise ValueError('cand_bboxes must be finite [C,7] aligned with cand_ids')
    return {'num': len(rows), 'missing_target': missing_target,
            'initial_target_survival': 1 - missing_target / len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round_json', required=True)
    parser.add_argument('--img_cache', required=True)
    args = parser.parse_args()
    print(json.dumps(validate_rows(json.load(open(args.round_json)), args.img_cache), indent=2))


if __name__ == '__main__':
    main()
