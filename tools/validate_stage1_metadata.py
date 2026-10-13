import argparse
import json


def validate_rows(rows):
    if not rows:
        raise ValueError('Empty MT annotations')
    keys = set()
    for row in rows:
        if row.get('eval_type') != 'mt' or not row.get('object_ids'):
            raise ValueError('MT rows require a nonempty query-compatible object_ids set')
        if not row.get('description', '').strip():
            raise ValueError('Empty query')
        if len(set(row['object_ids'])) < 2:
            raise ValueError('MT row has fewer than two distinct compatible objects')
        key = (row['scene_id'], str(row['ann_id']))
        if key in keys:
            raise ValueError(f'Duplicate scene/ann key {key}')
        keys.add(key)
        if row.get('true_target_id') is not None and row['true_target_id'] not in row['object_ids']:
            raise ValueError(f'Intended target is absent from compatible GT set: {key}')
    return {row['scene_id'] for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    args = parser.parse_args()
    train = json.load(open(args.train, encoding='utf-8'))
    val = json.load(open(args.val, encoding='utf-8'))
    overlap = validate_rows(train) & validate_rows(val)
    if overlap:
        raise ValueError(f'Train/validation scene overlap: {len(overlap)} scenes')
    print(f'MT metadata valid: train={len(train)}, val={len(val)}, no scene overlap')


if __name__ == '__main__':
    main()
