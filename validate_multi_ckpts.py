"""Evaluate independent Stage 1 checkpoints and export all grounding metrics."""
import argparse
import csv
import json
import math
from pathlib import Path

import hydra
import lightning.pytorch as pl

from m3drefclip.data.data_module import DataModule
from train import load_initial_weights


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', nargs='+', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--export_query_metrics', action='store_true')
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    with hydra.initialize(version_base=None, config_path='config'):
        cfg = hydra.compose(config_name='global_config', overrides=[
            'data=multi3drefer', f'project_root_path={root}', 'logger=null', *args.overrides])
    pl.seed_everything(cfg.test_seed, workers=True)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = DataModule(cfg.data)
    data.val_set = data.dataset(cfg.data, cfg.data.evaluation.split)
    rows = []
    for index, checkpoint in enumerate(args.ckpt):
        model = hydra.utils.instantiate(cfg.model.model_name, cfg)
        load_initial_weights(model, checkpoint, strict=True)
        trainer = pl.Trainer(accelerator=cfg.trainer.accelerator, devices=args.devices,
                             logger=False, enable_checkpointing=False)
        trainer.validate(model, datamodule=data)
        name = f'{index:02d}_{Path(checkpoint).stem}'
        metrics = model.last_validation_results
        row = {'checkpoint': checkpoint}
        for metric, groups in metrics.items():
            for group, value in groups.items():
                row[f'{metric}_{group}'] = float(value) if math.isfinite(float(value)) else None
        rows.append(row)
        if args.export_query_metrics:
            records = model.evaluator.last_query_results
            (output / f'{name}_queries.json').write_text(json.dumps(json_safe(records), indent=2, allow_nan=False))
    (output / 'val_metrics.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    with (output / 'val_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Metrics saved to {output}')


if __name__ == '__main__':
    main()
