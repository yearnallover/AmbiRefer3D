import os

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor

from m3drefclip.callback.gpu_cache_clean_callback import GPUCacheCleanCallback
from m3drefclip.callback.lr_decay_callback import LrDecayCallback
from m3drefclip.data.data_module import DataModule


def load_initial_weights(model, path, strict=False):
    if not path:
        return
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)
    result = model.load_state_dict(state, strict=bool(strict))
    print(f'Initial weights: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}')


@hydra.main(version_base=None, config_path='config', config_name='global_config')
def main(cfg):
    pl.seed_everything(cfg.train_seed, workers=True)
    os.makedirs(os.path.join(cfg.experiment_output_path, 'training'), exist_ok=True)
    data_module = DataModule(cfg.data)
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)
    if cfg.get('detector_path'):
        checkpoint = torch.load(cfg.detector_path, map_location='cpu', weights_only=False)
        model.detector.load_state_dict(checkpoint['state_dict'])
    load_initial_weights(model, cfg.get('init_from_ckpt'), cfg.get('init_from_ckpt_strict', False))
    logger = hydra.utils.instantiate(cfg.logger) if cfg.get('logger') is not None else False
    callbacks = [hydra.utils.instantiate(cfg.checkpoint_monitor), GPUCacheCleanCallback(), LrDecayCallback()]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.trainer)
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    main()
