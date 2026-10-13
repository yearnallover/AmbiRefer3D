from lightning.pytorch.callbacks import Callback
from math import cos, pi


class LrDecayCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # cosine learning rate decay
        current_epoch = trainer.current_epoch
        start_epoch = pl_module.hparams.cfg.model.lr_decay.start_epoch
        if trainer.current_epoch < start_epoch:
            return
        end_epoch = pl_module.hparams.cfg.trainer.max_epochs
        absolute_floor = 1e-6
        decay_epochs = max(end_epoch - start_epoch, 1)
        progress = min(max((current_epoch - start_epoch) / decay_epochs, 0.0), 1.0)
        for param_group in trainer.optimizers[0].param_groups:
            # Separate Stage1/Stage2 parameter groups can have different learning
            # rates. Preserve each group's own starting rate during decay and
            # never raise a deliberately small learning rate toward the global
            # floor.
            base_lr = param_group.setdefault("lr_decay_base_lr", param_group["lr"])
            floor_lr = min(absolute_floor, base_lr)
            param_group["lr"] = floor_lr + 0.5 * (base_lr - floor_lr) * (
                1 + cos(pi * progress)
            )
