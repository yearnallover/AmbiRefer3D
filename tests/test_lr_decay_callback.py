import unittest
from types import SimpleNamespace

from m3drefclip.callback.lr_decay_callback import LrDecayCallback


class _Optimizer:
    def __init__(self, learning_rates):
        self.param_groups = [{"lr": value} for value in learning_rates]


class LrDecayCallbackTest(unittest.TestCase):
    @staticmethod
    def _module(max_epochs=40, start_epoch=0):
        return SimpleNamespace(
            hparams=SimpleNamespace(
                cfg=SimpleNamespace(
                    model=SimpleNamespace(
                        lr_decay=SimpleNamespace(start_epoch=start_epoch)
                    ),
                    trainer=SimpleNamespace(max_epochs=max_epochs),
                )
            )
        )

    def test_global_floor_never_increases_small_parameter_group_lr(self):
        optimizer = _Optimizer([2e-7, 5e-7, 2e-6, 3e-4])
        trainer = SimpleNamespace(current_epoch=20, optimizers=[optimizer])

        LrDecayCallback().on_train_epoch_end(
            trainer, self._module(max_epochs=40)
        )

        learning_rates = [group["lr"] for group in optimizer.param_groups]
        self.assertEqual(learning_rates[0], 2e-7)
        self.assertEqual(learning_rates[1], 5e-7)
        self.assertLess(learning_rates[2], 2e-6)
        self.assertLess(learning_rates[3], 3e-4)

    def test_each_group_keeps_its_original_decay_base(self):
        optimizer = _Optimizer([2e-6])
        callback = LrDecayCallback()
        module = self._module(max_epochs=40)

        trainer = SimpleNamespace(current_epoch=10, optimizers=[optimizer])
        callback.on_train_epoch_end(trainer, module)
        first_lr = optimizer.param_groups[0]["lr"]
        trainer.current_epoch = 20
        callback.on_train_epoch_end(trainer, module)

        self.assertEqual(optimizer.param_groups[0]["lr_decay_base_lr"], 2e-6)
        self.assertLess(optimizer.param_groups[0]["lr"], first_lr)


if __name__ == "__main__":
    unittest.main()
