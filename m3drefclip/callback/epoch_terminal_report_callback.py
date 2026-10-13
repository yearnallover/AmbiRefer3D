from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback


def _scalar(value: Any):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class EpochTerminalReportCallback(Callback):
    """Append stable epoch summaries to stdout without progress-bar rewriting."""

    component_names = (
        "stage1", "detector", "grounding", "trajectory", "stage2", "fallback",
        "traj_survival", "traj_initial", "traj_coverage", "traj_dpo",
        "traj_set", "traj_raw_set",
        "stage2_selector", "stage2_mask", "stage2_stop",
        "stage2_preference", "stage2_metric", "stage2_dpo",
    )

    def __init__(self) -> None:
        super().__init__()
        self.loss_sum = 0.0
        self.loss_count = 0
        self.printed_train_epoch = None
        self.component_sums = {name: 0.0 for name in self.component_names}

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.loss_sum = 0.0
        self.loss_count = 0
        self.component_sums = {name: 0.0 for name in self.component_names}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        loss = _scalar(loss)
        if loss is None:
            return
        self.loss_sum += loss
        self.loss_count += 1
        components = getattr(pl_module, "_terminal_train_components", {})
        for name in self.component_names:
            value = _scalar(components.get(name))
            if value is not None:
                self.component_sums[name] += value

    @staticmethod
    def _emit(pl_module, line: str) -> None:
        print(line, flush=True)
        output_root = str(pl_module.hparams.cfg.experiment_output_path)
        path = os.path.join(output_root, "training", "epoch_metrics.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _print_train_summary(self, trainer, pl_module) -> None:
        epoch = int(trainer.current_epoch)
        if self.printed_train_epoch == epoch:
            return
        if self.loss_count == 0:
            return
        stats = torch.tensor(
            [
                self.loss_sum,
                float(self.loss_count),
                *(self.component_sums[name] for name in self.component_names),
            ],
            dtype=torch.float64,
            device=pl_module.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if not trainer.is_global_zero:
            self.printed_train_epoch = epoch
            return
        mean_loss = float(stats[0].item() / max(stats[1].item(), 1.0))
        component_means = {
            name: float(stats[index + 2].item() / max(stats[1].item(), 1.0))
            for index, name in enumerate(self.component_names)
        }
        lrs = ",".join(
            f"{group.get('name', index)}:{group['lr']:.3e}"
            for index, group in enumerate(trainer.optimizers[0].param_groups)
        )
        self._emit(
            pl_module,
            f"[TRAIN][epoch {trainer.current_epoch + 1:03d}/{trainer.max_epochs:03d}] "
            f"loss={mean_loss:.6f} stage1={component_means['stage1']:.6f} "
            f"detector={component_means['detector']:.6f} "
            f"grounding={component_means['grounding']:.6f} "
            f"trajectory={component_means['trajectory']:.6f} "
            f"stage2={component_means['stage2']:.6f} "
            f"traj[surv={component_means['traj_survival']:.4f},"
            f"init={component_means['traj_initial']:.4f},"
            f"cov={component_means['traj_coverage']:.4f},"
            f"set={component_means['traj_set']:.4f},"
            f"raw_set={component_means['traj_raw_set']:.4f},"
            f"dpo={component_means['traj_dpo']:.4f}] "
            f"s2[sel={component_means['stage2_selector']:.4f},"
            f"mask={component_means['stage2_mask']:.4f},"
            f"stop={component_means['stage2_stop']:.4f},"
            f"pref={component_means['stage2_preference']:.4f},"
            f"metric={component_means['stage2_metric']:.4f},"
            f"dpo={component_means['stage2_dpo']:.4f}] "
            f"fallback={100.0 * component_means['fallback']:.2f}% lr={lrs}",
        )
        self.printed_train_epoch = epoch

    def on_validation_start(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._print_train_summary(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._print_train_summary(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        # Lightning can replay only the saved validation-loop tail when a
        # checkpoint resumes at an epoch boundary. Those partial metrics do
        # not describe a validation epoch and must not enter the terminal log.
        if self.loss_count == 0:
            return
        metrics = trainer.callback_metrics

        def metric(name: str):
            return _scalar(metrics.get(name))

        def percent(value):
            return "NA" if value is None else f"{100.0 * value:.2f}%"

        ts50 = metric("val_eval/target_survival@0.5_mt")
        ts_no_iou = metric("val_eval/target_survival_no_iou_mt")
        iou25 = metric("val_eval/f1@0.25_mt")
        iou50 = metric("val_eval/f1@0.5_mt")
        final_acc = metric("val_stage2/acc_mt")
        conditional_acc = metric("val_stage2/conditional_acc_mt")
        initial_survival = metric("val_stage2/initial_target_survival_mt")
        final_survival = metric("val_stage2/final_target_survival_mt")
        joint_score = metric("val_eval/joint_score_mt")
        if ts50 is None:
            ts50 = metric("val_eval/target_survival@0.5_overall")
            ts_no_iou = metric("val_eval/target_survival_no_iou_overall")
            iou25 = metric("val_eval/f1@0.25_overall")
            iou50 = metric("val_eval/f1@0.5_overall")
            final_acc = metric("val_stage2/acc")
            conditional_acc = metric("val_stage2/conditional_acc")
            initial_survival = metric("val_stage2/initial_target_survival")
            final_survival = metric("val_stage2/final_target_survival")
            joint_score = metric("val_eval/joint_score_overall")
        self._emit(
            pl_module,
            f"[VAL-MT][epoch {trainer.current_epoch + 1:03d}] "
            f"TS@0.5={percent(ts50)} TS-no-IoU={percent(ts_no_iou)} "
            f"F1@IoU0.25={percent(iou25)} F1@IoU0.50={percent(iou50)} "
            f"Stage2-Final-Acc={percent(final_acc)} "
            f"Stage2-Conditional-Acc={percent(conditional_acc)} "
            f"Stage2-Survival={percent(initial_survival)}->{percent(final_survival)} "
            f"Joint={percent(joint_score)}",
        )
