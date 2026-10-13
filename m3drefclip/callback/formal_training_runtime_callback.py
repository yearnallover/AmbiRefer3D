from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


FORMAL_RUNTIME_SCHEMA_VERSION = "1.1.0"
SUPPORTED_TRACKS = {
    "query_conditioned_grounding",
    "proposal_detector_recovery",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get(mapping: Any, key: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    if isinstance(mapping, dict):
        value = mapping.get(key, default)
    else:
        value = getattr(mapping, key, default)
    return default if value is None else value


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _nested(mapping: Any, dotted_key: str) -> Any:
    value = mapping
    for key in dotted_key.split("."):
        if isinstance(value, dict):
            if key not in value:
                raise ValueError(f"runtime config is missing {dotted_key}")
            value = value[key]
        else:
            if not hasattr(value, key):
                raise ValueError(f"runtime config is missing {dotted_key}")
            value = getattr(value, key)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"formal runtime artifact must be a JSON object: {path}")
    return value


def _contract_value_matches(key: str, actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return actual is expected
    if key.endswith(("_data", "_path", "_manifest", "_preflight")):
        return Path(str(actual)).resolve() == Path(str(expected)).resolve()
    return actual == expected


def _scalar(value: Any) -> float | None:
    if not torch.is_tensor(value) or value.numel() != 1:
        return None
    return float(value.detach().cpu().item())


class FormalTrainingRuntimeCallback(Callback):
    """Emit fail-closed evidence from Lightning's actual training hooks."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        runtime = _get(cfg, "formal_training_runtime", {})
        self.track = str(
            _get(runtime, "track", "query_conditioned_grounding")
        ).strip()
        if self.track not in SUPPORTED_TRACKS:
            raise ValueError(f"unsupported formal training runtime track: {self.track}")
        self.model_name = str(_get(runtime, "model", "")).strip()
        self.bundle_path = Path(
            str(_get(runtime, "training_bundle_manifest", ""))
        )
        self.preflight_path = Path(str(_get(runtime, "trainer_preflight", "")))
        self.recovery_plan_path = Path(
            str(_get(runtime, "recovery_plan_manifest", ""))
        )
        self.recovery_data_audit_path = Path(
            str(_get(runtime, "recovery_data_audit_manifest", ""))
        )
        self.recovery_bundle_path = Path(
            str(_get(runtime, "recovery_training_bundle_manifest", ""))
        )
        self.recovery_preflight_path = Path(
            str(_get(runtime, "recovery_trainer_preflight", ""))
        )
        self.baseline_path = Path(str(_get(cfg, "init_from_ckpt", "")))
        output_value = str(_get(runtime, "output_report", "")).strip()
        self.output_path = Path(output_value) if output_value else None
        if not self.model_name:
            raise ValueError("formal training runtime model must be non-empty")
        required_inputs = [("baseline checkpoint", self.baseline_path)]
        if self.track == "query_conditioned_grounding":
            required_inputs.extend((
                ("training bundle", self.bundle_path),
                ("trainer preflight", self.preflight_path),
            ))
        else:
            required_inputs.extend((
                ("recovery plan", self.recovery_plan_path),
                ("recovery data audit", self.recovery_data_audit_path),
                ("recovery training bundle", self.recovery_bundle_path),
                ("recovery trainer preflight", self.recovery_preflight_path),
            ))
        for label, path in required_inputs:
            if not path.is_file():
                raise ValueError(f"formal training runtime {label} does not exist: {path}")
        if self.output_path is None:
            raise ValueError("formal training runtime output path must be non-empty")
        self.recovery_config_contract_verified = False
        if self.track == "proposal_detector_recovery":
            preflight = _read_json(self.recovery_preflight_path)
            contract = preflight.get("resolved_training_contract")
            if not isinstance(contract, dict) or not contract:
                raise ValueError(
                    "proposal recovery preflight lacks a resolved training contract"
                )
            mismatches = [
                key for key, expected in contract.items()
                if not _contract_value_matches(key, _nested(cfg, key), expected)
            ]
            if mismatches:
                raise ValueError(
                    "proposal recovery resolved config differs from preflight: "
                    + ", ".join(sorted(mismatches))
                )
            self.recovery_config_contract_verified = True

        self.forward_batch_count = 0
        self.loss_observation_count = 0
        self.backward_batch_count = 0
        self.gradient_check_batch_count = 0
        self.gradient_tensor_check_count = 0
        self.nonzero_detector_gradient_tensor_count = 0
        self.optimizer_step_count = 0
        self.validation_batch_count = 0
        self.detector_loss_observation_count = 0
        self.language_loss_nonzero_batch_count = 0
        self.all_detector_losses_finite = True
        self.all_losses_finite = True
        self.all_gradients_finite = True
        self.frozen_test_used_during_training = False
        self.detector_only_parameter_contract_verified = False
        self.optimizer_parameter_contract_verified = False
        self.trainable_parameter_count = 0

    def on_fit_start(self, trainer, pl_module) -> None:
        if self.track != "proposal_detector_recovery":
            return
        trainable = [
            name for name, parameter in pl_module.named_parameters()
            if parameter.requires_grad
        ]
        declared = set(getattr(
            pl_module, "detector_only_trainable_parameter_names", []
        ))
        if (
            getattr(pl_module, "detector_only_recovery_enabled", False) is not True
            or not trainable
            or any(not name.startswith("detector.") for name in trainable)
            or set(trainable) != declared
        ):
            raise ValueError(
                "proposal recovery requires the detector-only parameter contract"
            )
        self.trainable_parameter_count = len(trainable)
        self.detector_only_parameter_contract_verified = True

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx,
    ) -> None:
        self.forward_batch_count += 1
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        value = _scalar(loss)
        if value is not None:
            self.loss_observation_count += 1
            self.all_losses_finite &= math.isfinite(value)
        else:
            self.all_losses_finite = False
        if self.track == "proposal_detector_recovery":
            components = getattr(pl_module, "_terminal_train_components", {})
            detector = _scalar(components.get("detector"))
            if detector is None:
                self.all_detector_losses_finite = False
            else:
                self.detector_loss_observation_count += 1
                self.all_detector_losses_finite &= math.isfinite(detector)
            language_values = [
                _scalar(components.get(name))
                for name in ("grounding", "trajectory", "stage2")
            ]
            if any(item is None or not math.isfinite(item) for item in language_values):
                self.all_losses_finite = False
            elif any(abs(item) > 1e-12 for item in language_values):
                self.language_loss_nonzero_batch_count += 1

    def on_after_backward(self, trainer, pl_module) -> None:
        self.backward_batch_count += 1
        checked = 0
        finite = True
        for name, parameter in pl_module.named_parameters():
            if parameter.grad is None:
                continue
            checked += 1
            if not bool(torch.isfinite(parameter.grad).all().item()):
                finite = False
            elif (
                self.track == "proposal_detector_recovery"
                and name.startswith("detector.")
                and bool((parameter.grad != 0).any().item())
            ):
                self.nonzero_detector_gradient_tensor_count += 1
        self.gradient_check_batch_count += 1
        self.gradient_tensor_check_count += checked
        self.all_gradients_finite &= finite and checked > 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        if self.track == "proposal_detector_recovery":
            expected = {
                id(parameter) for name, parameter in pl_module.named_parameters()
                if parameter.requires_grad and name.startswith("detector.")
            }
            actual = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            if not expected or actual != expected:
                raise ValueError(
                    "proposal recovery optimizer violates the detector-only contract"
                )
            self.optimizer_parameter_contract_verified = True
        self.optimizer_step_count += 1

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,
    ) -> None:
        if not trainer.sanity_checking:
            self.validation_batch_count += 1

    def on_test_start(self, trainer, pl_module) -> None:
        self.frozen_test_used_during_training = True

    @staticmethod
    def _selected_checkpoint(trainer) -> tuple[Path, bool]:
        checkpoint = getattr(trainer, "checkpoint_callback", None)
        best = str(getattr(checkpoint, "best_model_path", "") or "")
        last = str(getattr(checkpoint, "last_model_path", "") or "")
        selected = Path(best or last)
        return selected, bool(best)

    def on_fit_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        tuned_path, selected_by_validation = self._selected_checkpoint(trainer)
        if not tuned_path.is_file():
            raise ValueError(
                "formal training runtime could not resolve a saved tuned checkpoint"
            )
        device = str(pl_module.device)
        max_epochs = int(trainer.max_epochs)
        epoch_progress = getattr(
            getattr(getattr(trainer, "fit_loop", None), "epoch_progress", None),
            "current", None,
        )
        completed_value = getattr(epoch_progress, "completed", None)
        completed_epochs = (
            int(trainer.current_epoch) + 1
            if completed_value is None else int(completed_value)
        )
        interrupted = bool(getattr(trainer, "interrupted", False))
        gates = {
            "fit_completed": True,
            "configured_epochs_completed": (
                max_epochs > 0 and completed_epochs >= max_epochs
            ),
            "trainer_not_interrupted": interrupted is False,
            "gpu_device_is_cuda": device.startswith("cuda"),
            "positive_forward_batch_count": self.forward_batch_count > 0,
            "loss_observed_for_every_forward_batch": (
                self.loss_observation_count == self.forward_batch_count
            ),
            "positive_backward_batch_count": self.backward_batch_count > 0,
            "gradient_checked_for_every_backward_batch": (
                self.gradient_check_batch_count == self.backward_batch_count
                and self.gradient_tensor_check_count > 0
            ),
            "positive_optimizer_step_count": self.optimizer_step_count > 0,
            "optimizer_steps_match_lightning_global_step": (
                self.optimizer_step_count == int(trainer.global_step)
            ),
            "optimizer_steps_do_not_exceed_backward_batches": (
                self.optimizer_step_count <= self.backward_batch_count
            ),
            "backward_batches_do_not_exceed_forward_batches": (
                self.backward_batch_count <= self.forward_batch_count
            ),
            "all_losses_finite": self.all_losses_finite,
            "all_gradients_finite": self.all_gradients_finite,
            "train_only_validation_used": self.validation_batch_count > 0,
            "tuned_checkpoint_selected_by_train_only_validation": selected_by_validation,
            "frozen_test_used_during_training": (
                self.frozen_test_used_during_training is False
            ),
        }
        if self.track == "proposal_detector_recovery":
            gates.update({
                "proposal_or_detector_loss_observed_for_every_forward_batch": (
                    self.detector_loss_observation_count
                    == self.forward_batch_count
                    and self.all_detector_losses_finite
                ),
                "language_grounding_loss_zero_for_every_forward_batch": (
                    self.language_loss_nonzero_batch_count == 0
                ),
                "detector_only_parameter_contract_verified": (
                    self.detector_only_parameter_contract_verified
                    and self.trainable_parameter_count > 0
                ),
                "recovery_config_contract_verified": (
                    self.recovery_config_contract_verified
                ),
                "optimizer_parameter_contract_verified": (
                    self.optimizer_parameter_contract_verified
                ),
                "positive_nonzero_detector_gradient_count": (
                    self.nonzero_detector_gradient_tensor_count > 0
                ),
                "frozen_evaluation_used_during_recovery": (
                    self.frozen_test_used_during_training is False
                ),
                "language_grounding_loss_optimized": (
                    self.language_loss_nonzero_batch_count == 0
                ),
                "proposal_or_detector_loss_optimized": (
                    self.detector_loss_observation_count
                    == self.forward_batch_count
                    and self.detector_loss_observation_count > 0
                    and self.all_detector_losses_finite
                ),
            })
        emitter_path = Path(__file__).resolve()
        train_path = emitter_path.parents[2] / "train.py"
        report = {
            "schema_version": FORMAL_RUNTIME_SCHEMA_VERSION,
            "status": (
                (
                    "gpu_failure_mode_recovery_runtime_complete"
                    if self.track == "proposal_detector_recovery"
                    else "gpu_training_runtime_complete"
                )
                if all(gates.values())
                else (
                    "gpu_failure_mode_recovery_runtime_incomplete"
                    if self.track == "proposal_detector_recovery"
                    else "gpu_training_runtime_incomplete"
                )
            ),
            "model": self.model_name,
            "training_track": self.track,
            "fit_completed": True,
            "gpu_device": device,
            "fit_global_step": int(trainer.global_step),
            "configured_max_epochs": max_epochs,
            "completed_epoch_count": completed_epochs,
            "trainer_interrupted": interrupted,
            "forward_batch_count": self.forward_batch_count,
            "loss_observation_count": self.loss_observation_count,
            "backward_batch_count": self.backward_batch_count,
            "gradient_check_batch_count": self.gradient_check_batch_count,
            "gradient_tensor_check_count": self.gradient_tensor_check_count,
            "nonzero_detector_gradient_tensor_count": (
                self.nonzero_detector_gradient_tensor_count
            ),
            "optimizer_step_count": self.optimizer_step_count,
            "validation_batch_count": self.validation_batch_count,
            "proposal_or_detector_loss_batch_count": (
                self.detector_loss_observation_count
            ),
            "language_grounding_loss_batch_count": (
                self.language_loss_nonzero_batch_count
            ),
            "trainable_parameter_count": self.trainable_parameter_count,
            "detector_only_parameter_contract_verified": (
                self.detector_only_parameter_contract_verified
            ),
            "recovery_config_contract_verified": (
                self.recovery_config_contract_verified
            ),
            "optimizer_parameter_contract_verified": (
                self.optimizer_parameter_contract_verified
            ),
            "all_losses_finite": self.all_losses_finite,
            "all_detector_losses_finite": self.all_detector_losses_finite,
            "all_gradients_finite": self.all_gradients_finite,
            "train_only_validation_used": self.validation_batch_count > 0,
            "tuned_checkpoint_selected_by_train_only_validation": selected_by_validation,
            "frozen_test_used_during_training": self.frozen_test_used_during_training,
            "frozen_evaluation_used_during_recovery": (
                self.frozen_test_used_during_training
            ),
            "language_grounding_loss_optimized": (
                self.language_loss_nonzero_batch_count > 0
            ),
            "proposal_or_detector_loss_optimized": (
                self.detector_loss_observation_count == self.forward_batch_count
                and self.detector_loss_observation_count > 0
            ),
            "runtime_gates": gates,
            "provenance": {
                "emitter": _artifact(emitter_path),
                "train_entrypoint": _artifact(train_path),
                "baseline_checkpoint": _artifact(self.baseline_path),
                "tuned_checkpoint": _artifact(tuned_path),
            },
        }
        if self.track == "query_conditioned_grounding":
            report["provenance"].update({
                "training_bundle_manifest": _artifact(self.bundle_path),
                "trainer_preflight": _artifact(self.preflight_path),
            })
        else:
            report["provenance"].update({
                "recovery_plan_manifest": _artifact(self.recovery_plan_path),
                "recovery_data_audit_manifest": _artifact(
                    self.recovery_data_audit_path
                ),
                "recovery_training_bundle_manifest": _artifact(
                    self.recovery_bundle_path
                ),
                "recovery_trainer_preflight": _artifact(
                    self.recovery_preflight_path
                ),
            })
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
