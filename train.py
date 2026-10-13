import os
import json
import torch
import hydra
import lightning.pytorch as pl
from m3drefclip.data.data_module import DataModule
from lightning.pytorch.callbacks import LearningRateMonitor
from m3drefclip.callback.gpu_cache_clean_callback import GPUCacheCleanCallback
from m3drefclip.callback.lr_decay_callback import LrDecayCallback
from m3drefclip.callback.epoch_terminal_report_callback import EpochTerminalReportCallback
from m3drefclip.callback.formal_training_runtime_callback import FormalTrainingRuntimeCallback
from stage2_refactor.learning.constraint_table import base_constraint_key


def _nested_get(mapping, *keys, default=None):
    value = mapping
    for key in keys:
        if value is None:
            return default
        if isinstance(value, dict):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
    return default if value is None else value


def _copy_named_rows(target, source, target_ids, source_ids):
    copied = 0
    result = target.clone()
    for name, target_idx in target_ids.items():
        source_idx = source_ids.get(name)
        if source_idx is None:
            continue
        if target_idx >= result.shape[0] or source_idx >= source.shape[0]:
            continue
        result[target_idx].copy_(source[source_idx])
        copied += 1
    return result, copied


def _copy_constraint_rows(target, source, target_ids, source_ids):
    """Map v2 cardinality-aware constraints to exact or legacy semantic rows."""
    copied = 0
    result = target.clone()
    for name, target_idx in target_ids.items():
        source_idx = source_ids.get(name)
        if source_idx is None:
            source_idx = source_ids.get(base_constraint_key(name))
        if source_idx is None:
            continue
        if target_idx >= result.shape[0] or source_idx >= source.shape[0]:
            continue
        result[target_idx].copy_(source[source_idx])
        copied += 1
    return result, copied


def _copy_vocab_rows(target, source, target_ids, source_ids):
    """Use the old unknown-token embedding for vocabulary added after training."""
    copied = 0
    result = target.clone()
    fallback_idx = source_ids.get("<unk>")
    for name, target_idx in target_ids.items():
        source_idx = source_ids.get(name, fallback_idx)
        if source_idx is None:
            continue
        if target_idx >= result.shape[0] or source_idx >= source.shape[0]:
            continue
        result[target_idx].copy_(source[source_idx])
        copied += 1
    return result, copied


def _remap_stage2_inventory(model, checkpoint, state_dict, ckpt_path):
    stage2 = getattr(model, "stage2", None)
    if stage2 is None:
        return []

    cfg = _nested_get(checkpoint, "hyper_parameters", "cfg", default={})
    old_stage2_cfg = _nested_get(cfg, "model", "stage2", default={})
    old_round_path = _nested_get(old_stage2_cfg, "round_json_path", default="")
    if old_round_path and not os.path.isabs(str(old_round_path)):
        old_round_path = os.path.abspath(str(old_round_path))
    if not old_round_path or not os.path.exists(str(old_round_path)):
        return []

    with open(old_round_path, "r", encoding="utf-8") as handle:
        old_samples = json.load(handle)
    old_vocab = stage2._build_vocab(type(stage2.vocab), old_samples)
    old_rel2id = dict(_nested_get(old_stage2_cfg, "rel2id", default={}) or {
        "near": 0,
        "behind": 1,
        "in_front_of": 2,
        "left_of": 3,
        "right_of": 4,
        "next_to": 5,
    })
    old_attr2id = dict(_nested_get(old_stage2_cfg, "attr2id", default={}) or {
        "color": 0,
        "material": 1,
        "shape": 2,
        "size": 3,
        "height": 4,
        "other": 5,
    })
    old_table = type(stage2.constraint_table).build_from_samples(
        old_samples, rel2id=old_rel2id, attr2id=old_attr2id
    )
    model_state = model.state_dict()
    reports = []

    for suffix in ("qenc.wemb.weight", "cenc.wemb.weight"):
        key = f"stage2.net.{suffix}"
        if key not in state_dict or key not in model_state:
            continue
        remapped, copied = _copy_vocab_rows(
            model_state[key], state_dict[key], stage2.vocab.stoi, old_vocab.stoi
        )
        state_dict[key] = remapped
        reports.append(f"{key}: copied {copied}/{len(stage2.vocab.itos)} tokens")

    named_embeddings = (
        ("stage2.net.cenc.e_rel.weight", stage2.constraint_table.rel2id, old_rel2id),
        ("stage2.net.cenc.e_attr.weight", stage2.constraint_table.attr2id, old_attr2id),
    )
    for key, new_ids, old_ids in named_embeddings:
        if key not in state_dict or key not in model_state:
            continue
        remapped, copied = _copy_named_rows(
            model_state[key], state_dict[key], new_ids, old_ids
        )
        state_dict[key] = remapped
        reports.append(f"{key}: copied {copied}/{len(new_ids)} named rows")

    for suffix in ("weight", "bias"):
        key = f"stage2.net.selector.out.{suffix}"
        if key not in state_dict or key not in model_state:
            continue
        remapped, copied = _copy_constraint_rows(
            model_state[key],
            state_dict[key],
            stage2.constraint_table.stoi,
            old_table.stoi,
        )
        state_dict[key] = remapped
        reports.append(
            f"{key}: copied {copied}/{stage2.num_constraints} constraint keys"
        )

    semantic_word_key = "stage2.net.selector.constraint_word.weight"
    old_word_key = "stage2.net.qenc.wemb.weight"
    semantic_source_key = (
        semantic_word_key if semantic_word_key in state_dict else old_word_key
    )
    if semantic_word_key in model_state and semantic_source_key in state_dict:
        remapped, copied = _copy_vocab_rows(
            model_state[semantic_word_key],
            state_dict[semantic_source_key],
            stage2.vocab.stoi,
            old_vocab.stoi,
        )
        state_dict[semantic_word_key] = remapped
        reports.append(
            f"{semantic_word_key}: remapped {copied}/{len(stage2.vocab.itos)} tokens "
            f"from {semantic_source_key}"
        )

    # The frozen DPO reference must exactly mirror the remapped policy model.
    # It is recreated by refresh_reference_model() after loading.
    for key in list(state_dict):
        if key.startswith("stage2.reference_net."):
            del state_dict[key]
    return reports


def load_initial_weights(model, ckpt_path, strict=False):
    if ckpt_path is None:
        return
    assert os.path.exists(ckpt_path), "Error: init_from_ckpt path does not exist."
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    remap_reports = _remap_stage2_inventory(
        model, checkpoint, state_dict, ckpt_path
    ) if isinstance(checkpoint, dict) else []
    reset_decision_heads = bool(
        _nested_get(
            getattr(model, "hparams", {}),
            "cfg", "model", "stage2", "reset_decision_heads_on_init",
            default=False,
        )
    )
    reset_head_keys = []
    if reset_decision_heads:
        reset_prefixes = (
            "stage2.net.stop.",
            "stage2.net.mask.out.4.",
            "stage2.reference_net.stop.",
            "stage2.reference_net.mask.out.4.",
        )
        for key in list(state_dict):
            if key.startswith(reset_prefixes):
                reset_head_keys.append(key)
                del state_dict[key]
    skipped_shape_keys = []
    expanded_shape_keys = []
    if not bool(strict):
        model_state = model.state_dict()
        compatible_state = {}
        for key, value in state_dict.items():
            if key not in model_state:
                compatible_state[key] = value
                continue
            if model_state[key].shape != value.shape:
                can_expand_stage2_input = (
                    key == "stage2.net.cand.stage1_mlp.0.weight"
                    and value.ndim == 2
                    and model_state[key].ndim == 2
                    and value.shape[0] == model_state[key].shape[0]
                    and value.shape[1] < model_state[key].shape[1]
                )
                if can_expand_stage2_input:
                    expanded = torch.zeros_like(model_state[key])
                    expanded[:, : value.shape[1]].copy_(value)
                    compatible_state[key] = expanded
                    expanded_shape_keys.append(
                        f"{key}: copied {value.shape[1]}/{model_state[key].shape[1]} input channels"
                    )
                    continue
                skipped_shape_keys.append(
                    f"{key}: checkpoint={tuple(value.shape)} model={tuple(model_state[key].shape)}"
                )
                continue
            compatible_state[key] = value
        state_dict = compatible_state
    load_result = model.load_state_dict(state_dict, strict=bool(strict))
    missing_keys = getattr(load_result, "missing_keys", [])
    unexpected_keys = getattr(load_result, "unexpected_keys", [])
    print(f"Initialized model weights from: {ckpt_path}")
    for report in remap_reports:
        print(f"Remapped Stage2 inventory: {report}")
    if missing_keys:
        print(f"Missing keys while loading init weights: {len(missing_keys)}")
        for key in missing_keys[:10]:
            print(f"  {key}")
    if unexpected_keys:
        print(f"Unexpected keys while loading init weights: {len(unexpected_keys)}")
        for key in unexpected_keys[:10]:
            print(f"  {key}")
    if skipped_shape_keys:
        print(f"Skipped shape-mismatched init weights: {len(skipped_shape_keys)}")
        for item in skipped_shape_keys[:10]:
            print(f"  {item}")
    if expanded_shape_keys:
        print(f"Expanded Stage2 input projections: {len(expanded_shape_keys)}")
        for item in expanded_shape_keys:
            print(f"  {item}")
    if reset_head_keys:
        print(
            "Reset Stage2 stop/filter decision outputs for the new trajectory "
            f"schema: {len(reset_head_keys)} tensors"
        )
    if getattr(model, "stage2", None) is not None and hasattr(model.stage2, "refresh_reference_model"):
        model.stage2.refresh_reference_model()


def init_callbacks(cfg, logger):
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    gpu_cache_clean_monitor = GPUCacheCleanCallback()
    lr_decay_callback = LrDecayCallback()
    callbacks = [
        checkpoint_monitor,
        gpu_cache_clean_monitor,
        lr_decay_callback,
        EpochTerminalReportCallback(),
    ]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    runtime_cfg = cfg.get("formal_training_runtime")
    if runtime_cfg is not None and bool(runtime_cfg.get("enabled", False)):
        callbacks.append(FormalTrainingRuntimeCallback(cfg))
    return callbacks


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.train_seed, workers=True)

    # create directories for training outputs
    os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)
    # print(cfg.data)
    # exit()
    # initialize data
    data_module = DataModule(cfg.data)

    # initialize model
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)

    if bool(cfg.get("reset_seed_after_model_init", False)):
        # Decouple DataLoader/augmentation randomness from the number of random
        # parameters consumed while constructing different model variants.
        pl.seed_everything(cfg.train_seed, workers=True)

    # load the pre-trained detector
    if "detector_path" in cfg:
        detector_weights = torch.load(cfg.detector_path, weights_only=False)["state_dict"]
        model.detector.load_state_dict(detector_weights)

    load_initial_weights(model, cfg.init_from_ckpt, cfg.init_from_ckpt_strict)

    # initialize logger
    logger = hydra.utils.instantiate(cfg.logger) if cfg.get("logger") is not None else False

    # initialize callbacks
    callbacks = init_callbacks(cfg, logger)

    # initialize trainer
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.trainer)

    # check the checkpoint
    if cfg.ckpt_path is not None:
        assert os.path.exists(cfg.ckpt_path), "Error: Checkpoint path does not exist."

    # start training
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    main()
