#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${INIT_CKPT:?Set INIT_CKPT to pretrained grounding weights}"
: "${MT_SOURCE_TRAIN_JSON:?Set MT_SOURCE_TRAIN_JSON}"
: "${MT_SOURCE_VAL_JSON:?Set MT_SOURCE_VAL_JSON}"
OUT="${AMBIREFER3D_OUTPUT_ROOT:-output}"
[[ "$OUT" = /* ]] || OUT="$ROOT/$OUT"
META="$OUT/ambirefer3d_metadata"
mkdir -p "$META"
python dataset/multi3drefer/metadata/make_mt_json.py --input "$MT_SOURCE_TRAIN_JSON" --output "$META/train_mt.json"
python dataset/multi3drefer/metadata/make_mt_json.py --input "$MT_SOURCE_VAL_JSON" --output "$META/val_mt.json"
python tools/validate_stage1_metadata.py --train "$META/train_mt.json" --val "$META/val_mt.json"
EXTRA=()
if [[ -n "${RESUME_CKPT:-}" ]]; then EXTRA+=("ckpt_path=$RESUME_CKPT"); fi
python train.py data=multi3drefer logger=null \
  "experiment_name=${EXP_NAME:-ambirefer3d_stage1}" \
  "experiment_output_path=$OUT/${EXP_NAME:-ambirefer3d_stage1}" \
  "init_from_ckpt=$INIT_CKPT" \
  "data.lang_metadata.train_language_data=$META/train_mt.json" \
  "data.lang_metadata.val_language_data=$META/val_mt.json" \
  "data.lang_metadata.test_language_data=$META/val_mt.json" \
  "trainer.max_epochs=${MAX_EPOCHS:-20}" \
  "trainer.check_val_every_n_epoch=${EVAL_EVERY:-5}" \
  "+trainer.devices=${DEVICES:-1}" \
  +trainer.strategy=ddp_find_unused_parameters_true \
  "data.dataloader.batch_size=${BATCH_SIZE:-4}" \
  "${EXTRA[@]}" "$@"
