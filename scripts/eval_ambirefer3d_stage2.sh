#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${STAGE2_CKPT:?Set STAGE2_CKPT}"
: "${STAGE2_SPEC:?Set the training STAGE2_SPEC}"
: "${STAGE2_VAL_JSON:?Set STAGE2_VAL_JSON}"
: "${STAGE2_IMG_CACHE:?Set STAGE2_IMG_CACHE}"
OUT="${AMBIREFER3D_OUTPUT_ROOT:-output}/ambirefer3d_stage2_eval"
mkdir -p "$OUT"
python tools/validate_stage2_inputs.py --round_json "$STAGE2_VAL_JSON" --img_cache "$STAGE2_IMG_CACHE"
python -m stage2_refactor.train \
  --round_json_path "$STAGE2_VAL_JSON" --img_cache_dir "$STAGE2_IMG_CACHE" \
  --spec_path "$STAGE2_SPEC" --build_spec 0 --test_only 1 \
  --load_from_pretrained "$STAGE2_CKPT" --load_strict 1 \
  --out_ckpt "$OUT/unused.pt" --out_eval_json "$OUT/metrics.json" \
  --epochs 1 --eval_batch_size 64 --max_round "${MAX_ROUNDS:-6}" \
  --max_len 32 --use_bbox_geom "${USE_BBOX_GEOM:-0}" \
  --eval_mode agent --selector_strategy ours --ablation_setting full \
  --seed "${SEED:-0}" "$@"
