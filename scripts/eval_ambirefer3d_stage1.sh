#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${STAGE1_CKPT:?Set STAGE1_CKPT}"
: "${MT_SOURCE_VAL_JSON:?Set MT_SOURCE_VAL_JSON}"
OUT="${AMBIREFER3D_OUTPUT_ROOT:-output}"
[[ "$OUT" = /* ]] || OUT="$ROOT/$OUT"
mkdir -p "$OUT/ambirefer3d_metadata"
python dataset/multi3drefer/metadata/make_mt_json.py --input "$MT_SOURCE_VAL_JSON" --output "$OUT/ambirefer3d_metadata/val_mt.json"
python validate_multi_ckpts.py --ckpt "$STAGE1_CKPT" \
  --out_dir "$OUT/ambirefer3d_stage1_eval" --devices 1 --export_query_metrics \
  data=multi3drefer "data.lang_metadata.val_language_data=$OUT/ambirefer3d_metadata/val_mt.json" \
  data.dataloader.batch_size=1 "$@"
python test.py data=multi3drefer "ckpt_path=$STAGE1_CKPT" \
  "pred_path=$OUT/ambirefer3d_stage1_predictions" \
  "data.lang_metadata.val_language_data=$OUT/ambirefer3d_metadata/val_mt.json" \ logger=null
