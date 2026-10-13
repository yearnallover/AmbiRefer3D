#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${FRONT_DIR:?Set FRONT_DIR}"
: "${FUTURE_FOLDER:?Set FUTURE_FOLDER}"
: "${FRONT_TEXTURE_ROOT:?Set FRONT_TEXTURE_ROOT}"
: "${FRONT_MODEL_INFO:?Set FRONT_MODEL_INFO}"
: "${CC_MATERIAL_ROOT:?Set CC_MATERIAL_ROOT}"
: "${SYNTHETIC_OUTPUT_ROOT:?Set SYNTHETIC_OUTPUT_ROOT}"
export AMBIREFER3D_SYNTHETIC_RENDER_ROOT="${AMBIREFER3D_SYNTHETIC_RENDER_ROOT:-$SYNTHETIC_OUTPUT_ROOT/render_multiview}"
blenderproc run data_construction/synthetic/generate_dataset.py \
  --front_dir "$FRONT_DIR" --future_folder "$FUTURE_FOLDER" \
  --front_3D_texture_path "$FRONT_TEXTURE_ROOT" --model_info_json "$FRONT_MODEL_INFO" \
  --output_root "$SYNTHETIC_OUTPUT_ROOT" --cc_material_folder "$CC_MATERIAL_ROOT" \
  --text_model "${AMBIREFER3D_TEXT_MODEL:-qwen-llm}" \
  --color_model "${AMBIREFER3D_VISION_MODEL:-qwen-vl}" \
  --num_splits "${NUM_SPLITS:-1}" --split "${SPLIT_INDEX:-0}" \
  --max_scenes "${MAX_SCENES:-0}" "$@"
