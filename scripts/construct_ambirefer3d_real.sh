#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${AMBIREFER3D_REAL_INPUT_JSON:?Set input descriptions JSON}"
: "${AMBIREFER3D_REAL_SCENEGRAPH_ROOT:?Set upstream objects.json/relationships.json root}"
: "${AMBIREFER3D_REAL_RENDER_ROOT:?Set existing object renders root}"
export AMBIREFER3D_REAL_AMBIGUOUS_JSONL="${AMBIREFER3D_REAL_AMBIGUOUS_JSONL:-data/real/ambiguous_queries.jsonl}"
export AMBIREFER3D_REAL_AMBIGUOUS_JSON="${AMBIREFER3D_REAL_AMBIGUOUS_JSON:-data/real/ambiguous_queries.json}"
export AMBIREFER3D_REAL_FILTERED_JSON="${AMBIREFER3D_REAL_FILTERED_JSON:-data/real/filtered_queries.json}"
export AMBIREFER3D_REAL_DIALOGUES_JSONL="${AMBIREFER3D_REAL_DIALOGUES_JSONL:-data/real/dialogues.jsonl}"
mkdir -p "$(dirname "$AMBIREFER3D_REAL_AMBIGUOUS_JSONL")" "$(dirname "$AMBIREFER3D_REAL_FILTERED_JSON")"
python -m data_construction.real.generate_ambiguous_queries
python tools/merge_jsonl.py --input "$AMBIREFER3D_REAL_AMBIGUOUS_JSONL" --output "$AMBIREFER3D_REAL_AMBIGUOUS_JSON"
python -m data_construction.real.filter_scene_objects
python -m data_construction.real.generate_clarifications --num_splits 1 --split 0
python tools/merge_jsonl.py --input "$AMBIREFER3D_REAL_DIALOGUES_JSONL" --output "${REAL_DIALOGUES_JSON:-data/real/dialogues.json}"
