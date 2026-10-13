# AmbiRefer3D Data Construction

The baseline model stages and the construction stages have different meanings.
Model Stage 1 means MT backbone fine-tuning, and model Stage 2 means interactive
constraint selection/filtering. The synthetic generator's internal `Stage0`,
`Stage1_2`, and `Stage3` mean scene processing, language generation/degradation,
and visual clarification, respectively.

## Real ScanNet Scenes

The release includes the provided query-degradation, object filtering, and
sharded clarification code in `data_construction/real/`. It also includes the
object renderer and annotation converters found outside the original pipeline
directory, because the pipeline needs their outputs.

| Step | Input | Output |
|---|---|---|
| Upstream scene graph preparation | ScanNet aligned instances and semantic relations | `scenegraph/<scene>/objects.json` and `relationships.json` |
| Optional object rendering | Scene mesh, object boxes, description CSV | Multi-view object JPGs and camera metadata |
| Query degradation | ScanRefer-style descriptions | JSONL with `scene_id`, `object_id`, `ann_id`, `variants` |
| JSONL conversion and filtering | Degraded queries and scene object labels | Filtered ambiguous-query JSON |
| Visual clarification | Queries, scene graph, object renders, optional annotated cameras | Dialogue JSONL with candidate compatibility and restored expressions |
| Grounding annotation conversion | Dialogue JSON plus instance labels | MT-compatible object IDs, intended target and query metadata |

Configure `.env`, provide existing upstream scene graphs and object renders,
and run from the repository root:

```bash
pip install -r data_construction/real/requirements.txt
bash scripts/construct_ambirefer3d_real.sh
```

The code calls your configured OpenAI-compatible text/VLM services. It does not
launch a model server. `AMBIREFER3D_TEXT_MODEL` and `AMBIREFER3D_VISION_MODEL`
must match the served model names. API credentials are read from the environment.

For manual sharding, each process writes an independent `_partN` JSONL file:

```bash
python -m data_construction.real.generate_clarifications --num_splits 4 --split 0
# Repeat split=1,2,3, then merge the four output files:
python tools/merge_jsonl.py --input /path/to/dialogues_part*.jsonl \
  --output /path/to/dialogues.json
```

The independent ambiguity variants (`minimal`, `no_spatial`, `no_visual`) are
different initial expressions. Their positions in a `dialogues` array do not
by themselves prove an ordered multi-turn interaction trajectory.

The supplied real-scene directories do not contain the original upstream scene
graph builder. Supply those artifacts separately. The legacy object renderer
also references `objects_list`, `render_object`, and `render_scene_overview`
that are absent from its provided renderer class. Its source is included for
traceability, but object-render regeneration needs those missing implementations;
the supported real construction launcher starts from existing object renders.

## Synthetic 3D-FRONT Scenes

`data_construction/synthetic/generate_dataset.py` is the supplied end-to-end
generator under a public descriptive filename. It imports BlenderProc/Blender,
builds scene graphs, exports point-cloud arrays, generates detailed descriptions,
removes attributes/relations to create ambiguity, renders multi-view candidates,
and asks the VLM for clarification.

External requirements are 3D-FRONT scene JSONs, 3D-FUTURE meshes/model info,
3D-FRONT textures, CC materials, BlenderProc and the text/VLM services.

```bash
bash scripts/construct_ambirefer3d_synthetic.sh
# Debug one scene without changing code:
MAX_SCENES=1 bash scripts/construct_ambirefer3d_synthetic.sh
# Process one of four disjoint shards:
NUM_SPLITS=4 SPLIT_INDEX=0 bash scripts/construct_ambirefer3d_synthetic.sh
```

The release removes the hard-coded one-scene slice and makes sharding effective.
The provided VLM clarification branch currently generates one clarification per
variant; the existing `--max_rounds` argument does not make it a recursive
multi-round dialogue constructor. A true ordered dialogue dataset needs repeated
state/query/feedback generation and termination verification.

The generator exports geometry arrays. Stage 1 additionally needs the repo's
`.pth` scene contract and, when multiview features are enabled, correctly aligned
ENet HDF5 features. Utilities for array-to-`.pth` conversion are retained under
`tools/` and `dataset/scannetv2/`. The legacy synthetic conversion helpers are
retained under `data_construction/synthetic/converters/`; their incomplete
ENet/image-loading placeholders are not a replacement for feature preparation.

## Stage 1 to Stage 2 Contract

MT Stage 1 annotations must contain `scene_id`, `ann_id`, `description`,
`object_ids`, and `eval_type=mt`. `object_ids` is the compatible object set;
`true_target_id` is the single intended target when known. Keep these distinct.

Stage 2 round samples require:

```json
{
  "scene_id": "scene0000_00",
  "ann_id": "0",
  "round": 0,
  "query": "the chair",
  "cand_ids": [4, 5],
  "target_id": 5,
  "target_index": 1,
  "constraint_gt": {"ctype": "REL", "rel": "near", "landmark_text": "table"},
  "stop_y": 0
}
```

For a Stage 1 miss, set `target_index=-1`. Never insert the target into the
predicted candidate set to repair survival. Candidate image caches use
`<scene>.npz`, `obj_ids [N]`, and finite `feats [N,768]`. Each `cand_id` must have
an image feature. Geometry-enabled rows additionally require aligned
`cand_bboxes [C,7]` (center XYZ, size XYZ, semantic ID).

```bash
python -m stage2_refactor.precompute_imgfeat \
  --json_path /path/to/rounds.json --render_root /path/to/object_renders \
  --out_dir /path/to/imgfeat_cache
python tools/validate_stage2_inputs.py \
  --round_json /path/to/rounds.json --img_cache /path/to/imgfeat_cache
```

The semantic converter keeps explicit candidate IDs and aligned per-candidate
arrays, and applies deterministic candidate shuffling. Without predicted
candidate IDs, it can only create a GT/dialogue-derived candidate experiment.

The legacy `conver_real2train.py` approximated proposal identity by truncating
distractor IDs to the number of predicted boxes. This is not a proposal-to-object
mapping. The release rejects that operation unless explicit candidate IDs are
provided. For actual backbone proposals, export proposal geometry, associate
candidate identity/image crops consistently, determine target survival using
the selected IoU or point criterion, and retain unmatched/missed queries in
the end-to-end metric denominator. `test.py` exports Stage 1 boxes, but box-only
predictions alone cannot create that full Stage 2 contract.

Consequently report GT-candidate Stage 2 scores and end-to-end scores separately.
`conditional_acc` is accuracy on initially surviving-target queries; `acc` is
accuracy on all supplied Stage 2 queries. Neither is automatically the overall
Stage 1-query accuracy if the conversion has dropped queries.
