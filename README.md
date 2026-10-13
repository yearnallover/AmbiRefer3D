# ambirefer3d

AmbiRefer3D addresses interactive 3D visual grounding with ambiguous language.
The baseline has two separately trained stages: Stage 1 fine-tunes a grounding
backbone on ambiguous multi-target (MT) annotations; Stage 2 selects clarification
constraints, filters candidates, and predicts when to stop. This release also
contains the real-scene and synthetic-scene data construction code.

The public project and launchers use `ambirefer3d`. The internal `m3drefclip`
module and `Multi3DRefer` dataset identifier retain their upstream names for
Python imports, checkpoint compatibility, and attribution.

| Component | Location |
|---|---|
| Stage 1 backbone, losses and evaluation | `m3drefclip/`, `train.py`, `test.py` |
| Stage 2 semantic constraint model | `stage2_refactor/` |
| Real scene ambiguous query generation and clarification | `data_construction/real/` |
| Real object/candidate rendering and annotation conversion | `data_construction/real/object_render/`, `data_construction/real/converters/` |
| 3D-FRONT generation, degradation and visual clarification | `data_construction/synthetic/generate_dataset.py` |
| Data contracts and known gaps | `docs/DATA_CONSTRUCTION.md`, `docs/RELEASE_AUDIT.md` |
| GitHub publication and source audit | `GITHUB_RELEASE.md`, `tools/audit_release.py` |

## Environment

The existing local training environment is `conda activate m3drefclip`. On a new
machine, create the public environment and compile the CUDA dependencies:

```bash
conda env create -f environment.yml
conda activate ambirefer3d
# Install matching MinkowskiEngine and PyTorch3D builds first.
pip install -e .
pip install -v ./m3drefclip/common_ops
```

The environment file follows the existing Python 3.10 / PyTorch 2.1 / CUDA 11.8
training setup. MinkowskiEngine and PyTorch3D are external compiled dependencies.
The synthetic generator runs through BlenderProc in its Blender environment;
see `data_construction/synthetic/requirements.txt` and the data construction guide.
The LLM/VLM clients require the `openai` package and an OpenAI-compatible server.

## Configure External Artifacts

```bash
cp .env.example .env
# Fill the paths in .env using your own data and checkpoints.
set -a
source .env
set +a
```

No point clouds, licensed 3D assets, generated annotations, features, model
weights, logs, or original Git history are bundled. Required artifacts are
listed in `docs/DATA_CONSTRUCTION.md`. Do not commit your `.env` file.

## Stage 1: MT Fine-tuning

Stage 1 uses all query-compatible annotated object IDs as its positive set,
while `true_target_id` records the user's intended object for Stage 2. Training
on the intended target alone changes the task and is not this MT baseline.

```bash
INIT_CKPT=/path/to/pretrained_grounding.ckpt \
MT_SOURCE_TRAIN_JSON=/path/to/ambiguous_train.json \
MT_SOURCE_VAL_JSON=/path/to/ambiguous_val.json \
bash scripts/train_ambirefer3d_stage1.sh
```

The launcher filters `eval_type=mt`, then disables Stage 2, the trajectory
adapter, online trajectory supervision and post-training objectives. It loads
initial weights using `init_from_ckpt`; `ckpt_path` is reserved for resuming the
same experiment. `MAX_EPOCHS`, `DEVICES`, `BATCH_SIZE`, `EXP_NAME`, and
`AMBIREFER3D_OUTPUT_ROOT` control runtime and output placement. Extra Hydra
overrides can be appended to the command.

```bash
STAGE1_CKPT=/path/to/trained_stage1.ckpt \
bash scripts/eval_ambirefer3d_stage1.sh
```

This exports Stage 1 metrics and predictions. Metrics include F1, precision
and recall at IoU 0.25/0.5, target survival with IoU, and target survival without
IoU when point-instance assignments and intended-target metadata are available.
Empty non-MT subgroup metrics are undefined on an MT-only split.

## Stage 2: Interactive Disambiguation

Generate round rows from dialogue metadata, then validate candidates and image
features before training:

```bash
python -m stage2_refactor.data.convert_dialogue_json_semantic \
  --scene_json /path/to/train_dialogues.json \
  --out_round_json /path/to/train_rounds.json --seed 0
python -m stage2_refactor.data.convert_dialogue_json_semantic \
  --scene_json /path/to/val_dialogues.json \
  --out_round_json /path/to/val_rounds.json --seed 0

STAGE2_TRAIN_JSON=/path/to/train_rounds.json \
STAGE2_VAL_JSON=/path/to/val_rounds.json \
STAGE2_IMG_CACHE=/path/to/imgfeat_cache \
bash scripts/train_ambirefer3d_stage2.sh
```

Use scene-disjoint official training/validation JSON files. The launcher freezes
the vocabulary/constraint inventory in `spec.json` built from training rows;
the validation split does not define trainable constraint classes. Evaluate
with the same specification and architecture:

```bash
STAGE2_CKPT=/path/to/stage2.pt \
STAGE2_SPEC=/path/to/spec.json \
STAGE2_VAL_JSON=/path/to/val_rounds.json \
STAGE2_IMG_CACHE=/path/to/imgfeat_cache \
bash scripts/eval_ambirefer3d_stage2.sh
```

Evaluation uses agent selection from the first round. `acc` and `success@k`
include rows whose target was absent initially; `conditional_acc` uses only
rows with an initially surviving target. A GT-derived candidate list can make
initial survival equal to one; it must be labelled as an oracle-candidate
experiment, not an end-to-end Stage 1 result. The release guide describes the
additional mapping needed for actual backbone proposals.

## Source Validation

```bash
python tools/audit_release.py
python -m unittest discover -s tests -p test_release_contracts.py
```

`docs/RELEASE_AUDIT.md` distinguishes corrected packaging issues, external
requirements, and unresolved reproduction gaps. Advanced trajectory/joint
modules from the current worktree are retained as optional implementation
code; the baseline launchers keep them disabled.

## Attribution

Stage 1 extends the Multi3DRefer / M3DRef-CLIP implementation. Its original
MIT copyright and license are retained in `LICENSE`. BlenderProc is an external
dependency; its license is included alongside the synthetic construction code.
See `THIRD_PARTY_NOTICES.md`. Preserve these notices when publishing.
