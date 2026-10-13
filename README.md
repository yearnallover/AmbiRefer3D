# AmbiRefer3D: 3D Visual Grounding with Referential Ambiguity [ICML2026]

[[paper]](https://openreview.net/pdf?id=tkOMvqB6E9)

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

## DataSet

### Data and Path Configuration

No point clouds, licensed 3D assets, generated annotations, features, model
weights, logs, or original Git history are bundled. Required artifacts and
their schemas are listed in [Data Construction](docs/DATA_CONSTRUCTION.md).

Configure your dataset, feature-cache and checkpoint paths before running the
construction, training or validation scripts:

```bash
cp .env.example .env
# Fill the paths in .env using your own data and checkpoints.
set -a
source .env
set +a
```

Keep `.env` outside Git. Use scene-disjoint official training/validation splits
for both stages.

### Real and Synthetic Data Construction

Real ScanNet scene query generation and clarification are provided under
`data_construction/real/`. The launcher requires upstream scene graphs,
existing object renders, input descriptions and configured text/VLM services:

```bash
bash scripts/construct_ambirefer3d_real.sh
```

Synthetic 3D-FRONT scene processing, query generation/degradation and visual
clarification are provided under `data_construction/synthetic/`. Configure the
licensed scene/model/texture assets and BlenderProc before running:

```bash
bash scripts/construct_ambirefer3d_synthetic.sh
```

See [Data Construction](docs/DATA_CONSTRUCTION.md) for the full input/output
contracts, sharding options and remaining construction requirements.

### Stage 2 Round Data

Convert training and validation dialogue metadata into separate round datasets:

```bash
python -m stage2_refactor.data.convert_dialogue_json_semantic \
  --scene_json /path/to/train_dialogues.json \
  --out_round_json /path/to/train_rounds.json --seed 0
python -m stage2_refactor.data.convert_dialogue_json_semantic \
  --scene_json /path/to/val_dialogues.json \
  --out_round_json /path/to/val_rounds.json --seed 0
```

Stage 2 additionally needs a candidate image-feature cache. Its preparation
and candidate-identity requirements are described in the data construction guide.
The training and validation launchers check candidate IDs and cached image
features before running.

## Training

### Stage 1: MT Fine-tuning

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

### Stage 2: Interactive Disambiguation

Train Stage 2 using the prepared round datasets and candidate image features:

```bash
STAGE2_TRAIN_JSON=/path/to/train_rounds.json \
STAGE2_VAL_JSON=/path/to/val_rounds.json \
STAGE2_IMG_CACHE=/path/to/imgfeat_cache \
bash scripts/train_ambirefer3d_stage2.sh
```

The launcher freezes the vocabulary/constraint inventory in `spec.json` built
from training rows; the validation split does not define trainable constraint
classes. Preserve this specification together with the trained checkpoint.

Advanced trajectory/joint modules from the current worktree are retained as
optional implementation code; the baseline launchers keep them disabled.

## Validation

### Stage 1

```bash
STAGE1_CKPT=/path/to/trained_stage1.ckpt \
bash scripts/eval_ambirefer3d_stage1.sh
```

This exports Stage 1 metrics and predictions. Metrics include F1, precision
and recall at IoU 0.25/0.5, target survival with IoU, and target survival without
IoU when point-instance assignments and intended-target metadata are available.
Empty non-MT subgroup metrics are undefined on an MT-only split.

### Stage 2

Evaluate with the same training specification and architecture:

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


## Bibtex
```
@inproceedings{zhuambirefer3d,
  title={AmbiRefer3D: 3D Visual Grounding with Referential Ambiguity},
  author={Zhu, Rongjiang and Kang, Wei and Liu, Zeqi and Yang, Shuo and Wu, Xinxiao and others},
  booktitle={Forty-third International Conference on Machine Learning}
}
```