# AmbiRefer3D Annotations and Construction

The downloadable annotation package contains real ScanNet data in JSON format.
The package root is `ambirefer3d_annotations/`. 3D-FRONT JSON files are excluded
from this download; the synthetic construction code remains available.

```text
ambirefer3d_annotations/
  manifest.json
  real/
    stage1/                 # ambiguous grounding annotations
    dialogues/              # clarification questions and restored expressions
    stage2/                 # per-constraint supervised round rows
  splits/                   # real-scene train/val ID lists
```

`manifest.json` lists each file's SHA256, number of rows and number of scenes.
Use the package files with the paths shown in `.env.example` and the README.
The additional `*_mt.json` files provide consistent MT subsets. The manifest
records their filtering rules and counts, while the original JSON labels remain
available unchanged.

## Grounding Annotations

Stage 1 rows contain `scene_id`, `ann_id`, `description`, `object_ids` and
`eval_type`. `object_ids` lists objects compatible with the ambiguous expression;
`true_target_id` identifies the intended target when available. MT fine-tuning
uses `eval_type=mt` and supervises the compatible set.

## Disambiguation Annotations

Real-scene dialogue records contain the initial query, removed information,
compatible candidates, a clarification question and a restored expression.
The variants `minimal`, `no_spatial` and `no_visual` define different ambiguity
settings. The supplied round converter preserves the candidate set and seeds
candidate shuffling for reproducibility.

Stage 2 rows use `query`, `cand_ids`, `target_index`, `constraint_gt` and `stop_y`.
Use `target_index=-1` when the target is absent. The feature cache must contain
finite `[N,768]` image features keyed by the same candidate IDs. Geometry-enabled
rows also supply `[C,7]` candidate boxes: center XYZ, size XYZ and semantic ID.

```bash
python -m stage2_refactor.precompute_imgfeat \
  --json_path /path/to/rounds.json --render_root /path/to/object_renders \
  --out_dir /path/to/imgfeat_cache
python tools/validate_stage2_inputs.py \
  --round_json /path/to/rounds.json --img_cache /path/to/imgfeat_cache
```

GT/dialogue-derived candidates and backbone-generated proposals have different
evaluation settings. For backbone proposals, retain their geometry, candidate
identities and matching image crops. Determine target survival through the chosen
IoU or point-instance criterion and retain missed queries in the denominator.
The candidate conversion helper requires explicit IDs rather than inferring
identity from the number of boxes.

## Real Scene Construction

`data_construction/real/` contains query degradation, scene-object filtering,
clarification generation, object rendering helpers and grounding format
converters. Configure input descriptions, upstream `objects.json` and
`relationships.json`, object renders and annotated cameras in `.env`.

```bash
pip install -r data_construction/real/requirements.txt
bash scripts/construct_ambirefer3d_real.sh
```

The supplied rendering helpers require completed object-render methods when
regenerating images from meshes. Existing renders can be used directly. The
text/VLM API endpoints and served model names are configurable through the
environment.

## Synthetic Scene Construction

`data_construction/synthetic/generate_dataset.py` runs through BlenderProc.
It loads 3D-FRONT scenes and 3D-FUTURE models, prepares scene geometry and
graphs, generates expressions, removes visual/spatial information, renders
candidate views and generates clarification metadata.

```bash
bash scripts/construct_ambirefer3d_synthetic.sh
MAX_SCENES=1 bash scripts/construct_ambirefer3d_synthetic.sh
NUM_SPLITS=4 SPLIT_INDEX=0 bash scripts/construct_ambirefer3d_synthetic.sh
```

Supply scene JSONs, model meshes, textures, CC materials, BlenderProc and
configured model services. The generator's internal construction stages
describe data preparation, whereas model Stage 1 and Stage 2 describe grounding
and disambiguation. Exported geometry arrays must be converted to the `.pth`
scene schema; multiview HDF5 features must align with the scene points.
