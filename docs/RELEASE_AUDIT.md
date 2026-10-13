# AmbiRefer3D Release Audit

This release follows the requested original baseline: MT backbone fine-tuning
for Stage 1 and independent semantic disambiguation for Stage 2. Data construction
sources are included alongside the model implementation. The experimental
joint/trajectory implementation remains in the core source as an optional
extension; the baseline launchers disable it.

## Corrected in the Release Copy

| Finding | Release action |
|---|---|
| Previous source package described joint trajectory training as the baseline | Rewrote README around the independent two-stage AmbiRefer3D method |
| Data construction existed outside the model repository | Included both real and synthetic construction, renderers and converters |
| Synthetic generator processed only the first scene and ignored sharding | Removed the one-scene slice; added effective `num_splits`, `split`, `max_scenes` |
| Real pipeline had fixed workstation paths and server addresses | Exposed supported paths, API endpoint/model and credentials as environment variables |
| Stage 2 used bare `utils/data/model` imports or obsolete `stage2` namespace | Normalized imports so `python -m stage2_refactor.train` works from repo root |
| Obsolete non-semantic modules referenced missing `ConstraintSpaces` | Excluded those three unused modules; retained the semantic model and converters |
| Stage 2 evaluated before each epoch's updates | Moved validation after training; checkpoint scores reflect the completed epoch |
| Best CPU checkpoint tensors shared mutable storage | Cloned tensors when recording the best state |
| Random row split could mix scenes/rounds across train and evaluation | Added explicit official scene-disjoint evaluation JSON and use it in the launcher |
| Validation vocabulary/constraint IDs could drift | Persist and reuse training spec; reject missing split keys |
| Converter shuffling could discard or misalign per-candidate arrays | Added seeded shuffling and reorder aligned candidate fields together |
| Proposal conversion truncated distractors based only on box count | Reject conversion without explicit candidate identities |
| Missing object images silently fell back to zero features | Added a preflight that requires every candidate's finite image feature |
| Public project naming exposed an internal experiment marker | Unified release branding and removed that marker from content and filenames |
| Packaging could include logs, weights, arrays, credentials or history | Created a source-only release with a fail-fast source audit and checksums |

The original working repository, existing datasets, checkpoints and earlier
source archive were not modified. The corrections above apply to the new
release copy only.

## Remaining Data and Reproduction Requirements

1. **Real scene graphs:** The provided real pipeline consumes `objects.json`
   and `relationships.json`; its upstream builder was not present in the supplied
   directories. Obtain the original graph-generation code or publish the graphs
   separately with their schema and license.
2. **Real object renderer:** The supplied `render_util.py` lacks `objects_list`,
   `render_object` and `render_scene_overview`, which its caller needs. Existing
   object renders support query construction and Stage 2 feature preparation;
   regeneration from raw meshes requires completing the renderer.
3. **True ordered trajectories:** Real `dialogues` variants and the synthetic
   generator's single visual clarification do not constitute a fully recursive
   ordered multi-turn dataset. This matters for any later trajectory-learning
   claims, but independent Stage 2 training can use the provided round contract.
4. **Actual proposal bridge:** A box-count-based distractor truncation cannot
   establish candidate identity or target survival. The exporter must retain all
   queries, actual proposal IDs/boxes and the corresponding image features.
   GT-derived candidate training/evaluation must be identified as such.
5. **Scene tensors and multiview features:** Licensed raw assets, `.pth` scene
   preprocessing, label maps and point-aligned HDF5 features remain external.
   Geometry export alone does not supply ENet features.
6. **Pretrained and trained weights:** Backbone, detector and Stage 2 weights
   are not bundled. Checkpoint paths are configurable; upload weights through
   a separate artifact host when ready.
7. **Additional backbones:** Adapters for BUTD-DETR, 3D-VisTA and TSP3D are
   retained, but their external upstream sources/checkpoints are not embedded.
   The documented default baseline is M3DRef-CLIP.
8. **LLM/VLM serving:** Data construction requires configured model services.
   No server weights or API credentials are bundled.

## Validation Scope

Source syntax, shell syntax, package metadata, canonical Stage 2 imports,
converter/validator regression tests, one-epoch CPU Stage 2 training and strict
checkpoint evaluation are checked during packaging. Generated verification
artifacts live outside the source tree. See `VERIFICATION.json` for the actual
results. Full GPU Stage 1 retraining and LLM/Blender dataset regeneration are
not part of this source-packaging validation.
