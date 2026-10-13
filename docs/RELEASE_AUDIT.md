# Source Validation

The implementation contains two independent stages: backbone fine-tuning for
multi-target grounding and supervised interactive disambiguation. The public
entry points are listed in `README.md`.

The source checks cover Python and Bash syntax, dependency/import consistency,
MT annotation contracts, candidate/image-feature alignment, and Stage 2
checkpoint loading. Small CPU fixtures verify training and evaluation without
substituting fixture scores for paper results.

Stage 1 validation exports grounding metrics at IoU 0.25 and 0.5 and candidate
survival. Stage 2 reports final accuracy, success at each interaction limit,
conditional accuracy and target survival. Queries whose targets are absent
initially remain in the Stage 2 evaluation denominator.

Use the original train/validation/test scene splits. Build the vocabulary and
constraint inventory from training samples and reuse the saved specification
for validation. Candidate IDs and image features must refer to the same objects.

The annotation download includes a manifest with file hashes, row counts and
scene counts. Meshes, multiview features and model weights use the paths in
`.env.example`. The upstream ScanNet/3D-FRONT assets follow their providers'
licenses.
