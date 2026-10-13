# Third-Party Notices

- **Multi3DRefer / M3DRef-CLIP:** Stage 1 is derived from the implementation by
  3dlg-hcvc. The original MIT license and copyright are retained in `LICENSE`.
  The public repository is AmbiRefer3D; it is not the upstream project's official
  repository. https://github.com/3dlg-hcvc/multi3drefer
- **BlenderProc:** Used by the synthetic generator. The upstream license is
  retained at `data_construction/synthetic/LICENSE.blenderproc`.
  https://github.com/DLR-RM/BlenderProc
- **PyTorch3D, MinkowskiEngine, CLIP, Transformers and torchvision:** Installed
  as external dependencies and governed by their upstream licenses. Their
  local clones, compiled binaries and model weights are not bundled.
- **ScanNet, ScanRefer, Multi3DRefer, 3D-FRONT, 3D-FUTURE and CC materials:**
  Dataset/model assets are external and subject to their providers' access and
  redistribution terms. This archive contains construction/preprocessing code,
  not those assets.

Internal Python module names and dataset identifiers retain their upstream
names to preserve checkpoint and configuration compatibility.
