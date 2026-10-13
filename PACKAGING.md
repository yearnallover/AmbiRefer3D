# AmbiRefer3D Source Package

The archive contains the Stage 1 backbone implementation, Stage 2 semantic
disambiguation code, both data construction pipelines, configuration,
preprocessing utilities, regression tests, and publication documentation.

It excludes raw datasets, generated dialogue/round JSON, feature caches,
checkpoints, compiled CUDA binaries, conda environments, logs, private `.env`
files, and original Git history. Compile custom operators after installing
compatible PyTorch/CUDA dependencies.

The authoritative method and entry points are in `README.md`. External
requirements and unresolved construction/interface gaps are documented in
`docs/DATA_CONSTRUCTION.md` and `docs/RELEASE_AUDIT.md`.

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/audit_release.py
PYTHONDONTWRITEBYTECODE=1 python tools/package_release.py \
  --output /path/to/external/ambirefer3d-code.tar.gz
```

To verify an archive:

```bash
cd /path/to/external
sha256sum -c ambirefer3d-code.tar.gz.sha256
tar -tzf ambirefer3d-code.tar.gz
```

Extracted source has a single `ambirefer3d/` root. See `GITHUB_RELEASE.md` for
initializing a fresh Git repository and publishing to your own GitHub account.
