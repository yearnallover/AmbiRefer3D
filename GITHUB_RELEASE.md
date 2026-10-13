# AmbiRefer3D Publication

The paper implementation is organized into independent Stage 1 grounding,
Stage 2 disambiguation and data construction components. Before publishing a
source update, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/audit_release.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_release_contracts.py
git diff --stat
```

The annotation archive and its SHA256 sidecar are published on the `annotations`
branch of `yearnallover/AmbiRefer3D`. The README links directly to those assets.
The annotation manifest records file hashes, scene splits and row counts.

The source archive contains a single `ambirefer3d/` directory and omits Git
metadata. Extract it before creating a fresh repository. Keep local `.env`,
point-cloud features, model weights and generated logs outside source commits.

```bash
python tools/package_release.py --output /path/to/ambirefer3d-paper-code.tar.gz
```
