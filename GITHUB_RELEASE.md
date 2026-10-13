# Publish ambirefer3d

The prepared source directory is suitable for a fresh GitHub repository. It
contains no original Git history and no inherited remote. Read
`docs/RELEASE_AUDIT.md` before documenting reproduction claims.

```bash
cd /path/to/ambirefer3d
python tools/audit_release.py
git init -b main
git add .
git diff --cached --stat
git commit -m "Release AmbiRefer3D source code"
git remote add origin git@github.com:YOUR_ACCOUNT/ambirefer3d.git
git push -u origin main
```

Create an empty GitHub repository named `ambirefer3d` before adding the remote.
The archive itself should be downloaded/extracted locally, not committed into
the repository. Keep credentials in an ignored `.env` file. Dataset and model
download links can be added after the corresponding artifacts are prepared.

Repackage the source to an external destination:

```bash
python tools/package_release.py --output /path/to/ambirefer3d-code.tar.gz
```

The packager runs the source audit, writes a single `ambirefer3d/` archive root,
and creates a SHA256 sidecar. It excludes `.git`, local `.env` and generated
artifacts. This release has not been pushed to GitHub automatically.
