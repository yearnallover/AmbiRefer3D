"""Create a source archive and digest in an explicit external output location."""
import argparse
import hashlib
import tarfile
from pathlib import Path

from audit_release import ROOT, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('Archive destination must be outside the source tree')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    report = audit(ROOT)
    if not report['passed']:
        raise RuntimeError(report['failures'])
    output.parent.mkdir(parents=True, exist_ok=True)
    def normalize_metadata(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ''
        return info

    with tarfile.open(output, 'w:gz') as archive:
        for path in sorted(ROOT.rglob('*')):
            relative = path.relative_to(ROOT)
            private_env = path.name.startswith('.env') and path.name != '.env.example'
            if '.git' in relative.parts or private_env or not path.is_file():
                continue
            archive.add(path, arcname=str(Path('ambirefer3d') / relative), recursive=False,
                        filter=normalize_metadata)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    Path(f'{output}.sha256').write_text(f'{digest}  {output.name}\n')
    print(f'Archive: {output}\nSize: {output.stat().st_size} bytes\nSHA256: {digest}')


if __name__ == '__main__':
    main()
