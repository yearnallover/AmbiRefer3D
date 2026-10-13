"""Audit a source release without importing CUDA or data-construction modules."""
import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = {'__pycache__', 'output', 'outputs', 'logs', 'checkpoints', 'predictions',
             '.venv', 'envs', 'build', '.pytest_cache'}
BINARY_SUFFIXES = {'.ckpt', '.pth', '.pt', '.npy', '.npz', '.hdf5', '.h5', '.ply',
                   '.pyc', '.so', '.o', '.sens', '.log', '.zip', '.gz'}
TEXT_SUFFIXES = {'.py', '.sh', '.md', '.txt', '.yaml', '.yml', '.json', '.example',
                 '.cpp', '.cu', '.h', '.hpp', '.cuh'}


def audit(root):
    failures = []
    count = 0
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if '.git' in relative.parts:
            continue
        if path.name.startswith('.env') and path.name != '.env.example':
            continue
        if path.is_symlink():
            failures.append(f'Symlink: {relative}')
            continue
        if path.is_dir():
            if path.name in GENERATED:
                failures.append(f'Generated directory: {relative}')
            continue
        count += 1
        if path.suffix in BINARY_SUFFIXES or path.stat().st_size > 50 * 1024 * 1024:
            failures.append(f'Binary/large artifact: {relative}')
            continue
        if re.search('v' + '16', str(relative), re.I):
            failures.append(f'Private experiment filename: {relative}')
        if path.suffix not in TEXT_SUFFIXES and path.name not in {'LICENSE', '.gitignore'}:
            continue
        text = path.read_text(encoding='utf-8')
        if re.search('v' + '16', text, re.I):
            failures.append(f'Private experiment marker: {relative}')
        if re.search(r'/(?:home|media)/[A-Za-z0-9_/.-]+', text):
            failures.append(f'Workstation-specific path: {relative}')
        if re.search(r'(?:sk-[A-Za-z0-9_-]{20,}|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY)', text):
            failures.append(f'Credential-like content: {relative}')
        if path.suffix == '.py':
            try:
                tree = ast.parse(text, filename=str(relative))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and re.search(r'(api_?key|password|token)$', target.id, re.I):
                                value = node.value.value
                                if isinstance(value, str) and value not in {'', 'EMPTY', 'YOUR_API_KEY'}:
                                    failures.append(f'Literal credential assignment: {relative}:{node.lineno}')
            except SyntaxError as exc:
                failures.append(f'Python syntax: {relative}:{exc.lineno}: {exc.msg}')
        if path.suffix == '.sh':
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            if result.returncode:
                failures.append(f'Shell syntax: {relative}: {result.stderr.strip()}')
    for required in ['README.md', 'LICENSE', 'setup.py', '.env.example', 'environment.yml',
                     'stage2_refactor/train.py', 'm3drefclip/common_ops/setup.py',
                     'data_construction/real/generate_clarifications.py',
                     'data_construction/synthetic/generate_dataset.py',
                     'scripts/train_ambirefer3d_stage1.sh', 'scripts/train_ambirefer3d_stage2.sh',
                     'docs/DATA_CONSTRUCTION.md', 'docs/RELEASE_AUDIT.md']:
        if not (root / required).is_file():
            failures.append(f'Missing required source: {required}')
    return {'passed': not failures, 'source_files': count, 'failures': failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default='')
    args = parser.parse_args()
    report = audit(ROOT)
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
