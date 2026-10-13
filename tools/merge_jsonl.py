"""Convert one or more JSONL files into an explicit JSON array."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = []
    for source in args.input:
        with open(source, encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f'{source}:{line_number}: invalid JSON') from exc
                    if not isinstance(row, dict):
                        raise ValueError(f'{source}:{line_number}: expected a JSON object')
                    rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {len(rows)} rows to {output}')


if __name__ == '__main__':
    main()
