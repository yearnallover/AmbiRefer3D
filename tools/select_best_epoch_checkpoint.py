#!/usr/bin/env python3
"""Select the checkpoint with the highest logged validation Joint score."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


VAL_PATTERN = re.compile(
    r"\[VAL-MT\]\[epoch\s+(?P<epoch>\d+)\].*?Joint=(?P<score>[0-9]+(?:\.[0-9]+)?)%"
)


def select_best(training_dir: Path) -> tuple[Path, int, float]:
    log_path = training_dir / "epoch_metrics.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing validation log: {log_path}")
    records = [
        (int(match.group("epoch")), float(match.group("score")))
        for match in VAL_PATTERN.finditer(log_path.read_text(encoding="utf-8"))
    ]
    if not records:
        raise RuntimeError(f"No finite Joint validation records found in {log_path}")
    epoch, score = max(records, key=lambda item: (item[1], -item[0]))
    checkpoint = training_dir / f"epoch={epoch - 1}.ckpt"
    if not checkpoint.is_file():
        last = training_dir / "last.ckpt"
        if epoch == max(item[0] for item in records) and last.is_file():
            checkpoint = last
        else:
            raise FileNotFoundError(
                f"Best epoch is {epoch}, but its checkpoint is missing: {checkpoint}"
            )
    return checkpoint.resolve(), epoch, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint, epoch, score = select_best(args.training_dir)
    print(
        f"selected validation epoch {epoch} with Joint={score:.2f}%",
        file=sys.stderr,
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
