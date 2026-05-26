"""Validate the expected Nexar split files and video paths."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _check_csv(path: Path, columns: list[str]) -> tuple[int, list[str]]:
    missing: list[str] = []
    count = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            count += 1
            for col in columns:
                value = row.get(col, "")
                if value and not Path(value).exists():
                    missing.append(value)
    return count, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-train", default="splits/processed_744/train.csv")
    parser.add_argument("--short-test", default="splits/processed_744/test.csv")
    parser.add_argument("--inner", default="splits/processed_744/train_inner.csv")
    parser.add_argument("--val", default="splits/processed_744/val.csv")
    parser.add_argument("--long-train", default="splits/processed_744_long/train.csv")
    parser.add_argument("--long-test", default="splits/processed_744_long/test.csv")
    args = parser.parse_args()

    specs = [
        (Path(args.short_train), ["path"]),
        (Path(args.short_test), ["path"]),
        (Path(args.inner), ["path"]),
        (Path(args.val), ["path"]),
        (Path(args.long_train), ["path", "source_path"]),
        (Path(args.long_test), ["path", "source_path"]),
    ]
    total_missing: list[str] = []
    for csv_path, columns in specs:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        count, missing = _check_csv(csv_path, columns)
        total_missing.extend(missing)
        print(f"{csv_path}: rows={count} missing_paths={len(missing)}", flush=True)
    if total_missing:
        preview = "\n".join(total_missing[:20])
        raise FileNotFoundError(f"missing video paths:\n{preview}")
    print("data ready", flush=True)


if __name__ == "__main__":
    main()
