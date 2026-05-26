"""Prepare BADAS-compatible manifests from existing collision/near_miss splits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .common import read_split_csv, stable_video_id


def _row(split: str, item: dict[str, Any]) -> dict[str, Any]:
    label = str(item.get("label_name") or ("collision" if int(item["label"]) == 1 else "near_miss"))
    return {
        "clip_id": stable_video_id(item["path"]),
        "clip_path": item["path"],
        "source_file_name": Path(item["path"]).name,
        "core_label": label,
        "label": int(item["label"]),
        "split": split,
        "event_center_time": "",
        "center_offset_in_clip": "",
        "status": "created",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="splits/processed_744/train.csv")
    parser.add_argument("--test-csv", default="splits/processed_744/test.csv")
    parser.add_argument("--train-inner-csv", default="splits/processed_744/train_inner.csv")
    parser.add_argument("--val-csv", default="splits/processed_744/val.csv")
    parser.add_argument("--out-dir", default="outputs/processed_744/badas_manifests")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "train": args.train_csv,
        "test": args.test_csv,
        "train_inner": args.train_inner_csv,
        "val": args.val_csv,
    }
    fieldnames = [
        "clip_id",
        "clip_path",
        "source_file_name",
        "core_label",
        "label",
        "split",
        "event_center_time",
        "center_offset_in_clip",
        "status",
    ]
    all_rows = []
    unique_rows: dict[str, dict[str, Any]] = {}
    for split, csv_path in specs.items():
        rows = [_row(split, item) for item in read_split_csv(csv_path)]
        all_rows.extend(rows)
        for row in rows:
            unique_rows.setdefault(row["clip_id"], {**row, "split": "unique"})
        with (out_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    with (out_dir / "all_splits.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with (out_dir / "all_unique.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(unique_rows.values(), key=lambda x: x["clip_id"]))
    print(f"wrote BADAS manifests to {out_dir}")


if __name__ == "__main__":
    main()
