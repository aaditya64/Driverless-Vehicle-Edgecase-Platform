"""Export compact object residual metrics used by the collision head."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_split_csv, stable_video_id


FIELDS = [
    "idx",
    "path",
    "label",
    "obj_score_ratio",
    "obj_score_z",
    "obj_energy_ratio",
    "obj_shift_ratio",
    "obj_diff_ratio",
    "obj_peak_time",
]


def _load_value(feature_dir: Path, video_path: str, name: str) -> float:
    data = np.load(feature_dir / f"{stable_video_id(video_path)}.npz", allow_pickle=True)
    names = [str(x) for x in data["summary_names"]]
    idx = names.index(name)
    return float(data["summary"][idx])


def _row(feature_dir: Path, idx: int, item: dict[str, Any]) -> dict[str, Any]:
    path = str(item["path"])
    return {
        "idx": idx,
        "path": path,
        "label": int(item["label"]),
        "obj_score_ratio": _load_value(feature_dir, path, "object_score_max_peak_early_ratio"),
        "obj_score_z": _load_value(feature_dir, path, "object_score_max_peak_z"),
        "obj_energy_ratio": _load_value(feature_dir, path, "object_res_energy_max_peak_early_ratio"),
        "obj_shift_ratio": _load_value(feature_dir, path, "object_center_shift_max_peak_early_ratio"),
        "obj_diff_ratio": _load_value(feature_dir, path, "object_diff_energy_max_peak_early_ratio"),
        "obj_peak_time": _load_value(feature_dir, path, "object_score_max_peak_time"),
    }


def _write(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="outputs/processed_744/object_residual_physics_w320_s4")
    parser.add_argument("--train-csv", default="splits/processed_744/train.csv")
    parser.add_argument("--test-csv", default="splits/processed_744/test.csv")
    parser.add_argument("--out-dir", default="analysis/impact_diagnostics_20260522")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    out_dir = Path(args.out_dir)
    train_rows = [_row(feature_dir, i, row) for i, row in enumerate(read_split_csv(args.train_csv))]
    test_rows = [_row(feature_dir, i, row) for i, row in enumerate(read_split_csv(args.test_csv))]
    _write(out_dir / "object_physics_train_metrics.csv", train_rows)
    _write(out_dir / "object_physics_test_metrics.csv", test_rows)
    print(f"wrote object metrics to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
