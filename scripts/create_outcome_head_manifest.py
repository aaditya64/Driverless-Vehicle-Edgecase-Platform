#!/usr/bin/env python3
"""Create a unified manifest for near-miss/collision outcome-head training."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEXAR_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clip_manifests"
    / "nexar_train_positive_event_clips.csv"
)
DEFAULT_SAVED_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "clip_manifests" / "saved_event_clips_fixed10.csv"
)
DEFAULT_OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clip_manifests"
    / "outcome_head_nexar_saved_fixed10.csv"
)

USABLE_STATUSES = {"created", "exists", "dry_run"}
SUPPORTED_LABELS = {"near_miss", "collision"}
FIELDNAMES = [
    "clip_id",
    "clip_path",
    "outcome_label",
    "source_dataset",
    "source_manifest",
    "source_clip_id",
    "source_video_id",
    "group_id",
    "status",
    "event_start_time",
    "event_end_time",
    "event_center_time",
    "clip_duration",
    "source_file_name",
    "manual_label",
    "event_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Nexar manual labels and fixed-length SAVeD labels into one "
            "manifest with a shared outcome_label column."
        )
    )
    parser.add_argument("--nexar-manifest", default=DEFAULT_NEXAR_MANIFEST, type=Path)
    parser.add_argument("--saved-manifest", default=DEFAULT_SAVED_MANIFEST, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_MANIFEST, type=Path)
    parser.add_argument("--nexar-label-column", default="manual_label")
    parser.add_argument("--saved-label-column", default="event_label")
    parser.add_argument("--output-label-column", default="outcome_label")
    parser.add_argument("--skip-nexar", action="store_true", help="Do not include Nexar rows.")
    parser.add_argument("--skip-saved", action="store_true", help="Do not include SAVeD rows.")
    return parser.parse_args()


def repo_relative_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text.startswith("collision"):
        return "collision"
    if text.startswith("near_miss") or text.startswith("nearmiss"):
        return "near_miss"
    return text


def require_columns(path: Path, fieldnames: list[str], columns: list[str]) -> None:
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")


def build_nexar_rows(
    manifest_path: Path,
    label_column: str,
    output_label_column: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    fieldnames, rows = read_csv(manifest_path)
    require_columns(manifest_path, fieldnames, ["clip_id", "clip_path", "status", label_column])

    output_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    manifest_text = repo_relative_path(manifest_path)

    for row in rows:
        status = str(row.get("status", "")).strip()
        if status not in USABLE_STATUSES:
            skipped["bad_status"] += 1
            continue

        label = normalize_label(row.get(label_column))
        if label not in SUPPORTED_LABELS:
            skipped[f"label:{label or 'empty'}"] += 1
            continue

        clip_id = str(row.get("clip_id", "")).strip()
        if not clip_id:
            skipped["missing_clip_id"] += 1
            continue

        output_rows.append(
            {
                "clip_id": clip_id,
                "clip_path": row.get("clip_path", ""),
                output_label_column: label,
                "source_dataset": "nexar_collision_prediction",
                "source_manifest": manifest_text,
                "source_clip_id": clip_id,
                "source_video_id": row.get("source_file_name", ""),
                "group_id": f"nexar:{clip_id}",
                "status": status,
                "event_start_time": row.get("time_of_event", ""),
                "event_end_time": row.get("time_of_event", ""),
                "event_center_time": row.get("event_center_time", ""),
                "clip_duration": row.get("clip_duration", ""),
                "source_file_name": row.get("source_file_name", ""),
                "manual_label": row.get(label_column, ""),
                "event_label": "",
            }
        )

    return output_rows, skipped


def build_saved_rows(
    manifest_path: Path,
    label_column: str,
    output_label_column: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    fieldnames, rows = read_csv(manifest_path)
    require_columns(manifest_path, fieldnames, ["clip_id", "clip_path", "status", label_column, "video_id"])

    output_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    manifest_text = repo_relative_path(manifest_path)

    for row in rows:
        status = str(row.get("status", "")).strip()
        if status not in USABLE_STATUSES:
            skipped["bad_status"] += 1
            continue

        label = normalize_label(row.get(label_column))
        if label not in SUPPORTED_LABELS:
            skipped[f"label:{label or 'empty'}"] += 1
            continue

        clip_id = str(row.get("clip_id", "")).strip()
        video_id = str(row.get("video_id", "")).strip()
        if not clip_id:
            skipped["missing_clip_id"] += 1
            continue
        if not video_id:
            skipped["missing_video_id"] += 1
            continue

        output_rows.append(
            {
                "clip_id": clip_id,
                "clip_path": row.get("clip_path", ""),
                output_label_column: label,
                "source_dataset": "saved_av_dataset",
                "source_manifest": manifest_text,
                "source_clip_id": row.get("source_manifest_clip_id", "") or clip_id,
                "source_video_id": video_id,
                "group_id": f"saved:{video_id}",
                "status": status,
                "event_start_time": row.get("event_start_time", ""),
                "event_end_time": row.get("event_end_time", ""),
                "event_center_time": row.get("event_center_time", ""),
                "clip_duration": row.get("clip_duration", ""),
                "source_file_name": "",
                "manual_label": "",
                "event_label": row.get(label_column, ""),
            }
        )

    return output_rows, skipped


def write_csv(path: Path, rows: list[dict[str, Any]], output_label_column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [output_label_column if name == "outcome_label" else name for name in FIELDNAMES]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.skip_nexar and args.skip_saved:
        raise ValueError("At least one source must be included.")

    output_rows: list[dict[str, Any]] = []
    skipped_by_source: dict[str, Counter[str]] = {}

    if not args.skip_nexar:
        nexar_rows, skipped = build_nexar_rows(
            args.nexar_manifest.expanduser().resolve(),
            args.nexar_label_column,
            args.output_label_column,
        )
        output_rows.extend(nexar_rows)
        skipped_by_source["nexar"] = skipped

    if not args.skip_saved:
        saved_rows, skipped = build_saved_rows(
            args.saved_manifest.expanduser().resolve(),
            args.saved_label_column,
            args.output_label_column,
        )
        output_rows.extend(saved_rows)
        skipped_by_source["saved"] = skipped

    seen: set[str] = set()
    duplicates: list[str] = []
    for row in output_rows:
        clip_id = str(row["clip_id"])
        if clip_id in seen:
            duplicates.append(clip_id)
        seen.add(clip_id)
    if duplicates:
        duplicate_preview = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate clip_id values in combined manifest: {duplicate_preview}")

    output_path = args.output.expanduser().resolve()
    write_csv(output_path, output_rows, args.output_label_column)

    label_counts = Counter(str(row[args.output_label_column]) for row in output_rows)
    source_counts = Counter(str(row["source_dataset"]) for row in output_rows)
    group_count = len({str(row["group_id"]) for row in output_rows})

    print(f"Wrote combined manifest: {output_path}")
    print(f"Rows: {len(output_rows)}")
    print(f"Source counts: {dict(source_counts)}")
    print(f"Label counts: {dict(label_counts)}")
    print(f"Group count: {group_count}")
    print(f"Skipped: { {source: dict(counter) for source, counter in skipped_by_source.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
