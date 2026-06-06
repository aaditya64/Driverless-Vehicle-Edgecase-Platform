#!/usr/bin/env python3
"""Create a manual QA sheet for SAVeD event clips."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "clip_manifests" / "saved_event_clips.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "clip_manifests" / "saved_event_clip_review.csv"

USABLE_STATUSES = {"created", "exists", "dry_run"}
FIELDNAMES = [
    "source_clip_id",
    "event_label",
    "video_id",
    "source_url",
    "source_video_path",
    "variable_clip_path",
    "original_time_range",
    "event_start_time",
    "event_end_time",
    "event_center_time",
    "clip_start_time",
    "clip_end_time",
    "clip_duration",
    "qa_status",
    "corrected_event_center_time",
    "corrected_event_start_time",
    "corrected_event_end_time",
    "qa_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a spreadsheet-friendly CSV for manually reviewing SAVeD rows. "
            "Use qa_status=include for usable rows, qa_status=exclude for rows "
            "whose clip does not contain the labelled event, and optional corrected "
            "event times for rows with bad timestamps."
        )
    )
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument(
        "--include",
        choices=["all", "collision", "near_miss"],
        default="all",
        help="Which event labels to include. Default: all.",
    )
    parser.add_argument("--limit", default=None, type=int, help="Write only the first N eligible rows.")
    parser.add_argument(
        "--include-unusable",
        action="store_true",
        help="Also include rows whose source manifest status is not created/exists/dry_run.",
    )
    return parser.parse_args()


def parse_float(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def round_time(value: float | None) -> float | str:
    if value is None:
        return ""
    return round(value, 3)


def read_rows(path: Path) -> list[dict[str, str]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source manifest does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def event_center(row: dict[str, str]) -> float | None:
    start = parse_float(row.get("event_start_time"))
    end = parse_float(row.get("event_end_time"))
    if start is None or end is None or end <= start:
        return None
    return (start + end) / 2.0


def review_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "source_clip_id": row.get("clip_id", ""),
        "event_label": row.get("event_label", ""),
        "video_id": row.get("video_id", ""),
        "source_url": row.get("source_url", ""),
        "source_video_path": row.get("source_video_path", ""),
        "variable_clip_path": row.get("clip_path", ""),
        "original_time_range": row.get("original_time_range", ""),
        "event_start_time": row.get("event_start_time", ""),
        "event_end_time": row.get("event_end_time", ""),
        "event_center_time": round_time(event_center(row)),
        "clip_start_time": row.get("clip_start_time", ""),
        "clip_end_time": row.get("clip_end_time", ""),
        "clip_duration": row.get("clip_duration", ""),
        "qa_status": "",
        "corrected_event_center_time": "",
        "corrected_event_start_time": "",
        "corrected_event_end_time": "",
        "qa_notes": "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = read_rows(args.source_manifest)

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        if args.limit is not None and len(output_rows) >= args.limit:
            break

        if args.include != "all" and row.get("event_label", "") != args.include:
            continue

        if not args.include_unusable and row.get("status", "") not in USABLE_STATUSES:
            continue

        output_rows.append(review_row(row))

    write_csv(args.output, output_rows)
    print(f"Wrote review sheet: {args.output.expanduser().resolve()}")
    print(f"Rows: {len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
