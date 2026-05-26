#!/usr/bin/env python3
"""Create fixed-length SAVeD event clips from the existing SAVeD manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from create_saved_event_clips import repo_relative_path, round_time, run_ffmpeg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "clip_manifests" / "saved_event_clips.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "event_clips" / "saved_fixed10"
DEFAULT_OUTPUT_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "clip_manifests" / "saved_event_clips_fixed10.csv"
)

USABLE_SOURCE_STATUSES = {"created", "exists", "dry_run"}
USABLE_OUTPUT_STATUSES = {"created", "exists", "dry_run"}
INCLUDE_QA_STATUSES = {"include", "included", "valid", "keep", "use", "usable", "ok"}
EXCLUDE_QA_STATUSES = {
    "exclude",
    "excluded",
    "drop",
    "invalid",
    "bad",
    "bad_timestamp",
    "no_event",
    "no_accident",
    "no_collision",
    "no_near_miss",
    "wrong_event",
}
EXTRA_FIELDNAMES = [
    "source_manifest_clip_id",
    "source_manifest_clip_path",
    "fixed_clip_seconds",
    "fixed_clip_policy",
    "qa_status",
    "qa_notes",
    "qa_corrected_event_center_time",
    "qa_corrected_event_start_time",
    "qa_corrected_event_end_time",
    "event_center_time",
    "center_offset_in_clip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create fixed-length, event-centred SAVeD clips for BADAS outcome-head "
            "training. The script reads the existing variable-length SAVeD manifest "
            "and writes a new manifest without modifying the source manifest."
        )
    )
    parser.add_argument(
        "--source-manifest",
        default=DEFAULT_SOURCE_MANIFEST,
        type=Path,
        help="Existing SAVeD event manifest. Default: saved_event_clips.csv.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        type=Path,
        help="Root directory for fixed-length clips.",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_OUTPUT_MANIFEST,
        type=Path,
        help="Output fixed-length manifest CSV.",
    )
    parser.add_argument(
        "--clip-seconds",
        default=10.0,
        type=float,
        help="Target clip length in seconds. Default: 10.",
    )
    parser.add_argument(
        "--include",
        choices=["all", "collision", "near_miss"],
        default="all",
        help="Which event labels to include. Default: all.",
    )
    parser.add_argument("--limit", default=None, type=int, help="Process only the first N eligible rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fixed clips.")
    parser.add_argument(
        "--review-csv",
        default=None,
        type=Path,
        help=(
            "Optional manual QA CSV keyed by source_clip_id. Recognized columns: "
            "qa_status, corrected_event_center_time, corrected_event_start_time, "
            "corrected_event_end_time, qa_notes."
        ),
    )
    parser.add_argument(
        "--review-mode",
        choices=["exclude-marked", "included-only", "all"],
        default="exclude-marked",
        help=(
            "How to apply --review-csv. exclude-marked skips explicit excludes; "
            "included-only keeps only explicit include/valid rows; all applies "
            "time corrections but does not skip by QA status."
        ),
    )
    parser.add_argument(
        "--copy-video",
        action="store_true",
        help="Use stream copy for faster but less frame-accurate clipping.",
    )
    parser.add_argument(
        "--seek-mode",
        choices=["output", "input", "hybrid"],
        default="output",
        help="Seek strategy passed to ffmpeg. Default: output.",
    )
    parser.add_argument(
        "--seek-preroll",
        default=10.0,
        type=float,
        help="Seconds before clip start used for hybrid input pre-seek. Default: 10.",
    )
    parser.add_argument("--no-audio", action="store_true", help="Drop audio tracks from clips.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest without running ffmpeg or creating clips.",
    )
    parser.add_argument(
        "--ffmpeg-timeout",
        default=180.0,
        type=float,
        help="Seconds before giving up on cutting one clip. Default: 180.",
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


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def event_center(row: dict[str, str]) -> float | None:
    start = parse_float(row.get("event_start_time"))
    end = parse_float(row.get("event_end_time"))
    if start is None or end is None or end <= start:
        return None
    return (start + end) / 2.0


def normalize_status(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def read_review_csv(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}

    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Review CSV does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "source_clip_id" in (reader.fieldnames or []):
            id_column = "source_clip_id"
        elif "clip_id" in (reader.fieldnames or []):
            id_column = "clip_id"
        else:
            raise ValueError(f"{path} must contain source_clip_id or clip_id.")

        reviews = {}
        for row in reader:
            source_clip_id = str(row.get(id_column, "")).strip()
            if source_clip_id:
                reviews[source_clip_id] = row
    return reviews


def review_decision(
    review: dict[str, str] | None,
    review_mode: str,
) -> tuple[bool, str]:
    if review is None:
        if review_mode == "included-only":
            return False, "qa_unreviewed"
        return True, ""

    qa_status = normalize_status(review.get("qa_status", ""))
    if review_mode == "all":
        return True, ""
    if review_mode == "included-only":
        if qa_status in INCLUDE_QA_STATUSES:
            return True, ""
        if qa_status in EXCLUDE_QA_STATUSES:
            return False, f"qa_{qa_status}"
        return False, "qa_not_included"
    if qa_status in EXCLUDE_QA_STATUSES:
        return False, f"qa_{qa_status}"
    return True, ""


def reviewed_event_center(
    source_row: dict[str, str],
    review: dict[str, str] | None,
) -> float | None:
    if review is not None:
        center = parse_float(review.get("corrected_event_center_time"))
        if center is not None:
            return center

        start = parse_float(review.get("corrected_event_start_time"))
        end = parse_float(review.get("corrected_event_end_time"))
        if start is not None and end is not None and end > start:
            return (start + end) / 2.0

    return event_center(source_row)


def reviewed_event_times(
    source_row: dict[str, str],
    review: dict[str, str] | None,
) -> tuple[str, str]:
    if review is None:
        return source_row.get("event_start_time", ""), source_row.get("event_end_time", "")

    start = str(review.get("corrected_event_start_time", "")).strip()
    end = str(review.get("corrected_event_end_time", "")).strip()
    return start or source_row.get("event_start_time", ""), end or source_row.get("event_end_time", "")


def review_has_time_correction(review: dict[str, str] | None) -> bool:
    if review is None:
        return False
    return any(
        str(review.get(column, "")).strip()
        for column in (
            "corrected_event_center_time",
            "corrected_event_start_time",
            "corrected_event_end_time",
        )
    )


def fixed_bounds(
    center_time: float,
    video_duration: float | None,
    clip_seconds: float,
) -> tuple[float, float, float] | None:
    if clip_seconds <= 0:
        raise ValueError("--clip-seconds must be positive.")
    if center_time < 0:
        return None
    if video_duration is not None:
        if video_duration <= 0 or center_time > video_duration:
            return None
        if video_duration + 1e-3 < clip_seconds:
            return None

    half = clip_seconds / 2.0
    start = center_time - half
    end = center_time + half

    if start < 0:
        end -= start
        start = 0.0

    if video_duration is not None and end > video_duration:
        start -= end - video_duration
        end = video_duration
        if start < 0:
            start = 0.0

    duration = end - start
    if duration <= 0:
        return None

    if video_duration is not None and abs(duration - clip_seconds) > 1e-2:
        return None

    return start, end, duration


def fixed_clip_id(source_clip_id: str) -> str:
    return f"{source_clip_id}_fixed10"


def output_fieldnames(source_fieldnames: list[str]) -> list[str]:
    fieldnames: list[str] = []
    for field in source_fieldnames:
        if field == "clip_id":
            fieldnames.extend(name for name in EXTRA_FIELDNAMES if name not in fieldnames)
        if field not in fieldnames:
            fieldnames.append(field)
    for field in EXTRA_FIELDNAMES:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def manifest_row(
    source_row: dict[str, str],
    clip_id: str,
    output_clip: Path,
    center_time: float | None,
    bounds: tuple[float, float, float] | None,
    clip_seconds: float,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = dict(source_row)
    source_clip_path = source_row.get("clip_path", "")

    row["source_manifest_clip_id"] = source_row.get("clip_id", "")
    row["source_manifest_clip_path"] = source_clip_path
    row["fixed_clip_seconds"] = round_time(clip_seconds)
    row["fixed_clip_policy"] = "event_centered_shift_to_video_bounds"
    row["clip_id"] = clip_id
    row["clip_path"] = repo_relative_path(output_clip) if status in USABLE_OUTPUT_STATUSES else ""
    row["event_center_time"] = round_time(center_time)

    if bounds is None:
        row["clip_start_time"] = ""
        row["clip_end_time"] = ""
        row["clip_duration"] = ""
        row["center_offset_in_clip"] = ""
    else:
        start, end, duration = bounds
        row["clip_start_time"] = round_time(start)
        row["clip_end_time"] = round_time(end)
        row["clip_duration"] = round_time(duration)
        row["center_offset_in_clip"] = round_time(center_time - start if center_time is not None else None)

    row["status"] = status
    row["error"] = error
    return row


def add_review_fields(
    row: dict[str, Any],
    review: dict[str, str] | None,
) -> None:
    row["qa_status"] = review.get("qa_status", "") if review else ""
    row["qa_notes"] = review.get("qa_notes", "") if review else ""
    row["qa_corrected_event_center_time"] = (
        review.get("corrected_event_center_time", "") if review else ""
    )
    row["qa_corrected_event_start_time"] = (
        review.get("corrected_event_start_time", "") if review else ""
    )
    row["qa_corrected_event_end_time"] = (
        review.get("corrected_event_end_time", "") if review else ""
    )

    if review is not None:
        start, end = reviewed_event_times(row, review)
        row["event_start_time"] = start
        row["event_end_time"] = end


def write_manifest(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.clip_seconds <= 0:
        raise ValueError("--clip-seconds must be positive.")
    if args.seek_preroll < 0:
        raise ValueError("--seek-preroll must be non-negative.")
    if args.ffmpeg_timeout <= 0:
        raise ValueError("--ffmpeg-timeout must be positive.")

    source_manifest = args.source_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()

    if not source_manifest.exists():
        raise FileNotFoundError(f"Source manifest does not exist: {source_manifest}")

    with source_manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        source_fieldnames = list(reader.fieldnames or [])
        source_rows = list(reader)
    reviews = read_review_csv(args.review_csv)

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    processed = 0

    print(f"Source manifest: {source_manifest}")
    print(f"Output root: {output_root}")
    print(f"Fixed manifest: {manifest_path}")
    print(f"Clip seconds: {args.clip_seconds}")
    print(f"Dry run: {args.dry_run}")
    print(f"Preserve audio: {not args.no_audio}")
    print(f"Review rows: {len(reviews)}")
    print(f"Review mode: {args.review_mode}")

    for source_row in source_rows:
        label = str(source_row.get("event_label", "")).strip()
        if args.include != "all" and label != args.include:
            continue
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1

        source_clip_id = str(source_row.get("clip_id", "")).strip()
        review = reviews.get(source_clip_id)
        include_row, excluded_status = review_decision(review, args.review_mode)
        clip_id = fixed_clip_id(source_clip_id) if source_clip_id else ""
        output_clip = output_root / label / f"{clip_id}.mp4" if clip_id else Path("")
        source_status = str(source_row.get("status", "")).strip()
        center_time = reviewed_event_center(source_row, review)
        video_duration = parse_float(source_row.get("video_duration"))

        if not include_row:
            counts["skipped"] += 1
            output_row = manifest_row(
                source_row,
                clip_id,
                output_clip,
                center_time,
                None,
                args.clip_seconds,
                excluded_status,
            )
            add_review_fields(output_row, review)
            rows.append(output_row)
            continue

        if source_status not in USABLE_SOURCE_STATUSES:
            status = f"source_{source_status or 'unusable'}"
            counts["skipped"] += 1
            output_row = manifest_row(
                source_row,
                clip_id,
                output_clip,
                center_time,
                None,
                args.clip_seconds,
                status,
            )
            add_review_fields(output_row, review)
            rows.append(output_row)
            continue

        source_video_path = str(source_row.get("source_video_path", "")).strip()
        if not source_video_path:
            counts["skipped"] += 1
            output_row = manifest_row(
                source_row,
                clip_id,
                output_clip,
                center_time,
                None,
                args.clip_seconds,
                "missing_source_video_path",
            )
            add_review_fields(output_row, review)
            rows.append(output_row)
            continue

        source_video = resolve_repo_path(source_video_path)
        if not source_video.exists():
            counts["skipped"] += 1
            output_row = manifest_row(
                source_row,
                clip_id,
                output_clip,
                center_time,
                None,
                args.clip_seconds,
                "missing_video",
                f"Source video does not exist: {source_video}",
            )
            add_review_fields(output_row, review)
            rows.append(output_row)
            continue

        if center_time is None:
            counts["skipped"] += 1
            output_row = manifest_row(
                source_row,
                clip_id,
                output_clip,
                center_time,
                None,
                args.clip_seconds,
                "invalid_event_times",
            )
            add_review_fields(output_row, review)
            rows.append(output_row)
            continue

        bounds = fixed_bounds(center_time, video_duration, args.clip_seconds)
        if bounds is None:
            counts["skipped"] += 1
            output_row = manifest_row(
                source_row,
                clip_id,
                output_clip,
                center_time,
                None,
                args.clip_seconds,
                "invalid_fixed_bounds",
            )
            add_review_fields(output_row, review)
            rows.append(output_row)
            continue

        start, _, duration = bounds

        if output_clip.exists() and not args.overwrite:
            if review_has_time_correction(review):
                counts["skipped"] += 1
                output_row = manifest_row(
                    source_row,
                    clip_id,
                    output_clip,
                    center_time,
                    bounds,
                    args.clip_seconds,
                    "needs_overwrite_for_correction",
                    "Existing clip is stale because review CSV changed the event time. Re-run with --overwrite.",
                )
                add_review_fields(output_row, review)
                rows.append(output_row)
                continue

            counts["exists"] += 1
            output_row = manifest_row(
                source_row,
                clip_id,
                output_clip,
                center_time,
                bounds,
                args.clip_seconds,
                "exists",
            )
            add_review_fields(output_row, review)
            rows.append(output_row)
            continue

        if args.dry_run:
            counts["dry_run"] += 1
            output_row = manifest_row(
                source_row,
                clip_id,
                output_clip,
                center_time,
                bounds,
                args.clip_seconds,
                "dry_run",
            )
            add_review_fields(output_row, review)
            rows.append(output_row)
            continue

        ok, error = run_ffmpeg(
            source_video=source_video,
            output_clip=output_clip,
            start_time=start,
            duration=duration,
            overwrite=args.overwrite,
            copy_video=args.copy_video,
            include_audio=not args.no_audio,
            seek_mode=args.seek_mode,
            seek_preroll=args.seek_preroll,
            timeout_seconds=args.ffmpeg_timeout,
        )
        status = "created" if ok else "ffmpeg_failed"
        counts[status if ok else "failed"] += 1
        output_row = manifest_row(
            source_row,
            clip_id,
            output_clip,
            center_time,
            bounds,
            args.clip_seconds,
            status,
            error,
        )
        add_review_fields(output_row, review)
        rows.append(output_row)

        if processed % 50 == 0:
            print(f"Processed {processed} eligible rows...")

    write_manifest(manifest_path, rows, output_fieldnames(source_fieldnames))

    print("Done.")
    print(f"Manifest rows: {len(rows)}")
    print(f"Created clips: {counts['created']}")
    print(f"Existing clips: {counts['exists']}")
    print(f"Dry-run clips: {counts['dry_run']}")
    print(f"Skipped rows: {counts['skipped']}")
    print(f"Failed rows: {counts['failed']}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
