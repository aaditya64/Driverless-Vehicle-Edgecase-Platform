#!/usr/bin/env python3
"""Create event clips from the saved AV crash and near-miss dataset."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "saved" / "raw"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "saved" / "videos_qt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "event_clips" / "saved"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "clip_manifests" / "saved_event_clips.csv"
)

BASE_FIELDNAMES = [
    "clip_id",
    "source_dataset",
    "event_label",
    "source_event_type",
    "source_csv",
    "source_row_index",
    "video_id",
    "source_url",
    "source_video_path",
    "clip_path",
    "original_time_range",
    "event_start_time",
    "event_end_time",
    "pre_buffer",
    "post_buffer",
    "clip_start_time",
    "clip_end_time",
    "clip_duration",
    "video_duration",
    "status",
    "error",
]


@dataclass(frozen=True)
class SourceSpec:
    event_type: str
    event_label: str
    csv_path: Path
    time_column: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cut one mp4 clip per valid saved AV crash/near-miss CSV row. "
            "Rows that cannot be cut are still written to the manifest with a status."
        )
    )
    parser.add_argument(
        "--crash-csv",
        default=DEFAULT_RAW_DIR / "AV_crash.csv",
        type=Path,
        help="Crash CSV path. Default: data/saved/raw/AV_crash.csv.",
    )
    parser.add_argument(
        "--nearmiss-csv",
        default=DEFAULT_RAW_DIR / "AV_nearmiss.csv",
        type=Path,
        help="Near-miss CSV path. Default: data/saved/raw/AV_nearmiss.csv.",
    )
    parser.add_argument(
        "--video-dir",
        default=DEFAULT_VIDEO_DIR,
        type=Path,
        help="Directory containing saved source .mp4 files named by video id.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        type=Path,
        help="Root directory where event clips will be written.",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        type=Path,
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--include",
        choices=["all", "crash", "nearmiss"],
        default="all",
        help="Which source CSVs to process. Default: all.",
    )
    parser.add_argument(
        "--pre-buffer",
        default=1.0,
        type=float,
        help="Seconds to include before the CSV event start time. Default: 1.",
    )
    parser.add_argument(
        "--post-buffer",
        default=1.0,
        type=float,
        help="Seconds to include after the CSV event end time. Default: 1.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Process only the first N CSV rows across selected sources.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing clips.",
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
        help=(
            "Seek strategy. output is frame-accurate but slower, input is faster "
            "but less exact, hybrid uses an input pre-seek followed by output seek. "
            "Default: output."
        ),
    )
    parser.add_argument(
        "--seek-preroll",
        default=10.0,
        type=float,
        help="Seconds before the clip start used for hybrid input pre-seek. Default: 10.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Drop audio tracks from generated clips. By default, audio is preserved.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest without running ffmpeg or creating clips.",
    )
    parser.add_argument(
        "--ffprobe-timeout",
        default=20.0,
        type=float,
        help="Seconds before giving up on probing one source video. Default: 20.",
    )
    parser.add_argument(
        "--ffmpeg-timeout",
        default=180.0,
        type=float,
        help="Seconds before giving up on cutting one clip. Default: 180.",
    )
    return parser.parse_args()


def repo_relative_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def round_time(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def normalize_url_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().strip("\"'").rstrip("\\").strip()


def extract_video_id(url_value: Any) -> str:
    text = normalize_url_text(url_value)
    if not text:
        return ""

    parsed = urlparse(text)
    host = parsed.netloc.lower()

    if "youtu.be" in host:
        video_id = parsed.path.strip("/").split("/")[0]
        return normalize_url_text(video_id)

    query_video_id = parse_qs(parsed.query).get("v", [""])[0]
    if query_video_id:
        return normalize_url_text(query_video_id)

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", text):
        return text

    return ""


def parse_timestamp(value: str, single_number_unit: str = "seconds") -> float | None:
    text = value.strip()
    if not text:
        return None

    parts = text.split(":")
    if len(parts) not in {1, 2, 3}:
        return None

    try:
        numbers = [float(part.strip()) for part in parts]
    except ValueError:
        return None

    if any(number < 0 for number in numbers):
        return None

    if len(numbers) == 1:
        (seconds,) = numbers
        if single_number_unit == "minutes":
            return seconds * 60
        return seconds

    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds

    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds


def parse_time_range(value: Any) -> tuple[float, float] | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None

    timestamp_pattern = r"\d+(?:\.\d+)?(?::\d+(?:\.\d+)?){0,2}"
    match = re.fullmatch(
        rf"\s*({timestamp_pattern})\s*[-\u2013\u2014~]\s*({timestamp_pattern})\s*",
        text,
    )
    if match is None:
        return None

    start_text = match.group(1)
    end_text = match.group(2)
    start_unit = "minutes" if ":" not in start_text and ":" in end_text else "seconds"
    end_unit = "minutes" if ":" not in end_text and ":" in start_text else "seconds"

    start_time = parse_timestamp(start_text, single_number_unit=start_unit)
    end_time = parse_timestamp(end_text, single_number_unit=end_unit)
    if start_time is None or end_time is None or end_time <= start_time:
        return None

    return start_time, end_time


def ffprobe_duration(video_path: Path, timeout_seconds: float) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None
    return parse_float(result.stdout)


def compute_clip_bounds(
    event_start_time: float,
    event_end_time: float,
    video_duration: float | None,
    pre_buffer: float,
    post_buffer: float,
) -> tuple[float, float, float] | None:
    clip_start_time = max(0.0, event_start_time - max(0.0, pre_buffer))
    clip_end_time = event_end_time + max(0.0, post_buffer)

    if video_duration is not None:
        clip_end_time = min(video_duration, clip_end_time)

    clip_duration = clip_end_time - clip_start_time
    if clip_duration <= 0:
        return None

    return clip_start_time, clip_end_time, clip_duration


def invalid_bounds_status(
    event_start_time: float,
    event_end_time: float,
    video_duration: float | None,
) -> str:
    if video_duration is not None and event_start_time >= video_duration:
        return "event_after_video_end"
    if video_duration is not None and event_end_time <= 0:
        return "event_before_video_start"
    return "invalid_clip_bounds"


def make_clip_id(
    event_label: str,
    source_row_index: int,
    video_id: str,
    event_start_time: float,
    event_end_time: float,
) -> str:
    return (
        f"saved_{event_label}_{source_row_index:06d}_"
        f"{video_id}_{event_start_time:.3f}_{event_end_time:.3f}"
    )


def run_ffmpeg(
    source_video: Path,
    output_clip: Path,
    start_time: float,
    duration: float,
    overwrite: bool,
    copy_video: bool,
    include_audio: bool,
    seek_mode: str,
    seek_preroll: float,
    timeout_seconds: float,
) -> tuple[bool, str]:
    output_clip.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
    ]

    if seek_mode == "input":
        cmd.extend(["-ss", f"{start_time:.3f}", "-i", str(source_video)])
    elif seek_mode == "hybrid":
        input_seek = max(0.0, start_time - seek_preroll)
        output_seek = start_time - input_seek
        cmd.extend(
            [
                "-ss",
                f"{input_seek:.3f}",
                "-i",
                str(source_video),
                "-ss",
                f"{output_seek:.3f}",
            ]
        )
    else:
        cmd.extend(["-i", str(source_video), "-ss", f"{start_time:.3f}"])

    cmd.extend(["-t", f"{duration:.3f}", "-map", "0:v:0"])
    if include_audio:
        cmd.extend(["-map", "0:a?"])
    else:
        cmd.append("-an")

    if copy_video:
        cmd.extend(["-c:v", "copy"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])

    if include_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])

    cmd.extend(["-movflags", "+faststart", str(output_clip)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"ffmpeg timed out after {timeout_seconds:.1f} seconds"

    if result.returncode == 0:
        return True, ""

    error = result.stderr.strip() or result.stdout.strip()
    return False, error


def prefixed_raw_fieldnames(sources: list[SourceSpec]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()

    for source in sources:
        with source.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for field in reader.fieldnames or []:
                prefixed = f"raw_{field}"
                if prefixed not in seen:
                    seen.add(prefixed)
                    fieldnames.append(prefixed)

    return fieldnames


def validate_sources(sources: list[SourceSpec]) -> None:
    for source in sources:
        if not source.csv_path.exists():
            raise FileNotFoundError(f"CSV does not exist: {source.csv_path}")


def build_manifest_row(
    source: SourceSpec,
    raw_row: dict[str, str],
    source_row_index: int,
    video_id: str,
    source_video: Path,
    output_clip: Path,
    event_times: tuple[float, float] | None,
    video_duration: float | None,
    clip_bounds: tuple[float, float, float] | None,
    pre_buffer: float,
    post_buffer: float,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    event_start_time: float | None = None
    event_end_time: float | None = None
    if event_times is not None:
        event_start_time, event_end_time = event_times

    clip_start_time: float | None = None
    clip_end_time: float | None = None
    clip_duration: float | None = None
    if clip_bounds is not None:
        clip_start_time, clip_end_time, clip_duration = clip_bounds

    row: dict[str, Any] = {
        "clip_id": output_clip.stem if output_clip.name else "",
        "source_dataset": "saved_av_dataset",
        "event_label": source.event_label,
        "source_event_type": source.event_type,
        "source_csv": repo_relative_path(source.csv_path),
        "source_row_index": source_row_index,
        "video_id": video_id,
        "source_url": raw_row.get("URL", ""),
        "source_video_path": repo_relative_path(source_video) if video_id else "",
        "clip_path": (
            repo_relative_path(output_clip)
            if status in {"created", "exists", "dry_run"}
            else ""
        ),
        "original_time_range": raw_row.get(source.time_column, ""),
        "event_start_time": round_time(event_start_time),
        "event_end_time": round_time(event_end_time),
        "pre_buffer": round_time(pre_buffer),
        "post_buffer": round_time(post_buffer),
        "clip_start_time": round_time(clip_start_time),
        "clip_end_time": round_time(clip_end_time),
        "clip_duration": round_time(clip_duration),
        "video_duration": round_time(video_duration),
        "status": status,
        "error": error,
    }

    for key, value in raw_row.items():
        row[f"raw_{key}"] = value

    return row


def selected_sources(args: argparse.Namespace) -> list[SourceSpec]:
    sources = [
        SourceSpec(
            event_type="crash",
            event_label="collision",
            csv_path=args.crash_csv.expanduser().resolve(),
            time_column="Time",
        ),
        SourceSpec(
            event_type="nearmiss",
            event_label="near_miss",
            csv_path=args.nearmiss_csv.expanduser().resolve(),
            time_column="time",
        ),
    ]

    if args.include == "all":
        return sources
    return [source for source in sources if source.event_type == args.include]


def write_manifest(
    manifest_path: Path,
    rows: list[dict[str, Any]],
    raw_fieldnames: list[str],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = BASE_FIELDNAMES + [
        field for field in raw_fieldnames if field not in BASE_FIELDNAMES
    ]

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def process_source(
    source: SourceSpec,
    video_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
    start_processed_rows: int,
    duration_cache: dict[Path, float | None],
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    if not source.csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {source.csv_path}")

    rows: list[dict[str, Any]] = []
    counts = {
        "created": 0,
        "exists": 0,
        "dry_run": 0,
        "skipped": 0,
        "failed": 0,
    }
    processed_rows = start_processed_rows

    with source.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for source_row_index, raw_row in enumerate(reader):
            if args.limit is not None and processed_rows >= args.limit:
                break
            processed_rows += 1

            video_id = extract_video_id(raw_row.get("URL", ""))
            event_times = parse_time_range(raw_row.get(source.time_column, ""))

            placeholder_output = Path("")
            source_video = video_dir / f"{video_id}.mp4" if video_id else Path("")

            if not video_id:
                counts["skipped"] += 1
                rows.append(
                    build_manifest_row(
                        source,
                        raw_row,
                        source_row_index,
                        video_id,
                        source_video,
                        placeholder_output,
                        event_times,
                        None,
                        None,
                        args.pre_buffer,
                        args.post_buffer,
                        "missing_video_id",
                    )
                )
                continue

            if event_times is None:
                counts["skipped"] += 1
                rows.append(
                    build_manifest_row(
                        source,
                        raw_row,
                        source_row_index,
                        video_id,
                        source_video,
                        placeholder_output,
                        event_times,
                        None,
                        None,
                        args.pre_buffer,
                        args.post_buffer,
                        "invalid_time_range",
                    )
                )
                continue

            event_start_time, event_end_time = event_times
            clip_id = make_clip_id(
                source.event_label,
                source_row_index,
                video_id,
                event_start_time,
                event_end_time,
            )
            output_clip = output_root / source.event_label / f"{clip_id}.mp4"

            if not source_video.exists():
                counts["skipped"] += 1
                rows.append(
                    build_manifest_row(
                        source,
                        raw_row,
                        source_row_index,
                        video_id,
                        source_video,
                        output_clip,
                        event_times,
                        None,
                        None,
                        args.pre_buffer,
                        args.post_buffer,
                        "missing_video",
                    )
                )
                continue

            if source_video not in duration_cache:
                duration_cache[source_video] = ffprobe_duration(
                    source_video,
                    timeout_seconds=args.ffprobe_timeout,
                )
            video_duration = duration_cache[source_video]
            clip_bounds = compute_clip_bounds(
                event_start_time=event_start_time,
                event_end_time=event_end_time,
                video_duration=video_duration,
                pre_buffer=args.pre_buffer,
                post_buffer=args.post_buffer,
            )

            if clip_bounds is None:
                status = invalid_bounds_status(
                    event_start_time=event_start_time,
                    event_end_time=event_end_time,
                    video_duration=video_duration,
                )
                counts["skipped"] += 1
                rows.append(
                    build_manifest_row(
                        source,
                        raw_row,
                        source_row_index,
                        video_id,
                        source_video,
                        output_clip,
                        event_times,
                        video_duration,
                        clip_bounds,
                        args.pre_buffer,
                        args.post_buffer,
                        status,
                    )
                )
                continue

            clip_start_time, _, clip_duration = clip_bounds

            if output_clip.exists() and not args.overwrite:
                counts["exists"] += 1
                rows.append(
                    build_manifest_row(
                        source,
                        raw_row,
                        source_row_index,
                        video_id,
                        source_video,
                        output_clip,
                        event_times,
                        video_duration,
                        clip_bounds,
                        args.pre_buffer,
                        args.post_buffer,
                        "exists",
                    )
                )
                continue

            if args.dry_run:
                counts["dry_run"] += 1
                rows.append(
                    build_manifest_row(
                        source,
                        raw_row,
                        source_row_index,
                        video_id,
                        source_video,
                        output_clip,
                        event_times,
                        video_duration,
                        clip_bounds,
                        args.pre_buffer,
                        args.post_buffer,
                        "dry_run",
                    )
                )
                continue

            ok, error = run_ffmpeg(
                source_video=source_video,
                output_clip=output_clip,
                start_time=clip_start_time,
                duration=clip_duration,
                overwrite=args.overwrite,
                copy_video=args.copy_video,
                include_audio=not args.no_audio,
                seek_mode=args.seek_mode,
                seek_preroll=args.seek_preroll,
                timeout_seconds=args.ffmpeg_timeout,
            )

            if ok:
                counts["created"] += 1
                status = "created"
            else:
                counts["failed"] += 1
                status = "ffmpeg_failed"

            rows.append(
                build_manifest_row(
                    source,
                    raw_row,
                    source_row_index,
                    video_id,
                    source_video,
                    output_clip,
                    event_times,
                    video_duration,
                    clip_bounds,
                    args.pre_buffer,
                    args.post_buffer,
                    status,
                    error,
                )
            )

            if processed_rows % 50 == 0:
                print(f"Processed {processed_rows} rows...")

    return rows, counts, processed_rows


def main() -> int:
    args = parse_args()
    sources = selected_sources(args)
    video_dir = args.video_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()

    if args.pre_buffer < 0 or args.post_buffer < 0:
        raise ValueError("--pre-buffer and --post-buffer must be non-negative.")
    if args.seek_preroll < 0:
        raise ValueError("--seek-preroll must be non-negative.")
    if args.ffprobe_timeout <= 0 or args.ffmpeg_timeout <= 0:
        raise ValueError("--ffprobe-timeout and --ffmpeg-timeout must be positive.")
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

    validate_sources(sources)

    print(f"Video directory: {video_dir}")
    print(f"Output root: {output_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Dry run: {args.dry_run}")
    print(f"Seek mode: {args.seek_mode}")
    print(f"Preserve audio: {not args.no_audio}")

    raw_fieldnames = prefixed_raw_fieldnames(sources)
    manifest_rows: list[dict[str, Any]] = []
    total_counts = {
        "created": 0,
        "exists": 0,
        "dry_run": 0,
        "skipped": 0,
        "failed": 0,
    }
    processed_rows = 0
    duration_cache: dict[Path, float | None] = {}

    for source in sources:
        if args.limit is not None and processed_rows >= args.limit:
            break

        print(f"Reading {source.event_type}: {source.csv_path}")
        rows, counts, processed_rows = process_source(
            source=source,
            video_dir=video_dir,
            output_root=output_root,
            args=args,
            start_processed_rows=processed_rows,
            duration_cache=duration_cache,
        )
        manifest_rows.extend(rows)
        for key, value in counts.items():
            total_counts[key] += value

    write_manifest(manifest_path, manifest_rows, raw_fieldnames)

    print("Done.")
    print(f"Manifest rows: {len(manifest_rows)}")
    print(f"Created clips: {total_counts['created']}")
    print(f"Existing clips: {total_counts['exists']}")
    print(f"Dry-run clips: {total_counts['dry_run']}")
    print(f"Skipped rows: {total_counts['skipped']}")
    print(f"Failed rows: {total_counts['failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
