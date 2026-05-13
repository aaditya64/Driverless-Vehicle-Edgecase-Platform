#!/usr/bin/env python3
"""Create fixed-length event-centred clips from Nexar positive videos."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "nexar_collision_prediction"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "event_clips" / "nexar" / "train" / "positive"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clip_manifests"
    / "nexar_train_positive_event_clips.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut 10-second event-centred clips from Nexar positive videos."
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        type=Path,
        help="Root directory of the Nexar collision prediction dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to process. Default: train.",
    )
    parser.add_argument(
        "--label-folder",
        default="positive",
        choices=["positive", "negative"],
        help="Nexar label folder to process. Default: positive.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        type=Path,
        help="Optional metadata CSV. Defaults to <data-root>/<split>/<label-folder>/metadata.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory where clipped mp4 files will be written.",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        type=Path,
        help="Output CSV manifest path.",
    )
    parser.add_argument(
        "--center-field",
        default="time_of_event",
        choices=["time_of_event", "time_of_alert"],
        help="Metadata column used as the event centre timestamp.",
    )
    parser.add_argument(
        "--clip-duration",
        default=10.0,
        type=float,
        help="Clip duration in seconds. Default: 10.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Process only the first N rows. Useful for testing.",
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
        "--dry-run",
        action="store_true",
        help="Write the manifest without running ffmpeg.",
    )
    return parser.parse_args()


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


def repo_relative_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def ffprobe_duration(video_path: Path) -> float | None:
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return parse_float(result.stdout)


def compute_clip_bounds(
    event_center_time: float,
    video_duration: float | None,
    clip_duration: float,
) -> tuple[float, float, float]:
    desired_start = event_center_time - clip_duration / 2.0

    if video_duration is None:
        start_time = max(0.0, desired_start)
        actual_duration = clip_duration
        center_offset = event_center_time - start_time
        return start_time, actual_duration, center_offset

    if video_duration <= clip_duration:
        start_time = 0.0
        actual_duration = max(0.0, video_duration)
        center_offset = event_center_time
        return start_time, actual_duration, center_offset

    latest_start = video_duration - clip_duration
    start_time = min(max(0.0, desired_start), latest_start)
    center_offset = event_center_time - start_time
    return start_time, clip_duration, center_offset


def make_clip_id(split: str, label_folder: str, file_name: str) -> str:
    stem = Path(file_name).stem
    return f"nexar_{split}_{label_folder}_{stem}"


def run_ffmpeg(
    source_video: Path,
    output_clip: Path,
    start_time: float,
    duration: float,
    overwrite: bool,
    copy_video: bool,
) -> tuple[bool, str]:
    output_clip.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    cmd.extend(["-y" if overwrite else "-n"])
    cmd.extend(
        [
            "-ss",
            f"{start_time:.3f}",
            "-i",
            str(source_video),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-an",
        ]
    )

    if copy_video:
        cmd.extend(["-c:v", "copy"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])

    cmd.append(str(output_clip))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return True, ""

    error = result.stderr.strip() or result.stdout.strip()
    return False, error


def build_manifest_row(
    row: dict[str, str],
    split: str,
    label_folder: str,
    source_video: Path,
    output_clip: Path,
    clip_id: str,
    event_center_time: float | None,
    video_duration: float | None,
    clip_start_time: float | None,
    clip_duration: float | None,
    center_offset: float | None,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    clip_end_time = None
    if clip_start_time is not None and clip_duration is not None:
        clip_end_time = clip_start_time + clip_duration

    return {
        "clip_id": clip_id,
        "source_dataset": "nexar_collision_prediction",
        "split": split,
        "source_label_folder": label_folder,
        "source_binary_label": "collision_or_near_miss" if label_folder == "positive" else "safe",
        "core_label": "needs_review" if label_folder == "positive" else "safe",
        "source_file_name": row.get("file_name", ""),
        "source_video_path": repo_relative_path(source_video),
        "clip_path": (
            repo_relative_path(output_clip)
            if status in {"created", "exists", "dry_run"}
            else ""
        ),
        "event_center_time": round_time(event_center_time),
        "video_duration": round_time(video_duration),
        "clip_start_time": round_time(clip_start_time),
        "clip_end_time": round_time(clip_end_time),
        "clip_duration": round_time(clip_duration),
        "center_offset_in_clip": round_time(center_offset),
        "time_of_event": row.get("time_of_event", ""),
        "time_of_alert": row.get("time_of_alert", ""),
        "light_conditions": row.get("light_conditions", ""),
        "weather": row.get("weather", ""),
        "scene": row.get("scene", ""),
        "time_to_accident": row.get("time_to_accident", ""),
        "status": status,
        "error": error,
    }


def write_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No manifest rows were produced.")

    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    metadata_path = (
        args.metadata.expanduser().resolve()
        if args.metadata
        else data_root / args.split / args.label_folder / "metadata.csv"
    )
    video_dir = data_root / args.split / args.label_folder
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_path}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

    print(f"Reading metadata: {metadata_path}")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {manifest_path}")

    manifest_rows: list[dict[str, Any]] = []
    created_count = 0
    exists_count = 0
    skipped_count = 0
    failed_count = 0

    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if args.limit is not None and row_idx >= args.limit:
                break

            file_name = row.get("file_name", "").strip()
            clip_id = make_clip_id(args.split, args.label_folder, file_name)
            source_video = video_dir / file_name
            output_clip = output_dir / f"{clip_id}.mp4"
            event_center_time = parse_float(row.get(args.center_field))

            if not file_name:
                skipped_count += 1
                manifest_rows.append(
                    build_manifest_row(
                        row,
                        args.split,
                        args.label_folder,
                        source_video,
                        output_clip,
                        clip_id,
                        event_center_time,
                        None,
                        None,
                        None,
                        None,
                        "missing_file_name",
                    )
                )
                continue

            if not source_video.exists():
                skipped_count += 1
                manifest_rows.append(
                    build_manifest_row(
                        row,
                        args.split,
                        args.label_folder,
                        source_video,
                        output_clip,
                        clip_id,
                        event_center_time,
                        None,
                        None,
                        None,
                        None,
                        "missing_video",
                    )
                )
                continue

            if event_center_time is None:
                skipped_count += 1
                manifest_rows.append(
                    build_manifest_row(
                        row,
                        args.split,
                        args.label_folder,
                        source_video,
                        output_clip,
                        clip_id,
                        event_center_time,
                        None,
                        None,
                        None,
                        None,
                        f"missing_{args.center_field}",
                    )
                )
                continue

            video_duration = ffprobe_duration(source_video)
            clip_start_time, actual_duration, center_offset = compute_clip_bounds(
                event_center_time=event_center_time,
                video_duration=video_duration,
                clip_duration=args.clip_duration,
            )

            if output_clip.exists() and not args.overwrite:
                exists_count += 1
                manifest_rows.append(
                    build_manifest_row(
                        row,
                        args.split,
                        args.label_folder,
                        source_video,
                        output_clip,
                        clip_id,
                        event_center_time,
                        video_duration,
                        clip_start_time,
                        actual_duration,
                        center_offset,
                        "exists",
                    )
                )
                continue

            if args.dry_run:
                manifest_rows.append(
                    build_manifest_row(
                        row,
                        args.split,
                        args.label_folder,
                        source_video,
                        output_clip,
                        clip_id,
                        event_center_time,
                        video_duration,
                        clip_start_time,
                        actual_duration,
                        center_offset,
                        "dry_run",
                    )
                )
                continue

            ok, error = run_ffmpeg(
                source_video=source_video,
                output_clip=output_clip,
                start_time=clip_start_time,
                duration=actual_duration,
                overwrite=args.overwrite,
                copy_video=args.copy_video,
            )

            if ok:
                created_count += 1
                status = "created"
            else:
                failed_count += 1
                status = "ffmpeg_failed"

            manifest_rows.append(
                build_manifest_row(
                    row,
                    args.split,
                    args.label_folder,
                    source_video,
                    output_clip,
                    clip_id,
                    event_center_time,
                    video_duration,
                    clip_start_time,
                    actual_duration,
                    center_offset,
                    status,
                    error,
                )
            )

            if (row_idx + 1) % 50 == 0:
                print(f"Processed {row_idx + 1} rows...")

    write_manifest(manifest_path, manifest_rows)

    print("Done.")
    print(f"Manifest rows: {len(manifest_rows)}")
    print(f"Created clips: {created_count}")
    print(f"Existing clips: {exists_count}")
    print(f"Skipped rows: {skipped_count}")
    print(f"Failed rows: {failed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
