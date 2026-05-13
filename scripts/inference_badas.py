#!/usr/bin/env python3
"""Run BADAS-Open inference on a dashcam video.

The local BADAS-Open loader tries to download a separate repo, so this script
loads the downloaded source and checkpoint directly from models/BADAS-Open.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "BADAS-Open"
DEFAULT_MODEL_NAME = "facebook/vjepa2-vitl-fpc16-256-ssv2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BADAS-Open risk prediction on one dashcam video."
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to an input dashcam .mp4 video.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        type=Path,
        help="Path to the downloaded BADAS-Open directory.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        type=Path,
        help="Optional path for the structured JSON output.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        type=Path,
        help="Optional path for a timestamp/risk_score CSV timeline.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device. Use auto unless you know what is available.",
    )
    parser.add_argument(
        "--threshold",
        default=0.8,
        type=float,
        help="Risk threshold used to mark high-risk timestamps.",
    )
    parser.add_argument(
        "--target-fps",
        default=8.0,
        type=float,
        help="Frame sampling rate used by BADAS.",
    )
    parser.add_argument(
        "--frame-count",
        default=16,
        type=int,
        help="Number of sampled frames per BADAS window.",
    )
    parser.add_argument(
        "--window-stride",
        default=1,
        type=int,
        help="Stride, in sampled frames, between sliding windows.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if device_arg == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            raise RuntimeError("MPS was requested, but it is not available in this Python.")
    return device_arg


def validate_paths(video_path: Path, model_dir: Path) -> tuple[Path, Path, Path]:
    video_path = video_path.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    src_dir = model_dir / "src"
    checkpoint_path = model_dir / "weights" / "badas_open.pth"

    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    if not src_dir.exists():
        raise FileNotFoundError(f"BADAS source directory does not exist: {src_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"BADAS checkpoint does not exist: {checkpoint_path}")

    return video_path, src_dir, checkpoint_path


def load_local_badas_model(
    src_dir: Path,
    checkpoint_path: Path,
    device: str,
    target_fps: float,
    frame_count: int,
    window_stride: int,
):
    sys.path.insert(0, str(src_dir))

    from models.vjepa import VJEPAModel

    model = VJEPAModel(
        model_name=DEFAULT_MODEL_NAME,
        checkpoint_path=str(checkpoint_path),
        device=device,
        frame_count=frame_count,
        img_size=224,
        window_stride=window_stride,
        target_fps=target_fps,
        use_sliding_window=True,
    )
    try:
        model.load()
    except RuntimeError as exc:
        message = str(exc)
        if "model type `vjepa2`" in message or "couldn't connect" in message:
            raise RuntimeError(
                "Failed to load the V-JEPA2 base model used by BADAS-Open. "
                "Install a Transformers version with V-JEPA2 support "
                "(`python -m pip install 'transformers==4.57.3'`) and make "
                "sure Hugging Face can download or find "
                f"{DEFAULT_MODEL_NAME} in the local cache."
            ) from exc
        raise
    return model


def finite_probabilities(predictions: np.ndarray) -> np.ndarray:
    preds = np.asarray(predictions, dtype=np.float32)
    if preds.ndim != 1:
        preds = preds.reshape(-1)
    return preds


def make_timeline(predictions: np.ndarray, target_fps: float) -> list[dict[str, Any]]:
    timeline = []
    for sampled_frame_idx, prob in enumerate(predictions):
        risk_score = None if math.isnan(float(prob)) else round(float(prob), 6)
        timeline.append(
            {
                "sampled_frame_idx": sampled_frame_idx,
                "time_sec": round(sampled_frame_idx / target_fps, 3),
                "risk_score": risk_score,
            }
        )
    return timeline


def high_risk_segments(
    predictions: np.ndarray,
    target_fps: float,
    threshold: float,
) -> list[dict[str, Any]]:
    segments = []
    start_idx = None
    last_idx = None

    for idx, prob in enumerate(predictions):
        is_high = not math.isnan(float(prob)) and float(prob) >= threshold
        if is_high and start_idx is None:
            start_idx = idx
        if is_high:
            last_idx = idx
        if not is_high and start_idx is not None:
            segments.append(
                {
                    "start_sec": round(start_idx / target_fps, 3),
                    "end_sec": round(last_idx / target_fps, 3),
                    "start_sampled_frame_idx": start_idx,
                    "end_sampled_frame_idx": last_idx,
                }
            )
            start_idx = None
            last_idx = None

    if start_idx is not None:
        segments.append(
            {
                "start_sec": round(start_idx / target_fps, 3),
                "end_sec": round(last_idx / target_fps, 3),
                "start_sampled_frame_idx": start_idx,
                "end_sampled_frame_idx": last_idx,
            }
        )

    return segments


def summarize_predictions(
    predictions: np.ndarray,
    target_fps: float,
    threshold: float,
) -> dict[str, Any]:
    valid_mask = ~np.isnan(predictions)
    if not valid_mask.any():
        return {
            "predicted_label": "unknown",
            "peak_risk_score": None,
            "peak_risk_time_sec": None,
            "peak_risk_sampled_frame_idx": None,
            "mean_risk_score": None,
            "high_risk_segments": [],
            "note": "All predictions were NaN. The video may be too short or unreadable.",
        }

    peak_idx = int(np.nanargmax(predictions))
    peak_score = float(predictions[peak_idx])
    mean_score = float(np.nanmean(predictions))
    predicted_label = "collision_or_near_miss" if peak_score >= threshold else "safe"

    return {
        "predicted_label": predicted_label,
        "peak_risk_score": round(peak_score, 6),
        "peak_risk_time_sec": round(peak_idx / target_fps, 3),
        "peak_risk_sampled_frame_idx": peak_idx,
        "mean_risk_score": round(mean_score, 6),
        "high_risk_segments": high_risk_segments(predictions, target_fps, threshold),
        "note": (
            "BADAS-Open is a binary risk model. It does not separate near_miss "
            "from collision without an additional classifier or relabelled data."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_csv(path: Path, timeline: list[dict[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sampled_frame_idx", "time_sec", "risk_score"],
        )
        writer.writeheader()
        writer.writerows(timeline)


def main() -> int:
    args = parse_args()
    video_path, src_dir, checkpoint_path = validate_paths(args.video, args.model_dir)
    device = resolve_device(args.device)

    print(f"Loading BADAS-Open from: {args.model_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    model = load_local_badas_model(
        src_dir=src_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        target_fps=args.target_fps,
        frame_count=args.frame_count,
        window_stride=args.window_stride,
    )

    print(f"Running inference on: {video_path}")
    predictions = finite_probabilities(model.predict(str(video_path)))
    timeline = make_timeline(predictions, args.target_fps)
    summary = summarize_predictions(predictions, args.target_fps, args.threshold)

    payload = {
        "video_path": str(video_path),
        "model": {
            "name": "BADAS-Open",
            "base_model": DEFAULT_MODEL_NAME,
            "checkpoint_path": str(checkpoint_path),
            "target_fps": args.target_fps,
            "frame_count": args.frame_count,
            "window_stride": args.window_stride,
            "threshold": args.threshold,
            "device": device,
        },
        "summary": summary,
        "timeline": timeline,
    }

    print(json.dumps({"summary": summary}, indent=2))

    if args.output_json:
        write_json(args.output_json, payload)
        print(f"Wrote JSON: {args.output_json}")

    if args.output_csv:
        write_csv(args.output_csv, timeline)
        print(f"Wrote CSV: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
