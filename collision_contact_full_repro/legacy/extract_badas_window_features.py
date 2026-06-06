#!/usr/bin/env python3
"""Extract BADAS/V-JEPA window-level features from processed event clips."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "BADAS-Open"
DEFAULT_MODEL_NAME = "facebook/vjepa2-vitl-fpc16-256-ssv2"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clip_manifests"
    / "nexar_train_positive_event_clips.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features"
_TORCHVISION_LIB = None


def _register_torchvision_nms_stub() -> None:
    global _TORCHVISION_LIB
    try:
        import torchvision  # noqa: F401
        return
    except RuntimeError as exc:
        if "torchvision::nms" not in str(exc):
            raise
    except Exception:
        return
    try:
        from torch.library import Library

        if _TORCHVISION_LIB is None:
            _TORCHVISION_LIB = Library("torchvision", "DEF")
            _TORCHVISION_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract classifier-input window embeddings and BADAS risk scores "
            "from processed 10-second positive clips."
        )
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        type=Path,
        help="Clip manifest CSV. Defaults to the Nexar positive event-clip manifest.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        type=Path,
        help="Optional directory of .mp4 clips. If provided, the manifest is not used.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory where .npz feature files and summaries will be written.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        type=Path,
        help="Path to the downloaded BADAS-Open directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device.",
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
        default=8,
        type=int,
        help="Stride, in sampled frames, between windows. Default 8 = 1 second at 8 fps.",
    )
    parser.add_argument(
        "--window-batch-size",
        default=3,
        type=int,
        help="Number of windows to run through BADAS in one forward pass.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Process only the first N clips. Useful for validation runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz feature files.",
    )
    parser.add_argument(
        "--summary-name",
        default="badas_window_features_summary.csv",
        help="Summary CSV filename inside output-dir.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force Hugging Face libraries to use local cache only.",
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
            raise RuntimeError("MPS was requested, but it is not available.")
    return device_arg


def validate_model_paths(model_dir: Path) -> tuple[Path, Path]:
    model_dir = model_dir.expanduser().resolve()
    src_dir = model_dir / "src"
    checkpoint_path = model_dir / "weights" / "badas_open.pth"

    if not src_dir.exists():
        raise FileNotFoundError(f"BADAS source directory does not exist: {src_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"BADAS checkpoint does not exist: {checkpoint_path}")
    return src_dir, checkpoint_path


def repo_relative_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_clip_rows(manifest_path: Path | None, input_dir: Path | None) -> list[dict[str, str]]:
    if input_dir is not None:
        input_dir = input_dir.expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        rows = []
        for path in sorted(input_dir.glob("*.mp4")):
            rows.append(
                {
                    "clip_id": path.stem,
                    "clip_path": repo_relative_path(path),
                    "source_file_name": "",
                    "core_label": "",
                    "event_center_time": "",
                    "center_offset_in_clip": "",
                }
            )
        return rows

    if manifest_path is None:
        raise ValueError("Either --manifest or --input-dir is required.")

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    usable_rows = []
    for row in rows:
        clip_path = str(row.get("clip_path", "")).strip()
        status = str(row.get("status", "")).strip()
        if not clip_path or status not in {"created", "exists", "dry_run"}:
            continue
        usable_rows.append(row)
    return usable_rows


def resolve_clip_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_local_badas_model(
    src_dir: Path,
    checkpoint_path: Path,
    device: str,
    target_fps: float,
    frame_count: int,
    window_stride: int,
):
    _register_torchvision_nms_stub()
    sys.path.insert(0, str(src_dir))

    try:
        import train.video_training  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import BADAS training modules from "
            f"{src_dir}. This usually means the active Python environment is "
            "missing a dependency used by models/BADAS-Open/src/train/"
            "video_training.py. Run `python -m pip install -r requirements.txt` "
            f"in the same environment, then retry. Original import error: {exc}"
        ) from exc

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
    model.load()
    return model


def preprocess_window_frames(model: Any, frames_array: np.ndarray) -> torch.Tensor:
    if model.processor:
        try:
            inputs = model.processor(videos=frames_array, return_tensors="pt")
            if "pixel_values_videos" in inputs:
                return inputs["pixel_values_videos"].squeeze(0)
            if "pixel_values" in inputs:
                return inputs["pixel_values"].squeeze(0)
            return list(inputs.values())[0].squeeze(0)
        except Exception as exc:
            print(f"Warning: processor failed ({exc}); using manual transform.")
    return model._manual_transform_frames(frames_array)


def extract_window_embeddings_and_risks(
    model: Any,
    processed_frames: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if processed_frames.dim() == 4:
        processed_frames = processed_frames.unsqueeze(0)

    classifier_model = model.model
    device = next(classifier_model.parameters()).device
    x = processed_frames.float().to(device)

    with torch.inference_mode():
        if not getattr(classifier_model, "use_custom_head", True):
            raise RuntimeError("Feature extraction currently expects BADAS custom-head model.")

        features = classifier_model._extract_video_features(x)
        if classifier_model.has_predictor and classifier_model.use_future_prediction:
            future_features = classifier_model._predict_future_features(features)
            features = classifier_model._combine_present_and_future(
                features,
                future_features,
            )

        embedding = classifier_model._apply_temporal_processing(features)
        logits = classifier_model.classifier(embedding)
        probs = torch.softmax(logits / 2.0, dim=1)

    embedding_np = embedding.detach().float().cpu().numpy()
    logits_np = logits.detach().float().cpu().numpy()
    risk_scores = probs[:, 1].detach().float().cpu().numpy()
    return embedding_np, risk_scores, logits_np


def extract_clip_features(
    model: Any,
    video_path: Path,
    target_fps: float,
    window_batch_size: int,
) -> dict[str, np.ndarray]:
    from utils.video import load_full_video_frames

    frames = load_full_video_frames(
        video_path=str(video_path),
        target_size=(224, 224),
        target_fps=target_fps,
    )
    total_frames = len(frames)
    windows = model.sliding_window_predictor.create_windows(total_frames)

    embeddings = []
    risk_scores = []
    logits = []
    window_start_idx = []
    window_end_idx = []
    target_frame_idx = []

    pending_tensors = []
    pending_windows = []

    def flush_pending() -> None:
        if not pending_tensors:
            return

        batch = torch.stack(pending_tensors, dim=0)
        batch_embeddings, batch_risks, batch_logits = extract_window_embeddings_and_risks(
            model,
            batch,
        )

        for local_idx, (start_idx, end_idx) in enumerate(pending_windows):
            embeddings.append(batch_embeddings[local_idx])
            risk_scores.append(float(batch_risks[local_idx]))
            logits.append(batch_logits[local_idx])
            window_start_idx.append(start_idx)
            window_end_idx.append(end_idx)
            target_frame_idx.append(end_idx)

        pending_tensors.clear()
        pending_windows.clear()

    for start_idx, end_idx in windows:
        window_frames = frames[start_idx:end_idx]
        padded_frames = model.sliding_window_predictor.pad_window_frames(
            window_frames,
            model.frame_count,
        )
        processed_frames = preprocess_window_frames(model, padded_frames)
        pending_tensors.append(processed_frames)
        pending_windows.append((start_idx, end_idx))

        if len(pending_tensors) >= window_batch_size:
            flush_pending()

    flush_pending()

    return {
        "features": np.stack(embeddings).astype(np.float32),
        "risk_scores": np.asarray(risk_scores, dtype=np.float32),
        "logits": np.stack(logits).astype(np.float32),
        "window_start_frame_idx": np.asarray(window_start_idx, dtype=np.int32),
        "window_end_frame_idx": np.asarray(window_end_idx, dtype=np.int32),
        "target_frame_idx": np.asarray(target_frame_idx, dtype=np.int32),
        "window_start_sec": np.asarray(window_start_idx, dtype=np.float32) / target_fps,
        "window_end_sec": np.asarray(window_end_idx, dtype=np.float32) / target_fps,
        "target_time_sec": np.asarray(target_frame_idx, dtype=np.float32) / target_fps,
        "total_sampled_frames": np.asarray(total_frames, dtype=np.int32),
    }


def finite_status(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all())


def summarize_feature_payload(
    row: dict[str, str],
    video_path: Path,
    output_path: Path,
    payload: dict[str, np.ndarray],
    status: str,
    error: str = "",
) -> dict[str, Any]:
    features = payload.get("features", np.empty((0, 0), dtype=np.float32))
    risk_scores = payload.get("risk_scores", np.empty((0,), dtype=np.float32))
    target_time_sec = payload.get("target_time_sec", np.empty((0,), dtype=np.float32))

    peak_idx = None
    peak_risk = None
    peak_time = None
    mean_risk = None
    if risk_scores.size and finite_status(risk_scores):
        peak_idx = int(np.argmax(risk_scores))
        peak_risk = round(float(risk_scores[peak_idx]), 6)
        peak_time = round(float(target_time_sec[peak_idx]), 3)
        mean_risk = round(float(np.mean(risk_scores)), 6)

    return {
        "clip_id": row.get("clip_id", video_path.stem),
        "status": status,
        "core_label": row.get("core_label", ""),
        "source_file_name": row.get("source_file_name", ""),
        "video_path": repo_relative_path(video_path),
        "feature_path": repo_relative_path(output_path) if status != "failed" else "",
        "num_windows": int(features.shape[0]) if features.ndim >= 1 else 0,
        "feature_dim": int(features.shape[1]) if features.ndim == 2 else "",
        "features_finite": finite_status(features) if features.size else False,
        "risk_finite": finite_status(risk_scores) if risk_scores.size else False,
        "peak_window_idx": peak_idx if peak_idx is not None else "",
        "peak_risk_score": peak_risk if peak_risk is not None else "",
        "peak_target_time_sec": peak_time if peak_time is not None else "",
        "mean_risk_score": mean_risk if mean_risk is not None else "",
        "event_center_time": row.get("event_center_time", ""),
        "center_offset_in_clip": row.get("center_offset_in_clip", ""),
        "error": error,
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "status",
        "core_label",
        "source_file_name",
        "video_path",
        "feature_path",
        "num_windows",
        "feature_dim",
        "features_finite",
        "risk_finite",
        "peak_window_idx",
        "peak_risk_score",
        "peak_target_time_sec",
        "mean_risk_score",
        "event_center_time",
        "center_offset_in_clip",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.window_batch_size < 1:
        raise ValueError("--window-batch-size must be >= 1.")

    src_dir, checkpoint_path = validate_model_paths(args.model_dir)
    device = resolve_device(args.device)
    rows = load_clip_rows(args.manifest, args.input_dir)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading BADAS-Open from: {args.model_dir}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Clips to process: {len(rows)}", flush=True)

    model = load_local_badas_model(
        src_dir=src_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        target_fps=args.target_fps,
        frame_count=args.frame_count,
        window_stride=args.window_stride,
    )
    if model.sliding_window_predictor is None:
        raise RuntimeError("BADAS model did not initialize a sliding-window predictor.")

    summary_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        clip_id = row.get("clip_id", "").strip()
        video_path = resolve_clip_path(row.get("clip_path", ""))
        output_path = output_dir / f"{clip_id or video_path.stem}.npz"

        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(rows)}] exists: {output_path.name}", flush=True)
            with np.load(output_path, allow_pickle=False) as existing:
                existing_payload = {
                    "features": existing["features"],
                    "risk_scores": existing["risk_scores"],
                    "target_time_sec": existing["target_time_sec"],
                }
            summary_rows.append(
                summarize_feature_payload(
                    row,
                    video_path,
                    output_path,
                    existing_payload,
                    "exists",
                )
            )
            continue

        try:
            if not video_path.exists():
                raise FileNotFoundError(f"Clip does not exist: {video_path}")

            payload = extract_clip_features(
                model,
                video_path,
                args.target_fps,
                args.window_batch_size,
            )
            metadata = {
                "clip_id": clip_id or video_path.stem,
                "video_path": repo_relative_path(video_path),
                "model_name": "BADAS-Open",
                "base_model": DEFAULT_MODEL_NAME,
                "checkpoint_path": str(checkpoint_path),
                "target_fps": args.target_fps,
                "frame_count": args.frame_count,
                "window_stride": args.window_stride,
                "feature_kind": "badas_classifier_input_embedding",
                "risk_kind": "badas_temperature_scaled_positive_probability",
                "core_label": row.get("core_label", ""),
                "event_center_time": row.get("event_center_time", ""),
                "center_offset_in_clip": row.get("center_offset_in_clip", ""),
            }

            np.savez_compressed(
                output_path,
                **payload,
                metadata=json.dumps(metadata),
            )

            summary = summarize_feature_payload(
                row,
                video_path,
                output_path,
                payload,
                "created",
            )
            summary_rows.append(summary)
            print(
                f"[{index}/{len(rows)}] created: {output_path.name} "
                f"windows={summary['num_windows']} dim={summary['feature_dim']} "
                f"peak={summary['peak_risk_score']}@{summary['peak_target_time_sec']}s",
                flush=True,
            )
        except Exception as exc:
            error = str(exc)
            print(f"[{index}/{len(rows)}] failed: {video_path} :: {error}", flush=True)
            summary_rows.append(
                summarize_feature_payload(
                    row,
                    video_path,
                    output_path,
                    {},
                    "failed",
                    error,
                )
            )

    summary_path = output_dir / args.summary_name
    write_summary(summary_path, summary_rows)
    write_run_config(
        output_dir / "badas_window_features_run_config.json",
        {
            "manifest": str(args.manifest.expanduser().resolve())
            if args.manifest is not None
            else None,
            "input_dir": str(args.input_dir.expanduser().resolve())
            if args.input_dir is not None
            else None,
            "output_dir": str(output_dir),
            "model_dir": str(args.model_dir.expanduser().resolve()),
            "checkpoint_path": str(checkpoint_path),
            "device": device,
            "target_fps": args.target_fps,
            "frame_count": args.frame_count,
            "window_stride": args.window_stride,
            "window_batch_size": args.window_batch_size,
            "limit": args.limit,
            "offline": args.offline,
            "feature_kind": "badas_classifier_input_embedding",
        },
    )

    failed_count = sum(1 for row in summary_rows if row["status"] == "failed")
    created_count = sum(1 for row in summary_rows if row["status"] == "created")
    print(
        f"Wrote summary: {summary_path} "
        f"(created={created_count}, failed={failed_count})",
        flush=True,
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
