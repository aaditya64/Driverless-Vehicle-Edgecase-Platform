"""Extract frozen VideoMAE embeddings for processed Nexar event clips."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .common import ensure_dirs, load_config, read_split_csv, stable_video_id, write_json

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


_register_torchvision_nms_stub()
from transformers import VideoMAEImageProcessor, VideoMAEModel


def _read_uniform_frames(video_path: str | Path, num_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = np.linspace(0, max(frame_count - 1, 0), num_frames).round().astype(int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]


def _summarize_tokens(tokens: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            tokens.mean(axis=0),
            tokens.std(axis=0),
            tokens.max(axis=0),
            tokens.min(axis=0),
            np.percentile(tokens, 25, axis=0),
            np.percentile(tokens, 75, axis=0),
        ],
        axis=0,
    ).astype(np.float32)


def _rows_from_csvs(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_split_csv(path):
            if row["path"] not in seen:
                rows.append(row)
                seen.add(row["path"])
    return rows


def extract_one(
    row: dict[str, Any],
    model: VideoMAEModel,
    processor: VideoMAEImageProcessor,
    device: torch.device,
    out_dir: Path,
    num_frames: int,
    force: bool,
) -> Path:
    video_path = row["path"]
    out_path = out_dir / f"{stable_video_id(video_path)}.npz"
    if out_path.exists() and not force:
        return out_path
    frames = _read_uniform_frames(video_path, num_frames)
    inputs = processor(frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        out = model(pixel_values=pixel_values)
    tokens = out.last_hidden_state.detach().float().cpu().numpy()[0]
    summary = _summarize_tokens(tokens)
    np.savez_compressed(
        out_path,
        tokens=tokens.astype(np.float32),
        summary=summary,
        video_path=str(video_path),
        model_name=str(model.config.name_or_path),
        num_frames=np.asarray([num_frames], dtype=np.int32),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wst_processed_744_val_small.yaml")
    parser.add_argument("--csv", action="append", default=None)
    parser.add_argument("--out-dir", default="outputs/processed_744/videomae_base_k400_16f")
    parser.add_argument("--model", default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    csvs = args.csv or [cfg["paths"]["train_csv"], cfg["paths"]["test_csv"]]
    rows = _rows_from_csvs(csvs)
    if args.limit > 0:
        rows = rows[: args.limit]
    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "cuda" else "cpu")
    print(f"device={device} model={args.model} videos={len(rows)}", flush=True)
    processor = VideoMAEImageProcessor.from_pretrained(args.model)
    model = VideoMAEModel.from_pretrained(args.model).eval().to(device)
    manifest = []
    for row in tqdm(rows, desc="videomae"):
        path = extract_one(row, model, processor, device, out_dir, args.num_frames, args.force)
        manifest.append({"path": row["path"], "label": int(row["label"]), "feature_path": str(path)})
    write_json(out_dir / "manifest.json", {"model": args.model, "count": len(manifest), "items": manifest})
    print(f"extracted {len(manifest)} VideoMAE feature files", flush=True)


if __name__ == "__main__":
    main()
