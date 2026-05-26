"""Extract frozen DINOv2 frame embeddings for dashcam videos."""

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


def _load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    _register_torchvision_nms_stub()
    import timm

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device)
    return model


def _read_uniform_frames(video_path: str | Path, num_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        frame_count = 300
    indices = np.linspace(0, max(frame_count - 1, 0), num_frames).round().astype(int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]


def _letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _preprocess(frames: list[np.ndarray], image_size: int) -> torch.Tensor:
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    batch = []
    for frame in frames:
        img = _letterbox(frame, image_size).astype(np.float32) / 255.0
        img = (img - mean) / std
        batch.append(np.transpose(img, (2, 0, 1)))
    return torch.from_numpy(np.stack(batch).astype(np.float32))


def _summarize_embeddings(emb: np.ndarray) -> np.ndarray:
    diffs = np.abs(np.diff(emb, axis=0)) if emb.shape[0] > 1 else np.zeros_like(emb)
    parts = [
        emb.mean(axis=0),
        emb.std(axis=0),
        emb.max(axis=0),
        emb.min(axis=0),
        diffs.mean(axis=0),
        diffs.max(axis=0),
    ]
    return np.concatenate(parts).astype(np.float32)


def extract_one(
    video_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    out_dir: str | Path,
    num_frames: int,
    image_size: int,
    batch_size: int,
    force: bool = False,
) -> Path:
    out_path = Path(out_dir) / f"{stable_video_id(video_path)}.npz"
    if out_path.exists() and not force:
        return out_path
    frames = _read_uniform_frames(video_path, num_frames)
    x = _preprocess(frames, image_size)
    embeddings = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = x[start : start + batch_size].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                y = model(batch)
            embeddings.append(y.detach().float().cpu().numpy())
    frame_embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    summary = _summarize_embeddings(frame_embeddings)
    np.savez_compressed(
        out_path,
        frame_embeddings=frame_embeddings,
        summary=summary,
        video_path=str(video_path),
        model_name=getattr(model, "default_cfg", {}).get("architecture", "unknown"),
        num_frames=np.asarray([num_frames], dtype=np.int32),
        image_size=np.asarray([image_size], dtype=np.int32),
    )
    return out_path


def _rows_from_csvs(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_split_csv(path):
            if row["path"] not in seen:
                rows.append(row)
                seen.add(row["path"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wst_processed_744_small.yaml")
    parser.add_argument("--csv", action="append", default=None)
    parser.add_argument("--out-dir", default="outputs/processed_744/dino_vits14")
    parser.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    csvs = args.csv or [cfg["paths"]["train_csv"], cfg["paths"]["test_csv"]]
    rows = _rows_from_csvs(csvs)
    if args.limit > 0:
        rows = rows[: args.limit]
    ensure_dirs(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "cuda" else "cpu")
    print(f"device={device} model={args.model} videos={len(rows)}")
    model = _load_model(args.model, device)
    manifest = []
    for row in tqdm(rows, desc="dino"):
        out_path = extract_one(
            row["path"],
            model,
            device,
            args.out_dir,
            args.num_frames,
            args.image_size,
            args.batch_size,
            force=args.force,
        )
        manifest.append({"path": row["path"], "label": int(row["label"]), "feature_path": str(out_path)})
    write_json(Path(args.out_dir) / "manifest.json", {"model": args.model, "count": len(manifest), "items": manifest})
    print(f"extracted {len(manifest)} DINO feature files")


if __name__ == "__main__":
    main()
