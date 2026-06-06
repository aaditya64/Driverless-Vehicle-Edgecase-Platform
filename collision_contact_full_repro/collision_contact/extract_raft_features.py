"""Extract RAFT dense optical-flow statistics for dashcam videos."""

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


def _load_raft(device: torch.device) -> tuple[torch.nn.Module, Any]:
    _register_torchvision_nms_stub()
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

    weights = Raft_Small_Weights.C_T_V2
    model = raft_small(weights=weights, progress=True).eval().to(device)
    return model, weights.transforms()


def _resize_for_raft(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / max(w, 1)
    new_h = max(8, int(round(h * scale / 8.0)) * 8)
    resized = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
    return resized


def _read_pair(cap: cv2.VideoCapture, idx1: int, idx2: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    frames = []
    for idx in [idx1, idx2]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(_resize_for_raft(frame, width))
    return frames[0], frames[1]


def _tensor_from_rgb(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(frame, (2, 0, 1))).unsqueeze(0)


def _stats(values: np.ndarray) -> list[float]:
    v = values.reshape(-1)
    if v.size == 0:
        return [0.0] * 8
    return [
        float(np.mean(v)),
        float(np.std(v)),
        float(np.percentile(v, 50)),
        float(np.percentile(v, 75)),
        float(np.percentile(v, 90)),
        float(np.percentile(v, 95)),
        float(np.percentile(v, 99)),
        float(np.max(v)),
    ]


def _direction_entropy(dx: np.ndarray, dy: np.ndarray, mag: np.ndarray, bins: int = 16) -> float:
    angle = np.arctan2(dy, dx)
    hist, _ = np.histogram(angle, bins=bins, range=(-np.pi, np.pi), weights=mag)
    total = float(hist.sum())
    if total <= 1e-8:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(bins))


def _affine_residual(flow: np.ndarray) -> tuple[np.ndarray, float, float]:
    h, w = flow.shape[:2]
    step = max(8, min(h, w) // 24)
    yy, xx = np.mgrid[step // 2 : h : step, step // 2 : w : step]
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    sampled = flow[yy.reshape(-1), xx.reshape(-1)].astype(np.float32)
    dst = pts + sampled
    if pts.shape[0] < 12:
        return flow, 0.0, 0.0
    mat, inliers = cv2.estimateAffinePartial2D(
        pts,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=1000,
        confidence=0.99,
    )
    if mat is None:
        median = np.median(flow.reshape(-1, 2), axis=0)
        residual = flow - median.reshape(1, 1, 2)
        return residual, 0.0, 0.0
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    expected_x = mat[0, 0] * grid_x + mat[0, 1] * grid_y + mat[0, 2] - grid_x
    expected_y = mat[1, 0] * grid_x + mat[1, 1] * grid_y + mat[1, 2] - grid_y
    expected = np.stack([expected_x, expected_y], axis=2).astype(np.float32)
    residual = flow - expected
    inlier_ratio = float(inliers.mean()) if inliers is not None else 0.0
    fitted = cv2.transform(pts.reshape(-1, 1, 2), mat).reshape(-1, 2)
    fit_error = float(np.median(np.linalg.norm(dst - fitted, axis=1)))
    return residual, inlier_ratio, fit_error


def _grid_means(mag: np.ndarray, grid: int = 4) -> list[float]:
    h, w = mag.shape
    vals = []
    for gy in range(grid):
        y0 = int(round(gy * h / grid))
        y1 = int(round((gy + 1) * h / grid))
        for gx in range(grid):
            x0 = int(round(gx * w / grid))
            x1 = int(round((gx + 1) * w / grid))
            vals.append(float(np.mean(mag[y0:y1, x0:x1])))
    return vals


def _flow_features(flow: np.ndarray) -> np.ndarray:
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    residual, inlier_ratio, fit_error = _affine_residual(flow)
    rdx = residual[..., 0]
    rdy = residual[..., 1]
    rmag = np.sqrt(rdx * rdx + rdy * rdy)

    h, w = mag.shape
    border = np.zeros_like(mag, dtype=bool)
    border[: h // 5, :] = True
    border[-h // 5 :, :] = True
    border[:, : w // 5] = True
    border[:, -w // 5 :] = True
    center = ~border

    features: list[float] = []
    features += _stats(mag)
    features += _stats(rmag)
    features += _stats(np.abs(dx))
    features += _stats(np.abs(dy))
    features += [float(np.mean(rmag[border])), float(np.mean(rmag[center])), float(np.mean(rmag[border]) / (np.mean(rmag[center]) + 1e-6))]
    grid_vals = _grid_means(rmag, grid=4)
    features += grid_vals
    features += [float(max(grid_vals)), float(np.std(grid_vals))]
    features += [_direction_entropy(dx, dy, mag), _direction_entropy(rdx, rdy, rmag)]
    features += [inlier_ratio, fit_error]
    return np.asarray(features, dtype=np.float32)


def extract_one(
    video_path: str | Path,
    model: torch.nn.Module,
    transforms: Any,
    device: torch.device,
    out_dir: str | Path,
    num_pairs: int,
    pair_gap: int,
    width: int,
    batch_size: int,
    force: bool = False,
) -> Path:
    out_path = Path(out_dir) / f"{stable_video_id(video_path)}.npz"
    if out_path.exists() and not force:
        return out_path
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_start = max(frame_count - pair_gap - 1, 1)
    starts = np.linspace(0, max_start, num_pairs).round().astype(int)
    pending_img1: list[torch.Tensor] = []
    pending_img2: list[torch.Tensor] = []
    pair_features = []
    def flush_pending() -> None:
        if not pending_img1:
            return
        img1 = torch.cat(pending_img1, dim=0)
        img2 = torch.cat(pending_img2, dim=0)
        pending_img1.clear()
        pending_img2.clear()
        img1, img2 = transforms(img1, img2)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            flows = model(img1.to(device), img2.to(device))[-1].detach().float().cpu().numpy()
        flows = np.transpose(flows, (0, 2, 3, 1))
        for flow in flows:
            pair_features.append(_flow_features(flow))

    for start in starts:
        pair = _read_pair(cap, int(start), int(start + pair_gap), width)
        if pair is None:
            continue
        pending_img1.append(_tensor_from_rgb(pair[0]))
        pending_img2.append(_tensor_from_rgb(pair[1]))
        if len(pending_img1) >= batch_size:
            flush_pending()
    flush_pending()
    cap.release()
    if not pair_features:
        raise RuntimeError(f"No RAFT pairs decoded from video: {video_path}")
    pair_arr = np.stack(pair_features).astype(np.float32)
    summary = np.concatenate([pair_arr.mean(axis=0), pair_arr.std(axis=0), pair_arr.max(axis=0), pair_arr.min(axis=0)]).astype(np.float32)
    np.savez_compressed(
        out_path,
        pair_features=pair_arr,
        summary=summary,
        video_path=str(video_path),
        num_pairs=np.asarray([num_pairs], dtype=np.int32),
        pair_gap=np.asarray([pair_gap], dtype=np.int32),
        width=np.asarray([width], dtype=np.int32),
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
    parser.add_argument("--out-dir", default="outputs/processed_744/raft_small")
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--pair-gap", type=int, default=2)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    csvs = args.csv or [cfg["paths"]["train_csv"], cfg["paths"]["test_csv"]]
    rows = _rows_from_csvs(csvs)
    if args.limit is not None:
        rows = rows[: args.limit]
    ensure_dirs(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "cuda" else "cpu")
    model, transforms = _load_raft(device)
    print(f"device={device} videos={len(rows)} pairs={args.num_pairs} width={args.width}")
    manifest = []
    for row in tqdm(rows, desc="raft"):
        out_path = extract_one(
            row["path"],
            model,
            transforms,
            device,
            args.out_dir,
            args.num_pairs,
            args.pair_gap,
            args.width,
            args.batch_size,
            force=args.force,
        )
        manifest.append({"path": row["path"], "label": int(row["label"]), "feature_path": str(out_path)})
    write_json(Path(args.out_dir) / "manifest.json", {"count": len(manifest), "items": manifest})
    print(f"extracted {len(manifest)} RAFT feature files")


if __name__ == "__main__":
    main()
