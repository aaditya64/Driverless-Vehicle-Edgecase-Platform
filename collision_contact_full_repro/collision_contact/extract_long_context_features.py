"""Extract low-FPS post-event behavior features from original 40s Nexar videos."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import ensure_dirs, stable_video_id, write_json
from .extract_yolo_interaction_features import _frame_features, _ultralytics_detections


def _resize(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if width <= 0 or w == width:
        return frame
    scale = width / max(w, 1)
    return cv2.resize(frame, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _read_timed_frames(video_path: str | Path, fps: float, width: int) -> tuple[np.ndarray, list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / src_fps if src_fps > 0 else 40.0
    times = np.arange(0.0, max(duration, 0.01), 1.0 / fps, dtype=np.float32)
    frames: list[np.ndarray] = []
    valid_times: list[float] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(_resize(frame, width))
        valid_times.append(float(t))
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"Too few sampled frames: {video_path}")
    return np.asarray(valid_times, dtype=np.float32), frames, duration


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _affine_pair(prev: np.ndarray, curr: np.ndarray) -> tuple[float, float, float, float, float, float]:
    p0 = cv2.goodFeaturesToTrack(prev, maxCorners=1200, qualityLevel=0.01, minDistance=5, blockSize=7)
    if p0 is None or len(p0) < 40:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    lk_params = dict(
        winSize=(25, 25),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **lk_params)
    if p1 is None or status is None:
        return 0.0, 0.0, 0.0, 0.0, float(len(p0)), 0.0
    keep = status.reshape(-1).astype(bool)
    q0 = p0.reshape(-1, 2)[keep]
    q1 = p1.reshape(-1, 2)[keep]
    if len(q0) < 40:
        return 0.0, 0.0, 0.0, 0.0, float(len(q0)), 0.0
    affine, inliers = cv2.estimateAffinePartial2D(q0, q1, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99)
    if affine is None or inliers is None:
        return 0.0, 0.0, 0.0, 0.0, float(len(q0)), 0.0
    mask = inliers.reshape(-1).astype(bool)
    if mask.sum() < 20:
        return 0.0, 0.0, 0.0, 0.0, float(len(q0)), float(mask.sum())
    pred = cv2.transform(q0.reshape(-1, 1, 2), affine).reshape(-1, 2)
    residual = np.linalg.norm(pred - q1, axis=1)
    a, _b, tx = affine[0]
    c, _d, ty = affine[1]
    theta_px = float(np.arctan2(c, a) * prev.shape[1])
    fit_error = float(np.median(residual[mask]))
    inlier_ratio = float(mask.sum() / max(len(q0), 1))
    return float(tx), float(ty), theta_px, fit_error, float(len(q0)), inlier_ratio


def _series_from_frames(times: np.ndarray, frames: list[np.ndarray], motion_mode: str) -> dict[str, np.ndarray]:
    grays = [_gray(f) for f in frames]
    rows = []
    prev = grays[0]
    for i, curr in enumerate(grays):
        if i == 0:
            dx = dy = theta_px = fit_error = num_tracks = inlier_ratio = frame_diff = 0.0
        else:
            frame_diff = float(np.mean(np.abs(curr.astype(np.float32) - prev.astype(np.float32))) / 255.0)
            if motion_mode == "affine":
                dx, dy, theta_px, fit_error, num_tracks, inlier_ratio = _affine_pair(prev, curr)
            else:
                dx = dy = theta_px = fit_error = num_tracks = inlier_ratio = 0.0
        brightness = float(np.mean(curr) / 255.0)
        blur = float(cv2.Laplacian(curr, cv2.CV_32F).var())
        speed = float(np.sqrt(dx * dx + dy * dy + theta_px * theta_px)) if motion_mode == "affine" else frame_diff * 100.0
        rows.append([speed, frame_diff, dx, dy, theta_px, fit_error, num_tracks, inlier_ratio, brightness, blur])
        prev = curr
    arr = np.asarray(rows, dtype=np.float32)
    return {
        "time": times,
        "speed": arr[:, 0],
        "frame_diff": arr[:, 1],
        "dx": arr[:, 2],
        "dy": arr[:, 3],
        "theta_px": arr[:, 4],
        "fit_error": arr[:, 5],
        "num_tracks": arr[:, 6],
        "inlier_ratio": arr[:, 7],
        "brightness": arr[:, 8],
        "blur": arr[:, 9],
    }


def _yolo_series(model: Any, frames: list[np.ndarray], batch_size: int) -> dict[str, np.ndarray]:
    feats = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        results = model.predict(batch, imgsz=640, conf=0.2, iou=0.5, verbose=False)
        for frame, result in zip(batch, results):
            dets = _ultralytics_detections(result)
            h, w = frame.shape[:2]
            feats.append(_frame_features(dets, w, h))
    arr = np.stack(feats).astype(np.float32)
    return {
        "vehicle_count": arr[:, 0],
        "vehicle_max_conf": arr[:, 1],
        "vehicle_max_area": arr[:, 2],
        "vehicle_sum_area": arr[:, 3],
        "vehicle_max_bottom": arr[:, 7],
        "vehicle_min_center_dist": arr[:, 8],
        "vehicle_max_threat": arr[:, 10],
        "risk_count": arr[:, 14],
        "risk_max_area": arr[:, 16],
        "risk_max_threat": arr[:, 24],
        "person_count": arr[:, 28],
        "bicycle_count": arr[:, 29],
        "car_count": arr[:, 30],
        "motorcycle_count": arr[:, 31],
        "bus_count": arr[:, 32],
        "truck_count": arr[:, 33],
    }


def _window_mask(times: np.ndarray, start: float, end: float) -> np.ndarray:
    return (times >= start) & (times < end)


def _safe_stats(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [0.0] * 8
    if x.size > 1:
        slope = float(np.polyfit(np.linspace(0.0, 1.0, x.size), x, 1)[0])
    else:
        slope = 0.0
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.percentile(x, 50)),
        float(np.percentile(x, 90)),
        float(x[-1] - x[0]),
        slope,
    ]


def _summarize(series: dict[str, np.ndarray], event_time: float, duration: float) -> tuple[np.ndarray, list[str]]:
    times = series["time"]
    windows = {
        "pre": (max(0.0, event_time - 8.0), max(0.0, event_time - 1.0)),
        "event": (max(0.0, event_time - 1.0), min(duration, event_time + 2.0)),
        "early_post": (min(duration, event_time + 2.0), min(duration, event_time + 10.0)),
        "late_post": (min(duration, event_time + 10.0), min(duration, event_time + 18.0)),
        "post": (min(duration, event_time + 2.0), duration),
        "full": (0.0, duration),
    }
    feature_names: list[str] = []
    values: list[float] = []
    channels = [k for k in series.keys() if k != "time"]
    for win_name, (start, end) in windows.items():
        mask = _window_mask(times, start, end)
        for ch in channels:
            stats = _safe_stats(series[ch][mask])
            names = ["mean", "std", "min", "max", "median", "p90", "last_first", "slope"]
            values.extend(stats)
            feature_names.extend([f"{win_name}_{ch}_{name}" for name in names])

    eps = 1e-6
    for ch in ["speed", "frame_diff", "vehicle_count", "vehicle_max_threat", "person_count", "risk_max_threat"]:
        if ch not in series:
            continue
        pre = series[ch][_window_mask(times, *windows["pre"])]
        post = series[ch][_window_mask(times, *windows["post"])]
        early = series[ch][_window_mask(times, *windows["early_post"])]
        pre_mean = float(np.mean(pre)) if pre.size else 0.0
        post_mean = float(np.mean(post)) if post.size else 0.0
        early_mean = float(np.mean(early)) if early.size else 0.0
        values.extend([post_mean - pre_mean, post_mean / (pre_mean + eps), early_mean - pre_mean, early_mean / (pre_mean + eps)])
        feature_names.extend([f"{ch}_post_minus_pre", f"{ch}_post_ratio_pre", f"{ch}_early_minus_pre", f"{ch}_early_ratio_pre"])

    speed = series["speed"]
    pre_speed = speed[_window_mask(times, *windows["pre"])]
    post_speed = speed[_window_mask(times, *windows["post"])]
    base = float(np.percentile(pre_speed, 25)) if pre_speed.size else float(np.percentile(speed, 25))
    stop_thr = max(0.5, base * 0.7)
    for win_name in ["early_post", "late_post", "post"]:
        mask = _window_mask(times, *windows[win_name])
        ratio = float(np.mean(speed[mask] <= stop_thr)) if mask.any() else 0.0
        values.append(ratio)
        feature_names.append(f"{win_name}_visual_stop_ratio")

    return np.asarray(values, dtype=np.float32), feature_names


def extract_one(
    row: pd.Series,
    out_dir: Path,
    sample_fps: float,
    width: int,
    motion_mode: str,
    yolo_model: Any | None,
    yolo_batch: int,
    force: bool,
) -> Path:
    source_path = Path(row["source_path"])
    out_path = out_dir / f"{stable_video_id(source_path)}.npz"
    if out_path.exists() and not force:
        return out_path
    times, frames, duration = _read_timed_frames(source_path, fps=sample_fps, width=width)
    series = _series_from_frames(times, frames, motion_mode)
    if yolo_model is not None:
        series.update(_yolo_series(yolo_model, frames, yolo_batch))
    else:
        zeros = np.zeros(len(times), dtype=np.float32)
        for ch in [
            "vehicle_count",
            "vehicle_max_conf",
            "vehicle_max_area",
            "vehicle_sum_area",
            "vehicle_max_bottom",
            "vehicle_min_center_dist",
            "vehicle_max_threat",
            "risk_count",
            "risk_max_area",
            "risk_max_threat",
            "person_count",
            "bicycle_count",
            "car_count",
            "motorcycle_count",
            "bus_count",
            "truck_count",
        ]:
            series[ch] = zeros.copy()
    summary, names = _summarize(series, float(row["time_of_event"]), duration)
    np.savez_compressed(
        out_path,
        summary=summary,
        feature_names=np.asarray(names),
        times=times,
        source_path=str(source_path),
        clip_path=str(row["path"]),
        label=np.asarray([int(row["label"])], dtype=np.int64),
        event_time=np.asarray([float(row["time_of_event"])], dtype=np.float32),
        duration=np.asarray([duration], dtype=np.float32),
        motion_mode=np.asarray([motion_mode]),
    )
    return out_path


def _load_rows(csvs: list[str]) -> pd.DataFrame:
    frames = []
    for csv in csvs:
        frames.append(pd.read_csv(csv))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("source_path").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", default=None)
    parser.add_argument("--out-dir", default="outputs/processed_744/long_context_features")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--motion-mode", choices=["affine", "diff"], default="affine")
    parser.add_argument("--use-yolo", action="store_true")
    parser.add_argument("--yolo-model", default="models/yolov8n.pt")
    parser.add_argument("--yolo-batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    csvs = args.csv or [
        "splits/processed_744_long/train.csv",
        "splits/processed_744_long/test.csv",
    ]
    rows = _load_rows(csvs)
    if args.limit > 0:
        rows = rows.head(args.limit)
    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)
    yolo_model = None
    if args.use_yolo:
        from ultralytics import YOLO

        yolo_model = YOLO(args.yolo_model)
    manifest = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="long-context"):
        out = extract_one(row, out_dir, args.sample_fps, args.width, args.motion_mode, yolo_model, args.yolo_batch, args.force)
        manifest.append({"source_path": row["source_path"], "clip_path": row["path"], "label": int(row["label"]), "feature_path": str(out)})
    write_json(
        out_dir / "manifest.json",
        {
            "count": len(manifest),
            "sample_fps": args.sample_fps,
            "width": args.width,
            "motion_mode": args.motion_mode,
            "use_yolo": bool(args.use_yolo),
            "items": manifest,
        },
    )
    print(f"extracted {len(manifest)} long-context feature files")


if __name__ == "__main__":
    main()
