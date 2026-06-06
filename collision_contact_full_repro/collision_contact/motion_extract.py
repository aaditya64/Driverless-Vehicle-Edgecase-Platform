from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


MOTION_COLUMNS = [
    "time",
    "frame_idx",
    "dx",
    "dy",
    "theta",
    "theta_px",
    "scale",
    "scale_delta",
    "shear",
    "inlier_ratio",
    "fit_error",
    "num_tracks",
    "num_inliers",
    "x_path",
    "y_path",
    "theta_path",
    "x_res",
    "y_res",
    "theta_res",
    "theta_res_px",
    "vx",
    "vy",
    "vtheta",
    "vtheta_px",
    "ax",
    "ay",
    "atheta",
    "atheta_px",
    "jerk_x",
    "jerk_y",
    "jerk_theta",
    "jerk_theta_px",
    "shake_energy",
    "jerk_energy",
    "valid",
]


def _resize_gray(frame: np.ndarray, resize_width: int, clahe: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    if resize_width > 0 and w != resize_width:
        scale = resize_width / float(w)
        frame = cv2.resize(frame, (resize_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if clahe:
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe_op.apply(gray)
    return gray


def _read_video_gray(video_path: str | Path, resize_width: int, clahe: bool) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(_resize_gray(frame, resize_width=resize_width, clahe=clahe))
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"Need at least two frames: {video_path}")
    return frames, fps


def _safe_savgol(x: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    n = len(x)
    if n < 5:
        return x.copy()
    window = min(window, n if n % 2 == 1 else n - 1)
    window = max(window, polyorder + 2)
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window <= polyorder or window < 3:
        return x.copy()
    return savgol_filter(x, window_length=window, polyorder=polyorder, mode="interp")


def _estimate_pair(prev: np.ndarray, curr: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    corners = cv2.goodFeaturesToTrack(
        prev,
        maxCorners=int(cfg["max_corners"]),
        qualityLevel=float(cfg["quality_level"]),
        minDistance=float(cfg["min_distance"]),
        blockSize=int(cfg["block_size"]),
    )
    if corners is None or len(corners) < int(cfg["min_tracks"]):
        return _empty_motion()

    lk_params = dict(
        winSize=(int(cfg["lk_win_size"]), int(cfg["lk_win_size"])),
        maxLevel=int(cfg["lk_max_level"]),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(cfg["lk_max_iter"]), float(cfg["lk_eps"])),
    )
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, corners, None, **lk_params)
    back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(curr, prev, next_pts, None, **lk_params)
    if next_pts is None or back_pts is None:
        return _empty_motion(num_tracks=float(len(corners)))

    p0 = corners.reshape(-1, 2)
    p1 = next_pts.reshape(-1, 2)
    pb = back_pts.reshape(-1, 2)
    status = status.reshape(-1).astype(bool)
    back_status = back_status.reshape(-1).astype(bool)
    fb_error = np.linalg.norm(p0 - pb, axis=1)
    keep = status & back_status & np.isfinite(fb_error) & (fb_error <= float(cfg["fb_error_threshold"]))
    p0 = p0[keep]
    p1 = p1[keep]
    num_tracks = float(len(p0))
    if len(p0) < int(cfg["min_tracks"]):
        return _empty_motion(num_tracks=num_tracks)

    affine, inliers = cv2.estimateAffinePartial2D(
        p0,
        p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg["ransac_threshold"]),
        maxIters=int(cfg["ransac_max_iters"]),
        confidence=float(cfg["ransac_confidence"]),
        refineIters=10,
    )
    if affine is None or inliers is None:
        return _empty_motion(num_tracks=num_tracks)

    inlier_mask = inliers.reshape(-1).astype(bool)
    num_inliers = float(inlier_mask.sum())
    if num_inliers < int(cfg["min_inliers"]):
        return _empty_motion(num_tracks=num_tracks, num_inliers=num_inliers)

    pred = cv2.transform(p0.reshape(-1, 1, 2), affine).reshape(-1, 2)
    residual = np.linalg.norm(pred - p1, axis=1)
    fit_error = float(np.median(residual[inlier_mask]))

    a, b, tx = affine[0]
    c, d, ty = affine[1]
    theta = float(np.arctan2(c, a))
    scale_x = float(np.sqrt(a * a + c * c))
    scale_y = float(np.sqrt(b * b + d * d))
    scale = 0.5 * (scale_x + scale_y)
    shear = float(a * b + c * d)

    return {
        "dx": float(tx),
        "dy": float(ty),
        "theta": theta,
        "scale": scale,
        "scale_delta": scale - 1.0,
        "shear": shear,
        "inlier_ratio": float(num_inliers / max(num_tracks, 1.0)),
        "fit_error": fit_error,
        "num_tracks": num_tracks,
        "num_inliers": num_inliers,
        "valid": 1.0,
    }


def _empty_motion(num_tracks: float = 0.0, num_inliers: float = 0.0) -> dict[str, float]:
    return {
        "dx": 0.0,
        "dy": 0.0,
        "theta": 0.0,
        "scale": 1.0,
        "scale_delta": 0.0,
        "shear": 0.0,
        "inlier_ratio": 0.0,
        "fit_error": 0.0,
        "num_tracks": float(num_tracks),
        "num_inliers": float(num_inliers),
        "valid": 0.0,
    }


def extract_global_motion(video_path: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    video_cfg = cfg["video"]
    motion_cfg = cfg["motion"]
    frames, fps = _read_video_gray(
        video_path,
        resize_width=int(video_cfg["resize_width"]),
        clahe=bool(video_cfg.get("clahe", True)),
    )
    rows = [_empty_motion()]
    rows[0]["valid"] = 1.0
    for i in range(1, len(frames)):
        rows.append(_estimate_pair(frames[i - 1], frames[i], motion_cfg))

    df = pd.DataFrame(rows)
    df.insert(0, "frame_idx", np.arange(len(df), dtype=np.int32))
    df.insert(0, "time", np.arange(len(df), dtype=np.float32) / fps)
    resize_width = float(video_cfg["resize_width"])
    df["theta_px"] = df["theta"].to_numpy() * resize_width

    for col in ["dx", "dy", "theta"]:
        arr = df[col].to_numpy(dtype=np.float64).copy()
        arr[~np.isfinite(arr)] = 0.0
        df[col] = arr

    df["x_path"] = df["dx"].cumsum()
    df["y_path"] = df["dy"].cumsum()
    df["theta_path"] = df["theta"].cumsum()

    window = int(motion_cfg["smoothing_window"])
    polyorder = int(motion_cfg["smoothing_polyorder"])
    x_base = _safe_savgol(df["x_path"].to_numpy(), window, polyorder)
    y_base = _safe_savgol(df["y_path"].to_numpy(), window, polyorder)
    th_base = _safe_savgol(df["theta_path"].to_numpy(), window, polyorder)

    df["x_res"] = df["x_path"] - x_base
    df["y_res"] = df["y_path"] - y_base
    df["theta_res"] = df["theta_path"] - th_base
    df["theta_res_px"] = df["theta_res"] * resize_width

    for source, velocity, accel, jerk in [
        ("x_res", "vx", "ax", "jerk_x"),
        ("y_res", "vy", "ay", "jerk_y"),
        ("theta_res", "vtheta", "atheta", "jerk_theta"),
    ]:
        r = df[source].to_numpy(dtype=np.float64)
        v = np.diff(r, prepend=r[0])
        a = np.diff(v, prepend=v[0])
        j = np.diff(a, prepend=a[0])
        df[velocity] = v
        df[accel] = a
        df[jerk] = j

    df["vtheta_px"] = df["vtheta"] * resize_width
    df["atheta_px"] = df["atheta"] * resize_width
    df["jerk_theta_px"] = df["jerk_theta"] * resize_width
    df["shake_energy"] = np.sqrt(df["x_res"] ** 2 + df["y_res"] ** 2 + df["theta_res_px"] ** 2)
    df["jerk_energy"] = np.sqrt(df["jerk_x"] ** 2 + df["jerk_y"] ** 2 + df["jerk_theta_px"] ** 2)

    for col in MOTION_COLUMNS:
        if col not in df:
            df[col] = 0.0
    return df[MOTION_COLUMNS].replace([np.inf, -np.inf], 0.0).fillna(0.0)
