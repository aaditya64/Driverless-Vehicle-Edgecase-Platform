"""Extract object-box CoTracker dynamics after ego-motion stabilization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .common import ensure_dirs, read_split_csv, stable_video_id, write_json
from .extract_residual_flow_physics_features import _fit_affine_from_flow
from .extract_yolo_interaction_features import RISK_CLASSES, _ensure_torchvision_nms_stub, _ultralytics_detections


EPS = 1e-6


def _read_uniform_bgr_frames(
    video_path: str | Path,
    num_frames: int,
    width: int,
    center_seconds: float | None,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if frame_count <= 0:
        frame_count = num_frames
    start = 0
    end = max(frame_count - 1, 0)
    if center_seconds is not None and center_seconds > 0 and fps > 0 and frame_count > 1:
        center = 0.5 * (frame_count - 1)
        half = 0.5 * center_seconds * fps
        start = int(round(max(0.0, center - half)))
        end = int(round(min(float(frame_count - 1), center + half)))
        if end <= start:
            start, end = 0, max(frame_count - 1, 0)
    frame_idxs = np.linspace(start, end, num_frames).round().astype(np.int32)
    frames: list[np.ndarray] = []
    last: np.ndarray | None = None
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            if width > 0 and w != width:
                frame = cv2.resize(frame, (width, int(round(h * width / max(w, 1)))), interpolation=cv2.INTER_AREA)
            last = frame
        if last is not None:
            frames.append(last.copy())
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames], frame_idxs, fps


def _box_geometry(box: dict[str, float], w: int, h: int) -> tuple[float, float, float, float, float, float]:
    bw = max(float(box["x2"] - box["x1"]), 1.0)
    bh = max(float(box["y2"] - box["y1"]), 1.0)
    cx = float((box["x1"] + box["x2"]) * 0.5)
    cy = float((box["y1"] + box["y2"]) * 0.5)
    area = bw * bh / max(float(w * h), 1.0)
    center_weight = max(0.0, 1.0 - abs(cx / max(float(w), 1.0) - 0.5) * 2.0)
    bottom_weight = 0.5 + 0.5 * float(box["y2"]) / max(float(h), 1.0)
    threat = area * float(box["conf"]) * (0.25 + 0.75 * center_weight) * bottom_weight
    return cx, cy, bw, bh, area, threat


def _points_in_box(box: dict[str, float], grid: int, frame_w: int, frame_h: int, margin: float = 0.16) -> np.ndarray:
    x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    xs = np.linspace(x1 + margin * bw, x2 - margin * bw, grid, dtype=np.float32)
    ys = np.linspace(y1 + margin * bh, y2 - margin * bh, grid, dtype=np.float32)
    pts = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, frame_w - 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, frame_h - 1.0)
    return pts


def _fallback_queries(frame_w: int, frame_h: int, query_t: int, grid: int) -> tuple[np.ndarray, list[int], list[dict[str, float]]]:
    xs = np.linspace(0.25 * frame_w, 0.75 * frame_w, grid, dtype=np.float32)
    ys = np.linspace(0.45 * frame_h, 0.90 * frame_h, grid, dtype=np.float32)
    pts = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float32)
    queries = np.c_[np.full(len(pts), query_t, dtype=np.float32), pts]
    meta = {
        "query_t": float(query_t),
        "cx": float(0.5 * frame_w),
        "cy": float(0.67 * frame_h),
        "bw": float(0.5 * frame_w),
        "bh": float(0.45 * frame_h),
        "area": 0.225,
        "threat": 0.0,
        "conf": 0.0,
        "cls": -1.0,
    }
    return queries.astype(np.float32), [0] * len(pts), [meta]


def _build_queries(
    detections_by_frame: dict[int, list[dict[str, float]]],
    query_frames: list[int],
    frame_w: int,
    frame_h: int,
    top_boxes: int,
    box_grid: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    queries: list[np.ndarray] = []
    groups: list[int] = []
    metas: list[dict[str, float]] = []
    group_id = 0
    for query_t in query_frames:
        dets = [d for d in detections_by_frame.get(query_t, []) if int(d["cls"]) in RISK_CLASSES]
        scored: list[tuple[float, dict[str, float], tuple[float, float, float, float, float, float]]] = []
        for det in dets:
            geom = _box_geometry(det, frame_w, frame_h)
            _cx, _cy, _bw, _bh, area, threat = geom
            score = threat + 0.15 * area + 0.01 * float(det["conf"])
            scored.append((score, det, geom))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _score, det, geom in scored[:top_boxes]:
            cx, cy, bw, bh, area, threat = geom
            pts = _points_in_box(det, box_grid, frame_w, frame_h)
            q = np.c_[np.full(len(pts), query_t, dtype=np.float32), pts]
            queries.append(q)
            groups.extend([group_id] * len(pts))
            metas.append(
                {
                    "query_t": float(query_t),
                    "cx": cx,
                    "cy": cy,
                    "bw": bw,
                    "bh": bh,
                    "area": area,
                    "threat": threat,
                    "conf": float(det["conf"]),
                    "cls": float(det["cls"]),
                }
            )
            group_id += 1
    if not queries:
        q, g, m = _fallback_queries(frame_w, frame_h, query_frames[len(query_frames) // 2], box_grid)
        return q, np.asarray(g, dtype=np.int32), m
    return np.concatenate(queries, axis=0).astype(np.float32), np.asarray(groups, dtype=np.int32), metas


def _apply_affine(mat: np.ndarray | None, point: np.ndarray) -> np.ndarray:
    if mat is None:
        return point.astype(np.float32, copy=True)
    x, y = float(point[0]), float(point[1])
    return np.asarray([mat[0, 0] * x + mat[0, 1] * y + mat[0, 2], mat[1, 0] * x + mat[1, 1] * y + mat[1, 2]], dtype=np.float32)


def _fit_pair_affines(frames_bgr: list[np.ndarray], sample_step: int, ransac_threshold: float) -> tuple[list[np.ndarray | None], np.ndarray, np.ndarray]:
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_bgr]
    mats: list[np.ndarray | None] = []
    inliers: list[float] = []
    errors: list[float] = []
    for i in range(1, len(gray)):
        flow = cv2.calcOpticalFlowFarneback(gray[i - 1], gray[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mat, inlier_ratio, fit_error = _fit_affine_from_flow(flow, sample_step=sample_step, ransac_threshold=ransac_threshold)
        mats.append(mat)
        inliers.append(float(inlier_ratio))
        errors.append(float(fit_error))
    return mats, np.asarray(inliers, dtype=np.float32), np.asarray(errors, dtype=np.float32)


def _robust_center(points: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, float, float]:
    pts = points[visible]
    if len(pts) == 0:
        return np.asarray([np.nan, np.nan], dtype=np.float32), np.nan, np.nan
    center = np.median(pts, axis=0).astype(np.float32)
    if len(pts) >= 3:
        lo = np.percentile(pts, 10, axis=0)
        hi = np.percentile(pts, 90, axis=0)
    else:
        lo = np.min(pts, axis=0)
        hi = np.max(pts, axis=0)
    wh = np.maximum(hi - lo, 1.0)
    return center, float(wh[0]), float(wh[1])


def _group_pair_rows(
    tracks: np.ndarray,
    visibility: np.ndarray,
    groups: np.ndarray,
    metas: list[dict[str, float]],
    mats: list[np.ndarray | None],
    inliers: np.ndarray,
    errors: np.ndarray,
    frame_idxs: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    n_frames = tracks.shape[0]
    n_groups = len(metas)
    centers = np.full((n_groups, n_frames, 2), np.nan, dtype=np.float32)
    widths = np.full((n_groups, n_frames), np.nan, dtype=np.float32)
    heights = np.full((n_groups, n_frames), np.nan, dtype=np.float32)
    vis_rates = np.zeros((n_groups, n_frames), dtype=np.float32)
    for gid in range(n_groups):
        idx = np.where(groups == gid)[0]
        for t in range(n_frames):
            vis = visibility[t, idx] > 0.5
            vis_rates[gid, t] = float(np.mean(vis)) if len(idx) else 0.0
            center, bw, bh = _robust_center(tracks[t, idx], vis)
            centers[gid, t] = center
            widths[gid, t] = bw
            heights[gid, t] = bh

    rows: list[list[float]] = []
    times: list[float] = []
    names = [
        "visible_group_count",
        "threat_max",
        "res_depth_max",
        "res_depth_mean",
        "res_x_depth_abs_max",
        "res_y_depth_abs_max",
        "vel_depth_max",
        "acc_depth_max",
        "jerk_depth_max",
        "scale_log_abs_max",
        "area_log_abs_max",
        "shape_change_max",
        "impulse_score_max",
        "motion_transfer_max",
        "affine_inlier_ratio",
        "affine_fit_error",
    ]
    prev_vel = np.zeros((n_groups, 2), dtype=np.float32)
    prev_acc = np.zeros((n_groups, 2), dtype=np.float32)
    has_prev_vel = np.zeros(n_groups, dtype=bool)
    has_prev_acc = np.zeros(n_groups, dtype=bool)
    group_level = np.zeros((n_groups, 8), dtype=np.float32)
    for t in range(1, n_frames):
        dt = float((frame_idxs[t] - frame_idxs[t - 1]) / max(fps, EPS))
        dt = max(dt, EPS)
        pair_values: list[dict[str, float]] = []
        for gid, meta in enumerate(metas):
            if vis_rates[gid, t] < 0.35 or vis_rates[gid, t - 1] < 0.35:
                continue
            c0 = centers[gid, t - 1]
            c1 = centers[gid, t]
            if not np.all(np.isfinite(c0)) or not np.all(np.isfinite(c1)):
                continue
            norm = max(float(meta["bh"]), 8.0)
            expected = _apply_affine(mats[t - 1], c0)
            residual = (c1 - expected).astype(np.float32)
            res_depth_vec = residual / norm
            vel = res_depth_vec / dt
            acc = (vel - prev_vel[gid]) / dt if has_prev_vel[gid] else np.zeros(2, dtype=np.float32)
            jerk = (acc - prev_acc[gid]) / dt if has_prev_acc[gid] else np.zeros(2, dtype=np.float32)
            prev_vel[gid] = vel
            prev_acc[gid] = acc
            has_prev_vel[gid] = True
            has_prev_acc[gid] = True
            scale_log = float(np.log((heights[gid, t] + EPS) / (heights[gid, t - 1] + EPS)))
            area_log = float(np.log((widths[gid, t] * heights[gid, t] + EPS) / (widths[gid, t - 1] * heights[gid, t - 1] + EPS)))
            idx = np.where(groups == gid)[0]
            vmask = (visibility[t, idx] > 0.5) & (visibility[t - 1, idx] > 0.5)
            if np.any(vmask):
                p0 = tracks[t - 1, idx][vmask] - c0[None]
                p1 = tracks[t, idx][vmask] - c1[None]
                shape_change = float(np.median(np.linalg.norm((p1 - p0) / norm, axis=1)))
            else:
                shape_change = 0.0
            res_depth = float(np.linalg.norm(res_depth_vec))
            vel_depth = float(np.linalg.norm(vel))
            acc_depth = float(np.linalg.norm(acc))
            jerk_depth = float(np.linalg.norm(jerk))
            impulse = float((1.0 + res_depth) * (1.0 + acc_depth) * (1.0 + 0.2 * jerk_depth) * (1.0 + abs(scale_log) + abs(area_log)) * (0.05 + meta["threat"]))
            transfer = float((1.0 + res_depth) * (1.0 + shape_change) * (1.0 + abs(area_log)) * (0.05 + meta["area"]))
            pair_values.append(
                {
                    "threat": float(meta["threat"]),
                    "res_depth": min(res_depth, 50.0),
                    "res_x_abs": float(abs(res_depth_vec[0])),
                    "res_y_abs": float(abs(res_depth_vec[1])),
                    "vel_depth": min(vel_depth, 300.0),
                    "acc_depth": min(acc_depth, 3000.0),
                    "jerk_depth": min(jerk_depth, 20000.0),
                    "scale_abs": float(abs(scale_log)),
                    "area_abs": float(abs(area_log)),
                    "shape_change": min(shape_change, 50.0),
                    "impulse": min(impulse, 1e7),
                    "transfer": min(transfer, 1e7),
                }
            )
            group_level[gid, 0] = max(group_level[gid, 0], res_depth)
            group_level[gid, 1] = max(group_level[gid, 1], acc_depth)
            group_level[gid, 2] = max(group_level[gid, 2], jerk_depth)
            group_level[gid, 3] = max(group_level[gid, 3], abs(scale_log))
            group_level[gid, 4] = max(group_level[gid, 4], abs(area_log))
            group_level[gid, 5] = max(group_level[gid, 5], shape_change)
            group_level[gid, 6] = max(group_level[gid, 6], impulse)
            group_level[gid, 7] = max(group_level[gid, 7], transfer)
        if pair_values:
            res_depths = np.asarray([v["res_depth"] for v in pair_values], dtype=np.float32)
            row = [
                float(len(pair_values)),
                float(max(v["threat"] for v in pair_values)),
                float(np.max(res_depths)),
                float(np.mean(res_depths)),
                float(max(v["res_x_abs"] for v in pair_values)),
                float(max(v["res_y_abs"] for v in pair_values)),
                float(max(v["vel_depth"] for v in pair_values)),
                float(max(v["acc_depth"] for v in pair_values)),
                float(max(v["jerk_depth"] for v in pair_values)),
                float(max(v["scale_abs"] for v in pair_values)),
                float(max(v["area_abs"] for v in pair_values)),
                float(max(v["shape_change"] for v in pair_values)),
                float(max(v["impulse"] for v in pair_values)),
                float(max(v["transfer"] for v in pair_values)),
                float(inliers[t - 1]),
                float(errors[t - 1]),
            ]
        else:
            row = [0.0] * 14 + [float(inliers[t - 1]), float(errors[t - 1])]
        rows.append(row)
        times.append(float(frame_idxs[t] / max(fps, EPS)))
    sequence = np.asarray(rows, dtype=np.float32)
    times_arr = np.asarray(times, dtype=np.float32)
    summary, summary_names = _dynamic_summary(times_arr, sequence, names)
    level_names = [
        "group_res_depth_max",
        "group_acc_depth_max",
        "group_jerk_depth_max",
        "group_scale_abs_max",
        "group_area_abs_max",
        "group_shape_change_max",
        "group_impulse_max",
        "group_transfer_max",
    ]
    if n_groups:
        level_summary = np.concatenate([group_level.max(axis=0), group_level.mean(axis=0), group_level.std(axis=0)]).astype(np.float32)
    else:
        level_summary = np.zeros(len(level_names) * 3, dtype=np.float32)
    level_summary_names = [f"{name}_{stat}" for stat in ["max", "mean", "std"] for name in level_names]
    return sequence, times_arr, names, np.concatenate([summary, level_summary]).astype(np.float32), summary_names + level_summary_names


def _dynamic_summary(times: np.ndarray, sequence: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    feats: list[float] = []
    feat_names: list[str] = []
    search = (times >= 2.0) & (times <= 8.0)
    early = (times >= 0.5) & (times < 2.5)
    if not np.any(search):
        search = np.ones_like(times, dtype=bool)
    if not np.any(early):
        early = np.ones_like(times, dtype=bool)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    dt = max(dt, EPS)
    for idx, name in enumerate(names):
        x = np.nan_to_num(sequence[:, idx].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        x = np.maximum(x, 0.0)
        xs = x[search]
        ts = times[search]
        peak_idx = int(np.argmax(xs)) if len(xs) else 0
        peak = float(xs[peak_idx]) if len(xs) else 0.0
        peak_time = float(ts[peak_idx]) if len(ts) else 0.0
        q25 = float(np.percentile(x, 25))
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        early_med = float(np.median(x[early]))
        early_p95 = float(np.percentile(x[early], 95))
        pre = x[(times >= peak_time - 1.25) & (times < peak_time - 0.25)]
        around = x[(times >= peak_time - 0.5) & (times <= peak_time + 0.5)]
        post = x[(times > peak_time + 0.25) & (times <= peak_time + 1.25)]
        if pre.size == 0:
            pre = x
        if around.size == 0:
            around = x
        if post.size == 0:
            post = x
        d1 = np.diff(x, prepend=x[0]) / dt
        d2 = np.diff(d1, prepend=d1[0]) / dt
        vals = [
            peak,
            float(np.log1p(peak)),
            float((peak + EPS) / (q25 + EPS)),
            float((peak + EPS) / (early_med + EPS)),
            float((peak + EPS) / (early_p95 + EPS)),
            float((peak - med) / (1.4826 * mad + 1e-3)),
            peak_time,
            float((peak + EPS) / (np.mean(around) + EPS)),
            float((np.mean(post) + EPS) / (peak + EPS)),
            float((np.mean(post) + EPS) / (np.mean(pre) + EPS)),
            float(np.percentile(xs, 95)) if len(xs) else 0.0,
            float(np.max(np.abs(d1[search]))) if np.any(search) else float(np.max(np.abs(d1))),
            float(np.max(np.abs(d2[search]))) if np.any(search) else float(np.max(np.abs(d2))),
        ]
        suffixes = [
            "peak",
            "log_peak",
            "peak_q25_ratio",
            "peak_early_ratio",
            "peak_early_p95_ratio",
            "peak_z",
            "peak_time",
            "sharpness",
            "post_over_peak",
            "post_over_pre",
            "p95",
            "d1_peak",
            "d2_peak",
        ]
        feats.extend(vals)
        feat_names.extend([f"{name}_{suffix}" for suffix in suffixes])
    return np.asarray(feats, dtype=np.float32), feat_names


def extract_one(
    video_path: str | Path,
    yolo_model: Any,
    cotracker_model: torch.nn.Module,
    device: torch.device,
    out_dir: str | Path,
    num_frames: int,
    width: int,
    center_seconds: float | None,
    query_positions: list[float],
    top_boxes: int,
    box_grid: int,
    image_size: int,
    conf: float,
    iou: float,
    sample_step: int,
    ransac_threshold: float,
    force: bool = False,
) -> Path:
    ensure_dirs(out_dir)
    out_path = Path(out_dir) / f"{stable_video_id(video_path)}.npz"
    if out_path.exists() and not force:
        return out_path
    frames_bgr, frame_idxs, fps = _read_uniform_bgr_frames(video_path, num_frames, width, center_seconds)
    frame_h, frame_w = frames_bgr[0].shape[:2]
    query_frames = sorted(set(int(round(np.clip(pos, 0.0, 1.0) * (num_frames - 1))) for pos in query_positions))
    yolo_frames = [frames_bgr[idx] for idx in query_frames]
    results = yolo_model.predict(yolo_frames, imgsz=image_size, conf=conf, iou=iou, device=0 if device.type == "cuda" else "cpu", batch=len(yolo_frames), verbose=False)
    detections_by_frame = {query_frames[i]: _ultralytics_detections(result) for i, result in enumerate(results)}
    queries_np, groups, metas = _build_queries(detections_by_frame, query_frames, frame_w, frame_h, top_boxes, box_grid)
    video_rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]).astype(np.uint8)
    video = torch.from_numpy(video_rgb).permute(0, 3, 1, 2)[None].float().to(device)
    queries = torch.from_numpy(queries_np)[None].float().to(device)
    with torch.no_grad():
        try:
            tracks, visibility = cotracker_model(video, queries=queries, grid_size=0, backward_tracking=True)
        except (TypeError, ValueError):
            tracks, visibility = cotracker_model(video, queries=queries, grid_size=0)
    tracks_np = tracks[0].detach().float().cpu().numpy()
    visibility_np = visibility[0].detach().float().cpu().numpy()
    mats, inliers, errors = _fit_pair_affines(frames_bgr, sample_step=sample_step, ransac_threshold=ransac_threshold)
    sequence, times, sequence_names, summary, summary_names = _group_pair_rows(
        tracks_np,
        visibility_np,
        groups,
        metas,
        mats,
        inliers,
        errors,
        frame_idxs,
        fps,
    )
    np.savez_compressed(
        out_path,
        sequence=sequence,
        summary=summary,
        times=times,
        sequence_names=np.asarray(sequence_names),
        summary_names=np.asarray(summary_names),
        tracks=tracks_np.astype(np.float32),
        visibility=visibility_np.astype(np.float32),
        groups=groups.astype(np.int32),
        queries=queries_np.astype(np.float32),
        object_meta=np.asarray([[m["query_t"], m["cx"], m["cy"], m["bw"], m["bh"], m["area"], m["threat"], m["conf"], m["cls"]] for m in metas], dtype=np.float32),
        video_path=str(video_path),
        width=np.asarray([frame_w], dtype=np.int32),
        height=np.asarray([frame_h], dtype=np.int32),
        fps=np.asarray([fps], dtype=np.float32),
        frame_idxs=frame_idxs,
    )
    return out_path


def _rows_from_csvs(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for csv_path in paths:
        for row in read_split_csv(csv_path):
            if row["path"] in seen:
                continue
            seen.add(row["path"])
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", default=[])
    parser.add_argument("--out-dir", default="outputs/processed_744/object_cotracker_dynamics_yolov8s_32f_w384_20260524")
    parser.add_argument("--cotracker-repo", default="external/co-tracker")
    parser.add_argument("--yolo-model", default="yolov8s.pt")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--center-seconds", type=float, default=6.0)
    parser.add_argument("--query-positions", nargs="+", type=float, default=[0.35, 0.5, 0.65])
    parser.add_argument("--top-boxes", type=int, default=4)
    parser.add_argument("--box-grid", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--sample-step", type=int, default=12)
    parser.add_argument("--ransac-threshold", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    csv_paths = args.csv or ["splits/processed_744/train.csv", "splits/processed_744/test.csv"]
    rows = _rows_from_csvs(csv_paths)
    if args.limit > 0:
        rows = rows[: args.limit]
    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)
    _ensure_torchvision_nms_stub()
    from ultralytics import YOLO

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cpu" or torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    yolo_model = YOLO(args.yolo_model)
    cotracker_model = torch.hub.load(args.cotracker_repo, "cotracker3_offline", source="local").to(device).eval()
    center_seconds = args.center_seconds if args.center_seconds > 0 else None
    manifest: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="object-cotracker"):
        out = extract_one(
            row["path"],
            yolo_model,
            cotracker_model,
            device,
            out_dir,
            args.num_frames,
            args.width,
            center_seconds,
            args.query_positions,
            args.top_boxes,
            args.box_grid,
            args.image_size,
            args.conf,
            args.iou,
            args.sample_step,
            args.ransac_threshold,
            args.force,
        )
        manifest.append({"path": row["path"], "label": int(row["label"]), "feature_path": str(out)})
    write_json(
        out_dir / "manifest.json",
        {
            "count": len(manifest),
            "items": manifest,
            "num_frames": args.num_frames,
            "width": args.width,
            "center_seconds": center_seconds,
            "query_positions": args.query_positions,
            "top_boxes": args.top_boxes,
            "box_grid": args.box_grid,
            "yolo_model": args.yolo_model,
        },
    )
    print(f"extracted {len(manifest)} object CoTracker dynamics feature files", flush=True)


if __name__ == "__main__":
    main()
