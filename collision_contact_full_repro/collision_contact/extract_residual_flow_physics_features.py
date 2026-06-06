"""Extract event-centered residual-flow physics features for collision analysis."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .common import ensure_dirs, read_split_csv, stable_video_id, write_json


EPS = 1e-6


def _read_sampled_gray(video_path: str | Path, width: int, frame_step: int) -> tuple[list[np.ndarray], np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_step == 0:
            h, w = frame.shape[:2]
            if width > 0 and w != width:
                frame = cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            frame_indices.append(idx)
        idx += 1
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"Need at least two sampled frames: {video_path}")
    return frames, np.asarray(frame_indices, dtype=np.int32), fps


def _direction_entropy(dx: np.ndarray, dy: np.ndarray, weights: np.ndarray, bins: int = 16) -> float:
    angles = np.arctan2(dy, dx)
    hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi), weights=weights)
    total = float(hist.sum())
    if total <= EPS:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(bins))


def _fit_affine_from_flow(flow: np.ndarray, sample_step: int, ransac_threshold: float) -> tuple[np.ndarray | None, float, float]:
    h, w = flow.shape[:2]
    step = max(8, int(sample_step))
    yy, xx = np.mgrid[step // 2 : h : step, step // 2 : w : step]
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    sampled = flow[yy.reshape(-1), xx.reshape(-1)].astype(np.float32)
    dst = pts + sampled
    if len(pts) < 12:
        return None, 0.0, 0.0
    mat, inliers = cv2.estimateAffinePartial2D(
        pts,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold),
        maxIters=1000,
        confidence=0.99,
    )
    if mat is None:
        return None, 0.0, 0.0
    fitted = cv2.transform(pts.reshape(-1, 1, 2), mat).reshape(-1, 2)
    fit_error = float(np.median(np.linalg.norm(dst - fitted, axis=1)))
    inlier_ratio = float(inliers.mean()) if inliers is not None else 0.0
    return mat.astype(np.float32), inlier_ratio, fit_error


def _affine_expected_flow(mat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    expected_x = mat[0, 0] * grid_x + mat[0, 1] * grid_y + mat[0, 2] - grid_x
    expected_y = mat[1, 0] * grid_x + mat[1, 1] * grid_y + mat[1, 2] - grid_y
    return np.stack([expected_x, expected_y], axis=2).astype(np.float32)


def _grid_reduce(values: np.ndarray, vec: np.ndarray | None, grid: int) -> dict[str, float]:
    h, w = values.shape
    p95s: list[float] = []
    energies: list[float] = []
    coherences: list[float] = []
    lower_values: list[np.ndarray] = []
    center_values: list[np.ndarray] = []
    border_values: list[np.ndarray] = []
    for gy in range(grid):
        for gx in range(grid):
            y0 = int(round(gy * h / grid))
            y1 = int(round((gy + 1) * h / grid))
            x0 = int(round(gx * w / grid))
            x1 = int(round((gx + 1) * w / grid))
            cell = values[y0:y1, x0:x1]
            p95s.append(float(np.percentile(cell, 95)))
            energies.append(float(np.mean(cell * cell)))
            if vec is not None:
                cell_vec = vec[y0:y1, x0:x1]
                mean_vec = np.mean(cell_vec.reshape(-1, 2), axis=0)
                coherences.append(float(np.linalg.norm(mean_vec) / (np.mean(cell) + EPS)))
            if gy >= grid // 2:
                lower_values.append(cell.reshape(-1))
            if abs(gx - (grid - 1) / 2.0) <= 1.0 and gy >= grid // 3:
                center_values.append(cell.reshape(-1))
            if gy == 0 or gx == 0 or gy == grid - 1 or gx == grid - 1:
                border_values.append(cell.reshape(-1))
    p95 = np.asarray(p95s, dtype=np.float32)
    energy = np.asarray(energies, dtype=np.float32)
    order = np.sort(energy)
    lower = np.concatenate(lower_values) if lower_values else values.reshape(-1)
    center = np.concatenate(center_values) if center_values else values.reshape(-1)
    border = np.concatenate(border_values) if border_values else values.reshape(-1)
    total_energy = float(np.sum(energy))
    return {
        "grid_p95_max": float(np.max(p95)),
        "grid_p95_mean": float(np.mean(p95)),
        "grid_p95_std": float(np.std(p95)),
        "grid_energy_max": float(np.max(energy)),
        "grid_energy_sum": total_energy,
        "grid_energy_concentration": float(np.max(energy) / (total_energy + EPS)),
        "grid_energy_top2_ratio": float(order[-1] / (order[-2] + EPS)) if len(order) >= 2 else 0.0,
        "grid_lower_p95": float(np.percentile(lower, 95)),
        "grid_center_p95": float(np.percentile(center, 95)),
        "grid_border_p95": float(np.percentile(border, 95)),
        "grid_max_cell_coherence": float(np.max(coherences)) if coherences else 0.0,
    }


def _affine_components(mat: np.ndarray | None, width: int) -> tuple[float, float, float]:
    if mat is None:
        return 0.0, 0.0, 0.0
    a, b, tx = mat[0]
    c, d, ty = mat[1]
    theta_px = float(np.arctan2(c, a) * width)
    scale_x = float(np.sqrt(a * a + c * c))
    scale_y = float(np.sqrt(b * b + d * d))
    scale_delta = 0.5 * (scale_x + scale_y) - 1.0
    trans = float(np.hypot(tx, ty))
    return trans, theta_px, scale_delta


def _pair_features(prev: np.ndarray, curr: np.ndarray, grid: int, sample_step: int, ransac_threshold: float) -> dict[str, float]:
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mat, inlier_ratio, fit_error = _fit_affine_from_flow(flow, sample_step=sample_step, ransac_threshold=ransac_threshold)
    if mat is not None:
        expected = _affine_expected_flow(mat, prev.shape)
        residual = flow - expected
        warped = cv2.warpAffine(prev, mat, (prev.shape[1], prev.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        median = np.median(flow.reshape(-1, 2), axis=0).reshape(1, 1, 2)
        residual = flow - median
        warped = prev
    raw_mag = np.linalg.norm(flow, axis=2)
    res_mag = np.linalg.norm(residual, axis=2)
    diff = np.abs(curr.astype(np.float32) - warped.astype(np.float32)) / 255.0
    trans, theta_px, scale_delta = _affine_components(mat, prev.shape[1])
    res_grid = _grid_reduce(res_mag, residual, grid=grid)
    diff_grid = _grid_reduce(diff, None, grid=grid)
    mean_vec = np.mean(residual.reshape(-1, 2), axis=0)
    res_mean_mag = float(np.mean(res_mag))
    out = {
        "raw_p95": float(np.percentile(raw_mag, 95)),
        "raw_p99": float(np.percentile(raw_mag, 99)),
        "raw_energy": float(np.mean(raw_mag * raw_mag)),
        "res_p90": float(np.percentile(res_mag, 90)),
        "res_p95": float(np.percentile(res_mag, 95)),
        "res_p99": float(np.percentile(res_mag, 99)),
        "res_max": float(np.max(res_mag)),
        "res_energy": float(np.mean(res_mag * res_mag)),
        "res_vector_coherence": float(np.linalg.norm(mean_vec) / (res_mean_mag + EPS)),
        "res_direction_entropy": _direction_entropy(residual[..., 0], residual[..., 1], res_mag),
        "affine_translation": trans,
        "affine_theta_px": theta_px,
        "affine_scale_delta_abs": abs(scale_delta),
        "affine_inlier_ratio": inlier_ratio,
        "affine_fit_error": fit_error,
        "res_to_global": float(res_grid["grid_p95_max"] / (trans + EPS)),
        "diff_p95": float(np.percentile(diff, 95)),
        "diff_p99": float(np.percentile(diff, 99)),
        "diff_energy": float(np.mean(diff * diff)),
    }
    out.update({f"res_{k}": v for k, v in res_grid.items()})
    out.update({f"diff_{k}": v for k, v in diff_grid.items()})
    return out


def _basic_stats(x: np.ndarray) -> list[float]:
    if x.size == 0:
        return [0.0] * 8
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.percentile(x, 50)),
        float(np.percentile(x, 75)),
        float(np.percentile(x, 90)),
        float(np.percentile(x, 95)),
    ]


def _window_stats(times: np.ndarray, values: np.ndarray, name: str) -> tuple[list[float], list[str]]:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pre = values[(times >= 1.0) & (times < 4.2)]
    impact = values[(times >= 4.4) & (times < 5.8)]
    post = values[(times >= 5.8) & (times < 7.2)]
    if pre.size == 0:
        pre = values
    if impact.size == 0:
        impact = values
    if post.size == 0:
        post = values
    pre_med = float(np.median(pre))
    pre_p95 = float(np.percentile(pre, 95))
    pre_mad = float(np.median(np.abs(pre - pre_med)))
    imp_max = float(np.max(impact))
    imp_mean = float(np.mean(impact))
    post_mean = float(np.mean(post))
    threshold = pre_med + 3.0 * 1.4826 * pre_mad
    vals = [
        imp_max,
        imp_mean,
        float(np.percentile(impact, 95)),
        float((imp_max + EPS) / (pre_med + EPS)),
        float((np.percentile(impact, 95) + EPS) / (pre_p95 + EPS)),
        float((imp_max - pre_med) / (1.4826 * pre_mad + 1e-3)),
        float((post_mean + EPS) / (imp_mean + EPS)),
        float(np.mean(impact > threshold)),
    ]
    names = [
        f"{name}_impact_max",
        f"{name}_impact_mean",
        f"{name}_impact_p95",
        f"{name}_impact_ratio",
        f"{name}_impact_p95_ratio",
        f"{name}_impact_z",
        f"{name}_post_over_impact",
        f"{name}_impact_width",
    ]
    for label, part in [("all", values), ("pre", pre), ("impact", impact), ("post", post)]:
        stats = _basic_stats(part)
        vals.extend(stats)
        names.extend([f"{name}_{label}_{stat}" for stat in ["mean", "std", "min", "max", "p50", "p75", "p90", "p95"]])
    return vals, names


def _derivative_stats(times: np.ndarray, values: np.ndarray, name: str) -> tuple[list[float], list[str]]:
    if values.size < 3:
        return [0.0] * 8, [
            f"{name}_d1_impact_max",
            f"{name}_d1_impact_ratio",
            f"{name}_d2_impact_max",
            f"{name}_d2_impact_ratio",
            f"{name}_d1_post_over_impact",
            f"{name}_d2_post_over_impact",
            f"{name}_rise_drop_ratio",
            f"{name}_signed_area_stop_index",
        ]
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    dt = max(dt, EPS)
    d1 = np.diff(values, prepend=values[0]) / dt
    d2 = np.diff(d1, prepend=d1[0]) / dt
    pre = (times >= 1.0) & (times < 4.2)
    impact = (times >= 4.4) & (times < 5.8)
    post = (times >= 5.8) & (times < 7.2)
    pre_abs_d1 = np.abs(d1[pre]) if np.any(pre) else np.abs(d1)
    pre_abs_d2 = np.abs(d2[pre]) if np.any(pre) else np.abs(d2)
    imp_abs_d1 = np.abs(d1[impact]) if np.any(impact) else np.abs(d1)
    imp_abs_d2 = np.abs(d2[impact]) if np.any(impact) else np.abs(d2)
    post_abs_d1 = np.abs(d1[post]) if np.any(post) else np.abs(d1)
    post_abs_d2 = np.abs(d2[post]) if np.any(post) else np.abs(d2)
    imp_values = values[impact] if np.any(impact) else values
    rise = float(np.max(np.maximum(d1[impact], 0.0))) if np.any(impact) else float(np.max(np.maximum(d1, 0.0)))
    drop = float(np.max(np.maximum(-d1[impact], 0.0))) if np.any(impact) else float(np.max(np.maximum(-d1, 0.0)))
    net_area = float(np.sum(imp_values))
    abs_area = float(np.sum(np.abs(imp_values)))
    vals = [
        float(np.max(imp_abs_d1)),
        float((np.max(imp_abs_d1) + EPS) / (np.median(pre_abs_d1) + EPS)),
        float(np.max(imp_abs_d2)),
        float((np.max(imp_abs_d2) + EPS) / (np.median(pre_abs_d2) + EPS)),
        float((np.mean(post_abs_d1) + EPS) / (np.mean(imp_abs_d1) + EPS)),
        float((np.mean(post_abs_d2) + EPS) / (np.mean(imp_abs_d2) + EPS)),
        float((rise + EPS) / (drop + EPS)),
        float(1.0 - abs(net_area) / (abs_area + EPS)),
    ]
    names = [
        f"{name}_d1_impact_max",
        f"{name}_d1_impact_ratio",
        f"{name}_d2_impact_max",
        f"{name}_d2_impact_ratio",
        f"{name}_d1_post_over_impact",
        f"{name}_d2_post_over_impact",
        f"{name}_rise_drop_ratio",
        f"{name}_signed_area_stop_index",
    ]
    return vals, names


def _summarize(sequence: np.ndarray, times: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    feats: list[float] = []
    feat_names: list[str] = []
    derivative_names = {
        "res_grid_p95_max",
        "res_grid_energy_max",
        "res_grid_energy_concentration",
        "res_to_global",
        "diff_grid_p95_max",
        "diff_grid_energy_max",
        "affine_translation",
        "affine_theta_px",
        "res_energy",
    }
    for idx, name in enumerate(names):
        vals, vals_names = _window_stats(times, sequence[:, idx], name)
        feats.extend(vals)
        feat_names.extend(vals_names)
        if name in derivative_names:
            vals, vals_names = _derivative_stats(times, sequence[:, idx], name)
            feats.extend(vals)
            feat_names.extend(vals_names)

    name_to_idx = {name: i for i, name in enumerate(names)}
    composite_specs = [
        ("residual_motion_energy_proxy", ["res_grid_energy_max", "res_grid_energy_concentration", "res_to_global"]),
        ("image_structure_energy_proxy", ["diff_grid_energy_max", "diff_grid_energy_concentration", "diff_p99"]),
        ("global_impulse_proxy", ["affine_translation", "affine_theta_px", "affine_fit_error"]),
    ]
    for comp_name, channels in composite_specs:
        present = [name_to_idx[c] for c in channels if c in name_to_idx]
        if not present:
            continue
        comp = np.prod(np.maximum(sequence[:, present], 0.0) + 1.0, axis=1) - 1.0
        vals, vals_names = _window_stats(times, comp, comp_name)
        feats.extend(vals)
        feat_names.extend(vals_names)
        vals, vals_names = _derivative_stats(times, comp, comp_name)
        feats.extend(vals)
        feat_names.extend(vals_names)
    arr = np.asarray(feats, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, feat_names


def extract_one(
    video_path: str | Path,
    out_dir: str | Path,
    width: int,
    frame_step: int,
    grid: int,
    sample_step: int,
    ransac_threshold: float,
    force: bool = False,
) -> Path:
    ensure_dirs(out_dir)
    out_path = Path(out_dir) / f"{stable_video_id(video_path)}.npz"
    if out_path.exists() and not force:
        return out_path
    frames, frame_indices, fps = _read_sampled_gray(video_path, width=width, frame_step=frame_step)
    rows: list[dict[str, float]] = []
    times: list[float] = []
    for i in range(1, len(frames)):
        rows.append(_pair_features(frames[i - 1], frames[i], grid=grid, sample_step=sample_step, ransac_threshold=ransac_threshold))
        times.append(float(frame_indices[i] / fps))
    names = list(rows[0].keys())
    sequence = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float32)
    times_arr = np.asarray(times, dtype=np.float32)
    summary, summary_names = _summarize(sequence, times_arr, names)
    np.savez_compressed(
        out_path,
        sequence=sequence,
        summary=summary,
        times=times_arr,
        sequence_names=np.asarray(names),
        summary_names=np.asarray(summary_names),
        video_path=str(video_path),
        width=np.asarray([width], dtype=np.int32),
        frame_step=np.asarray([frame_step], dtype=np.int32),
        fps=np.asarray([fps], dtype=np.float32),
    )
    return out_path


def _worker(args: tuple[str, str, int, int, int, int, float, bool]) -> str:
    video_path, out_dir, width, frame_step, grid, sample_step, ransac_threshold, force = args
    return str(extract_one(video_path, out_dir, width, frame_step, grid, sample_step, ransac_threshold, force=force))


def _unique_paths(csv_paths: list[str]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for csv_path in csv_paths:
        for row in read_split_csv(csv_path):
            if row["path"] not in seen:
                paths.append(row["path"])
                seen.add(row["path"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", default=[])
    parser.add_argument("--out-dir", default="outputs/processed_744/residual_flow_physics")
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--grid", type=int, default=6)
    parser.add_argument("--sample-step", type=int, default=12)
    parser.add_argument("--ransac-threshold", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    csv_paths = args.csv or ["splits/processed_744/train.csv", "splits/processed_744/test.csv"]
    paths = _unique_paths(csv_paths)
    ensure_dirs(args.out_dir)
    outputs: list[str] = []
    if args.workers <= 1:
        for path in tqdm(paths, desc="residual-flow"):
            outputs.append(
                str(
                    extract_one(
                        path,
                        args.out_dir,
                        args.width,
                        args.frame_step,
                        args.grid,
                        args.sample_step,
                        args.ransac_threshold,
                        force=args.force,
                    )
                )
            )
    else:
        worker_args = [
            (path, args.out_dir, args.width, args.frame_step, args.grid, args.sample_step, args.ransac_threshold, args.force)
            for path in paths
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, item) for item in worker_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc="residual-flow"):
                outputs.append(future.result())
    write_json(
        Path(args.out_dir) / "manifest.json",
        {
            "count": len(outputs),
            "features": sorted(outputs),
            "width": args.width,
            "frame_step": args.frame_step,
            "grid": args.grid,
            "sample_step": args.sample_step,
            "ransac_threshold": args.ransac_threshold,
        },
    )
    print(f"extracted {len(outputs)} residual-flow feature files")


if __name__ == "__main__":
    main()
