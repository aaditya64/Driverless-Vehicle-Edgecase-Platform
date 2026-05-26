"""Extract object-box residual motion features after global camera stabilization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .common import ensure_dirs, read_split_csv, stable_video_id, write_json
from .extract_residual_flow_physics_features import _fit_affine_from_flow, _affine_expected_flow
from .extract_yolo_interaction_features import RISK_CLASSES, _ensure_torchvision_nms_stub, _ultralytics_detections


EPS = 1e-6


def _read_sampled_frames(video_path: str | Path, width: int, frame_step: int) -> tuple[list[np.ndarray], np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    idxs: list[int] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_step == 0:
            h, w = frame.shape[:2]
            if width > 0 and w != width:
                frame = cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            idxs.append(idx)
        idx += 1
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"Need at least two sampled frames: {video_path}")
    return frames, np.asarray(idxs, dtype=np.int32), fps


def _box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    return float(inter / (area_a + area_b - inter + EPS))


def _box_geometry(box: dict[str, float], w: int, h: int) -> tuple[float, float, float, float, float]:
    bw = max(0.0, box["x2"] - box["x1"])
    bh = max(0.0, box["y2"] - box["y1"])
    cx = (box["x1"] + box["x2"]) * 0.5
    cy = (box["y1"] + box["y2"]) * 0.5
    area = bw * bh / max(w * h, 1)
    center_weight = max(0.0, 1.0 - abs(cx / max(w, 1) - 0.5) * 2.0)
    bottom_weight = 0.5 + 0.5 * box["y2"] / max(h, 1)
    threat = area * float(box["conf"]) * (0.25 + 0.75 * center_weight) * bottom_weight
    return cx, cy, area, threat, bottom_weight


def _mask_from_box(box: dict[str, float], h: int, w: int, pad: float = 0.08) -> tuple[slice, slice] | None:
    bw = box["x2"] - box["x1"]
    bh = box["y2"] - box["y1"]
    x1 = int(max(0, np.floor(box["x1"] - pad * bw)))
    y1 = int(max(0, np.floor(box["y1"] - pad * bh)))
    x2 = int(min(w, np.ceil(box["x2"] + pad * bw)))
    y2 = int(min(h, np.ceil(box["y2"] + pad * bh)))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    return slice(y1, y2), slice(x1, x2)


def _object_pair_features(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_dets: list[dict[str, float]],
    curr_dets: list[dict[str, float]],
    sample_step: int,
    ransac_threshold: float,
) -> dict[str, float]:
    h, w = curr_gray.shape
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mat, inlier_ratio, fit_error = _fit_affine_from_flow(flow, sample_step=sample_step, ransac_threshold=ransac_threshold)
    if mat is not None:
        residual = flow - _affine_expected_flow(mat, curr_gray.shape)
        warped = cv2.warpAffine(prev_gray, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        residual = flow - np.median(flow.reshape(-1, 2), axis=0).reshape(1, 1, 2)
        warped = prev_gray
    rmag = np.linalg.norm(residual, axis=2)
    diff = np.abs(curr_gray.astype(np.float32) - warped.astype(np.float32)) / 255.0
    object_rows: list[dict[str, float]] = []
    for det in curr_dets:
        if int(det["cls"]) not in RISK_CLASSES:
            continue
        mask = _mask_from_box(det, h, w)
        if mask is None:
            continue
        ys, xs = mask
        cell = rmag[ys, xs]
        vec = residual[ys, xs]
        diff_cell = diff[ys, xs]
        cx, cy, area, threat, bottom_weight = _box_geometry(det, w, h)
        best_iou = 0.0
        center_shift = 0.0
        area_change = 0.0
        for prev in prev_dets:
            if int(prev["cls"]) not in RISK_CLASSES:
                continue
            iou = _box_iou(det, prev)
            if iou > best_iou:
                pcx, pcy, parea, _, _ = _box_geometry(prev, w, h)
                best_iou = iou
                center_shift = float(np.hypot(cx - pcx, cy - pcy))
                area_change = float(abs(np.log((area + EPS) / (parea + EPS))))
        mean_vec = np.mean(vec.reshape(-1, 2), axis=0)
        mean_mag = float(np.mean(cell))
        object_rows.append(
            {
                "res_p95": float(np.percentile(cell, 95)),
                "res_p99": float(np.percentile(cell, 99)),
                "res_energy": float(np.mean(cell * cell)),
                "res_coherence": float(np.linalg.norm(mean_vec) / (mean_mag + EPS)),
                "diff_p95": float(np.percentile(diff_cell, 95)),
                "diff_energy": float(np.mean(diff_cell * diff_cell)),
                "area": area,
                "threat": threat,
                "bottom_weight": bottom_weight,
                "center_shift": center_shift,
                "area_change": area_change,
                "track_iou": best_iou,
                "score": float((np.percentile(cell, 95) + 1.0) * (np.mean(cell * cell) + 1.0) * (threat + EPS) * (diff_cell.mean() + 1.0)),
            }
        )
    if not object_rows:
        return {
            "object_count": 0.0,
            "object_score_max": 0.0,
            "object_res_p95_max": 0.0,
            "object_res_p99_max": 0.0,
            "object_res_energy_max": 0.0,
            "object_res_energy_sum": 0.0,
            "object_res_coherence_max": 0.0,
            "object_diff_p95_max": 0.0,
            "object_diff_energy_max": 0.0,
            "object_area_max": 0.0,
            "object_threat_max": 0.0,
            "object_center_shift_max": 0.0,
            "object_area_change_max": 0.0,
            "object_track_iou_max": 0.0,
            "affine_inlier_ratio": inlier_ratio,
            "affine_fit_error": fit_error,
        }
    keys = object_rows[0].keys()
    arr = {key: np.asarray([row[key] for row in object_rows], dtype=np.float32) for key in keys}
    return {
        "object_count": float(len(object_rows)),
        "object_score_max": float(np.max(arr["score"])),
        "object_res_p95_max": float(np.max(arr["res_p95"])),
        "object_res_p99_max": float(np.max(arr["res_p99"])),
        "object_res_energy_max": float(np.max(arr["res_energy"])),
        "object_res_energy_sum": float(np.sum(arr["res_energy"])),
        "object_res_coherence_max": float(np.max(arr["res_coherence"])),
        "object_diff_p95_max": float(np.max(arr["diff_p95"])),
        "object_diff_energy_max": float(np.max(arr["diff_energy"])),
        "object_area_max": float(np.max(arr["area"])),
        "object_threat_max": float(np.max(arr["threat"])),
        "object_center_shift_max": float(np.max(arr["center_shift"])),
        "object_area_change_max": float(np.max(arr["area_change"])),
        "object_track_iou_max": float(np.max(arr["track_iou"])),
        "affine_inlier_ratio": inlier_ratio,
        "affine_fit_error": fit_error,
    }


def _dynamic_summary(times: np.ndarray, sequence: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    feats: list[float] = []
    feat_names: list[str] = []
    search = (times >= 2.0) & (times <= 8.0)
    early = (times >= 0.5) & (times < 2.5)
    if not np.any(search):
        search = np.ones_like(times, dtype=bool)
    if not np.any(early):
        early = np.ones_like(times, dtype=bool)
    for idx, name in enumerate(names):
        x = np.nan_to_num(sequence[:, idx].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        x = np.maximum(x, 0.0)
        xs = x[search]
        ts = times[search]
        peak_idx = int(np.argmax(xs))
        peak = float(xs[peak_idx])
        peak_time = float(ts[peak_idx])
        q25 = float(np.percentile(x, 25))
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        early_med = float(np.median(x[early]))
        post = x[(times > peak_time + 0.25) & (times <= peak_time + 1.25)]
        around = x[(times >= peak_time - 0.5) & (times <= peak_time + 0.5)]
        if post.size == 0:
            post = x
        if around.size == 0:
            around = x
        dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        dt = max(dt, EPS)
        d1 = np.diff(x, prepend=x[0]) / dt
        d2 = np.diff(d1, prepend=d1[0]) / dt
        vals = [
            peak,
            float((peak + EPS) / (q25 + EPS)),
            float((peak + EPS) / (early_med + EPS)),
            float((peak - med) / (1.4826 * mad + 1e-3)),
            peak_time,
            float((peak + EPS) / (np.mean(around) + EPS)),
            float((np.mean(post) + EPS) / (peak + EPS)),
            float(np.percentile(xs, 95)),
            float(np.max(np.abs(d1[search]))),
            float(np.max(np.abs(d2[search]))),
        ]
        suffixes = ["peak", "peak_q25_ratio", "peak_early_ratio", "peak_z", "peak_time", "sharpness", "post_over_peak", "p95", "d1_peak", "d2_peak"]
        feats.extend(vals)
        feat_names.extend([f"{name}_{suffix}" for suffix in suffixes])
    return np.asarray(feats, dtype=np.float32), feat_names


def extract_one(
    video_path: str | Path,
    model: Any,
    out_dir: str | Path,
    width: int,
    frame_step: int,
    image_size: int,
    conf: float,
    iou: float,
    device: str | int,
    batch: int,
    sample_step: int,
    ransac_threshold: float,
    force: bool = False,
) -> Path:
    ensure_dirs(out_dir)
    out_path = Path(out_dir) / f"{stable_video_id(video_path)}.npz"
    if out_path.exists() and not force:
        return out_path
    frames, frame_idxs, fps = _read_sampled_frames(video_path, width=width, frame_step=frame_step)
    results = model.predict(frames, imgsz=image_size, conf=conf, iou=iou, device=device, batch=batch, verbose=False)
    detections = [_ultralytics_detections(result) for result in results]
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    rows: list[dict[str, float]] = []
    times: list[float] = []
    for i in range(1, len(frames)):
        rows.append(
            _object_pair_features(
                gray[i - 1],
                gray[i],
                detections[i - 1],
                detections[i],
                sample_step=sample_step,
                ransac_threshold=ransac_threshold,
            )
        )
        times.append(float(frame_idxs[i] / fps))
    names = list(rows[0].keys())
    sequence = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float32)
    times_arr = np.asarray(times, dtype=np.float32)
    summary, summary_names = _dynamic_summary(times_arr, sequence, names)
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


def _rows_from_csvs(paths: list[str]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for csv_path in paths:
        for row in read_split_csv(csv_path):
            if row["path"] not in seen:
                rows.append(row)
                seen.add(row["path"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", default=[])
    parser.add_argument("--out-dir", default="outputs/processed_744/object_residual_physics_w320_s4")
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--frame-step", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--sample-step", type=int, default=12)
    parser.add_argument("--ransac-threshold", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    csv_paths = args.csv or ["splits/processed_744/train.csv", "splits/processed_744/test.csv"]
    rows = _rows_from_csvs(csv_paths)
    if args.limit is not None:
        rows = rows[: args.limit]
    ensure_dirs(args.out_dir)
    _ensure_torchvision_nms_stub()
    import torch
    from ultralytics import YOLO

    device: str | int = 0 if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    model = YOLO(args.model)
    outputs: list[str] = []
    for row in tqdm(rows, desc="object-residual"):
        outputs.append(
            str(
                extract_one(
                    row["path"],
                    model,
                    args.out_dir,
                    args.width,
                    args.frame_step,
                    args.image_size,
                    args.conf,
                    args.iou,
                    device,
                    args.batch,
                    args.sample_step,
                    args.ransac_threshold,
                    force=args.force,
                )
            )
        )
    write_json(
        Path(args.out_dir) / "manifest.json",
        {
            "count": len(outputs),
            "features": outputs,
            "width": args.width,
            "frame_step": args.frame_step,
            "image_size": args.image_size,
            "conf": args.conf,
            "iou": args.iou,
        },
    )
    print(f"extracted {len(outputs)} object residual feature files")


if __name__ == "__main__":
    main()
