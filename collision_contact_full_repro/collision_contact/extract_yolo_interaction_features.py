"""Extract object-interaction features from YOLO vehicle detections."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .common import ensure_dirs, load_config, read_split_csv, stable_video_id, write_json


VEHICLE_CLASSES = {2, 3, 5, 7}
RISK_CLASSES = {0, 1, 2, 3, 5, 7}
_TORCHVISION_LIB = None


def _letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    h, w = frame.shape[:2]
    gain = min(size / max(h, 1), size / max(w, 1))
    new_w = int(round(w * gain))
    new_h = int(round(h * gain))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    canvas[int(round(pad_y)) : int(round(pad_y)) + new_h, int(round(pad_x)) : int(round(pad_x)) + new_w] = resized
    return canvas, gain, pad_x, pad_y


def _nms(boxes: list[list[float]], scores: list[float], iou_threshold: float) -> list[int]:
    if not boxes:
        return []
    arr = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores_arr.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union
        order = order[1:][iou <= iou_threshold]
    return keep


def _detect(
    net: cv2.dnn.Net,
    frame: np.ndarray,
    size: int,
    conf_threshold: float,
    iou_threshold: float,
) -> list[dict[str, float]]:
    h, w = frame.shape[:2]
    img, gain, pad_x, pad_y = _letterbox(frame, size)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0 / 255.0, size=(size, size), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    pred = np.squeeze(out)
    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
        pred = pred.T
    boxes: list[list[float]] = []
    scores: list[float] = []
    classes: list[int] = []
    for row in pred:
        cls_scores = row[4:]
        cls = int(np.argmax(cls_scores))
        conf = float(cls_scores[cls])
        if conf < conf_threshold or cls not in RISK_CLASSES:
            continue
        cx, cy, bw, bh = map(float, row[:4])
        x1 = (cx - bw / 2.0 - pad_x) / gain
        y1 = (cy - bh / 2.0 - pad_y) / gain
        x2 = (cx + bw / 2.0 - pad_x) / gain
        y2 = (cy + bh / 2.0 - pad_y) / gain
        x1 = float(np.clip(x1, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y2 = float(np.clip(y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        classes.append(cls)
    keep = _nms(boxes, scores, iou_threshold)
    detections = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": scores[i], "cls": float(classes[i])})
    return detections


def _ensure_torchvision_nms_stub() -> None:
    global _TORCHVISION_LIB
    import torch

    try:
        import torchvision  # noqa: F401
        return
    except RuntimeError as exc:
        if "torchvision::nms" not in str(exc):
            raise
    except Exception:
        return
    try:
        if _TORCHVISION_LIB is None:
            _TORCHVISION_LIB = torch.library.Library("torchvision", "DEF")
            _TORCHVISION_LIB.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass


def _ultralytics_detections(result: Any) -> list[dict[str, float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    detections = []
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
        if int(cls) not in RISK_CLASSES:
            continue
        detections.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "conf": float(conf), "cls": float(cls)})
    return detections


def _subset_stats(dets: list[dict[str, float]], w: int, h: int, classes: set[int]) -> list[float]:
    selected = [d for d in dets if int(d["cls"]) in classes]
    if not selected:
        return [0.0] * 14
    areas = []
    heights = []
    widths = []
    bottoms = []
    centers = []
    threats = []
    confs = []
    for d in selected:
        bw = max(0.0, d["x2"] - d["x1"])
        bh = max(0.0, d["y2"] - d["y1"])
        area = bw * bh / max(w * h, 1)
        cx = (d["x1"] + d["x2"]) * 0.5 / max(w, 1)
        cy = (d["y1"] + d["y2"]) * 0.5 / max(h, 1)
        bottom = d["y2"] / max(h, 1)
        center_dist = abs(cx - 0.5) * 2.0
        lower_weight = 0.5 + 0.5 * bottom
        center_weight = max(0.0, 1.0 - center_dist)
        threat = area * float(d["conf"]) * (0.25 + 0.75 * center_weight) * lower_weight
        areas.append(area)
        heights.append(bh / max(h, 1))
        widths.append(bw / max(w, 1))
        bottoms.append(bottom)
        centers.append(center_dist)
        threats.append(threat)
        confs.append(float(d["conf"]))
    return [
        float(len(selected)),
        float(max(confs)),
        float(np.max(areas)),
        float(np.sum(areas)),
        float(np.mean(areas)),
        float(np.max(heights)),
        float(np.max(widths)),
        float(np.max(bottoms)),
        float(np.min(centers)),
        float(np.mean(centers)),
        float(np.max(threats)),
        float(np.sum(threats)),
        float(np.mean(threats)),
        float(np.std(threats)),
    ]


def _frame_features(dets: list[dict[str, float]], w: int, h: int) -> np.ndarray:
    vehicle = _subset_stats(dets, w, h, VEHICLE_CLASSES)
    risk = _subset_stats(dets, w, h, RISK_CLASSES)
    cls_counts = []
    for cls in [0, 1, 2, 3, 5, 7]:
        cls_counts.append(float(sum(1 for d in dets if int(d["cls"]) == cls)))
    return np.asarray(vehicle + risk + cls_counts, dtype=np.float32)


def _safe_slope(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, y.size, dtype=np.float32)
    return float(np.polyfit(x, y.astype(np.float32), 1)[0])


def _summarize(seq: np.ndarray) -> np.ndarray:
    diffs = np.diff(seq, axis=0) if seq.shape[0] > 1 else np.zeros_like(seq)
    pos_diffs = np.maximum(diffs, 0.0)
    key_indices = [2, 7, 10, 11, 16, 21, 24, 25]
    slopes = np.asarray([_safe_slope(seq[:, i]) for i in key_indices], dtype=np.float32)
    max_pos = pos_diffs.max(axis=0) if pos_diffs.size else np.zeros(seq.shape[1], dtype=np.float32)
    last_first = seq[-1] - seq[0]
    return np.concatenate(
        [
            seq.mean(axis=0),
            seq.std(axis=0),
            seq.max(axis=0),
            seq.min(axis=0),
            last_first,
            max_pos,
            slopes,
        ]
    ).astype(np.float32)


def _read_uniform_frames(video_path: str | Path, num_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = np.linspace(0, max(n - 1, 0), num_frames).round().astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]


def extract_one(
    video_path: str | Path,
    net: cv2.dnn.Net,
    out_dir: str | Path,
    num_frames: int,
    image_size: int,
    conf_threshold: float,
    iou_threshold: float,
    force: bool = False,
) -> Path:
    out_path = Path(out_dir) / f"{stable_video_id(video_path)}.npz"
    if out_path.exists() and not force:
        return out_path
    frames = _read_uniform_frames(video_path, num_frames)
    feats = []
    for frame in frames:
        dets = _detect(net, frame, image_size, conf_threshold, iou_threshold)
        h, w = frame.shape[:2]
        feats.append(_frame_features(dets, w, h))
    frame_features = np.stack(feats).astype(np.float32)
    summary = _summarize(frame_features)
    np.savez_compressed(
        out_path,
        frame_features=frame_features,
        summary=summary,
        video_path=str(video_path),
        num_frames=np.asarray([num_frames], dtype=np.int32),
    )
    return out_path


def extract_one_ultralytics(
    video_path: str | Path,
    model: Any,
    out_dir: str | Path,
    num_frames: int,
    image_size: int,
    conf_threshold: float,
    iou_threshold: float,
    device: str | int,
    batch: int,
    force: bool = False,
) -> Path:
    out_path = Path(out_dir) / f"{stable_video_id(video_path)}.npz"
    if out_path.exists() and not force:
        return out_path
    frames = _read_uniform_frames(video_path, num_frames)
    results = model.predict(
        frames,
        imgsz=image_size,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        batch=batch,
        verbose=False,
    )
    feats = []
    for frame, result in zip(frames, results):
        h, w = frame.shape[:2]
        feats.append(_frame_features(_ultralytics_detections(result), w, h))
    frame_features = np.stack(feats).astype(np.float32)
    summary = _summarize(frame_features)
    np.savez_compressed(
        out_path,
        frame_features=frame_features,
        summary=summary,
        video_path=str(video_path),
        num_frames=np.asarray([num_frames], dtype=np.int32),
    )
    return out_path


def _save_ultralytics_features(
    video_path: str | Path,
    frames: list[np.ndarray],
    results: list[Any],
    out_path: str | Path,
    num_frames: int,
) -> Path:
    feats = []
    for frame, result in zip(frames, results):
        h, w = frame.shape[:2]
        feats.append(_frame_features(_ultralytics_detections(result), w, h))
    frame_features = np.stack(feats).astype(np.float32)
    summary = _summarize(frame_features)
    out_path = Path(out_path)
    np.savez_compressed(
        out_path,
        frame_features=frame_features,
        summary=summary,
        video_path=str(video_path),
        num_frames=np.asarray([num_frames], dtype=np.int32),
    )
    return out_path


def extract_batch_ultralytics(
    batch_items: list[tuple[dict[str, Any], Path, list[np.ndarray]]],
    model: Any,
    num_frames: int,
    image_size: int,
    conf_threshold: float,
    iou_threshold: float,
    device: str | int,
    batch: int,
) -> list[tuple[dict[str, Any], Path]]:
    if not batch_items:
        return []
    flat_frames = [frame for _, _, frames in batch_items for frame in frames]
    results = model.predict(
        flat_frames,
        imgsz=image_size,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        batch=batch,
        verbose=False,
    )
    saved = []
    offset = 0
    for row, out_path, frames in batch_items:
        count = len(frames)
        _save_ultralytics_features(row["path"], frames, results[offset : offset + count], out_path, num_frames)
        offset += count
        saved.append((row, out_path))
    return saved


def _rows_from_csvs(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
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
    parser.add_argument("--backend", choices=["ultralytics", "opencv"], default="ultralytics")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-dir", default="outputs/processed_744/yolo_interaction")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--video-batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = _rows_from_csvs(args.csv or [cfg["paths"]["train_csv"], cfg["paths"]["test_csv"]])
    if args.limit is not None:
        rows = rows[: args.limit]
    ensure_dirs(args.out_dir)
    model_path = args.model or ("models/yolov8n.pt" if args.backend == "ultralytics" else "models/yolov8n.onnx")
    if args.backend == "ultralytics":
        _ensure_torchvision_nms_stub()
        import torch
        from ultralytics import YOLO

        device: str | int = 0 if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
        detector = YOLO(model_path)
    else:
        device = "opencv"
        detector = cv2.dnn.readNetFromONNX(model_path)
    print(f"backend={args.backend} model={model_path} device={device} videos={len(rows)} frames={args.num_frames}")
    manifest = []
    if args.backend == "ultralytics":
        pending: list[tuple[dict[str, Any], Path, list[np.ndarray]]] = []
        for row in tqdm(rows, desc="yolo"):
            out_path = Path(args.out_dir) / f"{stable_video_id(row['path'])}.npz"
            if out_path.exists() and not args.force:
                manifest.append({"path": row["path"], "label": int(row["label"]), "feature_path": str(out_path)})
                continue
            pending.append((row, out_path, _read_uniform_frames(row["path"], args.num_frames)))
            if len(pending) >= args.video_batch:
                for saved_row, saved_path in extract_batch_ultralytics(pending, detector, args.num_frames, args.image_size, args.conf, args.iou, device, args.batch):
                    manifest.append({"path": saved_row["path"], "label": int(saved_row["label"]), "feature_path": str(saved_path)})
                pending = []
        for saved_row, saved_path in extract_batch_ultralytics(pending, detector, args.num_frames, args.image_size, args.conf, args.iou, device, args.batch):
            manifest.append({"path": saved_row["path"], "label": int(saved_row["label"]), "feature_path": str(saved_path)})
    else:
        for row in tqdm(rows, desc="yolo"):
            out_path = extract_one(row["path"], detector, args.out_dir, args.num_frames, args.image_size, args.conf, args.iou, force=args.force)
            manifest.append({"path": row["path"], "label": int(row["label"]), "feature_path": str(out_path)})
    write_json(Path(args.out_dir) / "manifest.json", {"count": len(manifest), "items": manifest})
    print(f"extracted {len(manifest)} YOLO interaction feature files")


if __name__ == "__main__":
    main()
