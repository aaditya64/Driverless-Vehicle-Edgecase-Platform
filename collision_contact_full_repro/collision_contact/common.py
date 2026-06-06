from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def read_split_csv(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "path": row["path"],
                    "label": int(row["label"]),
                    "label_name": row.get("label_name", ""),
                }
            )
    return rows


def stable_video_id(video_path: str | Path) -> str:
    path = str(video_path).replace(os.sep, "/")
    stem = Path(path).stem
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}"


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def robust_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    med = np.nanmedian(x, axis=0, keepdims=True)
    q75 = np.nanpercentile(x, 75, axis=0, keepdims=True)
    q25 = np.nanpercentile(x, 25, axis=0, keepdims=True)
    scale = np.maximum(q75 - q25, eps)
    return (x - med) / scale


def pad_or_trim_2d(x: np.ndarray, target_frames: int) -> np.ndarray:
    if x.shape[0] == target_frames:
        return x.astype(np.float32, copy=False)
    if x.shape[0] > target_frames:
        return x[:target_frames].astype(np.float32, copy=False)
    if x.shape[0] == 0:
        return np.zeros((target_frames, x.shape[1]), dtype=np.float32)
    pad_len = target_frames - x.shape[0]
    pad = np.repeat(x[-1:, :], pad_len, axis=0)
    return np.concatenate([x, pad], axis=0).astype(np.float32, copy=False)


def pad_or_trim_1d(x: np.ndarray, target_frames: int) -> np.ndarray:
    if x.shape[0] == target_frames:
        return x.astype(np.float32, copy=False)
    if x.shape[0] > target_frames:
        return x[:target_frames].astype(np.float32, copy=False)
    if x.shape[0] == 0:
        return np.zeros((target_frames,), dtype=np.float32)
    pad_len = target_frames - x.shape[0]
    pad = np.repeat(x[-1:], pad_len, axis=0)
    return np.concatenate([x, pad], axis=0).astype(np.float32, copy=False)

