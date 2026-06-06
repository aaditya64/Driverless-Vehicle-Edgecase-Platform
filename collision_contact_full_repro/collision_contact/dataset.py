from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import pad_or_trim_2d, read_split_csv, stable_video_id


@dataclass
class FeatureScaler:
    seq_mean: np.ndarray
    seq_std: np.ndarray
    hand_mean: np.ndarray
    hand_std: np.ndarray

    def transform_seq(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.seq_mean) / self.seq_std).astype(np.float32)

    def transform_hand(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.hand_mean) / self.hand_std).astype(np.float32)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            seq_mean=self.seq_mean,
            seq_std=self.seq_std,
            hand_mean=self.hand_mean,
            hand_std=self.hand_std,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        data = np.load(path, allow_pickle=True)
        return cls(
            seq_mean=data["seq_mean"],
            seq_std=data["seq_std"],
            hand_mean=data["hand_mean"],
            hand_std=data["hand_std"],
        )


def _feature_file(feature_dir: str | Path, video_path: str | Path) -> Path:
    return Path(feature_dir) / f"{stable_video_id(video_path)}.npz"


def _load_feature_arrays(
    feature_dir: str | Path,
    video_path: str | Path,
    sequence_keys: list[str] | None = None,
    handcrafted_key: str = "handcrafted",
) -> tuple[np.ndarray, np.ndarray]:
    path = _feature_file(feature_dir, video_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    data = np.load(path, allow_pickle=True)
    seq_parts = []
    keys = sequence_keys or ["raw_abs", "raw", "cwt_abs", "cwt", "swt_abs", "swt"]
    for key in keys:
        if key in data:
            seq_parts.append(data[key])
    if not seq_parts:
        seq_parts = [data["raw"], data["cwt"], data["swt"]]
    seq = np.concatenate(seq_parts, axis=1).astype(np.float32)
    if handcrafted_key in data:
        hand = data[handcrafted_key].astype(np.float32)
    else:
        hand = data["handcrafted"].astype(np.float32)
    return seq, hand


def fit_scaler(rows: list[dict[str, Any]], feature_dir: str | Path, cfg: dict[str, Any], eps: float = 1e-6) -> FeatureScaler:
    seqs = []
    hands = []
    sequence_keys = list(cfg["features"].get("sequence_keys", ["raw", "cwt", "swt"]))
    handcrafted_key = str(cfg["features"].get("handcrafted_key", "handcrafted"))
    for row in rows:
        seq, hand = _load_feature_arrays(feature_dir, row["path"], sequence_keys, handcrafted_key)
        seqs.append(seq)
        hands.append(hand)
    all_seq = np.concatenate(seqs, axis=0)
    all_hand = np.stack(hands, axis=0)
    seq_mean = all_seq.mean(axis=0, keepdims=True)
    seq_std = np.maximum(all_seq.std(axis=0, keepdims=True), eps)
    hand_mean = all_hand.mean(axis=0)
    hand_std = np.maximum(all_hand.std(axis=0), eps)
    return FeatureScaler(
        seq_mean=seq_mean.astype(np.float32),
        seq_std=seq_std.astype(np.float32),
        hand_mean=hand_mean.astype(np.float32),
        hand_std=hand_std.astype(np.float32),
    )


class NexarFeatureDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        cfg: dict[str, Any],
        scaler: FeatureScaler,
        train: bool,
    ) -> None:
        self.rows = rows
        self.cfg = cfg
        self.scaler = scaler
        self.train = train
        self.feature_dir = cfg["paths"]["feature_dir"]
        self.sequence_keys = list(cfg["features"].get("sequence_keys", ["raw", "cwt", "swt"]))
        self.handcrafted_key = str(cfg["features"].get("handcrafted_key", "handcrafted"))
        self.target_frames = int(cfg["video"]["target_frames"])
        self.aug_cfg = cfg["training"]["augmentation"]
        self.feature_slices = self._build_feature_slices()

    @classmethod
    def from_csv(cls, csv_path: str | Path, cfg: dict[str, Any], scaler: FeatureScaler, train: bool) -> "NexarFeatureDataset":
        return cls(read_split_csv(csv_path), cfg, scaler, train=train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        seq, hand = _load_feature_arrays(self.feature_dir, row["path"], self.sequence_keys, self.handcrafted_key)
        seq = pad_or_trim_2d(seq, self.target_frames)
        seq = self.scaler.transform_seq(seq)
        hand = self.scaler.transform_hand(hand)
        if self.train and bool(self.aug_cfg.get("enabled", False)):
            seq, hand = self._augment(seq, hand)
        return {
            "seq": torch.from_numpy(seq),
            "hand": torch.from_numpy(hand),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "path": row["path"],
        }

    def _build_feature_slices(self) -> dict[str, tuple[slice, int, int]]:
        raw_dim = len(self.cfg["features"].get("raw_channels", []))
        wavelet_dim = len(self.cfg["features"].get("wavelet_channels", []))
        cwt_freqs = int(self.cfg["features"].get("cwt", {}).get("n_freqs", 0))
        swt_levels = int(self.cfg["features"].get("swt", {}).get("levels", 0)) + 1
        dims = {
            "raw": raw_dim,
            "raw_abs": raw_dim,
            "cwt": wavelet_dim * cwt_freqs,
            "cwt_abs": wavelet_dim * cwt_freqs,
            "swt": wavelet_dim * swt_levels,
            "swt_abs": wavelet_dim * swt_levels,
        }
        out: dict[str, tuple[slice, int, int]] = {}
        start = 0
        for key in self.sequence_keys:
            dim = int(dims.get(key, 0))
            if dim <= 0:
                continue
            out[key] = (slice(start, start + dim), wavelet_dim if key.startswith(("cwt", "swt")) else dim, cwt_freqs if key.startswith("cwt") else swt_levels)
            start += dim
        return out

    @staticmethod
    def _edge_shift(seq: np.ndarray, shift: int) -> np.ndarray:
        if shift == 0:
            return seq
        out = np.empty_like(seq)
        if shift > 0:
            out[:shift] = seq[:1]
            out[shift:] = seq[:-shift]
        else:
            step = abs(shift)
            out[-step:] = seq[-1:]
            out[:-step] = seq[step:]
        return out

    @staticmethod
    def _smooth_warp(rng: np.random.Generator, length: int, std: float) -> np.ndarray:
        points = min(8, max(4, length // 40))
        anchors_x = np.linspace(0, length - 1, points)
        anchors_y = np.exp(rng.normal(0.0, std, size=points))
        return np.interp(np.arange(length), anchors_x, anchors_y).astype(np.float32)

    def _augment_group_amplitude(self, seq: np.ndarray, rng: np.random.Generator) -> None:
        amp_min = float(self.aug_cfg.get("amplitude_min", 1.0))
        amp_max = float(self.aug_cfg.get("amplitude_max", 1.0))
        if amp_min == 1.0 and amp_max == 1.0:
            return
        for feature_slice, _, _ in self.feature_slices.values():
            seq[:, feature_slice] *= rng.uniform(amp_min, amp_max)

    def _augment_magnitude_warp(self, seq: np.ndarray, rng: np.random.Generator) -> None:
        if rng.random() >= float(self.aug_cfg.get("magnitude_warp_prob", 0.0)):
            return
        std = float(self.aug_cfg.get("magnitude_warp_std", 0.04))
        if std <= 0:
            return
        for feature_slice, _, _ in self.feature_slices.values():
            seq[:, feature_slice] *= self._smooth_warp(rng, seq.shape[0], std)[:, None]

    def _augment_lowfreq_drift(self, seq: np.ndarray, rng: np.random.Generator) -> None:
        std = float(self.aug_cfg.get("lowfreq_drift_std", 0.0))
        raw = self.feature_slices.get("raw")
        if std <= 0 or raw is None:
            return
        feature_slice, _, _ = raw
        drift = np.cumsum(rng.normal(0.0, std, size=(seq.shape[0], 1)).astype(np.float32), axis=0)
        drift -= drift.mean(axis=0, keepdims=True)
        seq[:, feature_slice] += drift

    def _augment_time_mask(self, seq: np.ndarray, rng: np.random.Generator) -> None:
        if rng.random() >= float(self.aug_cfg.get("time_mask_prob", 0.0)):
            return
        max_len = min(int(self.aug_cfg.get("time_mask_max_frames", 0)), seq.shape[0])
        if max_len <= 1:
            return
        length = int(rng.integers(1, max_len + 1))
        start = int(rng.integers(0, max(seq.shape[0] - length + 1, 1)))
        seq[start : start + length] = 0.0

    def _augment_frequency_mask(self, seq: np.ndarray, rng: np.random.Generator) -> None:
        if rng.random() >= float(self.aug_cfg.get("freq_mask_prob", 0.0)):
            return
        max_bins = int(self.aug_cfg.get("freq_mask_max_bins", 0))
        for key in ["cwt", "cwt_abs"]:
            block = self.feature_slices.get(key)
            if block is None:
                continue
            feature_slice, channels, freqs = block
            if channels <= 0 or freqs <= 1 or max_bins <= 0:
                continue
            width = int(rng.integers(1, min(max_bins, freqs) + 1))
            start = int(rng.integers(0, freqs - width + 1))
            view = seq[:, feature_slice].reshape(seq.shape[0], channels, freqs)
            view[:, :, start : start + width] = 0.0

    def _augment_channel_dropout(self, seq: np.ndarray, rng: np.random.Generator) -> None:
        group_prob = float(self.aug_cfg.get("channel_group_dropout_prob", 0.0))
        if group_prob > 0:
            for key in ["cwt", "cwt_abs", "swt", "swt_abs"]:
                block = self.feature_slices.get(key)
                if block is None:
                    continue
                feature_slice, channels, width = block
                if channels <= 0 or width <= 0:
                    continue
                view = seq[:, feature_slice].reshape(seq.shape[0], channels, width)
                mask = rng.random(channels) < group_prob
                view[:, mask, :] = 0.0

        drop_prob = float(self.aug_cfg.get("feature_dropout_prob", 0.0))
        if drop_prob <= 0:
            return
        raw = self.feature_slices.get("raw")
        if raw is not None:
            feature_slice, dim, _ = raw
            mask = rng.random(dim) < drop_prob
            view = seq[:, feature_slice]
            view[:, mask] = 0.0

    def _augment(self, seq: np.ndarray, hand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng()
        seq = seq.copy()
        if rng.random() < float(self.aug_cfg.get("time_shift_prob", 0.0)):
            max_shift = int(self.aug_cfg.get("time_shift_max_frames", 0))
            if max_shift > 0:
                seq = self._edge_shift(seq, int(rng.integers(-max_shift, max_shift + 1)))
        self._augment_group_amplitude(seq, rng)
        self._augment_magnitude_warp(seq, rng)
        self._augment_lowfreq_drift(seq, rng)
        noise_std = float(self.aug_cfg.get("noise_std", 0.0))
        if noise_std > 0:
            seq = seq + rng.normal(0.0, noise_std, size=seq.shape).astype(np.float32)
        self._augment_time_mask(seq, rng)
        self._augment_frequency_mask(seq, rng)
        self._augment_channel_dropout(seq, rng)
        hand_noise = float(self.aug_cfg.get("hand_noise_std", 0.0))
        if hand_noise > 0:
            hand = hand + rng.normal(0.0, hand_noise, size=hand.shape).astype(np.float32)
        return seq.astype(np.float32), hand.astype(np.float32)


def collate_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, Any]:
    return {
        "seq": torch.stack([item["seq"] for item in batch]),
        "hand": torch.stack([item["hand"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "path": [str(item["path"]) for item in batch],
    }
