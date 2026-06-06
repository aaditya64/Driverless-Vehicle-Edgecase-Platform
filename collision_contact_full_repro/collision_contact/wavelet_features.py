from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pywt

from .common import pad_or_trim_1d, pad_or_trim_2d, robust_zscore


def _cwt_frequencies(cfg: dict[str, Any], fps: float) -> tuple[np.ndarray, np.ndarray]:
    cwt_cfg = cfg["features"]["cwt"]
    wavelet = cwt_cfg["wavelet"]
    freqs = np.geomspace(float(cwt_cfg["freq_min"]), float(cwt_cfg["freq_max"]), int(cwt_cfg["n_freqs"]))
    central = pywt.central_frequency(wavelet)
    scales = central * float(fps) / freqs
    return freqs.astype(np.float32), scales.astype(np.float32)


def _safe_channel(df: pd.DataFrame, channel: str) -> np.ndarray:
    if channel not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    arr = df[channel].to_numpy(dtype=np.float32).copy()
    arr[~np.isfinite(arr)] = 0.0
    return arr


def build_sequence_features(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    target_frames = int(cfg["video"]["target_frames"])
    fps = float(cfg["video"]["target_fps"])
    raw_channels = list(cfg["features"]["raw_channels"])
    wavelet_channels = list(cfg["features"]["wavelet_channels"])

    raw_abs = np.stack([_safe_channel(df, ch) for ch in raw_channels], axis=1)
    raw_abs = pad_or_trim_2d(raw_abs, target_frames)
    raw = robust_zscore(raw_abs).astype(np.float32)

    freqs, scales = _cwt_frequencies(cfg, fps)
    cwt_abs_parts: list[np.ndarray] = []
    for channel in wavelet_channels:
        signal = pad_or_trim_1d(_safe_channel(df, channel), target_frames)
        signal = robust_zscore(signal[:, None])[:, 0]
        coef, _ = pywt.cwt(signal, scales, cfg["features"]["cwt"]["wavelet"], sampling_period=1.0 / fps)
        energy = np.log1p(np.abs(coef) ** 2).astype(np.float32)
        cwt_abs_parts.append(energy.T)
    cwt_abs = np.concatenate(cwt_abs_parts, axis=1).astype(np.float32)
    cwt = robust_zscore(cwt_abs).astype(np.float32)

    swt_abs = _build_swt_features(df, cfg, wavelet_channels, target_frames, normalize=False)
    swt = _build_swt_features(df, cfg, wavelet_channels, target_frames, normalize=True)
    handcrafted_local = build_handcrafted_local(raw=raw, cwt=cwt, swt=swt, freqs=freqs, cfg=cfg)
    handcrafted = build_handcrafted(
        raw_abs=raw_abs,
        raw=raw,
        cwt_abs=cwt_abs,
        cwt=cwt,
        swt_abs=swt_abs,
        swt=swt,
        freqs=freqs,
        cfg=cfg,
    )
    time = pad_or_trim_1d(df["time"].to_numpy(dtype=np.float32), target_frames)
    return {
        "raw": raw,
        "raw_abs": raw_abs.astype(np.float32),
        "cwt": cwt,
        "cwt_abs": cwt_abs.astype(np.float32),
        "swt": swt,
        "swt_abs": swt_abs.astype(np.float32),
        "handcrafted": handcrafted,
        "handcrafted_local": handcrafted_local,
        "time": time,
        "freqs": freqs.astype(np.float32),
        "raw_channels": np.array(raw_channels),
        "wavelet_channels": np.array(wavelet_channels),
    }


def _build_swt_features(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    channels: list[str],
    target_frames: int,
    normalize: bool,
) -> np.ndarray:
    swt_cfg = cfg["features"]["swt"]
    wavelet = swt_cfg["wavelet"]
    levels = int(swt_cfg["levels"])
    block = 2**levels
    padded_len = int(np.ceil(target_frames / block) * block)
    parts: list[np.ndarray] = []
    for channel in channels:
        signal = pad_or_trim_1d(_safe_channel(df, channel), target_frames)
        if normalize:
            signal = robust_zscore(signal[:, None])[:, 0]
        if padded_len > target_frames:
            signal_pad = np.pad(signal, (0, padded_len - target_frames), mode="edge")
        else:
            signal_pad = signal
        coeffs = pywt.swt(signal_pad, wavelet=wavelet, level=levels, trim_approx=False)
        detail_parts = [cD[:target_frames] for _, cD in coeffs]
        detail_parts.append(coeffs[-1][0][:target_frames])
        parts.append(np.stack(detail_parts, axis=1))
    swt = np.concatenate(parts, axis=1).astype(np.float32)
    if normalize:
        return robust_zscore(swt).astype(np.float32)
    return swt


def build_handcrafted(
    raw_abs: np.ndarray,
    raw: np.ndarray,
    cwt_abs: np.ndarray,
    cwt: np.ndarray,
    swt_abs: np.ndarray,
    swt: np.ndarray,
    freqs: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    feats: list[float] = []
    for matrix in [raw_abs, raw, swt_abs, swt]:
        feats.extend(np.nanmean(matrix, axis=0).tolist())
        feats.extend(np.nanstd(matrix, axis=0).tolist())
        feats.extend(np.nanmax(np.abs(matrix), axis=0).tolist())
        feats.extend(np.nanpercentile(np.abs(matrix), 95, axis=0).tolist())

    n_wavelet_channels = len(cfg["features"]["wavelet_channels"])
    n_freqs = len(freqs)
    for matrix in [cwt_abs, cwt]:
        cwt3 = matrix.reshape(matrix.shape[0], n_wavelet_channels, n_freqs)
        low = (freqs >= 0.5) & (freqs < 2.0)
        mid = (freqs >= 2.0) & (freqs < 6.0)
        high = (freqs >= 6.0) & (freqs <= 14.0)
        for mask in [low, mid, high]:
            band = cwt3[:, :, mask]
            band_energy_t = band.mean(axis=(1, 2))
            feats.extend(
                [
                    float(np.mean(band_energy_t)),
                    float(np.std(band_energy_t)),
                    float(np.max(band_energy_t)),
                    float(np.percentile(band_energy_t, 95)),
                ]
            )
        total_energy = cwt3.mean(axis=(1, 2))
        peak_idx = int(np.argmax(total_energy))
        feats.extend(
            [
                float(peak_idx / max(len(total_energy) - 1, 1)),
                float(total_energy[peak_idx]),
                float(np.mean(total_energy)),
                float(np.std(total_energy)),
            ]
        )
    arr = np.asarray(feats, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def build_handcrafted_local(raw: np.ndarray, cwt: np.ndarray, swt: np.ndarray, freqs: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    feats: list[float] = []
    for matrix in [raw, swt]:
        feats.extend(np.nanmean(matrix, axis=0).tolist())
        feats.extend(np.nanstd(matrix, axis=0).tolist())
        feats.extend(np.nanmax(np.abs(matrix), axis=0).tolist())
        feats.extend(np.nanpercentile(np.abs(matrix), 95, axis=0).tolist())

    n_wavelet_channels = len(cfg["features"]["wavelet_channels"])
    n_freqs = len(freqs)
    cwt3 = cwt.reshape(cwt.shape[0], n_wavelet_channels, n_freqs)
    low = (freqs >= 0.5) & (freqs < 2.0)
    mid = (freqs >= 2.0) & (freqs < 6.0)
    high = (freqs >= 6.0) & (freqs <= 14.0)
    for mask in [low, mid, high]:
        band = cwt3[:, :, mask]
        band_energy_t = band.mean(axis=(1, 2))
        feats.extend(
            [
                float(np.mean(band_energy_t)),
                float(np.std(band_energy_t)),
                float(np.max(band_energy_t)),
                float(np.percentile(band_energy_t, 95)),
            ]
        )
    total_energy = cwt3.mean(axis=(1, 2))
    peak_idx = int(np.argmax(total_energy))
    feats.extend(
        [
            float(peak_idx / max(len(total_energy) - 1, 1)),
            float(total_energy[peak_idx]),
            float(np.mean(total_energy)),
            float(np.std(total_energy)),
        ]
    )
    arr = np.asarray(feats, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr
