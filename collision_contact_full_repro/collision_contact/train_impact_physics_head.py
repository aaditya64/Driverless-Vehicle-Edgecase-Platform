"""Train an event-centered impact-physics head from cached motion/wavelet features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import read_split_csv, stable_video_id, write_json


EPS = 1e-6


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    p = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    pred = (p >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, np.c_[1.0 - p, p], labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def _best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    for threshold in np.linspace(0.2, 0.8, 241):
        item = _metrics(y, p, float(threshold))
        if best_metrics is None or item["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = item
            best_threshold = float(threshold)
    assert best_metrics is not None
    return best_threshold, best_metrics


def _feature_path(feature_dir: str | Path, video_path: str | Path) -> Path:
    return Path(feature_dir) / f"{stable_video_id(video_path)}.npz"


def _safe_ratio(a: float, b: float) -> float:
    return float((a + EPS) / (b + EPS))


def _basic_stats(x: np.ndarray) -> list[float]:
    if x.size == 0:
        return [0.0] * 9
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.percentile(x, 50)),
        float(np.percentile(x, 75)),
        float(np.percentile(x, 90)),
        float(np.percentile(x, 95)),
        float(np.percentile(x, 99)),
    ]


def _window_feature_block(signal: np.ndarray, times: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    signal = np.nan_to_num(signal.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    abs_signal = np.abs(signal)
    pre = abs_signal[(times >= 1.0) & (times < 4.2)]
    impact = abs_signal[(times >= 4.4) & (times < 5.8)]
    post = abs_signal[(times >= 5.8) & (times < 7.2)]
    if pre.size == 0:
        pre = abs_signal
    if impact.size == 0:
        impact = abs_signal
    if post.size == 0:
        post = abs_signal
    pre_med = float(np.median(pre))
    pre_mad = float(np.median(np.abs(pre - pre_med)))
    imp_max = float(np.max(impact))
    imp_mean = float(np.mean(impact))
    imp_p95 = float(np.percentile(impact, 95))
    post_mean = float(np.mean(post))
    threshold = pre_med + 3.0 * 1.4826 * pre_mad
    width = float(np.mean(impact > threshold))
    imp_peak_idx = int(np.argmax(impact))
    imp_peak_time = float(np.linspace(4.4, 5.8, len(impact), endpoint=False)[imp_peak_idx]) if impact.size else 0.0
    features = [
        imp_max,
        imp_mean,
        imp_p95,
        _safe_ratio(imp_max, pre_med),
        _safe_ratio(imp_p95, float(np.percentile(pre, 95))),
        float((imp_max - pre_med) / (1.4826 * pre_mad + 1e-3)),
        _safe_ratio(post_mean, imp_mean),
        width,
        imp_peak_time,
    ]
    names = [
        f"{prefix}_impact_max",
        f"{prefix}_impact_mean",
        f"{prefix}_impact_p95",
        f"{prefix}_impact_max_ratio",
        f"{prefix}_impact_p95_ratio",
        f"{prefix}_impact_z",
        f"{prefix}_post_over_impact",
        f"{prefix}_impact_width",
        f"{prefix}_impact_peak_time",
    ]
    for window_name, values in [("pre", pre), ("impact", impact), ("post", post)]:
        stats = _basic_stats(values)
        features.extend(stats)
        names.extend([f"{prefix}_{window_name}_{stat}" for stat in ["mean", "std", "min", "max", "p50", "p75", "p90", "p95", "p99"]])
    return features, names


def _signed_shape_features(signal: np.ndarray, times: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    signal = np.nan_to_num(signal.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    impact = signal[(times >= 4.4) & (times < 5.8)]
    post = signal[(times >= 5.8) & (times < 7.2)]
    if impact.size < 3:
        impact = signal
    if post.size < 3:
        post = signal
    peak = float(impact[np.argmax(np.abs(impact))])
    post_peak = float(post[np.argmax(np.abs(post))])
    sign_flip = float(np.any(np.sign(impact[:-1]) * np.sign(impact[1:]) < 0.0)) if impact.size > 1 else 0.0
    rebound = float(np.sign(peak) * np.sign(post_peak) < 0.0)
    net_change = float(np.sum(impact))
    abs_change = float(np.sum(np.abs(impact)))
    stop_index = float(1.0 - abs(net_change) / (abs_change + EPS))
    features = [peak, post_peak, sign_flip, rebound, net_change, abs_change, stop_index]
    names = [
        f"{prefix}_signed_peak",
        f"{prefix}_post_signed_peak",
        f"{prefix}_sign_flip",
        f"{prefix}_post_rebound",
        f"{prefix}_impact_net_change",
        f"{prefix}_impact_abs_change",
        f"{prefix}_impulse_stop_index",
    ]
    return features, names


def build_impact_features(feature_dir: str | Path, video_path: str | Path) -> tuple[np.ndarray, list[str]]:
    data = np.load(_feature_path(feature_dir, video_path), allow_pickle=True)
    raw = np.nan_to_num(data["raw"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    raw_abs = np.nan_to_num(data["raw_abs"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cwt_abs = np.nan_to_num(data["cwt_abs"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    swt_abs = np.nan_to_num(data["swt_abs"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    times = np.nan_to_num(data["time"].astype(np.float32), nan=0.0)
    freqs = data["freqs"].astype(np.float32)
    raw_channels = [str(x) for x in data["raw_channels"]]
    wavelet_channels = [str(x) for x in data["wavelet_channels"]]
    features: list[float] = []
    names: list[str] = []
    for idx, channel in enumerate(raw_channels):
        block, block_names = _window_feature_block(raw_abs[:, idx], times, f"raw_{channel}")
        features.extend(block)
        names.extend(block_names)
        if channel in {"dx", "dy", "theta_px", "x_res", "y_res", "theta_res_px", "ax", "ay", "atheta_px", "jerk_x", "jerk_y", "jerk_theta_px"}:
            block, block_names = _signed_shape_features(raw[:, idx], times, f"signed_{channel}")
            features.extend(block)
            names.extend(block_names)

    n_channels = len(wavelet_channels)
    n_freqs = len(freqs)
    cwt3 = cwt_abs.reshape(cwt_abs.shape[0], n_channels, n_freqs)
    bands = {
        "low": (freqs >= 0.5) & (freqs < 2.0),
        "mid": (freqs >= 2.0) & (freqs < 6.0),
        "high": (freqs >= 6.0) & (freqs <= 14.0),
        "all": np.ones_like(freqs, dtype=bool),
    }
    for channel_idx, channel in enumerate(wavelet_channels):
        for band_name, mask in bands.items():
            signal = cwt3[:, channel_idx, mask].mean(axis=1)
            block, block_names = _window_feature_block(signal, times, f"cwt_{channel}_{band_name}")
            features.extend(block)
            names.extend(block_names)
    for band_name, mask in bands.items():
        signal = cwt3[:, :, mask].mean(axis=(1, 2))
        block, block_names = _window_feature_block(signal, times, f"cwt_total_{band_name}")
        features.extend(block)
        names.extend(block_names)

    for idx in range(swt_abs.shape[1]):
        if idx >= 24:
            break
        block, block_names = _window_feature_block(swt_abs[:, idx], times, f"swt_{idx:02d}")
        features.extend(block)
        names.extend(block_names)
    arr = np.asarray(features, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, names


def _matrix(rows: list[dict[str, Any]], feature_dir: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    names: list[str] | None = None
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    for row in rows:
        vec, vec_names = build_impact_features(feature_dir, row["path"])
        xs.append(vec)
        if names is None:
            names = vec_names
    assert names is not None
    return np.stack(xs).astype(np.float32), y, names


def _models(seed: int) -> dict[str, Any]:
    return {
        "logreg": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)),
        "hgb": make_pipeline(
            SimpleImputer(),
            HistGradientBoostingClassifier(
                learning_rate=0.035,
                max_iter=450,
                max_leaf_nodes=15,
                l2_regularization=0.08,
                min_samples_leaf=12,
                random_state=seed,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(),
            ExtraTreesClassifier(
                n_estimators=900,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    }


def _oof_predict(model: Any, x: np.ndarray, y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf.split(x, y):
        fold_model = clone(model)
        fold_model.fit(x[train_idx], y[train_idx])
        oof[val_idx] = fold_model.predict_proba(x[val_idx])[:, 1]
    return oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="splits/processed_744/train.csv")
    parser.add_argument("--test-csv", default="splits/processed_744/test.csv")
    parser.add_argument("--feature-dir", default="outputs/processed_744/features")
    parser.add_argument("--out-dir", default="outputs/processed_744_impact_physics_head")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_split_csv(args.train_csv)
    test_rows = read_split_csv(args.test_csv)
    x_train, y_train, feature_names = _matrix(train_rows, args.feature_dir)
    x_test, y_test, _ = _matrix(test_rows, args.feature_dir)

    report: dict[str, Any] = {
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "feature_dir": args.feature_dir,
        "feature_dim": int(x_train.shape[1]),
        "models": {},
    }
    best_name = ""
    best_score = -1.0
    best_model: Any | None = None
    best_threshold = 0.5
    best_test_p: np.ndarray | None = None
    for name, model in _models(args.seed).items():
        oof_p = _oof_predict(model, x_train, y_train, folds=args.folds, seed=args.seed)
        threshold, oof_best = _best_threshold(y_train, oof_p)
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        test_p = fitted.predict_proba(x_test)[:, 1]
        item = {
            "oof_default": _metrics(y_train, oof_p, 0.5),
            "oof_best": oof_best,
            "test_default": _metrics(y_test, test_p, 0.5),
            "test_at_oof_threshold": _metrics(y_test, test_p, threshold),
        }
        report["models"][name] = item
        score = item["oof_best"]["macro_f1"]
        if score > best_score:
            best_score = score
            best_name = name
            best_model = fitted
            best_threshold = threshold
            best_test_p = test_p

    assert best_model is not None and best_test_p is not None
    report["selected_model"] = best_name
    report["selected_threshold"] = float(best_threshold)
    report["selected_test"] = _metrics(y_test, best_test_p, best_threshold)
    joblib.dump({"model": best_model, "threshold": best_threshold, "feature_names": feature_names}, out_dir / "impact_physics_head.joblib")
    write_json(out_dir / "impact_physics_summary.json", report)
    with open(out_dir / "impact_physics_test_predictions.csv", "w", encoding="utf-8") as f:
        f.write("idx,path,true_label,prob_collision,pred_label\n")
        pred = (best_test_p >= best_threshold).astype(np.int64)
        for idx, (row, prob, label) in enumerate(zip(test_rows, best_test_p, pred)):
            f.write(f"{idx},{row['path']},{row['label']},{float(prob):.8f},{int(label)}\n")
    print(f"selected={best_name} threshold={best_threshold:.4f}")
    print(report["selected_test"])


if __name__ == "__main__":
    main()
