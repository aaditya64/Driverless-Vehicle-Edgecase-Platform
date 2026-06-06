"""Train a collision head from impulse, damping, wavelet, and object-energy physics."""

from __future__ import annotations

import argparse
import csv
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
from .train_impact_physics_head import build_impact_features

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


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


def _safe_stats(x: np.ndarray) -> list[float]:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return [0.0] * 10
    return [
        float(x.mean()),
        float(x.std()),
        float(x.min()),
        float(x.max()),
        float(np.percentile(x, 25)),
        float(np.percentile(x, 50)),
        float(np.percentile(x, 75)),
        float(np.percentile(x, 90)),
        float(np.percentile(x, 95)),
        float(np.percentile(x, 99)),
    ]


def _window(x: np.ndarray, times: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (times >= lo) & (times < hi)
    return x[mask] if np.any(mask) else x


def _stop_features(x: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return [0.0] * 8, [
            f"{prefix}_signed_peak",
            f"{prefix}_abs_integral",
            f"{prefix}_net_integral",
            f"{prefix}_stop_index",
            f"{prefix}_sign_flip_count",
            f"{prefix}_positive_integral",
            f"{prefix}_negative_integral",
            f"{prefix}_peak_to_mean_abs",
        ]
    abs_integral = float(np.sum(np.abs(x)))
    net_integral = float(np.sum(x))
    signs = np.sign(x)
    sign_flip_count = float(np.sum(signs[:-1] * signs[1:] < 0.0)) if x.size > 1 else 0.0
    peak = float(x[np.argmax(np.abs(x))])
    pos = float(np.sum(np.maximum(x, 0.0)))
    neg = float(np.sum(np.maximum(-x, 0.0)))
    features = [
        peak,
        abs_integral,
        net_integral,
        float(1.0 - abs(net_integral) / (abs_integral + EPS)),
        sign_flip_count,
        pos,
        neg,
        float(abs(peak) / (np.mean(np.abs(x)) + EPS)),
    ]
    names = [
        f"{prefix}_signed_peak",
        f"{prefix}_abs_integral",
        f"{prefix}_net_integral",
        f"{prefix}_stop_index",
        f"{prefix}_sign_flip_count",
        f"{prefix}_positive_integral",
        f"{prefix}_negative_integral",
        f"{prefix}_peak_to_mean_abs",
    ]
    return features, names


def _ringdown_features(signal: np.ndarray, times: np.ndarray, peak_t: float, prefix: str) -> tuple[list[float], list[str]]:
    signal = np.nan_to_num(signal.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pre = np.abs(_window(signal, times, peak_t - 2.0, peak_t - 0.45))
    imp = np.abs(_window(signal, times, peak_t - 0.25, peak_t + 0.35))
    post = np.abs(_window(signal, times, peak_t + 0.35, peak_t + 1.35))
    post_times = _window(times, times, peak_t + 0.35, peak_t + 1.35)
    pre_e = float(np.sum(pre * pre))
    imp_e = float(np.sum(imp * imp))
    post_e = float(np.sum(post * post))
    slope = 0.0
    if post.size >= 4 and post_times.size == post.size:
        y = np.log(post + np.percentile(post, 25) + EPS)
        x = post_times - float(post_times[0])
        slope = float(np.polyfit(x, y, 1)[0])
    features = [
        imp_e,
        post_e,
        float((imp_e + EPS) / (pre_e + EPS)),
        float((post_e + EPS) / (imp_e + EPS)),
        slope,
        float(np.mean(post > (np.median(pre) + 3.0 * np.std(pre) + EPS))) if pre.size else 0.0,
    ]
    names = [
        f"{prefix}_impact_energy",
        f"{prefix}_post_energy",
        f"{prefix}_impact_over_pre",
        f"{prefix}_post_over_impact",
        f"{prefix}_log_envelope_slope",
        f"{prefix}_post_width_over_pre3std",
    ]
    return features, names


def _read_object_metrics(paths: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for path in paths:
        if not path:
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                vals = np.asarray(
                    [
                        np.log1p(float(row["obj_score_ratio"])),
                        float(row["obj_score_z"]),
                        np.log1p(float(row["obj_energy_ratio"])),
                        np.log1p(float(row["obj_shift_ratio"])),
                        np.log1p(float(row["obj_diff_ratio"])),
                        float(row["obj_peak_time"]),
                        float(2.8 <= float(row["obj_peak_time"]) <= 6.6),
                    ],
                    dtype=np.float32,
                )
                out[str(row["path"])] = vals
    return out


def build_deep_features(feature_dir: str | Path, video_path: str | Path, object_metrics: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    base, base_names = build_impact_features(feature_dir, video_path)
    data = np.load(_feature_path(feature_dir, video_path), allow_pickle=True)
    raw = np.nan_to_num(data["raw"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    raw_abs = np.nan_to_num(data["raw_abs"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cwt_abs = np.nan_to_num(data["cwt_abs"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    times = np.nan_to_num(data["time"].astype(np.float32), nan=0.0)
    freqs = data["freqs"].astype(np.float32)
    raw_channels = [str(x) for x in data["raw_channels"]]
    wavelet_channels = [str(x) for x in data["wavelet_channels"]]
    raw_idx = {name: i for i, name in enumerate(raw_channels)}
    wav_idx = {name: i for i, name in enumerate(wavelet_channels)}

    n_channels = len(wavelet_channels)
    n_freqs = len(freqs)
    cwt3 = cwt_abs.reshape(cwt_abs.shape[0], n_channels, n_freqs)
    high = (freqs >= 6.0) & (freqs <= 14.0)
    mid = (freqs >= 2.0) & (freqs < 6.0)
    jerk_waves = [wav_idx[name] for name in ["jerk_x", "jerk_y", "jerk_theta_px"] if name in wav_idx]
    if jerk_waves:
        high_jerk = cwt3[:, jerk_waves][:, :, high].mean(axis=(1, 2))
        mid_jerk = cwt3[:, jerk_waves][:, :, mid].mean(axis=(1, 2))
    else:
        high_jerk = cwt3[:, :, high].mean(axis=(1, 2))
        mid_jerk = cwt3[:, :, mid].mean(axis=(1, 2))
    search = (times >= 2.0) & (times <= 8.0)
    score = high_jerk.copy()
    for name in ["jerk_energy", "shake_energy", "fit_error"]:
        if name in raw_idx:
            score = score + raw_abs[:, raw_idx[name]]
    peak_index = int(np.argmax(np.where(search, score, -np.inf))) if np.any(search) else int(np.argmax(score))
    peak_t = float(times[peak_index])

    features: list[float] = [peak_t, float(score[peak_index])]
    names: list[str] = ["deep_peak_time", "deep_peak_score"]

    vector_groups = {
        "ego_accel": ["ax", "ay", "atheta_px"],
        "ego_jerk": ["jerk_x", "jerk_y", "jerk_theta_px"],
        "ego_residual": ["x_res", "y_res", "theta_res_px"],
        "ego_velocity": ["vx", "vy", "vtheta_px"],
    }
    for group_name, channels in vector_groups.items():
        present = [raw_idx[name] for name in channels if name in raw_idx]
        if not present:
            continue
        mag = np.linalg.norm(raw[:, present], axis=1)
        pre = _window(mag, times, peak_t - 2.0, peak_t - 0.45)
        imp = _window(mag, times, peak_t - 0.25, peak_t + 0.35)
        post = _window(mag, times, peak_t + 0.35, peak_t + 1.35)
        features.extend(_safe_stats(pre))
        names.extend([f"{group_name}_pre_{s}" for s in ["mean", "std", "min", "max", "p25", "p50", "p75", "p90", "p95", "p99"]])
        features.extend(_safe_stats(imp))
        names.extend([f"{group_name}_impact_{s}" for s in ["mean", "std", "min", "max", "p25", "p50", "p75", "p90", "p95", "p99"]])
        features.extend(_safe_stats(post))
        names.extend([f"{group_name}_post_{s}" for s in ["mean", "std", "min", "max", "p25", "p50", "p75", "p90", "p95", "p99"]])
        features.extend(
            [
                float((np.max(imp) + EPS) / (np.percentile(pre, 95) + EPS)) if pre.size else 0.0,
                float((np.sum(post * post) + EPS) / (np.sum(imp * imp) + EPS)) if imp.size else 0.0,
                float(np.mean(imp > (np.median(pre) + 3.0 * np.std(pre) + EPS))) if pre.size else 0.0,
            ]
        )
        names.extend([f"{group_name}_impact_p95_ratio", f"{group_name}_ringdown_energy_ratio", f"{group_name}_impact_width"])

    for channel in ["dx", "dy", "theta_px", "ax", "ay", "atheta_px", "jerk_x", "jerk_y", "jerk_theta_px"]:
        if channel not in raw_idx:
            continue
        imp = _window(raw[:, raw_idx[channel]], times, peak_t - 0.25, peak_t + 0.35)
        block, block_names = _stop_features(imp, f"deep_{channel}")
        features.extend(block)
        names.extend(block_names)
        block, block_names = _ringdown_features(raw[:, raw_idx[channel]], times, peak_t, f"deep_{channel}")
        features.extend(block)
        names.extend(block_names)

    for wave_name, signal in [("high_jerk", high_jerk), ("mid_jerk", mid_jerk)]:
        pre = _window(signal, times, peak_t - 2.0, peak_t - 0.45)
        imp = _window(signal, times, peak_t - 0.25, peak_t + 0.35)
        post = _window(signal, times, peak_t + 0.35, peak_t + 1.35)
        features.extend(_safe_stats(imp))
        names.extend([f"deep_cwt_{wave_name}_impact_{s}" for s in ["mean", "std", "min", "max", "p25", "p50", "p75", "p90", "p95", "p99"]])
        features.extend(
            [
                float((np.max(imp) + EPS) / (np.percentile(pre, 95) + EPS)) if pre.size else 0.0,
                float((np.mean(post) + EPS) / (np.mean(imp) + EPS)) if imp.size else 0.0,
            ]
        )
        names.extend([f"deep_cwt_{wave_name}_impact_p95_ratio", f"deep_cwt_{wave_name}_post_over_impact"])

    obj = object_metrics.get(str(video_path), np.zeros((7,), dtype=np.float32))
    features.extend(obj.tolist())
    names.extend(
        [
            "object_log_score_ratio",
            "object_score_z",
            "object_log_energy_ratio",
            "object_log_shift_ratio",
            "object_log_diff_ratio",
            "object_peak_time",
            "object_peak_in_event_window",
        ]
    )

    arr = np.concatenate([base, np.asarray(features, dtype=np.float32)])
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), base_names + names


def _matrix(rows: list[dict[str, Any]], feature_dir: str | Path, object_metrics: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    names: list[str] | None = None
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    for row in rows:
        vec, vec_names = build_deep_features(feature_dir, row["path"], object_metrics)
        xs.append(vec)
        if names is None:
            names = vec_names
    assert names is not None
    return np.stack(xs).astype(np.float32), y, names


def _models(seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logreg": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.35)),
        "hgb": make_pipeline(
            SimpleImputer(),
            HistGradientBoostingClassifier(
                learning_rate=0.028,
                max_iter=700,
                max_leaf_nodes=15,
                l2_regularization=0.12,
                min_samples_leaf=10,
                random_state=seed,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(),
            ExtraTreesClassifier(
                n_estimators=1000,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    }
    if LGBMClassifier is not None:
        models["lgbm"] = make_pipeline(
            SimpleImputer(),
            LGBMClassifier(
                n_estimators=420,
                learning_rate=0.025,
                num_leaves=15,
                max_depth=4,
                min_child_samples=14,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.2,
                reg_lambda=4.0,
                class_weight="balanced",
                random_state=seed + 11,
                n_jobs=4,
                verbose=-1,
            ),
        )
    if XGBClassifier is not None:
        models["xgb"] = make_pipeline(
            SimpleImputer(),
            XGBClassifier(
                n_estimators=320,
                max_depth=2,
                learning_rate=0.025,
                subsample=0.85,
                colsample_bytree=0.75,
                min_child_weight=8,
                reg_alpha=0.6,
                reg_lambda=8.0,
                eval_metric="logloss",
                random_state=seed + 17,
                n_jobs=4,
            ),
        )
    return models


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
    parser.add_argument("--object-metrics-csv", action="append", default=["analysis/impact_diagnostics_20260522/object_physics_train_metrics.csv", "analysis/impact_diagnostics_20260522/object_physics_test_metrics.csv"])
    parser.add_argument("--out-dir", default="outputs/processed_744_deep_impulse_physics_head_20260523")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    object_metrics = _read_object_metrics(args.object_metrics_csv)
    train_rows = read_split_csv(args.train_csv)
    test_rows = read_split_csv(args.test_csv)
    x_train, y_train, feature_names = _matrix(train_rows, args.feature_dir, object_metrics)
    x_test, y_test, _ = _matrix(test_rows, args.feature_dir, object_metrics)

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
    joblib.dump({"model": best_model, "threshold": best_threshold, "feature_names": feature_names}, out_dir / "deep_impulse_physics_head.joblib")
    write_json(out_dir / "deep_impulse_physics_summary.json", report)
    with open(out_dir / "deep_impulse_physics_test_predictions.csv", "w", encoding="utf-8") as f:
        f.write("idx,path,true_label,prob_collision,pred_label\n")
        pred = (best_test_p >= best_threshold).astype(np.int64)
        for idx, (row, prob, label) in enumerate(zip(test_rows, best_test_p, pred)):
            f.write(f"{idx},{row['path']},{row['label']},{float(prob):.8f},{int(label)}\n")
    print(f"selected={best_name} threshold={best_threshold:.4f}")
    print(report["selected_test"])


if __name__ == "__main__":
    main()
