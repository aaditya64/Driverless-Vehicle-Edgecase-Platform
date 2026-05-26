"""Train reliability-constrained residual fusion using 40-second context features."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .common import read_split_csv, stable_video_id, write_json


EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(np.float64), EPS, 1.0 - EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    p = _clip(p)
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


def _best_threshold(y: np.ndarray, p: np.ndarray, metric: str) -> tuple[float, dict[str, Any]]:
    best_thr = 0.5
    best_item: dict[str, Any] | None = None
    for thr in np.linspace(0.2, 0.8, 241):
        item = _metrics(y, p, float(thr))
        if best_item is None or item[metric] > best_item[metric]:
            best_item = item
            best_thr = float(thr)
    assert best_item is not None
    return best_thr, best_item


def _selective_by_margin(
    y: np.ndarray,
    p: np.ndarray,
    class_threshold: float,
    target_accuracy: float,
    min_coverage: float,
) -> dict[str, Any] | None:
    p = _clip(p)
    pred = (p >= class_threshold).astype(np.int64)
    margin = np.abs(p - class_threshold)
    best: dict[str, Any] | None = None
    for margin_threshold in np.linspace(0.0, 0.5, 501):
        mask = margin >= float(margin_threshold)
        coverage = float(mask.mean())
        if coverage < min_coverage or not mask.any():
            continue
        acc = float(accuracy_score(y[mask], pred[mask]))
        if acc < target_accuracy:
            continue
        item = {
            "margin_threshold": float(margin_threshold),
            "coverage": coverage,
            "n": int(mask.sum()),
            "accuracy": acc,
            "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred[mask])),
            "macro_f1": float(f1_score(y[mask], pred[mask], average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y[mask], pred[mask], labels=[0, 1]).tolist(),
        }
        if best is None or item["coverage"] > best["coverage"]:
            best = item
    return best


def _apply_selective_by_margin(y: np.ndarray, p: np.ndarray, class_threshold: float, margin_threshold: float) -> dict[str, Any]:
    p = _clip(p)
    pred = (p >= class_threshold).astype(np.int64)
    mask = np.abs(p - class_threshold) >= margin_threshold
    if not mask.any():
        return {
            "coverage": 0.0,
            "n": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    return {
        "coverage": float(mask.mean()),
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(y[mask], pred[mask])),
        "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred[mask])),
        "macro_f1": float(f1_score(y[mask], pred[mask], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y[mask], pred[mask], labels=[0, 1]).tolist(),
    }


def _npz_feature_path(feature_dir: str | Path, path: str) -> Path:
    return Path(feature_dir) / f"{stable_video_id(path)}.npz"


def _read_long_map(path: str | Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[str(row["path"])] = row
    return out


def _load_summary(feature_dir: str | Path, path: str) -> np.ndarray:
    data = np.load(_npz_feature_path(feature_dir, path), allow_pickle=True)
    return np.nan_to_num(data["summary"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _load_long_dino(feature_dir: str | Path, path: str, mode: str) -> np.ndarray:
    data = np.load(_npz_feature_path(feature_dir, path), allow_pickle=True)
    if mode == "summary":
        x = data["summary"].astype(np.float32)
    else:
        emb = data["window_embeddings"].astype(np.float32)
        names = [str(x) for x in data["window_names"]]
        index = {name: i for i, name in enumerate(names)}
        means = emb.mean(axis=1)
        stds = emb.std(axis=1)
        parts = [means.reshape(-1), stds.reshape(-1)]
        for a, b in [
            ("event", "pre"),
            ("early_post", "pre"),
            ("late_post", "pre"),
            ("late_post", "event"),
            ("early_post", "event"),
        ]:
            if a in index and b in index:
                parts.append(means[index[a]] - means[index[b]])
                parts.append(stds[index[a]] - stds[index[b]])
        x = np.concatenate(parts, axis=0).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _load_badas_context(feature_dir: str | Path, clip_path: str) -> np.ndarray:
    data = np.load(_npz_feature_path(feature_dir, clip_path), allow_pickle=True)
    risk = np.nan_to_num(data["risk_scores"].astype(np.float32), nan=0.0)
    logits = np.nan_to_num(data["logits"].astype(np.float32), nan=0.0)
    target_t = data["target_time_sec"].astype(np.float32)

    def stats(x: np.ndarray) -> list[float]:
        if x.size == 0:
            return [0.0] * 10
        d = np.diff(x) if x.size > 1 else np.zeros(1, dtype=np.float32)
        return [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(np.percentile(x, 25)),
            float(np.percentile(x, 50)),
            float(np.percentile(x, 75)),
            float(x[-1] - x[0]),
            float(np.max(np.abs(d))),
            float(np.mean(np.abs(d))),
        ]

    parts: list[float] = []
    parts.extend(stats(risk))
    if risk.size:
        peak_idx = int(np.argmax(risk))
        parts.extend(
            [
                float(target_t[peak_idx]) if target_t.size else 0.0,
                float(1.0 - np.prod(1.0 - np.clip(risk, 0.0, 1.0))),
                float(np.mean(risk[:3])) if risk.size >= 3 else float(np.mean(risk)),
                float(np.mean(risk[3:6])) if risk.size >= 6 else float(np.mean(risk)),
                float(np.mean(risk[6:])) if risk.size >= 7 else float(np.mean(risk)),
            ]
        )
    else:
        parts.extend([0.0] * 5)
    parts.extend(logits.mean(axis=0).tolist())
    parts.extend(logits.std(axis=0).tolist())
    parts.extend((logits[-1] - logits[0]).tolist() if logits.shape[0] > 1 else [0.0] * logits.shape[1])
    return np.asarray(parts, dtype=np.float32)


def _metadata_features(rows: list[dict[str, Any]], long_maps: dict[str, dict[str, str]], train_levels: dict[str, list[str]] | None = None) -> tuple[np.ndarray, dict[str, list[str]]]:
    cats = ["light_conditions", "weather", "scene"]
    if train_levels is None:
        train_levels = {c: sorted({long_maps[row["path"]].get(c, "") for row in rows}) for c in cats}
    out = []
    for row in rows:
        meta = long_maps[row["path"]]
        vals = [
            float(meta.get("video_duration") or 0.0),
            float(meta.get("time_of_event") or 0.0),
            float(meta.get("time_of_alert") or 0.0),
            float(meta.get("event_center_time") or 0.0),
            float(meta.get("clip_start_time") or 0.0),
            float(meta.get("clip_end_time") or 0.0),
        ]
        vals.append(vals[1] - vals[2])
        for cat in cats:
            value = meta.get(cat, "")
            vals.extend([1.0 if value == level else 0.0 for level in train_levels[cat]])
        out.append(vals)
    return np.asarray(out, dtype=np.float32), train_levels


def _build_context_matrix(
    rows: list[dict[str, Any]],
    long_csv: str,
    *,
    long_context_dir: str,
    long_dino_dir: str,
    badas_dir: str,
    include_long_context: bool,
    include_long_dino: bool,
    include_badas: bool,
    include_meta: bool,
    dino_mode: str,
    train_levels: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, dict[str, list[str]] | None]:
    long_map = _read_long_map(long_csv)
    parts: list[np.ndarray] = []
    for row in rows:
        clip_path = row["path"]
        source_path = long_map[clip_path]["source_path"]
        item_parts = []
        if include_long_context:
            item_parts.append(_load_summary(long_context_dir, source_path))
        if include_long_dino:
            item_parts.append(_load_long_dino(long_dino_dir, source_path, dino_mode))
        if include_badas:
            item_parts.append(_load_badas_context(badas_dir, clip_path))
        parts.append(np.concatenate(item_parts).astype(np.float32) if item_parts else np.zeros(1, dtype=np.float32))
    x = np.stack(parts)
    levels = train_levels
    if include_meta:
        meta, levels = _metadata_features(rows, long_map, train_levels)
        x = np.concatenate([x, meta], axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), levels


def _load_event_probs(run_dir: str | Path, primary: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    z = np.load(Path(run_dir) / "strong_fusion_probabilities.npz", allow_pickle=True)
    y_train = z["y_train"].astype(np.int64)
    y_test = z["y_test"].astype(np.int64)
    oof_base = _clip(z["oof_base"])
    test_base = _clip(z["test_base"])
    if f"oof_{primary}" in z and f"test_{primary}" in z:
        p_train = _clip(z[f"oof_{primary}"])
        p_test = _clip(z[f"test_{primary}"])
    else:
        p_train = _clip(oof_base.mean(axis=1))
        p_test = _clip(test_base.mean(axis=1))
    names = [str(x) for x in z["spec_names"]]
    return p_train, p_test, oof_base, test_base, y_train, y_test, names


def _event_features(p_primary: np.ndarray, base: np.ndarray) -> np.ndarray:
    p_primary = _clip(p_primary)
    base = _clip(base)
    stats = np.stack(
        [
            base.mean(axis=1),
            np.median(base, axis=1),
            base.std(axis=1),
            base.min(axis=1),
            base.max(axis=1),
            (base >= 0.5).mean(axis=1),
            (np.abs(base - 0.5) * 2.0).mean(axis=1),
            np.abs(p_primary - 0.5) * 2.0,
        ],
        axis=1,
    )
    return np.concatenate([_logit(p_primary)[:, None], p_primary[:, None], _logit(base), base, stats], axis=1)


def _models(seed: int, n_features: int, n_train: int) -> dict[str, Any]:
    pca64 = max(2, min(64, n_features, n_train - 1))
    pca128 = max(2, min(128, n_features, n_train - 1))
    pca220 = max(2, min(220, n_features, n_train - 1))
    return {
        "logreg_l2": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.25, max_iter=4000, class_weight="balanced", random_state=seed)),
        "pca64_logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=pca64, random_state=seed), LogisticRegression(C=0.45, max_iter=4000, class_weight="balanced", random_state=seed)),
        "pca128_svc": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=pca128, random_state=seed), SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
        "pca220_logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=pca220, random_state=seed), LogisticRegression(C=0.35, max_iter=4000, class_weight="balanced", random_state=seed)),
        "extra_trees": ExtraTreesClassifier(n_estimators=700, max_depth=6, min_samples_leaf=6, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=-1),
        "hgb": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=180, learning_rate=0.025, max_leaf_nodes=15, l2_regularization=0.6, random_state=seed)),
        "xgb_shallow": XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.025, subsample=0.9, colsample_bytree=0.65, min_child_weight=8, reg_alpha=0.5, reg_lambda=6.0, eval_metric="logloss", random_state=seed, n_jobs=4),
        "lgbm_shallow": make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=260,
                max_depth=3,
                num_leaves=7,
                learning_rate=0.025,
                subsample=0.9,
                colsample_bytree=0.65,
                min_child_samples=14,
                reg_alpha=0.5,
                reg_lambda=6.0,
                class_weight="balanced",
                random_state=seed,
                n_jobs=4,
                verbose=-1,
            ),
        ),
        "catboost_ordered": CatBoostClassifier(
            iterations=260,
            depth=3,
            learning_rate=0.025,
            l2_leaf_reg=8.0,
            loss_function="Logloss",
            eval_metric="Logloss",
            bootstrap_type="Bayesian",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        ),
    }


def _fit(model: Any, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> Any:
    if sample_weight is None:
        model.fit(x, y)
        return model
    if hasattr(model, "steps"):
        last_name = model.steps[-1][0]
        model.fit(x, y, **{f"{last_name}__sample_weight": sample_weight})
        return model
    try:
        model.fit(x, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(x, y)
    return model


def _predict(model: Any, x: np.ndarray) -> np.ndarray:
    return _clip(model.predict_proba(x)[:, 1])


def _oof_predict(
    x: np.ndarray,
    y: np.ndarray,
    model: Any,
    folds: int,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    fold_reports = []
    for fold, (tr, va) in enumerate(cv.split(x, y), start=1):
        m = clone(model)
        fold_weight = sample_weight[tr] if sample_weight is not None else None
        _fit(m, x[tr], y[tr], fold_weight)
        oof[va] = _predict(m, x[va])
        fold_reports.append({"fold": fold, "metrics": _metrics(y[va], oof[va])})
    return oof, fold_reports


def _hard_weights(
    y: np.ndarray,
    p_ref: np.ndarray,
    mode: str,
    *,
    alpha: float,
    gamma: float,
    max_weight: float,
) -> np.ndarray | None:
    if mode == "none":
        return None
    p_ref = _clip(p_ref)
    y = y.astype(np.int64)
    pt = np.where(y == 1, p_ref, 1.0 - p_ref)
    if mode == "focal":
        raw = np.power(1.0 - pt, gamma)
    elif mode == "margin":
        raw = 1.0 / (np.abs(p_ref - 0.5) + 0.05)
    elif mode == "error":
        raw = 1.0 + alpha * ((p_ref >= 0.5).astype(np.int64) != y).astype(np.float64)
        raw = raw / max(float(np.mean(raw)), EPS)
        return np.clip(raw, 0.1, max_weight)
    else:
        raise ValueError(mode)
    raw = raw / max(float(np.mean(raw)), EPS)
    weights = 1.0 + alpha * (raw - 1.0)
    weights = weights / max(float(np.mean(weights)), EPS)
    return np.clip(weights, 0.1, max_weight)


def _path_indices(rows: list[dict[str, Any]], subset_rows: list[dict[str, Any]]) -> np.ndarray:
    index = {row["path"]: i for i, row in enumerate(rows)}
    missing = [row["path"] for row in subset_rows if row["path"] not in index]
    if missing:
        raise ValueError(f"subset rows are not contained in train rows, first missing={missing[0]}")
    return np.asarray([index[row["path"]] for row in subset_rows], dtype=np.int64)


def _fuse_grid(y: np.ndarray, p_event: np.ndarray, p_context: np.ndarray, metric: str) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    z_event = _logit(p_event)
    z_context = _logit(p_context)
    for w in np.linspace(-0.5, 1.2, 69):
        p = _sigmoid(z_event + float(w) * z_context)
        thr, item = _best_threshold(y, p, metric)
        candidate = {"weight": float(w), "threshold": float(thr), "metrics": item, "prob": p}
        if best is None or item[metric] > best["metrics"][metric]:
            best = candidate
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="splits/processed_744/train.csv")
    parser.add_argument("--train-inner-csv", default="splits/processed_744/train_inner.csv")
    parser.add_argument("--val-csv", default="splits/processed_744/val.csv")
    parser.add_argument("--test-csv", default="splits/processed_744/test.csv")
    parser.add_argument("--long-train-csv", default="splits/processed_744_long/train.csv")
    parser.add_argument("--long-test-csv", default="splits/processed_744_long/test.csv")
    parser.add_argument("--event-run-dir", default="outputs/processed_744_strong_fusion_boosted")
    parser.add_argument("--event-primary", default="blend_logreg_xgb")
    parser.add_argument("--long-context-dir", default="outputs/processed_744/long_context_diff_1fps_yolo")
    parser.add_argument("--long-dino-dir", default="outputs/processed_744/long_dino_vits14_5w8f")
    parser.add_argument("--badas-dir", default="outputs/processed_744/badas_window_features")
    parser.add_argument("--out-dir", default="outputs/processed_744_long_context_residual_fusion")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--threshold-metric", choices=["accuracy", "macro_f1", "balanced_accuracy"], default="accuracy")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--selection-mode", choices=["oof", "holdout"], default="oof")
    parser.add_argument("--target-accuracy", type=float, default=0.98)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--hard-weight-mode", choices=["none", "focal", "margin", "error"], default="none")
    parser.add_argument("--hard-weight-alpha", type=float, default=1.0)
    parser.add_argument("--hard-weight-gamma", type=float, default=2.0)
    parser.add_argument("--hard-weight-max", type=float, default=4.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_split_csv(args.train_csv)
    train_inner_rows = read_split_csv(args.train_inner_csv)
    val_rows = read_split_csv(args.val_csv)
    test_rows = read_split_csv(args.test_csv)
    p_event_train, p_event_test, base_train, base_test, y_train, y_test, spec_names = _load_event_probs(args.event_run_dir, args.event_primary)
    if not np.array_equal(y_train, np.asarray([int(r["label"]) for r in train_rows])) or not np.array_equal(y_test, np.asarray([int(r["label"]) for r in test_rows])):
        raise ValueError("event probability labels do not match split CSV order")

    variants = {
        "context_behavior": dict(include_long_context=True, include_long_dino=False, include_badas=False, include_meta=True, dino_mode="compact"),
        "context_behavior_badas": dict(include_long_context=True, include_long_dino=False, include_badas=True, include_meta=True, dino_mode="compact"),
        "dino_compact": dict(include_long_context=False, include_long_dino=True, include_badas=False, include_meta=True, dino_mode="compact"),
        "behavior_dino_compact": dict(include_long_context=True, include_long_dino=True, include_badas=False, include_meta=True, dino_mode="compact"),
        "behavior_dino_badas_compact": dict(include_long_context=True, include_long_dino=True, include_badas=True, include_meta=True, dino_mode="compact"),
    }
    if args.variants:
        unknown = sorted(set(args.variants) - set(variants))
        if unknown:
            raise ValueError(f"Unknown variants: {unknown}")
        variants = {name: variants[name] for name in args.variants}

    event_train = _event_features(p_event_train, base_train)
    event_test = _event_features(p_event_test, base_test)
    report: dict[str, Any] = {
        "event_run_dir": str(args.event_run_dir),
        "event_primary": args.event_primary,
        "selection_mode": args.selection_mode,
        "hard_weight": {
            "mode": args.hard_weight_mode,
            "alpha": args.hard_weight_alpha,
            "gamma": args.hard_weight_gamma,
            "max": args.hard_weight_max,
        },
        "event_default_train": _metrics(y_train, p_event_train, 0.5),
        "event_default_test": _metrics(y_test, p_event_test, 0.5),
        "variants": {},
        "spec_names": spec_names,
    }
    event_thr, event_oof_best = _best_threshold(y_train, p_event_train, args.threshold_metric)
    report["event_oof_threshold"] = event_oof_best
    report["event_test_at_oof_threshold"] = _metrics(y_test, p_event_test, event_thr)
    inner_idx = _path_indices(train_rows, train_inner_rows)
    val_idx = _path_indices(train_rows, val_rows)
    event_val_thr, event_val_best = _best_threshold(y_train[val_idx], p_event_train[val_idx], args.threshold_metric)
    report["event_val_threshold"] = event_val_best
    report["event_test_at_val_threshold"] = _metrics(y_test, p_event_test, event_val_thr)
    sample_weight = _hard_weights(
        y_train,
        p_event_train,
        args.hard_weight_mode,
        alpha=args.hard_weight_alpha,
        gamma=args.hard_weight_gamma,
        max_weight=args.hard_weight_max,
    )
    if sample_weight is not None:
        report["hard_weight"]["stats"] = {
            "mean": float(np.mean(sample_weight)),
            "std": float(np.std(sample_weight)),
            "min": float(np.min(sample_weight)),
            "max": float(np.max(sample_weight)),
            "p50": float(np.percentile(sample_weight, 50)),
            "p90": float(np.percentile(sample_weight, 90)),
            "p99": float(np.percentile(sample_weight, 99)),
        }

    best: dict[str, Any] | None = None
    for variant_name, options in variants.items():
        print(f"variant={variant_name}", flush=True)
        x_ctx_train, levels = _build_context_matrix(
            train_rows,
            args.long_train_csv,
            long_context_dir=args.long_context_dir,
            long_dino_dir=args.long_dino_dir,
            badas_dir=args.badas_dir,
            **options,
        )
        x_ctx_test, _ = _build_context_matrix(
            test_rows,
            args.long_test_csv,
            long_context_dir=args.long_context_dir,
            long_dino_dir=args.long_dino_dir,
            badas_dir=args.badas_dir,
            train_levels=levels,
            **options,
        )
        x_train = np.concatenate([event_train, x_ctx_train], axis=1)
        x_test = np.concatenate([event_test, x_ctx_test], axis=1)
        variant_report = {
            "context_dim": int(x_ctx_train.shape[1]),
            "total_dim": int(x_train.shape[1]),
            "models": {},
        }
        models = _models(args.seed, x_train.shape[1], x_train.shape[0])
        if args.models:
            unknown_models = sorted(set(args.models) - set(models))
            if unknown_models:
                raise ValueError(f"Unknown models: {unknown_models}")
            models = {name: models[name] for name in args.models}
        for model_name, model in models.items():
            print(f"  model={model_name}", flush=True)
            oof_context, folds = _oof_predict(x_train, y_train, model, args.folds, args.seed, sample_weight)
            val_model = clone(model)
            val_weight = sample_weight[inner_idx] if sample_weight is not None else None
            _fit(val_model, x_train[inner_idx], y_train[inner_idx], val_weight)
            val_context = _predict(val_model, x_train[val_idx])
            val_fuse = _fuse_grid(y_train[val_idx], p_event_train[val_idx], val_context, args.threshold_metric)
            oof_fuse = _fuse_grid(y_train, p_event_train, oof_context, args.threshold_metric)
            fuse = val_fuse if args.selection_mode == "holdout" else oof_fuse
            full_model = clone(model)
            _fit(full_model, x_train, y_train, sample_weight)
            test_context = _predict(full_model, x_test)
            test_fused = _sigmoid(_logit(p_event_test) + float(fuse["weight"]) * _logit(test_context))
            selected_y = y_train[val_idx] if args.selection_mode == "holdout" else y_train
            selected_selective = _selective_by_margin(
                selected_y,
                fuse["prob"],
                float(fuse["threshold"]),
                args.target_accuracy,
                args.min_coverage,
            )
            test_selective = None
            if selected_selective is not None:
                test_selective = _apply_selective_by_margin(y_test, test_fused, float(fuse["threshold"]), float(selected_selective["margin_threshold"]))
            item = {
                "context_oof_default": _metrics(y_train, oof_context, 0.5),
                "context_val_default": _metrics(y_train[val_idx], val_context, 0.5),
                "context_test_default": _metrics(y_test, test_context, 0.5),
                "fused_oof_best": {k: v for k, v in oof_fuse.items() if k != "prob"},
                "fused_val_best": {k: v for k, v in val_fuse.items() if k != "prob"},
                "selected_fuse": {k: v for k, v in fuse.items() if k != "prob"},
                "fused_test_at_oof": _metrics(y_test, test_fused, float(fuse["threshold"])),
                "fused_test_default": _metrics(y_test, test_fused, 0.5),
                "selective_selected": selected_selective,
                "selective_test": test_selective,
                "folds": folds,
            }
            variant_report["models"][model_name] = item
            score_key = "fused_val_best" if args.selection_mode == "holdout" else "fused_oof_best"
            score = item[score_key]["metrics"][args.threshold_metric]
            print(
                f"    oof_fused_acc={item['fused_oof_best']['metrics']['accuracy']:.4f} "
                f"val_fused_acc={item['fused_val_best']['metrics']['accuracy']:.4f} "
                f"test_acc={item['fused_test_at_oof']['accuracy']:.4f} "
                f"test_auc={item['fused_test_at_oof']['auroc']:.4f} "
                f"w={item['selected_fuse']['weight']:.3f}",
                flush=True,
            )
            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "variant": variant_name,
                    "model": model_name,
                    "options": deepcopy(options),
                    "fused_test_at_oof": item["fused_test_at_oof"],
                    "context_test_default": item["context_test_default"],
                    "fused_oof_best": item["fused_oof_best"],
                    "fused_val_best": item["fused_val_best"],
                    "selected_fuse": item["selected_fuse"],
                    "selective_selected": item["selective_selected"],
                    "selective_test": item["selective_test"],
                    "model_object": full_model,
                    "metadata_levels": levels,
                }
        report["variants"][variant_name] = variant_report

    assert best is not None
    summary_best = {k: v for k, v in best.items() if k not in {"model_object", "metadata_levels"}}
    report["best_by_oof"] = summary_best
    write_json(out_dir / "long_context_residual_summary.json", report)
    joblib.dump(
        {
            "model": best["model_object"],
            "variant": best["variant"],
            "model_name": best["model"],
            "options": best["options"],
            "metadata_levels": best["metadata_levels"],
            "event_run_dir": args.event_run_dir,
            "event_primary": args.event_primary,
            "config": vars(args),
        },
        out_dir / "long_context_residual_model.joblib",
    )
    print("BEST", summary_best, flush=True)


if __name__ == "__main__":
    main()
