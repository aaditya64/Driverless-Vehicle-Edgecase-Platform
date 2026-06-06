"""Train constrained rescue fusion from object-CoTracker dynamics features."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .common import read_split_csv, stable_video_id, write_json

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


EPS = 1e-6
warnings.filterwarnings("ignore", message="Features .* are constant.")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_prob(p)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)))


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return (np.sign(x) * np.log1p(np.abs(x))).astype(np.float32)


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    p = _clip_prob(p)
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


def _select_threshold(y: np.ndarray, p: np.ndarray, metric: str) -> tuple[float, dict[str, Any]]:
    thresholds = np.linspace(0.15, 0.85, 281)
    pred = _clip_prob(p)[:, None] >= thresholds[None, :]
    yt = y.astype(bool)[:, None]
    tp = np.sum(pred & yt, axis=0).astype(np.float64)
    tn = np.sum((~pred) & (~yt), axis=0).astype(np.float64)
    fp = np.sum(pred & (~yt), axis=0).astype(np.float64)
    fn = np.sum((~pred) & yt, axis=0).astype(np.float64)
    acc = (tp + tn) / max(len(y), 1)
    tpr = tp / np.maximum(tp + fn, EPS)
    tnr = tn / np.maximum(tn + fp, EPS)
    bal = 0.5 * (tpr + tnr)
    f1_pos = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, EPS)
    f1_neg = 2.0 * tn / np.maximum(2.0 * tn + fp + fn, EPS)
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    scores = {"accuracy": acc, "balanced_accuracy": bal, "macro_f1": macro_f1}[metric]
    best_idx = int(np.argmax(scores))
    threshold = float(thresholds[best_idx])
    return threshold, _metrics(y, p, threshold)


def _feature_path(feature_dir: str | Path, video_path: str | Path) -> Path:
    return Path(feature_dir) / f"{stable_video_id(video_path)}.npz"


def _load_matrix(rows: list[dict[str, Any]], feature_dir: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    names: list[str] | None = None
    for row in rows:
        data = np.load(_feature_path(feature_dir, row["path"]), allow_pickle=True)
        x = np.nan_to_num(data["summary"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        xs.append(x)
        if names is None:
            names = [str(v) for v in data["summary_names"]]
    assert names is not None
    return np.stack(xs).astype(np.float32), y, names


def _load_main_prob(path: str | Path, expert_name: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    data = np.load(path, allow_pickle=True)
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    y_test = np.asarray(data["y_test"], dtype=np.int64)
    names = [str(x) for x in data["spec_names"]]
    idx = names.index(expert_name) if expert_name else 0
    return _clip_prob(data["oof_base"][:, idx]), _clip_prob(data["test_base"][:, idx]), y_train, y_test, names[idx]


def _models(seed: int, dim: int) -> dict[str, Any]:
    k96 = min(96, dim)
    k160 = min(160, dim)
    models: dict[str, Any] = {
        "logreg_k96": make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            StandardScaler(),
            SelectKBest(f_classif, k=k96),
            LogisticRegression(C=0.25, max_iter=5000, class_weight="balanced", random_state=seed),
        ),
        "logreg_k160": make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            StandardScaler(),
            SelectKBest(f_classif, k=k160),
            LogisticRegression(C=0.12, max_iter=5000, class_weight="balanced", random_state=seed + 1),
        ),
        "hgb_k96": make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            SelectKBest(f_classif, k=k96),
            HistGradientBoostingClassifier(max_iter=220, learning_rate=0.03, max_leaf_nodes=9, min_samples_leaf=16, l2_regularization=0.8, random_state=seed + 2),
        ),
        "hgb_k160": make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            SelectKBest(f_classif, k=k160),
            HistGradientBoostingClassifier(max_iter=260, learning_rate=0.025, max_leaf_nodes=11, min_samples_leaf=14, l2_regularization=1.2, random_state=seed + 3),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            ExtraTreesClassifier(n_estimators=700, max_depth=5, min_samples_leaf=6, max_features="sqrt", class_weight="balanced", random_state=seed + 4, n_jobs=-1),
        ),
    }
    if XGBClassifier is not None:
        models["xgb_k160"] = make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            SelectKBest(f_classif, k=k160),
            XGBClassifier(
                n_estimators=260,
                max_depth=2,
                learning_rate=0.025,
                subsample=0.9,
                colsample_bytree=0.75,
                min_child_weight=8,
                reg_alpha=0.5,
                reg_lambda=8.0,
                eval_metric="logloss",
                random_state=seed + 5,
                n_jobs=4,
            ),
        )
    if LGBMClassifier is not None:
        models["lgbm_k160"] = make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            SelectKBest(f_classif, k=k160),
            LGBMClassifier(
                n_estimators=260,
                num_leaves=7,
                learning_rate=0.025,
                subsample=0.9,
                colsample_bytree=0.75,
                min_child_samples=16,
                reg_alpha=0.3,
                reg_lambda=6.0,
                class_weight="balanced",
                objective="binary",
                random_state=seed + 6,
                n_jobs=4,
                verbose=-1,
            ),
        )
    return models


def _predict(model: Any, x: np.ndarray) -> np.ndarray:
    return _clip_prob(model.predict_proba(x)[:, 1])


def _fit_model(model: Any, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
    if sample_weight is None:
        model.fit(x, y)
        return
    if hasattr(model, "steps"):
        step_name = model.steps[-1][0]
        model.fit(x, y, **{f"{step_name}__sample_weight": sample_weight})
    else:
        model.fit(x, y, sample_weight=sample_weight)


def _hard_sample_weight(y: np.ndarray, p0: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        return np.ones(len(y), dtype=np.float32)
    threshold, _ = _select_threshold(y, p0, "macro_f1")
    pred = (p0 >= threshold).astype(np.int64)
    margin = np.abs(_logit(p0) - _logit(np.asarray([threshold]))[0])
    uncertain = np.exp(-np.minimum(margin, 6.0))
    missed = (pred != y).astype(np.float64)
    w = 1.0 + scale * (0.65 * uncertain + 1.35 * missed)
    return np.clip(w / np.mean(w), 0.35, 4.0).astype(np.float32)


def _oof_and_test(model: Any, x: np.ndarray, y: np.ndarray, x_test: np.ndarray, folds: int, seed: int, sample_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, Any, list[dict[str, Any]]]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    test_fold = np.zeros((len(x_test), folds), dtype=np.float64)
    fold_report = []
    for fold, (tr, va) in enumerate(skf.split(x, y), start=1):
        clf = clone(model)
        _fit_model(clf, x[tr], y[tr], None if sample_weight is None else sample_weight[tr])
        oof[va] = _predict(clf, x[va])
        test_fold[:, fold - 1] = _predict(clf, x_test)
        fold_report.append({"fold": fold, "metrics": _metrics(y[va], oof[va], 0.5)})
    full = clone(model)
    _fit_model(full, x, y, sample_weight)
    return _clip_prob(oof), _clip_prob(test_fold.mean(axis=1)), full, fold_report


def _fusion_candidates(y: np.ndarray, p0: np.ndarray, p0_test: np.ndarray, q: np.ndarray, q_test: np.ndarray, metric: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    l0 = _logit(p0)
    l0_test = _logit(p0_test)
    lq = _logit(q)
    lq_test = _logit(q_test)
    quantiles = sorted(set(float(v) for v in np.quantile(q, [0.50, 0.60, 0.70, 0.80, 0.88, 0.93, 0.96])))
    forms: list[tuple[str, str, np.ndarray, np.ndarray]] = [
        ("direct", "direct", lq, lq_test),
        ("centered", "centered", lq - float(np.median(lq)), lq_test - float(np.median(lq))),
    ]
    for tau_p in quantiles:
        tau_l = float(_logit(np.asarray([tau_p]))[0])
        forms.append((f"rescue_q{tau_p:.3f}", "rescue", np.maximum(0.0, lq - tau_l), np.maximum(0.0, lq_test - tau_l)))
        forms.append((f"window_q{tau_p:.3f}", "window", np.maximum(0.0, lq - tau_l) - 0.25 * np.maximum(0.0, tau_l - lq), np.maximum(0.0, lq_test - tau_l) - 0.25 * np.maximum(0.0, tau_l - lq_test)))
    weights = np.asarray([*np.linspace(0.0, 1.6, 33), 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0], dtype=np.float64)
    for form_name, family, delta, delta_test in forms:
        for weight in weights:
            p = _sigmoid(l0 + float(weight) * delta)
            pt = _sigmoid(l0_test + float(weight) * delta_test)
            threshold, oof = _select_threshold(y, p, metric)
            candidates.append({"name": f"{form_name}/w{weight:.2f}", "family": family, "p_oof": p, "p_test": pt, "threshold": threshold, "oof": oof})
            candidates.append({"name": f"{form_name}/w{weight:.2f}/fixed0p5", "family": family, "p_oof": p, "p_test": pt, "threshold": 0.5, "oof": _metrics(y, p, 0.5)})
    for alpha in [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]:
        p = 1.0 - (1.0 - p0) * np.power(1.0 - q, alpha)
        pt = 1.0 - (1.0 - p0_test) * np.power(1.0 - q_test, alpha)
        threshold, oof = _select_threshold(y, p, metric)
        candidates.append({"name": f"noisy_or/a{alpha:.2f}", "family": "rescue", "p_oof": p, "p_test": pt, "threshold": threshold, "oof": oof})
        candidates.append({"name": f"noisy_or/a{alpha:.2f}/fixed0p5", "family": "rescue", "p_oof": p, "p_test": pt, "threshold": 0.5, "oof": _metrics(y, p, 0.5)})
    for tau_p in quantiles:
        active = q >= tau_p
        active_test = q_test >= tau_p
        for alpha in [0.35, 0.50, 0.75, 1.00]:
            p = np.where(active, np.maximum(p0, alpha * q + (1.0 - alpha) * p0), p0)
            pt = np.where(active_test, np.maximum(p0_test, alpha * q_test + (1.0 - alpha) * p0_test), p0_test)
            threshold, oof = _select_threshold(y, p, metric)
            candidates.append({"name": f"gated_max_q{tau_p:.3f}/a{alpha:.2f}", "family": "rescue", "p_oof": p, "p_test": pt, "threshold": threshold, "oof": oof})
            candidates.append({"name": f"gated_max_q{tau_p:.3f}/a{alpha:.2f}/fixed0p5", "family": "rescue", "p_oof": p, "p_test": pt, "threshold": 0.5, "oof": _metrics(y, p, 0.5)})
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="splits/processed_744/train.csv")
    parser.add_argument("--test-csv", default="splits/processed_744/test.csv")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--main-prob", default="outputs/processed_744_long_vjepa2_dinov3_oof_experts_20260523/strong_fusion_probabilities.npz")
    parser.add_argument("--main-expert-name", default="processed_744_strong_fusion_boosted:blend_logreg_xgb/context_behavior/none/hgb/context")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--threshold-metric", choices=["accuracy", "balanced_accuracy", "macro_f1"], default="macro_f1")
    parser.add_argument("--hard-weight-scale", type=float, default=1.5)
    parser.add_argument("--select-families", nargs="+", default=["anchor", "rescue"])
    parser.add_argument("--out-dir", default="outputs/processed_744_object_cotracker_rescue_20260524")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_split_csv(args.train_csv)
    test_rows = read_split_csv(args.test_csv)
    x_train, y_train_csv, feature_names = _load_matrix(train_rows, args.feature_dir)
    x_test, y_test_csv, _ = _load_matrix(test_rows, args.feature_dir)
    p0_train, p0_test, y_train, y_test, main_name = _load_main_prob(args.main_prob, args.main_expert_name)
    if not np.array_equal(y_train, y_train_csv) or not np.array_equal(y_test, y_test_csv):
        raise ValueError("feature labels and anchor labels do not match")

    report: dict[str, Any] = {
        "feature_dir": args.feature_dir,
        "feature_dim": int(x_train.shape[1]),
        "main_prob": args.main_prob,
        "main_expert": main_name,
        "main_test": _metrics(y_test, p0_test, 0.5),
        "models": [],
        "select_families": args.select_families,
        "hard_weight_scale": args.hard_weight_scale,
        "fusion_candidates_top_by_oof": [],
        "fusion_candidates_top_by_selectable_oof": [],
        "fusion_candidates_top_by_test_diagnostic": [],
    }
    models_out: dict[str, Any] = {}
    all_candidates: list[dict[str, Any]] = []
    threshold0, oof0 = _select_threshold(y_train, p0_train, args.threshold_metric)
    all_candidates.append({"name": "main_anchor", "family": "anchor", "p_oof": p0_train, "p_test": p0_test, "threshold": threshold0, "oof": oof0})
    all_candidates.append({"name": "main_anchor_fixed0p5", "family": "anchor", "p_oof": p0_train, "p_test": p0_test, "threshold": 0.5, "oof": _metrics(y_train, p0_train, 0.5)})
    hard_weight = _hard_sample_weight(y_train, p0_train, args.hard_weight_scale)
    report["hard_weight_stats"] = {"min": float(np.min(hard_weight)), "mean": float(np.mean(hard_weight)), "max": float(np.max(hard_weight))}
    object_oof_cols: list[np.ndarray] = []
    object_test_cols: list[np.ndarray] = []
    object_names: list[str] = []
    for name, model in _models(args.seed, x_train.shape[1]).items():
        print(f"model={name}", flush=True)
        q_oof, q_test, full, folds = _oof_and_test(model, x_train, y_train, x_test, args.folds, args.seed, hard_weight)
        threshold_q, oof_q = _select_threshold(y_train, q_oof, args.threshold_metric)
        test_q = _metrics(y_test, q_test, threshold_q)
        report["models"].append({"name": name, "oof_best": oof_q, "test_at_oof_threshold": test_q, "folds": folds})
        object_oof_cols.append(q_oof.astype(np.float32))
        object_test_cols.append(q_test.astype(np.float32))
        object_names.append(name)
        models_out[name] = full
        all_candidates.append({"name": f"object/{name}", "family": "object", "p_oof": q_oof, "p_test": q_test, "threshold": threshold_q, "oof": oof_q})
        for cand in _fusion_candidates(y_train, p0_train, p0_test, q_oof, q_test, args.threshold_metric):
            cand["name"] = f"{name}/{cand['name']}"
            all_candidates.append(cand)
        print(f"  oof_f1={oof_q['macro_f1']:.4f} test_acc={test_q['accuracy']:.4f} test_auc={test_q['auroc']:.4f}", flush=True)

    ranked = []
    for cand in all_candidates:
        test = _metrics(y_test, cand["p_test"], cand["threshold"])
        ranked.append({"name": cand["name"], "family": cand["family"], "threshold": float(cand["threshold"]), "oof": cand["oof"], "test": test})
    ranked_oof = sorted(ranked, key=lambda x: (x["oof"][args.threshold_metric], x["oof"]["auroc"]), reverse=True)
    ranked_test = sorted(ranked, key=lambda x: (x["test"]["accuracy"], x["test"]["auroc"]), reverse=True)
    selectable_families = set(args.select_families)
    ranked_selectable = [row for row in ranked_oof if row["family"] in selectable_families]
    if not ranked_selectable:
        raise ValueError(f"No candidates matched --select-families={args.select_families}")
    selected = ranked_selectable[0]
    selected_src = next(c for c in all_candidates if c["name"] == selected["name"])
    pred = (selected_src["p_test"] >= selected_src["threshold"]).astype(np.int64)
    predictions = [
        {
            "idx": int(i),
            "path": row["path"],
            "true_label": int(y),
            "prob_collision": float(p),
            "pred_label": int(pr),
            "correct": bool(int(y) == int(pr)),
        }
        for i, (row, y, p, pr) in enumerate(zip(test_rows, y_test, selected_src["p_test"], pred))
    ]
    report["selected_by_oof"] = selected
    report["reached_95_accuracy"] = bool(selected["test"]["accuracy"] >= 0.95)
    report["reached_98_accuracy"] = bool(selected["test"]["accuracy"] >= 0.98)
    report["fusion_candidates_top_by_oof"] = ranked_oof[:50]
    report["fusion_candidates_top_by_selectable_oof"] = ranked_selectable[:50]
    report["fusion_candidates_top_by_test_diagnostic"] = ranked_test[:50]
    write_json(out_dir / "object_cotracker_rescue_summary.json", report)
    write_json(out_dir / "object_cotracker_rescue_predictions.json", {"predictions": predictions})
    write_json(out_dir / "object_cotracker_rescue_errors.json", {"errors": [row for row in predictions if not row["correct"]]})
    np.savez_compressed(
        out_dir / "object_cotracker_rescue_probabilities.npz",
        y_train=y_train,
        y_test=y_test,
        p0_train=p0_train.astype(np.float32),
        p0_test=p0_test.astype(np.float32),
        selected_oof=selected_src["p_oof"].astype(np.float32),
        selected_test=selected_src["p_test"].astype(np.float32),
        selected_threshold=np.asarray([selected_src["threshold"]], dtype=np.float32),
        selected_name=np.asarray([selected["name"]]),
        object_oof=np.stack(object_oof_cols, axis=1).astype(np.float32),
        object_test=np.stack(object_test_cols, axis=1).astype(np.float32),
        object_names=np.asarray(object_names),
    )
    joblib.dump({"models": models_out, "feature_names": feature_names, "selected": selected, "args": vars(args)}, out_dir / "object_cotracker_rescue_models.joblib")
    print("BEST", json.dumps(selected, indent=2), flush=True)
    print(f"errors={int(np.sum(pred != y_test))} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
