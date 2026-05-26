"""Train constrained rescue fusion from native-frame impulse physics features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .common import read_split_csv, stable_video_id, write_json
from .train_deep_impulse_physics_head import _read_object_metrics, build_deep_features
from .train_object_cotracker_rescue_fusion import (
    _fusion_candidates,
    _hard_sample_weight,
    _load_main_prob,
    _metrics,
    _oof_and_test,
    _select_threshold,
    _signed_log1p,
)

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def _cotracker_summary(feature_dir: str | Path, video_path: str | Path) -> tuple[np.ndarray, list[str]]:
    path = Path(feature_dir) / f"{stable_video_id(video_path)}.npz"
    data = np.load(path, allow_pickle=True)
    return np.nan_to_num(data["summary"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), [f"cotracker_{x}" for x in data["summary_names"]]


def _matrix(rows: list[dict[str, Any]], feature_dir: str | Path, object_metrics: dict[str, np.ndarray], cotracker_dir: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    names: list[str] | None = None
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    for row in rows:
        vec, vec_names = build_deep_features(feature_dir, row["path"], object_metrics)
        if cotracker_dir:
            cvec, cnames = _cotracker_summary(cotracker_dir, row["path"])
            vec = np.concatenate([vec, cvec])
            vec_names = vec_names + cnames
        xs.append(np.nan_to_num(vec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        if names is None:
            names = vec_names
    if names is None:
        raise ValueError("empty split")
    return np.stack(xs).astype(np.float32), y, names


def _models(seed: int, dim: int) -> dict[str, Any]:
    k256 = min(256, dim)
    k512 = min(512, dim)
    models: dict[str, Any] = {
        "logreg_k256": make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            StandardScaler(),
            SelectKBest(f_classif, k=k256),
            LogisticRegression(C=0.12, max_iter=5000, class_weight="balanced", random_state=seed),
        ),
        "hgb_k256": make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            SelectKBest(f_classif, k=k256),
            HistGradientBoostingClassifier(max_iter=280, learning_rate=0.025, max_leaf_nodes=11, min_samples_leaf=14, l2_regularization=1.0, random_state=seed + 1),
        ),
        "hgb_k512": make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            SelectKBest(f_classif, k=k512),
            HistGradientBoostingClassifier(max_iter=320, learning_rate=0.022, max_leaf_nodes=13, min_samples_leaf=12, l2_regularization=1.5, random_state=seed + 2),
        ),
        "extra_trees_k512": make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            SelectKBest(f_classif, k=k512),
            ExtraTreesClassifier(n_estimators=900, max_depth=6, min_samples_leaf=4, max_features="sqrt", class_weight="balanced", random_state=seed + 3, n_jobs=-1),
        ),
    }
    if XGBClassifier is not None:
        models["xgb_k512"] = make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            SelectKBest(f_classif, k=k512),
            XGBClassifier(
                n_estimators=340,
                max_depth=2,
                learning_rate=0.022,
                subsample=0.88,
                colsample_bytree=0.70,
                min_child_weight=8,
                reg_alpha=0.7,
                reg_lambda=9.0,
                eval_metric="logloss",
                random_state=seed + 4,
                n_jobs=4,
            ),
        )
    if LGBMClassifier is not None:
        models["lgbm_k512"] = make_pipeline(
            SimpleImputer(),
            FunctionTransformer(_signed_log1p, validate=False),
            SelectKBest(f_classif, k=k512),
            LGBMClassifier(
                n_estimators=340,
                num_leaves=7,
                learning_rate=0.022,
                subsample=0.88,
                colsample_bytree=0.70,
                min_child_samples=16,
                reg_alpha=0.4,
                reg_lambda=7.0,
                class_weight="balanced",
                objective="binary",
                random_state=seed + 5,
                n_jobs=4,
                verbose=-1,
            ),
        )
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="splits/processed_744/train.csv")
    parser.add_argument("--test-csv", default="splits/processed_744/test.csv")
    parser.add_argument("--feature-dir", default="outputs/processed_744/features")
    parser.add_argument("--cotracker-feature-dir", default="")
    parser.add_argument("--object-metrics-csv", action="append", default=["analysis/impact_diagnostics_20260522/object_physics_train_metrics.csv", "analysis/impact_diagnostics_20260522/object_physics_test_metrics.csv"])
    parser.add_argument("--main-prob", default="outputs/processed_744_long_vjepa2_dinov3_oof_experts_20260523/strong_fusion_probabilities.npz")
    parser.add_argument("--main-expert-name", default="processed_744_strong_fusion_boosted:blend_logreg_xgb/context_behavior/none/hgb/context")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--threshold-metric", choices=["accuracy", "balanced_accuracy", "macro_f1"], default="macro_f1")
    parser.add_argument("--hard-weight-scale", type=float, default=1.5)
    parser.add_argument("--select-families", nargs="+", default=["anchor", "rescue"])
    parser.add_argument("--out-dir", default="outputs/processed_744_deep_impulse_rescue_20260524")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    object_metrics = _read_object_metrics(args.object_metrics_csv)
    train_rows = read_split_csv(args.train_csv)
    test_rows = read_split_csv(args.test_csv)
    cotracker_dir = args.cotracker_feature_dir or None
    x_train, y_train_csv, feature_names = _matrix(train_rows, args.feature_dir, object_metrics, cotracker_dir)
    x_test, y_test_csv, _ = _matrix(test_rows, args.feature_dir, object_metrics, cotracker_dir)
    p0_train, p0_test, y_train, y_test, main_name = _load_main_prob(args.main_prob, args.main_expert_name)
    if not np.array_equal(y_train, y_train_csv) or not np.array_equal(y_test, y_test_csv):
        raise ValueError("feature labels and anchor labels do not match")

    report: dict[str, Any] = {
        "feature_dir": args.feature_dir,
        "cotracker_feature_dir": args.cotracker_feature_dir,
        "feature_dim": int(x_train.shape[1]),
        "main_prob": args.main_prob,
        "main_expert": main_name,
        "main_test": _metrics(y_test, p0_test, 0.5),
        "select_families": args.select_families,
        "hard_weight_scale": args.hard_weight_scale,
        "models": [],
    }
    hard_weight = _hard_sample_weight(y_train, p0_train, args.hard_weight_scale)
    report["hard_weight_stats"] = {"min": float(np.min(hard_weight)), "mean": float(np.mean(hard_weight)), "max": float(np.max(hard_weight))}
    candidates: list[dict[str, Any]] = []
    threshold0, oof0 = _select_threshold(y_train, p0_train, args.threshold_metric)
    candidates.append({"name": "main_anchor", "family": "anchor", "p_oof": p0_train, "p_test": p0_test, "threshold": threshold0, "oof": oof0})
    candidates.append({"name": "main_anchor_fixed0p5", "family": "anchor", "p_oof": p0_train, "p_test": p0_test, "threshold": 0.5, "oof": _metrics(y_train, p0_train, 0.5)})
    models_out = {}
    expert_oof_cols = []
    expert_test_cols = []
    expert_names = []
    for name, model in _models(args.seed, x_train.shape[1]).items():
        print(f"model={name}", flush=True)
        q_oof, q_test, full, folds = _oof_and_test(model, x_train, y_train, x_test, args.folds, args.seed, hard_weight)
        threshold_q, oof_q = _select_threshold(y_train, q_oof, args.threshold_metric)
        test_q = _metrics(y_test, q_test, threshold_q)
        report["models"].append({"name": name, "oof_best": oof_q, "test_at_oof_threshold": test_q, "folds": folds})
        models_out[name] = full
        expert_oof_cols.append(q_oof.astype(np.float32))
        expert_test_cols.append(q_test.astype(np.float32))
        expert_names.append(name)
        candidates.append({"name": f"impulse/{name}", "family": "object", "p_oof": q_oof, "p_test": q_test, "threshold": threshold_q, "oof": oof_q})
        for cand in _fusion_candidates(y_train, p0_train, p0_test, q_oof, q_test, args.threshold_metric):
            cand["name"] = f"{name}/{cand['name']}"
            candidates.append(cand)
        print(f"  oof_f1={oof_q['macro_f1']:.4f} test_acc={test_q['accuracy']:.4f} test_auc={test_q['auroc']:.4f}", flush=True)

    ranked = []
    for cand in candidates:
        ranked.append({"name": cand["name"], "family": cand["family"], "threshold": float(cand["threshold"]), "oof": cand["oof"], "test": _metrics(y_test, cand["p_test"], cand["threshold"])})
    ranked_oof = sorted(ranked, key=lambda x: (x["oof"][args.threshold_metric], x["oof"]["auroc"]), reverse=True)
    selectable = [row for row in ranked_oof if row["family"] in set(args.select_families)]
    if not selectable:
        raise ValueError(f"No candidates matched --select-families={args.select_families}")
    selected = selectable[0]
    selected_src = next(c for c in candidates if c["name"] == selected["name"])
    pred = (selected_src["p_test"] >= selected_src["threshold"]).astype(np.int64)
    predictions = [
        {"idx": int(i), "path": row["path"], "true_label": int(y), "prob_collision": float(p), "pred_label": int(pr), "correct": bool(int(y) == int(pr))}
        for i, (row, y, p, pr) in enumerate(zip(test_rows, y_test, selected_src["p_test"], pred))
    ]
    report["selected_by_oof"] = selected
    report["reached_95_accuracy"] = bool(selected["test"]["accuracy"] >= 0.95)
    report["reached_98_accuracy"] = bool(selected["test"]["accuracy"] >= 0.98)
    report["fusion_candidates_top_by_oof"] = ranked_oof[:50]
    report["fusion_candidates_top_by_selectable_oof"] = selectable[:50]
    report["fusion_candidates_top_by_test_diagnostic"] = sorted(ranked, key=lambda x: (x["test"]["accuracy"], x["test"]["auroc"]), reverse=True)[:50]
    write_json(out_dir / "deep_impulse_rescue_summary.json", report)
    write_json(out_dir / "deep_impulse_rescue_predictions.json", {"predictions": predictions})
    write_json(out_dir / "deep_impulse_rescue_errors.json", {"errors": [row for row in predictions if not row["correct"]]})
    np.savez_compressed(
        out_dir / "deep_impulse_rescue_probabilities.npz",
        y_train=y_train,
        y_test=y_test,
        p0_train=p0_train.astype(np.float32),
        p0_test=p0_test.astype(np.float32),
        selected_oof=selected_src["p_oof"].astype(np.float32),
        selected_test=selected_src["p_test"].astype(np.float32),
        selected_threshold=np.asarray([selected_src["threshold"]], dtype=np.float32),
        selected_name=np.asarray([selected["name"]]),
        expert_oof=np.stack(expert_oof_cols, axis=1).astype(np.float32),
        expert_test=np.stack(expert_test_cols, axis=1).astype(np.float32),
        expert_names=np.asarray(expert_names),
    )
    joblib.dump({"models": models_out, "feature_names": feature_names, "selected": selected, "args": vars(args)}, out_dir / "deep_impulse_rescue_models.joblib")
    print("BEST", json.dumps(selected, indent=2), flush=True)
    print(f"errors={int(np.sum(pred != y_test))} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
