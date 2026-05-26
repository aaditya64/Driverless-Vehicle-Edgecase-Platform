"""Select native-frame impulse rescue candidates on train-inner validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .common import read_split_csv, write_json
from .train_deep_impulse_physics_head import _read_object_metrics
from .train_deep_impulse_rescue_fusion import _matrix, _models
from .train_object_cotracker_rescue_fusion import _fit_model, _fusion_candidates, _hard_sample_weight, _load_main_prob, _metrics, _predict, _select_threshold


def _anchor_for_rows(rows: list[dict[str, Any]], full_train_rows: list[dict[str, Any]], full_test_rows: list[dict[str, Any]], p_train: np.ndarray, p_test: np.ndarray) -> np.ndarray:
    by_path = {row["path"]: float(p) for row, p in zip(full_train_rows, p_train)}
    by_path.update({row["path"]: float(p) for row, p in zip(full_test_rows, p_test)})
    return np.asarray([by_path[row["path"]] for row in rows], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-inner-csv", default="splits/processed_744/train_inner.csv")
    parser.add_argument("--val-csv", default="splits/processed_744/val.csv")
    parser.add_argument("--train-csv", default="splits/processed_744/train.csv")
    parser.add_argument("--test-csv", default="splits/processed_744/test.csv")
    parser.add_argument("--feature-dir", default="outputs/processed_744/features")
    parser.add_argument("--object-metrics-csv", action="append", default=["analysis/impact_diagnostics_20260522/object_physics_train_metrics.csv", "analysis/impact_diagnostics_20260522/object_physics_test_metrics.csv"])
    parser.add_argument("--cotracker-feature-dir", default="")
    parser.add_argument("--main-prob", default="outputs/processed_744_long_vjepa2_dinov3_oof_experts_20260523/strong_fusion_probabilities.npz")
    parser.add_argument("--main-expert-name", default="processed_744_strong_fusion_boosted:blend_logreg_xgb/context_behavior/none/hgb/context")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--hard-weight-scale", type=float, default=1.5)
    parser.add_argument("--threshold-metric", choices=["accuracy", "balanced_accuracy", "macro_f1"], default="macro_f1")
    parser.add_argument("--select-threshold-min", type=float, default=0.45)
    parser.add_argument("--select-threshold-max", type=float, default=0.55)
    parser.add_argument("--out-dir", default="outputs/processed_744_val_selected_deep_rescue_20260524")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    object_metrics = _read_object_metrics(args.object_metrics_csv)
    inner_rows = read_split_csv(args.train_inner_csv)
    val_rows = read_split_csv(args.val_csv)
    train_rows = read_split_csv(args.train_csv)
    test_rows = read_split_csv(args.test_csv)
    cotracker_dir = args.cotracker_feature_dir or None
    x_inner, y_inner, feature_names = _matrix(inner_rows, args.feature_dir, object_metrics, cotracker_dir)
    x_val, y_val, _ = _matrix(val_rows, args.feature_dir, object_metrics, cotracker_dir)
    x_test, y_test_csv, _ = _matrix(test_rows, args.feature_dir, object_metrics, cotracker_dir)
    p0_train, p0_test, y_train_full, y_test, main_name = _load_main_prob(args.main_prob, args.main_expert_name)
    if not np.array_equal(y_test, y_test_csv):
        raise ValueError("test labels do not match anchor")
    p0_inner = _anchor_for_rows(inner_rows, train_rows, test_rows, p0_train, p0_test)
    p0_val = _anchor_for_rows(val_rows, train_rows, test_rows, p0_train, p0_test)
    if not np.array_equal(np.asarray([int(r["label"]) for r in train_rows], dtype=np.int64), y_train_full):
        raise ValueError("train labels do not match anchor")
    weight = _hard_sample_weight(y_inner, p0_inner, args.hard_weight_scale)

    candidates: list[dict[str, Any]] = []
    threshold0, val0 = _select_threshold(y_val, p0_val, args.threshold_metric)
    candidates.append({"name": "main_anchor", "family": "anchor", "p_val": p0_val, "p_test": p0_test, "threshold": threshold0, "val": val0})
    candidates.append({"name": "main_anchor_fixed0p5", "family": "anchor", "p_val": p0_val, "p_test": p0_test, "threshold": 0.5, "val": _metrics(y_val, p0_val, 0.5)})
    models_out = {}
    model_reports = []
    for name, model in _models(args.seed, x_inner.shape[1]).items():
        print(f"model={name}", flush=True)
        fitted = model
        _fit_model(fitted, x_inner, y_inner, weight)
        q_val = _predict(fitted, x_val)
        q_test = _predict(fitted, x_test)
        threshold_q, val_q = _select_threshold(y_val, q_val, args.threshold_metric)
        test_q = _metrics(y_test, q_test, threshold_q)
        model_reports.append({"name": name, "val_best": val_q, "test_at_val_threshold": test_q})
        models_out[name] = fitted
        candidates.append({"name": f"impulse/{name}", "family": "object", "p_val": q_val, "p_test": q_test, "threshold": threshold_q, "val": val_q})
        for cand in _fusion_candidates(y_val, p0_val, p0_test, q_val, q_test, args.threshold_metric):
            candidates.append({"name": f"{name}/{cand['name']}", "family": cand["family"], "p_val": cand["p_oof"], "p_test": cand["p_test"], "threshold": cand["threshold"], "val": cand["oof"]})
        print(f"  val_f1={val_q['macro_f1']:.4f} test_acc={test_q['accuracy']:.4f} test_auc={test_q['auroc']:.4f}", flush=True)

    ranked = []
    for cand in candidates:
        ranked.append({"name": cand["name"], "family": cand["family"], "threshold": float(cand["threshold"]), "val": cand["val"], "test": _metrics(y_test, cand["p_test"], cand["threshold"])})
    ranked_val = sorted(ranked, key=lambda x: (x["val"][args.threshold_metric], x["val"]["auroc"]), reverse=True)
    selectable = [
        row
        for row in ranked_val
        if row["family"] in {"anchor", "rescue"} and args.select_threshold_min <= row["threshold"] <= args.select_threshold_max
    ]
    if not selectable:
        raise ValueError("no selectable candidates")
    selected = selectable[0]
    selected_src = next(c for c in candidates if c["name"] == selected["name"])
    pred = (selected_src["p_test"] >= selected_src["threshold"]).astype(np.int64)
    predictions = [
        {"idx": int(i), "path": row["path"], "true_label": int(y), "prob_collision": float(p), "pred_label": int(pr), "correct": bool(int(y) == int(pr))}
        for i, (row, y, p, pr) in enumerate(zip(test_rows, y_test, selected_src["p_test"], pred))
    ]
    report = {
        "main_expert": main_name,
        "feature_dir": args.feature_dir,
        "cotracker_feature_dir": args.cotracker_feature_dir,
        "feature_dim": int(x_inner.shape[1]),
        "threshold_constraint": [args.select_threshold_min, args.select_threshold_max],
        "models": model_reports,
        "selected_by_val": selected,
        "reached_95_accuracy": bool(selected["test"]["accuracy"] >= 0.95),
        "reached_98_accuracy": bool(selected["test"]["accuracy"] >= 0.98),
        "candidates_top_by_val": ranked_val[:80],
        "candidates_top_by_selectable_val": selectable[:80],
        "candidates_top_by_test_diagnostic": sorted(ranked, key=lambda x: (x["test"]["accuracy"], x["test"]["auroc"]), reverse=True)[:80],
    }
    write_json(out_dir / "val_selected_deep_rescue_summary.json", report)
    write_json(out_dir / "val_selected_deep_rescue_predictions.json", {"predictions": predictions})
    write_json(out_dir / "val_selected_deep_rescue_errors.json", {"errors": [row for row in predictions if not row["correct"]]})
    np.savez_compressed(
        out_dir / "val_selected_deep_rescue_probabilities.npz",
        y_val=y_val,
        y_test=y_test,
        p0_val=p0_val.astype(np.float32),
        p0_test=p0_test.astype(np.float32),
        selected_val=selected_src["p_val"].astype(np.float32),
        selected_test=selected_src["p_test"].astype(np.float32),
        selected_threshold=np.asarray([selected_src["threshold"]], dtype=np.float32),
        selected_name=np.asarray([selected["name"]]),
    )
    joblib.dump({"models": models_out, "feature_names": feature_names, "selected": selected, "args": vars(args)}, out_dir / "val_selected_deep_rescue_models.joblib")
    print("BEST", json.dumps(selected, indent=2), flush=True)
    print(f"errors={int(np.sum(pred != y_test))} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
