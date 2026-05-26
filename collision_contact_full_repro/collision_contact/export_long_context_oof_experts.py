"""Export leakage-safe 40-second context experts as cached OOF probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone

from .common import ensure_dirs, read_split_csv, write_json
from .train_long_context_residual_fusion import (
    _build_context_matrix,
    _event_features,
    _fit,
    _fuse_grid,
    _hard_weights,
    _load_event_probs,
    _metrics,
    _models,
    _oof_predict,
    _predict,
    _sigmoid,
    _logit,
)


def _parse_event_specs(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for item in items:
        if "::" in item:
            run_dir, primary = item.split("::", 1)
        else:
            run_dir, primary = item, "blend_logreg_xgb"
        out.append((run_dir, primary))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="splits/processed_744/train.csv")
    parser.add_argument("--test-csv", default="splits/processed_744/test.csv")
    parser.add_argument("--long-train-csv", default="splits/processed_744_long/train.csv")
    parser.add_argument("--long-test-csv", default="splits/processed_744_long/test.csv")
    parser.add_argument("--long-context-dir", default="outputs/processed_744/long_context_diff_1fps_yolo")
    parser.add_argument("--long-dino-dir", default="outputs/processed_744/long_dino_vits14_5w8f")
    parser.add_argument("--badas-dir", default="outputs/processed_744/badas_window_features")
    parser.add_argument(
        "--event-specs",
        nargs="+",
        default=[
            "outputs/processed_744_strong_fusion_boosted::blend_logreg_xgb",
            "outputs/processed_744_strong_fusion_long_context::blend_logreg_xgb",
            "outputs/processed_744_strong_fusion_long_dino::blend_logreg_xgb",
            "outputs/processed_744_strong_fusion_videomae::blend_logreg_xgb",
        ],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "dino_compact",
            "behavior_dino_compact",
            "behavior_dino_badas_compact",
            "context_behavior",
            "context_behavior_badas",
        ],
    )
    parser.add_argument("--models", nargs="+", default=["hgb", "xgb_shallow", "pca128_svc", "pca64_logreg", "extra_trees"])
    parser.add_argument("--hard-weight-modes", nargs="+", default=["none", "margin", "focal", "error"])
    parser.add_argument("--hard-weight-alpha", type=float, default=1.5)
    parser.add_argument("--hard-weight-gamma", type=float, default=2.0)
    parser.add_argument("--hard-weight-max", type=float, default=5.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--threshold-metric", choices=["accuracy", "macro_f1", "balanced_accuracy"], default="accuracy")
    parser.add_argument("--out-dir", default="outputs/processed_744_long_context_oof_experts_20260522")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)
    train_rows = read_split_csv(args.train_csv)
    test_rows = read_split_csv(args.test_csv)
    y_from_train = np.asarray([int(r["label"]) for r in train_rows], dtype=np.int64)
    y_from_test = np.asarray([int(r["label"]) for r in test_rows], dtype=np.int64)

    variant_defs = {
        "context_behavior": dict(include_long_context=True, include_long_dino=False, include_badas=False, include_meta=True, dino_mode="compact"),
        "context_behavior_badas": dict(include_long_context=True, include_long_dino=False, include_badas=True, include_meta=True, dino_mode="compact"),
        "dino_compact": dict(include_long_context=False, include_long_dino=True, include_badas=False, include_meta=True, dino_mode="compact"),
        "behavior_dino_compact": dict(include_long_context=True, include_long_dino=True, include_badas=False, include_meta=True, dino_mode="compact"),
        "behavior_dino_badas_compact": dict(include_long_context=True, include_long_dino=True, include_badas=True, include_meta=True, dino_mode="compact"),
    }
    missing_variants = sorted(set(args.variants) - set(variant_defs))
    if missing_variants:
        raise ValueError(f"Unknown variants: {missing_variants}")

    context_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for variant_name in args.variants:
        opts = variant_defs[variant_name]
        x_ctx_train, levels = _build_context_matrix(
            train_rows,
            args.long_train_csv,
            long_context_dir=args.long_context_dir,
            long_dino_dir=args.long_dino_dir,
            badas_dir=args.badas_dir,
            **opts,
        )
        x_ctx_test, _ = _build_context_matrix(
            test_rows,
            args.long_test_csv,
            long_context_dir=args.long_context_dir,
            long_dino_dir=args.long_dino_dir,
            badas_dir=args.badas_dir,
            train_levels=levels,
            **opts,
        )
        context_cache[variant_name] = (x_ctx_train, x_ctx_test)

    oof_cols: list[np.ndarray] = []
    test_cols: list[np.ndarray] = []
    names: list[str] = []
    report: dict[str, Any] = {
        "event_specs": args.event_specs,
        "variants": args.variants,
        "models": args.models,
        "hard_weight_modes": args.hard_weight_modes,
        "experts": [],
    }
    y_train = None
    y_test = None

    for spec_idx, (event_run_dir, event_primary) in enumerate(_parse_event_specs(args.event_specs)):
        p_event_train, p_event_test, base_train, base_test, y_tr, y_te, _ = _load_event_probs(event_run_dir, event_primary)
        if y_train is None:
            y_train, y_test = y_tr, y_te
        elif not np.array_equal(y_train, y_tr) or not np.array_equal(y_test, y_te):
            raise ValueError(f"label mismatch for {event_run_dir}")
        if not np.array_equal(y_tr, y_from_train) or not np.array_equal(y_te, y_from_test):
            raise ValueError(f"split labels do not match event probabilities for {event_run_dir}")
        event_train = _event_features(p_event_train, base_train)
        event_test = _event_features(p_event_test, base_test)

        for variant_name in args.variants:
            x_ctx_train, x_ctx_test = context_cache[variant_name]
            x_train = np.concatenate([event_train, x_ctx_train], axis=1)
            x_test = np.concatenate([event_test, x_ctx_test], axis=1)
            all_models = _models(args.seed + 31 * spec_idx, x_train.shape[1], x_train.shape[0])
            unknown_models = sorted(set(args.models) - set(all_models))
            if unknown_models:
                raise ValueError(f"Unknown models: {unknown_models}")
            for hard_mode in args.hard_weight_modes:
                weights = _hard_weights(
                    y_train,
                    p_event_train,
                    hard_mode,
                    alpha=args.hard_weight_alpha,
                    gamma=args.hard_weight_gamma,
                    max_weight=args.hard_weight_max,
                )
                for model_name in args.models:
                    model = clone(all_models[model_name])
                    print(f"event={Path(event_run_dir).name} variant={variant_name} weight={hard_mode} model={model_name}", flush=True)
                    oof_context, folds = _oof_predict(
                        x_train,
                        y_train,
                        model,
                        args.folds,
                        args.seed + spec_idx * 101,
                        weights,
                    )
                    full_model = clone(all_models[model_name])
                    _fit(full_model, x_train, y_train, weights)
                    test_context = _predict(full_model, x_test)
                    fuse = _fuse_grid(y_train, p_event_train, oof_context, args.threshold_metric)
                    oof_fused = fuse["prob"]
                    test_fused = _sigmoid(_logit(p_event_test) + float(fuse["weight"]) * _logit(test_context))

                    prefix = f"{Path(event_run_dir).name}:{event_primary}/{variant_name}/{hard_mode}/{model_name}"
                    for suffix, tr_prob, te_prob in [
                        ("context", oof_context, test_context),
                        (f"fused_w{float(fuse['weight']):.3f}", oof_fused, test_fused),
                    ]:
                        name = f"{prefix}/{suffix}"
                        oof_cols.append(tr_prob.astype(np.float32))
                        test_cols.append(te_prob.astype(np.float32))
                        names.append(name)
                        item = {
                            "name": name,
                            "oof_default": _metrics(y_train, tr_prob, 0.5),
                            "test_default": _metrics(y_test, te_prob, 0.5),
                            "fuse": {k: v for k, v in fuse.items() if k != "prob"} if suffix.startswith("fused") else None,
                            "folds": folds,
                        }
                        report["experts"].append(item)

    assert y_train is not None and y_test is not None
    oof_base = np.stack(oof_cols, axis=1)
    test_base = np.stack(test_cols, axis=1)
    np.savez_compressed(
        out_dir / "strong_fusion_probabilities.npz",
        oof_base=oof_base,
        test_base=test_base,
        y_train=y_train,
        y_test=y_test,
        spec_names=np.asarray(names),
    )
    report["n_experts"] = len(names)
    report["n_train"] = int(y_train.shape[0])
    report["n_test"] = int(y_test.shape[0])
    report["best_by_test_default"] = sorted(
        [
            {
                "name": item["name"],
                "accuracy": item["test_default"]["accuracy"],
                "auroc": item["test_default"]["auroc"],
                "confusion_matrix": item["test_default"]["confusion_matrix"],
            }
            for item in report["experts"]
        ],
        key=lambda x: (x["accuracy"], x["auroc"]),
        reverse=True,
    )[:25]
    write_json(out_dir / "long_context_oof_experts_summary.json", report)
    print(f"wrote {out_dir / 'strong_fusion_probabilities.npz'} with {len(names)} experts", flush=True)


if __name__ == "__main__":
    main()
