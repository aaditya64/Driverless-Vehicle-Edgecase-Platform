"""Run the strongest cached-feature fusion experiment for processed Nexar clips."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .common import load_config, read_split_csv, stable_video_id, write_json
from .dataset import _load_feature_arrays

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


@dataclass(frozen=True)
class BaseSpec:
    name: str
    group: str
    model: Any


def _feature_file(feature_dir: str | Path, video_path: str | Path) -> Path:
    return Path(feature_dir) / f"{stable_video_id(video_path)}.npz"


def _clean(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def _stats_2d(x: np.ndarray) -> np.ndarray:
    x = _clean(x)
    parts = [
        np.nanmean(x, axis=0),
        np.nanstd(x, axis=0),
        np.nanmin(x, axis=0),
        np.nanmax(x, axis=0),
        np.nanpercentile(x, 25, axis=0),
        np.nanpercentile(x, 50, axis=0),
        np.nanpercentile(x, 75, axis=0),
        np.nanpercentile(np.abs(x), 95, axis=0),
    ]
    if x.shape[0] > 1:
        d = np.diff(x, axis=0)
        parts.extend([np.nanmean(np.abs(d), axis=0), np.nanmax(np.abs(d), axis=0)])
    else:
        parts.extend([np.zeros(x.shape[1], dtype=np.float32), np.zeros(x.shape[1], dtype=np.float32)])
    return _clean(np.concatenate(parts, axis=0))


def _wavelet_extra_summary(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    feats: list[np.ndarray] = []
    for key in ["raw", "raw_abs", "swt", "swt_abs"]:
        if key in data:
            feats.append(_stats_2d(data[key]))
    if "cwt_abs" in data and "freqs" in data and "wavelet_channels" in data:
        cwt = _clean(data["cwt_abs"])
        freqs = data["freqs"].astype(np.float32)
        n_ch = len(data["wavelet_channels"])
        n_freq = len(freqs)
        cwt3 = cwt.reshape(cwt.shape[0], n_ch, n_freq)
        bands = [
            (freqs >= 0.5) & (freqs < 2.0),
            (freqs >= 2.0) & (freqs < 6.0),
            (freqs >= 6.0) & (freqs < 10.0),
            (freqs >= 10.0) & (freqs <= 14.0),
        ]
        band_series = []
        for mask in bands:
            band = cwt3[:, :, mask].mean(axis=2)
            band_series.append(band)
            total = band.mean(axis=1, keepdims=True)
            band_series.append(total)
        feats.append(_stats_2d(np.concatenate(band_series, axis=1)))
    return _clean(np.concatenate(feats, axis=0))


def _load_summary(feature_dir: str | Path, video_path: str, key: str = "summary") -> np.ndarray:
    path = _feature_file(feature_dir, video_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return _clean(np.load(path, allow_pickle=True)[key])


def _load_badas_summary(feature_dir: str | Path, video_path: str) -> np.ndarray:
    path = _feature_file(feature_dir, video_path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    z = _clean(data["features"])
    risk = _clean(data["risk_scores"]).reshape(-1, 1)
    logits = _clean(data["logits"])
    parts = [_stats_2d(z), _stats_2d(risk), _stats_2d(logits)]
    if z.shape[0] > 1:
        slope = z[-1] - z[0]
        parts.append(slope)
    risk_flat = risk[:, 0]
    parts.append(
        np.asarray(
            [
                risk_flat.mean(),
                risk_flat.std(),
                risk_flat.min(),
                risk_flat.max(),
                np.percentile(risk_flat, 25),
                np.percentile(risk_flat, 50),
                np.percentile(risk_flat, 75),
                risk_flat[-1] - risk_flat[0],
                risk_flat[:3].mean(),
                risk_flat[3:6].mean(),
                risk_flat[6:].mean(),
                1.0 - float(np.prod(1.0 - np.clip(risk_flat, 0.0, 1.0))),
            ],
            dtype=np.float32,
        )
    )
    return _clean(np.concatenate(parts, axis=0))


def _read_source_map(csv_path: str | None) -> dict[str, str]:
    if not csv_path:
        return {}
    out: dict[str, str] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_path = row.get("source_path")
            if source_path:
                out[str(row["path"])] = str(source_path)
    return out


def _load_groups(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    args: argparse.Namespace,
    source_by_clip: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    sequence_keys = list(cfg["features"].get("sequence_keys", ["raw", "cwt", "swt"]))
    paths: list[str] = []
    labels: list[int] = []
    hand: list[np.ndarray] = []
    wave: list[np.ndarray] = []
    dino2: list[np.ndarray] = []
    dino3: list[np.ndarray] = []
    videomae: list[np.ndarray] = []
    raft: list[np.ndarray] = []
    yolo: list[np.ndarray] = []
    badas: list[np.ndarray] = []
    long_context: list[np.ndarray] = []
    long_dino: list[np.ndarray] = []
    source_by_clip = source_by_clip or {}
    for row in rows:
        video_path = row["path"]
        feature_path = _feature_file(cfg["paths"]["feature_dir"], video_path)
        _, hand_local = _load_feature_arrays(cfg["paths"]["feature_dir"], video_path, sequence_keys, "handcrafted_local")
        _, hand_global = _load_feature_arrays(cfg["paths"]["feature_dir"], video_path, sequence_keys, "handcrafted")
        hand.append(_clean(np.concatenate([hand_local, hand_global], axis=0)))
        wave.append(_wavelet_extra_summary(feature_path))
        dino2.append(_load_summary(args.dino2_dir, video_path))
        dino3.append(_load_summary(args.dino3_dir, video_path))
        videomae.append(_load_summary(args.videomae_dir, video_path))
        raft.append(_load_summary(args.raft_dir, video_path))
        yolo.append(_load_summary(args.yolo_dir, video_path))
        badas.append(_load_badas_summary(args.badas_dir, video_path))
        if args.long_context_dir:
            source_path = source_by_clip.get(video_path)
            if not source_path:
                raise KeyError(f"Missing long-context source path for clip: {video_path}")
            long_context.append(_load_summary(args.long_context_dir, source_path))
        if args.long_dino_dir:
            source_path = source_by_clip.get(video_path)
            if not source_path:
                raise KeyError(f"Missing long-DINO source path for clip: {video_path}")
            long_dino.append(_load_summary(args.long_dino_dir, source_path))
        labels.append(int(row["label"]))
        paths.append(video_path)
    groups = {
        "hand": np.stack(hand),
        "wave": np.stack(wave),
        "physics": np.concatenate([np.stack(hand), np.stack(wave)], axis=1),
        "flow_obj": np.concatenate([np.stack(raft), np.stack(yolo)], axis=1),
        "visual": np.concatenate([np.stack(dino2), np.stack(dino3)], axis=1),
        "videomae": np.stack(videomae),
        "badas": np.stack(badas),
    }
    groups["video_semantic"] = np.concatenate([groups["visual"], groups["videomae"]], axis=1)
    groups["physics_flow_obj"] = np.concatenate([groups["physics"], groups["flow_obj"]], axis=1)
    groups["all"] = np.concatenate([groups["physics"], groups["flow_obj"], groups["video_semantic"], groups["badas"]], axis=1)
    if long_context:
        groups["long_context"] = np.stack(long_context)
        groups["long_context_physics"] = np.concatenate([groups["long_context"], groups["physics"]], axis=1)
        groups["long_context_flow_obj"] = np.concatenate([groups["long_context"], groups["flow_obj"]], axis=1)
        groups["all_short"] = groups["all"]
        groups["all"] = np.concatenate([groups["all"], groups["long_context"]], axis=1)
    if long_dino:
        groups["long_dino"] = np.stack(long_dino)
        groups["long_dino_physics"] = np.concatenate([groups["long_dino"], groups["physics"]], axis=1)
        groups["long_dino_flow_obj"] = np.concatenate([groups["long_dino"], groups["flow_obj"]], axis=1)
        if "long_context" in groups:
            groups["long_behavior_semantic"] = np.concatenate([groups["long_context"], groups["long_dino"]], axis=1)
            groups["long_behavior_semantic_physics"] = np.concatenate([groups["long_context"], groups["long_dino"], groups["physics"]], axis=1)
        groups["all"] = np.concatenate([groups["all"], groups["long_dino"]], axis=1)
    return groups, np.asarray(labels, dtype=np.int64), paths


def _xgb(seed: int, n_estimators: int, depth: int, lr: float, subsample: float = 0.8) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=depth,
        learning_rate=lr,
        subsample=subsample,
        colsample_bytree=0.82,
        min_child_weight=2.0,
        reg_alpha=0.08,
        reg_lambda=1.2,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def _pca_model(n_components: int, clf: Any) -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=n_components, random_state=20260520), clf)


def _scaled_model(clf: Any) -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)


def _base_specs(seed: int, use_long_context: bool = False, use_long_dino: bool = False) -> list[BaseSpec]:
    specs = [
        BaseSpec("physics_xgb_shallow", "physics", _xgb(seed, 320, 2, 0.035, 0.78)),
        BaseSpec("physics_xgb_deeper", "physics", _xgb(seed + 1, 260, 3, 0.035, 0.78)),
        BaseSpec("physics_flow_obj_xgb", "physics_flow_obj", _xgb(seed + 2, 300, 2, 0.035, 0.82)),
        BaseSpec("physics_extra_trees", "physics", ExtraTreesClassifier(n_estimators=900, max_depth=9, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)),
        BaseSpec("physics_flow_obj_hgb", "physics_flow_obj", make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=280, learning_rate=0.028, l2_regularization=0.12, random_state=seed))),
        BaseSpec("badas_svc_pca96", "badas", _pca_model(96, SVC(C=1.5, gamma="scale", class_weight="balanced", probability=True, random_state=seed))),
        BaseSpec("badas_logreg_pca160", "badas", _pca_model(160, LogisticRegression(max_iter=2000, C=0.8, class_weight="balanced", random_state=seed))),
        BaseSpec("visual_svc_pca160", "visual", _pca_model(160, SVC(C=1.2, gamma="scale", class_weight="balanced", probability=True, random_state=seed))),
        BaseSpec("visual_logreg_pca192", "visual", _pca_model(192, LogisticRegression(max_iter=2000, C=0.7, class_weight="balanced", random_state=seed))),
        BaseSpec("videomae_svc_pca128", "videomae", _pca_model(128, SVC(C=1.2, gamma="scale", class_weight="balanced", probability=True, random_state=seed))),
        BaseSpec("videomae_logreg_pca160", "videomae", _pca_model(160, LogisticRegression(max_iter=2000, C=0.7, class_weight="balanced", random_state=seed))),
        BaseSpec("video_semantic_svc_pca220", "video_semantic", _pca_model(220, SVC(C=1.1, gamma="scale", class_weight="balanced", probability=True, random_state=seed))),
        BaseSpec("all_svc_pca220", "all", _pca_model(220, SVC(C=1.2, gamma="scale", class_weight="balanced", probability=True, random_state=seed))),
        BaseSpec("all_logreg_pca260", "all", _pca_model(260, LogisticRegression(max_iter=2500, C=0.55, class_weight="balanced", random_state=seed))),
        BaseSpec("all_rf_pca220", "all", _pca_model(220, RandomForestClassifier(n_estimators=700, max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=-1))),
        BaseSpec("all_xgb_pca220", "all", _pca_model(220, _xgb(seed + 3, 240, 2, 0.035, 0.82))),
    ]
    if use_long_context:
        specs.extend(
            [
                BaseSpec("long_context_xgb", "long_context", _xgb(seed + 20, 360, 2, 0.03, 0.86)),
                BaseSpec("long_context_xgb_deeper", "long_context", _xgb(seed + 21, 300, 3, 0.028, 0.84)),
                BaseSpec(
                    "long_context_hgb",
                    "long_context",
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        HistGradientBoostingClassifier(max_iter=320, learning_rate=0.025, l2_regularization=0.18, random_state=seed + 22),
                    ),
                ),
                BaseSpec(
                    "long_context_extra_trees",
                    "long_context",
                    ExtraTreesClassifier(
                        n_estimators=900,
                        max_depth=9,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=seed + 23,
                        n_jobs=-1,
                    ),
                ),
                BaseSpec("long_context_svc_pca160", "long_context", _pca_model(160, SVC(C=1.2, gamma="scale", class_weight="balanced", probability=True, random_state=seed + 24))),
                BaseSpec("long_context_logreg_pca220", "long_context", _pca_model(220, LogisticRegression(max_iter=2500, C=0.55, class_weight="balanced", random_state=seed + 25))),
                BaseSpec("long_context_physics_xgb", "long_context_physics", _xgb(seed + 26, 340, 2, 0.03, 0.86)),
                BaseSpec("long_context_flow_obj_xgb", "long_context_flow_obj", _xgb(seed + 27, 320, 2, 0.03, 0.86)),
                BaseSpec("all_long_svc_pca320", "all", _pca_model(320, SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed + 28))),
                BaseSpec("all_long_logreg_pca360", "all", _pca_model(360, LogisticRegression(max_iter=3000, C=0.45, class_weight="balanced", random_state=seed + 29))),
            ]
        )
    if use_long_dino:
        specs.extend(
            [
                BaseSpec("long_dino_svc_pca160", "long_dino", _pca_model(160, SVC(C=1.2, gamma="scale", class_weight="balanced", probability=True, random_state=seed + 40))),
                BaseSpec("long_dino_logreg_pca220", "long_dino", _pca_model(220, LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=seed + 41))),
                BaseSpec("long_dino_physics_svc_pca260", "long_dino_physics", _pca_model(260, SVC(C=1.1, gamma="scale", class_weight="balanced", probability=True, random_state=seed + 42))),
                BaseSpec("long_dino_physics_logreg_pca320", "long_dino_physics", _pca_model(320, LogisticRegression(max_iter=3000, C=0.45, class_weight="balanced", random_state=seed + 43))),
                BaseSpec("long_dino_flow_obj_svc_pca220", "long_dino_flow_obj", _pca_model(220, SVC(C=1.1, gamma="scale", class_weight="balanced", probability=True, random_state=seed + 44))),
                BaseSpec("all_long_semantic_svc_pca360", "all", _pca_model(360, SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed + 45))),
                BaseSpec("all_long_semantic_logreg_pca420", "all", _pca_model(420, LogisticRegression(max_iter=3500, C=0.4, class_weight="balanced", random_state=seed + 46))),
            ]
        )
        if use_long_context:
            specs.extend(
                [
                    BaseSpec("long_behavior_semantic_svc_pca220", "long_behavior_semantic", _pca_model(220, SVC(C=1.1, gamma="scale", class_weight="balanced", probability=True, random_state=seed + 47))),
                    BaseSpec("long_behavior_semantic_physics_logreg_pca360", "long_behavior_semantic_physics", _pca_model(360, LogisticRegression(max_iter=3500, C=0.45, class_weight="balanced", random_state=seed + 48))),
                    BaseSpec("long_behavior_semantic_physics_svc_pca360", "long_behavior_semantic_physics", _pca_model(360, SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed + 49))),
                ]
            )
    if LGBMClassifier is not None:
        specs.extend(
            [
                BaseSpec(
                    "physics_lgbm",
                    "physics",
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        LGBMClassifier(
                            n_estimators=420,
                            learning_rate=0.025,
                            num_leaves=15,
                            max_depth=4,
                            subsample=0.82,
                            colsample_bytree=0.82,
                            reg_alpha=0.08,
                            reg_lambda=1.2,
                            class_weight="balanced",
                            random_state=seed,
                            n_jobs=-1,
                            verbosity=-1,
                        ),
                    ),
                ),
                BaseSpec(
                    "all_lgbm_pca220",
                    "all",
                    _pca_model(
                        220,
                        LGBMClassifier(
                            n_estimators=320,
                            learning_rate=0.025,
                            num_leaves=15,
                            max_depth=4,
                            subsample=0.82,
                            colsample_bytree=0.82,
                            reg_alpha=0.08,
                            reg_lambda=1.2,
                            class_weight="balanced",
                            random_state=seed + 7,
                            n_jobs=-1,
                            verbosity=-1,
                        ),
                    ),
                ),
            ]
        )
        if use_long_context:
            specs.extend(
                [
                    BaseSpec(
                        "long_context_lgbm",
                        "long_context",
                        make_pipeline(
                            SimpleImputer(strategy="median"),
                            LGBMClassifier(
                                n_estimators=460,
                                learning_rate=0.022,
                                num_leaves=15,
                                max_depth=4,
                                subsample=0.84,
                                colsample_bytree=0.84,
                                reg_alpha=0.1,
                                reg_lambda=1.5,
                                class_weight="balanced",
                                random_state=seed + 30,
                                n_jobs=-1,
                                verbosity=-1,
                            ),
                        ),
                    ),
                    BaseSpec(
                        "all_long_lgbm_pca320",
                        "all",
                        _pca_model(
                            320,
                            LGBMClassifier(
                                n_estimators=360,
                                learning_rate=0.022,
                                num_leaves=15,
                                max_depth=4,
                                subsample=0.84,
                                colsample_bytree=0.84,
                                reg_alpha=0.1,
                                reg_lambda=1.5,
                                class_weight="balanced",
                                random_state=seed + 31,
                                n_jobs=-1,
                                verbosity=-1,
                            ),
                        ),
                    ),
                ]
            )
        if use_long_dino:
            specs.extend(
                [
                    BaseSpec(
                        "long_dino_lgbm_pca220",
                        "long_dino",
                        _pca_model(
                            220,
                            LGBMClassifier(
                                n_estimators=360,
                                learning_rate=0.022,
                                num_leaves=15,
                                max_depth=4,
                                subsample=0.84,
                                colsample_bytree=0.84,
                                reg_alpha=0.12,
                                reg_lambda=1.6,
                                class_weight="balanced",
                                random_state=seed + 50,
                                n_jobs=-1,
                                verbosity=-1,
                            ),
                        ),
                    ),
                    BaseSpec(
                        "long_dino_physics_lgbm_pca320",
                        "long_dino_physics",
                        _pca_model(
                            320,
                            LGBMClassifier(
                                n_estimators=360,
                                learning_rate=0.022,
                                num_leaves=15,
                                max_depth=4,
                                subsample=0.84,
                                colsample_bytree=0.84,
                                reg_alpha=0.12,
                                reg_lambda=1.6,
                                class_weight="balanced",
                                random_state=seed + 51,
                                n_jobs=-1,
                                verbosity=-1,
                            ),
                        ),
                    ),
                ]
            )
    if CatBoostClassifier is not None:
        cat_params = {
            "iterations": 360,
            "depth": 4,
            "learning_rate": 0.035,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
            "l2_leaf_reg": 5.0,
        }
        specs.extend(
            [
                BaseSpec("physics_catboost", "physics", make_pipeline(SimpleImputer(strategy="median"), CatBoostClassifier(**cat_params))),
                BaseSpec("physics_flow_obj_catboost", "physics_flow_obj", make_pipeline(SimpleImputer(strategy="median"), CatBoostClassifier(**cat_params))),
                BaseSpec("all_catboost_pca220", "all", _pca_model(220, CatBoostClassifier(**{**cat_params, "random_seed": seed + 9}))),
            ]
        )
        if use_long_context:
            specs.extend(
                [
                    BaseSpec("long_context_catboost", "long_context", make_pipeline(SimpleImputer(strategy="median"), CatBoostClassifier(**{**cat_params, "random_seed": seed + 32}))),
                    BaseSpec("all_long_catboost_pca320", "all", _pca_model(320, CatBoostClassifier(**{**cat_params, "random_seed": seed + 33}))),
                ]
            )
        if use_long_dino:
            specs.extend(
                [
                    BaseSpec("long_dino_catboost_pca220", "long_dino", _pca_model(220, CatBoostClassifier(**{**cat_params, "random_seed": seed + 52}))),
                    BaseSpec("long_dino_physics_catboost_pca320", "long_dino_physics", _pca_model(320, CatBoostClassifier(**{**cat_params, "random_seed": seed + 53}))),
                ]
            )
    return specs


def _proba(model: Any, x: np.ndarray) -> np.ndarray:
    p = model.predict_proba(x)
    p = np.asarray(p, dtype=np.float64)
    if p.shape[1] == 2:
        return p
    out = np.zeros((p.shape[0], 2), dtype=np.float64)
    for idx, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, idx]
    return out


def _metrics(y_true: np.ndarray, p_collision: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (p_collision >= threshold).astype(np.int64)
    prob = np.stack([1.0 - p_collision, p_collision], axis=1)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "collision_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "collision_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "near_miss_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "near_miss_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "auroc": float(roc_auc_score(y_true, p_collision)),
    }


def _select_threshold(y_true: np.ndarray, p_collision: np.ndarray, metric: str) -> tuple[float, dict[str, Any]]:
    best_thr = 0.5
    best_score = -1.0
    best_metrics: dict[str, Any] = {}
    for thr in np.linspace(0.2, 0.8, 121):
        m = _metrics(y_true, p_collision, float(thr))
        score = float(m[metric])
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = m
    return best_thr, best_metrics


def _meta_features(probs: np.ndarray) -> np.ndarray:
    p = probs.astype(np.float64)
    stats = np.stack(
        [
            p.mean(axis=1),
            p.std(axis=1),
            p.min(axis=1),
            p.max(axis=1),
            np.percentile(p, 25, axis=1),
            np.percentile(p, 50, axis=1),
            np.percentile(p, 75, axis=1),
        ],
        axis=1,
    )
    return np.concatenate([p, stats], axis=1)


def _train_oof(
    train_groups: dict[str, np.ndarray],
    y_train: np.ndarray,
    test_groups: dict[str, np.ndarray],
    specs: list[BaseSpec],
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros((len(y_train), len(specs)), dtype=np.float64)
    test_probs = np.zeros((test_groups[specs[0].group].shape[0], len(specs)), dtype=np.float64)
    trained: dict[str, Any] = {}
    report: dict[str, Any] = {}
    for spec_idx, spec in enumerate(specs):
        x = train_groups[spec.group]
        x_test = test_groups[spec.group]
        fold_items = []
        print(f"base {spec_idx + 1}/{len(specs)} {spec.name} group={spec.group} dim={x.shape[1]}", flush=True)
        for fold, (fit_idx, val_idx) in enumerate(cv.split(x, y_train), start=1):
            model = clone(spec.model)
            model.fit(x[fit_idx], y_train[fit_idx])
            prob = _proba(model, x[val_idx])[:, 1]
            oof[val_idx, spec_idx] = prob
            fold_metric = _metrics(y_train[val_idx], prob, 0.5)
            fold_items.append({k: fold_metric[k] for k in ["accuracy", "macro_f1", "log_loss", "auroc", "confusion_matrix"]})
            print(
                f"  fold={fold} acc={fold_metric['accuracy']:.4f} macro_f1={fold_metric['macro_f1']:.4f} "
                f"log_loss={fold_metric['log_loss']:.4f} auroc={fold_metric['auroc']:.4f}",
                flush=True,
            )
        full_model = clone(spec.model)
        full_model.fit(x, y_train)
        test_probs[:, spec_idx] = _proba(full_model, x_test)[:, 1]
        trained[spec.name] = full_model
        oof_metric = _metrics(y_train, oof[:, spec_idx], 0.5)
        report[spec.name] = {
            "group": spec.group,
            "folds": fold_items,
            "oof_default": oof_metric,
        }
        print(
            f"  oof acc={oof_metric['accuracy']:.4f} macro_f1={oof_metric['macro_f1']:.4f} "
            f"log_loss={oof_metric['log_loss']:.4f} auroc={oof_metric['auroc']:.4f}",
            flush=True,
        )
    return oof, test_probs, report, trained


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wst_processed_744_val_small.yaml")
    parser.add_argument("--train-csv", default="splits/processed_744/train.csv")
    parser.add_argument("--test-csv", default="splits/processed_744/test.csv")
    parser.add_argument("--dino2-dir", default="outputs/processed_744/dino_vits14")
    parser.add_argument("--dino3-dir", default="outputs/processed_744/dinov3_vits16_4f")
    parser.add_argument("--videomae-dir", default="outputs/processed_744/videomae_base_k400_16f")
    parser.add_argument("--raft-dir", default="outputs/processed_744/raft_small_4pairs")
    parser.add_argument("--yolo-dir", default="outputs/processed_744/yolo_interaction")
    parser.add_argument("--badas-dir", default="outputs/processed_744/badas_window_features")
    parser.add_argument("--long-context-dir", default="")
    parser.add_argument("--long-dino-dir", default="")
    parser.add_argument("--long-train-csv", default="")
    parser.add_argument("--long-test-csv", default="")
    parser.add_argument("--out-dir", default="outputs/processed_744_strong_fusion")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--threshold-metric", default="accuracy", choices=["accuracy", "macro_f1"])
    parser.add_argument("--primary-meta", default="logreg_cv", choices=["logreg_cv", "mean", "median", "xgb_meta", "blend_logreg_xgb"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_split_csv(args.train_csv)
    test_rows = read_split_csv(args.test_csv)
    if args.long_context_dir or args.long_dino_dir:
        args.long_train_csv = args.long_train_csv or "splits/processed_744_long/train.csv"
        args.long_test_csv = args.long_test_csv or "splits/processed_744_long/test.csv"
    train_source_by_clip = _read_source_map(args.long_train_csv)
    test_source_by_clip = _read_source_map(args.long_test_csv)
    print(f"loading features train={len(train_rows)} test={len(test_rows)}", flush=True)
    train_groups, y_train, train_paths = _load_groups(train_rows, cfg, args, train_source_by_clip)
    test_groups, y_test, test_paths = _load_groups(test_rows, cfg, args, test_source_by_clip)
    print("feature dimensions:", {k: int(v.shape[1]) for k, v in train_groups.items()}, flush=True)

    specs = _base_specs(seed, use_long_context=bool(args.long_context_dir), use_long_dino=bool(args.long_dino_dir))
    oof, test_base, base_report, trained = _train_oof(train_groups, y_train, test_groups, specs, args.folds, seed)

    oof_meta = _meta_features(oof)
    test_meta = _meta_features(test_base)
    meta_models = {
        "mean": None,
        "median": None,
        "logreg_cv": make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                cv=5,
                scoring="neg_log_loss",
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "xgb_meta": _xgb(seed + 11, 160, 2, 0.035, 0.95),
    }
    ensemble_report: dict[str, Any] = {}
    meta_test_probs: dict[str, np.ndarray] = {}
    meta_oof_probs: dict[str, np.ndarray] = {}
    meta_fitted: dict[str, Any] = {}
    for name, model in meta_models.items():
        if name == "mean":
            p_oof = oof.mean(axis=1)
            p_test = test_base.mean(axis=1)
        elif name == "median":
            p_oof = np.median(oof, axis=1)
            p_test = np.median(test_base, axis=1)
        else:
            model.fit(oof_meta, y_train)
            p_oof = _proba(model, oof_meta)[:, 1]
            p_test = _proba(model, test_meta)[:, 1]
        thr_acc, oof_acc = _select_threshold(y_train, p_oof, "accuracy")
        thr_f1, oof_f1 = _select_threshold(y_train, p_oof, "macro_f1")
        test_default = _metrics(y_test, p_test, 0.5)
        test_acc_thr = _metrics(y_test, p_test, thr_acc)
        test_f1_thr = _metrics(y_test, p_test, thr_f1)
        ensemble_report[name] = {
            "oof_default": _metrics(y_train, p_oof, 0.5),
            "oof_best_accuracy": oof_acc,
            "oof_best_macro_f1": oof_f1,
            "test_default": test_default,
            "test_oof_accuracy_threshold": test_acc_thr,
            "test_oof_macro_f1_threshold": test_f1_thr,
        }
        print(
            f"ensemble {name}: oof_acc={oof_acc['accuracy']:.4f} oof_f1={oof_f1['macro_f1']:.4f} "
            f"test_acc@acc_thr={test_acc_thr['accuracy']:.4f} test_f1@f1_thr={test_f1_thr['macro_f1']:.4f} "
            f"test_default={test_default['accuracy']:.4f}",
            flush=True,
        )
        meta_oof_probs[name] = p_oof
        meta_test_probs[name] = p_test
        meta_fitted[name] = model

    if "logreg_cv" in meta_oof_probs and "xgb_meta" in meta_oof_probs:
        best_blend: dict[str, Any] | None = None
        for w in np.linspace(0.0, 1.0, 51):
            p_oof = w * meta_oof_probs["logreg_cv"] + (1.0 - w) * meta_oof_probs["xgb_meta"]
            thr, oof_f1 = _select_threshold(y_train, p_oof, "macro_f1")
            if best_blend is None or oof_f1["macro_f1"] > best_blend["oof_best_macro_f1"]["macro_f1"]:
                p_test = w * meta_test_probs["logreg_cv"] + (1.0 - w) * meta_test_probs["xgb_meta"]
                best_blend = {
                    "weight_logreg": float(w),
                    "weight_xgb": float(1.0 - w),
                    "threshold": float(thr),
                    "oof_best_macro_f1": oof_f1,
                    "test_oof_macro_f1_threshold": _metrics(y_test, p_test, thr),
                    "test_default": _metrics(y_test, p_test, 0.5),
                    "p_oof": p_oof,
                    "p_test": p_test,
                }
        assert best_blend is not None
        meta_oof_probs["blend_logreg_xgb"] = best_blend["p_oof"]
        meta_test_probs["blend_logreg_xgb"] = best_blend["p_test"]
        meta_fitted["blend_logreg_xgb"] = {"weight_logreg": best_blend["weight_logreg"], "weight_xgb": best_blend["weight_xgb"]}
        ensemble_report["blend_logreg_xgb"] = {
            k: v for k, v in best_blend.items() if k not in {"p_oof", "p_test"}
        }
        print(
            f"ensemble blend_logreg_xgb: w_logreg={best_blend['weight_logreg']:.2f} "
            f"thr={best_blend['threshold']:.3f} oof_f1={best_blend['oof_best_macro_f1']['macro_f1']:.4f} "
            f"test_acc={best_blend['test_oof_macro_f1_threshold']['accuracy']:.4f} "
            f"test_f1={best_blend['test_oof_macro_f1_threshold']['macro_f1']:.4f}",
            flush=True,
        )

    best_name = args.primary_meta
    best_test_prob = meta_test_probs[best_name]
    threshold_key = "oof_best_accuracy" if args.threshold_metric == "accuracy" else "oof_best_macro_f1"
    if best_name == "blend_logreg_xgb":
        best_thr = float(ensemble_report[best_name].get("threshold", 0.5))
    else:
        best_thr = float(ensemble_report[best_name][threshold_key]["threshold"])
    best_model = meta_fitted[best_name]
    assert best_test_prob is not None
    best_metrics = _metrics(y_test, best_test_prob, best_thr)
    y_pred = (best_test_prob >= best_thr).astype(np.int64)
    predictions = [
        {
            "path": path,
            "true_label": int(y),
            "pred_label": int(pred),
            "prob_collision": float(prob),
            "prob_near_miss": float(1.0 - prob),
            "correct": bool(pred == y),
        }
        for path, y, pred, prob in zip(test_paths, y_test, y_pred, best_test_prob)
    ]
    errors = [p for p in predictions if not p["correct"]]
    summary = {
        "target_accuracy": float(args.target_accuracy),
        "reached_target": bool(best_metrics["accuracy"] >= float(args.target_accuracy)),
        "best_ensemble_by_oof_macro_f1": best_name,
        "best_threshold_selected_on_oof": best_thr,
        "threshold_metric": args.threshold_metric,
        "best_test_metrics": best_metrics,
        "base_report": base_report,
        "ensemble_report": ensemble_report,
        "feature_dimensions": {k: int(v.shape[1]) for k, v in train_groups.items()},
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "long_context_dir": args.long_context_dir,
        "long_dino_dir": args.long_dino_dir,
        "long_train_csv": args.long_train_csv,
        "long_test_csv": args.long_test_csv,
        "folds": args.folds,
    }
    write_json(out_dir / "strong_fusion_summary.json", summary)
    write_json(out_dir / "strong_fusion_predictions.json", {"predictions": predictions})
    write_json(out_dir / "strong_fusion_errors.json", {"errors": errors})
    np.savez_compressed(
        out_dir / "strong_fusion_probabilities.npz",
        oof_base=oof.astype(np.float32),
        test_base=test_base.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        spec_names=np.asarray([s.name for s in specs]),
        **{f"oof_{k}": v.astype(np.float32) for k, v in meta_oof_probs.items()},
        **{f"test_{k}": v.astype(np.float32) for k, v in meta_test_probs.items()},
    )
    joblib.dump(
        {
            "specs": specs,
            "trained_base_models": trained,
            "meta_model_name": best_name,
            "meta_model": best_model,
            "threshold": best_thr,
            "config": vars(args),
            "label_map": {"near_miss": 0, "collision": 1},
        },
        out_dir / "strong_fusion_model.joblib",
    )
    print("BEST", json.dumps({k: best_metrics[k] for k in ["accuracy", "macro_f1", "balanced_accuracy", "auroc", "log_loss", "confusion_matrix"]}, indent=2), flush=True)
    print(f"errors={len(errors)} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
