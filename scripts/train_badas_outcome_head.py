#!/usr/bin/env python3
"""Train a small collision-vs-near-miss head on cached BADAS window features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clip_manifests"
    / "nexar_train_positive_event_clips.csv"
)
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "outcome_head"

LABEL_TO_TARGET = {
    "near_miss": 0.0,
    "collision": 1.0,
}
IGNORED_LABELS = {
    "",
    "needs_review",
    "ambiguous",
    "uncertain",
    "unknown",
    "discard",
    "safe",
}


@dataclass(frozen=True)
class FeatureSample:
    clip_id: str
    feature_path: Path
    label: str
    target: float


class FeatureNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, arrays: list[np.ndarray]) -> "FeatureNormalizer":
        stacked = np.concatenate(arrays, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


class OutcomeFeatureDataset(Dataset):
    def __init__(
        self,
        samples: list[FeatureSample],
        add_risk: bool,
        add_time: bool,
        normalizer: FeatureNormalizer | None = None,
    ):
        self.samples = samples
        self.add_risk = add_risk
        self.add_time = add_time
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.samples)

    def input_dim(self) -> int:
        x = self._load_features(self.samples[0].feature_path)
        return int(x.shape[1])

    def raw_arrays(self) -> list[np.ndarray]:
        return [self._load_features(sample.feature_path) for sample in self.samples]

    def _load_features(self, path: Path) -> np.ndarray:
        with np.load(path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32)
            extras = []

            if self.add_risk:
                risk_scores = data["risk_scores"].astype(np.float32)
                extras.append(risk_scores[:, None])

            if self.add_time:
                target_time_sec = data["target_time_sec"].astype(np.float32)
                span = float(target_time_sec.max() - target_time_sec.min())
                if span < 1e-6:
                    time_feature = np.zeros_like(target_time_sec, dtype=np.float32)
                else:
                    center = float(target_time_sec.mean())
                    time_feature = (target_time_sec - center) / span
                extras.append(time_feature[:, None].astype(np.float32))

            if extras:
                features = np.concatenate([features, *extras], axis=1)

        if self.normalizer is not None:
            features = self.normalizer.transform(features)
        return features.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        features = self._load_features(sample.feature_path)
        return {
            "features": torch.from_numpy(features),
            "target": torch.tensor(sample.target, dtype=torch.float32),
            "clip_id": sample.clip_id,
            "label": sample.label,
        }


class TemporalConvOutcomeHead(nn.Module):
    """Small temporal head for q=P(collision | risky)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        conv_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so temporal length is preserved.")

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        conv_blocks = []
        padding = kernel_size // 2
        for _ in range(conv_layers):
            conv_blocks.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.temporal_conv = nn.Sequential(*conv_blocks)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, windows, channels]
        x = self.input_projection(x)
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)

        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(pooled).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a conditional outcome head on cached BADAS window features. "
            "The model outputs q=P(collision | risky)."
        )
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument(
        "--labels-csv",
        default=None,
        type=Path,
        help=(
            "Optional CSV with clip_id and label column. Values here override "
            "manifest labels, useful while annotators work in a separate file."
        ),
    )
    parser.add_argument("--features-dir", default=DEFAULT_FEATURES_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--label-column", default="core_label")
    parser.add_argument("--clip-id-column", default="clip_id")
    parser.add_argument("--train-fraction", default=0.70, type=float)
    parser.add_argument("--val-fraction", default=0.15, type=float)
    parser.add_argument("--test-fraction", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--patience", default=8, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument(
        "--pos-weight",
        default="auto",
        help="Use 'auto', 'none', or a numeric BCE positive-class weight.",
    )

    parser.add_argument("--hidden-dim", default=256, type=int)
    parser.add_argument("--conv-layers", default=2, type=int)
    parser.add_argument("--kernel-size", default=3, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--no-risk", action="store_true", help="Do not append risk score.")
    parser.add_argument("--no-time", action="store_true", help="Do not append time feature.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if device_arg == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_labels(path: Path, clip_id_column: str, label_column: str) -> dict[str, str]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Label CSV does not exist: {path}")

    labels = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if clip_id_column not in reader.fieldnames:
            raise ValueError(f"Missing clip id column '{clip_id_column}' in {path}")
        if label_column not in reader.fieldnames:
            raise ValueError(f"Missing label column '{label_column}' in {path}")

        for row in reader:
            clip_id = str(row.get(clip_id_column, "")).strip()
            label = normalize_label(row.get(label_column, ""))
            if clip_id:
                labels[clip_id] = label
    return labels


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def build_samples(
    manifest: Path,
    labels_csv: Path | None,
    features_dir: Path,
    clip_id_column: str,
    label_column: str,
) -> tuple[list[FeatureSample], dict[str, int]]:
    labels = read_labels(manifest, clip_id_column, label_column)
    if labels_csv is not None:
        labels.update(read_labels(labels_csv, clip_id_column, label_column))

    features_dir = features_dir.expanduser().resolve()
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory does not exist: {features_dir}")

    samples = []
    skipped = {
        "missing_feature": 0,
        "ignored_label": 0,
        "unknown_label": 0,
    }

    for clip_id, label in sorted(labels.items()):
        feature_path = features_dir / f"{clip_id}.npz"
        if not feature_path.exists():
            skipped["missing_feature"] += 1
            continue

        if label in LABEL_TO_TARGET:
            samples.append(
                FeatureSample(
                    clip_id=clip_id,
                    feature_path=feature_path,
                    label=label,
                    target=LABEL_TO_TARGET[label],
                )
            )
        elif label in IGNORED_LABELS:
            skipped["ignored_label"] += 1
        else:
            skipped["unknown_label"] += 1

    return samples, skipped


def stratified_train_val_test_split(
    samples: list[FeatureSample],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[FeatureSample], list[FeatureSample], list[FeatureSample]]:
    fractions = {
        "train": train_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }
    if any(value <= 0.0 for value in fractions.values()):
        raise ValueError("Train/val/test fractions must all be > 0.")
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "Train/val/test fractions must sum to 1.0. "
            f"Got {fractions} with sum={sum(fractions.values())}."
        )

    by_label: dict[str, list[FeatureSample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    if set(by_label) != set(LABEL_TO_TARGET):
        raise ValueError(
            "Training requires at least one near_miss and one collision sample. "
            f"Observed labels: {sorted(by_label)}"
        )

    rng = random.Random(seed)
    train_samples = []
    val_samples = []
    test_samples = []

    for label, group in sorted(by_label.items()):
        rng.shuffle(group)
        if len(group) < 3:
            raise ValueError(f"Label '{label}' has only {len(group)} sample(s). Need at least 3.")

        test_count = max(1, int(round(len(group) * test_fraction)))
        val_count = max(1, int(round(len(group) * val_fraction)))

        if test_count + val_count >= len(group):
            test_count = 1
            val_count = 1

        train_count = len(group) - val_count - test_count
        if train_count < 1:
            raise ValueError(f"Label '{label}' does not have enough samples for train/val/test split.")

        train_samples.extend(group[:train_count])
        val_samples.extend(group[train_count : train_count + val_count])
        test_samples.extend(group[train_count + val_count :])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return train_samples, val_samples, test_samples


def make_pos_weight(samples: list[FeatureSample], pos_weight_arg: str, device: torch.device) -> torch.Tensor | None:
    if pos_weight_arg == "none":
        return None
    if pos_weight_arg != "auto":
        value = float(pos_weight_arg)
        return torch.tensor([value], dtype=torch.float32, device=device)

    positives = sum(1 for sample in samples if sample.target == 1.0)
    negatives = sum(1 for sample in samples if sample.target == 0.0)
    if positives == 0:
        raise ValueError("Cannot compute pos_weight because there are no collision samples.")
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def collate_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, Any]:
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "clip_id": [item["clip_id"] for item in batch],
        "label": [item["label"] for item in batch],
    }


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    y_pred = (y_score >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    accuracy = safe_div(tp + tn, len(y_true))

    return {
        "accuracy": accuracy,
        "precision_collision": precision,
        "recall_collision": recall,
        "f1_collision": f1,
        "auroc": binary_auroc(y_true, y_score),
        "average_precision": average_precision(y_true, y_score),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    rank_sum_pos = float(ranks[positives].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = int((y_true == 1).sum())
    if positives == 0:
        return None

    order = np.argsort(-y_score)
    sorted_true = y_true[order]
    hit_count = 0
    precision_sum = 0.0
    for rank, target in enumerate(sorted_true, start=1):
        if target == 1:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / positives


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_scores = []
    all_clip_ids = []

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["target"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(features)
            loss = criterion(logits, targets)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        all_targets.append(targets.detach().cpu().numpy())
        all_scores.append(torch.sigmoid(logits.detach()).cpu().numpy())
        all_clip_ids.extend(batch["clip_id"])

    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, np.concatenate(all_targets), np.concatenate(all_scores), all_clip_ids


def prediction_rows(
    clip_ids: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for clip_id, target, score in zip(clip_ids, y_true, y_score):
        target_int = int(target)
        rows.append(
            {
                "clip_id": clip_id,
                "target": target_int,
                "label": "collision" if target_int == 1 else "near_miss",
                "collision_probability": float(score),
                "predicted_label": "collision" if score >= threshold else "near_miss",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(val) for val in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, skipped = build_samples(
        manifest=args.manifest,
        labels_csv=args.labels_csv,
        features_dir=args.features_dir,
        clip_id_column=args.clip_id_column,
        label_column=args.label_column,
    )

    label_counts = {
        label: sum(1 for sample in samples if sample.label == label)
        for label in LABEL_TO_TARGET
    }
    print(f"Usable labelled feature samples: {len(samples)}")
    print(f"Label counts: {label_counts}")
    print(f"Skipped: {skipped}")

    if len(samples) < 6:
        raise RuntimeError(
            "Not enough labelled feature samples to train. Need labelled "
            "`near_miss` and `collision` rows in the manifest or --labels-csv."
        )

    train_samples, val_samples, test_samples = stratified_train_val_test_split(
        samples,
        args.train_fraction,
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )
    print(f"Train/val/test: {len(train_samples)}/{len(val_samples)}/{len(test_samples)}")

    add_risk = not args.no_risk
    add_time = not args.no_time
    train_dataset_for_norm = OutcomeFeatureDataset(train_samples, add_risk, add_time)
    normalizer = FeatureNormalizer.fit(train_dataset_for_norm.raw_arrays())
    train_dataset = OutcomeFeatureDataset(train_samples, add_risk, add_time, normalizer)
    val_dataset = OutcomeFeatureDataset(val_samples, add_risk, add_time, normalizer)
    test_dataset = OutcomeFeatureDataset(test_samples, add_risk, add_time, normalizer)

    input_dim = train_dataset.input_dim()
    model = TemporalConvOutcomeHead(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        conv_layers=args.conv_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    pos_weight = make_pos_weight(train_samples, args.pos_weight, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    best_ap = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []
    best_path = output_dir / "best_outcome_head.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_true, train_score, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        val_loss, val_true, val_score, val_clip_ids = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        train_metrics = binary_metrics(train_true, train_score, args.threshold)
        val_metrics = binary_metrics(val_true, val_score, args.threshold)
        val_ap = val_metrics["average_precision"]
        comparable_ap = -1.0 if val_ap is None else float(val_ap)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history_rows.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_ap={val_ap if val_ap is not None else 'n/a'} "
            f"val_f1={val_metrics['f1_collision'] if val_metrics['f1_collision'] is not None else 'n/a'} "
            f"tp/fp/fn={val_metrics['tp']}/{val_metrics['fp']}/{val_metrics['fn']}",
            flush=True,
        )

        if comparable_ap > best_ap:
            best_ap = comparable_ap
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "conv_layers": args.conv_layers,
                    "kernel_size": args.kernel_size,
                    "dropout": args.dropout,
                    "add_risk": add_risk,
                    "add_time": add_time,
                    "normalizer": normalizer.to_dict(),
                    "label_to_target": LABEL_TO_TARGET,
                    "threshold": args.threshold,
                    "best_epoch": best_epoch,
                    "best_val_average_precision": best_ap,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs. Best epoch: {best_epoch}")
            break

    history_path = output_dir / "training_history.csv"
    all_metric_keys = sorted({key for row in history_rows for key in row})
    write_csv(history_path, [json_safe(row) for row in history_rows], all_metric_keys)

    split_rows = (
        [
            {"split": "train", "clip_id": sample.clip_id, "label": sample.label}
            for sample in train_samples
        ]
        + [
            {"split": "val", "clip_id": sample.clip_id, "label": sample.label}
            for sample in val_samples
        ]
        + [
            {"split": "test", "clip_id": sample.clip_id, "label": sample.label}
            for sample in test_samples
        ]
    )
    write_csv(output_dir / "train_val_test_split.csv", split_rows, ["split", "clip_id", "label"])

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_true, val_score, val_clip_ids = run_epoch(model, val_loader, criterion, device)
    test_loss, test_true, test_score, test_clip_ids = run_epoch(model, test_loader, criterion, device)

    val_metrics = binary_metrics(val_true, val_score, args.threshold)
    test_metrics = binary_metrics(test_true, test_score, args.threshold)

    write_csv(
        output_dir / "val_predictions.csv",
        prediction_rows(val_clip_ids, val_true, val_score, args.threshold),
        ["clip_id", "target", "label", "collision_probability", "predicted_label"],
    )
    write_csv(
        output_dir / "test_predictions.csv",
        prediction_rows(test_clip_ids, test_true, test_score, args.threshold),
        ["clip_id", "target", "label", "collision_probability", "predicted_label"],
    )

    final_metrics = {
        "best_epoch": best_epoch,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "val": val_metrics,
        "test": test_metrics,
    }
    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(final_metrics), f, indent=2)
        f.write("\n")

    config = {
        "manifest": args.manifest.expanduser().resolve(),
        "labels_csv": args.labels_csv.expanduser().resolve() if args.labels_csv else None,
        "features_dir": args.features_dir.expanduser().resolve(),
        "output_dir": output_dir,
        "device": str(device),
        "usable_samples": len(samples),
        "label_counts": label_counts,
        "skipped": skipped,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "input_dim": input_dim,
        "add_risk": add_risk,
        "add_time": add_time,
        "pos_weight": None if pos_weight is None else float(pos_weight.detach().cpu().item()),
        "best_epoch": best_epoch,
        "best_val_average_precision": best_ap,
        "args": vars(args),
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(config), f, indent=2)
        f.write("\n")

    print(f"Wrote best model: {best_path}")
    print(f"Wrote history: {history_path}")
    print(f"Wrote final metrics: {output_dir / 'final_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
