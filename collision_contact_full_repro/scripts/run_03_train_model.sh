#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" -m collision_contact.train_val_selected_deep_rescue \
  --train-inner-csv splits/processed_744/train_inner.csv \
  --val-csv splits/processed_744/val.csv \
  --train-csv splits/processed_744/train.csv \
  --test-csv splits/processed_744/test.csv \
  --feature-dir outputs/processed_744/features \
  --object-metrics-csv analysis/impact_diagnostics_20260522/object_physics_train_metrics.csv \
  --object-metrics-csv analysis/impact_diagnostics_20260522/object_physics_test_metrics.csv \
  --cotracker-feature-dir outputs/processed_744/object_cotracker_dynamics_yolov8s_32f_w384_20260524 \
  --main-prob outputs/processed_744_long_context_anchor/strong_fusion_probabilities.npz \
  --main-expert-name processed_744_strong_fusion_boosted:blend_logreg_xgb/context_behavior/none/hgb/context \
  --seed 20260524 \
  --hard-weight-scale 1.5 \
  --threshold-metric macro_f1 \
  --select-threshold-min 0.45 \
  --select-threshold-max 0.55 \
  --out-dir outputs/collision_contact_model
