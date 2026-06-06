#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" -m collision_contact.install_released_anchor \
  --source-dir assets/released_anchor \
  --out-dir outputs/processed_744_long_context_anchor

if [[ "${RETRAIN_ANCHOR:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m collision_contact.train_strong_fusion \
    --config configs/wst_processed_744_small.yaml \
    --train-csv splits/processed_744/train.csv \
    --test-csv splits/processed_744/test.csv \
    --dino2-dir outputs/processed_744/dino_vits14 \
    --dino3-dir outputs/processed_744/dinov3_vits16_4f \
    --videomae-dir outputs/processed_744/videomae_base_k400_16f \
    --raft-dir outputs/processed_744/raft_small_4pairs \
    --yolo-dir outputs/processed_744/yolo_interaction \
    --badas-dir outputs/processed_744/badas_window_features \
    --out-dir outputs/processed_744_strong_fusion_boosted_retrained \
    --folds 5 \
    --primary-meta blend_logreg_xgb \
    --threshold-metric macro_f1

  "$PYTHON_BIN" -m collision_contact.export_long_context_oof_experts \
    --train-csv splits/processed_744/train.csv \
    --test-csv splits/processed_744/test.csv \
    --long-train-csv splits/processed_744_long/train.csv \
    --long-test-csv splits/processed_744_long/test.csv \
    --long-context-dir outputs/processed_744/long_context_diff_1fps_yolo \
    --event-specs outputs/processed_744_strong_fusion_boosted_retrained::blend_logreg_xgb \
    --variants context_behavior \
    --models hgb \
    --hard-weight-modes none \
    --folds 5 \
    --out-dir outputs/processed_744_long_context_anchor_retrained
fi
