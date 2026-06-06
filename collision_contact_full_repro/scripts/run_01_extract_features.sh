#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

PYTHON_BIN="${PYTHON:-python}"
WORKERS="${WORKERS:-1}"
DINO_BATCH="${DINO_BATCH:-8}"
VIDEO_BATCH="${VIDEO_BATCH:-1}"
YOLO_BATCH="${YOLO_BATCH:-16}"
RAFT_BATCH="${RAFT_BATCH:-4}"
BADAS_BATCH="${BADAS_BATCH:-3}"
COTRACKER_DEVICE="${COTRACKER_DEVICE:-cuda}"

limit_args=()
if [[ "${LIMIT:-0}" != "0" ]]; then
  limit_args=(--limit "$LIMIT")
fi

force_args=()
overwrite_args=()
if [[ "${FORCE:-0}" == "1" ]]; then
  force_args=(--force)
  overwrite_args=(--overwrite)
fi

"$PYTHON_BIN" -m collision_contact.extract_features \
  --config configs/wst_processed_744_small.yaml \
  --workers "$WORKERS" \
  "${limit_args[@]}" \
  "${force_args[@]}"

"$PYTHON_BIN" -m collision_contact.extract_dino_features \
  --config configs/wst_processed_744_small.yaml \
  --out-dir outputs/processed_744/dino_vits14 \
  --model vit_small_patch14_dinov2.lvd142m \
  --num-frames 16 \
  --image-size 518 \
  --batch-size "$DINO_BATCH" \
  "${limit_args[@]}" \
  "${force_args[@]}"

"$PYTHON_BIN" -m collision_contact.extract_dino_features \
  --config configs/wst_processed_744_small.yaml \
  --out-dir outputs/processed_744/dinov3_vits16_4f \
  --model vit_small_patch16_dinov3 \
  --num-frames 4 \
  --image-size 256 \
  --batch-size "$DINO_BATCH" \
  "${limit_args[@]}" \
  "${force_args[@]}"

"$PYTHON_BIN" -m collision_contact.extract_videomae_features \
  --config configs/wst_processed_744_small.yaml \
  --out-dir outputs/processed_744/videomae_base_k400_16f \
  --model MCG-NJU/videomae-base-finetuned-kinetics \
  --num-frames 16 \
  "${limit_args[@]}" \
  "${force_args[@]}"

"$PYTHON_BIN" -m collision_contact.extract_raft_features \
  --config configs/wst_processed_744_small.yaml \
  --out-dir outputs/processed_744/raft_small_4pairs \
  --num-pairs 4 \
  --pair-gap 2 \
  --width 384 \
  --batch-size "$RAFT_BATCH" \
  "${limit_args[@]}" \
  "${force_args[@]}"

"$PYTHON_BIN" -m collision_contact.extract_yolo_interaction_features \
  --config configs/wst_processed_744_small.yaml \
  --backend ultralytics \
  --model yolov8n.pt \
  --out-dir outputs/processed_744/yolo_interaction \
  --num-frames 24 \
  --image-size 640 \
  --batch "$YOLO_BATCH" \
  --video-batch "$VIDEO_BATCH" \
  "${limit_args[@]}" \
  "${force_args[@]}"

"$PYTHON_BIN" -m collision_contact.prepare_badas_manifest \
  --out-dir outputs/processed_744/badas_manifests

"$PYTHON_BIN" legacy/extract_badas_window_features.py \
  --manifest outputs/processed_744/badas_manifests/all_unique.csv \
  --output-dir outputs/processed_744/badas_window_features \
  --model-dir models/BADAS-Open \
  --device auto \
  --window-batch-size "$BADAS_BATCH" \
  "${limit_args[@]}" \
  "${overwrite_args[@]}"

"$PYTHON_BIN" -m collision_contact.extract_long_context_features \
  --out-dir outputs/processed_744/long_context_diff_1fps_yolo \
  --sample-fps 1.0 \
  --width 480 \
  --motion-mode diff \
  --use-yolo \
  --yolo-model yolov8n.pt \
  --yolo-batch "$YOLO_BATCH" \
  "${limit_args[@]}" \
  "${force_args[@]}"

"$PYTHON_BIN" -m collision_contact.extract_object_residual_physics_features \
  --out-dir outputs/processed_744/object_residual_physics_w320_s4 \
  --model yolov8n.pt \
  --width 320 \
  --frame-step 4 \
  --image-size 640 \
  --batch "$YOLO_BATCH" \
  "${limit_args[@]}" \
  "${force_args[@]}"

if [[ "${LIMIT:-0}" == "0" ]]; then
  "$PYTHON_BIN" -m collision_contact.export_object_metrics \
    --feature-dir outputs/processed_744/object_residual_physics_w320_s4 \
    --out-dir analysis/impact_diagnostics_20260522
else
  echo "limited extraction: skipped full object metrics export"
fi

"$PYTHON_BIN" -m collision_contact.extract_object_cotracker_dynamics_features \
  --out-dir outputs/processed_744/object_cotracker_dynamics_yolov8s_32f_w384_20260524 \
  --cotracker-repo external/co-tracker \
  --yolo-model yolov8s.pt \
  --num-frames 32 \
  --width 384 \
  --center-seconds 6.0 \
  --device "$COTRACKER_DEVICE" \
  "${limit_args[@]}" \
  "${force_args[@]}"
