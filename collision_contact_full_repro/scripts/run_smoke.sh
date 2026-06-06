#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" -m collision_contact.check_data
"$PYTHON_BIN" -m collision_contact.extract_features \
  --config configs/wst_smoke.yaml \
  --workers 1 \
  --limit 1 \
  --force
"$PYTHON_BIN" -m collision_contact.extract_yolo_interaction_features \
  --config configs/wst_smoke.yaml \
  --backend ultralytics \
  --model yolov8n.pt \
  --out-dir outputs/smoke/yolo_interaction \
  --num-frames 2 \
  --image-size 320 \
  --batch 2 \
  --video-batch 1 \
  --limit 1 \
  --force
"$PYTHON_BIN" -m collision_contact.extract_object_residual_physics_features \
  --out-dir outputs/smoke/object_residual \
  --model yolov8n.pt \
  --width 240 \
  --frame-step 12 \
  --image-size 320 \
  --batch 2 \
  --limit 1 \
  --force
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
for path in [
    Path("outputs/smoke/features"),
    Path("outputs/smoke/yolo_interaction"),
    Path("outputs/smoke/object_residual"),
]:
    files = list(path.glob("*.npz"))
    if not files:
        raise SystemExit(f"no features written in {path}")
    print(f"{path}: {len(files)} npz")
PY
