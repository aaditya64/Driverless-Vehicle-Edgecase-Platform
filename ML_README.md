# Driverless Vehicle Edge-Case Platform

This repository contains the web app scaffold plus the ML/data-preparation code for classifying and organizing driverless/dashcam edge-case videos.

The current best ML result is in:

```text
collision_contact_full_repro/
```

It solves the current positive-only Nexar task:

```text
10-second event-centered positive clip -> near_miss or collision
```

Current best fixed-split result:

```text
selected=lgbm_k512/rescue_q0.371/w2.50/fixed0p5
threshold=0.5
val_accuracy=0.915966386555
val_macro_f1=0.915674603175
test_accuracy=0.939597315436
test_auroc=0.964556962025
test_confusion=[[75, 4], [5, 65]]
```

Confusion matrix label order is:

```text
[near_miss, collision]
```

So the final test result has 9 errors out of 149 test clips.

## Important: There Are Two Reproduction Paths

You should choose one path depending on what you want to do.

### Path A: Fast Reproduction of the Current Best Result

Use this if you only want to reproduce the reported `93.96%` test accuracy or retrain the final classifier from already extracted features.

This path does not require:

```text
raw Nexar videos
BADAS model download
V-JEPA2 model download
YOLO model download
CoTracker download
DINO / VideoMAE / RAFT model downloads
feature extraction from video
```

Why: the repository/package already contains the required precomputed feature assets:

```text
collision_contact_full_repro/outputs/processed_744/features/
collision_contact_full_repro/outputs/processed_744/object_cotracker_dynamics_yolov8s_32f_w384_20260524/
collision_contact_full_repro/analysis/impact_diagnostics_20260522/
collision_contact_full_repro/assets/released_anchor/
```

Run:

```bash
cd collision_contact_full_repro

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

On macOS, install OpenMP for LightGBM/XGBoost:

```bash
brew install libomp
```

Then run:

```bash
PYTHON=.venv/bin/python bash scripts/run_02_train_anchor.sh
PYTHON=.venv/bin/python bash scripts/run_03_train_model.sh
PYTHON=.venv/bin/python bash scripts/run_04_report.sh
```

Expected output:

```text
selected=lgbm_k512/rescue_q0.371/w2.50/fixed0p5
threshold=0.5
val_accuracy=0.915966386555
val_macro_f1=0.915674603175
test_accuracy=0.939597315436
test_auroc=0.964556962025
test_confusion=[[75, 4], [5, 65]]
```

Do not run these commands for Path A:

```bash
bash scripts/run_00_check_and_prepare.sh
bash scripts/run_01_extract_features.sh
```

Those commands are for Path B and expect raw videos and model downloads.

### Path B: Full Rebuild From Raw Videos

Use this if you want to reproduce the full method from original videos, regenerate all features, or modify the feature extraction pipeline.

This path requires:

```text
raw Nexar videos
generated 10-second event clips
Hugging Face access/token for gated models
BADAS-Open
V-JEPA2
YOLOv8n / YOLOv8s
CoTracker
DINO / DINOv3
VideoMAE
RAFT
ffmpeg / ffprobe
```

Recommended hardware:

```text
CUDA GPU preferred
Apple Silicon MPS possible but slower, especially for CoTracker
CPU-only is not recommended for full extraction
```

Install environment:

```bash
cd collision_contact_full_repro

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Install system dependencies:

```bash
brew install ffmpeg
brew install libomp
```

Log in to Hugging Face if rebuilding gated-model features:

```bash
huggingface-cli login
export HF_TOKEN=your_huggingface_token
```

Place raw/processed data in the expected structure:

```text
collision_contact_full_repro/data/nexar_collision_prediction/train/positive/*.mp4
collision_contact_full_repro/data/nexar_collision_prediction/test/positive/*.mp4
collision_contact_full_repro/data/processed_positive/processed/event_clips/nexar/train/positive/*.mp4
collision_contact_full_repro/data/processed_positive/processed/event_clips/nexar/test/positive/*.mp4
```

The split CSVs already exist in:

```text
collision_contact_full_repro/splits/processed_744/
collision_contact_full_repro/splits/processed_744_long/
```

Run data check and model-asset preparation:

```bash
PYTHON=.venv/bin/python bash scripts/run_00_check_and_prepare.sh
```

Run a small smoke test:

```bash
PYTHON=.venv/bin/python bash scripts/run_smoke.sh
```

Run full feature extraction:

```bash
PYTHON=.venv/bin/python bash scripts/run_01_extract_features.sh
```

For Apple Silicon / lower memory:

```bash
COTRACKER_DEVICE=mps \
DINO_BATCH=2 \
RAFT_BATCH=1 \
YOLO_BATCH=4 \
BADAS_BATCH=1 \
PYTHON=.venv/bin/python \
bash scripts/run_01_extract_features.sh
```

For quick debugging only:

```bash
LIMIT=8 PYTHON=.venv/bin/python bash scripts/run_01_extract_features.sh
```

Important: `LIMIT` mode is not enough for final training because full object metrics are not exported in limited mode.

After full extraction:

```bash
PYTHON=.venv/bin/python bash scripts/run_02_train_anchor.sh
PYTHON=.venv/bin/python bash scripts/run_03_train_model.sh
PYTHON=.venv/bin/python bash scripts/run_04_report.sh
```

## What the Final Model Uses

The final classifier is not only BADAS. It is a feature-fusion/rescue pipeline:

1. Camera/global-motion and wavelet features.
2. Object residual physics metrics.
3. CoTracker object dynamics features.
4. A released long-context anchor.
5. A LightGBM rescue-fusion classifier selected on validation.

Key files:

```text
collision_contact_full_repro/collision_contact/train_val_selected_deep_rescue.py
collision_contact_full_repro/scripts/run_03_train_model.sh
collision_contact_full_repro/outputs/collision_contact_model/val_selected_deep_rescue_summary.json
collision_contact_full_repro/outputs/collision_contact_model/val_selected_deep_rescue_models.joblib
```

Feature dimension:

```text
4063
```

Final selected model:

```text
lgbm_k512/rescue_q0.371/w2.50/fixed0p5
```

## Repository Structure

```text
frontend/                         existing Vite/React frontend
backend/                          existing backend scaffold
scripts/                          earlier BADAS/data-prep scripts
doc/                              project notes and experiment documents
ML_PIPELINE_CONTEXT_AND_PROGRESS.md
collision_contact_full_repro/     current best reproducible ML package
```

Important `collision_contact_full_repro/` structure:

```text
collision_contact_full_repro/
  README.md
  requirements.txt
  collision_contact/
  scripts/
  configs/
  splits/
  assets/released_anchor/
  legacy/
  outputs/processed_744/features/
  outputs/processed_744/object_cotracker_dynamics_yolov8s_32f_w384_20260524/
  outputs/processed_744_long_context_anchor/
  outputs/collision_contact_model/
  analysis/impact_diagnostics_20260522/
```

## Outputs for Full-Stack

The easiest file to consume today is:

```text
collision_contact_full_repro/outputs/collision_contact_model/val_selected_deep_rescue_predictions.json
```

Each prediction has:

```text
idx
path
true_label
prob_collision
pred_label
correct
```

Label mapping:

```text
pred_label=0 -> near_miss
pred_label=1 -> collision
```

The current repo does not yet have a production single-video inference API. To support backend inference on a new uploaded video, we still need a wrapper that:

1. Cuts or receives a 10-second event-centered clip.
2. Extracts the same feature stack.
3. Builds the 4063-dimensional feature vector in the saved order.
4. Loads `val_selected_deep_rescue_models.joblib`.
5. Applies the selected rescue-fusion rule and threshold.
6. Returns `prob_collision`, `pred_label`, and metadata.

## Earlier BADAS/Data-Preparation Scripts

The root-level `scripts/` folder contains earlier pipeline pieces:

```text
scripts/download_badas_open.py
scripts/inference_badas.py
scripts/create_event_clips.py
scripts/create_saved_event_clips.py
scripts/extract_badas_window_features.py
scripts/train_badas_outcome_head.py
```

These were used during earlier experiments and baseline building. The current best model is under `collision_contact_full_repro/`.

## Large File / GitHub Notes

Use Git LFS for feature and model artifacts:

```bash
git lfs install
git lfs track "*.npz"
git lfs track "*.joblib"
git add .gitattributes
```

Do not commit:

```text
.venv/
.cache/
external/co-tracker/
models/BADAS-Open/
raw Nexar videos
generated raw event clips unless the team explicitly wants them in external storage
```

The precomputed feature folder is large:

```text
collision_contact_full_repro/outputs/processed_744/features/   about 1.1 GB
```

So LFS or a release asset is strongly recommended.

## More Detail

Detailed ML status and history:

```text
ML_PIPELINE_CONTEXT_AND_PROGRESS.md
```

Detailed reproduction package documentation:

```text
collision_contact_full_repro/README.md
```
