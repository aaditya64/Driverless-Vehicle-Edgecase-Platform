# Collision Contact Reproduction Pipeline

This folder contains the current best collision-vs-near-miss reproduction package for the Nexar positive clips. It is intended for teammates who were not involved in the ML work and need to reproduce the reported result or use the trained model artifacts.

Current fixed-split result:

```text
selected=lgbm_k512/rescue_q0.371/w2.50/fixed0p5
threshold=0.5
val_accuracy=0.915966386555
val_macro_f1=0.915674603175
test_accuracy=0.939597315436
test_auroc=0.964556962025
test_confusion=[[75, 4], [5, 65]]
```

The confusion matrix uses label order `[near_miss, collision]`. On the 149-video test set, the selected model gets 75 near-miss clips and 65 collision clips correct, with 9 total errors.

## What This Model Does

Input task:

```text
10-second event-centered Nexar positive clip -> near_miss or collision
```

This model is not a full `safe / near_miss / collision` pipeline yet. The safe/risky gate was handled in earlier BADAS work. This package focuses on the hard positive-only boundary:

```text
near_miss vs physical-contact collision
```

The final method combines:

1. Camera-shake/global-motion features.
   - Estimate frame-to-frame camera motion.
   - Build motion channels such as displacement, velocity, acceleration, jerk, residual motion, fit error, and shake energy.
   - Apply wavelet transforms to capture short transient vibration/contact patterns.

2. Object residual physics.
   - Use YOLO detections around the event window.
   - Export object-energy, object-shift, residual-motion, and peak-time metrics.

3. CoTracker object dynamics.
   - Use YOLOv8s to choose object regions.
   - Use CoTracker to track object points over 32 frames at width 384.
   - Summarize contact-like object motion into fixed-length features.

4. Released long-context anchor.
   - A precomputed calibrated probability asset from the stronger long-context fusion pipeline.
   - Stored at `assets/released_anchor/strong_fusion_probabilities.npz`.
   - Installed into `outputs/processed_744_long_context_anchor/` by `scripts/run_02_train_anchor.sh`.

5. Final rescue fusion.
   - Train several compact tabular heads on `train_inner`.
   - Evaluate fusion/rescue candidates on `val`.
   - Select the candidate by validation macro-F1 under a threshold constraint.
   - Final selected candidate is `lgbm_k512/rescue_q0.371/w2.50/fixed0p5`.

## Data Split

The package uses a fixed cleaned 744-sample Nexar positive set:

```text
total usable samples: 744
train: 595
  train_inner: 476
  val: 119
test: 149
```

Class counts:

```text
train:       collision=280, near_miss=315
train_inner: collision=224, near_miss=252
val:          collision=56,  near_miss=63
test:         collision=70,  near_miss=79
```

Short split CSV columns:

```text
path,label,label_name
```

Long-context split CSV columns:

```text
path,source_path,label,label_name,clip_id,source_file_name,manual_label,
video_duration,time_of_event,time_of_alert,clip_start_time,clip_end_time,
event_center_time,light_conditions,weather,scene
```

Label encoding:

```text
0 = near_miss
1 = collision
```

## Fast Reproduction: Use Precomputed Features

This is the recommended path for full-stack teammates. It reproduces the reported result without re-extracting visual features.

Fast reproduction requires these precomputed assets to already be present:

```text
outputs/processed_744/features/
outputs/processed_744/object_cotracker_dynamics_yolov8s_32f_w384_20260524/
analysis/impact_diagnostics_20260522/
assets/released_anchor/
```

Raw `.mp4` videos are not needed for this fast path because the model trains from the saved feature files. Do not run `scripts/run_00_check_and_prepare.sh` if the raw videos are not present; that script validates video paths and will fail without the dataset.

### 1. Create Python Environment

From inside this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

On macOS, install OpenMP runtime for LightGBM/XGBoost:

```bash
brew install libomp
```

If `ffmpeg` is not installed and you plan to rebuild features from videos later:

```bash
brew install ffmpeg
```

### 2. Verify Key Python Imports

```bash
python - <<'PY'
for name in ["numpy", "sklearn", "lightgbm", "xgboost", "catboost"]:
    mod = __import__(name)
    print(name, getattr(mod, "__version__", "OK"))
PY
```

`lightgbm` and `xgboost` must import successfully. If they fail on macOS with an OpenMP error, install `libomp` and retry.

### 3. Install Released Anchor

```bash
PYTHON=.venv/bin/python bash scripts/run_02_train_anchor.sh
```

This copies the released anchor from:

```text
assets/released_anchor/
```

to:

```text
outputs/processed_744_long_context_anchor/
```

### 4. Train the Final Head

```bash
PYTHON=.venv/bin/python bash scripts/run_03_train_model.sh
```

Expected final line:

```text
BEST {
  "name": "lgbm_k512/rescue_q0.371/w2.50/fixed0p5",
  ...
}
errors=9 out_dir=outputs/collision_contact_model
```

### 5. Print Report

```bash
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

## Full Rebuild From Raw Videos

This path rebuilds features from the original videos. It is slower and requires model downloads, raw data, and a GPU or a patient machine.

Use this only if:

- precomputed features are missing,
- feature extraction code changed,
- the split changed,
- labels changed, or
- new clips need to be added.

### 1. Download/Place Raw Data

The raw Nexar dataset is not included in git because of size and licensing. Download it from the team-approved source and place it with this structure:

```text
data/nexar_collision_prediction/train/positive/*.mp4
data/nexar_collision_prediction/test/positive/*.mp4
data/nexar_collision_prediction/train/positive/metadata.csv
data/nexar_collision_prediction/test/positive/metadata.csv
```

The fixed split CSVs in this package also reference generated 10-second clips:

```text
data/processed_positive/processed/event_clips/nexar/train/positive/*.mp4
data/processed_positive/processed/event_clips/nexar/test/positive/*.mp4
```

If your local preprocessing writes clips under `data/processed/event_clips/...`, copy or symlink them into the `data/processed_positive/processed/event_clips/...` layout before running the check script.

### 2. Download External Model Assets

Some models are public and some are gated. For gated Hugging Face models, first log in with an account that has access:

```bash
huggingface-cli login
export HF_TOKEN=your_huggingface_token
```

Then run:

```bash
PYTHON=.venv/bin/python bash scripts/run_00_check_and_prepare.sh
```

This script does two things:

1. Validates paths in the split CSVs.
2. Downloads or prepares external assets.

Downloaded/prepared assets include:

```text
nexar-ai/badas-open
facebook/vjepa2-vitl-fpc16-256-ssv2
yolov8n.pt
yolov8s.pt
facebookresearch/co-tracker
```

Other feature extractors may also cache models on first use:

```text
vit_small_patch14_dinov2.lvd142m
vit_small_patch16_dinov3
MCG-NJU/videomae-base-finetuned-kinetics
torchvision RAFT Small C_T_V2
```

Do not commit downloaded model weights, `.cache/`, or `external/co-tracker/` unless the team explicitly decides to vendor them.

### 3. Run a Smoke Test

```bash
PYTHON=.venv/bin/python bash scripts/run_smoke.sh
```

The smoke test checks data paths and extracts a tiny number of features. It is not a full training run.

### 4. Extract Full Feature Stack

Default full extraction:

```bash
PYTHON=.venv/bin/python bash scripts/run_01_extract_features.sh
```

On Apple Silicon, use MPS where possible and reduce batch sizes:

```bash
COTRACKER_DEVICE=mps \
DINO_BATCH=2 \
RAFT_BATCH=1 \
YOLO_BATCH=4 \
BADAS_BATCH=1 \
PYTHON=.venv/bin/python \
bash scripts/run_01_extract_features.sh
```

On CUDA machines:

```bash
COTRACKER_DEVICE=cuda PYTHON=.venv/bin/python bash scripts/run_01_extract_features.sh
```

Useful controls:

```bash
# Limit to a small number of videos for debugging.
LIMIT=8 PYTHON=.venv/bin/python bash scripts/run_01_extract_features.sh

# Recompute files even if outputs already exist.
FORCE=1 PYTHON=.venv/bin/python bash scripts/run_01_extract_features.sh
```

Important note: when `LIMIT` is used, the script intentionally skips full object-metrics export. Full training requires the complete object metrics CSVs in:

```text
analysis/impact_diagnostics_20260522/
```

### 5. Install or Retrain Long-Context Anchor

For normal reproduction, install the released anchor:

```bash
PYTHON=.venv/bin/python bash scripts/run_02_train_anchor.sh
```

To retrain the anchor from extracted DINO/DINOv3/VideoMAE/RAFT/YOLO/BADAS/long-context features:

```bash
RETRAIN_ANCHOR=1 PYTHON=.venv/bin/python bash scripts/run_02_train_anchor.sh
```

The current reported result uses the released anchor, not a newly retrained anchor.

### 6. Train and Report

```bash
PYTHON=.venv/bin/python bash scripts/run_03_train_model.sh
PYTHON=.venv/bin/python bash scripts/run_04_report.sh
```

## Output Artifacts

After successful training:

```text
outputs/collision_contact_model/val_selected_deep_rescue_models.joblib
```

contains:

```text
models          fitted sklearn/LightGBM/XGBoost candidate models
feature_names   4063 feature names
selected        selected fusion candidate and threshold
args            training arguments
```

The selected model metadata is also saved in:

```text
outputs/collision_contact_model/val_selected_deep_rescue_summary.json
```

Test predictions:

```text
outputs/collision_contact_model/val_selected_deep_rescue_predictions.json
```

Test errors:

```text
outputs/collision_contact_model/val_selected_deep_rescue_errors.json
```

Current test errors:

```text
collision -> near_miss:
  nexar_train_positive_00016
  nexar_train_positive_00175
  nexar_train_positive_00308
  nexar_train_positive_00517
  nexar_train_positive_00529

near_miss -> collision:
  nexar_train_positive_00206
  nexar_train_positive_00263
  nexar_train_positive_00563
  nexar_train_positive_00882
```

## Using the Trained Model in an App

The current package provides reproducible model artifacts and split-level predictions. It does not yet provide a production HTTP service or a one-command `predict_one_video.py` inference CLI.

For full-stack integration today, the most direct consumable outputs are:

```text
outputs/collision_contact_model/val_selected_deep_rescue_summary.json
outputs/collision_contact_model/val_selected_deep_rescue_predictions.json
outputs/collision_contact_model/val_selected_deep_rescue_errors.json
```

Each prediction row contains:

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

For new videos, the required work is:

1. Generate the same 10-second event-centered clip.
2. Extract the same feature stack.
3. Build a feature vector with the same `4063` feature order.
4. Load `val_selected_deep_rescue_models.joblib`.
5. Apply the selected rescue-fusion rule and threshold.

That final single-video wrapper should be added before backend production integration.

## GitHub / Large File Notes

Recommended files to track with Git LFS:

```bash
git lfs track "*.npz"
git lfs track "*.joblib"
git add .gitattributes
```

If this package was assembled from `final_model_repro_transfer`, make sure the Git repo receives the real files, not local symlinks. In particular, `outputs/processed_744/features/`, `outputs/processed_744/object_cotracker_dynamics_yolov8s_32f_w384_20260524/`, and `analysis/impact_diagnostics_20260522/` may be symlinks in one local workspace. Copy the target directory contents into the Git repo.

Do not commit:

```text
.venv/
.cache/
external/co-tracker/
models/BADAS-Open/
data/nexar_collision_prediction/
data/processed_positive/
outputs/smoke/
outputs/processed_744_before_transfer/
analysis/impact_diagnostics_20260522_before_transfer/
```

The precomputed `outputs/processed_744/features/` directory is large, about 1.1 GB. Single files are below GitHub's 100 MB hard limit, but the repo will still become heavy without LFS or an external release asset.

## Troubleshooting

### `Library not loaded: @rpath/libomp.dylib`

Install OpenMP:

```bash
brew install libomp
```

Then verify:

```bash
python - <<'PY'
import lightgbm, xgboost
print(lightgbm.__version__)
print(xgboost.__version__)
PY
```

### Missing raw video paths

This only affects data checking and feature extraction. Fast reproduction from precomputed features can skip `run_00_check_and_prepare.sh`.

### CoTracker is slow on laptop

CoTracker is the slowest feature extraction stage. Use batching and `LIMIT` while testing:

```bash
LIMIT=8 COTRACKER_DEVICE=mps PYTHON=.venv/bin/python bash scripts/run_01_extract_features.sh
```

For full extraction, a CUDA machine is preferred.
