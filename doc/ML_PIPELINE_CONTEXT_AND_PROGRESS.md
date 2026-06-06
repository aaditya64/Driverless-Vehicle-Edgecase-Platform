# ML Pipeline Context and Progress

Last updated: 2026-05-26

This document summarizes the ML/data pipeline for the Driverless Vehicle Edge-Case Intelligence Platform. It is written for teammates who did not participate in the ML experiments and need to understand what was built, what files matter, how to reproduce the current result, and what still needs engineering work before backend/full-stack integration.

## 1. Current Executive Summary

The current best model solves the positive-only Nexar task:

```text
10-second event-centered positive clip -> near_miss or collision
```

It does not yet solve the full three-class task:

```text
safe / near_miss / collision
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

Label order in the confusion matrix:

```text
[near_miss, collision]
```

So the final test result means:

```text
near_miss correct: 75
near_miss predicted as collision: 4
collision predicted as near_miss: 5
collision correct: 65
total test samples: 149
total errors: 9
```

The reproducible package is:

```text
collision_contact_full_repro/
```

The final trained model/report artifacts are:

```text
collision_contact_full_repro/outputs/collision_contact_model/
```

## 2. What Changed Since the Earlier BADAS Baseline

The project started with BADAS/V-JEPA features as a risk-oriented baseline. That baseline was useful but not strong enough for the fine boundary between physical contact and very close near-misses.

Earlier BADAS conditional head:

```text
input: BADAS window embeddings, risk scores, timestamps
task: P(collision | risky)
data: 750 manually labelled Nexar train positive clips
result: about 0.81 test accuracy and about 0.88 AUROC over multiple seeds
```

After label cleanup and bad/unclear clip removal, the usable set became:

```text
usable samples: 744
task: near_miss vs collision
```

The fixed07 BADAS-style baseline was still limited:

```text
test AP:       about 0.886
test AUROC:    about 0.895
test accuracy: about 0.818
collision F1:  about 0.812
```

Main conclusion from the baseline stage:

```text
BADAS features capture risk severity, but they often miss the physical-contact evidence needed to separate collision from collision-like near_miss.
```

The current final method improves the result by adding camera transient vibration, wavelet/global-motion features, object residual physics, CoTracker object dynamics, and a released long-context anchor.

## 3. Dataset and Labels

Primary dataset:

```text
Nexar Collision Prediction positive clips
```

Original Nexar positive labels only mean:

```text
collision_or_near_miss
```

They do not distinguish:

```text
near_miss vs collision
```

So the team manually labelled the positive clips. Labels were stored in a CSV column named:

```text
manual_label
```

After several review passes:

- `not_sure` labels were removed or resolved.
- Visibly bad/unclear videos were removed from the final fixed split.
- The final reproducible split uses 744 samples.

Final fixed split:

```text
splits/processed_744/train.csv        595 rows
splits/processed_744/train_inner.csv  476 rows
splits/processed_744/val.csv          119 rows
splits/processed_744/test.csv         149 rows
```

Class counts:

```text
train:        collision=280, near_miss=315
train_inner:  collision=224, near_miss=252
val:          collision=56,  near_miss=63
test:         collision=70,  near_miss=79
```

Label encoding:

```text
0 = near_miss
1 = collision
```

Important raw/processed video paths expected by the full rebuild flow:

```text
data/nexar_collision_prediction/train/positive/*.mp4
data/nexar_collision_prediction/test/positive/*.mp4
data/processed_positive/processed/event_clips/nexar/train/positive/*.mp4
data/processed_positive/processed/event_clips/nexar/test/positive/*.mp4
```

Fast reproduction does not need raw `.mp4` files if the precomputed feature assets are present.

## 4. Final Method

The final model is not a single end-to-end neural network. It is a reproducible feature-fusion and rescue pipeline.

### 4.1 Camera Motion and Wavelet Features

Location:

```text
collision_contact_full_repro/collision_contact/extract_features.py
collision_contact_full_repro/collision_contact/motion_extract.py
collision_contact_full_repro/collision_contact/wavelet_features.py
```

Purpose:

- Estimate global camera motion between adjacent frames.
- Extract motion channels such as displacement, residuals, velocity, acceleration, jerk, fit error, and shake energy.
- Use continuous and stationary wavelet transforms to represent short transient vibration/contact patterns.

Key output:

```text
outputs/processed_744/features/
```

This directory contains:

```text
744 .npz feature files
744 _motion.csv files
```

### 4.2 Object Residual Physics

Location:

```text
collision_contact_full_repro/collision_contact/extract_object_residual_physics_features.py
collision_contact_full_repro/collision_contact/export_object_metrics.py
```

Purpose:

- Use YOLO object detections around the event.
- Compute object residual/energy/shift metrics that correlate with physical contact.
- Export compact tabular metrics used by the final model.

Key outputs:

```text
analysis/impact_diagnostics_20260522/object_physics_train_metrics.csv
analysis/impact_diagnostics_20260522/object_physics_test_metrics.csv
```

### 4.3 CoTracker Object Dynamics

Location:

```text
collision_contact_full_repro/collision_contact/extract_object_cotracker_dynamics_features.py
```

Purpose:

- Use YOLOv8s to choose object regions.
- Use CoTracker to track object points around the event.
- Summarize object dynamics over 32 frames at width 384.

Key output:

```text
outputs/processed_744/object_cotracker_dynamics_yolov8s_32f_w384_20260524/
```

This directory contains:

```text
744 .npz feature files
manifest.json
```

### 4.4 Released Long-Context Anchor

Location:

```text
assets/released_anchor/strong_fusion_probabilities.npz
assets/released_anchor/long_context_oof_experts_summary.json
```

Installed output:

```text
outputs/processed_744_long_context_anchor/
```

Purpose:

- Provide calibrated train/test probabilities from a stronger long-context model.
- This anchor gives the final rescue fusion a stable baseline probability.
- The final reported result uses this released anchor, not a newly retrained anchor.

Important caveat:

```text
Long-context features use a wider source-video context, so this is an offline/post-event classification setting, not a pre-event crash prediction model.
```

### 4.5 Final Rescue Fusion

Location:

```text
collision_contact_full_repro/collision_contact/train_val_selected_deep_rescue.py
```

Training command:

```bash
PYTHON=.venv/bin/python bash scripts/run_03_train_model.sh
```

How selection works:

- Build a 4063-dimensional feature vector.
- Train candidate compact tabular models on `train_inner`.
- Evaluate candidates on `val`.
- Combine candidate probabilities with the released anchor using rescue/noisy-or/gated rules.
- Select the best validation macro-F1 candidate whose threshold is constrained to `[0.45, 0.55]`.
- Report final test metrics once on the fixed test split.

Selected final candidate:

```text
lgbm_k512/rescue_q0.371/w2.50/fixed0p5
```

## 5. Reproduction Package

Main folder:

```text
collision_contact_full_repro/
```

Required code/config folders:

```text
README.md
requirements.txt
collision_contact/
scripts/
configs/
splits/
assets/released_anchor/
legacy/
```

Required precomputed feature/data artifacts for fast reproduction:

```text
outputs/processed_744/features/
outputs/processed_744/object_cotracker_dynamics_yolov8s_32f_w384_20260524/
analysis/impact_diagnostics_20260522/
outputs/processed_744_long_context_anchor/
```

Final trained model/report artifacts:

```text
outputs/collision_contact_model/val_selected_deep_rescue_models.joblib
outputs/collision_contact_model/val_selected_deep_rescue_summary.json
outputs/collision_contact_model/val_selected_deep_rescue_predictions.json
outputs/collision_contact_model/val_selected_deep_rescue_errors.json
outputs/collision_contact_model/val_selected_deep_rescue_probabilities.npz
```

The full reproduction instructions are now documented in:

```text
collision_contact_full_repro/README.md
```

## 6. Minimal Reproduction Commands

From inside `collision_contact_full_repro/`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

On macOS:

```bash
brew install libomp
```

Install released anchor:

```bash
PYTHON=.venv/bin/python bash scripts/run_02_train_anchor.sh
```

Train final model:

```bash
PYTHON=.venv/bin/python bash scripts/run_03_train_model.sh
```

Print report:

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

## 7. Full Feature Rebuild Commands

Only run this if precomputed features are missing or must be regenerated.

Install/check raw data and download model assets:

```bash
PYTHON=.venv/bin/python bash scripts/run_00_check_and_prepare.sh
```

Full extraction:

```bash
PYTHON=.venv/bin/python bash scripts/run_01_extract_features.sh
```

Apple Silicon lower-memory version:

```bash
COTRACKER_DEVICE=mps \
DINO_BATCH=2 \
RAFT_BATCH=1 \
YOLO_BATCH=4 \
BADAS_BATCH=1 \
PYTHON=.venv/bin/python \
bash scripts/run_01_extract_features.sh
```

Debug with a limit:

```bash
LIMIT=8 PYTHON=.venv/bin/python bash scripts/run_01_extract_features.sh
```

Important:

```text
LIMIT mode is only for debugging. It does not generate full object metrics for final training.
```

External model assets downloaded or cached during the full rebuild:

```text
nexar-ai/badas-open
facebook/vjepa2-vitl-fpc16-256-ssv2
vit_small_patch14_dinov2.lvd142m
vit_small_patch16_dinov3
MCG-NJU/videomae-base-finetuned-kinetics
torchvision RAFT Small C_T_V2
yolov8n.pt
yolov8s.pt
facebookresearch/co-tracker
```

Some are gated Hugging Face models, so a valid token may be required:

```bash
huggingface-cli login
export HF_TOKEN=your_huggingface_token
```

## 9. Full-Stack Integration Notes

The current output that full-stack can consume immediately:

```text
outputs/collision_contact_model/val_selected_deep_rescue_predictions.json
```

Each prediction entry contains:

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

Current limitation:

```text
There is no production-ready API service or single-video prediction CLI yet.
```

To score a new uploaded video in the platform, we still need to build a wrapper that:

1. Cuts or receives the 10-second event-centered clip.
2. Runs the same feature extraction stack.
3. Builds the 4063-dimensional feature vector in the saved order.
4. Loads `val_selected_deep_rescue_models.joblib`.
5. Applies the selected rescue-fusion candidate and threshold.
6. Returns `prob_collision`, `pred_label`, and supporting metadata.

Until that wrapper exists, the current trained model is best treated as a reproducible research artifact plus fixed-split prediction output, not a deployed inference service.

## 11. Other Dataset Work

SAVeD / Saved AV dataset was also prepared earlier:

```text
AV_crash.csv rows: 1040
AV_nearmiss.csv rows: 602
downloaded source videos: 248 mp4 files
generated event clips: 1606
collision clips: 1020
near_miss clips: 586
```

Current SAVeD status:

```text
Prepared locally, but not used in the final 93.96% Nexar fixed-split result.
```

Possible future uses:

- external validation,
- additional training data,
- hard-case mining,
- semantic scenario tagging.

## 13. Definition of Done for Current Stage

Completed:

- Manual positive label cleanup.
- Removal of unclear/bad clips from the final fixed set.
- Fixed 744-sample train/val/test split.
- BADAS conditional-head baseline.
- Camera-motion and wavelet feature extraction.
- Object residual physics feature extraction and metrics export.
- CoTracker object-dynamics feature extraction.
- Released long-context anchor installation.
- Final LightGBM rescue-fusion training.
- Reproduced final result exactly:
  - test accuracy `0.939597315436`
  - test AUROC `0.964556962025`
  - test confusion `[[75, 4], [5, 65]]`
- README updated with reproduction instructions for teammates.
- Final model artifacts saved under `outputs/collision_contact_model/`.
