# ML Pipeline Context and Progress

Last updated: 2026-05-16

This document records the current understanding, implementation state, file layout, and next steps for the ML/data-preparation part of the Driverless Vehicle Edge-Case Intelligence Platform.

## 1. Overall Goal

The ML component takes ego-centric dashcam videos as input and produces structured traffic-risk analysis outputs for a searchable edge-case intelligence platform.

The planned pipeline separates temporal risk detection from semantic interpretation:

1. BADAS-based risk model
   - Detects whether a clip contains ego-centric traffic risk.
   - Produces a risk timeline over the video.
   - Supports key-risk timestamp extraction.
   - Currently works as a binary model: `safe` vs `collision_or_near_miss`.

2. Three-class core outcome classifier
   - Target labels: `safe`, `near_miss`, `collision`.
   - Initial conditional head training script is implemented.
   - Current design keeps BADAS as the binary risk gate and trains a small head for `P(collision | risky)`.
   - Manual relabelling of the 750 Nexar train positive clips has now been completed by classmates.
   - The manual labels were added to the annotation CSV in a new column named `manual_label`.

3. Qwen3-VL semantic tag classifier
   - Planned for later.
   - Will consume BADAS-extracted 10-second clips.
   - Expected outputs include actor type, scenario type, road type, weather, avoidance behaviour, collision geometry, near-miss type, and a structured incident summary.

## 2. Dataset Strategy

### Nexar

Nexar is the current primary target-domain dataset.

Local expected path:

```text
data/nexar_collision_prediction/
```

Relevant original metadata:

```text
data/nexar_collision_prediction/train/positive/metadata.csv
data/nexar_collision_prediction/train/negative/metadata.csv
```

Important Nexar metadata columns:

```text
file_name
time_of_event
time_of_alert
light_conditions
weather
scene
time_to_accident
```

For positive training videos, `time_of_event` is available and is used as the event-centre timestamp for clip extraction.

Important limitation:

```text
Nexar positive = collision_or_near_miss
```

Nexar does not directly separate `near_miss` and `collision`, so positive clips originally required manual review before a three-class model could be trained.

Current update:

```text
All 750 Nexar train positive clips have been manually labelled.
The labels are stored in a CSV column named: manual_label
```

Important compatibility note:

```text
Older manifest/training flow expects: core_label
Current annotated CSV uses: manual_label
```

Formal training can now run from the default manifest by passing `--label-column manual_label`.

### SAVeD and NIDB

SAVeD is planned as an external hard-sample dataset with richer semantic annotations. It has not been integrated into the local scripts yet.

NIDB remains optional and has not been integrated.

## 3. Current Repository State

Current branch used for the full ML upload:

```text
ml-badas-preprocessing
```

GitHub branch:

```text
https://github.com/aaditya64/Driverless-Vehicle-Edgecase-Platform/tree/ml-badas-preprocessing
```

Tracked files include:

```text
README.md
requirements.txt
scripts/download_badas_open.py
scripts/inference_badas.py
scripts/create_event_clips.py
scripts/extract_badas_window_features.py
scripts/train_badas_outcome_head.py
data/processed/clip_manifests/nexar_train_positive_event_clips.csv
doc/
frontend/
backend/
```

Large local assets are intentionally not tracked by git:

```text
data/nexar_collision_prediction/
data/processed/event_clips/
models/
outputs/
```

This means each team member must download datasets and model weights locally.

## 4. Dependencies and Environment

Python dependencies are listed in:

```text
requirements.txt
```

Important requirement:

```text
transformers==4.57.3
psutil>=5.9.0
```

This version is needed because older versions such as `4.52.4` do not recognize the `vjepa2` architecture used by BADAS-Open.

`psutil` is required by the BADAS-Open training module that is reused when loading the checkpoint and extracting features.

External command-line tools required for clip extraction:

```text
ffmpeg
ffprobe
```

Check availability:

```bash
which ffmpeg
which ffprobe
```

On macOS, install with:

```bash
brew install ffmpeg
```

## 5. BADAS-Open Model Setup

BADAS-Open is downloaded from Hugging Face after access approval.

Local expected path:

```text
models/BADAS-Open/
```

Important local files after download:

```text
models/BADAS-Open/src/
models/BADAS-Open/weights/badas_open.pth
models/BADAS-Open/config.json
models/BADAS-Open/README.md
```

Download helper:

```text
scripts/download_badas_open.py
```

Usage:

```bash
huggingface-cli login
python scripts/download_badas_open.py
```

Notes:

- The script assumes the user has accepted the BADAS-Open Hugging Face access conditions.
- BADAS-Open also needs the base model `facebook/vjepa2-vitl-fpc16-256-ssv2`.
- The first inference run may download/cache the V-JEPA2 base model through Hugging Face.

## 6. BADAS Inference Script

Implemented script:

```text
scripts/inference_badas.py
```

Purpose:

- Load local BADAS-Open source and checkpoint directly.
- Run BADAS risk inference on one dashcam video.
- Produce a risk timeline.
- Extract peak-risk timestamp.
- Produce high-risk segments above a threshold.
- Write JSON and CSV outputs when requested.

Example command:

```bash
python scripts/inference_badas.py \
  --video data/nexar_collision_prediction/train/positive/00024.mp4 \
  --output-json outputs/badas_positive_00024.json \
  --output-csv outputs/badas_positive_00024.csv \
  --window-stride 16 \
  --device auto
```

Important arguments:

```text
--video          input mp4 path
--model-dir      BADAS-Open model directory, default models/BADAS-Open
--output-json    optional structured JSON output
--output-csv     optional timestamp/risk_score timeline CSV
--device         auto, cpu, cuda, or mps
--threshold      high-risk threshold, default 0.8
--target-fps     sampling rate, default 8.0
--frame-count    BADAS window size, default 16 frames
--window-stride  sliding-window stride in sampled frames
```

Model interpretation:

```text
target_fps = 8 fps
frame_count = 16 frames
16 frames / 8 fps = 2 seconds per BADAS window
```

The model output is binary:

```text
safe
collision_or_near_miss
```

It does not directly output:

```text
near_miss
collision
```

Example verified output on one Nexar positive video:

```text
video: data/nexar_collision_prediction/train/positive/00024.mp4
predicted_label: collision_or_near_miss
peak_risk_score: 0.992672
peak_risk_time_sec: 20.0
peak_risk_sampled_frame_idx: 160
mean_risk_score: 0.481607
```

Because sampling is 8 fps:

```text
160 sampled frames / 8 fps = 20.0 seconds
```

## 7. Event-Centred Clip Extraction

Implemented script:

```text
scripts/create_event_clips.py
```

Purpose:

- Read Nexar metadata.
- Use `time_of_event` as event centre for positive videos.
- Cut fixed-length event-centred clips.
- Generate a manifest for annotation and training.

Default input:

```text
data/nexar_collision_prediction/train/positive/metadata.csv
data/nexar_collision_prediction/train/positive/*.mp4
```

Default output:

```text
data/processed/event_clips/nexar/train/positive/
```

Default manifest:

```text
data/processed/clip_manifests/nexar_train_positive_event_clips.csv
```

Default command:

```bash
python scripts/create_event_clips.py
```

This processes all rows in:

```text
data/nexar_collision_prediction/train/positive/metadata.csv
```

Useful test command:

```bash
python scripts/create_event_clips.py --limit 5
```

Overwrite existing clips:

```bash
python scripts/create_event_clips.py --overwrite
```

Use `time_of_alert` instead of `time_of_event`:

```bash
python scripts/create_event_clips.py --center-field time_of_alert
```

Dry run without running ffmpeg:

```bash
python scripts/create_event_clips.py --dry-run
```

Current extraction rule:

```text
clip_start_time = time_of_event - 5 seconds
clip_end_time = time_of_event + 5 seconds
clip_duration = 10 seconds
```

Boundary handling:

- If the desired start time is before video start, the clip starts at 0.
- If the desired end time would exceed video duration, the clip is shifted earlier where possible.
- `center_offset_in_clip` records where the event centre appears inside the extracted clip.

Encoding:

```text
video codec: libx264
preset: veryfast
crf: 18
audio: removed
```

Optional faster but less frame-accurate mode:

```bash
python scripts/create_event_clips.py --copy-video
```

Practical note:

- Nexar videos checked locally do not contain an audio stream.
- The script removes audio with `-an`, but this does not currently discard useful Nexar signal because the source videos are video-only.

## 8. Generated Clip Manifest

Generated manifest:

```text
data/processed/clip_manifests/nexar_train_positive_event_clips.csv
```

Current local generation result:

```text
Manifest rows: 750
Created clips: 747
Existing clips: 3
Skipped rows: 0
Failed rows: 0
```

The three existing clips came from earlier test runs. Total local clip count after full generation:

```text
750 positive event-centred clips
```

Approximate local size:

```text
data/processed/event_clips/nexar/train/positive/ = about 4.5 GB
```

The manifest is tracked in git, but the actual clips are not.

Important manifest columns:

```text
clip_id
source_dataset
split
source_label_folder
source_binary_label
core_label
source_file_name
source_video_path
clip_path
event_center_time
video_duration
clip_start_time
clip_end_time
clip_duration
center_offset_in_clip
time_of_event
time_of_alert
light_conditions
weather
scene
status
error
```

For positive Nexar samples:

```text
source_binary_label = collision_or_near_miss
core_label = needs_review
```

This was the original generated state. The current annotated CSV now has an additional human label column:

```text
manual_label
```

Current status:

```text
clip_id column restored and present for all 750 rows
750 / 750 train positive clips manually labelled
human label source column = manual_label
legacy/default training label column = core_label
collision labels = 337
near_miss labels = 413
not_sure / ambiguous labels = 0
```

The training code can use a different label column, but the column choice must be explicit. The current default manifest is training-ready with `--label-column manual_label`; no `--clip-id-column` override is needed.

## 9. Annotation Status

The positive clip annotation pass is complete.

Current annotation source:

```text
CSV containing all 750 Nexar train positive clips
label column: manual_label
current cleaned distribution: 337 collision, 413 near_miss
```

Allowed final labels for Nexar positive clips:

```text
near_miss
collision
```

The legacy `core_label` column may still contain:

```text
needs_review
```

Do not assume `core_label` is the current human-reviewed label unless it has been explicitly synchronized from `manual_label`.

Label definitions:

- `collision`: physical contact happens between the ego vehicle and another actor/object, or the event clearly includes a crash involving the ego vehicle.
- `near_miss`: no physical contact, but an imminent collision is avoided through braking, swerving, stopping, or another emergency manoeuvre.
- `needs_review`: not yet labelled or ambiguous.

Training-ready options:

1. Use the annotated manifest directly:

```bash
python -u scripts/train_badas_outcome_head.py \
  --label-column manual_label \
  --device mps \
  --epochs 50 \
  --batch-size 32 \
  --output-dir outputs/outcome_head
```

2. Or create a training label CSV with this schema and use it through `--labels-csv`:

```text
clip_id,core_label
nexar_train_positive_00024,collision
nexar_train_positive_00072,near_miss
```

The second option works with the script defaults because `core_label` already exists in the original manifest and labels from `--labels-csv` override manifest labels.

If files are moved into class folders later, a future sync script should update the manifest so that training still uses one clean metadata source.

## 10. BADAS Window-Level Feature Extraction

Implemented script:

```text
scripts/extract_badas_window_features.py
```

Purpose:

- Load BADAS-Open and the frozen V-JEPA2 backbone.
- Run the processed 10-second clips through BADAS sliding windows.
- Extract the classifier-input window embedding from BADAS before the original binary risk classifier.
- Save reusable `.npz` files for training a lightweight near-miss/collision outcome head.
- Save BADAS risk scores and timestamp alignment for each window.

Default inputs:

```text
data/processed/clip_manifests/nexar_train_positive_event_clips.csv
data/processed/event_clips/nexar/train/positive/*.mp4
```

Default output directory:

```text
outputs/features/
```

Default command:

```bash
python -u scripts/extract_badas_window_features.py --offline --device auto
```

Recommended Apple Silicon command:

```bash
python -u scripts/extract_badas_window_features.py \
  --device mps \
  --offline \
  --window-batch-size 3
```

Important arguments:

```text
--target-fps          default 8.0
--frame-count         default 16
--window-stride       default 8
--window-batch-size   controls extraction speed/memory only; it does not change saved feature content
--limit               useful for pilot extraction
--offline             forces Hugging Face libraries to use local cache
```

Feature interpretation for the current defaults:

```text
10-second clip
target_fps = 8 fps
about 80 sampled frames
frame_count = 16 sampled frames = 2 seconds
window_stride = 8 sampled frames = 1 second
expected windows per clip = 9
feature shape per clip = [9, 1024]
```

Each `.npz` contains:

```text
features              float32, shape [num_windows, 1024]
risk_scores           float32, shape [num_windows]
logits                float32, shape [num_windows, 2]
window_start_sec      float32, shape [num_windows]
window_end_sec        float32, shape [num_windows]
target_time_sec       float32, shape [num_windows]
metadata              JSON string
```

Current verified local extraction state:

```text
feature_dir: outputs/features/
feature files for Nexar train positive clips: 750 / 750
standard num_windows per 10-second clip: 9
feature_dim: 1024
features_finite: True
risk_finite: True
npz_reload_check: ok
```

The full 750 positive clip feature extraction has now been completed locally. Re-running the extraction should not be necessary unless clips are regenerated, feature settings change, files are missing, or `--overwrite` is intentionally used.

The script writes one `.npz` per clip and skips existing files unless `--overwrite` is provided, so interrupted or partial future runs can still be resumed.

## 11. Conditional Near-Miss/Collision Outcome Head

Implemented script:

```text
scripts/train_badas_outcome_head.py
```

Purpose:

- Read cached BADAS window feature `.npz` files.
- Read human labels from the manifest or an optional separate labels CSV.
- Use only rows labelled:

```text
near_miss
collision
```

- Skip rows labelled:

```text
needs_review
ambiguous
safe
```

- Train a lightweight conditional classifier:

```text
q = P(collision | risky)
```

This keeps the BADAS risk model frozen and separates:

```text
P(risky)
```

from:

```text
P(collision | risky)
```

At inference time, the intended probability composition is:

```text
P(safe) = 1 - P(risky)
P(collision) = P(risky) * P(collision | risky)
P(near_miss) = P(risky) * (1 - P(collision | risky))
```

Current model architecture:

```text
window features [T, 1024]
+ risk score
+ relative time
        |
linear projection
        |
2-layer 1D temporal convolution
        |
mean pooling + max pooling
        |
small MLP
        |
collision logit
```

Default training command if the manifest already has final labels in `core_label`:

```bash
python -u scripts/train_badas_outcome_head.py \
  --device mps \
  --epochs 50 \
  --batch-size 32 \
  --output-dir outputs/outcome_head
```

Command for the current annotated manifest with `manual_label`:

```bash
python -u scripts/train_badas_outcome_head.py \
  --label-column manual_label \
  --device mps \
  --epochs 50 \
  --batch-size 32 \
  --output-dir outputs/outcome_head
```

If annotators keep labels in a separate CSV, the easiest compatible format is:

```text
clip_id,core_label
nexar_train_positive_00024,collision
nexar_train_positive_00072,near_miss
```

Then train with:

```bash
python -u scripts/train_badas_outcome_head.py \
  --labels-csv path/to/labels.csv \
  --device mps \
  --epochs 50 \
  --batch-size 32 \
  --output-dir outputs/outcome_head
```

Training outputs:

```text
outputs/outcome_head/best_outcome_head.pt
outputs/outcome_head/training_history.csv
outputs/outcome_head/train_val_test_split.csv
outputs/outcome_head/val_predictions.csv
outputs/outcome_head/test_predictions.csv
outputs/outcome_head/final_metrics.json
outputs/outcome_head/training_config.json
```

Default split:

```text
train = 70%
validation = 15%
test = 15%
```

The split is stratified by `near_miss` and `collision`. Validation is used for early stopping and model selection. Test should be treated as the final internal evaluation split and should not be used for repeated threshold or architecture tuning.

Current status:

- Manual labels for all 750 train positive clips are now available in `manual_label`.
- Label values have been cleaned through multiple hard-case review passes: 337 `collision`, 413 `near_miss`, 0 `not_sure`.
- The manifest has a valid `clip_id` column for all 750 rows.
- Cached BADAS window features for all 750 train positive clips are now available.
- Formal real-label training has been run on the cleaned labels.
- Current best baseline is `manual_label_fixed04_*_mps` / `manual_label_fixed05_*_mps`; these two runs produced identical results because the effective label set did not change between them.

Current baseline command pattern:

```bash
for seed in 1 2 3 4 5 42 2014 917 517; do
  python -u scripts/train_badas_outcome_head.py \
    --label-column manual_label \
    --device mps \
    --seed "$seed" \
    --epochs 50 \
    --batch-size 32 \
    --output-dir "outputs/outcome_head/manual_label_fixed05_seed${seed}_mps"
done
```

Current 9-seed baseline result with threshold `0.5`:

```text
test AP        = 0.857 +/- 0.042
test AUROC     = 0.879 +/- 0.028
test accuracy  = 0.808 +/- 0.035
test precision = 0.790 +/- 0.070
test recall    = 0.795 +/- 0.054
test F1        = 0.790 +/- 0.033
```

Validation-selected thresholds did not materially improve the result:

```text
threshold 0.5 test F1          = 0.790 +/- 0.033
validation-selected test F1    = 0.790 +/- 0.031
```

Interpretation:

- The outcome head is a meaningful and stable baseline for `near_miss` vs `collision`.
- The model is not random: AUROC is about `0.88` and F1 is about `0.79` across seeds.
- The remaining failure mode is semantic: BADAS frozen features capture risk severity well, but they do not always capture the fine-grained visual evidence of physical contact.
- Very collision-like near misses are still often predicted as `collision`.
- Some visually subtle collisions are still predicted as `near_miss`.

## 12. What Is Not Done Yet

### Safe Clip Extraction

Only Nexar positive event-centred clips have been generated so far.

Safe/negative clips still need to be generated from:

```text
data/nexar_collision_prediction/train/negative/
```

Because negative samples do not have `time_of_event`, a separate rule is needed, such as:

- middle 10 seconds
- random 10-second crop
- multiple random crops per video
- lowest-risk 10-second region based on BADAS, if we want harder safe samples

### Label Column Selection

Manual relabelling is complete. The current human-reviewed label column differs from the earlier script default:

```text
current annotated CSV: manual_label
default training label column: core_label
```

Before formal outcome-head training, use:

```bash
--label-column manual_label
```

No `--clip-id-column` override is needed now that the manifest has a normal `clip_id` column.

### Positive Outcome Head Baseline

The conditional `near_miss` vs `collision` outcome head has been formally trained and evaluated on real manual labels.

Current effective label version:

```text
manual_label_fixed04 / manual_label_fixed05
```

`fixed05` produced the same metrics as `fixed04`; the effective manifest labels were unchanged.

Current label distribution:

```text
collision = 337
near_miss = 413
total = 750
```

Current 9-seed test result at threshold `0.5`:

```text
test AP        = 0.857 +/- 0.042
test AUROC     = 0.879 +/- 0.028
test accuracy  = 0.808 +/- 0.035
test precision = 0.790 +/- 0.070
test recall    = 0.795 +/- 0.054
test F1        = 0.790 +/- 0.033
```

This is the current baseline to report. It is useful, stable, and clearly above a majority-class baseline, but it is not a final robust contact detector.

Current high-confidence repeated error examples:

```text
collision -> near_miss:
00180, 00466, 00423, 00818, 00117, 00210, 00383

near_miss -> collision:
00882, 01031, 00758, 01000, 00556, 00927, 00986
```

These examples should be treated as a hard-case evaluation set. If their labels are confirmed, the next improvement likely requires a finer physical-contact verifier rather than only retuning the current BADAS-feature head.

### Three-Class Model Training

The conditional positive-outcome head is implemented and has a real-label baseline. The full three-class system is not complete yet.

Target classes:

```text
safe
near_miss
collision
```

Possible implementation options:

1. Use BADAS/V-JEPA features and train the new conditional outcome head.
2. Fine-tune the BADAS-based model with a three-class output head.
3. Train a separate video classifier on the generated 10-second clips.

Recommended first step:

- Treat `manual_label_fixed04/fixed05` as the current positive-outcome baseline.
- Report 9-seed mean/std instead of a single seed.
- Keep threshold `0.5` as the baseline threshold unless a later validation protocol improves it.
- Use the repeated-error clips as a hard-case set for qualitative review.
- Add safe clips later to complete the full `safe`, `near_miss`, `collision` pipeline.

### Qwen3-VL Semantic Tagging

Qwen3-VL integration is not implemented yet.

Planned input:

```text
BADAS-extracted or event-centred 10-second clip
```

Planned output JSON fields:

```text
actor_type
scenario_type
road_type
weather
avoidance_behavior
collision_geometry
near_miss_type
short_summary
```

Initial recommendation:

- Start with prompt-based zero-shot/few-shot inference.
- Validate JSON strictly.
- Manually review a subset.
- Consider LoRA fine-tuning only after enough high-quality structured tags exist.

## 13. Reproduction Steps for a New Team Member

Clone the branch:

```bash
git clone -b ml-badas-preprocessing https://github.com/aaditya64/Driverless-Vehicle-Edgecase-Platform.git
cd Driverless-Vehicle-Edgecase-Platform
```

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install ffmpeg if missing:

```bash
brew install ffmpeg
```

Download or place Nexar dataset at:

```text
data/nexar_collision_prediction/
```

Expected example:

```text
data/nexar_collision_prediction/train/positive/metadata.csv
data/nexar_collision_prediction/train/positive/00024.mp4
data/nexar_collision_prediction/train/negative/metadata.csv
```

Download BADAS-Open:

```bash
huggingface-cli login
python scripts/download_badas_open.py
```

Generate Nexar positive event-centred clips:

```bash
python scripts/create_event_clips.py
```

Run BADAS inference on one video:

```bash
python scripts/inference_badas.py \
  --video data/nexar_collision_prediction/train/positive/00024.mp4 \
  --output-json outputs/badas_positive_00024.json \
  --output-csv outputs/badas_positive_00024.csv \
  --window-stride 16 \
  --device auto
```

Extract cached BADAS window features:

```bash
python -u scripts/extract_badas_window_features.py \
  --device auto \
  --offline
```

For the current local state, the 750 positive clip features have already been extracted. Re-run this only to regenerate missing or changed feature files.

Train the conditional outcome head from the annotated manifest:

```bash
python -u scripts/train_badas_outcome_head.py \
  --label-column manual_label \
  --device mps \
  --epochs 50 \
  --batch-size 32
```

If `manual_label` is later copied into `core_label`, the `--label-column` argument can be omitted.

## 14. Known Issues and Practical Notes

### Git Tracking

The following are intentionally ignored:

```text
data/nexar_collision_prediction/
data/processed/event_clips/
models/
outputs/
```

Do not commit raw videos or model weights to GitHub.

### Local Paths

The manifest uses repo-relative paths, for example:

```text
data/nexar_collision_prediction/train/positive/00822.mp4
data/processed/event_clips/nexar/train/positive/nexar_train_positive_00822.mp4
```

This allows other team members to reproduce the same structure locally.

### BADAS Warning

`albumentations` may print a warning about failing to fetch version info because of SSL certificate checks. This does not affect inference.

Example:

```text
albumentations/check_version.py: UserWarning: Error fetching version info
```

### Device Selection

`--device auto` chooses:

1. CUDA if available
2. Apple MPS if available
3. CPU otherwise

On Apple Silicon Macs, `mps` may be selected automatically.

For some Python/PyTorch installations, Apple MPS may not be available even when the hardware supports Metal. A clean virtual environment with an official macOS arm64 PyTorch build fixed this locally:

```bash
python -m venv .venv-mps
source .venv-mps/bin/activate
python -m pip install -U pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -c "import torch; print(torch.backends.mps.is_available())"
```

Expected output before using `--device mps`:

```text
True
```

Known working feature-extraction command inside this environment:

```bash
python -u scripts/extract_badas_window_features.py \
  --device mps \
  --offline \
  --window-batch-size 3
```

### Runtime

Full positive clip extraction with `scripts/create_event_clips.py` processes 750 videos and may take several minutes.

The local generated positive clip directory is several GB in size.

Full BADAS window feature extraction for 750 positive clips may take hours on a laptop because it runs the V-JEPA/BADAS model. This one-time cache-building step has now been completed for the 750 train positive clips. The resulting `.npz` files are small compared with the videos.

## 15. Suggested Next Tasks

1. Preserve and report the current positive-outcome baseline.
   - Current human label column is `manual_label`.
   - Current label counts: 337 `collision`, 413 `near_miss`.
   - Report the `manual_label_fixed04/fixed05` 9-seed result: F1 about `0.790 +/- 0.033`, AUROC about `0.879 +/- 0.028`.
   - Use threshold `0.5` as the baseline threshold.

2. Review or formalize the hard-case set.
   - `collision -> near_miss`: `00180`, `00466`, `00423`, `00818`, `00117`, `00210`, `00383`.
   - `near_miss -> collision`: `00882`, `01031`, `00758`, `01000`, `00556`, `00927`, `00986`.
   - Decide whether these labels are final or should be corrected.

3. Verify the completed feature cache before future training.
   - Expected positive feature files: 750.
   - Expected feature shape per standard clip: `[9, 1024]`.
   - Check for missing `.npz` files only if training reports missing features.

4. Generate safe clips from Nexar negative videos.
   - Add rows with `core_label=safe`.

5. Create a combined training manifest.
   - Include `safe`, `near_miss`, and `collision`.
   - Split into train/validation/test.

6. Add Qwen3-VL semantic tagging prototype.
   - Prompt for strict JSON.
   - Store tags in a structured output file.

7. Connect ML outputs to the platform backend/search layer.
   - Risk score
   - Core label
   - Event timestamp
   - Semantic tags
   - Summary

## 16. Current Definition of Done for This Stage

Completed:

- BADAS-Open download helper script.
- BADAS inference script.
- Nexar positive event-centred clip extraction script.
- Full local generation of 750 positive 10-second clips.
- Generated positive clip manifest.
- BADAS window-level feature extraction script.
- Verified cached `.npz` feature format and reload checks.
- Full local BADAS feature extraction for all 750 train positive clips.
- Manual positive relabelling for all 750 train positive clips, stored in `manual_label`.
- Cleaned label values after hard-case review: 337 `collision`, 413 `near_miss`, no `not_sure` labels remaining.
- Restored normal `clip_id` column in the positive clip manifest.
- Conditional `near_miss` vs `collision` outcome head training script.
- Smoke-tested outcome head training, checkpoint writing, history writing, and validation prediction output.
- Formal real-label positive-outcome baseline over 9 seeds.
- Current baseline metrics: test F1 `0.790 +/- 0.033`, AUROC `0.879 +/- 0.028`, AP `0.857 +/- 0.042`, accuracy `0.808 +/- 0.035`.
- Initial repeated-error hard-case set identified for collision-like near misses and subtle collisions.
- Git branch with code, docs, README, requirements, and manifest.

Not completed:

- Safe clip generation.
- Full three-class model assembly and evaluation.
- Physical-contact verifier for hard near-miss/collision boundary cases.
- Qwen3-VL tagging.
- Backend integration of ML outputs.
