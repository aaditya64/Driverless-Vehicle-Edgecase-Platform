# ML Technical Documentation

## Scope and evidence boundary

This document describes the machine-learning and computer-vision work present in this repository. It deliberately excludes frontend, database, deployment, and other full-stack concerns.

Every implementation claim below is grounded in repository files. The canonical collision/contact implementation is under [`collision_contact_full_repro/`](collision_contact_full_repro/). Qwen 2.5-VL is documented in [`API_USAGE_README.md`](API_USAGE_README.md); its server source and adapter weights are referenced there as files on a separate cloud machine, but are not present in this repository. Consequently, this document records the Qwen configuration, inputs, outputs, schema, and measured results that are actually documented, but does not invent its prompt, decoding parameters, adapter method, training data, or internal server logic.

## 1. ML problem decomposition

The repository contains two separate ML functions:

1. **Traffic-event outcome inference**: classify an event-centred positive dashcam clip as `near_miss` or `collision`.
2. **Semantic event generation**: use Qwen 2.5-VL to turn video evidence into a short summary and structured semantic tags such as scenario type, actor, environment, behaviour, collision geometry, and near-miss type.

These functions are not the same model and should not be described as one end-to-end network.

```text
Dashcam video
  |
  +-- outcome branch ----------------------------------------------+
  |   10 s event clip                                               |
  |     -> camera motion + wavelets                                 |
  |     -> DINOv2 / DINOv3 / VideoMAE / RAFT / YOLO / BADAS        |
  |     -> long-context anchor                                      |
  |     -> object residual physics + YOLO-guided CoTracker         |
  |     -> LightGBM rescue fusion                                  |
  |     -> P(collision), near_miss/collision                       |
  |                                                                 |
  +-- semantic branch ----------------------------------------------+
      video frames + optional upstream outcome
        -> Qwen/Qwen2.5-VL-7B-Instruct + configured adapter
        -> summary + tags + structured traffic-event attributes
```

The outcome package explicitly solves:

```text
10-second event-centred Nexar positive clip -> near_miss or collision
```

It does **not** by itself solve the full `safe / near_miss / collision` problem. The earlier BADAS work supplies a binary risk-oriented `safe` versus `collision_or_near_miss` stage, while the current best package concentrates on the harder positive-only physical-contact boundary. See [`README.md`](README.md), [`ML_PIPELINE_CONTEXT_AND_PROGRESS.md`](ML_PIPELINE_CONTEXT_AND_PROGRESS.md), and [`collision_contact_full_repro/README.md`](collision_contact_full_repro/README.md).

## 2. Dataset, labels, and evaluation protocol

### 2.1 Data and target labels

The final outcome model uses manually refined labels for Nexar Collision Prediction positive clips. Nexar's original positive label means `collision_or_near_miss`; the team added a `manual_label` to separate the two outcomes.

The cleaned fixed dataset contains 744 usable positive clips:

| Split | Samples | Near miss | Collision | Role |
|---|---:|---:|---:|---|
| `train` | 595 | 315 | 280 | Full training set and anchor OOF set |
| `train_inner` | 476 | 252 | 224 | Fits final rescue-head candidates |
| `val` | 119 | 63 | 56 | Selects the final candidate |
| `test` | 149 | 79 | 70 | Final fixed evaluation |

The encoding is:

```text
0 = near_miss
1 = collision
```

The split files are stored under:

```text
collision_contact_full_repro/splits/processed_744/
collision_contact_full_repro/splits/processed_744_long/
```

The short split has `path,label,label_name`. The long-context split additionally records the source-video path and timing/context metadata such as `time_of_event`, `time_of_alert`, lighting, weather, and scene.

### 2.2 Event and long-context inputs

The main outcome features are extracted from 10-second, event-centred clips. The motion configuration fixes the representation to 300 frames at a nominal 30 FPS. The long-context branch reads the original source video and samples it at 1 FPS in the current extraction script, using pre-event, event, early-post, late-post, post, and full-video windows. This makes the reported outcome model an **offline/post-event classifier**, not a causal pre-event collision predictor.

### 2.3 Selection discipline

The final rescue head is fitted on `train_inner`, evaluated on `val`, and selected by validation macro-F1. Only anchor/rescue-family candidates with a decision threshold in `[0.45, 0.55]` are eligible. The selected candidate is then reported on the fixed test set. This logic is implemented in [`train_val_selected_deep_rescue.py`](collision_contact_full_repro/collision_contact/train_val_selected_deep_rescue.py).

The released anchor contains out-of-fold probabilities for all 595 training samples and predictions for all 149 test samples. Its shape is `(595, 72)` for `oof_base` and `(149, 72)` for `test_base`, representing 72 cached expert probability columns.

## 3. Outcome-model architecture at a glance

The current best result is a **feature-fusion system**, not a single fine-tuned video transformer. Frozen/pretrained neural models produce representations or tracks; classical vision produces physically interpretable motion signals; compact tabular classifiers combine the evidence.

| Component | Repository model/configuration | Function in the pipeline | Status in the reported result |
|---|---|---|---|
| Global motion | Shi-Tomasi corners, pyramidal Lucas-Kanade, RANSAC affine fit | Ego/camera displacement and shake | Direct final rescue features |
| Wavelets | CWT `cmor1.5-1.0`, SWT `sym4` | Transient impact/vibration energy | Direct final rescue features and anchor inputs |
| DINOv2 | `vit_small_patch14_dinov2.lvd142m` | Frozen frame-level visual embeddings | Anchor/event-fusion input |
| DINOv3 | `vit_small_patch16_dinov3` | Frozen sparse-frame visual embeddings | Anchor/event-fusion input |
| VideoMAE | `MCG-NJU/videomae-base-finetuned-kinetics` | Frozen spatiotemporal video tokens | Anchor/event-fusion input |
| RAFT | TorchVision RAFT Small `C_T_V2` | Dense optical-flow statistics | Anchor/event-fusion input |
| YOLOv8n | `yolov8n.pt` | Object interaction, residual physics, and long-context detections | Anchor inputs and exported object metrics |
| YOLOv8s | `yolov8s.pt` | Select object boxes for point tracking | Direct CoTracker feature construction |
| CoTracker | local `cotracker3_offline` | Long-range point tracks inside threat-ranked objects | 232 direct final rescue features |
| BADAS-Open / V-JEPA2 | `nexar-ai/badas-open`, base `facebook/vjepa2-vitl-fpc16-256-ssv2` | Window embeddings and risk scores | Anchor/event-fusion input; earlier baseline |
| Long-context expert | HGB context expert over event probability, 40-second behaviour, and metadata | Stable base collision probability | Released anchor used by final rescue |
| Final rescue model | LightGBM after univariate feature selection | Correct high-confidence collision misses from the anchor | Selected final model |
| Qwen semantics | `Qwen/Qwen2.5-VL-7B-Instruct` plus configured adapter directory | Summary and structured semantic tags | Separate semantic-output service |

The full extraction order is defined in [`run_01_extract_features.sh`](collision_contact_full_repro/scripts/run_01_extract_features.sh). The released anchor is installed or optionally retrained by [`run_02_train_anchor.sh`](collision_contact_full_repro/scripts/run_02_train_anchor.sh). The final model is trained by [`run_03_train_model.sh`](collision_contact_full_repro/scripts/run_03_train_model.sh).

## 4. Camera/global-motion estimation

Implementation: [`motion_extract.py`](collision_contact_full_repro/collision_contact/motion_extract.py).

This stage estimates motion attributable to the dashcam/ego platform and then isolates high-frequency residual shake.

### 4.1 Frame preprocessing

- Frames are resized to width 640 while preserving aspect ratio.
- Frames are converted to grayscale.
- CLAHE is enabled with clip limit `2.0` and tile grid `(8, 8)`.
- The final sequence representation is padded or trimmed to 300 frames.

### 4.2 Sparse correspondence and affine motion

For every adjacent frame pair:

1. Up to 2,500 Shi-Tomasi corners are detected with quality level `0.01`, minimum distance `7`, and block size `7`.
2. Pyramidal Lucas-Kanade optical flow uses a `21 x 21` window, three pyramid levels, at most 30 iterations, and epsilon `0.01`.
3. Forward-backward tracking rejects points whose round-trip error exceeds `1.5` pixels.
4. At least 80 tracks must survive.
5. `cv2.estimateAffinePartial2D` fits a partial affine transform with RANSAC threshold `2.0`, confidence `0.995`, and at most 3,000 iterations.
6. At least 50 affine inliers are required.

The fitted matrix yields translation `(dx, dy)`, rotation `theta`, scale, scale change, shear, inlier ratio, and median inlier reprojection error. Rotation is also converted to a pixel-equivalent signal:

```text
theta_px = theta * resize_width
```

### 4.3 Residual shake and derivatives

Translations and rotation are cumulatively integrated to form camera paths. A Savitzky-Golay baseline with window 31 and polynomial order 3 removes slow trajectory motion:

```text
x_res     = x_path     - smooth(x_path)
y_res     = y_path     - smooth(y_path)
theta_res = theta_path - smooth(theta_path)
```

First, second, and third finite differences produce residual velocity, acceleration, and jerk. Two vector magnitudes are then formed:

```text
shake_energy = sqrt(x_res^2 + y_res^2 + theta_res_px^2)
jerk_energy  = sqrt(jerk_x^2 + jerk_y^2 + jerk_theta_px^2)
```

The configured 20 raw channels are:

```text
dx, dy, theta_px, scale_delta,
x_res, y_res, theta_res_px,
vx, vy, vtheta_px,
ax, ay, atheta_px,
jerk_x, jerk_y, jerk_theta_px,
shake_energy, jerk_energy,
inlier_ratio, fit_error
```

## 5. Wavelet representations and engineered impact features

Implementation: [`wavelet_features.py`](collision_contact_full_repro/collision_contact/wavelet_features.py), [`train_impact_physics_head.py`](collision_contact_full_repro/collision_contact/train_impact_physics_head.py), and [`train_deep_impulse_physics_head.py`](collision_contact_full_repro/collision_contact/train_deep_impulse_physics_head.py).

### 5.1 Normalisation

Raw and wavelet matrices are robustly normalised per feature. Absolute/non-normalised versions are retained as well, because absolute motion magnitude carries impact information that z-scoring would remove.

### 5.2 Continuous wavelet transform

The CWT is applied to 12 residual/derivative channels:

```text
x_res, y_res, theta_res_px,
ax, ay, atheta_px,
jerk_x, jerk_y, jerk_theta_px,
shake_energy, jerk_energy, fit_error
```

Configuration:

```text
wavelet: cmor1.5-1.0
frequency range: 0.5-14.0 Hz
frequency bins: 48, geometrically spaced
```

For coefficient `c`, stored CWT energy is:

```text
log(1 + |c|^2)
```

This produces `300 x (12 * 48) = 300 x 576` CWT matrices for both normalised and absolute variants.

### 5.3 Stationary wavelet transform

The SWT uses `sym4` with four levels. Each channel contributes four detail sequences plus the final approximation sequence, giving `12 * 5 = 60` values per frame. Normalised and absolute versions are stored.

Each motion feature file therefore contains:

| Array | Shape |
|---|---:|
| `raw`, `raw_abs` | `300 x 20` each |
| `cwt`, `cwt_abs` | `300 x 576` each |
| `swt`, `swt_abs` | `300 x 60` each |
| `handcrafted` | `672` |
| `handcrafted_local` | `336` |

### 5.4 Fixed event-window feature block

The final rescue representation starts with 3,540 fixed-window impact features. For each relevant signal, the code divides the 10-second clip into:

```text
pre-impact: 1.0 <= t < 4.2 s
impact:     4.4 <= t < 5.8 s
post:       5.8 <= t < 7.2 s
```

It records impact maximum, mean, p95, ratios against the pre-event baseline, robust z-score using median absolute deviation, post/impact ratio, impact width above a robust baseline, peak time, and nine distribution statistics for each of the three windows.

The 3,540 dimensions are composed exactly as follows:

- `20` raw channels x `36` window features = `720`.
- `12` signed raw channels x `7` impulse/rebound features = `84`.
- `12` CWT channels x `4` frequency bands x `36` = `1,728`.
- `4` total CWT bands x `36` = `144`.
- The first `24` SWT channels x `36` = `864`.
- Total: `720 + 84 + 1,728 + 144 + 864 = 3,540`.

The CWT bands used here are `0.5-2 Hz`, `2-6 Hz`, `6-14 Hz`, and all frequencies.

### 5.5 Adaptive deep-impulse block

An additional 291 features locate a data-dependent impact candidate between 2 and 8 seconds. The peak score combines high-frequency jerk CWT energy with absolute `jerk_energy`, `shake_energy`, and affine `fit_error`.

Around that peak, the code computes:

- pre, impact, and post statistics for acceleration, jerk, residual displacement, and velocity vector magnitudes;
- signed impulse, integral, sign-flip, stop-index, and peak-to-mean measures;
- post-impact ring-down energy and log-envelope slope;
- high- and mid-frequency jerk-wavelet statistics;
- seven compact YOLO object-residual metrics.

The direct non-CoTracker feature vector is therefore:

```text
3,540 fixed event-window features
+ 291 adaptive impulse/object features
= 3,831 features
```

## 6. DINOv2 and DINOv3 visual embeddings

Implementation: [`extract_dino_features.py`](collision_contact_full_repro/collision_contact/extract_dino_features.py).

Two `timm` models are used as frozen feature extractors with `num_classes=0`; the code calls `eval()` and uses `torch.no_grad()`, so these backbones are not fine-tuned in this pipeline.

### 6.1 DINOv2 path

```text
model: vit_small_patch14_dinov2.lvd142m
frames: 16 uniformly sampled across the event clip
input size: 518 x 518
```

### 6.2 DINOv3 path

```text
model: vit_small_patch16_dinov3
frames: 4 uniformly sampled across the event clip
input size: 256 x 256
```

For both variants:

- OpenCV decodes uniformly spaced frames and repeats the last decoded frame if decoding returns fewer than requested.
- Each frame is aspect-preserving letterboxed onto a square canvas.
- Pixel values are divided by 255 and normalised with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- Per-frame embeddings are retained.
- A fixed summary concatenates embedding-wise mean, standard deviation, maximum, minimum, mean absolute temporal difference, and maximum absolute temporal difference.

DINO features are used by the retrainable strong event-fusion/anchor path. They are not concatenated directly into the final 4,063-dimensional rescue vector.

## 7. VideoMAE temporal embeddings

Implementation: [`extract_videomae_features.py`](collision_contact_full_repro/collision_contact/extract_videomae_features.py).

Configuration:

```text
model: MCG-NJU/videomae-base-finetuned-kinetics
frames: 16 uniformly sampled across the event clip
processor: VideoMAEImageProcessor.from_pretrained(model)
network: VideoMAEModel.from_pretrained(model)
```

The model runs in evaluation mode without gradient calculation. The code stores `last_hidden_state` tokens and a summary formed by token-wise mean, standard deviation, maximum, minimum, 25th percentile, and 75th percentile.

VideoMAE is grouped with DINO features under the internal strong-fusion name `video_semantic`. That internal name refers to learned visual/video representations; it is separate from the Qwen-generated human-readable semantic tags.

## 8. RAFT dense optical flow

Implementation: [`extract_raft_features.py`](collision_contact_full_repro/collision_contact/extract_raft_features.py).

Configuration in the full extraction script:

```text
model: torchvision raft_small
weights: Raft_Small_Weights.C_T_V2
frame pairs: 4 uniformly distributed starts
pair gap: 2 source frames
resize width: 384
```

Frame height is rounded to a multiple of 8, matching the network's spatial constraints. For each predicted dense flow field, the pipeline computes:

- distribution statistics of total flow magnitude;
- distribution statistics after subtracting a RANSAC partial-affine global-flow model;
- statistics of absolute horizontal and vertical flow;
- border versus centre residual magnitude and their ratio;
- a `4 x 4` grid of residual-flow means, its maximum, and its standard deviation;
- normalised 16-bin direction entropy for raw and residual flow;
- affine inlier ratio and fit error.

Each statistic vector is then summarised across the selected frame pairs using mean, standard deviation, maximum, and minimum. RAFT contributes to the strong event-fusion/anchor path rather than the direct 4,063-dimensional final vector.

## 9. YOLO object understanding

Implementation: [`extract_yolo_interaction_features.py`](collision_contact_full_repro/collision_contact/extract_yolo_interaction_features.py), [`extract_object_residual_physics_features.py`](collision_contact_full_repro/collision_contact/extract_object_residual_physics_features.py), and [`extract_object_cotracker_dynamics_features.py`](collision_contact_full_repro/collision_contact/extract_object_cotracker_dynamics_features.py).

The repository uses two YOLOv8 weight sizes for different purposes.

### 9.1 Class scope

The COCO IDs treated as risk-relevant are:

```text
0 person
1 bicycle
2 car
3 motorcycle
5 bus
7 truck
```

The vehicle-only subset is `{2, 3, 5, 7}`.

### 9.2 YOLOv8n interaction features

The short-clip interaction extractor uses `yolov8n.pt`, 24 uniformly sampled frames, inference size 640, confidence threshold `0.25`, and IoU threshold `0.5`.

For both the vehicle subset and all risk classes, each frame records object count, maximum confidence, bounding-box area statistics, height/width, bottom position, distance from image centre, and threat statistics. Six per-class counts are appended.

The threat heuristic is explicitly:

```text
area = box_area / frame_area
center_weight = max(0, 1 - 2 * |normalised_center_x - 0.5|)
lower_weight = 0.5 + 0.5 * normalised_box_bottom
threat = area * confidence * (0.25 + 0.75 * center_weight) * lower_weight
```

Temporal summarisation concatenates per-feature mean, standard deviation, maximum, minimum, last-minus-first, maximum positive change, and slopes for eight selected interaction signals.

### 9.3 YOLO residual-object physics

The direct object-physics extractor uses `yolov8n.pt`, width 320, every fourth source frame, inference size 640, confidence `0.2`, and IoU `0.5`.

For each sampled pair:

1. Farneback flow is computed.
2. A RANSAC affine flow model estimates global camera motion.
3. Expected affine flow is subtracted to obtain residual flow.
4. The previous grayscale frame is affine-warped, and absolute appearance difference is measured.
5. YOLO boxes select object-local residual-flow and difference regions.
6. Current boxes are associated with previous boxes by maximum IoU.

Object-local signals include residual p95/p99, residual energy, residual coherence, difference p95/energy, normalised area, threat, centre shift, area change, and tracking IoU. A composite score multiplies residual, energy, threat, and appearance-difference terms.

Dynamic summaries search for peaks between 2 and 8 seconds and record peak ratios, robust peak z-scores, peak time, sharpness, post-peak decay, p95, and first/second derivative peaks. Seven compact metrics are exported for the final rescue representation:

```text
log(object score peak/early ratio)
object score robust z-score
log(object residual-energy peak/early ratio)
log(object centre-shift peak/early ratio)
log(object difference-energy peak/early ratio)
object peak time
indicator that 2.8 <= peak time <= 6.6
```

### 9.4 YOLOv8s as the CoTracker query selector

`yolov8s.pt` is used for the tracking stage because the code separately configures it as the object-box source. It runs on three relative query positions, `0.35`, `0.50`, and `0.65`, in a six-second centre window. Confidence is `0.12`, IoU is `0.55`, and at most four threat-ranked boxes per query frame are retained.

Each retained box receives a `3 x 3` grid of point queries inset by 16% from the box border. If no risk-relevant box is found, a fallback grid covers the lower central image region.

## 10. CoTracker object dynamics

Implementation: [`extract_object_cotracker_dynamics_features.py`](collision_contact_full_repro/collision_contact/extract_object_cotracker_dynamics_features.py).

Configuration:

```text
model loader: torch.hub.load(local_repo, "cotracker3_offline", source="local")
frames: 32
resize width: 384
temporal region: central 6 seconds
backward tracking: requested when supported
```

CoTracker receives explicit `(query_time, x, y)` points generated inside the YOLOv8s boxes. The resulting arrays contain point tracks and visibility across all 32 frames.

### 10.1 Ego-motion removal

The tracking branch independently computes Farneback flow between adjacent sampled frames and fits a partial affine transform with RANSAC. For an object-group centre `c_(t-1)`, the expected camera-induced location is the affine transform of that centre. Residual object displacement is:

```text
r_t = c_t - affine_t(c_(t-1))
```

It is divided by the object's original box height, giving a scale/depth-normalised displacement. Dividing by elapsed source-video time gives velocity; finite differences give acceleration and jerk.

### 10.2 Group-level signals

Point groups require visibility of at least `0.35` in consecutive frames. Robust group centres are medians of visible tracks; 10th-to-90th percentile extents estimate width and height when enough points are visible.

The 16 per-pair signals are:

```text
visible_group_count
threat_max
res_depth_max, res_depth_mean
res_x_depth_abs_max, res_y_depth_abs_max
vel_depth_max, acc_depth_max, jerk_depth_max
scale_log_abs_max, area_log_abs_max
shape_change_max
impulse_score_max, motion_transfer_max
affine_inlier_ratio, affine_fit_error
```

The contact-like composite terms are implemented as:

```text
impulse = (1 + residual)
        * (1 + acceleration)
        * (1 + 0.2 * jerk)
        * (1 + |log scale change| + |log area change|)
        * (0.05 + threat)

motion_transfer = (1 + residual)
                * (1 + shape change)
                * (1 + |log area change|)
                * (0.05 + normalised object area)
```

### 10.3 The 232-dimensional CoTracker summary

Each of the 16 temporal signals is reduced to 13 dynamic statistics: peak, log peak, peak/q25 ratio, peak/early-median ratio, peak/early-p95 ratio, robust peak z-score, peak time, sharpness, post/peak, post/pre, p95, first-derivative peak, and second-derivative peak. This gives `16 * 13 = 208` features.

Eight group-level maxima are additionally aggregated by max, mean, and standard deviation, giving `8 * 3 = 24` features.

```text
208 temporal summary features
+ 24 group-level summary features
= 232 CoTracker features
```

These 232 values are directly appended to the 3,831 impact/object vector in the final rescue model.

## 11. BADAS-Open and V-JEPA2

Implementation: [`legacy/extract_badas_window_features.py`](collision_contact_full_repro/legacy/extract_badas_window_features.py), [`scripts/inference_badas.py`](scripts/inference_badas.py), and [`scripts/train_badas_outcome_head.py`](scripts/train_badas_outcome_head.py).

The BADAS model is loaded from a local `nexar-ai/badas-open` checkout and checkpoint `weights/badas_open.pth`. Its base model identifier is:

```text
facebook/vjepa2-vitl-fpc16-256-ssv2
```

The feature-extraction configuration uses:

```text
target FPS: 8
frames per window: 16
window stride: 8 sampled frames = 1 second
image size passed to BADAS: 224
window batch size in full extraction: configurable, default 3
```

For a 10-second clip, the stored assets contain nine windows. For each window the code extracts:

- the classifier-input embedding after BADAS temporal processing (`1024` dimensions in the stored feature files);
- two logits;
- a temperature-scaled positive/risk probability, computed as `softmax(logits / 2)[:, 1]`;
- window start/end indices and target timestamp.

If the BADAS custom head exposes future prediction, predicted future features are combined with present features before temporal processing.

The earlier standalone BADAS outcome head appended risk and normalised time to each window embedding, projected windows into a hidden space, applied one-dimensional temporal convolutions, concatenated mean and max temporal pooling, and produced `q = P(collision | risky)` with a binary head. This was an earlier baseline; it is not the final selected 93.96% model.

In the strong-fusion anchor, BADAS embeddings, risks, and logits are summarised across windows with distribution and temporal-difference statistics. BADAS is useful for risk severity but, according to the repository's experiment notes, was insufficient alone for distinguishing physical contact from very close near misses.

## 12. Strong event fusion and released long-context anchor

Implementation: [`train_strong_fusion.py`](collision_contact_full_repro/collision_contact/train_strong_fusion.py), [`extract_long_context_features.py`](collision_contact_full_repro/collision_contact/extract_long_context_features.py), [`export_long_context_oof_experts.py`](collision_contact_full_repro/collision_contact/export_long_context_oof_experts.py), and the released report [`long_context_oof_experts_summary.json`](collision_contact_full_repro/assets/released_anchor/long_context_oof_experts_summary.json).

### 12.1 Event-level feature groups

The retrainable strong-fusion stage constructs:

```text
physics       = handcrafted motion/wavelet + extended wavelet summaries
flow_obj      = RAFT summary + YOLO interaction summary
visual        = DINOv2 summary + DINOv3 summary
videomae      = VideoMAE summary
badas         = BADAS embedding/risk/logit summary
video_semantic = visual + videomae
all           = physics + flow_obj + video_semantic + badas
```

Candidate base classifiers include shallow and deeper XGBoost models, Extra Trees, histogram gradient boosting, PCA+SVC, PCA+logistic regression, random forest, LightGBM, and CatBoost when installed. Five-fold stratified out-of-fold predictions are generated for base models. Meta features concatenate the base probabilities with row-wise mean, standard deviation, min, max, and quartiles. Meta alternatives include mean, median, logistic regression with internal CV, XGBoost, and a validation-selected blend of logistic-regression and XGBoost meta probabilities.

The released anchor refers to the event primary model:

```text
processed_744_strong_fusion_boosted:blend_logreg_xgb
```

Thus, DINO/DINOv3, VideoMAE, RAFT, YOLO, BADAS, and motion/wavelet representations contribute through the event probability supplied to the long-context expert.

### 12.2 Long-context behaviour features

The current extractor samples the original source video at 1 FPS and width 480, using frame difference rather than affine motion in the invoked configuration. YOLOv8n is enabled.

Channels include visual speed/frame difference, translation/rotation placeholders or affine values depending on mode, fit quality, brightness, blur, vehicle/risk counts and threat, and per-class counts. Eight statistics are computed for each channel in six windows:

```text
pre:        event_time - 8 s to event_time - 1 s
event:      event_time - 1 s to event_time + 2 s
early_post: event_time + 2 s to event_time + 10 s
late_post:  event_time + 10 s to event_time + 18 s
post:       event_time + 2 s to source-video end
full:       complete source video
```

The extractor also adds post-minus-pre and post/pre ratios for selected signals and visual-stop ratios based on a pre-event speed baseline.

### 12.3 Selected released anchor expert

The final rescue head selects this column from the released 72-expert probability asset:

```text
processed_744_strong_fusion_boosted:
  blend_logreg_xgb/context_behavior/none/hgb/context
```

The name encodes:

- event source: the boosted strong-fusion `blend_logreg_xgb` probability;
- context variant: `context_behavior`, which includes long-context behaviour, event-derived features, and long-split metadata, but excludes the optional long-DINO and context-BADAS additions;
- hard-weight mode: `none`;
- context classifier: histogram gradient boosting (`hgb`);
- output: the context expert probability rather than its additional fused variant.

The selected anchor expert's released fixed-test default metrics are accuracy `0.9194630872483222`, AUROC `0.9415913200723327`, and confusion matrix `[[76, 3], [9, 61]]`.

The released anchor is a cached probability asset, not a complete single-video inference model. Rebuilding it requires the omitted raw videos and all feature models; using it directly is valid only for the fixed split whose rows match the cached arrays.

## 13. Final 4,063-dimensional rescue model

Implementation: [`train_deep_impulse_rescue_fusion.py`](collision_contact_full_repro/collision_contact/train_deep_impulse_rescue_fusion.py), [`train_object_cotracker_rescue_fusion.py`](collision_contact_full_repro/collision_contact/train_object_cotracker_rescue_fusion.py), and [`train_val_selected_deep_rescue.py`](collision_contact_full_repro/collision_contact/train_val_selected_deep_rescue.py).

### 13.1 Exact input composition

```text
3,540 fixed-window impact/wavelet features
+ 291 adaptive impulse and compact object-residual features
+ 232 YOLO-guided CoTracker dynamics features
= 4,063 input features
```

NaN and infinite values are replaced by zero before fitting.

### 13.2 Candidate preprocessing and estimators

The final candidate set applies:

1. `SimpleImputer`;
2. signed logarithmic compression, `sign(x) * log(1 + |x|)`;
3. for logistic regression only, standard scaling;
4. `SelectKBest(f_classif)` with 256 or 512 selected features;
5. one of logistic regression, histogram gradient boosting, Extra Trees, XGBoost, or LightGBM.

The selected raw rescue estimator is `lgbm_k512`, configured as:

```text
selected features: 512 by univariate ANOVA F score
trees: 340
num_leaves: 7
learning_rate: 0.022
subsample: 0.88
colsample_bytree: 0.70
min_child_samples: 16
reg_alpha: 0.4
reg_lambda: 7.0
class_weight: balanced
objective: binary
random seed: 20260529 (base seed 20260524 + 5)
```

### 13.3 Anchor-aware hard-sample weighting

Candidate rescue heads are fitted with weights derived from anchor difficulty. Let `p0` be the anchor probability and `t0` the anchor threshold selected for macro-F1 on the fitting split:

```text
margin    = |logit(p0) - logit(t0)|
uncertain = exp(-min(margin, 6))
missed    = 1(anchor prediction != label)

w = 1 + 1.5 * (0.65 * uncertain + 1.35 * missed)
```

Weights are divided by their mean and clipped to `[0.35, 4.0]`. This makes the rescue classifier focus on examples the anchor misses or considers uncertain.

### 13.4 Selected rescue equation

The selected validation candidate is:

```text
lgbm_k512/rescue_q0.371/w2.50/fixed0p5
```

Let:

```text
p0 = released long-context anchor probability
q  = lgbm_k512 collision probability from the 4,063 features
tau = 0.371
w = 2.50
```

The rescue only increases the anchor logit when `q` exceeds `tau` in logit space:

```text
delta = max(0, logit(q) - logit(0.371))
p_final = sigmoid(logit(p0) + 2.50 * delta)
prediction = collision if p_final >= 0.5 else near_miss
```

This is intentionally asymmetric: weak rescue evidence leaves the anchor unchanged, while sufficiently strong impact/CoTracker evidence can raise collision probability.

The code also evaluates direct-logit, centred-logit, window, noisy-OR, and gated-maximum fusion families, but the validation-selected final result is the positive-only rescue equation above.

## 14. Qwen 2.5-VL semantic generation

Repository evidence: [`API_USAGE_README.md`](API_USAGE_README.md).

Qwen is a separate semantic layer. It generates tags and a short human-readable briefing; it is not described in the repository as a source of the final collision/near-miss probability.

### 14.1 Documented model and visual input configuration

The service startup configuration specifies:

```text
base model: Qwen/Qwen2.5-VL-7B-Instruct
adapter directory: model_adapter/deploy_24f_r8_e1_clean
frames presented by the API pipeline: 16
minimum pixels: 50,176
maximum pixels: 102,400
```

The adapter directory name is recorded exactly as documented. The repository does not contain adapter metadata or training code, so the meaning of `24f_r8_e1`, the adapter technique, trainable modules, optimiser, learning rate, dataset, and checkpoint-selection rule cannot be verified here.

The API accepts a video and optional fields:

```text
clip_id
provided_outcome: collision | near_miss | safe | auto
event_time_sec
include_diagnostics
```

When the dedicated outcome model or backend already knows the outcome, the documented production recommendation is to pass `provided_outcome`. This cleanly separates numerical outcome detection from semantic description.

### 14.2 Semantic outputs

Qwen produces a structured `traffic_event.v1` response with:

- `briefing.summary`: concise event summary;
- `classification.primary_type`;
- `classification.scenario_tags` and `classification.type_tags`;
- primary actor type and position;
- environment: lighting, weather, road surface, road type, and visibility issues;
- behaviour: ego action, other-actor action, and who avoided the conflict;
- collision target, geometry, and severity when the outcome is `collision`;
- near-miss type and severity when the outcome is `near_miss`;
- risk, review, and runtime information.

The allowed scenario/type vocabulary documented by the API is:

```text
front_vehicle_hard_brake
rear_end_risk
cut_in_lane_change
intersection_crossing
turning_conflict
pedestrian_crossing
cyclist_conflict
motorcyclist_conflict
animal_crossing
passing_overtaking
oncoming_head_on
static_obstacle
roadwork_or_infrastructure
occlusion_emergence
loss_of_control
non_ego_visible_collision
normal_interaction_hard_negative
```

Actor types are `none`, `car`, `large_vehicle`, `pedestrian`, `cyclist`, `motorcyclist`, `animal`, or `infrastructure_or_static_object`. The environment taxonomy covers daylight/dawn/dark/glare, clear/cloudy/rain/snow/fog, dry/wet/snow-ice surfaces, multiple road families, and visibility issues such as blur, shake, low light, glare, occlusion, windshield rain, and snow/fog.

For collision events, the schema includes geometries such as front-to-rear, front-to-side, head-on, same/opposite-direction sideswipe, rear impact, fixed object, vulnerable-road-user/animal, and off-road/rollover. For near misses, it includes close following, crossing, cut-in, vulnerable-road-user/animal close pass, static-obstacle avoidance, and loss-of-control avoidance.

### 14.3 Documented semantic evaluation

The API README records a 26-example evaluation:

```text
primary_actor:                    26/26 = 1.0000
primary_type:                     25/26 = 0.9615
lighting:                         26/26 = 1.0000
weather:                          26/26 = 1.0000
road_family:                      26/26 = 1.0000
collision_target_when_collision:  18/18 = 1.0000
request latency mean / max:       12.22 s / 14.88 s
```

The repository does not include the 26-example evaluation set, evaluator implementation, per-field matching rules, generation prompt, temperature, top-p, maximum generated tokens, or failure analysis. The numbers should therefore be reported as documented service metrics, not independently reproduced metrics.

### 14.4 Repository limitation for Qwen reproducibility

The referenced server lives at:

```text
/mnt/nexar_qwen_api/nexar_traffic_event_api_clean_20260606/api/qwen_api_server.py
```

That implementation and `model_adapter/deploy_24f_r8_e1_clean` are not in this repository. The local [`backend/main.py`](backend/main.py) contains only a health endpoint. Therefore, the following details cannot be asserted from this repository:

- exact frame-sampling positions;
- prompt template and system instructions;
- Qwen processor/Transformers versions;
- adapter architecture or fine-tuning method;
- training/validation datasets;
- decoding strategy;
- JSON repair, schema validation, retry, or hallucination-control logic;
- whether the summary and every field are produced in one generation or post-processed across stages.

## 15. Earlier and alternative ML work

These components exist in the repository but are not the selected final path.

### 15.1 Wavelet-Shake Transformer

Implementation: [`doc/nexar_wst/src/model_wst.py`](doc/nexar_wst/src/model_wst.py).

The Wavelet-Shake Transformer patchifies the `raw + CWT + SWT` time sequence, projects flattened temporal patches into a learned embedding, prepends a learned class token, adds learned positional embeddings, and applies a pre-normalised `nn.TransformerEncoder`. A separate MLP embeds handcrafted features; the class-token and handcrafted embeddings are concatenated, projected, and classified.

The documented default experiment uses 300 frames, 20 raw channels, `12 x 48` CWT channels, `12 x 5` SWT channels, four transformer layers, `d_model=128`, four attention heads, feed-forward dimension 256, dropout `0.15`, patch length/stride `4`, AdamW, class-weighted cross entropy, label smoothing, gradient clipping, and early stopping on a validation composite dominated by macro-F1. A supervised contrastive-loss implementation exists, but the checked configuration sets its weight to `0.0`.

Handcrafted-feature baselines include balanced logistic regression and a 300-tree depth-4 balanced random forest. The final reported package moved to tabular rescue fusion rather than using this transformer as the selected outcome classifier.

### 15.2 Standalone impact and residual-flow heads

The package includes standalone impact-physics, deep-impulse, object-CoTracker, and residual-flow training scripts. They evaluate logistic regression, histogram gradient boosting, Extra Trees, XGBoost, LightGBM, and related fusion rules. They document the experiment path that led to the final rescue design; their outputs are not the selected `val_selected_deep_rescue` artifact unless explicitly referenced by the final training script.

### 15.3 SAVeD dataset preparation

The repository notes that SAVeD/AV data was downloaded and clipped, with 1,020 collision clips and 586 near-miss clips after preparation. It was **not used** in the reported 93.96% Nexar fixed-split result.

## 16. Final reported metrics

The selected fixed-split result is stored in [`val_selected_deep_rescue_summary.json`](collision_contact_full_repro/outputs/collision_contact_model/val_selected_deep_rescue_summary.json):

```text
selected model: lgbm_k512/rescue_q0.371/w2.50/fixed0p5
decision threshold: 0.5

validation accuracy: 0.915966386555
validation macro-F1: 0.915674603175
validation AUROC: 0.968537414966

test accuracy: 0.939597315436
test macro-F1: 0.939324012488
test balanced accuracy: 0.938969258590
test AUROC: 0.964556962025
test log loss: 0.263939987181
test confusion matrix: [[75, 4], [5, 65]]
```

The confusion-matrix order is `[near_miss, collision]`. The model makes nine errors on 149 test clips: four near misses predicted as collisions and five collisions predicted as near misses.

## 17. Saved ML artifacts

### 17.1 Final rescue artifacts

```text
collision_contact_full_repro/outputs/collision_contact_model/
  val_selected_deep_rescue_models.joblib
  val_selected_deep_rescue_summary.json
  val_selected_deep_rescue_predictions.json
  val_selected_deep_rescue_errors.json
  val_selected_deep_rescue_probabilities.npz
```

The Joblib artifact stores fitted candidate models, all 4,063 feature names, the selected fusion metadata, and training arguments. The prediction JSON stores `path`, true label, collision probability, predicted label, and correctness for the fixed test set.

### 17.2 Precomputed direct features

```text
collision_contact_full_repro/outputs/processed_744/features/
collision_contact_full_repro/outputs/processed_744/
  object_cotracker_dynamics_yolov8s_32f_w384_20260524/
collision_contact_full_repro/analysis/impact_diagnostics_20260522/
```

There are 744 motion/wavelet feature files, 744 motion CSVs, 744 CoTracker files, and train/test object-metrics CSVs.

### 17.3 Released anchor

```text
collision_contact_full_repro/assets/released_anchor/
  strong_fusion_probabilities.npz
  long_context_oof_experts_summary.json
```

The normal reproduction path copies these files into `outputs/processed_744_long_context_anchor/`; it does not retrain DINO, VideoMAE, RAFT, YOLO, BADAS, or the long-context experts.

## 18. Advantages of the method

### 18.1 Lightweight final decision stage compared with VLM generation

The final outcome decision is computationally small once features and the anchor probability are available. It does not run an autoregressive vision-language decoder to decide `near_miss` versus `collision`. Instead, it:

1. reads a fixed 4,063-dimensional numerical vector;
2. applies imputation and `sign(x) * log(1 + |x|)`;
3. retains 512 features with a pre-fitted ANOVA F-score selector;
4. evaluates a LightGBM classifier with 340 trees and seven leaves per tree;
5. applies one scalar rescue equation to the cached anchor probability.

This makes the **final classifier stage** substantially smaller than invoking the documented 7-billion-parameter Qwen 2.5-VL model for outcome classification. It also avoids autoregressive token generation for the binary decision. The design reserves Qwen 2.5-VL for the task where a VLM is useful: producing summaries and semantic tags.

This advantage must be stated precisely. The repository does not contain an end-to-end latency, FLOP, energy, or memory benchmark comparing the complete raw-video outcome pipeline with Qwen. A full feature rebuild still runs several neural extractors, including DINO/DINOv3, VideoMAE, RAFT, YOLO, BADAS/V-JEPA2, and CoTracker. Therefore, the evidence supports a low-cost **downstream decision head** and efficient reuse of cached features, but not a measured claim that first-time extraction of the entire outcome stack is always cheaper than one VLM request.

### 18.2 Feature extraction can be cached and reused

The expensive visual and tracking operations are separated from classifier training. DINO, VideoMAE, RAFT, YOLO, BADAS, motion/wavelet, and CoTracker outputs are saved as `.npz` feature files. This provides two practical advantages:

- classifier experiments do not repeatedly decode every video or rerun every backbone;
- the final LightGBM head can be retrained and evaluated from precomputed features without raw videos or model downloads.

The repository's fast-reproduction path explicitly skips raw-video feature extraction and trains from the saved feature assets. This is particularly useful when testing feature selection, tree-model hyperparameters, thresholds, or rescue-fusion rules.

### 18.3 Specialised models are used for specialised evidence

The system does not ask one general-purpose model to infer every property implicitly. Each component contributes evidence aligned with its implementation:

- DINO/DINOv3 encode frame appearance;
- VideoMAE encodes spatiotemporal visual content;
- RAFT measures dense optical flow;
- YOLO identifies risk-relevant actors and regions;
- CoTracker follows points inside selected object boxes;
- BADAS/V-JEPA2 supplies learned risk-oriented video features;
- affine motion and wavelets expose camera shake and short impact transients;
- Qwen 2.5-VL converts visual evidence into human-readable semantics.

This modularity allows the collision/contact classifier to use numerical motion and physical-contact evidence while the VLM handles open-ended language generation.

### 18.4 Physically interpretable intermediate signals

Many direct final features have an explicit meaning: residual displacement after ego-motion removal, acceleration, jerk, wavelet-band energy, object-box shift, scale/area change, point-track visibility, shape change, impulse score, and post-impact ring-down. The final artifact also stores all 4,063 feature names.

These signals are easier to inspect during error analysis than an outcome produced only as generated text. The repository can trace a prediction back to motion CSVs, feature arrays, object metrics, CoTracker summaries, the anchor probability, and the rescue probability.

### 18.5 Stable structured outcome interface

The outcome branch produces a scalar `prob_collision` and applies an explicit threshold. Its class mapping and fusion equation are fixed in code. Semantic generation remains downstream and can receive `provided_outcome`, so wording or tag-generation behaviour does not determine the numerical collision decision.

This separation also permits independent evaluation:

- outcome quality is measured with accuracy, balanced accuracy, macro-F1, AUROC, log loss, and a confusion matrix;
- Qwen semantics are measured field by field for actor, scenario type, lighting, weather, road family, and collision target.

### 18.6 Rescue fusion preserves a strong baseline unless extra evidence is sufficient

The selected fusion is not an unrestricted average of the anchor and LightGBM probabilities. When the LightGBM probability `q` is at or below the rescue point `0.371`, the rescue delta is zero and the anchor is unchanged. Only stronger LightGBM evidence increases the collision logit.

This is a useful property for a specialised contact detector: the additional physics/CoTracker head is used to recover collision cases only when its evidence crosses the validation-selected activation point. The final decision remains an explicit, auditable formula rather than an opaque second-stage generation.

### 18.7 Works with a relatively compact labelled outcome dataset

The final supervised outcome heads are trained on the fixed 744-video labelled set rather than fine-tuning all visual backbones. The pretrained extractors remain frozen, and only compact downstream classifiers are fitted. This reduces the number of trainable parameters in the project-specific outcome model and makes the fixed-split experiment feasible without training a large video or vision-language backbone from scratch.

This is an architectural advantage, not evidence that the method is universally more data-efficient than every VLM approach; the repository contains no controlled data-scaling comparison.

## 19. Reproducibility and interpretation constraints

1. **Frozen feature models versus learned heads**: DINO, DINOv3, VideoMAE, RAFT, YOLO, CoTracker, and BADAS are loaded from pretrained weights for extraction. The repository trains downstream tabular/temporal heads, not these backbones.
2. **Final direct inputs versus anchor ancestry**: the final LightGBM directly reads motion/wavelet/object/CoTracker features. DINO, VideoMAE, RAFT, YOLO-interaction, and BADAS influence it through the released anchor probability.
3. **Offline context**: long-context features include post-event observations and metadata. Reported accuracy must not be presented as pre-event crash forecasting accuracy.
4. **Fixed-split anchor asset**: the released anchor is indexed to the existing train/test split. It is not sufficient for inference on an arbitrary new video.
5. **No production single-video outcome wrapper in this repository**: new-video inference still requires identical clip creation, feature extraction, 4,063-feature ordering, anchor generation, and rescue application.
6. **Qwen is semantically separate**: passing `provided_outcome` lets Qwen describe an upstream outcome; Qwen should not be credited with the LightGBM collision/contact metrics.
7. **Qwen reproducibility is partial**: only the API contract and runtime configuration are present locally; server and adapter implementation details are external.
8. **SAVeD is not part of the final metric**: it must not be included in the reported final training-set count.

## 20. Primary source-file map

| Topic | Primary repository evidence |
|---|---|
| Overall final method and data | [`README.md`](README.md), [`ML_PIPELINE_CONTEXT_AND_PROGRESS.md`](ML_PIPELINE_CONTEXT_AND_PROGRESS.md), [`collision_contact_full_repro/README.md`](collision_contact_full_repro/README.md) |
| Full feature-extraction invocation | [`run_01_extract_features.sh`](collision_contact_full_repro/scripts/run_01_extract_features.sh) |
| Motion estimation | [`motion_extract.py`](collision_contact_full_repro/collision_contact/motion_extract.py) |
| CWT/SWT construction | [`wavelet_features.py`](collision_contact_full_repro/collision_contact/wavelet_features.py), [`wst_processed_744_small.yaml`](collision_contact_full_repro/configs/wst_processed_744_small.yaml) |
| DINOv2/DINOv3 | [`extract_dino_features.py`](collision_contact_full_repro/collision_contact/extract_dino_features.py) |
| VideoMAE | [`extract_videomae_features.py`](collision_contact_full_repro/collision_contact/extract_videomae_features.py) |
| RAFT | [`extract_raft_features.py`](collision_contact_full_repro/collision_contact/extract_raft_features.py) |
| YOLO interaction | [`extract_yolo_interaction_features.py`](collision_contact_full_repro/collision_contact/extract_yolo_interaction_features.py) |
| YOLO object residuals | [`extract_object_residual_physics_features.py`](collision_contact_full_repro/collision_contact/extract_object_residual_physics_features.py), [`export_object_metrics.py`](collision_contact_full_repro/collision_contact/export_object_metrics.py) |
| CoTracker | [`extract_object_cotracker_dynamics_features.py`](collision_contact_full_repro/collision_contact/extract_object_cotracker_dynamics_features.py) |
| BADAS/V-JEPA2 | [`legacy/extract_badas_window_features.py`](collision_contact_full_repro/legacy/extract_badas_window_features.py) |
| Strong event fusion | [`train_strong_fusion.py`](collision_contact_full_repro/collision_contact/train_strong_fusion.py) |
| Long context and anchor | [`extract_long_context_features.py`](collision_contact_full_repro/collision_contact/extract_long_context_features.py), [`export_long_context_oof_experts.py`](collision_contact_full_repro/collision_contact/export_long_context_oof_experts.py) |
| Final 4,063 features | [`train_impact_physics_head.py`](collision_contact_full_repro/collision_contact/train_impact_physics_head.py), [`train_deep_impulse_physics_head.py`](collision_contact_full_repro/collision_contact/train_deep_impulse_physics_head.py), [`train_deep_impulse_rescue_fusion.py`](collision_contact_full_repro/collision_contact/train_deep_impulse_rescue_fusion.py) |
| Final selection and report | [`train_val_selected_deep_rescue.py`](collision_contact_full_repro/collision_contact/train_val_selected_deep_rescue.py), [`val_selected_deep_rescue_summary.json`](collision_contact_full_repro/outputs/collision_contact_model/val_selected_deep_rescue_summary.json) |
| Qwen 2.5-VL semantics | [`API_USAGE_README.md`](API_USAGE_README.md) |
