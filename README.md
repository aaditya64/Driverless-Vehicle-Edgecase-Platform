# Driverless Vehicle Edge-Case Platform

This workspace contains the current ML/data-preparation code for the driverless vehicle edge-case intelligence platform.

## What is tracked

- `scripts/download_badas_open.py`: downloads BADAS-Open from Hugging Face after access is approved.
- `scripts/inference_badas.py`: runs BADAS-Open risk timeline inference on a dashcam video.
- `scripts/create_event_clips.py`: cuts 10-second Nexar positive clips centred on `time_of_event`.
- `data/processed/clip_manifests/nexar_train_positive_event_clips.csv`: generated clip manifest for annotation.
- `doc/`: project planning documents.

Large local assets are intentionally not tracked:

- raw Nexar videos
- generated event clips
- downloaded BADAS/V-JEPA model weights
- inference outputs

## Setup

```bash
python -m pip install -r requirements.txt
```

For BADAS-Open, first request access on Hugging Face, then log in:

```bash
huggingface-cli login
python scripts/download_badas_open.py
```

## BADAS Inference

```bash
python scripts/inference_badas.py \
  --video data/nexar_collision_prediction/train/positive/00024.mp4 \
  --output-json outputs/badas_positive_00024.json \
  --output-csv outputs/badas_positive_00024.csv \
  --window-stride 16 \
  --device auto
```

BADAS-Open is currently used as a binary risk model:

```text
safe vs collision_or_near_miss
```

It does not distinguish `near_miss` from `collision` without additional labelled data and a separate classifier/head.

## Event-Centred Clip Generation

Create 10-second clips from Nexar positive videos:

```bash
python scripts/create_event_clips.py
```

The script uses `time_of_event` as the event centre and writes clips to:

```text
data/processed/event_clips/nexar/train/positive/
```

It also writes the annotation manifest:

```text
data/processed/clip_manifests/nexar_train_positive_event_clips.csv
```

The manifest starts with `core_label=needs_review` for positive samples. Human annotation should change this to:

```text
near_miss
collision
```
