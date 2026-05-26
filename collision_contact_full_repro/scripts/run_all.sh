#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

bash scripts/run_00_check_and_prepare.sh
bash scripts/run_01_extract_features.sh
bash scripts/run_02_train_anchor.sh
bash scripts/run_03_train_model.sh
bash scripts/run_04_report.sh
