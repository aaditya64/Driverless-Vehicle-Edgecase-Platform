#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" -m collision_contact.check_data
"$PYTHON_BIN" -m collision_contact.prepare_assets
