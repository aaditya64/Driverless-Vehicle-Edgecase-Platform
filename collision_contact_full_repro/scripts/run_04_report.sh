#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" -m collision_contact.print_report \
  --summary outputs/collision_contact_model/val_selected_deep_rescue_summary.json
