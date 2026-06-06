"""Print the selected model metrics from a completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/collision_contact_model/val_selected_deep_rescue_summary.json")
    args = parser.parse_args()

    data = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    selected = data["selected_by_val"]
    test = selected["test"]
    val = selected["val"]
    print(f"selected={selected['name']}")
    print(f"threshold={selected['threshold']}")
    print(f"val_accuracy={val['accuracy']:.12f}")
    print(f"val_macro_f1={val['macro_f1']:.12f}")
    print(f"test_accuracy={test['accuracy']:.12f}")
    print(f"test_auroc={test['auroc']:.12f}")
    print(f"test_confusion={test['confusion_matrix']}")


if __name__ == "__main__":
    main()
