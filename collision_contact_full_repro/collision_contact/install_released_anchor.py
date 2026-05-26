"""Install the validated long-context risk anchor used by the collision head."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="assets/released_anchor")
    parser.add_argument("--out-dir", default="outputs/processed_744_long_context_anchor")
    args = parser.parse_args()

    source = Path(args.source_dir)
    target = Path(args.out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in ["strong_fusion_probabilities.npz", "long_context_oof_experts_summary.json"]:
        src = source / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, target / name)
    print(f"installed anchor to {target}", flush=True)


if __name__ == "__main__":
    main()
