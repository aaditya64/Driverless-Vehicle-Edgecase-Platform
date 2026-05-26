"""Download external model assets used by the reproduction pipeline."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _download_badas(target: Path) -> None:
    if (target / "weights" / "badas_open.pth").exists():
        print(f"BADAS asset exists: {target}", flush=True)
        return
    from huggingface_hub import snapshot_download

    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="nexar-ai/badas-open",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def _download_yolo() -> None:
    from ultralytics import YOLO

    for name in ["yolov8n.pt", "yolov8s.pt"]:
        YOLO(name)


def _clone_cotracker(target: Path) -> None:
    if (target / "hubconf.py").exists():
        print(f"CoTracker repo exists: {target}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "https://github.com/facebookresearch/co-tracker.git", str(target)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--badas-dir", default="models/BADAS-Open")
    parser.add_argument("--cotracker-dir", default="external/co-tracker")
    parser.add_argument("--skip-badas", action="store_true")
    parser.add_argument("--skip-cotracker", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    args = parser.parse_args()

    if not args.skip_badas:
        _download_badas(Path(args.badas_dir))
    if not args.skip_yolo:
        _download_yolo()
    if not args.skip_cotracker:
        _clone_cotracker(Path(args.cotracker_dir))
    print("assets ready", flush=True)


if __name__ == "__main__":
    main()
