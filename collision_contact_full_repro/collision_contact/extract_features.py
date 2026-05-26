from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .common import ensure_dirs, load_config, read_split_csv, stable_video_id, write_json
from .motion_extract import extract_global_motion
from .visualize import save_motion_report
from .wavelet_features import build_sequence_features


def feature_path(feature_dir: str | Path, video_path: str | Path) -> Path:
    return Path(feature_dir) / f"{stable_video_id(video_path)}.npz"


def motion_csv_path(feature_dir: str | Path, video_path: str | Path) -> Path:
    return Path(feature_dir) / f"{stable_video_id(video_path)}_motion.csv"


def report_png_path(report_dir: str | Path, video_path: str | Path) -> Path:
    return Path(report_dir) / "motion" / f"{stable_video_id(video_path)}.png"


def extract_one(video_path: str | Path, cfg: dict, force: bool = False) -> Path:
    fdir = Path(cfg["paths"]["feature_dir"])
    rdir = Path(cfg["paths"]["report_dir"])
    ensure_dirs(fdir, rdir / "motion")
    out = feature_path(fdir, video_path)
    motion_out = motion_csv_path(fdir, video_path)
    if out.exists() and motion_out.exists() and not force:
        return out
    df = extract_global_motion(video_path, cfg)
    features = build_sequence_features(df, cfg)
    np.savez_compressed(out, **features, video_path=str(video_path))
    df.to_csv(motion_out, index=False)
    if bool(cfg.get("features", {}).get("save_motion_reports", True)):
        save_motion_report(df, features, report_png_path(rdir, video_path), title=str(video_path))
    return out


def _extract_worker(args: tuple[str, dict, bool]) -> str:
    path, cfg, force = args
    return str(extract_one(path, cfg, force=force))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wst_3070_8gb.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = read_split_csv(cfg["paths"]["train_csv"]) + read_split_csv(cfg["paths"]["test_csv"])
    seen = []
    for row in rows:
        if row["path"] not in seen:
            seen.append(row["path"])
    if args.limit > 0:
        seen = seen[: args.limit]
    outputs = []
    if args.workers <= 1:
        for path in tqdm(seen, desc="extract"):
            outputs.append(str(extract_one(path, cfg, force=args.force)))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_extract_worker, (path, cfg, args.force)) for path in seen]
            for future in tqdm(as_completed(futures), total=len(futures), desc="extract"):
                outputs.append(future.result())
    write_json(Path(cfg["paths"]["report_dir"]) / "feature_manifest.json", {"features": outputs})
    print(f"extracted {len(outputs)} feature files")


if __name__ == "__main__":
    main()
