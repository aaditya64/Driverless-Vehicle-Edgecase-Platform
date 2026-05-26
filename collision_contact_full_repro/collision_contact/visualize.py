from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_motion_report(df: pd.DataFrame, feature_npz: dict, out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time = df["time"].to_numpy()
    freqs = feature_npz["freqs"]
    cwt = feature_npz["cwt"]
    n_freqs = len(freqs)
    n_channels = len(feature_npz["wavelet_channels"])
    cwt3 = cwt.reshape(cwt.shape[0], n_channels, n_freqs)
    scalogram = cwt3.mean(axis=1).T

    fig, axes = plt.subplots(5, 1, figsize=(13, 12), constrained_layout=True)
    fig.suptitle(title)
    axes[0].plot(time, df["dx"], label="dx")
    axes[0].plot(time, df["dy"], label="dy")
    axes[0].plot(time, df["theta_px"], label="theta_px")
    axes[0].set_title("Frame-to-frame global affine motion")
    axes[0].legend(loc="upper right")

    axes[1].plot(time, df["x_res"], label="x_res")
    axes[1].plot(time, df["y_res"], label="y_res")
    axes[1].plot(time, df["theta_res_px"], label="theta_res_px")
    axes[1].set_title("Detrended camera-shake residual")
    axes[1].legend(loc="upper right")

    axes[2].plot(time, df["jerk_x"], label="jerk_x")
    axes[2].plot(time, df["jerk_y"], label="jerk_y")
    axes[2].plot(time, df["jerk_theta_px"], label="jerk_theta_px")
    axes[2].plot(time, df["jerk_energy"], label="jerk_energy", linewidth=2.0)
    axes[2].set_title("Jerk and jerk energy")
    axes[2].legend(loc="upper right")

    axes[3].plot(time, df["inlier_ratio"], label="inlier_ratio")
    axes[3].plot(time, df["fit_error"], label="fit_error")
    axes[3].set_title("Tracking quality")
    axes[3].legend(loc="upper right")

    extent = [float(time[0]), float(time[min(len(time) - 1, scalogram.shape[1] - 1)]), float(freqs[0]), float(freqs[-1])]
    im = axes[4].imshow(scalogram, aspect="auto", origin="lower", extent=extent, cmap="magma")
    axes[4].set_title("Mean CWT log-energy scalogram")
    axes[4].set_xlabel("time (s)")
    axes[4].set_ylabel("frequency (Hz)")
    fig.colorbar(im, ax=axes[4], label="log-energy")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

