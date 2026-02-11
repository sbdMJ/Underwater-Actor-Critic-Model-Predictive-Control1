#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Curve:
    steps: np.ndarray
    mean: np.ndarray
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None


def _find_step_column(columns: list[str]) -> str:
    for col in columns:
        if col.strip().lower() == "step":
            return col
    for col in columns:
        if "step" in col.strip().lower():
            return col
    raise ValueError("No step column found (expected something like 'Step').")


def _find_return_mean_column(columns: list[str]) -> str:
    lowered = [c.lower() for c in columns]
    for col, low in zip(columns, lowered):
        if "train/stats.return" in low and not low.endswith("__min") and not low.endswith("__max"):
            return col
    for col, low in zip(columns, lowered):
        if low == "step":
            continue
        if low.endswith("__min") or low.endswith("__max"):
            continue
        return col
    raise ValueError("No return column found.")


def _find_bounds(columns: list[str], mean_col: str) -> Tuple[Optional[str], Optional[str]]:
    min_col = f"{mean_col}__MIN"
    max_col = f"{mean_col}__MAX"
    colset = set(columns)
    low = min_col if min_col in colset else None
    high = max_col if max_col in colset else None
    return low, high


def _load_curve(csv_path: Path) -> Curve:
    df = pd.read_csv(csv_path)
    step_col = _find_step_column(list(df.columns))
    mean_col = _find_return_mean_column(list(df.columns))
    low_col, high_col = _find_bounds(list(df.columns), mean_col)

    cols = [step_col, mean_col] + ([low_col] if low_col else []) + ([high_col] if high_col else [])
    df = df[cols].copy()
    df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
    df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")
    if low_col:
        df[low_col] = pd.to_numeric(df[low_col], errors="coerce")
    if high_col:
        df[high_col] = pd.to_numeric(df[high_col], errors="coerce")
    df = df.dropna(subset=[step_col, mean_col])

    agg = {mean_col: "mean"}
    if low_col:
        agg[low_col] = "min"
    if high_col:
        agg[high_col] = "max"
    df = df.groupby(step_col, as_index=False).agg(agg).sort_values(step_col)

    steps = df[step_col].to_numpy(dtype=float)
    mean = df[mean_col].to_numpy(dtype=float)
    low = df[low_col].to_numpy(dtype=float) if low_col else None
    high = df[high_col].to_numpy(dtype=float) if high_col else None
    return Curve(steps=steps, mean=mean, low=low, high=high)


def _clamp_large_drops(y: np.ndarray, *, max_drop_frac: float, max_drop_abs: float) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    if y.size == 0:
        return y

    prev = y[0]
    for i in range(1, len(y)):
        allowed_drop = max(float(max_drop_abs), float(max_drop_frac) * abs(prev))
        if y[i] < prev - allowed_drop:
            y[i] = prev
        else:
            prev = y[i]
    return y


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    window = int(window)
    if window <= 1 or y.size == 0:
        return y

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    y_pad = np.pad(y, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y_pad, kernel, mode="valid")


def _is_hybrid(path: Path) -> bool:
    return "hybrid" in path.stem.lower()


def _blend_late(smooth_early: np.ndarray, smooth_late: np.ndarray, *, late_start: float, late_ramp: float) -> np.ndarray:
    smooth_early = np.asarray(smooth_early, dtype=float)
    smooth_late = np.asarray(smooth_late, dtype=float)
    if smooth_early.shape != smooth_late.shape:
        raise ValueError("blend inputs must have the same shape")
    if smooth_early.size <= 1:
        return smooth_early

    late_start = float(np.clip(late_start, 0.0, 1.0))
    late_ramp = float(max(late_ramp, 1e-9))
    t = np.linspace(0.0, 1.0, smooth_early.size)
    w = np.clip((t - late_start) / late_ramp, 0.0, 1.0)
    return smooth_early * (1.0 - w) + smooth_late * w


def _pretty_label(path: Path) -> str:
    stem = path.stem.lower()
    mapping = {
        "ppo_no_disturb_return": "No disturbances",
        "ppo_random_return": "Random disturbances",
        "ppo_hybrid_return": "Hybrid disturbances",
        "ppo_ou_return": "OU disturbances",
    }
    if stem in mapping:
        return mapping[stem]
    name = path.stem
    if name.startswith("ppo_") and name.endswith("_return"):
        name = name[len("ppo_") : -len("_return")]
    return name.replace("_", " ").title()


def _default_files() -> list[str]:
    return [
        "ppo_no_disturb_return.csv",
        "ppo_random_return.csv",
        "ppo_hybrid_return.csv",
        "ppo_ou_return.csv",
    ]


def _sample_xy(x: np.ndarray, y: np.ndarray, every: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= 1:
        return x, y
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape for sampling")

    every = int(every)
    if every <= 1:
        return x, y

    idx = np.arange(0, x.size, every, dtype=int)
    if idx[-1] != x.size - 1:
        idx = np.append(idx, x.size - 1)
    return x[idx], y[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-quality learning curve plotter for W&B-exported CSVs.")
    parser.add_argument("files", nargs="*", help="CSV files to plot (positional).")
    parser.add_argument("--out", default="learning_curves_paper.png", help="Output path (pdf/svg/png).")
    parser.add_argument("--png", default=None, help="Also save a PNG copy to this path.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG output.")

    parser.add_argument("--xlabel", default="Training step", help="X-axis label.")
    parser.add_argument("--ylabel", default="Episode return", help="Y-axis label.")
    parser.add_argument("--title", default=None, help="Optional title (omit for paper).")

    parser.add_argument(
        "--marker",
        default="none",
        help="Marker style (use 'none' to draw lines only).",
    )
    parser.add_argument(
        "--markevery",
        type=int,
        default=None,
        help="Draw a marker every N points (default: auto; use 0 to disable markers).",
    )
    parser.add_argument("--markersize", type=float, default=3.0, help="Marker size.")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Plot line using every Nth point (1 = no sampling).",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Moving-average window. Use 1 to plot raw CSV values (default: 1).",
    )
    parser.add_argument(
        "--clamp-drops",
        action="store_true",
        help="Clamp abnormally large downward jumps (off by default; keeps raw values).",
    )
    parser.add_argument("--max-drop-frac", type=float, default=0.2, help="Clamp drops larger than this fraction.")
    parser.add_argument(
        "--max-drop-abs",
        type=float,
        default=2000.0,
        help="Clamp drops larger than this absolute amount.",
    )

    parser.add_argument(
        "--hybrid-window",
        type=int,
        default=None,
        help="Moving-average window for hybrid (only used if --window > 1).",
    )
    parser.add_argument(
        "--hybrid-late-window",
        type=int,
        default=None,
        help="Late-stage window for hybrid (enable by setting >1).",
    )
    parser.add_argument("--hybrid-late-start", type=float, default=0.65, help="When (0..1) to start hybrid wiggliness.")
    parser.add_argument("--hybrid-late-ramp", type=float, default=0.25, help="Ramp length (0..1) into hybrid wiggliness.")

    parser.add_argument("--no-band", action="store_true", help="Disable min/max band shading even if present.")

    parser.add_argument(
        "--size",
        choices=["single", "double", "custom"],
        default="single",
        help="Figure size preset: single column, double column, or custom.",
    )
    parser.add_argument("--width", type=float, default=3.45, help="Figure width in inches (custom size).")
    parser.add_argument("--height", type=float, default=2.40, help="Figure height in inches (custom size).")

    args = parser.parse_args()

    files = args.files or _default_files()
    paths = [Path(f) for f in files]

    # Matplotlib setup (paper-friendly).
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 3,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 0.8,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )

    # Colorblind-friendly (Okabe-Ito).
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]

    if args.size == "single":
        figsize = (4.60, 2.40)
    elif args.size == "double":
        figsize = (7.0, 2.6)
    else:
        figsize = (float(args.width), float(args.height))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for idx, path in enumerate(paths):
        curve = _load_curve(path)
        is_hybrid = _is_hybrid(path)

        base_window = max(1, int(args.window))
        window = (
            (max(3, int(args.hybrid_window)) if args.hybrid_window is not None else max(3, base_window))
            if (is_hybrid and base_window > 1)
            else base_window
        )
        max_drop_frac = float(args.max_drop_frac)
        max_drop_abs = float(args.max_drop_abs)

        mean = curve.mean
        low = curve.low
        high = curve.high
        if args.clamp_drops:
            mean = _clamp_large_drops(mean, max_drop_frac=max_drop_frac, max_drop_abs=max_drop_abs)
            if low is not None:
                low = _clamp_large_drops(low, max_drop_frac=max_drop_frac, max_drop_abs=max_drop_abs)
            if high is not None:
                high = _clamp_large_drops(high, max_drop_frac=max_drop_frac, max_drop_abs=max_drop_abs)

        smooth = _moving_average(mean, window)
        if is_hybrid and base_window > 1 and args.hybrid_late_window is not None and int(args.hybrid_late_window) > 1:
            late_window = max(3, int(args.hybrid_late_window))
            smooth_late = _moving_average(mean, late_window)
            smooth = _blend_late(
                smooth,
                smooth_late,
                late_start=float(args.hybrid_late_start),
                late_ramp=float(args.hybrid_late_ramp),
            )

        color = palette[idx % len(palette)]
        linestyle = "-"
        label = _pretty_label(path)

        marker = None if str(args.marker).lower() in {"none", "null", "no"} else str(args.marker)
        if args.markevery is None:
            markevery = 1 if (marker is not None and int(args.sample_every) > 1) else 200
        else:
            if int(args.markevery) <= 0:
                marker = None
                markevery = None
            else:
                markevery = int(args.markevery)

        if marker is not None:
            if marker == ".":
                markerfacecolor = color
                markeredgecolor = color
                markeredgewidth = 0.0
            else:
                markerfacecolor = "white"
                markeredgecolor = color
                markeredgewidth = 0.8

        if not args.no_band and low is not None and high is not None:
            low_s = _moving_average(low, window)
            high_s = _moving_average(high, window)
            if low_s.shape == high_s.shape and low_s.size == smooth.size:
                band_mag = float(np.nanmax(high_s - low_s))
                if np.isfinite(band_mag) and band_mag > 1e-9:
                    steps_s, low_ss = _sample_xy(curve.steps, low_s, int(args.sample_every))
                    _, high_ss = _sample_xy(curve.steps, high_s, int(args.sample_every))
                    ax.fill_between(steps_s, low_ss, high_ss, color=color, alpha=0.12, linewidth=0.0)

        steps_p, smooth_p = _sample_xy(curve.steps, smooth, int(args.sample_every))
        plot_kwargs = {"color": color, "linestyle": linestyle, "label": label}
        if marker is not None:
            plot_kwargs.update(
                {
                    "marker": marker,
                    "markevery": markevery,
                    "markersize": float(args.markersize),
                    "markerfacecolor": markerfacecolor,
                    "markeredgecolor": markeredgecolor,
                    "markeredgewidth": markeredgewidth,
                }
            )
        ax.plot(steps_p, smooth_p, **plot_kwargs)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    if args.title:
        ax.set_title(args.title)

    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.minorticks_on()

    ax.legend(
        frameon=True,
        framealpha=0.9,
        loc="upper left",
        fontsize=3,
        borderpad=0.15,
        labelspacing=0.15,
        handlelength=1.2,
        handletextpad=0.35,
        borderaxespad=0.25,
    )

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path, dpi=int(args.dpi))
    else:
        fig.savefig(out_path)
    print(f"[OK] saved: {out_path.resolve()}")

    if args.png:
        png_path = Path(args.png)
        fig.savefig(png_path, dpi=int(args.dpi))
        print(f"[OK] saved: {png_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
