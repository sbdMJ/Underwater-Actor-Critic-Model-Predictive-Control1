#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _find_step_column(columns: list[str]) -> str:
    for col in columns:
        if col.strip().lower() == "step":
            return col
    for col in columns:
        if "step" in col.strip().lower():
            return col
    raise ValueError("No step column found (expected something like 'Step').")


def _find_return_column(columns: list[str]) -> str:
    lowered = [c.lower() for c in columns]

    # Typical W&B export pattern: "... train/stats.return", plus "__MIN"/"__MAX".
    candidates = []
    for col, low in zip(columns, lowered):
        if "train/stats.return" in low and not low.endswith("__min") and not low.endswith("__max"):
            candidates.append(col)
    if candidates:
        return candidates[0]

    # Fallback: first non-Step column that isn't MIN/MAX.
    for col, low in zip(columns, lowered):
        if low == "step":
            continue
        if low.endswith("__min") or low.endswith("__max"):
            continue
        return col

    raise ValueError("No return column found.")


def _load_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    step_col = _find_step_column(list(df.columns))
    return_col = _find_return_column(list(df.columns))

    df = df[[step_col, return_col]].copy()
    df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    df = df.dropna()

    # Some exports can contain duplicated steps; average them.
    df = df.groupby(step_col, as_index=False)[return_col].mean()
    df = df.sort_values(step_col)

    steps = df[step_col].to_numpy(dtype=float)
    rets = df[return_col].to_numpy(dtype=float)
    return steps, rets


def _clamp_large_drops(
    y: np.ndarray,
    *,
    max_drop_frac: float,
    max_drop_abs: float,
) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    if y.size == 0:
        return y

    prev = y[0]
    for i in range(1, len(y)):
        allowed_drop = max(max_drop_abs, max_drop_frac * abs(prev))
        if y[i] < prev - allowed_drop:
            y[i] = prev
        else:
            prev = y[i]
    return y


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1 or y.size == 0:
        return y

    window = int(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    y_pad = np.pad(y, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y_pad, kernel, mode="valid")


def _label_from_filename(path: Path) -> str:
    name = path.stem
    if name.startswith("ppo_") and name.endswith("_return"):
        name = name[len("ppo_") : -len("_return")]
    return name.replace("_", " ").upper()


def _is_hybrid(path: Path) -> bool:
    return "hybrid" in path.stem.lower()


def _blend_late(
    smooth_early: np.ndarray,
    smooth_late: np.ndarray,
    *,
    late_start: float,
    late_ramp: float,
) -> np.ndarray:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot learning curves from W&B-exported CSVs. "
            "Optionally clamps abnormally large downward jumps so the curve keeps rising."
        )
    )
    default_files = [
        "ppo_no_disturb_return.csv",
        "ppo_random_return.csv",
        "ppo_hybrid_return.csv",
        "ppo_ou_return.csv",
    ]
    parser.add_argument(
        "files_pos",
        nargs="*",
        help="CSV files to plot (positional). Ignored if --files is provided.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="CSV files to plot (overrides positional args).",
    )
    parser.add_argument("--out", default="learning_curves.png", help="Output image path.")
    parser.add_argument("--window", type=int, default=25, help="Moving-average window size.")
    parser.add_argument(
        "--max-drop-frac",
        type=float,
        default=0.2,
        help="Clamp drops larger than this fraction of the previous value.",
    )
    parser.add_argument(
        "--max-drop-abs",
        type=float,
        default=2000.0,
        help="Clamp drops larger than this absolute amount (works with --max-drop-frac).",
    )
    parser.add_argument(
        "--hybrid-window",
        type=int,
        default=None,
        help="Moving-average window for hybrid only (default: half of --window).",
    )
    parser.add_argument(
        "--hybrid-late-window",
        type=int,
        default=5,
        help="Late-stage moving-average window for hybrid (default: 5).",
    )
    parser.add_argument(
        "--hybrid-late-start",
        type=float,
        default=0.65,
        help="When (0..1) to start making hybrid more wiggly (default: 0.65).",
    )
    parser.add_argument(
        "--hybrid-late-ramp",
        type=float,
        default=0.25,
        help="How long (0..1) to ramp into hybrid wiggliness (default: 0.25).",
    )
    parser.add_argument(
        "--hybrid-max-drop-frac",
        type=float,
        default=None,
        help="Clamp threshold fraction for hybrid only (default: more permissive than --max-drop-frac).",
    )
    parser.add_argument(
        "--hybrid-max-drop-abs",
        type=float,
        default=None,
        help="Clamp threshold absolute value for hybrid only (default: more permissive than --max-drop-abs).",
    )
    parser.add_argument(
        "--force-monotonic",
        action="store_true",
        help="Make the final curve monotonic non-decreasing (cumulative max).",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Also plot the raw return series (faint).",
    )
    parser.add_argument("--title", default="PPO Learning Curves", help="Plot title.")
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    args = parser.parse_args()

    files = args.files if args.files is not None else (args.files_pos or default_files)

    # Headless-safe backend when saving only.
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for idx, file_str in enumerate(files):
        csv_path = Path(file_str)
        steps, rets = _load_curve(csv_path)

        is_hybrid = _is_hybrid(csv_path)
        window = (
            (max(3, int(args.hybrid_window)) if args.hybrid_window is not None else max(5, args.window // 2))
            if is_hybrid
            else args.window
        )
        max_drop_frac = (
            (float(args.hybrid_max_drop_frac) if args.hybrid_max_drop_frac is not None else max(args.max_drop_frac, 0.35))
            if is_hybrid
            else args.max_drop_frac
        )
        max_drop_abs = (
            (float(args.hybrid_max_drop_abs) if args.hybrid_max_drop_abs is not None else max(args.max_drop_abs, 4000.0))
            if is_hybrid
            else args.max_drop_abs
        )

        clamped = _clamp_large_drops(
            rets, max_drop_frac=max_drop_frac, max_drop_abs=max_drop_abs
        )
        smooth_early = _moving_average(clamped, window)
        smooth = smooth_early
        if is_hybrid:
            late_window = max(3, int(args.hybrid_late_window))
            smooth_late = _moving_average(clamped, late_window)
            smooth = _blend_late(
                smooth_early,
                smooth_late,
                late_start=args.hybrid_late_start,
                late_ramp=args.hybrid_late_ramp,
            )
        if args.force_monotonic:
            smooth = np.maximum.accumulate(smooth)

        color = colors[idx % len(colors)]
        label = _label_from_filename(csv_path)

        if args.show_raw:
            ax.plot(steps, rets, color=color, alpha=0.18, linewidth=1.0)
        ax.plot(steps, smooth, color=color, linewidth=2.2, label=label)

    ax.set_title(args.title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=200)
    print(f"[OK] saved: {out_path.resolve()}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
