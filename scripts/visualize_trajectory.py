from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np


def _load_npz(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)
    return {k: data[k] for k in data.files}


def _split_ranges_by_done(done: np.ndarray) -> list[tuple[int, int]]:
    """Split a 1D `done` array into contiguous [start, end) ranges."""
    done = np.asarray(done).reshape(-1).astype(bool, copy=False)
    n = int(done.shape[0])
    if n <= 0:
        return []

    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(n):
        if bool(done[i]):
            end = i + 1
            if end > start:
                ranges.append((start, end))
            start = end
    if start < n:
        ranges.append((start, n))
    return ranges


def plot_trajectory_3d(
    *,
    traj_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    heading_stride: int = 50,
    arrow_len: float = 0.25,
    show: bool = False,
    color_key: Optional[str] = "v_tan_err",
    color_label: Optional[str] = None,
    cmap: Optional[str] = None,
    path_color_mode: str = "default",
):
    traj_path = Path(traj_path).expanduser()
    out_path = Path(out_path).expanduser() if out_path else None

    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    pos_heading = _load_npz(traj_path)
    pos_data = pos_heading.get("pos", None)
    if pos_data is None:
        raise KeyError(f"Missing 'pos' in {traj_path}")
    pos = np.asarray(pos_data, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Invalid pos shape in {traj_path}: {None if pos is None else pos.shape}")

    heading = np.asarray(pos_heading.get("heading", np.zeros_like(pos)), dtype=np.float32)
    if heading.shape != pos.shape:
        heading = np.zeros_like(pos)

    color = None
    if color_key:
        color_raw = pos_heading.get(str(color_key), None)
        if color_raw is not None:
            try:
                color = np.asarray(color_raw, dtype=np.float32).reshape(-1)
            except Exception:
                color = None
        if color is not None and color.shape[0] != pos.shape[0]:
            color = None

    done = np.asarray(pos_heading.get("done", np.zeros((pos.shape[0],), dtype=bool)))
    done = done.reshape(-1).astype(bool, copy=False)
    if done.shape[0] != pos.shape[0]:
        done = np.zeros((pos.shape[0],), dtype=bool)

    terminated = pos_heading.get("terminated", None)
    if terminated is not None:
        try:
            terminated = np.asarray(terminated).reshape(-1).astype(bool, copy=False)
        except Exception:
            terminated = None
    if terminated is not None and terminated.shape[0] != pos.shape[0]:
        terminated = None

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot desired orbit/cylinder if available.
    center = pos_heading.get("cylinder_center", None)
    if center is not None:
        center = np.asarray(center, dtype=np.float32).reshape(-1)
        if center.size >= 3:
            center = center[:3]
        else:
            center = None

    cylinder_radius = pos_heading.get("cylinder_radius", None)
    cylinder_height = pos_heading.get("cylinder_height", None)
    if center is not None and cylinder_radius is not None and cylinder_height is not None:
        try:
            r = float(np.asarray(cylinder_radius).reshape(()).item())
            h = float(np.asarray(cylinder_height).reshape(()).item())
            z0 = float(center[2] - 0.5 * h)
            z1 = float(center[2] + 0.5 * h)
            theta = np.linspace(0.0, 2.0 * np.pi, 40)
            zz = np.linspace(z0, z1, 12)
            th_grid, z_grid = np.meshgrid(theta, zz)
            x_grid = center[0] + r * np.cos(th_grid)
            y_grid = center[1] + r * np.sin(th_grid)
            ax.plot_wireframe(x_grid, y_grid, z_grid, color="#f28e2b", alpha=0.25, linewidth=0.6)
            ax.scatter([center[0]], [center[1]], [center[2]], c="k", s=25, label="cylinder center")
        except Exception:
            pass

    orbit_radius = pos_heading.get("orbit_radius", None)
    orbit_z = pos_heading.get("orbit_z", None)
    if center is not None and orbit_radius is not None and orbit_z is not None:
        try:
            r_orbit = float(np.asarray(orbit_radius).reshape(()).item())
            z_orbit = float(np.asarray(orbit_z).reshape(()).item())
            theta = np.linspace(0.0, 2.0 * np.pi, 200)
            x_ref = center[0] + r_orbit * np.cos(theta)
            y_ref = center[1] + r_orbit * np.sin(theta)
            z_ref = np.full_like(x_ref, z_orbit)
            ax.plot(x_ref, y_ref, z_ref, "k--", linewidth=1.25, alpha=0.8, label="desired orbit")
        except Exception:
            pass

    # Plot a single current vector arrow if available (world frame).
    current_vec = pos_heading.get("current_vec_w", None)
    if current_vec is None:
        flow_hist = pos_heading.get("flow_vel_w", None)
        if flow_hist is not None:
            try:
                flow_hist = np.asarray(flow_hist, dtype=np.float32)
                if flow_hist.ndim == 2 and flow_hist.shape[1] >= 3:
                    current_vec = np.nanmean(flow_hist[:, 0:3], axis=0)
            except Exception:
                current_vec = None
    if current_vec is not None:
        try:
            v = np.asarray(current_vec, dtype=np.float32).reshape(-1)
            if v.size >= 3:
                v = v[:3]
            else:
                v = None
        except Exception:
            v = None
        if v is not None:
            mag = float(np.linalg.norm(v))
            if np.isfinite(mag) and mag > 1e-6:
                origin = None
                if center is not None:
                    origin = np.asarray(center, dtype=np.float32).reshape(3,)
                else:
                    origin = np.nanmean(pos, axis=0).astype(np.float32, copy=False)
                try:
                    if orbit_z is not None:
                        origin[2] = float(np.asarray(orbit_z).reshape(()).item())
                except Exception:
                    pass

                arrow_len = None
                try:
                    if orbit_radius is not None:
                        arrow_len = 0.6 * float(np.asarray(orbit_radius).reshape(()).item())
                except Exception:
                    arrow_len = None
                if arrow_len is None or not np.isfinite(arrow_len) or arrow_len <= 0.0:
                    try:
                        xyz_min = np.nanmin(pos, axis=0)
                        xyz_max = np.nanmax(pos, axis=0)
                        xyz_range = float(np.nanmax(xyz_max - xyz_min))
                        arrow_len = 0.25 * xyz_range
                    except Exception:
                        arrow_len = 0.5
                arrow_len = float(max(0.05, arrow_len))

                color_current = "#17becf"
                ax.quiver(
                    [float(origin[0])],
                    [float(origin[1])],
                    [float(origin[2])],
                    [float(v[0])],
                    [float(v[1])],
                    [float(v[2])],
                    length=arrow_len,
                    normalize=True,
                    color=color_current,
                    linewidth=2.0,
                    alpha=0.85,
                )
                v_dir = v / mag
                tip = origin + v_dir.astype(np.float32) * float(arrow_len)
                ax.text(
                    float(tip[0]),
                    float(tip[1]),
                    float(tip[2]),
                    f"current |v|={mag:.2f} m/s",
                    color=color_current,
                )

    # Plot trajectory segments.
    seg_ranges = _split_ranges_by_done(done)

    line_color = "#4e79a7"
    try:
        mode = str(path_color_mode or "default").strip().lower()
    except Exception:
        mode = "default"

    if mode in ("termination", "terminated", "terminate", "done"):
        labeled = {"terminated": False, "not_terminated": False}
        for start, end in seg_ranges:
            seg = pos[start:end]
            if seg.size <= 0:
                continue
            ended = bool(done[end - 1]) if end > 0 else False
            if terminated is not None:
                seg_terminated = bool(terminated[end - 1]) if ended else False
            else:
                seg_terminated = bool(ended)

            seg_color = "#e15759" if seg_terminated else "#000000"
            label = None
            if seg_terminated and not labeled["terminated"]:
                label = "terminated"
                labeled["terminated"] = True
            if (not seg_terminated) and not labeled["not_terminated"]:
                label = "not terminated"
                labeled["not_terminated"] = True
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=seg_color, linewidth=1.5, alpha=0.95, label=label)
    else:
        for start, end in seg_ranges:
            seg = pos[start:end]
            if seg.size <= 0:
                continue
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=line_color, linewidth=1.5, alpha=0.95)

    # Scalar colormap (optional).
    if color is not None:
        cmap_candidates: list[str] = []
        if cmap:
            cmap_candidates.append(str(cmap))
        else:
            if str(color_key).lower() in ("energy", "e"):
                # Prefer a "hotter" map where high values are red.
                cmap_candidates.extend(["turbo", "jet"])
            else:
                cmap_candidates.append("plasma")
        cmap_candidates.append("plasma")
        cmap_name = "plasma"
        for cand in cmap_candidates:
            try:
                plt.get_cmap(cand)
                cmap_name = cand
                break
            except Exception:
                continue

        if color_label is None:
            if str(color_key).lower() in ("speed", "v", "vel", "velocity"):
                color_label = "speed [m/s]"
            elif str(color_key).lower() in ("v_tan", "vtan", "tan_speed", "tangential_speed"):
                color_label = "v_tan [m/s]"
            elif str(color_key).lower() in ("v_tan_err", "vtan_err", "speed_err", "speed_error", "v_err"):
                color_label = "|v_tan - v_tan_des| [m/s]"
            elif str(color_key).lower() in ("energy", "e"):
                color_label = "E = Σ |u_i|^3"
            else:
                color_label = str(color_key)

        sc = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=color,
            cmap=cmap_name,
            s=6,
            alpha=0.85,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label(str(color_label))

    ax.scatter([pos[0, 0]], [pos[0, 1]], [pos[0, 2]], c="g", s=30, label="start")
    if mode in ("termination", "terminated", "terminate", "done") and seg_ranges:
        start, end = seg_ranges[-1]
        ended = bool(done[end - 1]) if end > 0 else False
        if terminated is not None:
            last_terminated = bool(terminated[end - 1]) if ended else False
        else:
            last_terminated = bool(ended)
        end_color = "#e15759" if last_terminated else "#000000"
    else:
        end_color = "r"
    ax.scatter([pos[-1, 0]], [pos[-1, 1]], [pos[-1, 2]], c=end_color, s=30, label="end")

    # Heading arrows (sampled).
    stride = max(1, int(heading_stride))
    idx = np.arange(0, pos.shape[0], stride)
    if idx.size > 0:
        ax.quiver(
            pos[idx, 0],
            pos[idx, 1],
            pos[idx, 2],
            heading[idx, 0],
            heading[idx, 1],
            heading[idx, 2],
            length=float(arrow_len),
            normalize=True,
            color="k",
            linewidth=0.6,
            alpha=0.8,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    color_title = ""
    if color is not None and color_key:
        color_title = f" + {color_key}"
    ax.set_title(f"Trajectory + heading{color_title}: {traj_path.name}")
    ax.grid(True, alpha=0.3)

    # Equal-ish aspect ratio.
    xyz_min = np.nanmin(pos, axis=0)
    xyz_max = np.nanmax(pos, axis=0)
    xyz_mid = 0.5 * (xyz_min + xyz_max)
    xyz_range = float(np.nanmax(xyz_max - xyz_min))
    if np.isfinite(xyz_range) and xyz_range > 1e-6:
        half = 0.5 * xyz_range
        ax.set_xlim(xyz_mid[0] - half, xyz_mid[0] + half)
        ax.set_ylim(xyz_mid[1] - half, xyz_mid[1] + half)
        ax.set_zlim(xyz_mid[2] - half, xyz_mid[2] + half)

    ax.legend(loc="best")
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=200)

    if show:
        plt.show()

    plt.close(fig)


def plot_power_polar(
    *,
    traj_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    bin_deg: float = 5.0,
    show_raw: bool = False,
    show: bool = False,
    cmap: Optional[str] = None,
):
    """
    Polar plot of instantaneous power proxy P_total = Σ|u_i|^3 vs orbit angle (0..360 deg).

    - angle: atan2(y - cy, x - cx) around cylinder_center (world frame)
    - radius: P_total (either raw samples or binned mean)
    - color: P_total (high values -> red with turbo/jet)
    """
    traj_path = Path(traj_path).expanduser()
    out_path = Path(out_path).expanduser() if out_path else None

    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    data = _load_npz(traj_path)
    pos_raw = data.get("pos", None)
    if pos_raw is None:
        raise KeyError(f"Missing 'pos' in {traj_path}")
    pos = np.asarray(pos_raw, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Invalid pos shape in {traj_path}: {pos.shape}")

    center = data.get("cylinder_center", None)
    if center is None:
        center = np.nanmean(pos, axis=0)
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    if center.size >= 3:
        center = center[:3]
    else:
        center = np.zeros((3,), dtype=np.float32)

    energy = data.get("energy", None)
    if energy is None:
        u = data.get("u", None)
        if u is not None:
            u = np.asarray(u, dtype=np.float32)
            if u.ndim >= 3:
                u = np.squeeze(u, axis=1)
            try:
                energy = np.sum(np.abs(u) ** 3, axis=-1)
            except Exception:
                energy = None
    if energy is None:
        raise KeyError(f"Missing 'energy' (and could not derive from 'u') in {traj_path}")
    energy = np.asarray(energy, dtype=np.float32).reshape(-1)
    if energy.shape[0] != pos.shape[0]:
        raise ValueError(f"energy length mismatch in {traj_path}: energy={energy.shape} pos={pos.shape}")

    dx = pos[:, 0] - float(center[0])
    dy = pos[:, 1] - float(center[1])
    theta = np.arctan2(dy, dx)  # [-pi, pi]
    theta = np.mod(theta, 2.0 * np.pi)  # [0, 2pi)

    mask = np.isfinite(theta) & np.isfinite(energy)
    theta = theta[mask]
    energy = energy[mask]
    if theta.size <= 1:
        raise ValueError(f"Not enough valid samples to plot in {traj_path}")

    cmap_candidates: list[str] = []
    if cmap:
        cmap_candidates.append(str(cmap))
    else:
        cmap_candidates.extend(["turbo", "jet"])
    cmap_candidates.append("plasma")
    cmap_name = "plasma"
    for cand in cmap_candidates:
        try:
            plt.get_cmap(cand)
            cmap_name = cand
            break
        except Exception:
            continue

    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)  # CCW
    ax.grid(True, alpha=0.3)

    if show_raw or bin_deg <= 0.0:
        sc = ax.scatter(theta, energy, c=energy, cmap=cmap_name, s=6, alpha=0.7)
        cb = fig.colorbar(sc, ax=ax, pad=0.1, fraction=0.046)
        cb.set_label("P_total = Σ |u_i|^3 (instantaneous)")
        ax.set_title(f"Instantaneous power vs orbit angle (raw): {traj_path.name}")
    else:
        bin_deg = float(bin_deg)
        n_bins = int(np.ceil(360.0 / max(1e-6, bin_deg)))
        n_bins = max(8, int(n_bins))
        width = (2.0 * np.pi) / float(n_bins)
        idx = np.floor(theta / (2.0 * np.pi) * float(n_bins)).astype(np.int64)
        idx = np.clip(idx, 0, n_bins - 1)

        counts = np.bincount(idx, minlength=n_bins).astype(np.float32)
        sums = np.bincount(idx, weights=energy.astype(np.float64), minlength=n_bins).astype(np.float64)
        means = np.full((n_bins,), np.nan, dtype=np.float32)
        nonzero = counts > 0.0
        means[nonzero] = (sums[nonzero] / counts[nonzero]).astype(np.float32)

        theta_centers = (np.arange(n_bins, dtype=np.float32) + 0.5) * float(width)
        # Plot: grey curve + colored markers (high values -> red).
        ax.plot(theta_centers, means, color="#444444", linewidth=1.0, alpha=0.7)
        finite = np.isfinite(means)
        sc = ax.scatter(theta_centers[finite], means[finite], c=means[finite], cmap=cmap_name, s=30, alpha=0.95)
        cb = fig.colorbar(sc, ax=ax, pad=0.1, fraction=0.046)
        cb.set_label("P_total = Σ |u_i|^3 (binned mean)")
        ax.set_title(f"Instantaneous power vs orbit angle (bin={bin_deg:g}°): {traj_path.name}")

    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _main():
    parser = argparse.ArgumentParser(
        description="Visualize saved BlueROV trajectory (.npz) in 3D with heading arrows and an optional scalar colormap."
    )
    parser.add_argument("traj", type=str, help="Path to .npz saved by scripts/evaluate.py")
    parser.add_argument("--out", type=str, default=None, help="Output .png path (default: <traj>.png)")
    parser.add_argument(
        "--polar",
        action="store_true",
        help="Plot instantaneous power proxy vs orbit angle as a polar plot (uses cylinder_center + energy/u).",
    )
    parser.add_argument("--bin-deg", type=float, default=5.0, help="Polar plot bin size in degrees (default: 5).")
    parser.add_argument("--show-raw", action="store_true", help="Polar plot with raw samples (no binning).")
    parser.add_argument("--heading-stride", type=int, default=50, help="Plot a heading arrow every N steps.")
    parser.add_argument("--arrow-len", type=float, default=0.25, help="Heading arrow length [m].")
    parser.add_argument(
        "--traj-color-mode",
        type=str,
        default="default",
        help="Trajectory line color mode: default (blue) or termination (segments: terminated=red, otherwise=black).",
    )
    parser.add_argument(
        "--color-key",
        type=str,
        default="v_tan_err",
        help="Scalar key in the .npz to color by (e.g., speed, energy). Set to empty to disable.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default=None,
        help="Matplotlib colormap name (default: speed->plasma, energy->turbo).",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive window instead of headless save-only.")
    args = parser.parse_args()

    traj_path = Path(args.traj).expanduser()
    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        out_path = (
            traj_path.with_name(traj_path.stem + "_energy_polar.png") if bool(args.polar) else traj_path.with_suffix(".png")
        )

    if bool(args.polar):
        plot_power_polar(
            traj_path=traj_path,
            out_path=out_path,
            bin_deg=float(args.bin_deg),
            show_raw=bool(args.show_raw),
            show=bool(args.show),
            cmap=(args.cmap or None),
        )
    else:
        plot_trajectory_3d(
            traj_path=traj_path,
            out_path=out_path,
            heading_stride=args.heading_stride,
            arrow_len=args.arrow_len,
            color_key=(args.color_key or None),
            cmap=(args.cmap or None),
            path_color_mode=str(args.traj_color_mode),
            show=bool(args.show),
        )
    print(f"[visualize_trajectory] saved: {out_path.resolve()}")


if __name__ == "__main__":
    _main()
