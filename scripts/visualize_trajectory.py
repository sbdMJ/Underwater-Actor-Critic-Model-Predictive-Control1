from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np


def _load_npz(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)
    return {k: data[k] for k in data.files}


def _split_by_done(pos: np.ndarray, done: np.ndarray) -> list[np.ndarray]:
    pos = np.asarray(pos)
    done = np.asarray(done).reshape(-1).astype(bool, copy=False)
    if pos.shape[0] != done.shape[0]:
        done = np.zeros((pos.shape[0],), dtype=bool)

    segments: list[np.ndarray] = []
    start = 0
    for i in range(pos.shape[0]):
        if done[i]:
            segments.append(pos[start : i + 1])
            start = i + 1
    if start < pos.shape[0]:
        segments.append(pos[start:])
    return [seg for seg in segments if seg.size > 0]


def plot_trajectory_3d(
    *,
    traj_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    heading_stride: int = 50,
    arrow_len: float = 0.25,
    show: bool = False,
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

    speed = pos_heading.get("speed", None)
    if speed is not None:
        try:
            speed = np.asarray(speed, dtype=np.float32).reshape(-1)
        except Exception:
            speed = None
    if speed is not None and speed.shape[0] != pos.shape[0]:
        speed = None

    done = np.asarray(pos_heading.get("done", np.zeros((pos.shape[0],), dtype=bool)))
    done = done.reshape(-1).astype(bool, copy=False)
    if done.shape[0] != pos.shape[0]:
        done = np.zeros((pos.shape[0],), dtype=bool)

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

    # Plot trajectory segments.
    segments = _split_by_done(pos, done)
    for seg in segments:
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#4e79a7", linewidth=1.5, alpha=0.95)

    # Speed colormap (optional).
    if speed is not None:
        sc = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=speed,
            cmap="plasma",
            s=6,
            alpha=0.85,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("speed [m/s]")

    ax.scatter([pos[0, 0]], [pos[0, 1]], [pos[0, 2]], c="g", s=30, label="start")
    ax.scatter([pos[-1, 0]], [pos[-1, 1]], [pos[-1, 2]], c="r", s=30, label="end")

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
    ax.set_title(f"Trajectory + heading{' + speed' if speed is not None else ''}: {traj_path.name}")
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


def _main():
    parser = argparse.ArgumentParser(
        description="Visualize saved BlueROV trajectory (.npz) in 3D with heading arrows and optional speed colormap."
    )
    parser.add_argument("traj", type=str, help="Path to .npz saved by scripts/evaluate.py")
    parser.add_argument("--out", type=str, default=None, help="Output .png path (default: <traj>.png)")
    parser.add_argument("--heading-stride", type=int, default=50, help="Plot a heading arrow every N steps.")
    parser.add_argument("--arrow-len", type=float, default=0.25, help="Heading arrow length [m].")
    parser.add_argument("--show", action="store_true", help="Show interactive window instead of headless save-only.")
    args = parser.parse_args()

    traj_path = Path(args.traj).expanduser()
    out_path = Path(args.out).expanduser() if args.out else traj_path.with_suffix(".png")
    plot_trajectory_3d(
        traj_path=traj_path,
        out_path=out_path,
        heading_stride=args.heading_stride,
        arrow_len=args.arrow_len,
        show=bool(args.show),
    )
    print(f"[visualize_trajectory] saved: {out_path.resolve()}")


if __name__ == "__main__":
    _main()
