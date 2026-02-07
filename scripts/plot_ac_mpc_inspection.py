import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _evaluate_critic(critic_net, states):
    """Evaluate critic on a batch of states and return a 1D numpy array."""
    try:
        values = critic_net(states)
    except Exception:
        values = np.array([critic_net(state) for state in states])

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1)


def _trajectory_line(xy, velocities, cmap="jet", linewidth=2.0):
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line = LineCollection(segments, cmap=cmap, linewidth=linewidth)
    line.set_array(velocities[:-1])
    return line


def plot_ac_mpc_inspection(
    actual_traj,
    velocities,
    mpc_preds,
    critic_net=None,
    *,
    cylinder_radius=0.5,
    target_orbit=2.0,
    target_depth=None,
    grid_size=150,
    pred_stride=10,
    values_xy=None,
    values_rz=None,
    grid_extent=None,
):
    """Plot AC-MPC inspection performance for a cylinder orbit task."""
    actual_traj = np.asarray(actual_traj)
    velocities = np.asarray(velocities)

    if target_depth is None:
        target_depth = float(np.mean(actual_traj[:, 2]))

    if grid_extent is None:
        x_min, x_max = actual_traj[:, 0].min(), actual_traj[:, 0].max()
        y_min, y_max = actual_traj[:, 1].min(), actual_traj[:, 1].max()
        pad = max(1.0, 0.3 * target_orbit)
        x_min, x_max = x_min - pad, x_max + pad
        y_min, y_max = y_min - pad, y_max + pad
        r_max = max(target_orbit + pad, np.sqrt(x_max**2 + y_max**2))
        z_min = actual_traj[:, 2].min() - pad
        z_max = actual_traj[:, 2].max() + pad
    else:
        x_min, x_max, y_min, y_max, r_max, z_min, z_max = grid_extent

    if values_xy is None or values_rz is None:
        if critic_net is None:
            raise ValueError("critic_net is required when value grids are not provided.")
        xs = np.linspace(x_min, x_max, grid_size)
        ys = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        yaw = np.arctan2(-yy, -xx)
        flat_states = np.stack(
            [xx.ravel(), yy.ravel(), np.full(xx.size, target_depth), yaw.ravel()], axis=1
        )
        values_xy = _evaluate_critic(critic_net, flat_states).reshape(xx.shape)

        rs = np.linspace(0.0, r_max, grid_size)
        zs = np.linspace(z_min, z_max, grid_size)
        rr, zz = np.meshgrid(rs, zs)
        xx_r = rr
        yy_r = np.zeros_like(rr)
        yaw_r = np.arctan2(-yy_r, -xx_r)
        flat_states_rz = np.stack(
            [xx_r.ravel(), yy_r.ravel(), zz.ravel(), yaw_r.ravel()], axis=1
        )
        values_rz = _evaluate_critic(critic_net, flat_states_rz).reshape(rr.shape)
    else:
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, values_xy.shape[1]), np.linspace(y_min, y_max, values_xy.shape[0]))
        rr, zz = np.meshgrid(np.linspace(0.0, r_max, values_rz.shape[1]), np.linspace(z_min, z_max, values_rz.shape[0]))

    fig, (ax_xy, ax_rz) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    heat_xy = ax_xy.imshow(
        values_xy,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    line_xy = _trajectory_line(actual_traj[:, :2], velocities)
    ax_xy.add_collection(line_xy)

    for idx in range(0, len(mpc_preds), pred_stride):
        preds = np.asarray(mpc_preds[idx])
        ax_xy.plot(preds[:, 0], preds[:, 1], "x", color="black", markersize=4)

    cylinder = plt.Circle((0.0, 0.0), cylinder_radius, color="gray", fill=False, lw=2)
    orbit = plt.Circle((0.0, 0.0), target_orbit, color="white", fill=False, ls="--", lw=1.5)
    ax_xy.add_patch(cylinder)
    ax_xy.add_patch(orbit)
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("상단 공전 뷰 (XY 평면)")

    heat_rz = ax_rz.imshow(
        values_rz,
        extent=(0.0, r_max, z_min, z_max),
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )
    r_actual = np.sqrt(actual_traj[:, 0] ** 2 + actual_traj[:, 1] ** 2)
    traj_rz = np.column_stack([r_actual, actual_traj[:, 2]])
    line_rz = _trajectory_line(traj_rz, velocities)
    ax_rz.add_collection(line_rz)

    for idx in range(0, len(mpc_preds), pred_stride):
        preds = np.asarray(mpc_preds[idx])
        r_preds = np.sqrt(preds[:, 0] ** 2 + preds[:, 1] ** 2)
        ax_rz.plot(r_preds, preds[:, 2], "x", color="black", markersize=4)

    ax_rz.axvline(target_orbit, color="white", ls="--", lw=1.5)
    ax_rz.set_xlabel("중심으로부터의 거리 (m)")
    ax_rz.set_ylabel("수심 (m)")
    ax_rz.set_title("반경-수직 단면 뷰 (RZ 평면)")

    cbar_value = fig.colorbar(heat_rz, ax=[ax_xy, ax_rz], shrink=0.8, label="가치(Value)")
    cbar_speed = fig.colorbar(line_xy, ax=ax_xy, shrink=0.8, label="속도 (m/s)")
    cbar_speed.ax.set_title("속도 (m/s)")

    return fig, (ax_xy, ax_rz)


def load_eval_plot_data(path):
    data = np.load(path, allow_pickle=True)
    actual_traj = data["actual_traj"]
    velocities = data["velocities"]
    mpc_preds = list(data["mpc_preds"]) if "mpc_preds" in data.files else []
    values_xy = data["values_xy"] if "values_xy" in data.files else None
    values_rz = data["values_rz"] if "values_rz" in data.files else None
    grid_extent = (
        float(data["x_min"]),
        float(data["x_max"]),
        float(data["y_min"]),
        float(data["y_max"]),
        float(data["r_max"]),
        float(data["z_min"]),
        float(data["z_max"]),
    )
    meta = {
        "cylinder_radius": float(data["cylinder_radius"]),
        "target_orbit": float(data["target_orbit"]),
        "target_depth": float(data["target_depth"]),
        "grid_size": int(data["grid_size"]),
    }
    return actual_traj, velocities, mpc_preds, values_xy, values_rz, grid_extent, meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot AC-MPC cylinder inspection visualization.")
    parser.add_argument("--eval-data", required=True, help="Path to evaluation plot data (.npz).")
    parser.add_argument(
        "--save",
        default="outputs/ac_mpc_inspection.png",
        help="Path to save the generated figure.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    args = parser.parse_args()

    (
        actual_traj,
        velocities,
        mpc_preds,
        values_xy,
        values_rz,
        grid_extent,
        meta,
    ) = load_eval_plot_data(args.eval_data)

    fig, _ = plot_ac_mpc_inspection(
        actual_traj,
        velocities,
        mpc_preds,
        None,
        cylinder_radius=meta["cylinder_radius"],
        target_orbit=meta["target_orbit"],
        target_depth=meta["target_depth"],
        grid_size=meta["grid_size"],
        values_xy=values_xy,
        values_rz=values_rz,
        grid_extent=grid_extent,
    )
    fig.savefig(args.save, dpi=200)

    if args.show:
        plt.show()
