import argparse
import os
from pathlib import Path

import numpy as np


def _write_pgm_u8(path: str, img_u8: np.ndarray) -> None:
    if img_u8.dtype != np.uint8:
        raise ValueError("img_u8 must be uint8")
    if img_u8.ndim != 2:
        raise ValueError("img_u8 must be HxW")
    h, w = img_u8.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(img_u8.tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--enable-livestream", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="outputs/sonarUse")
    parser.add_argument("--beams", type=int, default=256)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--vertical-samples", type=int, default=128)
    parser.add_argument("--hfov-deg", type=float, default=120.0)
    parser.add_argument("--vfov-deg", type=float, default=30.0)
    parser.add_argument("--range-min", type=float, default=0.5)
    parser.add_argument("--range-max", type=float, default=10.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--noise-mul", type=float, default=0.01)
    parser.add_argument("--noise-add", type=float, default=0.02)
    parser.add_argument("--background-level", type=float, default=0.02)
    parser.add_argument("--range-noise-slope", type=float, default=0.03)
    parser.add_argument("--speckle-sigma", type=float, default=0.35)
    parser.add_argument("--incidence-power", type=float, default=1.5)
    parser.add_argument("--spreading-power", type=float, default=1.0)
    parser.add_argument("--absorption-db-per-m", type=float, default=0.02)
    parser.add_argument("--tvg-power", type=float, default=1.5)
    parser.add_argument("--log-k", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--blur-sigma-range", type=float, default=1.2)
    parser.add_argument("--blur-sigma-beam", type=float, default=0.8)
    parser.add_argument("--normals-annotator", type=str, default="")
    parser.add_argument(
        "--depth-annotator",
        type=str,
        default="distance_to_camera",
        help="Replicator depth annotator name (e.g. distance_to_camera, distance_to_image_plane).",
    )
    parser.add_argument("--depth-is-range", action="store_true", default=True)
    parser.add_argument("--warmup-frames", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))

    from marinegym import init_simulation_app

    sim_cfg = {"headless": args.headless, "enable_livestream": args.enable_livestream}
    simulation_app = init_simulation_app(sim_cfg)

    try:
        from omni.isaac.core import World
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.core.simulation_context import SimulationContext

        from marinegym.sensors.sonar import Sonar, SonarCfg

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()

        prim_utils.create_prim("/World/Box", prim_type="Cube", position=(4.0, 0.0, 0.75), scale=(1.5, 1.5, 1.5))
        prim_utils.create_prim("/World/Wall", prim_type="Cube", position=(8.0, 0.0, 1.0), scale=(0.2, 6.0, 2.0))

        world.reset()

        sonar_cfg = SonarCfg(
            num_beams=args.beams,
            num_bins=args.bins,
            vertical_samples=args.vertical_samples,
            horizontal_fov_deg=args.hfov_deg,
            vertical_fov_deg=args.vfov_deg,
            range_min=args.range_min,
            range_max=args.range_max,
            gain=args.gain,
            noise_multiplicative_std=args.noise_mul,
            noise_additive_std=args.noise_add,
            background_level=args.background_level,
            range_noise_slope=args.range_noise_slope,
            speckle_sigma=args.speckle_sigma,
            incidence_power=args.incidence_power,
            spreading_power=args.spreading_power,
            absorption_db_per_m=args.absorption_db_per_m,
            tvg_power=args.tvg_power,
            log_compression_k=args.log_k,
            gamma=args.gamma,
            blur_sigma_range=args.blur_sigma_range,
            blur_sigma_beam=args.blur_sigma_beam,
            output_dtype="float32",
            depth_annotator=args.depth_annotator,
            normals_annotator=(args.normals_annotator or None),
            depth_is_range=bool(args.depth_is_range),
        )
        sonar = Sonar(sonar_cfg)

        sonar.spawn(
            ["/World/Sonar_0"],
            translations=[(0.0, 0.0, 1.0)],
            targets=[(10.0, 0.0, 1.0)],
        )
        sonar.initialize(prim_paths_expr="/World/Sonar_.*")

        for _ in range(max(0, int(args.warmup_frames))):
            world.step(render=True)

        SimulationContext.instance().render()
        td = sonar.get_images()
        sonar_img = td["sonar"][0, 0].detach().cpu().numpy()  # (bins, beams)

        npy_path = output_dir / "sonar.npy"
        np.save(npy_path, sonar_img)

        pgm_path = output_dir / "sonar.pgm"
        img_u8 = np.clip(np.round(sonar_img * 255.0), 0, 255).astype(np.uint8)
        _write_pgm_u8(str(pgm_path), img_u8)

        print("[sonarUse] sonar:", sonar_img.shape, sonar_img.dtype, "min/max:", float(sonar_img.min()), float(sonar_img.max()))
        print("[sonarUse] saved:", str(npy_path))
        print("[sonarUse] saved:", str(pgm_path))

        for _ in range(30):
            world.step(render=not args.headless)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
