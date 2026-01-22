# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
from omni.isaac.core.simulation_context import SimulationContext
from pxr import Gf, UsdGeom
from tensordict import TensorDict

from .camera import orientation_from_view


@dataclass
class SonarCfg:
    sensor_tick: float = 0.0
    num_beams: int = 256
    num_bins: int = 256
    vertical_samples: int = 128
    horizontal_fov_deg: float = 120.0
    vertical_fov_deg: float = 30.0
    range_min: float = 0.5
    range_max: float = 10.0
    gain: float = 1.0
    noise_multiplicative_std: float = 0.01
    noise_additive_std: float = 0.02
    background_level: float = 0.02
    range_noise_slope: float = 0.03
    speckle_sigma: float = 0.35
    incidence_power: float = 1.5
    spreading_power: float = 1.0
    absorption_db_per_m: float = 0.02
    tvg_power: float = 1.5
    log_compression_k: float = 10.0
    gamma: float = 0.7
    blur_sigma_range: float = 1.2
    blur_sigma_beam: float = 0.8
    output_dtype: str = "float32"  # "float32" | "uint8" | "uint16"
    depth_annotator: str = "distance_to_camera"
    normals_annotator: Optional[str] = None
    depth_is_range: bool = True
    horizontal_aperture_mm: float = 20.955


class Sonar:
    """
    Sonar image synthesized from Replicator depth (optionally normals).

    Notes:
    - This spawns USD Camera prim(s) and reads depth through Replicator annotators.
    - The returned sonar image is a tensor with shape (1, num_bins, num_beams).
    """

    def __init__(self, cfg: Optional[SonarCfg] = None) -> None:
        self.cfg = cfg or SonarCfg()
        self.sim = SimulationContext.instance()
        self.torch_device = self.sim.device
        self.device = self.sim.device
        if isinstance(self.device, str) and "cuda" in self.device:
            self.device = self.device.split(":")[0]

        self._annotators: list[dict[str, rep.Annotator]] = []
        self._ray_dirs_cam: Optional[torch.Tensor] = None
        self._ray_cos_cam: Optional[torch.Tensor] = None

    def spawn(
        self,
        prim_paths: Sequence[str],
        translations=None,
        targets=None,
    ) -> None:
        n = len(prim_paths)
        if translations is None:
            translations = [(0, 0, 0) for _ in range(n)]
        translations = torch.atleast_2d(torch.as_tensor(translations)).expand(n, 3).tolist()

        if targets is None:
            targets = [(1, 0, 0) for _ in range(n)]
        targets = torch.atleast_2d(torch.as_tensor(targets)).expand(n, 3).tolist()

        if not len(translations) == len(prim_paths) == len(targets):
            raise ValueError("translations/targets must match prim_paths length.")

        for prim_path, translation, target in zip(prim_paths, translations, targets):
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")
            prim_utils.create_prim(
                prim_path,
                prim_type="Camera",
                translation=translation,
                orientation=orientation_from_view(translation, target),
            )
            self._configure_usd_camera(prim_path)

    def initialize(self, prim_paths_expr: Optional[str] = None) -> None:
        if prim_paths_expr is None:
            prim_paths_expr = f"/World/envs/.*/Sonar_.*"

        prim_paths = prim_utils.find_matching_prim_paths(prim_paths_expr)
        if not prim_paths:
            raise RuntimeError(f"No sonar camera prims matched pattern: {prim_paths_expr}")

        self._annotators.clear()
        for prim_path in prim_paths:
            render_product = rep.create.render_product(
                prim_path, resolution=(self.cfg.num_beams, self.cfg.vertical_samples)
            )
            annotators: dict[str, rep.Annotator] = {}
            for name in (self.cfg.depth_annotator, self.cfg.normals_annotator):
                if not name:
                    continue
                annotator = rep.AnnotatorRegistry.get_annotator(name=name, device=self.device)
                annotator.attach([render_product])
                annotators[name] = annotator
            self._annotators.append(annotators)

        self._build_ray_tables()

        for _ in range(2):
            self.sim.render()

    def get_images(self) -> TensorDict:
        if not self._annotators:
            raise RuntimeError(
                "Sonar has no annotators. Call `initialize()` with a prim_paths_expr that matches the spawned sonar prim(s)."
            )

        images_list = []
        for annotators in self._annotators:
            depth = self._get_annotator_tensor(annotators, self.cfg.depth_annotator)
            normals = None
            if self.cfg.normals_annotator:
                normals = self._get_annotator_tensor(annotators, self.cfg.normals_annotator)

            sonar = self._depth_to_sonar(depth, normals=normals)
            images_list.append(TensorDict({"sonar": sonar}, []))

        return torch.stack(images_list)

    def _get_annotator_tensor(self, annotators: dict[str, rep.Annotator], name: str) -> torch.Tensor:
        if name not in annotators:
            raise RuntimeError(f"Annotator '{name}' not attached. Available: {list(annotators.keys())}")
        data = annotators[name].get_data(device=self.device)
        if isinstance(data, torch.Tensor):
            tensor = data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            try:
                tensor = wp.to_torch(data)
            except Exception:
                tensor = torch.as_tensor(data)

        if tensor.device.type == "cpu" and isinstance(self.torch_device, str) and "cuda" in self.torch_device:
            tensor = tensor.to(self.torch_device)
        return tensor

    def _configure_usd_camera(self, prim_path: str) -> None:
        prim = prim_utils.get_prim_at_path(prim_path)
        usd_cam = UsdGeom.Camera(prim)

        hfov = float(self.cfg.horizontal_fov_deg)
        if not (0.0 < hfov < 179.0):
            raise ValueError(f"horizontal_fov_deg must be in (0, 179); got {hfov}")
        vfov = float(self.cfg.vertical_fov_deg)
        if not (0.0 < vfov < 179.0):
            raise ValueError(f"vertical_fov_deg must be in (0, 179); got {vfov}")

        horiz_ap = float(self.cfg.horizontal_aperture_mm)
        focal = horiz_ap / (2.0 * math.tan(math.radians(hfov) / 2.0))
        vert_ap = 2.0 * focal * math.tan(math.radians(vfov) / 2.0)

        usd_cam.GetHorizontalApertureAttr().Set(horiz_ap)
        usd_cam.GetVerticalApertureAttr().Set(vert_ap)
        usd_cam.GetFocalLengthAttr().Set(focal)
        usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(self.cfg.range_min, self.cfg.range_max))

    def _build_ray_tables(self) -> None:
        device = self.torch_device
        w = int(self.cfg.num_beams)
        h = int(self.cfg.vertical_samples)
        hfov = math.radians(float(self.cfg.horizontal_fov_deg))
        vfov = math.radians(float(self.cfg.vertical_fov_deg))

        u = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2.0 - 1.0
        v = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2.0 - 1.0
        uu, vv = torch.meshgrid(u, v, indexing="xy")
        tx = math.tan(hfov / 2.0)
        ty = math.tan(vfov / 2.0)
        dirs = torch.stack([uu * tx, vv * ty, -torch.ones_like(uu)], dim=-1)
        dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp_min(1e-12)
        self._ray_dirs_cam = dirs
        self._ray_cos_cam = (-dirs[..., 2]).clamp_min(1e-6)

    @staticmethod
    def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if sigma <= 0.0:
            return torch.tensor([1.0], device=device, dtype=dtype)
        radius = int(max(1, math.ceil(3.0 * sigma)))
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        return kernel

    def _gaussian_blur2d(self, img: torch.Tensor, sigma_y: float, sigma_x: float) -> torch.Tensor:
        if sigma_x <= 0.0 and sigma_y <= 0.0:
            return img
        if img.dim() != 2:
            raise ValueError(f"expected HxW image, got: {tuple(img.shape)}")

        x_kernel = self._gaussian_kernel1d(float(sigma_x), img.device, img.dtype)
        y_kernel = self._gaussian_kernel1d(float(sigma_y), img.device, img.dtype)

        x_kernel_2d = x_kernel.view(1, 1, 1, -1)
        y_kernel_2d = y_kernel.view(1, 1, -1, 1)

        x_pad = (x_kernel.numel() - 1) // 2
        y_pad = (y_kernel.numel() - 1) // 2

        x = img[None, None, ...]
        x = F.pad(x, (x_pad, x_pad, 0, 0), mode="replicate")
        x = F.conv2d(x, x_kernel_2d)
        x = F.pad(x, (0, 0, y_pad, y_pad), mode="replicate")
        x = F.conv2d(x, y_kernel_2d)
        return x[0, 0]

    @staticmethod
    def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
        t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _depth_to_sonar(self, depth: torch.Tensor, normals: Optional[torch.Tensor]) -> torch.Tensor:
        if self._ray_dirs_cam is None or self._ray_cos_cam is None:
            self._build_ray_tables()

        depth = depth.to(dtype=torch.float32)
        if depth.dim() == 4 and depth.shape[0] == 1:
            depth = depth.squeeze(0)
        if depth.dim() == 3 and depth.shape[0] == 1:
            depth = depth.squeeze(0)
        if depth.dim() == 3 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        if depth.dim() != 2:
            raise ValueError(f"Unexpected depth tensor shape: {tuple(depth.shape)}")
        if tuple(depth.shape) != (self.cfg.vertical_samples, self.cfg.num_beams):
            depth = depth.reshape(self.cfg.vertical_samples, self.cfg.num_beams)

        if self.cfg.depth_is_range:
            ranges = depth
        else:
            ranges = depth / self._ray_cos_cam

        h = int(self.cfg.vertical_samples)
        w = int(self.cfg.num_beams)
        n_bins = int(self.cfg.num_bins)
        r0 = float(self.cfg.range_min)
        r1 = float(self.cfg.range_max)
        step = (r1 - r0) / float(n_bins)
        if step <= 0:
            raise ValueError("range_max must be > range_min.")

        bins = torch.floor((ranges - r0) / step).to(torch.long)
        valid = (bins >= 0) & (bins < n_bins) & torch.isfinite(ranges)

        intensity = torch.ones((h, w), device=ranges.device, dtype=torch.float32)
        if normals is not None:
            normals = normals.to(dtype=torch.float32)
            if normals.dim() == 4 and normals.shape[0] == 1:
                normals = normals.squeeze(0)
            if normals.dim() == 3 and normals.shape[0] == 3:
                normals = normals.permute(1, 2, 0)
            if normals.dim() == 3 and normals.shape[-1] >= 3 and normals.shape[:2] == (h, w):
                n = normals[..., :3]
                view_dir = -self._ray_dirs_cam
                intensity = (n * view_dir).sum(dim=-1).clamp_min(0.0).pow(float(self.cfg.incidence_power))

        t = (torch.arange(h, device=ranges.device, dtype=torch.float32) + 0.5) / h
        v_weight = self._smoothstep(0.0, 0.2, t) * (1.0 - self._smoothstep(0.8, 1.0, t))
        weights = v_weight[:, None].expand(h, w)

        r = ranges.clamp_min(r0)
        atten = (r / r0).pow(-float(self.cfg.spreading_power))
        if float(self.cfg.absorption_db_per_m) > 0.0:
            atten = atten * torch.pow(
                torch.tensor(10.0, device=ranges.device, dtype=torch.float32),
                (-float(self.cfg.absorption_db_per_m) * r / 20.0),
            )

        sample_val = (intensity * weights * atten).to(torch.float32)

        flat_beam = torch.arange(w, device=ranges.device).repeat(h)
        flat_bins = bins.reshape(-1)
        flat_vals = sample_val.reshape(-1)
        flat_valid = valid.reshape(-1)
        flat_idx = flat_beam[flat_valid] * n_bins + flat_bins[flat_valid]

        sum_flat = torch.zeros(w * n_bins, device=ranges.device, dtype=torch.float32)
        cnt_flat = torch.zeros(w * n_bins, device=ranges.device, dtype=torch.float32)
        sum_flat.scatter_add_(0, flat_idx, flat_vals[flat_valid])
        cnt_flat.scatter_add_(0, flat_idx, torch.ones_like(flat_vals[flat_valid]))

        sums = sum_flat.view(w, n_bins)
        cnts = cnt_flat.view(w, n_bins)

        bin_centers = (torch.arange(n_bins, device=ranges.device, dtype=torch.float32) + 0.5) * step + r0
        tvg = (bin_centers / r0).pow(float(self.cfg.tvg_power))[None, :]
        mul_beam = torch.randn((w, 1), device=ranges.device, dtype=torch.float32) * float(
            self.cfg.noise_multiplicative_std
        ) + 1.0
        add = (
            torch.randn((w, n_bins), device=ranges.device, dtype=torch.float32) * float(self.cfg.noise_additive_std)
            + float(self.cfg.background_level)
            + float(self.cfg.range_noise_slope) * (bin_centers / r1)[None, :]
        )

        sonar = float(self.cfg.gain) * add
        has = cnts > 0
        echo = (sums / cnts.clamp_min(1.0)) * tvg * mul_beam
        sonar = sonar + torch.where(has, float(self.cfg.gain) * echo, 0.0)

        if float(self.cfg.speckle_sigma) > 0.0:
            s = float(self.cfg.speckle_sigma)
            speckle = torch.exp(torch.randn_like(sonar) * s - 0.5 * (s * s))
            sonar = sonar * speckle

        if float(self.cfg.blur_sigma_range) > 0.0 or float(self.cfg.blur_sigma_beam) > 0.0:
            sonar = self._gaussian_blur2d(sonar.transpose(0, 1), float(self.cfg.blur_sigma_range), float(self.cfg.blur_sigma_beam)).transpose(0, 1)

        k = float(self.cfg.log_compression_k)
        if k > 0.0:
            sonar = torch.log1p(k * sonar) / math.log1p(k)
        g = float(self.cfg.gamma)
        if g > 0.0 and g != 1.0:
            sonar = sonar.clamp_min(0.0).pow(g)

        sonar = sonar.clamp(0.0, 1.0)

        sonar = sonar.transpose(0, 1).contiguous().unsqueeze(0)

        if self.cfg.output_dtype == "float32":
            return sonar
        if self.cfg.output_dtype == "uint8":
            return (sonar * 255.0).round().to(torch.uint8)
        if self.cfg.output_dtype == "uint16":
            return (sonar * 65535.0).round().to(torch.uint16)
        raise ValueError(f"Unsupported output_dtype: {self.cfg.output_dtype}")
