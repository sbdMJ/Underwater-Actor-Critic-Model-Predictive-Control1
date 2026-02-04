import logging
from typing import Type, Dict

import torch
import torch.distributions as D
import yaml
from pxr import PhysxSchema, UsdPhysics
from functorch import vmap
#from tensordict.nn import make_functional
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDict
from tensordict import from_module

from marinegym.views import RigidPrimView
import omni.isaac.core.utils.prims as prim_utils
from marinegym.actuators.rotor_group import RotorGroup
from marinegym.actuators.t200 import T200
from marinegym.controllers import LeePositionController
from omni.isaac.core.prims.xform_prim_view import XFormPrimView

from marinegym.robots import RobotBase, RobotCfg
from marinegym.robots.robot import TEMPLATE_PRIM_PATH
from marinegym.utils.torch import (
    normalize, off_diag, quat_rotate, quat_rotate_inverse, quat_axis, symlog, quaternion_to_euler
)

from dataclasses import dataclass
from collections import defaultdict

import pprint
import torch.distributions as D

@dataclass
class UnderwaterVehicleCfg(RobotCfg):
    force_sensor: bool = False
    
class UnderwaterVehicle(RobotBase):
    
    param_path: str
    DEFAULT_CONTROLLER: Type = LeePositionController
    cfg_cls = UnderwaterVehicleCfg

    def __init__(
        self, 
        name: str = None, 
        cfg: UnderwaterVehicleCfg=None, 
        is_articulation: bool = True
    ) -> None:
        super().__init__(name, cfg, is_articulation)

        with open(self.param_path, "r") as f:
            logging.info(f"Reading {self.name}'s params from {self.param_path}.")
            self.params = yaml.safe_load(f)
        # Model-authored orientation correction (multiples of pi). Used by tasks when
        # task.model_rpy_offset is null.
        self.model_rpy_offset_pi = self.params.get("model_rpy_offset", [0.0, 0.0, 0.0])
        # Buoyancy lever arm direction (in the base_link local frame). Some assets have a visual
        # frame rotated relative to the rigid-body frame; in that case, using +Z here can cause the
        # vehicle to "self-right" into a 90deg-pitched pose. This vector defines the axis along which
        # `coBM` is applied (defaults to +Z).
        cobm_axis = self.params.get("cobm_axis", [0.0, 0.0, 1.0])
        if isinstance(cobm_axis, str):
            axis = cobm_axis.strip().lower()
            if axis == "x":
                cobm_axis = [1.0, 0.0, 0.0]
            elif axis == "y":
                cobm_axis = [0.0, 1.0, 0.0]
            elif axis == "z":
                cobm_axis = [0.0, 0.0, 1.0]
            else:
                raise ValueError(f"cobm_axis must be one of 'x','y','z' or a 3-vector. got {cobm_axis!r}")
        cobm_axis_t = torch.as_tensor(cobm_axis, device=self.device, dtype=torch.float32).flatten()
        if cobm_axis_t.numel() != 3:
            raise ValueError(f"cobm_axis must have 3 elements. got {cobm_axis!r}")
        nrm = torch.norm(cobm_axis_t).clamp_min(1e-9)
        self._cobm_axis_unit = (cobm_axis_t / nrm).to(self.device)

        # Thruster local axis convention for direct rotor-force mode (use_virtual_rotors == False).
        # Default is +X (0). Accepts int 0/1/2 or strings like "x", "-y", "+z".
        thrust_axis = self.params.get("thrust_axis", 0)
        thrust_axis_sign = 1.0
        if isinstance(thrust_axis, str):
            axis = thrust_axis.strip().lower()
            if axis.startswith(("+", "-")):
                if axis[0] == "-":
                    thrust_axis_sign = -1.0
                axis = axis[1:]
            if axis == "x":
                thrust_axis = 0
            elif axis == "y":
                thrust_axis = 1
            elif axis == "z":
                thrust_axis = 2
            else:
                raise ValueError(f"thrust_axis must be 0/1/2 or 'x'/'y'/'z' (optionally signed). got {thrust_axis!r}")
        thrust_axis = int(thrust_axis)
        if thrust_axis not in (0, 1, 2):
            raise ValueError(f"thrust_axis must be 0,1,2. got {thrust_axis}")
        self._thrust_axis_idx = thrust_axis
        self._thrust_axis_sign = float(self.params.get("thrust_axis_sign", thrust_axis_sign))
        # Optional per-rotor thrust axis overrides (length num_rotors).
        # Each entry may be 0/1/2 or 'x'/'y'/'z' with optional sign prefix.
        self._thrust_axes_idx = None
        self._thrust_axes_sign = None
        thrust_axes = self.params.get("thrust_axes", None)
        if thrust_axes is not None:
            axes_idx = torch.zeros((int(self.num_rotors),), device=self.device, dtype=torch.long)
            axes_sign = torch.ones((int(self.num_rotors),), device=self.device, dtype=torch.float32)
            if len(thrust_axes) != int(self.num_rotors):
                raise ValueError(
                    f"thrust_axes must have length num_rotors={int(self.num_rotors)}. got {len(thrust_axes)}"
                )
            for i, entry in enumerate(list(thrust_axes)):
                sign_i = 1.0
                axis_i = entry
                if isinstance(axis_i, str):
                    s = axis_i.strip().lower()
                    if s.startswith(("+", "-")):
                        if s[0] == "-":
                            sign_i = -1.0
                        s = s[1:]
                    if s == "x":
                        axis_i = 0
                    elif s == "y":
                        axis_i = 1
                    elif s == "z":
                        axis_i = 2
                    else:
                        raise ValueError(
                            f"thrust_axes[{i}] must be 0/1/2 or 'x'/'y'/'z' (optionally signed). got {entry!r}"
                        )
                axis_i = int(axis_i)
                if axis_i not in (0, 1, 2):
                    raise ValueError(f"thrust_axes[{i}] must be 0,1,2. got {axis_i}")
                axes_idx[i] = axis_i
                axes_sign[i] = float(sign_i)
            self._thrust_axes_idx = axes_idx
            self._thrust_axes_sign = axes_sign
        self.num_rotors = self.params["rotor_configuration"]["num_rotors"]
        self.thruster_allocation = None
        if "thruster_allocation" in self.params:
            self.thruster_allocation = torch.tensor(
                self.params["thruster_allocation"],
                dtype=torch.float32,
                device=self.device,
            )

        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors, device=self.device)
        self.intrinsics_spec = CompositeSpec({
            "mass": UnboundedContinuousTensorSpec(1),
            "inertia": UnboundedContinuousTensorSpec(3),
            "com": UnboundedContinuousTensorSpec(3),
            "KF": UnboundedContinuousTensorSpec(self.num_rotors),
            "KM": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_up": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_down": UnboundedContinuousTensorSpec(self.num_rotors),
            "drag_coef": UnboundedContinuousTensorSpec(1),
        }).to(self.device)
        
        if self.cfg.force_sensor:
            self.use_force_sensor = True
            state_dim = 19 + self.num_rotors + 6
        else:
            self.use_force_sensor = False
            state_dim = 19 + self.num_rotors
        self.state_spec = UnboundedContinuousTensorSpec(state_dim, device=self.device)
        self.randomization = defaultdict(dict)

    def initialize(
        self, 
        prim_paths_expr: str = None,
        track_contact_forces: bool = False
    ):
        if self.is_articulation:
            super().initialize(prim_paths_expr=prim_paths_expr)
            base_rel = self.params.get("base_link_relpath", "base_link")
            base_expr = (
                f"{self.prim_paths_expr}/{base_rel.lstrip('/')}"
                if base_rel
                else f"{self.prim_paths_expr}/base_link"
            )
            try:
                self.base_link = RigidPrimView(
                    prim_paths_expr=base_expr,
                    name="base_link",
                    track_contact_forces=track_contact_forces,
                    shape=self.shape,
                )
                self.base_link.initialize()
            except Exception:
                template_root = f"{TEMPLATE_PRIM_PATH}/{self.name}_0"
                root_prim = prim_utils.get_prim_at_path(template_root)
                if not root_prim.IsValid():
                    raise RuntimeError(
                        f"Cannot resolve base_link for articulation '{self.name}'. "
                        f"Expected robot root prim at '{template_root}' to exist."
                    )

                queue = [root_prim]
                candidates: list[str] = []
                sample_paths: list[str] = []
                while queue:
                    prim = queue.pop(0)
                    for child in prim_utils.get_prim_children(prim):
                        queue.append(child)
                        path = prim_utils.get_prim_path(child)
                        low = path.lower()
                        if "rotor" in low or "thruster" in low:
                            continue
                        if len(sample_paths) < 80:
                            sample_paths.append(path)
                        if not (UsdPhysics.RigidBodyAPI(child) or PhysxSchema.PhysxRigidBodyAPI(child)):
                            continue
                        candidates.append(path)

                chosen = None
                if candidates:
                    base_like = next(
                        (p for p in candidates if p.lower().endswith("/base_link")),
                        None,
                    )
                    body_like = next(
                        (p for p in candidates if p.lower().endswith("/body")),
                        None,
                    )
                    chosen = base_like or body_like or min(
                        set(candidates), key=lambda p: (p.count("/"), len(p))
                    )

                if chosen is None or not chosen.startswith(template_root + "/"):
                    raise RuntimeError(
                        "Prim 'base_link' not found for articulation asset and no descendant rigid bodies "
                        f"could be inferred under '{template_root}'. "
                        "Either author a rigid body prim named 'base_link' (recommended) or set "
                        "`base_link_relpath` in the robot YAML to the correct link prim. "
                        f"Sample prims under root: {sample_paths[:30]}"
                    )

                rel = chosen[len(template_root) + 1 :]
                num_envs = int(self.shape[0])
                num_robots = int(self.shape[1])
                prim_paths = [
                    f"/World/envs/env_{env_i}/{self.name}_{robot_i}/{rel}"
                    for env_i in range(num_envs)
                    for robot_i in range(num_robots)
                ]
                print(
                    f"[UnderwaterVehicle] articulation base_link prim not found at '{base_expr}'. "
                    f"Using inferred rigid body '{rel}'."
                )
                self.base_link = RigidPrimView(
                    prim_paths_expr=prim_paths,
                    name="base_link",
                    track_contact_forces=track_contact_forces,
                    shape=self.shape,
                )
                self.base_link.initialize()
            print(self._view.dof_names)
            print(self._view._dof_indices)
            rotor_joint_indices = [
                i for i, dof_name in enumerate(self._view._dof_names) 
                if dof_name.startswith("rotor")
            ]
            if len(rotor_joint_indices):
                self.rotor_joint_indices = torch.tensor(
                    rotor_joint_indices,
                    device=self.device
                )
            else:
                self.rotor_joint_indices = None
        else:
            if prim_paths_expr is None:
                prim_paths_expr = f"/World/envs/.*/{self.name}_.*"
            super().initialize(prim_paths_expr=prim_paths_expr)
            self.base_link = self._view
            self.prim_paths_expr = prim_paths_expr

        rotor_pattern = self.params.get("rotor_prim_pattern", "rotor_*")
        rotor_expr = f"{self.prim_paths_expr}/{rotor_pattern}"
        try:
            self.rotors_view = RigidPrimView(
                prim_paths_expr=rotor_expr,
                name="rotors",
                shape=(*self.shape, self.num_rotors),
            )
            self.rotors_view.initialize()
        except Exception:
            def _ensure_rigid_body(path: str) -> bool:
                prim = prim_utils.get_prim_at_path(path)
                if not prim.IsValid():
                    return False
                # Do not force rotor prims to become rigid bodies here.
                # Some assets are visual-only or use instancing/references where editing APIs can
                # cause unexpected behavior (e.g., rotors becoming independent dynamic actors).
                return True

            # Non-articulation USD들은 최상위 prim이 rigid body가 아닐 수 있어서,
            # self.prim_paths(=base rigid body view) 기준으로 rotor를 탐색하면 누락될 수 있다.
            # 이 경우 container root(/World/envs/env_i/{name}_{j})를 탐색 루트로 사용한다.
            search_root_paths = list(self.prim_paths)
            if not bool(getattr(self, "is_articulation", True)):
                try:
                    num_envs = int(self.shape[0])
                    num_agents = int(self.shape[1])
                    search_root_paths = [
                        f"/World/envs/env_{env_i}/{self.name}_{agent_i}"
                        for env_i in range(num_envs)
                        for agent_i in range(num_agents)
                    ]
                except Exception:
                    search_root_paths = list(self.prim_paths)

            rotor_paths = []
            all_child_paths = []
            rotor_explicit = self.params.get("rotor_prim_paths", None)
            if rotor_explicit:
                for prim_path in search_root_paths:
                    for rel in rotor_explicit:
                        rel = str(rel)
                        full_path = rel if rel.startswith("/") else (prim_path.rstrip("/") + "/" + rel.lstrip("/"))
                        if _ensure_rigid_body(full_path):
                            rotor_paths.append(full_path)
                expected_explicit = int(len(search_root_paths)) * int(self.num_rotors)
                # If explicit paths did not yield the expected count, fall back to hint-based search.
                if len(rotor_paths) != expected_explicit:
                    rotor_explicit = None
                    rotor_paths = []

            if not rotor_explicit:
                rotor_hints = self.params.get(
                    "rotor_prim_hints", ["rotor", "thruster", "prop", "motor", "fan"]
                )
                for prim_path in search_root_paths:
                    root_prim = prim_utils.get_prim_at_path(prim_path)
                    queue = [root_prim]
                    found: dict[int, str] = {}
                    while queue:
                        prim = queue.pop(0)
                        for child in prim_utils.get_prim_children(prim):
                            queue.append(child)
                            path = prim_utils.get_prim_path(child)
                            if path == prim_path:
                                continue
                            all_child_paths.append(path)
                            if rotor_hints and not any(
                                h in path.lower() for h in rotor_hints
                            ):
                                continue
                            # Prefer canonical rotor prim names to avoid collecting rotor sub-meshes.
                            leaf = path.rsplit("/", 1)[-1].lower()
                            if not (
                                leaf.startswith("rotor_")
                                and leaf[6:].isdigit()
                            ):
                                continue
                            rotor_idx = int(leaf[6:])
                            if rotor_idx < 0 or rotor_idx >= int(self.num_rotors):
                                continue
                            if rotor_idx in found:
                                continue
                            if _ensure_rigid_body(path):
                                found[rotor_idx] = path
                                if len(found) >= int(self.num_rotors):
                                    queue.clear()
                                    break
                    for rotor_idx in range(int(self.num_rotors)):
                        if rotor_idx in found:
                            rotor_paths.append(found[rotor_idx])

            expected = int(self.shape[0]) * int(self.num_rotors)
            if len(rotor_paths) != expected:
                if all_child_paths:
                    unique_paths = sorted(set(all_child_paths))
                    non_looks = [
                        p for p in unique_paths
                        if "/looks" not in p.lower()
                        and "/materials" not in p.lower()
                        and "/shaders" not in p.lower()
                    ]
                    sample_non_looks = ", ".join(non_looks[:80])
                    sample_all = ", ".join(unique_paths[:50])
                    print(
                        "[UnderwaterVehicle] rotor lookup failed. "
                        f"children={len(unique_paths)} non_looks={len(non_looks)}"
                    )
                    if sample_non_looks:
                        print(
                            "[UnderwaterVehicle] sample non-looks prims: "
                            f"{sample_non_looks}"
                        )
                    else:
                        print(
                            "[UnderwaterVehicle] sample prims: "
                            f"{sample_all}"
                        )
                if self.thruster_allocation is None:
                    raise ValueError(
                        "Rotor prim count mismatch: "
                        f"found={len(rotor_paths)} expected={expected}. "
                        "Set rotor_prim_pattern/rotor_prim_hints/rotor_prim_paths to match rotor prims."
                    )
                if self.thruster_allocation.shape != (6, self.num_rotors):
                    raise ValueError(
                        "thruster_allocation must have shape (6, num_rotors). "
                        f"got {tuple(self.thruster_allocation.shape)}"
                    )
                print(
                    "[UnderwaterVehicle] using thruster_allocation fallback "
                    "for virtual rotors."
                )
                self.rotors_view = None
                self.use_virtual_rotors = True

            if rotor_paths:
                # Prefer rigid-body rotor view (can apply forces directly). If rotors are not rigid bodies
                # (visual-only USD), fall back to XForm view and use virtual-rotor wrench application.
                try:
                    self.rotors_view = RigidPrimView(
                        prim_paths_expr=rotor_paths,
                        name="rotors",
                        shape=(*self.shape, self.num_rotors),
                    )
                    self.rotors_view.initialize()
                except Exception:
                    self.rotors_view = None
                    self.use_virtual_rotors = True

                    rotor_xform = XFormPrimView(
                        prim_paths_expr=rotor_paths,
                        name="rotors_xform",
                        reset_xform_properties=False,
                    )
                    rotor_xform.initialize()
                    rotor_pos_w, rotor_rot_wxyz = rotor_xform.get_world_poses()
                    rotor_pos_w = rotor_pos_w.reshape(*self.shape, self.num_rotors, 3)
                    rotor_rot_wxyz = rotor_rot_wxyz.reshape(*self.shape, self.num_rotors, 4)

                    base_pos_w, base_rot_wxyz = self.base_link.get_world_poses(clone=True)
                    base_pos_w0 = base_pos_w[0, 0]
                    base_rot_wxyz0 = base_rot_wxyz[0, 0]
                    rotor_pos_w0 = rotor_pos_w[0, 0]
                    rotor_rot_wxyz0 = rotor_rot_wxyz[0, 0]

                    from marinegym.controllers.thruster_allocation import (
                        compute_thruster_allocation_matrix_from_views,
                    )

                    thrust_axis = int(self.params.get("thrust_axis", 0))
                    B = compute_thruster_allocation_matrix_from_views(
                        base_pos_w=base_pos_w0,
                        base_rot_wxyz=base_rot_wxyz0,
                        rotor_pos_w=rotor_pos_w0,
                        rotor_rot_wxyz=rotor_rot_wxyz0,
                        thrust_axis=thrust_axis,
                    )
                    self.thruster_allocation = torch.tensor(
                        B, dtype=torch.float32, device=self.device
                    )

        # Allow forcing virtual-rotor mode via YAML (decouples thrust direction from USD rotor local axes).
        if bool(self.params.get("use_virtual_rotors", False)):
            self.use_virtual_rotors = True

        # If virtual mode is enabled and no allocation was provided, build one from explicit frames
        # (preferred) or from the rotor prim poses (fallback).
        if bool(getattr(self, "use_virtual_rotors", False)) and self.thruster_allocation is None:
            frames = self.params.get("virtual_rotor_frames", None)
            if frames:
                pos_b = torch.zeros((int(self.num_rotors), 3), device=self.device, dtype=torch.float32)
                dir_b = torch.zeros((int(self.num_rotors), 3), device=self.device, dtype=torch.float32)
                filled = torch.zeros((int(self.num_rotors),), device=self.device, dtype=torch.bool)
                for frame in frames:
                    try:
                        name = str(frame.get("name", ""))
                        idx = None
                        if name.startswith("rotor_") and name[6:].isdigit():
                            idx = int(name[6:])
                        if idx is None or idx < 0 or idx >= int(self.num_rotors):
                            continue
                        pos = torch.as_tensor(frame.get("pos", [0.0, 0.0, 0.0]), device=self.device, dtype=torch.float32)
                        direction = torch.as_tensor(frame.get("dir", [1.0, 0.0, 0.0]), device=self.device, dtype=torch.float32)
                        if pos.numel() == 3 and direction.numel() == 3:
                            pos_b[idx] = pos.flatten()
                            dir_b[idx] = direction.flatten()
                            filled[idx] = True
                    except Exception:
                        continue
                if bool(filled.all()):
                    dir_b = torch.nn.functional.normalize(dir_b, dim=-1)
                    m_b = torch.cross(pos_b, dir_b, dim=-1)
                    B = torch.cat([dir_b, m_b], dim=-1).T  # (6, n)
                    self.thruster_allocation = B.to(device=self.device, dtype=torch.float32)
                else:
                    missing = [i for i in range(int(self.num_rotors)) if not bool(filled[i].item())]
                    raise ValueError(
                        f"virtual_rotor_frames is missing entries for rotor indices: {missing}. "
                        "Expected frames named rotor_0..rotor_{num_rotors-1}."
                    )
            elif self.rotors_view is not None:
                # Fallback: compute from current rotor prim poses.
                from marinegym.controllers.thruster_allocation import (
                    compute_thruster_allocation_matrix_from_drone,
                )
                thrust_axis = int(self.params.get("thrust_axis", 0))
                B = compute_thruster_allocation_matrix_from_drone(self, thrust_axis=thrust_axis)
                self.thruster_allocation = torch.tensor(B, dtype=torch.float32, device=self.device)

        rotor_config = self.params["rotor_configuration"]
        self.rotors = T200(rotor_config, dt=self.dt).to(self.device)

        rotor_params = from_module(self.rotors, lock=False)
        self.rotor_params = rotor_params.expand(self.shape).clone()
        self.TIME_CONSTANTS_0 = torch.tensor(self.params["rotor_configuration"]['time_constants'], device=self.device)
        self.FORCE_CONSTANTS_0 = torch.tensor(self.params["rotor_configuration"]['force_constants'], device=self.device)

        # self.tau_up = self.rotor_params["tau_up"]
        # self.tau_down = self.rotor_params["tau_down"]
        # self.throttle = self.rotor_params["throttle"]
        # self.directions = self.rotor_params["directions"]

        self.tau_up = self.rotor_params["tau_up"]
        self.tau_down = self.rotor_params["tau_down"]
        self.directions = self.rotor_params["directions"]
        self.throttle = torch.zeros(*self.shape, self.num_rotors, device=self.device)
        self.rpm = torch.zeros_like(self.throttle)

        self.thrusts = torch.zeros(*self.shape, self.num_rotors, 3, device=self.device)
        self.torques = torch.zeros(*self.shape, 3, device=self.device)
        self.forces = torch.zeros(*self.shape, 3, device=self.device)

        self.pos, self.rot = self.get_world_poses(True)
        self.throttle_difference = torch.zeros(self.throttle.shape[:-1], device=self.device)
        self.heading = torch.zeros(*self.shape, 3, device=self.device)
        self.up = torch.zeros(*self.shape, 3, device=self.device)
        self.vel = self.vel_w = torch.zeros(*self.shape, 6, device=self.device)
        self.vel_b = torch.zeros_like(self.vel_w)
        self.acc = self.acc_w = torch.zeros(*self.shape, 6, device=self.device)
        self.acc_b = torch.zeros_like(self.acc_w)

        self.alpha = 0.9
        
        mass_override = self.params.get("mass", None)
        inertia_override = self.params.get("inertia", None)
        if mass_override is not None:
            if bool(getattr(self, "is_articulation", False)) and hasattr(self, "_view") and hasattr(self._view, "set_body_masses"):
                # Articulation assets may have many links (e.g., rotor links). If only the base_link mass is
                # overridden, the remaining links can keep their default masses and the vehicle becomes
                # strongly negatively/positively buoyant. Treat the YAML `mass` as the *total* mass and
                # distribute it across links, keeping most mass on the base-like link.
                try:
                    body_masses = self._view.get_body_masses(clone=True)  # (*shape, num_bodies)
                    num_bodies = int(body_masses.shape[-1])
                    target_total = float(mass_override)
                    min_link_mass = float(self.params.get("articulation_min_link_mass", 0.01))
                    min_link_mass = max(0.0, min_link_mass)

                    body_names = [str(x) for x in getattr(self._view, "_body_names", [])]
                    root_idx = 0
                    # Prefer an obvious base link name; otherwise keep the root body (idx 0).
                    for i, name in enumerate(body_names):
                        low = name.lower()
                        if low.endswith("base_link") or low.endswith("/base_link"):
                            root_idx = i
                            break
                    else:
                        for i, name in enumerate(body_names):
                            low = name.lower()
                            if low.endswith("default") or low.endswith("/default"):
                                root_idx = i
                                break

                    masses_new = torch.full_like(body_masses, min_link_mass)
                    # Put the remaining mass on the chosen base-like link.
                    remaining = target_total - min_link_mass * float(max(0, num_bodies - 1))
                    remaining = max(min_link_mass, remaining)
                    masses_new[..., root_idx] = remaining

                    # Debug: print total before/after once.
                    try:
                        total_before = float(body_masses[0, 0].sum().item())
                        total_after = float(masses_new[0, 0].sum().item())
                        print(
                            f"[UnderwaterVehicle] articulation mass override: total_before={total_before:.3f} "
                            f"total_after={total_after:.3f} root_idx={root_idx} root_name={body_names[root_idx] if body_names else root_idx}"
                        )
                    except Exception:
                        pass

                    self._view.set_body_masses(masses_new)
                except Exception as e:
                    try:
                        print(
                            f"[UnderwaterVehicle] articulation mass override failed; "
                            f"fallback to base rigid mass only. ({type(e).__name__}: {e})"
                        )
                    except Exception:
                        pass
                    # Fallback: override only the base rigid body mass.
                    masses = torch.full(
                        (self.base_link.count,),
                        float(mass_override),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    self.base_link.set_masses(masses)
            else:
                masses = torch.full(
                    (self.base_link.count,),
                    float(mass_override),
                    device=self.device,
                    dtype=torch.float32,
                )
                self.base_link.set_masses(masses)
        if inertia_override is not None:
            inertia_vals = torch.as_tensor(inertia_override, device=self.device, dtype=torch.float32).flatten()
            if inertia_vals.numel() == 3:
                xx, yy, zz = inertia_vals.tolist()
                inertia_vals = torch.tensor(
                    [xx, 0.0, 0.0, 0.0, yy, 0.0, 0.0, 0.0, zz],
                    device=self.device,
                    dtype=torch.float32,
                )
            if inertia_vals.numel() != 9:
                raise ValueError("inertia must be length 3 or 9")
            inertias = inertia_vals.repeat(self.base_link.count, 1)
            self.base_link.set_inertias(inertias)

        self.masses = self.base_link.get_masses().clone()
        self.gravity = self.masses * 9.81
        self.inertias = self.base_link.get_inertias().reshape(*self.shape, 3, 3).diagonal(0, -2, -1)
        self.volumes = torch.full_like(self.masses, self.params["volume"])
        self.coBMs = torch.full_like(self.masses, self.params["coBM"])
        # Total mass (useful for buoyancy tuning). For articulation assets, sum link masses.
        self.total_masses = self.masses.clone()
        if bool(getattr(self, "is_articulation", False)) and hasattr(self, "_view") and hasattr(self._view, "get_body_masses"):
            try:
                body_masses = self._view.get_body_masses(clone=True)  # (*shape, num_bodies)
                self.total_masses = body_masses.sum(dim=-1, keepdim=True)
            except Exception:
                self.total_masses = self.masses.clone()
        # Cache constants for buoyancy computation (avoid hard-coded 9.8 vs 9.81 mismatch).
        try:
            from omni.isaac.core.simulation_context import SimulationContext

            g_vec = SimulationContext.instance().get_physics_context().get_gravity()
            self._g_mag = float(abs(float(g_vec[2])))
        except Exception:
            self._g_mag = 9.81
        self._water_density = float(self.params.get("water_density", 997.0))
        self._buoyancy_mode = str(self.params.get("buoyancy_mode", "volume")).lower()
        self.MASS_0 = self.masses[0].clone()
        self.INERTIA_0 = (
            self.base_link
            .get_inertias()
            .reshape(*self.shape, 3, 3)[0]
            .diagonal(0, -2, -1)
            .clone()
        )
        self.VOLUME_0 = torch.tensor([[self.params["volume"]]]).to(self.MASS_0)
        self.CoBM_0 = torch.tensor([[self.params["coBM"]]]).to(self.MASS_0)
        self.ADDED_MASS_0 = torch.tensor(self.params["hydro_coef"]["added_mass"]).to(self.MASS_0)
        self.LINEAR_DAMPING_0 = torch.tensor(self.params["hydro_coef"]["linear_damping"]).to(self.MASS_0)
        self.QUADRATIC_DAMPING_0 = torch.tensor(self.params["hydro_coef"]["quadratic_damping"]).to(self.MASS_0)

        logging.info(str(self))

        self.intrinsics = self.intrinsics_spec.expand(self.shape).zero()
        
        self.prev_body_vels = torch.zeros(*self.shape, 6, device=self.device)     
        self.prev_body_acc= torch.zeros(*self.shape, 6, device=self.device)   
        hydro_coef=self.params['hydro_coef']
        self.added_mass_matrix = torch.diag(torch.tensor(hydro_coef["added_mass"])).repeat(*self.shape, 1, 1).to(self.device)
        self.linear_damping_matrix = torch.diag(torch.tensor(hydro_coef["linear_damping"])).repeat(*self.shape, 1, 1).to(self.device)
        self.quadratic_damping_matrix = torch.diag(torch.tensor(hydro_coef["quadratic_damping"])).repeat(*self.shape, 1, 1).to(self.device)
        self.flow_vels = torch.zeros(*self.shape, 6, device=self.device)
        self.max_flow_vel = torch.zeros(*self.shape, 6, device=self.device)
        self.flow_noise_scale = torch.zeros(*self.shape, 6, device=self.device)

    # def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
    #     rotor_cmds = actions.expand(*self.shape, self.num_rotors)        
    #     last_throttle = self.throttle.clone()
    #     thrusts, moments = vmap(vmap(self.rotors, randomness="different"), randomness="same")(
    #         rotor_cmds, self.rotor_params
    #     )

    #     rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
    #     torque_axis = quat_axis(rotor_rot.flatten(end_dim=-2), axis=2).unflatten(0, (*self.shape, self.num_rotors))

    #     self.thrusts[..., 0] = thrusts
    #     self.torques[:] = (moments.unsqueeze(-1) * torque_axis).sum(-2)
    #     self.forces.zero_()
    #     flow_vels = self.flow_vels  + torch.rand_like(self.flow_vels) * self.flow_noise_scale
    #     hydro_forces, hydro_torques = self.apply_hydrodynamic_forces(flow_vels)
    #     self.forces += hydro_forces
    #     self.torques += hydro_torques

    #     self.rotors_view.apply_forces_and_torques_at_pos(
    #         self.thrusts.reshape(-1, 3), 
    #         is_global=False
    #     )

    #     self.base_link.apply_forces_and_torques_at_pos(
    #         self.forces.reshape(-1, 3), 
    #         self.torques.reshape(-1, 3),
    #         is_global=False
    #     )
    #     self.throttle_difference[:] = torch.norm(self.throttle - last_throttle, dim=-1)
    #     return self.throttle.sum(-1)

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)
        last_throttle = self.throttle.clone()

        # ✅ vmap 제거: 상태(throttle/rpm) 입력/출력으로 계산
        thrusts, moments, new_throttle, new_rpm = self.rotors(
            rotor_cmds,
            self.throttle,
            self.rpm,
            tau_up=self.tau_up,
            tau_down=self.tau_down,
            directions=self.directions,
            time_constants=self.TIME_CONSTANTS_0,
            force_constants=self.FORCE_CONSTANTS_0,
        )

        # ✅ 상태 업데이트 (Parameter 말고 Tensor여야 함!)
        self.throttle.copy_(new_throttle)
        self.rpm.copy_(new_rpm)

        use_virtual_rotors = bool(getattr(self, "use_virtual_rotors", False))
        if not use_virtual_rotors:
            rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
            torque_axis = quat_axis(
                rotor_rot.flatten(end_dim=-2), axis=2
            ).unflatten(0, (*self.shape, self.num_rotors))
            self.thrusts.zero_()
            axes_idx = getattr(self, "_thrust_axes_idx", None)
            axes_sign = getattr(self, "_thrust_axes_sign", None)
            if axes_idx is None or axes_sign is None:
                axis = int(getattr(self, "_thrust_axis_idx", 0))
                sign = float(getattr(self, "_thrust_axis_sign", 1.0))
                self.thrusts[..., axis] = thrusts * sign
            else:
                # Fill per-rotor axis components efficiently.
                # thrusts: (*shape, R)
                # thrusts_vec: (*shape, R, 3)
                thrusts_vec = torch.zeros_like(self.thrusts)
                s = axes_sign.to(device=thrusts.device, dtype=thrusts.dtype).view(1, 1, -1, 1)
                u = thrusts.unsqueeze(-1) * s  # (*shape, R, 1)
                idx = axes_idx.to(device=thrusts.device).view(1, 1, -1, 1).expand(*thrusts.shape, 1)
                thrusts_vec.scatter_(dim=-1, index=idx, src=u)
                self.thrusts.copy_(thrusts_vec)
            self.torques[:] = (moments.unsqueeze(-1) * torque_axis).sum(-2)
        else:
            self.thrusts[..., 0] = thrusts

        self.forces.zero_()
        flow_vels = self.flow_vels + torch.rand_like(self.flow_vels) * self.flow_noise_scale
        hydro_forces, hydro_torques = self.apply_hydrodynamic_forces(flow_vels)
        self.forces += hydro_forces
        self.torques += hydro_torques

        if not use_virtual_rotors:
            self.rotors_view.apply_forces_and_torques_at_pos(
                self.thrusts.reshape(-1, 3),
                is_global=False,
            )
        else:
            B = self.thruster_allocation.to(self.thrusts.device, dtype=self.thrusts.dtype)
            tau = torch.einsum("ij,...j->...i", B, thrusts)
            self.forces += tau[..., 0:3]
            self.torques += tau[..., 3:6]

        # Apply wrenches in the global frame to avoid relying on a particular base_link local-axis convention.
        forces_w = quat_rotate(self.rot, self.forces)
        torques_w = quat_rotate(self.rot, self.torques)
        self.base_link.apply_forces_and_torques_at_pos(
            forces_w.reshape(-1, 3),
            torques_w.reshape(-1, 3),
            is_global=True,
        )

        self.throttle_difference[:] = torch.norm(self.throttle - last_throttle, dim=-1)
        return self.throttle.sum(-1)


    def apply_hydrodynamic_forces(self, flow_vels_w) -> TensorDict:

        body_vels = self.vel_b.clone()
        flow_vels_b = torch.cat([
            quat_rotate_inverse(self.rot, flow_vels_w[..., :3]),
            quat_rotate_inverse(self.rot, flow_vels_w[..., 3:])
        ], dim=-1)
        body_vels -=  flow_vels_b
        body_vels[..., [1,2,4,5]] *= -1

        body_acc = self.calculate_acc(body_vels)        
        damping = self.calculate_damping(body_vels.squeeze(1))
        added_mass = self.calculate_added_mass(body_acc.squeeze(1))
        coriolis = self.calculate_corilis(body_vels.squeeze(1))
        buoyancy = self.calculate_buoyancy(self.rot.squeeze(1))
        
        hydro = - (added_mass + coriolis + damping)
        hydro[:, [1, 2, 4, 5]] *= -1
        hydro = hydro.unsqueeze(1)
        buoyancy = buoyancy.unsqueeze(1)

        return hydro[..., 0:3] + buoyancy[..., 0:3], hydro[..., 3:6] + buoyancy[..., 3:6]
    
    def calculate_acc(self, body_vels):
        alpha = 0.3
        acc = (body_vels - self.prev_body_vels) / self.dt
        filteredAcc = (1.0-alpha)* self.prev_body_acc + alpha * acc
        self.prev_body_vels = body_vels.clone()
        self.prev_body_acc = filteredAcc.clone() 

        return filteredAcc
    
    def calculate_damping(self, body_vels):
        maintained_body_vels = torch.diag_embed(body_vels)
        maintained_body_vels[:, 1, 5] = body_vels[:, 5]
        maintained_body_vels[:, 2, 4] = body_vels[:, 4]
        maintained_body_vels[:, 4, 2] = body_vels[:, 2]
        maintained_body_vels[:, 5, 1] = body_vels[:, 1] 
        damping_matrix = self.linear_damping_matrix[:,0,:,:] + self.quadratic_damping_matrix[:,0,:,:] * torch.abs(maintained_body_vels)
        damping = damping_matrix @ body_vels.unsqueeze(2)
        damping = damping.squeeze(2)   
        
        return damping
    
    def calculate_added_mass(self, body_acc):
        added_mass = self.added_mass_matrix[:,0,:,:] @ body_acc.unsqueeze(2)
        added_mass = added_mass.squeeze(2)
        
        return added_mass
    
    def calculate_corilis(self, body_vels):
        ab = self.added_mass_matrix[:,0,:,:] @ body_vels.unsqueeze(2)
        ab =ab.squeeze(2)
        coriolis = torch.zeros(*self.shape, 6, device=self.device)
        coriolis.squeeze_(dim=1)
        coriolis[:, 0:3] = - torch.cross(ab[:, 0:3], body_vels[:, 3:6], dim=1)
        coriolis[:, 3:6] = - (torch.cross(ab[:, 0:3], body_vels[:, 0:3], dim=1) + torch.cross(ab[:, 3:6], body_vels[:, 3:6], dim=1))
        
        return coriolis
    
    def calculate_buoyancy(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        """
        Returns buoyancy wrench in the base_link local frame.

        We compute buoyancy as a world-up force (magnitude rho*g*V) and rotate it into the body frame
        via quaternion, avoiding fragile Euler/sign conventions that differ across assets.
        """
        # quat_wxyz: (..., 4) where ... flattens to (N,)
        q = quat_wxyz.reshape(-1, 4)
        n = q.shape[0]

        rho = float(getattr(self, "_water_density", 997.0))
        g_mag = float(getattr(self, "_g_mag", 9.81))
        mode = str(getattr(self, "_buoyancy_mode", "volume")).lower()
        if mode in ("neutral", "neutral_mass", "match_weight"):
            buoyancy_force = (g_mag * self.total_masses[:, 0, 0]).reshape(-1)
        else:
            buoyancy_force = (rho * g_mag * self.volumes[:, 0, 0]).reshape(-1)

        f_w = torch.zeros((n, 3), device=self.device, dtype=q.dtype)
        f_w[:, 2] = buoyancy_force
        # World -> body
        f_b = quat_rotate_inverse(q, f_w)

        # CoB offset along configured axis in the body frame (restoring torque).
        # Note: coBM is a signed scalar; axis is unit-length.
        dis = self.coBMs[:, 0, 0].reshape(-1).to(dtype=q.dtype)
        axis_b = self._cobm_axis_unit.to(dtype=q.dtype).view(1, 3).expand(n, 3)
        r_cb = axis_b * dis.unsqueeze(-1)
        tau_b = -torch.cross(r_cb, f_b, dim=-1)

        wrench = torch.zeros((n, 6), device=self.device, dtype=q.dtype)
        wrench[:, 0:3] = f_b
        wrench[:, 3:6] = tau_b
        return wrench.view(*self.shape, 6).squeeze(1)

    def get_state(self, check_nan: bool=False, env_frame: bool=True):
        self.pos[:], self.rot[:] = self.get_world_poses(True)
        if env_frame and hasattr(self, "_envs_positions"):
            self.pos.sub_(self._envs_positions)
        
        vel_w = self.get_velocities(True)
        vel_b = torch.cat([
            quat_rotate_inverse(self.rot, vel_w[..., :3]),
            quat_rotate_inverse(self.rot, vel_w[..., 3:])
        ], dim=-1)
        self.vel_w[:] = vel_w
        self.vel_b[:] = vel_b
        
        self.heading[:] = quat_axis(self.rot, axis=0)
        self.up[:] = quat_axis(self.rot, axis=2)
        # throttle is already in [-1, 1] (supports reverse); do not rescale.
        state = [self.pos, self.rot, self.vel, self.heading, self.up, self.throttle]
        if self.use_force_sensor:
            self.force_readings, self.torque_readings = self.get_force_sensor_forces().chunk(2, -1)
            # normalize by mass and inertia
            force_reading_norms = self.force_readings.norm(dim=-1, keepdim=True)
            force_readings = (
                self.force_readings
                / force_reading_norms
                * symlog(force_reading_norms)
                / self.gravity.unsqueeze(-2)
            )
            torque_readings = self.torque_readings / self.INERTIA_0.unsqueeze(-2)
            state.append(force_readings.flatten(-2))
            state.append(torque_readings.flatten(-2))
        state = torch.cat(state, dim=-1)
        if check_nan:
            assert not torch.isnan(state).any()
        return state

    def _reset_idx(self, env_ids: torch.Tensor, train: bool=True):
        if env_ids is None:
            env_ids = torch.arange(self.shape[0], device=self.device)
    
        self.thrusts[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.vel[env_ids] = 0.
        self.acc[env_ids] = 0.
    
        # ✅ 여기 추가
        self.throttle[env_ids] = 0.0
        self.rpm[env_ids] = 0.0
    
        self.throttle_difference[env_ids] = 0.0
        self.prev_body_acc[env_ids] = 0.0
        self.prev_body_vels[env_ids] = 0.0
        self.flow_vels[env_ids] = torch.rand_like(self.flow_vels[env_ids]) * self.max_flow_vel[env_ids]
        return env_ids

    
    def set_flow_velocities(self, env_ids, max_flow_velocity, flow_velocity_gaussian_noise):
        self.max_flow_vel[env_ids,0,:] = torch.tensor(max_flow_velocity,dtype=torch.float32,device=self.device)
        self.flow_noise_scale[env_ids,0,:] = torch.tensor(flow_velocity_gaussian_noise,dtype=torch.float32,device=self.device)
    
    def get_thrust_to_weight_ratio(self):
        return self.KF.sum(-1, keepdim=True) / (self.masses * 9.81)

    def get_linear_smoothness(self):
        return - (
            torch.norm(self.acc[..., :3], dim=-1) 
            + torch.norm(self.jerk[..., :3], dim=-1)
        )
    
    def get_angular_smoothness(self):
        return - (
            torch.sum(self.acc[..., 3:].abs(), dim=-1)
            + torch.sum(self.jerk[..., 3:].abs(), dim=-1)
        )
    
    def __str__(self):
        default_params = "\n".join([
            "Default parameters:",
            f"Mass: {self.MASS_0.tolist()}",
            f"Inertia: {self.INERTIA_0.tolist()}",
        ])
        return default_params

    @staticmethod
    def make(
        drone_model: str,
        controller: str = None,
        device: str = "cpu",
        is_articulation: bool | None = None,
    ):
        drone_cls = UnderwaterVehicle.REGISTRY[drone_model]
        if is_articulation is None:
            drone = drone_cls()
        else:
            drone = drone_cls(is_articulation=bool(is_articulation))
        controller_cls = None
        if controller is not None:
            from marinegym.controllers import ControllerBase
            controller_cls = (
                ControllerBase.REGISTRY.get(controller)
                or ControllerBase.REGISTRY.get(str(controller).lower())
            )
        return drone, controller_cls
