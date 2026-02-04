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


import abc
import os.path as osp
from contextlib import contextmanager
from typing import Dict, Sequence, Type

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.torch as torch_utils

import omni.timeline
import torch
from pxr import PhysxSchema, UsdPhysics
from marinegym.views import ArticulationView, RigidPrimView
from omni.isaac.core.simulation_context import SimulationContext
from torchrl.data import TensorSpec

import marinegym.utils.kit as kit_utils

from marinegym.robots.config import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
    RobotCfg,
)

ASSET_PATH = osp.join(osp.dirname(__file__), "assets")
TEMPLATE_PRIM_PATH = "/World/envs/env_0"


class RobotBase(abc.ABC):

    usd_path: str
    fixed_usd_path: str
    cfg_cls = RobotCfg

    _robots = {}
    _envs_positions: torch.Tensor

    REGISTRY: Dict[str, Type["RobotBase"]] = {}

    def __init__(self, name: str, cfg: RobotCfg = None, is_articulation=True) -> None:
        if name is None:
            name = self.__class__.__name__
        if name in RobotBase._robots:
            raise RuntimeError
        RobotBase._robots[name] = self
        if cfg is None:
            cfg = self.cfg_cls()

        self.name = name
        self.is_articulation = is_articulation
        self.rigid_props: RigidBodyPropertiesCfg = cfg.rigid_props
        self.articulation_props: ArticulationRootPropertiesCfg = cfg.articulation_props

        self.n = 0

        if SimulationContext._instance is None:
            raise RuntimeError("The SimulationContext is not created.")

        self.cfg = cfg
        self.device = SimulationContext.instance()._device
        self.dt = SimulationContext.instance().get_physics_dt()
        self.gravity = SimulationContext.instance().get_physics_context().get_gravity()
        self.state_spec: TensorSpec
        self.action_spec: TensorSpec
        self.initialized = False
        self._non_articulation_base_relpath: str | None = None

    def _ensure_non_articulation_base_rigid_body(self, prim_path: str) -> None:
        root_prim = prim_utils.get_prim_at_path(prim_path)
        if not root_prim.IsValid():
            return

        # If the robot class specifies a non-articulation base rigid body under its root,
        # prefer that and avoid forcing the top-level prim to become a rigid body.
        if self._non_articulation_base_relpath:
            try:
                base_path = prim_path.rstrip("/") + "/" + str(self._non_articulation_base_relpath).lstrip("/")
                base_prim = prim_utils.get_prim_at_path(base_path)
                if base_prim.IsValid():
                    if not UsdPhysics.RigidBodyAPI(base_prim):
                        UsdPhysics.RigidBodyAPI.Apply(base_prim)
                    if not PhysxSchema.PhysxRigidBodyAPI(base_prim):
                        PhysxSchema.PhysxRigidBodyAPI.Apply(base_prim)
            except Exception:
                pass
            return

        # Minimal guarantee: make the spawned root a rigid body so that RigidPrimView patterns
        # like "/World/envs/*/{name}_*" match at least one rigid body after sim.reset().
        try:
            if not UsdPhysics.RigidBodyAPI(root_prim):
                UsdPhysics.RigidBodyAPI.Apply(root_prim)
            if not PhysxSchema.PhysxRigidBodyAPI(root_prim):
                PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        except Exception:
            pass

        # Heuristic: pick a descendant that likely represents the main body, not a rotor/sensor.
        exclude_hints = ("rotor", "thruster", "prop", "motor", "fan", "camera", "sensor", "sonar", "dvl")
        prefer_exact = ("/base_link",)
        prefer_contains = ("base", "hull", "chassis", "body", "frame", "link")

        candidates: list[tuple[tuple[int, int, int], str]] = []
        queue: list[tuple[object, int]] = [(root_prim, 0)]
        while queue:
            prim, depth = queue.pop(0)
            for child in prim_utils.get_prim_children(prim):
                queue.append((child, depth + 1))
                path = prim_utils.get_prim_path(child)
                low = path.lower()
                if any(h in low for h in exclude_hints):
                    continue
                # Prefer xform-ish nodes higher in the tree.
                priority = 2
                if any(low.endswith(suf) for suf in prefer_exact):
                    priority = 0
                elif any(h in low for h in prefer_contains):
                    priority = 1
                candidates.append(((priority, depth, len(path)), path))

        if not candidates:
            return

        _, chosen = min(candidates, key=lambda it: it[0])
        rel = chosen[len(prim_path) + 1 :] if chosen.startswith(prim_path + "/") else None
        if not rel:
            return

        chosen_prim = prim_utils.get_prim_at_path(chosen)
        if not chosen_prim.IsValid():
            return

        if not UsdPhysics.RigidBodyAPI(chosen_prim):
            UsdPhysics.RigidBodyAPI.Apply(chosen_prim)
        if not PhysxSchema.PhysxRigidBodyAPI(chosen_prim):
            PhysxSchema.PhysxRigidBodyAPI.Apply(chosen_prim)

        self._non_articulation_base_relpath = rel

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if cls.__name__ in RobotBase.REGISTRY:
            raise ValueError
        super().__init_subclass__(**kwargs)
        RobotBase.REGISTRY[cls.__name__] = cls
        RobotBase.REGISTRY[cls.__name__.lower()] = cls

    def spawn(
        self,
        translations=[(0.0, 0.0, 0.5)],
        orientations=None,
        prim_paths: Sequence[str] = None
    ):
        if SimulationContext.instance()._physics_sim_view is not None:
            raise RuntimeError(
                "Cannot spawn robots after simulation_context.reset() is called."
            )

        translations = torch.atleast_2d(
            torch.as_tensor(translations, device=self.device)
        )
        n = translations.shape[0]

        if orientations is None:
            orientations = [None for _ in range(n)]

        if prim_paths is None:
            prim_paths = [f"{TEMPLATE_PRIM_PATH}/{self.name}_{i}" for i in range(n)]

        if not len(translations) == len(prim_paths):
            raise ValueError

        prims = []
        for prim_path, translation, orientation in zip(prim_paths, translations, orientations):
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")
            prim = self._create_prim(prim_path, translation, orientation)
            if not self.is_articulation:
                # Ensure at least one rigid body exists under this asset before sim.reset(),
                # otherwise RigidPrimView will not match any rigid bodies.
                self._ensure_non_articulation_base_rigid_body(prim_path)
            # apply rigid body properties
            kit_utils.set_nested_rigid_body_properties(
                prim_path,
                linear_damping=self.rigid_props.linear_damping,
                angular_damping=self.rigid_props.angular_damping,
                max_linear_velocity=self.rigid_props.max_linear_velocity,
                max_angular_velocity=self.rigid_props.max_angular_velocity,
                max_depenetration_velocity=self.rigid_props.max_depenetration_velocity,
                enable_gyroscopic_forces=True,
                disable_gravity=self.rigid_props.disable_gravity,
                retain_accelerations=self.rigid_props.retain_accelerations,
            )
            # articulation root settings
            if self.is_articulation:
                kit_utils.set_articulation_properties(
                    prim_path,
                    enable_self_collisions=self.articulation_props.enable_self_collisions,
                    solver_position_iteration_count=self.articulation_props.solver_position_iteration_count,
                    solver_velocity_iteration_count=self.articulation_props.solver_velocity_iteration_count,
                )
            prims.append(prim)

        self.n += n
        return prims

    def _create_prim(self, prim_path, translation, orientation):
        prim = prim_utils.create_prim(
            prim_path,
            usd_path=self.usd_path,
            translation=translation,
            orientation=orientation,
        )
        return prim

    def initialize(
        self,
        prim_paths_expr: str = None,
    ):
        if SimulationContext.instance()._physics_sim_view is None:
            raise RuntimeError(
                f"Cannot initialize {self.__class__.__name__} before the simulation context resets."
                "Call simulation_context.reset() first."
            )
        if prim_paths_expr is None:
            prim_paths_expr = f"/World/envs/.*/{self.name}_.*"
        self.prim_paths_expr = prim_paths_expr

        # create handles
        # -- robot articulation
        if self.is_articulation:
            prim_paths_expr_art = self.prim_paths_expr

            # Some USD assets reference an articulation root under the spawned container prim
            # (e.g., /World/envs/env_0/Robot_0/<articulation_root>). In that case, the container prim
            # itself does not have ArticulationRootAPI and using it for ArticulationView leads to
            # mismatched root transforms (often visible as flipped/offset visuals).
            try:
                template_root = f"{TEMPLATE_PRIM_PATH}/{self.name}_0"
                root_prim = prim_utils.get_prim_at_path(template_root)
                if root_prim.IsValid() and not UsdPhysics.ArticulationRootAPI(root_prim):
                    queue = [root_prim]
                    candidates: list[str] = []
                    while queue:
                        prim = queue.pop(0)
                        for child in prim_utils.get_prim_children(prim):
                            queue.append(child)
                            path = prim_utils.get_prim_path(child)
                            low = path.lower()
                            if "rotor" in low or "thruster" in low:
                                continue
                            if UsdPhysics.ArticulationRootAPI(child):
                                candidates.append(path)
                    if candidates:
                        base_like = next(
                            (p for p in candidates if p.lower().endswith("/base_link")),
                            None,
                        )
                        chosen = base_like or min(
                            set(candidates), key=lambda p: (p.count("/"), len(p))
                        )
                        if chosen.startswith(template_root + "/"):
                            rel = chosen[len(template_root) + 1 :]
                            num_envs = int(self.shape[0]) if getattr(self, "shape", None) else 1
                            try:
                                if getattr(RobotBase, "_envs_positions", None) is not None:
                                    num_envs = int(RobotBase._envs_positions.shape[0])
                            except Exception:
                                num_envs = 1
                            prim_paths = [
                                f"/World/envs/env_{env_i}/{self.name}_{robot_i}/{rel}"
                                for env_i in range(num_envs)
                                for robot_i in range(int(self.n))
                            ]
                            print(
                                f"[RobotBase] articulation root for '{self.name}' is nested; "
                                f"using '{rel}' instead of container prim."
                            )
                            prim_paths_expr_art = prim_paths
            except Exception:
                prim_paths_expr_art = self.prim_paths_expr

            self._view = ArticulationView(
                prim_paths_expr_art,
                reset_xform_properties=False,
                shape=(-1, self.n)
            )
            self.articulation = self
        else:
            # NOTE: Some USD assets (e.g., BlueROVHeavy) do not have a rigid-body schema on the
            # top-level prim (e.g., /World/envs/env_0/BlueROVHeavy_0). For those, we pick a
            # descendant rigid body (prefer base_link) and build an explicit prim-path list.
            num_envs = 1
            try:
                if getattr(RobotBase, "_envs_positions", None) is not None:
                    num_envs = int(RobotBase._envs_positions.shape[0])
            except Exception:
                num_envs = 1

            template_root = f"{TEMPLATE_PRIM_PATH}/{self.name}_0"
            root_prim = prim_utils.get_prim_at_path(template_root)

            # Prefer the root prim if it is already a rigid body (common when the asset was authored
            # correctly or when the robot class creates a root collider).
            if root_prim.IsValid() and (
                UsdPhysics.RigidBodyAPI(root_prim) or PhysxSchema.PhysxRigidBodyAPI(root_prim)
            ):
                rel = None
                self._non_articulation_base_relpath = None
            else:
                rel = self._non_articulation_base_relpath
            if rel is None:
                # Fallback: resolve a descendant rigid body after cloning/reset (schema may only be
                # discoverable reliably at this point for some assets).
                try:
                    if root_prim.IsValid():
                        queue = [root_prim]
                        candidates: list[str] = []
                        while queue:
                            prim = queue.pop(0)
                            for child in prim_utils.get_prim_children(prim):
                                queue.append(child)
                                path = prim_utils.get_prim_path(child)
                                low = path.lower()
                                if "rotor" in low or "thruster" in low:
                                    continue
                                if not (UsdPhysics.RigidBodyAPI(child) or PhysxSchema.PhysxRigidBodyAPI(child)):
                                    continue
                                candidates.append(path)
                        if candidates:
                            base_like = next(
                                (p for p in candidates if p.lower().endswith("/base_link")),
                                None,
                            )
                            chosen = base_like or min(
                                set(candidates), key=lambda p: (p.count("/"), len(p))
                            )
                            if chosen.startswith(template_root + "/"):
                                rel = chosen[len(template_root) + 1 :]
                                self._non_articulation_base_relpath = rel
                        else:
                            # Debug aid: if we cannot find any rigid bodies under the robot root,
                            # print a small summary to help identify the correct base prim.
                            queue = [root_prim]
                            sample_paths: list[str] = []
                            rigid_like = 0
                            rotor_like = 0
                            coll_like = 0
                            artic_like = 0
                            rotor0_path = None
                            while queue and len(sample_paths) < 80:
                                prim = queue.pop(0)
                                for child in prim_utils.get_prim_children(prim):
                                    queue.append(child)
                                    path = prim_utils.get_prim_path(child)
                                    low = path.lower()
                                    if "/looks" in low or "/materials" in low or "/shaders" in low:
                                        continue
                                    if "rotor" in low:
                                        rotor_like += 1
                                    if "rotor_0" in low and rotor0_path is None:
                                        rotor0_path = path
                                    if UsdPhysics.RigidBodyAPI(child) or PhysxSchema.PhysxRigidBodyAPI(child):
                                        rigid_like += 1
                                    if UsdPhysics.CollisionAPI(child) or PhysxSchema.PhysxCollisionAPI(child):
                                        coll_like += 1
                                    if UsdPhysics.ArticulationRootAPI(child):
                                        artic_like += 1
                                    sample_paths.append(path)
                                    if len(sample_paths) >= 80:
                                        break
                            print(
                                f"[RobotBase] non-articulation robot '{self.name}': "
                                f"no descendant rigid bodies found under '{template_root}'. "
                                f"sample_prims={len(sample_paths)} rotor_named={rotor_like} "
                                f"rigid_schema={rigid_like} collision_schema={coll_like} articulation_schema={artic_like}"
                            )
                            if rotor0_path is not None:
                                print(f"[RobotBase] found rotor_0-like prim: {rotor0_path}")
                            if sample_paths:
                                print(
                                    "[RobotBase] sample prim paths:\n  - "
                                    + "\n  - ".join(sample_paths[:50])
                                )
                except Exception:
                    rel = None

            prim_paths_expr_rb = self.prim_paths_expr
            if rel:
                prim_paths = [
                    f"/World/envs/env_{env_i}/{self.name}_{robot_i}/{rel}"
                    for env_i in range(num_envs)
                    for robot_i in range(int(self.n))
                ]
                print(
                    f"[RobotBase] non-articulation root prim '{self.name}' "
                    f"does not appear to be a rigid body; using descendant rigid body '{rel}'."
                )
                prim_paths_expr_rb = prim_paths

            self._view = RigidPrimView(
                prim_paths_expr_rb,
                reset_xform_properties=False,
                shape=(-1, self.n),
                # track_contact_forces=True
            )
            self.articulation = None
            self.articulation_indices = None

        self._view.initialize()
        # set the default state
        self._view.post_reset()
        self.shape = torch.arange(self._view.count).reshape(-1, self.n).shape

        self.prim_paths = self._view.prim_paths
        self.initialized = True

    @abc.abstractmethod
    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor, train: bool=True):
        raise NotImplementedError

    def get_world_poses(self, clone: bool=False):
        return self._view.get_world_poses(clone=clone)

    def set_world_poses(self, positions: torch.Tensor=None, orientations: torch.Tensor=None, env_indices: torch.Tensor = None):
        return self._view.set_world_poses(positions, orientations, env_indices=env_indices)

    def get_velocities(self, clone: bool=False):
        return self._view.get_velocities(clone=clone)

    def set_velocities(self, velocities: torch.Tensor, env_indices: torch.Tensor = None):
        return self._view.set_velocities(velocities, env_indices=env_indices)

    def get_joint_positions(self, clone: bool=False):
        if not self.is_articulation:
            raise NotImplementedError
        return self._view.get_joint_positions(clone=clone)

    def set_joint_positions(self, pos: torch.Tensor, env_indices: torch.Tensor = None):
        if not self.is_articulation:
            raise NotImplementedError
        return self._view.set_joint_positions(pos, env_indices=env_indices)

    def set_joint_position_targets(self, pos: torch.Tensor, env_indices: torch.Tensor = None):
        if not self.is_articulation:
            raise NotImplementedError
        self._view.set_joint_position_targets(pos, env_indices=env_indices)

    def get_joint_velocities(self, clone: bool=False):
        return self._view.get_joint_velocities(clone=clone)

    def set_joint_velocities(self, vel: torch.Tensor, env_indices: torch.Tensor = None):
        return self._view.set_joint_velocities(vel, env_indices=env_indices)

    def get_force_sensor_forces(self, clone: bool=False):
        if self.is_articulation:
            forces = self._view.get_force_sensor_forces(clone=clone)
        else:
            forces = self.articulation._view.get_force_sensor_forces(clone=clone)
            forces = forces[..., self.articulation_indices, :]
            forces = forces.reshape(*self.shape, 1, 6)
        return forces

    def get_state(self):
        raise NotImplementedError
