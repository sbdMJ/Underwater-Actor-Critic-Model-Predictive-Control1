import omni.isaac.core.utils.prims as prim_utils
from pxr import PhysxSchema, UsdGeom, UsdPhysics

from marinegym.robots.robot import ASSET_PATH
from marinegym.robots.drone.underwaterVehicle import UnderwaterVehicle

import math
import re


def _normalize3(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    n = math.sqrt(x * x + y * y + z * z)
    if n <= 0.0:
        return 1.0, 0.0, 0.0
    return x / n, y / n, z / n


def _quat_from_two_vectors(v_from, v_to):
    """
    Returns a unit quaternion (w, x, y, z) rotating v_from -> v_to.
    """
    ax, ay, az = _normalize3(v_from)
    bx, by, bz = _normalize3(v_to)
    dot = ax * bx + ay * by + az * bz

    # 180-degree rotation: pick an arbitrary orthogonal axis.
    if dot < -0.999999:
        if abs(ax) < 0.9:
            ox, oy, oz = 1.0, 0.0, 0.0
        else:
            ox, oy, oz = 0.0, 1.0, 0.0
        rx = ay * oz - az * oy
        ry = az * ox - ax * oz
        rz = ax * oy - ay * ox
        rx, ry, rz = _normalize3((rx, ry, rz))
        return 0.0, rx, ry, rz

    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    qw = 1.0 + dot
    qx, qy, qz = cx, cy, cz
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return qw / n, qx / n, qy / n, qz / n


class BlueROVHeavy(UnderwaterVehicle):

    # NOTE: Asset file is stored as the original assembly USD name.
    usd_path: str = ASSET_PATH + "/usd/BlueROVHeavy/BlueROVHeavy.usd"
    param_path: str = ASSET_PATH + "/usd/BlueROVHeavy/BlueROVHeavy.yaml"
    fixed_usd_path: str = ASSET_PATH + "/usd/BlueROVHeavy/fixed_usd/BlueROVHeavy/BlueROVHeavy.usd"

    def _prune_visual_subtree(self, visual_root_path: str) -> None:
        """
        Hide all prims under `visual_root_path` except:
        - any prim named "Default"
        - any prim named "rotor_0" .. "rotor_{num_rotors-1}"

        This is for stage hygiene / debugging: keep only the main body + rotors visible.
        """
        keep_leaf_names = {"Default"} | {f"rotor_{i}" for i in range(int(self.num_rotors))}
        rotor_leaf_re = re.compile(r"^rotor_(\d+)$")

        visual_root = prim_utils.get_prim_at_path(visual_root_path)
        if not visual_root.IsValid():
            return

        all_paths: list[str] = []
        queue = [visual_root]
        while queue:
            prim = queue.pop(0)
            for child in prim_utils.get_prim_children(prim):
                queue.append(child)
                all_paths.append(prim_utils.get_prim_path(child))

        keep_roots: list[str] = []
        for path in all_paths:
            leaf = path.rsplit("/", 1)[-1]
            if leaf in keep_leaf_names:
                keep_roots.append(path)
                continue
            m = rotor_leaf_re.match(leaf)
            if m and int(m.group(1)) < int(self.num_rotors):
                keep_roots.append(path)

        if not keep_roots:
            return

        keep_set: set[str] = {visual_root_path}

        # Keep all ancestors of keep roots (up to visual root).
        for kp in keep_roots:
            cur = kp
            while True:
                keep_set.add(cur)
                if cur == visual_root_path:
                    break
                if not cur.startswith(visual_root_path + "/"):
                    break
                parent = cur.rsplit("/", 1)[0]
                if parent == cur:
                    break
                cur = parent

        # Keep the full subtree under each keep root.
        for path in all_paths:
            if any(path == kp or path.startswith(kp + "/") for kp in keep_roots):
                keep_set.add(path)

        def _set_visible(path: str, visible: bool) -> None:
            prim = prim_utils.get_prim_at_path(path)
            if not prim.IsValid():
                return
            try:
                imageable = UsdGeom.Imageable(prim)
                if not imageable.GetPrim().IsValid():
                    return
                attr = imageable.GetVisibilityAttr() or imageable.CreateVisibilityAttr()
                attr.Set("inherited" if visible else "invisible")
            except Exception:
                return

        # First hide everything under visual, then re-enable the keep set.
        for path in all_paths:
            _set_visible(path, False)
        for path in sorted(keep_set, key=lambda s: (s.count("/"), len(s))):
            _set_visible(path, True)

    def __init__(self, name: str = None, cfg=None, is_articulation: bool = True) -> None:
        # is_articulation=False: visual-only USD를 안전하게 쓰기 위해 wrapper rigid body("body")를 생성하는 경로.
        # is_articulation=True: USD를 그대로 스폰(articulation 기반)하는 경로. (USD에 articulation root가 있어야 함)
        super().__init__(name=name, cfg=cfg, is_articulation=is_articulation)
        if not bool(is_articulation):
            # This asset is visual-only; we create a dedicated rigid body under the root.
            self._non_articulation_base_relpath = "body"

    def _create_prim(self, prim_path, translation, orientation):
        if bool(getattr(self, "is_articulation", False)):
            return super()._create_prim(prim_path, translation, orientation)

        # Create a wrapper root, then a simple rigid-body "body" and attach the visual USD under it.
        try:
            import marinegym.utils.kit as kit_utils

            root = prim_utils.create_prim(
                prim_path,
                "Xform",
                translation=translation,
                orientation=orientation,
            )

            body_path = f"{prim_path}/body"
            body = prim_utils.create_prim(
                body_path,
                "Capsule",
                translation=(0.0, 0.0, 0.0),
                attributes={"radius": 0.18, "height": 0.70},
            )
            UsdPhysics.RigidBodyAPI.Apply(body)
            UsdPhysics.CollisionAPI.Apply(body)
            PhysxSchema.PhysxRigidBodyAPI.Apply(body)
            PhysxSchema.PhysxCollisionAPI.Apply(body)
            mass_api = UsdPhysics.MassAPI.Apply(body)
            mass_attr = mass_api.GetMassAttr() or mass_api.CreateMassAttr()
            if mass_attr.Get() in (None, 0.0):
                mass_attr.Set(15.0)
            try:
                UsdGeom.Imageable(body).MakeInvisible()
            except Exception:
                pass

            # Attach visual assembly under the rigid body so it follows the physics actor.
            prim_utils.create_prim(
                f"{body_path}/visual",
                usd_path=self.usd_path,
                translation=(0.0, 0.0, 0.0),
            )

            # Create lightweight rotor frame prims under the rigid body.
            # The referenced visual assembly may use payloads/references such that deep prim paths
            # (e.g., ".../visual/.../rotor_0") are not available during early initialization.
            # UnderwaterVehicle can use these Xform prims to compute the thruster allocation matrix.
            frames = self.params.get("virtual_rotor_frames", None)
            if not frames:
                c45 = 0.70710678
                frames = [
                    {"name": "rotor_0", "pos": [0.25, 0.15, 0.0], "dir": [c45, c45, 0.0]},
                    {"name": "rotor_1", "pos": [0.25, -0.15, 0.0], "dir": [c45, -c45, 0.0]},
                    {"name": "rotor_2", "pos": [-0.25, 0.15, 0.0], "dir": [-c45, c45, 0.0]},
                    {"name": "rotor_3", "pos": [-0.25, -0.15, 0.0], "dir": [-c45, -c45, 0.0]},
                    {"name": "rotor_4", "pos": [0.20, 0.15, 0.0], "dir": [0.0, 0.0, 1.0]},
                    {"name": "rotor_5", "pos": [0.20, -0.15, 0.0], "dir": [0.0, 0.0, 1.0]},
                    {"name": "rotor_6", "pos": [-0.20, 0.15, 0.0], "dir": [0.0, 0.0, 1.0]},
                    {"name": "rotor_7", "pos": [-0.20, -0.15, 0.0], "dir": [0.0, 0.0, 1.0]},
                ]
            for frame in frames:
                name = str(frame.get("name", "rotor"))
                pos = frame.get("pos", [0.0, 0.0, 0.0])
                direction = frame.get("dir", [1.0, 0.0, 0.0])
                quat_wxyz = _quat_from_two_vectors((1.0, 0.0, 0.0), direction)
                prim_utils.create_prim(
                    f"{body_path}/{name}",
                    "Xform",
                    translation=tuple(float(x) for x in pos),
                    orientation=tuple(float(x) for x in quat_wxyz),
                )

            # IMPORTANT: keep the referenced visual assembly purely visual.
            # If the referenced USD contains rigid bodies/colliders, they would become independent
            # physics actors (parenting does not weld bodies), causing parts like rotors to "fly off".
            kit_utils.set_nested_rigid_body_properties(
                f"{body_path}/visual",
                rigid_body_enabled=False,
                disable_gravity=True,
            )
            kit_utils.set_nested_collision_properties(
                f"{body_path}/visual",
                collision_enabled=False,
            )
        except Exception as e:
            print(f"[BlueROVHeavy] failed to create wrapped rigid body asset: {e}")
            # Fallback to the default behavior (may not be controllable).
            return super()._create_prim(prim_path, translation, orientation)

        return root

    def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False):
        super().initialize(prim_paths_expr=prim_paths_expr, track_contact_forces=track_contact_forces)
        if not bool(self.params.get("visual_keep_only_default_and_rotors", False)):
            return
        try:
            num_envs = int(self.shape[0])
            num_agents = int(self.shape[1])
            for env_i in range(num_envs):
                for agent_i in range(num_agents):
                    if bool(getattr(self, "is_articulation", False)):
                        self._prune_visual_subtree(
                            f"/World/envs/env_{env_i}/{self.name}_{agent_i}"
                        )
                    else:
                        self._prune_visual_subtree(
                            f"/World/envs/env_{env_i}/{self.name}_{agent_i}/body/visual"
                        )
        except Exception:
            return
