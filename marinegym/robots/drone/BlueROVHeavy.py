import torch

import omni.isaac.core.utils.prims as prim_utils
from pxr import PhysxSchema, UsdPhysics

from marinegym.robots.robot import ASSET_PATH
from marinegym.robots.drone.underwaterVehicle import UnderwaterVehicle

class BlueROVHeavy(UnderwaterVehicle):

    usd_path: str = ASSET_PATH + "/usd/BlueROVHeavy/BlueROVHeavy.usd"
    param_path: str = ASSET_PATH + "/usd/BlueROVHeavy/BlueROVHeavy.yaml"
    fixed_usd_path: str = ASSET_PATH + "/usd/BlueROVHeavy/fixed_usd/BlueROVHeavy/BlueROVHeavy.usd"

    def __init__(self, name: str = None, cfg=None, is_articulation: bool = False) -> None:
        super().__init__(name=name, cfg=cfg, is_articulation=is_articulation)

    def _create_prim(self, prim_path, translation, orientation):
        prim = prim_utils.create_prim(
            prim_path,
            usd_path=self.usd_path,
            translation=translation,
            orientation=orientation,
        )
        usd_prim = prim_utils.get_prim_at_path(prim_path)
        if not UsdPhysics.RigidBodyAPI(usd_prim):
            UsdPhysics.RigidBodyAPI.Apply(usd_prim)
        if not PhysxSchema.PhysxRigidBodyAPI(usd_prim):
            PhysxSchema.PhysxRigidBodyAPI.Apply(usd_prim)
        return prim
