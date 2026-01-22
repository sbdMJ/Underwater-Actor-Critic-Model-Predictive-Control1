import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor

from marinegym.controllers.pypose_mpc_controller import PyPoseMPCController
from marinegym.utils.torch import euler_to_quaternion, normalize, quat_rotate_inverse

from .modules.distributions import IndependentNormal
from .ppo.common import GAE
from .utils.valuenorm import ValueNorm1


@dataclass
class PPOPyposeMPCQRConstConfig:
    name: str = "ppo_pypose_mpc_qrconst"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16

    priv_actor: bool = False
    priv_critic: bool = False
    checkpoint_path: Optional[str] = None

    entropy_coef: float = 0.001
    clip_param: float = 0.1
    actor_std: float = 0.15

    # Observation layout hints (populated by scripts/train.py).
    obs_has_target_quat: bool = False
    obs_time_encoding_dim: int = 0

    # PyPose MPC settings (populated by scripts/train.py / CLI).
    mpc_dt: float = 0.05
    mpc_horizon: int = 15
    mpc_nu: int = 6
    mpc_ilqr_iters: int = 6
    mpc_ilqr_reg: float = 1e-3
    terminal_weight_mult: float = 10.0
    max_thruster_force: float = 40.0

    # Drone parameters (populated by scripts/train.py).
    mpc_param_yaml: Optional[str] = None
    mpc_alloc_npz: Optional[str] = None
    mpc_mass: Optional[float] = None
    mpc_inertia: Optional[List[float]] = None  # [xx, yy, zz]

    water_density: float = 997.0
    gravity: float = 9.81

    # Learnable scalar cost bounds.
    q_lb: float = 1e-3
    q_ub: float = 1e2
    r_lb: float = 1e-4
    r_ub: float = 1.0
    weights_log_scale: bool = True

    # Optional init (populated by scripts/train.py).
    q_pos_init: Optional[float] = None
    q_quat_init: Optional[float] = None
    q_vel_init: Optional[float] = None
    q_omega_init: Optional[float] = None
    r_u_init: Optional[float] = None


cs = ConfigStore.instance()
cs.store("ppo_pypose_mpc_qrconst", node=PPOPyposeMPCQRConstConfig, group="algo")
cs.store(
    "ppo_pypose_mpc_qrconst_priv",
    node=PPOPyposeMPCQRConstConfig(priv_actor=True, priv_critic=True),
    group="algo",
)
cs.store(
    "ppo_pypose_mpc_qrconst_priv_critic",
    node=PPOPyposeMPCQRConstConfig(priv_critic=True),
    group="algo",
)


def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


def _inv_sigmoid(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = y.clamp(eps, 1.0 - eps)
    return torch.log(y) - torch.log1p(-y)


def _map_raw_to_positive(
    raw: torch.Tensor,
    lb: float,
    ub: float,
    *,
    log_scale: bool,
) -> torch.Tensor:
    if ub <= lb:
        raise ValueError(f"invalid bounds: lb={lb} ub={ub}")
    s = torch.sigmoid(raw)
    if log_scale:
        log_lb = math.log(lb)
        log_ub = math.log(ub)
        return torch.exp(log_lb + (log_ub - log_lb) * s)
    return lb + (ub - lb) * s


def _load_uav_params(cfg: PPOPyposeMPCQRConstConfig) -> dict:
    if cfg.mpc_param_yaml is None:
        raise ValueError("cfg.mpc_param_yaml is required (set by scripts/train.py).")
    if cfg.mpc_alloc_npz is None:
        raise ValueError("cfg.mpc_alloc_npz is required (set by scripts/train.py).")
    if cfg.mpc_mass is None:
        raise ValueError("cfg.mpc_mass is required (set by scripts/train.py).")
    if cfg.mpc_inertia is None or len(cfg.mpc_inertia) != 3:
        raise ValueError("cfg.mpc_inertia must be [xx, yy, zz] (set by scripts/train.py).")

    with open(cfg.mpc_param_yaml, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    hydro = params.get("hydro_coef", None)
    if hydro is None:
        raise ValueError(f"hydro_coef not found in {cfg.mpc_param_yaml}")

    alloc = np.load(cfg.mpc_alloc_npz)
    if "B" not in alloc:
        raise ValueError(f"'B' not found in {cfg.mpc_alloc_npz}")
    B = np.asarray(alloc["B"], dtype=np.float64)

    inertia_xx, inertia_yy, inertia_zz = [float(x) for x in cfg.mpc_inertia]
    return {
        "name": params.get("name", "uav"),
        "mass": float(cfg.mpc_mass),
        "inertia": {"xx": inertia_xx, "yy": inertia_yy, "zz": inertia_zz},
        "hydro_coef": hydro,
        "thruster_allocation": B,
        "volume": float(params.get("volume", 0.0)),
        "coBM": float(params.get("coBM", 0.0)),
        "rho": float(cfg.water_density),
        "g": float(cfg.gravity),
    }


class _ConstScalarQR(nn.Module):
    def __init__(self, cfg: PPOPyposeMPCQRConstConfig):
        super().__init__()
        self.cfg = cfg

        q_pos0 = float(cfg.q_pos_init) if cfg.q_pos_init is not None else 50.0
        q_quat0 = float(cfg.q_quat_init) if cfg.q_quat_init is not None else 5.0
        q_vel0 = float(cfg.q_vel_init) if cfg.q_vel_init is not None else 2.0
        q_omega0 = float(cfg.q_omega_init) if cfg.q_omega_init is not None else 0.5
        r_u0 = float(cfg.r_u_init) if cfg.r_u_init is not None else 0.02

        q0 = torch.tensor([q_pos0, q_quat0, q_vel0, q_omega0], dtype=torch.float32)
        r0 = torch.tensor([r_u0], dtype=torch.float32)

        q0 = q0.clamp(min=float(cfg.q_lb), max=float(cfg.q_ub))
        r0 = r0.clamp(min=float(cfg.r_lb), max=float(cfg.r_ub))

        if cfg.weights_log_scale:
            sq0 = (torch.log(q0) - math.log(cfg.q_lb)) / (math.log(cfg.q_ub) - math.log(cfg.q_lb))
            sr0 = (torch.log(r0) - math.log(cfg.r_lb)) / (math.log(cfg.r_ub) - math.log(cfg.r_lb))
        else:
            sq0 = (q0 - cfg.q_lb) / (cfg.q_ub - cfg.q_lb)
            sr0 = (r0 - cfg.r_lb) / (cfg.r_ub - cfg.r_lb)

        self.raw_q = nn.Parameter(_inv_sigmoid(sq0))
        self.raw_r = nn.Parameter(_inv_sigmoid(sr0))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = _map_raw_to_positive(
            self.raw_q,
            float(self.cfg.q_lb),
            float(self.cfg.q_ub),
            log_scale=bool(self.cfg.weights_log_scale),
        )
        r = _map_raw_to_positive(
            self.raw_r,
            float(self.cfg.r_lb),
            float(self.cfg.r_ub),
            log_scale=bool(self.cfg.weights_log_scale),
        )
        return q[0], q[1], q[2], q[3], r[0]


class _MPCMeanActorConstQR(nn.Module):
    """
    Differentiable MPC policy: a = MPC(x, x_ref; q_pos/q_quat/q_vel/q_omega/r_u) + Gaussian noise.

    Unlike `ppo_pypose_mpc_qrdiag`, this learns only the 5 scalar weights used in
    `cfg/task/Hover_PyPose_MPC.yaml` and shares them across all envs/timesteps.
    """

    def __init__(
        self,
        cfg: PPOPyposeMPCQRConstConfig,
        *,
        nu: int,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.nu = int(nu)
        self.nx = 13
        self.device = device

        uav_params = _load_uav_params(cfg)
        self.mpc = PyPoseMPCController(
            uav_params=uav_params,
            dt=float(cfg.mpc_dt),
            horizon=int(cfg.mpc_horizon),
            batch_size=1,  # will adapt on first call
            ilqr_iters=int(cfg.mpc_ilqr_iters),
            ilqr_reg=float(cfg.mpc_ilqr_reg),
            terminal_weight_mult=float(cfg.terminal_weight_mult),
            max_thruster_force=float(cfg.max_thruster_force),
        ).to(device)
        self.qr = _ConstScalarQR(cfg).to(device)

    def _extract_target_quat(self, obs_flat: torch.Tensor) -> torch.Tensor:
        obs_dim = int(obs_flat.shape[-1])
        time_dim = int(getattr(self.cfg, "obs_time_encoding_dim", 0))
        has_target_quat = bool(getattr(self.cfg, "obs_has_target_quat", False))

        if has_target_quat:
            drone_state_dim = obs_dim - time_dim - 3 - 4
            if drone_state_dim <= 0 or (drone_state_dim + 3 + 4) > (obs_dim - time_dim):
                raise ValueError(f"invalid obs layout: obs_dim={obs_dim} time_dim={time_dim}")
            tq_start = drone_state_dim + 3
            tq = obs_flat[:, tq_start : tq_start + 4]
            return normalize(tq)

        drone_state_dim = obs_dim - time_dim - 3
        if drone_state_dim <= 0 or (drone_state_dim + 3) > (obs_dim - time_dim):
            raise ValueError(f"invalid obs layout: obs_dim={obs_dim} time_dim={time_dim}")
        rheading = obs_flat[:, drone_state_dim : drone_state_dim + 3]
        drone_heading = obs_flat[:, 13:16]  # after rpos(3), quat(4), vel_w(6)
        target_heading = drone_heading + rheading
        yaw = torch.atan2(target_heading[:, 1], target_heading[:, 0])
        rpy = torch.zeros((obs_flat.shape[0], 3), device=obs_flat.device, dtype=obs_flat.dtype)
        rpy[:, 2] = yaw
        return euler_to_quaternion(rpy)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.squeeze(-2)
        batch_shape = obs.shape[:-1]
        obs_flat = obs.reshape(-1, obs.shape[-1])
        if not torch.isfinite(obs_flat).all():
            obs_flat = torch.nan_to_num(obs_flat)

        rpos = obs_flat[:, 0:3]
        quat = normalize(obs_flat[:, 3:7])
        vel_w = obs_flat[:, 7:13]
        v_b = quat_rotate_inverse(quat, vel_w[:, 0:3])
        w_b = quat_rotate_inverse(quat, vel_w[:, 3:6])
        vel_b = torch.cat([v_b, w_b], dim=-1)

        pos_rel = -rpos  # drone_pos - target_pos (target at origin)
        root_state = torch.cat([pos_rel, quat, vel_b], dim=-1)  # (B, 13)
        target_pos = torch.zeros_like(pos_rel)  # (B, 3)
        target_quat = self._extract_target_quat(obs_flat)

        q_pos, q_quat, q_vel, q_omega, r_u = self.qr()
        w_x = torch.cat(
            [
                q_pos.repeat(3),
                q_quat.repeat(4),
                q_vel.repeat(3),
                q_omega.repeat(3),
            ],
            dim=-1,
        ).to(device=obs_flat.device, dtype=torch.float64)
        w_u = r_u.repeat(self.nu).to(device=obs_flat.device, dtype=torch.float64)

        try:
            u0 = self.mpc.compute(
                root_state,
                target_pos,
                target_quat=target_quat,
                w_x=w_x,
                w_u=w_u,
            )
        except Exception:
            u0 = torch.zeros((root_state.shape[0], self.nu), device=root_state.device, dtype=root_state.dtype)
        if not torch.isfinite(u0).all():
            u0 = torch.nan_to_num(u0).clamp(-1.0, 1.0)
        loc = u0.view(*batch_shape, 1, self.nu).to(dtype=obs.dtype)
        scale = torch.full_like(loc, float(self.cfg.actor_std))
        return loc, scale


class PPOPyposeMPCQRConstPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: PPOPyposeMPCQRConstConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = float(cfg.entropy_coef)
        self.clip_param = float(cfg.clip_param)
        self.critic_loss_fn = nn.HuberLoss(delta=10)

        self.n_agents, self.action_dim = action_spec.shape[-2:]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        if self.n_agents != 1:
            raise ValueError(f"Expected n_agents=1. got {self.n_agents}")

        mpc_mean = _MPCMeanActorConstQR(cfg, nu=self.action_dim, device=device)

        if self.cfg.priv_actor:
            raise NotImplementedError(
                "priv_actor is disabled for ppo_pypose_mpc_qrconst to keep the actor learnables limited to scalar Q/R."
            )

        actor_module = TensorDictModule(
            mpc_mean,
            in_keys=[("agents", "observation")],
            out_keys=["loc", "scale"],
        )

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        if self.cfg.priv_critic:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            self.critic = nn.ModuleDict(
                {
                    "obs_net": TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                    "ctx_net": TensorDictModule(
                        nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                        [("agents", "intrinsics")],
                        ["context"],
                    ),
                    "cat": CatTensors(["feature", "context"], "feature"),
                    "head": TensorDictModule(nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)), ["feature"], ["state_value"]),
                }
            )

            class _CriticWrapper(TensorDictModuleBase):
                def __init__(self, mod: nn.ModuleDict):
                    super().__init__()
                    self.mod = mod

                def forward(self, tensordict: TensorDict) -> TensorDict:
                    self.mod["obs_net"](tensordict)
                    self.mod["ctx_net"](tensordict)
                    self.mod["cat"](tensordict)
                    self.mod["head"](tensordict)
                    return tensordict

            self.critic = _CriticWrapper(self.critic).to(self.device)
        else:
            self.critic = TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)),
                [("agents", "observation")],
                ["state_value"],
            ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.value_norm = ValueNorm1(reward_spec.shape[-2:]).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "feature", "context", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        dones = tensordict[("next", "terminated")].unsqueeze(-1)
        values = tensordict["state_value"]

        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for _epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        policy_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = -self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(-self.clip_param, self.clip_param)
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        loss = policy_loss + entropy_loss + value_loss
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()

        explained_var = 1.0 - F.mse_loss(values, b_returns) / b_returns.var().clamp_min(1e-8)
        return TensorDict(
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "actor_grad_norm": actor_grad_norm,
                "critic_grad_norm": critic_grad_norm,
                "explained_var": explained_var,
            },
            [],
        )


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    usable = (tensordict.shape[0] // num_minibatches) * num_minibatches
    perm = torch.randperm(usable, device=tensordict.device).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
