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
class PPOPyposeMPCQRDiagConfig:
    name: str = "ppo_pypose_mpc_qrdiag"
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

    # Learnable diagonal cost bounds.
    wx_lb: float = 1e-3
    wx_ub: float = 1e2
    wu_lb: float = 1e-4
    wu_ub: float = 1.0
    weights_log_scale: bool = True
    cost_hidden: int = 256

    # Optional init (populated by scripts/train.py).
    wx_init: Optional[List[float]] = None  # (13,)
    wu_init: Optional[List[float]] = None  # (nu,)


cs = ConfigStore.instance()
cs.store("ppo_pypose_mpc_qrdiag", node=PPOPyposeMPCQRDiagConfig, group="algo")
cs.store(
    "ppo_pypose_mpc_qrdiag_priv",
    node=PPOPyposeMPCQRDiagConfig(priv_actor=True, priv_critic=True),
    group="algo",
)
cs.store(
    "ppo_pypose_mpc_qrdiag_priv_critic",
    node=PPOPyposeMPCQRDiagConfig(priv_critic=True),
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


def _load_uav_params(cfg: PPOPyposeMPCQRDiagConfig) -> dict:
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


class _NeuralDiagCostMapHorizon(nn.Module):
    def __init__(self, cfg: PPOPyposeMPCQRDiagConfig, *, horizon: int, nx: int, nu: int):
        super().__init__()
        self.cfg = cfg
        self.horizon = int(horizon)
        self.nx = int(nx)
        self.nu = int(nu)
        out_dim = self.horizon * (self.nx + self.nu)

        hidden = int(cfg.cost_hidden)
        self.trunk = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.head = nn.Linear(hidden, out_dim)

        wx0 = torch.as_tensor(cfg.wx_init, dtype=torch.float32) if cfg.wx_init is not None else None
        wu0 = torch.as_tensor(cfg.wu_init, dtype=torch.float32) if cfg.wu_init is not None else None
        if wx0 is None:
            wx0 = torch.full((self.nx,), 1.0, dtype=torch.float32)
        if wu0 is None:
            wu0 = torch.full((self.nu,), 0.02, dtype=torch.float32)
        if int(wx0.numel()) != self.nx:
            raise ValueError(f"wx_init must have length {self.nx}. got {int(wx0.numel())}")
        if int(wu0.numel()) != self.nu:
            raise ValueError(f"wu_init must have length {self.nu}. got {int(wu0.numel())}")

        wx0 = wx0.clamp(min=cfg.wx_lb, max=cfg.wx_ub)
        wu0 = wu0.clamp(min=cfg.wu_lb, max=cfg.wu_ub)

        if cfg.weights_log_scale:
            sx0 = (torch.log(wx0) - math.log(cfg.wx_lb)) / (math.log(cfg.wx_ub) - math.log(cfg.wx_lb))
            su0 = (torch.log(wu0) - math.log(cfg.wu_lb)) / (math.log(cfg.wu_ub) - math.log(cfg.wu_lb))
        else:
            sx0 = (wx0 - cfg.wx_lb) / (cfg.wx_ub - cfg.wx_lb)
            su0 = (wu0 - cfg.wu_lb) / (cfg.wu_ub - cfg.wu_lb)

        raw_x0 = _inv_sigmoid(sx0)
        raw_u0 = _inv_sigmoid(su0)
        raw0 = torch.cat([raw_x0, raw_u0], dim=-1)  # (nx+nu,)
        raw0 = raw0.repeat(self.horizon)  # (horizon*(nx+nu),)
        with torch.no_grad():
            self.head.bias.copy_(raw0)

    def forward(self, obs_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(obs_flat)
        raw = self.head(feat).view(obs_flat.shape[0], self.horizon, self.nx + self.nu)
        raw_x = raw[..., : self.nx]
        raw_u = raw[..., self.nx :]

        w_x = _map_raw_to_positive(
            raw_x,
            float(self.cfg.wx_lb),
            float(self.cfg.wx_ub),
            log_scale=bool(self.cfg.weights_log_scale),
        )
        w_u = _map_raw_to_positive(
            raw_u,
            float(self.cfg.wu_lb),
            float(self.cfg.wu_ub),
            log_scale=bool(self.cfg.weights_log_scale),
        )
        return w_x, w_u


class _MPCMeanActor(nn.Module):
    """
    Differentiable MPC policy: a = MPC(x, x_ref; diag(Q), diag(R)) + Gaussian noise.

    - Learns only diag(Q)=w_x (13,) and diag(R)=w_u (nu,) via raw parameters.
    - Keeps tracking structure (x_ref via target_pos/target_quat).
    - No linear term p.
    """

    def __init__(
        self,
        cfg: PPOPyposeMPCQRDiagConfig,
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
            batch_size=1,  # PyPoseMPCController.compute will adapt on first call
            ilqr_iters=int(cfg.mpc_ilqr_iters),
            ilqr_reg=float(cfg.mpc_ilqr_reg),
            terminal_weight_mult=float(cfg.terminal_weight_mult),
            max_thruster_force=float(cfg.max_thruster_force),
        ).to(device)
        self.cost_map = _NeuralDiagCostMapHorizon(
            cfg, horizon=int(cfg.mpc_horizon), nx=self.nx, nu=self.nu
        ).to(device)

    def _extract_target_quat(self, obs_flat: torch.Tensor) -> torch.Tensor:
        obs_dim = int(obs_flat.shape[-1])
        time_dim = int(getattr(self.cfg, "obs_time_encoding_dim", 0))
        has_target_quat = bool(getattr(self.cfg, "obs_has_target_quat", False))

        if has_target_quat:
            drone_state_dim = obs_dim - time_dim - 3 - 4
            if drone_state_dim <= 0 or (drone_state_dim + 3 + 4) > (obs_dim - time_dim):
                raise ValueError(f"invalid obs layout: obs_dim={obs_dim} time_dim={time_dim}")
            tq_start = drone_state_dim + 3  # after rheading
            tq = obs_flat[:, tq_start : tq_start + 4]
            return normalize(tq)

        # Fallback: assume target is yaw-only. Reconstruct from (drone_heading + rheading).
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
        # obs: (*batch, 1, obs_dim)
        obs = obs.squeeze(-2)
        batch_shape = obs.shape[:-1]
        obs_flat = obs.reshape(-1, obs.shape[-1])

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

        w_x_seq, w_u_seq = self.cost_map(obs_flat)
        u0 = self.mpc.compute(
            root_state,
            target_pos,
            target_quat=target_quat,
            w_x_seq=w_x_seq,
            w_u_seq=w_u_seq,
        )
        loc = u0.view(*batch_shape, 1, self.nu).to(dtype=obs.dtype)
        scale = torch.full_like(loc, float(self.cfg.actor_std))
        return loc, scale


class PPOPyposeMPCQRDiagPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: PPOPyposeMPCQRDiagConfig,
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

        mpc_mean = _MPCMeanActor(cfg, nu=self.action_dim, device=device)

        if self.cfg.priv_actor:
            raise NotImplementedError(
                "priv_actor is disabled for ppo_pypose_mpc_qrdiag to keep the actor learnables limited to diag(Q)/diag(R)."
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
