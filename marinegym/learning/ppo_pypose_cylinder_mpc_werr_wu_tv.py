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

from marinegym.controllers.pypose_cylinder_orbit_mpc_controller import PyPoseCylinderOrbitMPCController
from marinegym.utils.torch import euler_to_quaternion, normalize, quat_rotate_inverse

from .modules.distributions import IndependentNormal
from .ppo.common import GAE
from .utils.valuenorm import ValueNorm1


@dataclass
class PPOPyposeCylinderMPCWErrWUTVConfig:
    """
    PPO(actor-critic) + PyPose differentiable Cylinder-Orbit MPC(iLQR) as the last layer of the actor.

    The actor predicts time-varying diagonal weights for:
      - w_err_seq: orbit error vector e(x) (10-dim)
      - w_u_seq: input penalty (nu-dim), interpreted as force-penalty weights (like acados)
    """

    name: str = "ppo_pypose_cylinder_mpc_werr_wu_tv"
    train_every: int = 64
    ppo_epochs: int = 5
    num_minibatches: int = 8

    priv_actor: bool = False
    priv_critic: bool = False
    checkpoint_path: Optional[str] = None

    entropy_coef: float = 0.0001
    clip_param: float = 0.1

    actor_log_std_init: float = -1.0
    actor_log_std_min: float = -3.0
    actor_log_std_max: float = 0.0

    # Observation layout hints (populated by scripts/train.py).
    obs_has_target_quat: bool = False
    obs_time_encoding_dim: int = 0
    # If true, the last 3 dims of the observation are interpreted as (cylinder_center - target_pos) in env-frame.
    obs_has_cylinder_rel: bool = False

    # PyPose MPC settings (populated by scripts/train.py / CLI).
    mpc_dtype: str = "float32"
    mpc_dt: float = 0.05
    mpc_horizon: int = 5
    mpc_nu: int = 6
    mpc_ilqr_iters: int = 2
    mpc_ilqr_reg: float = 1e-3
    terminal_weight_mult: float = 10.0
    max_thruster_force: float = 40.0

    # Orbit constants (task-space). When the actor constructs a relative state (target at origin),
    # z is interpreted in that same relative frame (typically orbit_z - cylinder_center_z).
    orbit_radius: float = 1.4
    orbit_z: float = 0.0
    orbit_v_tan: float = 1.0
    orbit_direction: float = 1.0
    orbit_yaw_offset: float = 0.0

    # Drone parameters (populated by scripts/train.py).
    mpc_param_yaml: Optional[str] = None
    mpc_alloc_npz: Optional[str] = None
    mpc_mass: Optional[float] = None
    mpc_inertia: Optional[List[float]] = None  # [xx, yy, zz]

    water_density: float = 997.0
    gravity: float = 9.81

    # Learnable diagonal cost bounds.
    werr_lb: float = 1e-3
    werr_ub: float = 1e3
    wu_lb: float = 1e-3
    wu_ub: float = 0.2
    weights_log_scale: bool = True
    cost_hidden: int = 256
    R_min_coeff: float = 0.2
    gamma_d_max: float = 5.0

    critic_worstcase_k: int = 1

    # Optional init (populated by scripts/train.py).
    werr_init: Optional[List[float]] = None  # (10,)
    wu_init: Optional[List[float]] = None  # (nu,)


cs = ConfigStore.instance()
cs.store("ppo_pypose_cylinder_mpc_werr_wu_tv", node=PPOPyposeCylinderMPCWErrWUTVConfig, group="algo")
cs.store(
    "ppo_pypose_cylinder_mpc_werr_wu_tv_priv",
    node=PPOPyposeCylinderMPCWErrWUTVConfig(priv_actor=True, priv_critic=True),
    group="algo",
)
cs.store(
    "ppo_pypose_cylinder_mpc_werr_wu_tv_priv_critic",
    node=PPOPyposeCylinderMPCWErrWUTVConfig(priv_critic=True),
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


def _map_raw_to_positive(raw: torch.Tensor, lb: float, ub: float, *, log_scale: bool) -> torch.Tensor:
    if ub <= lb:
        raise ValueError(f"invalid bounds: lb={lb} ub={ub}")
    s = torch.sigmoid(raw)
    if log_scale:
        log_lb = math.log(lb)
        log_ub = math.log(ub)
        return torch.exp(log_lb + (log_ub - log_lb) * s)
    return lb + (ub - lb) * s


def _parse_torch_dtype(dtype_str: str) -> torch.dtype:
    s = str(dtype_str).strip().lower()
    if s.startswith("torch."):
        s = s[len("torch.") :]

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if s not in dtype_map:
        raise ValueError(f"Unsupported cfg.mpc_dtype={dtype_str!r}. Use one of: {sorted(dtype_map.keys())}")
    return dtype_map[s]


def _load_uav_params(cfg: PPOPyposeCylinderMPCWErrWUTVConfig) -> dict:
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


class _NeuralDiagCostMapOrbitErrHorizon(nn.Module):
    def __init__(self, cfg: PPOPyposeCylinderMPCWErrWUTVConfig, *, horizon: int, ne: int, nu: int):
        super().__init__()
        self.cfg = cfg
        self.horizon = int(horizon)
        self.ne = int(ne)
        self.nu = int(nu)
        out_dim = self.horizon * (self.ne + self.nu)

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

        werr0 = torch.as_tensor(cfg.werr_init, dtype=torch.float32) if cfg.werr_init is not None else None
        wu0 = torch.as_tensor(cfg.wu_init, dtype=torch.float32) if cfg.wu_init is not None else None
        if werr0 is None:
            werr0 = torch.full((self.ne,), 1.0, dtype=torch.float32)
        if wu0 is None:
            wu0 = torch.full((self.nu,), 0.02, dtype=torch.float32)
        if int(werr0.numel()) != self.ne:
            raise ValueError(f"werr_init must have length {self.ne}. got {int(werr0.numel())}")
        if int(wu0.numel()) != self.nu:
            raise ValueError(f"wu_init must have length {self.nu}. got {int(wu0.numel())}")

        werr0 = werr0.clamp(min=cfg.werr_lb, max=cfg.werr_ub)
        wu0 = wu0.clamp(min=cfg.wu_lb, max=cfg.wu_ub)

        if cfg.weights_log_scale:
            se0 = (torch.log(werr0) - math.log(cfg.werr_lb)) / (math.log(cfg.werr_ub) - math.log(cfg.werr_lb))
            su0 = (torch.log(wu0) - math.log(cfg.wu_lb)) / (math.log(cfg.wu_ub) - math.log(cfg.wu_lb))
        else:
            se0 = (werr0 - cfg.werr_lb) / (cfg.werr_ub - cfg.werr_lb)
            su0 = (wu0 - cfg.wu_lb) / (cfg.wu_ub - cfg.wu_lb)

        raw_e0 = _inv_sigmoid(se0)
        raw_u0 = _inv_sigmoid(su0)
        raw0 = torch.cat([raw_e0, raw_u0], dim=-1)  # (ne+nu,)
        raw0 = raw0.repeat(self.horizon)  # (horizon*(ne+nu),)
        with torch.no_grad():
            self.head.bias.copy_(raw0)

    def forward(self, obs_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(obs_flat)
        raw = self.head(feat).view(obs_flat.shape[0], self.horizon, self.ne + self.nu)
        raw_e = raw[..., : self.ne]
        raw_u = raw[..., self.ne :]

        w_err = _map_raw_to_positive(
            raw_e,
            float(self.cfg.werr_lb),
            float(self.cfg.werr_ub),
            log_scale=bool(self.cfg.weights_log_scale),
        )
        w_u = _map_raw_to_positive(
            raw_u,
            float(self.cfg.wu_lb),
            float(self.cfg.wu_ub),
            log_scale=bool(self.cfg.weights_log_scale),
        )
        return w_err, w_u


class _MPCMeanActorOrbitErrTV(nn.Module):
    def __init__(self, cfg: PPOPyposeCylinderMPCWErrWUTVConfig, *, nu: int, device):
        super().__init__()
        self.cfg = cfg
        self.nu = int(nu)
        self.ne = 10
        self.device = device

        uav_params = _load_uav_params(cfg)
        mpc_dtype = _parse_torch_dtype(cfg.mpc_dtype)
        self.mpc = PyPoseCylinderOrbitMPCController(
            uav_params=uav_params,
            mpc_dtype=mpc_dtype,
            dt=float(cfg.mpc_dt),
            horizon=int(cfg.mpc_horizon),
            batch_size=1,  # compute adapts on first call
            ilqr_iters=int(cfg.mpc_ilqr_iters),
            ilqr_reg=float(cfg.mpc_ilqr_reg),
            terminal_weight_mult=float(cfg.terminal_weight_mult),
            max_thruster_force=float(cfg.max_thruster_force),
            # These are used only as defaults when w_err_seq/w_u_seq are not provided.
            q_radial=1.0,
            q_z=1.0,
            q_tan=1.0,
            q_radial_speed=1.0,
            q_heading=1.0,
            q_roll=1.0,
            q_pitch=1.0,
            q_wxy=1.0,
            r_u=1.0,
        ).to(device)
        self.cost_map = _NeuralDiagCostMapOrbitErrHorizon(cfg, horizon=int(cfg.mpc_horizon), ne=self.ne, nu=self.nu).to(device)

        hidden = int(cfg.cost_hidden)
        self.actor_trunk = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.actor_head = nn.Linear(hidden, 2)
        # actor outputs:
        #   actor_out[..., 0] -> raw R
        #   actor_out[..., 1] -> raw disturbance gain gamma_d
        self.actor_log_std = nn.Parameter(torch.full((self.nu,), float(cfg.actor_log_std_init)))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.squeeze(-2)
        batch_shape = obs.shape[:-1]
        obs_flat = obs.reshape(-1, obs.shape[-1])
        obs_flat = torch.nan_to_num(obs_flat)

        rpos = obs_flat[:, 0:3]
        quat = normalize(obs_flat[:, 3:7])
        vel_w = obs_flat[:, 7:13]
        v_b = quat_rotate_inverse(quat, vel_w[:, 0:3])
        w_b = quat_rotate_inverse(quat, vel_w[:, 3:6])
        vel_b = torch.cat([v_b, w_b], dim=-1)

        pos_rel = -rpos  # drone_pos - target_pos (target at origin)
        root_state = torch.cat([pos_rel, quat, vel_b], dim=-1)  # (B, 13)

        actor_feat = self.actor_trunk(obs_flat)
        actor_out = self.actor_head(actor_feat)
        r_raw = actor_out[:, 0]
        gamma_d_raw = actor_out[:, 1]

        # IMPORTANT: w_u_seq is interpreted as a "force penalty" weight (like acados r_u) and will be
        # multiplied by max_thruster_force^2 inside the MPC. Therefore, any additive term here must
        # be on the same order as the task's r_u (~1e-3), otherwise the MPC will output near-zero
        # actions and the vehicle will fail to orbit at initialization.
        ru_ref = None
        try:
            if getattr(self.cfg, "wu_init", None):
                ru_ref = float(np.mean(self.cfg.wu_init))
        except Exception:
            ru_ref = None
        if ru_ref is None:
            ru_ref = float(getattr(self.cfg, "wu_lb", 1e-3))
        ru_ref = max(ru_ref, float(getattr(self.cfg, "wu_lb", 1e-3)))

        # Clamp raw output to avoid exp overflow and overly stiff input penalties.
        r_raw = r_raw.clamp(min=-5.0, max=5.0)
        R_base = ru_ref * torch.exp(r_raw)
        dist_xy = torch.linalg.norm(rpos[:, :2], dim=-1)
        orbit_err = torch.abs(dist_xy - float(self.cfg.orbit_radius))
        R_min = ru_ref * float(self.cfg.R_min_coeff) * orbit_err
        R = R_base + R_min
        gamma_d = torch.sigmoid(gamma_d_raw) * float(self.cfg.gamma_d_max)

        w_err_seq, w_u_seq = self.cost_map(obs_flat)
        w_u_seq = w_u_seq + R.view(-1, 1, 1)

        if bool(getattr(self.cfg, "obs_has_cylinder_rel", False)):
            center = obs_flat[:, -3:].to(dtype=root_state.dtype)  # cylinder_center - target_pos
        else:
            center = torch.zeros_like(pos_rel)  # cylinder_center == target_pos
        try:
            u0 = self.mpc.compute(
                root_state,
                center_w=center,
                radius=float(self.cfg.orbit_radius),
                z=float(self.cfg.orbit_z),
                v_tan=float(self.cfg.orbit_v_tan),
                direction=float(self.cfg.orbit_direction),
                yaw_offset=float(self.cfg.orbit_yaw_offset),
                w_err_seq=w_err_seq,
                w_u_seq=w_u_seq,
                gamma_d=gamma_d,
            )
        except Exception:
            u0 = torch.zeros((root_state.shape[0], self.nu), device=root_state.device, dtype=root_state.dtype)
        u0 = torch.nan_to_num(u0).clamp(-1.0, 1.0)

        loc = u0.view(*batch_shape, 1, self.nu).to(dtype=obs.dtype)

        log_std = self.actor_log_std.clamp(
            min=float(self.cfg.actor_log_std_min),
            max=float(self.cfg.actor_log_std_max),
        )
        scale = torch.exp(log_std).view(*(1,) * (loc.dim() - 1), -1).expand_as(loc).to(dtype=obs.dtype)
        return loc, scale


class PPOPyposeCylinderMPCWErrWUTVPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: PPOPyposeCylinderMPCWErrWUTVConfig,
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

        mpc_mean = _MPCMeanActorOrbitErrTV(cfg, nu=self.action_dim, device=device)

        if self.cfg.priv_actor:
            raise NotImplementedError(
                "priv_actor is disabled for ppo_pypose_cylinder_mpc_werr_wu_tv to keep the actor learnables limited to the cost map and log_std."
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

        k = int(getattr(self.cfg, "critic_worstcase_k", 1))
        if k > 1:
            returns_flat = b_returns.view(-1)
            values_flat = values.view(-1)
            b_values_flat = b_values.view(-1)
            B = returns_flat.shape[0] // k
            if B > 0 and returns_flat.shape[0] % k == 0:
                returns_reshaped = returns_flat.view(B, k)
                worst_returns, _ = returns_reshaped.min(dim=1)
                values = values_flat.view(B, k).mean(dim=1)
                b_values = b_values_flat.view(B, k).mean(dim=1)
                b_returns = worst_returns
            else:
                b_returns = b_returns.view(-1)
                values = values_flat
                b_values = b_values_flat

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
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
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
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
