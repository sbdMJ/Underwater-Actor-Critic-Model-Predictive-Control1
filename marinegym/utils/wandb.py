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


import datetime
import logging
import os
import re

import wandb
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig


def dict_flatten(a: dict, delim="."):
    """Flatten a dict recursively.
    Examples:
        >>> a = {
                "a": 1,
                "b":{
                    "c": 3,
                    "d": 4,
                    "e": {
                        "f": 5
                    }
                }
            }
        >>> dict_flatten(a)
        {'a': 1, 'b.c': 3, 'b.d': 4, 'b.e.f': 5}
    """
    result = {}
    for k, v in a.items():
        if isinstance(v, dict):
            result.update({k + delim + kk: vv for kk, vv in dict_flatten(v).items()})
        else:
            result[k] = v
    return result


def _sanitize_run_name(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    return re.sub(r"[^0-9a-zA-Z_.=,+-]+", "_", value)


def _collect_override_tags(cfg) -> list[str]:
    if not HydraConfig.initialized():
        return []
    overrides = list(HydraConfig.get().overrides.task)
    if not overrides:
        return []
    name_keys = getattr(cfg.wandb, "name_keys", None)
    if name_keys:
        key_set = {str(k) for k in name_keys}
        filtered = []
        for item in overrides:
            key = item.split("=", 1)[0].split("+", 1)[-1]
            if key in key_set:
                filtered.append(item)
        overrides = filtered
    return [_sanitize_run_name(item) for item in overrides if item]


def init_wandb(cfg):
    """Initialize WandB.

    If only `run_id` is given, resume from the run specified by `run_id`.
    If only `run_path` is given, start a new run from that specified by `run_path`,
        possibly restoring trained models.

    Otherwise, start a fresh new run.

    """
    wandb_cfg = cfg.wandb
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    run_prefix = str(getattr(wandb_cfg, "run_prefix", "") or "").strip()
    base_name = str(wandb_cfg.run_name)
    if run_prefix:
        base_name = f"{run_prefix}-{base_name}"
    override_tags = _collect_override_tags(cfg)
    if override_tags:
        base_name = f"{base_name}__{'__'.join(override_tags)}"
    run_name = f"{base_name}/{time_str}"
    kwargs = dict(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        name=run_name,
        mode=wandb_cfg.mode,
        tags=wandb_cfg.tags,
    )
    entity = getattr(wandb_cfg, "entity", None)
    if entity is not None:
        entity_str = str(entity).strip()
        if entity_str and entity_str != "your_wandb_entity_here":
            kwargs["entity"] = entity_str
    if wandb_cfg.run_id is not None:
        kwargs["id"] = wandb_cfg.run_id
        kwargs["resume"] = "must"
    else:
        kwargs["id"] = wandb.util.generate_id()
    run = wandb.init(**kwargs)
    cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
    run.config.update(cfg_dict)
    return run
