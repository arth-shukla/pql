from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from omegaconf import OmegaConf

from pql.wrappers.flatten_ob import FlatObEnvWrapper
from pql.wrappers.reset import ResetEnvWrapper


def create_maniskill_task_env(cfg, num_envs=None):
    import mani_skill.envs
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    @dataclass
    class ResetEnvWrapper:
        env: ManiSkillVectorEnv

        def __post_init__(self):
            self.observation_space = self.env.unwrapped.single_observation_space
            self.action_space = self.env.unwrapped.single_action_space
            self.max_episode_length = self.env.unwrapped.max_episode_steps

        def reset(self):
            return self.env.reset()[0]

        def step(self, actions):
            obs, rew, term, trunc, info = self.env.step(actions)
            done = term | trunc
            return obs, rew, done, info

    task_cfg = OmegaConf.to_container(cfg.task, resolve=True, throw_on_missing=True)

    # for now just support exactly 1 GPU
    assert "cuda" in cfg.rl_device and "cuda" in cfg.sim_device
    assert cfg.graphics_device_id == 0

    if not cfg.headless:
        task_cfg["render_mode"] = "human"
    else:
        task_cfg["render_mode"] = "rgb_array"

    env = gym.make(
        task_cfg["name"],
        num_envs=num_envs if num_envs is not None else task_cfg["num_envs"],
        max_episode_steps=task_cfg["max_episode_steps"],
        obs_mode="state",
        reward_mode="normalized_dense",
        control_mode=task_cfg.get("control_mode", "pd_joint_delta_pos"),
        render_mode="rgb_array" if cfg.headless else "human",
        shader_dir="minimal",
        sim_backend="gpu",
        sim_config=dict(
            sim_freq=100, control_freq=20, **task_cfg.get("sim_config", dict())
        ),
    )
    env = ManiSkillVectorEnv(
        env,
        ignore_terminations=task_cfg.get("continuous_task", True),
        staggered_reset=task_cfg.get("staggered_reset", False),
    )
    return ResetEnvWrapper(env)


def create_isaacgym_task_env(cfg, num_envs=None):
    from isaacgymenvs.tasks import isaacgym_task_map

    task_cfg = OmegaConf.to_container(cfg.task, resolve=True, throw_on_missing=True)
    if num_envs is not None:
        task_cfg["env"]["numEnvs"] = num_envs

    env = isaacgym_task_map[cfg.task.name](
        cfg=task_cfg,
        rl_device=cfg.rl_device,
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,
        force_render=not cfg.headless,
    )
    env = ResetEnvWrapper(env)
    env = FlatObEnvWrapper(env)
    return env


def create_task_env(cfg, num_envs=None):
    if cfg.task.get("suite", "maniskill") == "maniskill":
        return create_maniskill_task_env(cfg, num_envs)
    else:
        return create_isaacgym_task_env(cfg, num_envs)
