import sys


if sys.version_info < (3, 9):
    # NOTE (arth): import first to avoid importing after torch:
    import isaacgym

import os
from pathlib import Path
import torch
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import argparse
import statistics
import yaml

import pql
from pql.utils.common import set_random_seed
from pql.utils.isaacgym_util import create_task_env 
from pql.utils.common import capture_keyboard_interrupt
from pql.utils.common import load_class_from_path
from pql.models import model_name_to_path
from pql.utils.torch_util import RunningMeanStd

def default_rollout(env, cfg, actor, normalizer):
    with torch.inference_mode():
        num_envs = cfg.eval_num_envs
        max_step = env.max_episode_length

        eval_rewbuffer = []
        eval_lenbuffer = []
        eval_soncebuffer = []
        eval_sendbuffer = []
        eval_cur_reward_sum = torch.zeros(
            num_envs, dtype=torch.float, device=cfg.device
        )
        eval_cur_episode_length = torch.zeros(
            num_envs, dtype=torch.float, device=cfg.device
        )
        eval_cur_sonce = torch.zeros(
            num_envs, dtype=torch.bool, device=cfg.device
        )
        eval_cur_send = torch.zeros(
            num_envs, dtype=torch.bool, device=cfg.device
        )
        obs = env.reset()
        for _ in range(max_step):  # run an episode
            if cfg.algo.obs_norm:
                action = actor(normalizer.normalize(obs))
            else:
                action = actor(obs)
            next_obs, reward, done, info = env.step(action)
            reward = reward.to(torch.float)
            done = done.to(torch.bool)
            eval_cur_reward_sum += reward
            eval_cur_episode_length += 1

            if "success" in info:
                success = info["success"].to(torch.bool)
                eval_cur_sonce |= success
                eval_cur_send[:] = success

            if torch.any(done):
                eval_rewbuffer.extend(
                    eval_cur_reward_sum[done].tolist()
                )
                eval_lenbuffer.extend(
                    eval_cur_episode_length[done].tolist()
                )
                eval_cur_reward_sum[done] = 0
                eval_cur_episode_length[done] = 0
                if "success" in info:
                    eval_soncebuffer.extend(
                        eval_cur_sonce[done].tolist()
                    )
                    eval_sendbuffer.extend(
                        eval_cur_send[done].tolist()
                    )
                    eval_cur_sonce[done] = 0
                    eval_cur_send[done] = 0

            obs = next_obs

    return {
        "eval/ep_return": statistics.mean(eval_rewbuffer),
        "eval/len": statistics.mean(eval_lenbuffer),
        "eval/success_once": statistics.mean(eval_soncebuffer),
        "eval/success_at_end": statistics.mean(eval_sendbuffer),
    }


def main(cfg: DictConfig, ckpt_dir):
    ckpt_dir = Path(ckpt_dir)

    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    env = create_task_env(cfg, num_envs=cfg.eval_num_envs, eval=True)
    device = torch.device(cfg.device)
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    act_class = load_class_from_path(cfg.algo.act_class,
                                            model_name_to_path[cfg.algo.act_class])
    actor = act_class(obs_dim, action_dim).to(device)
    obs_rms = RunningMeanStd(shape=obs_dim, device=device)

    running_eval_logs = {
        "time/rollout+update_time": [],
        "eval/ep_return": [],
        "eval/len": [],
        "eval/success_once": [],
        "eval/success_at_end": [],
    }
    eval_log_path = ckpt_dir / "eval_log.yml"
    file_names = sorted([x for x in os.listdir(ckpt_dir) if x.endswith(".pt")], key=lambda x: float(x[:-3]))
    for ckpt_file in file_names:
        ckpt_path = ckpt_dir / ckpt_file
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        actor.load_state_dict(ckpt["actor"])
        obs_rms.load_state_dict(ckpt["obs_rms"])
        eval_logs = default_rollout(env, cfg, actor, obs_rms)

        running_eval_logs["time/rollout+update_time"].append(float(ckpt_path.stem))
        running_eval_logs["eval/ep_return"].append(eval_logs["eval/ep_return"])
        running_eval_logs["eval/len"].append(eval_logs["eval/len"])
        running_eval_logs["eval/success_once"].append(eval_logs["eval/success_once"])
        running_eval_logs["eval/success_at_end"].append(eval_logs["eval/success_at_end"])

        # Write evaluation logs to yaml file
        with open(eval_log_path, "w") as f:
            yaml.dump(running_eval_logs, f)
        print(f"{float(ckpt_path.stem):.4f}\t{eval_logs['eval/ep_return']:.4f}\t{eval_logs['eval/len']:.4f}\t{eval_logs['eval/success_once']:.4f}\t{eval_logs['eval/success_at_end']:.4f}", sep="\t")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=pql.LIB_PATH.joinpath('cfg/pql.yaml').as_posix(),
                        help='Path to the config file')
    parser.add_argument('--ckpt_dir', type=str, help='Path to ckpt dir')
    args = parser.parse_args()
    
    # Load the config file
    cfg = OmegaConf.load(args.config_path)
    main(cfg, args.ckpt_dir)
