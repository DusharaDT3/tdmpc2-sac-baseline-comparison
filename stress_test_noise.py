"""
Stress test (b): Gaussian observation noise robustness.

Evaluates a pretrained TD-MPC2 checkpoint with i.i.d. Gaussian noise N(0, sigma^2) added to the
(un-normalized, raw-state) observation at every env step, before the agent sees it. Mirrors the
repo's own evaluate.py eval loop exactly (same agent.act/env.step calls), just with noise injection
added between env.step() and agent.act() so the planner/policy never sees the clean observation.

Usage:
    python stress_test_noise.py --task cheetah-run --checkpoint <path> --sigmas 0 0.05 0.1 0.2 --out <csv>
"""
import argparse
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from omegaconf import OmegaConf

import sys
sys.path.insert(0, os.path.dirname(__file__) + '/../tdmpc2/tdmpc2')

import hydra.utils
hydra.utils.get_original_cwd = lambda: os.path.dirname(__file__) + '/../tdmpc2/tdmpc2'

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2


def build_cfg(task, checkpoint, model_size=5, seed=1):
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), '../tdmpc2/tdmpc2/config.yaml'))
    cfg.task = task
    cfg.checkpoint = checkpoint
    cfg.model_size = model_size
    cfg.seed = seed
    cfg.eval_episodes = 10
    cfg.save_video = False
    cfg.multitask = False
    cfg.obs = 'state'
    cfg = parse_cfg(cfg)
    return cfg


def evaluate_with_noise(cfg, sigma, n_episodes=10):
    set_seed(cfg.seed)
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    agent.load(cfg.checkpoint)

    ep_rewards = []
    for _ in range(n_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if sigma > 0:
            obs = obs + torch.randn_like(obs) * sigma
        while not done:
            action = agent.act(obs, t0=t == 0)
            obs, reward, done, info = env.step(action)
            if sigma > 0:
                obs = obs + torch.randn_like(obs) * sigma
            ep_reward += reward
            t += 1
        ep_rewards.append(ep_reward)
    return float(np.mean(ep_rewards)), float(np.std(ep_rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0, 0.05, 0.1, 0.2])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    write_header = not os.path.exists(args.out)
    with open(args.out, 'a') as f:
        if write_header:
            f.write('task,sigma,mean_return,std_return\n')
        for sigma in args.sigmas:
            cfg = build_cfg(args.task, args.checkpoint, seed=args.seed)
            mean_r, std_r = evaluate_with_noise(cfg, sigma)
            f.write(f'{args.task},{sigma},{mean_r:.2f},{std_r:.2f}\n')
            f.flush()
            print(f'DONE task={args.task} sigma={sigma} mean_return={mean_r:.2f} std={std_r:.2f}')


if __name__ == '__main__':
    main()
