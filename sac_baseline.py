"""
SAC baseline on the same 3 DMControl tasks TD-MPC2 was evaluated on, built to match
TD-MPC2's own env setup exactly (action_repeat=2, max_episode_steps=500, state obs
flattened to float32, actions rescaled to [-1, 1]) — see tdmpc2/envs/dmcontrol.py.

This is a plain gymnasium.Env wrapper around dm_control.suite rather than reusing the
repo's DMControlWrapper directly, because that wrapper returns the old 4-tuple gym API
(obs, reward, done, info) and torch tensors, while stable-baselines3 requires the
gymnasium 5-tuple API (obs, reward, terminated, truncated, info) and numpy arrays.
"""
import argparse
import time

import numpy as np
import gymnasium as gym
from dm_control import suite
from dm_control.suite.wrappers import action_scale


class DMControlGymEnv(gym.Env):
    def __init__(self, domain, task, seed=1, action_repeat=2, max_episode_steps=500):
        super().__init__()
        self._env = suite.load(domain, task, task_kwargs={'random': seed}, visualize_reward=False)
        self._env = action_scale.Wrapper(self._env, minimum=-1., maximum=1.)
        self.action_repeat = action_repeat
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        obs_spec = self._env.observation_spec()
        obs_dim = sum(int(np.prod(v.shape)) if v.shape else 1 for v in obs_spec.values())
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        act_spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            low=act_spec.minimum.astype(np.float32),
            high=act_spec.maximum.astype(np.float32),
            dtype=np.float32)

    def _flatten_obs(self, obs):
        return np.concatenate([np.asarray(v).flatten() for v in obs.values()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ts = self._env.reset()
        self._step_count = 0
        return self._flatten_obs(ts.observation), {}

    def step(self, action):
        reward = 0.0
        ts = None
        for _ in range(self.action_repeat):
            ts = self._env.step(action)
            reward += ts.reward or 0.0
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        return self._flatten_obs(ts.observation), reward, terminated, truncated, {}


TASKS = {
    'cheetah-run': ('cheetah', 'run'),
    'walker-walk': ('walker', 'walk'),
    'hopper-hop': ('hopper', 'hop'),
}


def make_env(task_name, seed=1):
    domain, task = TASKS[task_name]
    return DMControlGymEnv(domain, task, seed=seed)


def evaluate_policy(model, env, n_episodes=10):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=list(TASKS.keys()))
    parser.add_argument('--steps', type=int, default=200_000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    from stable_baselines3 import SAC

    env = make_env(args.task, seed=args.seed)
    model = SAC('MlpPolicy', env, verbose=1, seed=args.seed, device='auto')

    t0 = time.time()
    model.learn(total_timesteps=args.steps, progress_bar=False)
    train_time = time.time() - t0

    mean_return, std_return = evaluate_policy(model, env, n_episodes=10)

    with open(args.out, 'a') as f:
        f.write(f"{args.task},{args.steps},{args.seed},{mean_return:.2f},{std_return:.2f},{train_time:.1f}\n")

    print(f"DONE task={args.task} steps={args.steps} mean_return={mean_return:.2f} "
          f"std={std_return:.2f} train_time={train_time:.1f}s")


if __name__ == '__main__':
    main()
