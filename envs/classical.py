from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

from common.types import Observation, Action, Reward


class Classical(gym.Env):
    """ Static classical conditioning environment based on pre-generated states and rewards """

    def __init__(self, states: np.array, rewards: np.array):
        super(Classical, self).__init__()

        self.states = states
        self.rewards = rewards

        self.dim = states.shape[1]
        self.observation_space = gym.spaces.Box(np.min(states), np.max(states), shape=(self.dim,), dtype=float)
        self.action_space = gym.spaces.Discrete(1)

        if len(states) != len(rewards):
            raise ValueError(f'Different numbers of states and rewards ({len(states)} != {len(rewards)})')

    def _get_obs(self) -> Observation:
        return

    def reset(self, seed=None, options=None) -> Tuple[Observation, Dict]:
        super().reset(seed=seed)

        self._state_index = 0

        # First observation is irrelevant in classical conditioning environments because it is not accompanied by reward
        return np.zeros((self.dim,)), {}

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, bool, Dict]:
        state = self.states[self._state_index]
        reward = self.rewards[self._state_index]
        terminated = self._state_index + 1 == len(self.states)

        self._state_index += 1
        return state, reward, terminated, False, {}
