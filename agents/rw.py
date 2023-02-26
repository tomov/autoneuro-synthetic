from typing import Dict, Any

import gymnasium as gym
import numpy as np

from agent import Agent
from common.types import Observation, Action, Reward


class RW(Agent):
    """
    Rescorla-Wagner agent
    """

    def __init__(self, env: gym.Env, learning_rate: float = 0.1):
        super().__init__(self)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(f"Observation space needs to be Box, not {type(env.observation_space)}")
        if len(env.observation_space.shape) > 1:
            raise ValueError(f"Observation space needs to be 1D; shape is {env.observation_space.shape}")

        self.dim = env.observation_space.shape[0]
        self.learning_rate = learning_rate
        self.action_space = env.action_space

        self.reset()

    def reset(self):
        self.weights = np.zeros((self.dim,))

    def act(self, observation: Observation) -> Action:
        """
        Return random action
        :param observation: observation from environment
        :return: random action
        """
        return self.action_space.sample()

    def observe(self,
                observation: Observation,
                action: Action,
                reward: Reward,
                next_observation: Observation) -> Dict[str, Any]:
        """
        Process events (SARS tuple) from environment.
        :param observation: previous observation
        :param action: previous action
        :param reward: obtained reward
        :param next_observation: following observation
        :return: info dict
        """
        value = self.weights.dot(observation)
        rpe = reward - value
        self.weights += self.learning_rate * rpe * observation
        return {'value': value, 'rpe': rpe, 'weight': self.weights}
