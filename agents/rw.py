from typing import Dict

import gymnasium as gym
import numpy as np

from agents.agent import Agent
from common.types import Observation, Action, Reward
from utils.spaces_helpers import get_1D_box_observation_space_dim


class RW(Agent):
    """
    Rescorla-Wagner agent
    """

    def __init__(self, env: gym.Env, learning_rate: float = 0.1):
        super(RW, self).__init__()

        self.dim = get_1D_box_observation_space_dim(env)
        self.learning_rate = learning_rate
        self.action_space = env.action_space

        self.reset()

    def __str__(self):
        return f'RW_alpha={self.learning_rate:.3f}'

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
                next_observation: Observation) -> Dict[str, np.array]:
        """
        Update weights using Rescorla-Wagner rule.
        https://en.wikipedia.org/wiki/Rescorla%E2%80%93Wagner_model
        :param observation: previous observation
        :param action: previous action
        :param reward: obtained reward
        :param next_observation: following observation
        :return: info dict
        """
        value = self.weights.dot(observation)
        rpe = reward - value
        self.weights += self.learning_rate * rpe * observation
        return {'value': value, 'rpe': rpe}
