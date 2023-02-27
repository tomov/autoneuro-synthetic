import random
from typing import Dict, Any

import gymnasium as gym
import numpy as np

from agents.agent import Agent
from common.types import Observation, Action, Reward
from utils.spaces_helpers import get_1D_box_observation_space_dim, get_discrete_action_space_n


class TD(Agent):
    """
    TD-learning agent
    """

    def __init__(self, env: gym.Env, learning_rate: float = 0.1, discount_rate: float = 0.9, eps: float = 0.1):
        super(TD, self).__init__()

        self.dim = get_1D_box_observation_space_dim(env)
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
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
        :param observation: previous observation
        :param action: previous action
        :param reward: obtained reward
        :param next_observation: following observation
        :return: info dict
        """
        value = self.weights.dot(observation)
        next_value = self.weights.dot(next_observation)
        rpe = reward + self.discount_rate * next_value - value
        self.weights[action] += self.learning_rate * rpe * observation
        return {'value': value, 'next_value': next_value, 'rpe': rpe}
