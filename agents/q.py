import random
from typing import Dict, Any

import gymnasium as gym
import numpy as np

from agents.agent import Agent
from common.types import Observation, Action, Reward
from utils.spaces_helpers import get_1D_box_observation_space_dim, get_discrete_action_space_n


class Q(Agent):
    """
    Q-learning agent
    """

    def __init__(self, env: gym.Env, learning_rate: float = 0.1, discount_rate: float = 0.9, eps: float = 0.1):
        super(Q, self).__init__()

        self.dim = get_1D_box_observation_space_dim(env)
        self.num_actions = get_discrete_action_space_n(env)
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.eps = eps
        self.action_space = env.action_space

        self.reset()

    def reset(self):
        self.weights = np.zeros((self.num_actions, self.dim))
        self.weights = np.random.random((self.num_actions, self.dim)) * 0.001

    def act(self, observation: Observation) -> Action:
        """
        Return eps-greedy action
        :param observation: observation from environment
        :return: eps-greedy action
        """
        if random.random() < self.eps:
            return self.action_space.sample()
        return self._act_greedy(observation)

    def _act_greedy(self, observation: Observation) -> Action:
        """
        Return greedy action
        :param observation: observation from environment
        :return: greedy action
        """
        values = self.weights.dot(observation)
        return np.argmax(values)

    def observe(self,
                observation: Observation,
                action: Action,
                reward: Reward,
                next_observation: Observation) -> Dict[str, Any]:
        """
        Update state-action weights using Q-learning
        :param observation: previous observation
        :param action: previous action
        :param reward: obtained reward
        :param next_observation: following observation
        :return: info dict
        """
        value = self.weights[action].dot(observation)
        next_action = self._act_greedy(next_observation)
        next_value = self.weights[next_action].dot(next_observation)
        rpe = reward + self.discount_rate * next_value - value
        self.weights[action] += self.learning_rate * rpe * observation
        return {'value': value, 'next_action': next_action, 'next_value': next_value, 'rpe': rpe}
