import abc
from typing import Dict

import numpy as np

from common.types import Observation, Action, Reward


class Agent(abc.ABC):
    """
    Abstract base class for agents
    """

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def act(self, observation: Observation) -> Action:
        """
        Perform action given observation
        :param observation: observation from environment
        :return: action
        """
        pass

    @abc.abstractmethod
    def observe(self,
                observation: Observation,
                action: Action,
                reward: Reward,
                next_observation: Observation) -> Dict[str, np.array]:
        """
        Process events (SARS tuple) from environment.
        :param observation: previous observation
        :param action: previous action
        :param reward: obtained reward
        :param next_observation: following observation
        :return: info dict
        """
        pass
