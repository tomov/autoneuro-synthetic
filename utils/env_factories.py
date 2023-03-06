""" Helper functions for environments """

import os

import numpy as np

from envs.classical import Classical


def classical_factory(dirpath: os.path) -> Classical:
    states = np.genfromtxt(os.path.join(dirpath, 'states.csv'), delimiter=',')
    rewards = np.genfromtxt(os.path.join(dirpath, 'rewards.csv'), delimiter=',')
    return Classical(states=states, rewards=rewards)
