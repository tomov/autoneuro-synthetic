""" Helper functions for observation and action spaces """

import gymnasium as gym


def get_1D_box_observation_space_dim(env: gym.Env) -> int:
    """
    Get dimension of 1D Box observation space
    :param env: gym environment
    :return: Box dimension
    :raise: ValueError if observation space is not Box or 1D
    """
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError(f"Observation space needs to be Box, not {type(env.observation_space)}")
    if len(env.observation_space.shape) > 1:
        raise ValueError(f"Observation space needs to be 1D; shape is {env.observation_space.shape}")
    return env.observation_space.shape[0]


def get_discrete_action_space_n(env: gym.Env) -> int:
    """
    Get dimension of discrete action space
    :param env: gym environment
    :return: number of actions
    :raise: ValueError if action space is not Discrete
    """
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError(f"Action space needs to be Discrete, not {type(env.action_space)}")
    return env.action_space.n
