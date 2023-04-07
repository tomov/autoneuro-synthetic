""" Generate synthetic data for instrumental conditioning """
import json
import os
import typing
from typing import Dict, List, Tuple, Any

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from agents.agent import Agent
from agents.q import Q
from envs.classical import Classical


def save_data(dirpath: os.path, agent_inputs: List[Dict[str, np.array]], agent_outputs: List[Dict[str, np.array]],
              agent_kwargs: Dict[str, Any]) -> None:
    os.makedirs(dirpath, exist_ok=True)

    # save metadata
    with open(os.path.join(dirpath, 'agent_kwargs.json'), "w") as f:
        json.dump(agent_kwargs, f)

    # save inputs and outputs
    for subdir, infos in [('inputs', agent_inputs), ('outputs', agent_outputs)]:
        subdirpath = os.path.join(dirpath, subdir)
        os.makedirs(subdirpath, exist_ok=True)

        variable_names = infos[0].keys()
        for variable_name in variable_names:
            variable = np.vstack([info[variable_name] for info in infos])
            np.savetxt(os.path.join(subdirpath, f'{variable_name}s.csv'), variable, delimiter=',')


def sim_agent(env: Classical, agent: Agent, num_steps: int) -> Tuple[
    List[Dict[str, np.array]], List[Dict[str, np.array]], List[float]]:
    # reset
    observation, env_info = env.reset()
    agent.reset()

    # bookkeeping
    agent_outputs = []
    agent_inputs = []
    episode_reward = 0
    episode_rewards = []

    for step in range(num_steps):
        # agent -> environment
        action = agent.act(observation)
        next_observation, reward, terminated, truncated, env_info = env.step(action)

        # environment -> agent
        agent_output = agent.observe(observation, action, reward, next_observation)
        agent_output['action'] = action

        # bookkeeping
        agent_outputs.append(agent_output)
        agent_inputs.append({'observation': observation, 'reward': reward, 'next_observation': next_observation,
                             'terminated': terminated, 'truncated': truncated})
        episode_reward += reward

        # iterate
        observation = next_observation

        # reset env (optionally)
        if terminated or truncated:
            observation, env_info = env.reset()

            episode_rewards.append(episode_reward)
            episode_reward = 0

    return agent_inputs, agent_outputs, episode_rewards


def gen_agent_data(root_data_dir: os.path, env_name: str, agent_class: typing.Type[Agent],
                   agent_kwargs: Dict[str, Any] = {}, num_steps: int = 100, do_plot: bool = False) -> None:
    # create env and agent
    env = gym.make(env_name)
    agent = agent_class(env, **agent_kwargs)

    # simulate agent
    agent_inputs, agent_outputs, episode_rewards = sim_agent(env, agent, num_steps)

    # save data
    data_dir = os.path.join(root_data_dir, env_name, str(agent))
    print(f'Saving data to {data_dir}')
    save_data(data_dir, agent_inputs, agent_outputs, agent_kwargs)

    # plot (optionally)
    if do_plot:
        plt.plot(range(len(episode_rewards)), episode_rewards)
        plt.show()
        plt.xlabel('episode')
        plt.ylabel('cumulative reward')


if __name__ == '__main__':
    for lr in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
        gen_agent_data(root_data_dir=os.path.join('data', 'instrumental'),
                       env_name="CartPole-v1",
                       agent_class=Q,
                       agent_kwargs={'learning_rate': 0.50},
                       num_steps=10000,
                       do_plot=False)
