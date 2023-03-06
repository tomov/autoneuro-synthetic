""" Generate synthetic data for classical conditioning """

import os
import typing
from typing import Dict, List

import numpy as np

from agents.agent import Agent
from agents.rw import RW
from agents.td import TD
from envs.classical import Classical
from utils.env_factories import classical_factory

dirs = [
    os.path.join('data', 'classical', 'overshadowing')
]


def save_outputs(output_dir: os.path, agent_infos: List[Dict[str, np.array]]) -> None:
    os.makedirs(output_dir, exist_ok=True)

    output_names = agent_infos[0].keys()
    for output_name in output_names:
        output = np.vstack([agent_info[output_name] for agent_info in agent_infos])
        np.savetxt(os.path.join(output_dir, f'{output_name}s.csv'), output, delimiter=',')


def sim_agent(env: Classical, agent: Agent) -> List[Dict[str, np.array]]:
    observation, env_info = env.reset()
    agent.reset()

    agent_infos = []
    while True:
        action = agent.act(observation)
        next_observation, reward, terminated, truncated, env_info = env.step(action)

        agent_info = agent.observe(observation, action, reward, next_observation)
        agent_infos.append(agent_info)

        observation = next_observation
        if terminated or truncated:
            break

    return agent_infos


def gen_agent_data(input_dir: os.path, output_dir: os.path, agent_class: typing.Type[Agent]) -> None:
    env = classical_factory(input_dir)
    agent = agent_class(env)
    agent_infos = sim_agent(env, agent)
    save_outputs(output_dir, agent_infos)


if __name__ == '__main__':
    for dirpath in dirs:
        gen_agent_data(input_dir=os.path.join(dirpath, 'input'), output_dir=os.path.join(dirpath, 'output', 'RW'),
                       agent_class=RW)
        gen_agent_data(input_dir=os.path.join(dirpath, 'input'), output_dir=os.path.join(dirpath, 'output', 'TD'),
                       agent_class=TD)
