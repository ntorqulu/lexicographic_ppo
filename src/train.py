import os

import gym
import torch
from EthicalGatheringGame.presets import tiny
from TrainingParameters import TrainingParameters
from LexicoPPO import LexicoPPO


def train_from_params(train_params: TrainingParameters, session_pref: str, show_progress: bool = True):
    device = torch.device("cpu")


if __name__ == '__main__':
    # Create the environment
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)

    # Set parameters
    params = TrainingParameters(env_name="MultiAgentEthicalGathering-v1", tot_steps=15000)
    prioritize_collecting_over_sharing = True

    # initialize lexico_ppo agent
    agent = LexicoPPO(train_params=params, env=env)
    if prioritize_collecting_over_sharing:
        mode = 1  # first component positive, second negative
    else:  # prioritize sharing over collecting
        mode = 2  # first component negative, second positive

    # Train the agent
    agent.train()  # without reset and set_agents for now
