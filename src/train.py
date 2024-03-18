import os

import gym
import numpy as np
import torch
from EthicalGatheringGame.presets import tiny
from TrainingParameters import TrainingParameters
from LexicoPPO import LexicoPPO

if __name__ == '__main__':
    # Create the environment
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)

    # Set parameters
    params = TrainingParameters(env_name="MultiAgentEthicalGathering-v1")

    # initialize lexico_ppo agent
    agent = LexicoPPO(train_params=params, env=env)

    # Train the agent
    agent.train()  # without reset and set_agents for now
