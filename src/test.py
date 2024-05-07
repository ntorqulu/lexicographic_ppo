import time
import os
import gym
import matplotlib
from EthicalGatheringGame.presets import tiny
from EthicalGatheringGame.wrappers import NormalizeReward
from TrainingParameters import TrainingParameters
from callbacks import PrintAverageReward, TensorBoardLogging
import re

from PPO import PPO
from LPPO import LPPO

matplotlib.use("TkAgg")

# Define the directory where the files are expected to be generated
directory_path = "StoreNuria/tiny/2500_30000_1_(1)"
tiny["we"] = [1, 10]
tiny["reward_mode"] = "vectorial"
tiny["inequality_mode"] = "loss"
tiny["efficiency"] = [0.85, 0.2]
env = gym.make("MultiAgentEthicalGathering-v1", **tiny)

agents = LPPO.actors_from_file(directory_path)

# Run a simulation of the trained agents
env.toggleTrack(True)
env.toggleStash(True)
env.reset()

n_sims = 50
for sim in range(n_sims):
    obs, info = env.reset()
    done = False
    while not done:
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
        obs, rewards, done, info = env.step(actions)
        done = all(done)
        #env.render()

env.plot_results("median")
env.print_results()
