import time
import os
import gym
import matplotlib
from EthicalGatheringGame.presets import tiny
from EthicalGatheringGame.wrappers import NormalizeReward
from LexicoPPO import LexicoPPO
from TrainingParameters import TrainingParameters
from callbacks import PrintAverageReward, TensorBoardLogging
import re

matplotlib.use("TkAgg")
env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
env = NormalizeReward(env)
params = TrainingParameters(env_name="MultiAgentEthicalGathering-v1")

# initialize lexico_ppo agent
ppo = LexicoPPO(train_params=params, env=env)
ppo.addCallbacks(PrintAverageReward(ppo, 1))
ppo.addCallbacks(TensorBoardLogging(ppo, "example_data"))
ppo.train()

# Define the directory where the files are expected to be generated
directory_path = "example_data/tiny"

# Wait for the file to be generated
while True:
    matching_files = [file for file in os.listdir(directory_path) if re.match(r'^\d+_\d+_\d+_\(\d+\)', file)]
    if matching_files:
        latest_file = max(matching_files, key=lambda x: int(re.search(r'\((\d+)\)', x).group(1)))
        file_path = os.path.join(directory_path, latest_file)
        break
    else:
        time.sleep(1)  # Check every second if there's a matching file

# Fetch agents from the generated file
print(f"Fetching agents from file: {file_path}")
agents = LexicoPPO.actors_from_file(file_path)

# Run a simulation of the trained agents
obs, info = env.reset()
done = False
while not done:
    actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
    obs, rewards, done, info = env.step(actions)
    done = all(done)
    env.render()
