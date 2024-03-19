import time
import gym
from EthicalGatheringGame.presets import tiny
from EthicalGatheringGame.wrappers import NormalizeReward
from LexicoPPO import LexicoPPO
from src.TrainingParameters import TrainingParameters

env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
params = TrainingParameters(env_name="MultiAgentEthicalGathering-v1")
env = NormalizeReward(env)
ppo = LexicoPPO(train_params=params, env=env)
start = time.time()
agents = ppo.actors_from_file("example_data/tiny/15000_5_1_(1)")

# Run a simulation of the trained agents
obs, info = env.reset()
done = False
while not done:
    actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
    obs, rewards, done, info = env.step(actions)
    done = all(done)
    env.render()
