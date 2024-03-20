import time
import gym
from EthicalGatheringGame.wrappers import NormalizeReward
from EthicalGatheringGame.presets import tiny
from TrainingParameters import TrainingParameters
from LexicoPPO import LexicoPPO
from src.callbacks import PrintAverageReward

if __name__ == '__main__':
    # Create the environment
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
    env = NormalizeReward(env)
    # Set parameters
    params = TrainingParameters(env_name="MultiAgentEthicalGathering-v1")

    # initialize lexico_ppo agent
    ppo = LexicoPPO(train_params=params, env=env)
    ppo.addCallbacks(PrintAverageReward(ppo, 1))
    ppo.train()
