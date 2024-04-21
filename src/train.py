import time
import gym
from EthicalGatheringGame.wrappers import NormalizeReward
from EthicalGatheringGame.presets import tiny
from TrainingParameters import TrainingParameters
from PPO import PPO
from LexicoPPO import LexicoPPO
from PPOmultiRewards import PPOmultiRewards
from callbacks import PrintAverageReward, TensorBoardLogging

if __name__ == '__main__':
    # Create the environment
    # scalarised or vectorial
    tiny["reward_mode"] = "vectorial"
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
    # Set parameters
    params = TrainingParameters(env_name="MultiAgentEthicalGathering-v1")
    # initialize lexico_ppo agent
    ppo = LexicoPPO(train_params=params, env=env)
    ppo.addCallbacks(PrintAverageReward(ppo, 300))
    ppo.addCallbacks(TensorBoardLogging(ppo, f"StoreNuria/tensorboard_logs/{ppo.run_name}"))
    ppo.train()
