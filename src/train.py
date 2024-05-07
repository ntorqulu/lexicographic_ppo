import argparse
import time
import gym
from EthicalGatheringGame.wrappers import NormalizeReward
from EthicalGatheringGame.presets import tiny
from TrainingParameters import TrainingParameters
from PPO import PPO
from LPPO import LPPO
from callbacks import PrintAverageReward, TensorBoardLogging


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize agent with custom parameters")
    parser.add_argument("--we_reward0", type=float, default=1, help="Reward for we[0]")
    parser.add_argument("--we_reward1", type=float, default=10, help="Reward for we[1]")
    parser.add_argument("--tot_steps", type=int, default=15000000, help="Total number of steps")
    parser.add_argument("--execution_class", type=str, default="LPPO", help="Execution class")
    # if type is PPO, one reward is used, if LexicoPPO, two rewards are used
    if parser.parse_known_args()[0].execution_class == "PPO":
        parser.add_argument("--reward_size", type=int, default=1, help="Size of the reward vector")
    else:
        parser.add_argument("--reward_size", type=int, default=2, help="Size of the reward vector")
    parser.add_argument("--beta_values", nargs='+', type=float, default=None, help="Beta values")
    parser.add_argument("--eta_values", nargs='+', type=float,  default=None, help="Eta values")
    return parser.parse_args()


if __name__ == '__main__':
    # initialize environment parameters
    args = parse_args()
    tiny["we"] = [args.we_reward0, args.we_reward1]
    if args.reward_size == 1:
        tiny["reward_mode"] = "scalarised"
    else:
        tiny["reward_mode"] = "vectorial"
    tiny["inequality_mode"] = "loss"
    tiny["efficiency"] = [0.85, 0.2]

    # Create the environment
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)

    training_params = TrainingParameters(**vars(args))
    if training_params.execution_class == "PPO":
        ppo = PPO(training_params, env=env)
    elif training_params.execution_class == "LPPO":
        ppo = LPPO(training_params, env=env)
    else:
        raise ValueError("Execution class not recognized")

    # Add callbacks
    ppo.addCallbacks(PrintAverageReward(ppo, 300))
    ppo.addCallbacks(TensorBoardLogging(ppo, f"StoreNuria/tensorboard_logs/{type(ppo).__name__}/{ppo.run_name}"))

    # Train the agent
    ppo.train()
