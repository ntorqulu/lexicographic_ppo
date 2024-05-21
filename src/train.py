import argparse
import time
import logging
import gym
from EthicalGatheringGame.wrappers import NormalizeReward
from EthicalGatheringGame.presets import tiny, large, small
from TrainingParameters import TrainingParameters
from PPO import PPO
from LPPO import LPPO
from callbacks import PrintAverageReward, TensorBoardLogging, AnnealEntropy
from lr_schedules import IndependentPPOAnnealing

"""
Best average reward LPPO policy for performance:
2; 0.5, 1

"""

# Constants for execution classes
PPO_CLASS = "PPO"
LPPO_CLASS = "LPPO"


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize agent with custom parameters")
    parser.add_argument("--we_reward0", type=float, default=1, help="Reward for we[0]")
    parser.add_argument("--we_reward1", type=float, default=10, help="Reward for we[1]")
    parser.add_argument("--tot_steps", type=int, default=15000000, help="Total number of steps")
    parser.add_argument("--execution_class", type=str, default=LPPO_CLASS, help="Execution class")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--prioritize_performance_over_safety", action='store_true', help="Prioritize performance over safety")

    # Conditional argument for reward size based on execution class
    args, _ = parser.parse_known_args()
    if args.execution_class == PPO_CLASS:
        parser.add_argument("--reward_size", type=int, default=1, help="Size of the reward vector")
    else:
        parser.add_argument("--reward_size", type=int, default=2, help="Size of the reward vector")

    parser.add_argument("--beta_values", nargs='+', type=float, default=[2, 1], help="Beta values")
    parser.add_argument("--eta_value", type=float, default=0.1, help="Eta value")
    return parser.parse_args()


def configure_environment(args):
    tiny["we"] = [args.we_reward0, args.we_reward1]
    tiny["reward_mode"] = "scalarised" if args.reward_size == 1 else "vectorial"
    tiny["inequality_mode"] = "loss"
    tiny["efficiency"] = [0.85, 0.2]
    tiny["n_agents"] = 2
    return tiny


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # Configure environment parameters
    env_config = configure_environment(args)

    # Create the environment
    env = gym.make("MultiAgentEthicalGathering-v1", **env_config)

    training_params = TrainingParameters(**vars(args))

    if training_params.execution_class == PPO_CLASS:
        ppo = PPO(training_params, env=env)
    elif training_params.execution_class == LPPO_CLASS:
        ppo = LPPO(training_params, env=env)
    else:
        raise ValueError("Execution class not recognized")

    ppo.lr_scheduler = IndependentPPOAnnealing(ppo, {
        0: {"actor_lr": 0.0003, "critic_lr": 0.001},
        1: {"actor_lr": 0.0003, "critic_lr": 0.001},
    })

    # Add callbacks
    ppo.addCallbacks(AnnealEntropy(ppo, 1.0, 0.5, training_params.concavity_entropy))
    ppo.addCallbacks(PrintAverageReward(ppo, 300))
    ppo.addCallbacks(TensorBoardLogging(ppo, f"StoreNuria/tensorboard_logs/{type(ppo).__name__}/{ppo.run_name}"))

    # Track training time
    t0 = time.time()
    ppo.train()
    t = time.time() - t0

    logging.info(f"Steps per second: {ppo.tot_steps / t:.2f}")
    logging.info(f"Time elapsed: {t:.2f} seconds")


if __name__ == '__main__':
    main()

