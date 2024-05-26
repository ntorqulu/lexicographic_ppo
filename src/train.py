import argparse
import json
import time
import logging
import gym
from EthicalGatheringGame.presets import tiny, large, small
from TrainingParameters import TrainingParameters
from PPO import PPO
from LPPO import LPPO
from callbacks import PrintAverageReward, TensorBoardLogging, AnnealEntropy
from lr_schedules import IndependentPPOAnnealing

"""
Best average reward LPPO policy for safety:
eta = 2.5; beta = 1, 0.5

Best average reward LPPO policy for performance:
eta = 2.5; beta = 0.5, 1

"""

# Constants for execution classes
PPO_CLASS = "PPO"
LPPO_CLASS = "LPPO"


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Initialize agent with custom parameters")
    parser.add_argument("--config", type=str, help="Path to the configuration JSON file")
    parser.add_argument("--we_reward0", type=float, default=1, help="Reward for we[0]")
    parser.add_argument("--we_reward1", type=float, default=10, help="Reward for we[1]")
    parser.add_argument("--tot_steps", type=int, default=15000000, help="Total number of steps")
    parser.add_argument("--execution_class", type=str, default=LPPO_CLASS, help="Execution class")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--prioritize_performance_over_safety", action='store_true',
                        help="Prioritize performance over safety")

    # Conditional argument for reward size based on execution class
    args, _ = parser.parse_known_args()
    if args.execution_class == PPO_CLASS:
        parser.add_argument("--reward_size", type=int, default=1, help="Size of the reward vector")
    else:
        parser.add_argument("--reward_size", type=int, default=2, help="Size of the reward vector")

    parser.add_argument("--beta_values", nargs='+', type=float, default=[1, 0.5], help="Beta values")
    parser.add_argument("--eta_value", type=float, default=2.5, help="Eta value")
    return parser.parse_args()


def load_config(config_path):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def configure_environment(config):
    """
    Configure the environment based on parsed arguments.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Configuration dictionary for the environment.
    """
    tiny["we"] = [config['we_reward0'], config['we_reward1']]
    tiny["reward_mode"] = "scalarised" if config['reward_size'] == 1 else "vectorial"
    tiny["inequality_mode"] = "loss"
    tiny["efficiency"] = [0.85, 0.2]
    tiny["n_agents"] = 2
    return tiny


def filter_training_parameters(config):
    """
    Filter the training parameters from the config dictionary.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Filtered configuration dictionary for training parameters.
    """
    training_param_keys = [
        'env_name', 'tag', 'seed', 'th_deterministic', 'save_dir', 'n_agents', 'reward_size',
        'tot_steps', 'batch_size', 'max_steps', 'max_grad_norm', 'n_epochs', 'critic_times',
        'actor_lr', 'critic_lr', 'anneal_lr', 'anneal_entropy', 'concavity_entropy', 'clip',
        'ent_coef', 'gae_lambda', 'gamma', 'v_coef', 'n_cpus', 'cuda', 'h_size', 'h_layers',
        'we_reward0', 'we_reward1', 'execution_class', 'eval_mode', 'beta_values', 'eta_value',
        'prioritize_performance_over_safety'
    ]
    return {key: config[key] for key in training_param_keys if key in config}


def main():
    """
    Main function to set up and train the agent.
    """
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # Load configuration from JSON file if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = vars(args)

    # Configure environment parameters
    env_config = configure_environment(config)

    # Create the environment
    env = gym.make("MultiAgentEthicalGathering-v1", **env_config)

    # Initialize training parameters
    filtered_config = filter_training_parameters(config)
    training_params = TrainingParameters(**filtered_config)

    # Select the appropriate PPO class
    if training_params.execution_class == PPO_CLASS:
        ppo = PPO(training_params, env=env)
    elif training_params.execution_class == LPPO_CLASS:
        ppo = LPPO(training_params, env=env)
    else:
        raise ValueError("Execution class not recognized")

    # Set the learning rate scheduler
    ppo.lr_scheduler = IndependentPPOAnnealing(ppo, {
        0: {"actor_lr": 0.0003, "critic_lr": 0.001},
        1: {"actor_lr": 0.0003, "critic_lr": 0.001},
    })

    # Add callbacks for training
    ppo.addCallbacks(AnnealEntropy(ppo, 1.0, 0.5, training_params.concavity_entropy))
    ppo.addCallbacks(PrintAverageReward(ppo, 300))
    ppo.addCallbacks(TensorBoardLogging(ppo, f"StoreNuria/tensorboard_logs/{type(ppo).__name__}/{ppo.run_name}"))

    # Track training time and start training
    t0 = time.time()
    ppo.train()
    t = time.time() - t0

    # Log training performance
    logging.info(f"Steps per second: {ppo.tot_steps / t:.2f}")
    logging.info(f"Time elapsed: {t:.2f} seconds")


if __name__ == '__main__':
    main()
