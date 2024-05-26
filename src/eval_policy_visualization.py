import argparse
import os
import gym
import logging
import matplotlib
from EthicalGatheringGame.presets import tiny
from PPO import PPO
from LPPO import LPPO

matplotlib.use("TkAgg")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure_environment(reward_mode: str) -> gym.Env:
    """
    Configures the environment with the specified parameters.

    Parameters:
    reward_mode (str): The mode of the reward ('scalarised' or 'vectorial').

    Returns:
    gym.Env: Configured gym environment.
    """
    tiny["we"] = [1, 10]
    tiny["reward_mode"] = reward_mode
    tiny["inequality_mode"] = "loss"
    tiny["efficiency"] = [0.85, 0.2]
    tiny["n_agents"] = 2
    return gym.make("MultiAgentEthicalGathering-v1", **tiny)


def load_agents(directory_path: str, execution_class: str) -> list:
    """
    Loads trained agents from the specified directory.

    Parameters:
    directory_path (str): Path to the directory containing the trained agents.
    execution_class (str): The execution class ('PPO' or 'LPPO').

    Returns:
    list: List of loaded agents.
    """
    if not os.path.exists(directory_path):
        logger.error(f"Directory {directory_path} does not exist.")
        raise FileNotFoundError(f"Directory {directory_path} does not exist.")
    if execution_class == "PPO":
        return PPO.actors_from_file(directory_path)
    elif execution_class == "LPPO":
        return LPPO.actors_from_file(directory_path)
    else:
        logger.error(f"Unsupported execution class: {execution_class}")
        raise ValueError(f"Unsupported execution class: {execution_class}")


def run_simulations(env: gym.Env, agents: list, n_sims: int = 1000, render: bool = False):
    """
    Runs a number of simulations with the trained agents.

    Parameters:
    env (gym.Env): The environment in which to run the simulations.
    agents (list): List of trained agents.
    n_sims (int): Number of simulations to run.
    """
    env.toggleTrack(True)
    env.toggleStash(True)

    for sim in range(n_sims):
        obs, info = env.reset()
        done = False

        while not done:
            actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
            obs, rewards, done, info = env.step(actions)
            done = all(done)
            if render:
                env.render()
    env.plot_results("median")
    env.print_results()


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained policies.")
    parser.add_argument("--directory_path", type=str, default="src/StoreNuria/policies/LPPOSafetySeed/2500_50000_1",
                        help="Directory path for saving models.")
    parser.add_argument("--execution_class", type=str, choices=["PPO", "LPPO"], default="LPPO",
                        help="Execution class ('PPO' or 'LPPO').")
    parser.add_argument("--n_sims", type=int, default=100, help="Number of simulations to run.")
    parser.add_argument("--render", action="store_true", help="Render the environment during simulations.")
    return parser.parse_args()


def main():
    """
    Main function to execute the testing of trained policies.
    """
    args = parse_args()

    directory_path = args.directory_path
    execution_class = args.execution_class
    reward_mode = "vectorial" if execution_class == "LPPO" else "scalarised"

    logger.info("Configuring environment...")
    env = configure_environment(reward_mode)

    logger.info("Loading trained agents...")
    agents = load_agents(directory_path, execution_class)

    logger.info("Running simulations...")
    run_simulations(env, agents, n_sims=args.n_sims, render=args.render)


if __name__ == "__main__":
    main()
