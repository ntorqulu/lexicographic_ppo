import os
import gym
import logging
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
from EthicalGatheringGame.presets import tiny
from LPPO import LPPO
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure_environment() -> gym.Env:
    tiny["we"] = [1, 10]
    tiny["reward_mode"] = "vectorial"
    tiny["inequality_mode"] = "loss"
    tiny["efficiency"] = [0.85, 0.2]
    tiny["n_agents"] = 2
    return gym.make("MultiAgentEthicalGathering-v1", **tiny)


def load_agents(directory_path: str) -> list:
    if not os.path.exists(directory_path):
        logger.error(f"Directory {directory_path} does not exist.")
        raise FileNotFoundError(f"Directory {directory_path} does not exist.")
    return LPPO.actors_from_file(directory_path)


def run_simulation(env: gym.Env, agents: list) -> dict:
    obs, info = env.reset()
    done = False

    total_rewards = np.zeros(len(agents))

    while not done:
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
        obs, rewards, done, info = env.step(actions)
        done = all(done)
        for i in range(len(agents)):
            total_rewards[i] += np.dot(rewards[i], [1, 10])

    return {
        "total_rewards": total_rewards
    }


def collect_rewards_across_seeds(base_directory_path: str, seeds: range, n_sims: int) -> pd.DataFrame:
    reward_data = []

    for seed in seeds:
        logger.info(f"Loading trained agents for seed {seed}...")
        directory_path = f"{base_directory_path}/2500_50000_{seed}"
        agents = load_agents(directory_path)

        logger.info(f"Running simulations for seed {seed}...")
        env = configure_environment()

        for sim in range(n_sims):
            result = run_simulation(env, agents)
            for i in range(len(agents)):
                reward_data.append({"Seed": seed, "Agent": f"Agent {i}", "Reward": result["total_rewards"][i]})

    return pd.DataFrame(reward_data)


def plot_rewards(reward_data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Reward", y="Agent", data=reward_data)
    plt.title("Reward per agent")
    plt.show()


def main():
    base_directory_path = "StoreNuria/LPPOseed"
    seeds = range(1, 21)
    n_sims = 100

    reward_data = collect_rewards_across_seeds(base_directory_path, seeds, n_sims)
    plot_rewards(reward_data)


if __name__ == "__main__":
    main()
