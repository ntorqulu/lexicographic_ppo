import numpy as np

from eval_policy_visualization import *

matplotlib.use("TkAgg")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_simulation(env: gym.Env, agents: list) -> dict:
    """
    Run a single simulation with the given environment and agents.
    """
    obs, info = env.reset()
    done = False

    total_rewards = np.zeros(len(agents))
    R_missedEthical_counts = np.zeros(len(agents))
    R_nonEthical_counts = np.zeros(len(agents))
    suboptimal_counts = np.zeros(len(agents))

    while not done:
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
        obs, rewards, done, info = env.step(actions)
        done = all(done)
        for i in range(len(agents)):
            # np.dot with weights [1, 10]
            total_rewards[i] += rewards[i]
            R_missedEthical_counts[i] += info["R'_E"][i]
            R_nonEthical_counts[i] += info["R'_N"][i]

            # Check for suboptimal actions
            agent_id = chr(65 + i)  # Convert index to corresponding letter (0 -> A, 1 -> B)
            if 'suboptimal' in info[agent_id]['events']:
                logger.info(f"Agent {agent_id} took suboptimal action.")
                suboptimal_counts[i] += 1

    sim_data = info["sim_data"]
    return {
        "total_rewards": total_rewards,
        "R_missedEthical_counts": R_missedEthical_counts,
        "R_nonEthical_counts": R_nonEthical_counts,
        "suboptimal_counts": suboptimal_counts,
        "time_to_survival": sim_data["time_to_survival"],
        "donation_box_full": sim_data["donation_box_full"]
    }


def aggregate_simulation_results(results: list, n_agents: int) -> dict:
    """
    Aggregate the results of multiple simulations.
    """
    survival_times = np.array([result["time_to_survival"] for result in results])
    total_rewards = np.array([result["total_rewards"] for result in results])
    R_missedEthical_counts = np.array([result["R_missedEthical_counts"] for result in results])
    R_nonEthical_counts = np.array([result["R_nonEthical_counts"] for result in results])
    suboptimal_counts = np.array([result["suboptimal_counts"] for result in results])
    donation_full_times = np.array([result["donation_box_full"] for result in results])

    times_not_survived = np.sum(survival_times == -1, axis=0)
    times_not_full = np.sum(donation_full_times == -1)

    aggregated_results = {
        "mean_survival_times": np.mean(survival_times, axis=0),
        "std_survival_times": np.std(survival_times, axis=0),
        "times_not_survived": times_not_survived,
        "mean_total_rewards": np.mean(total_rewards, axis=0),
        "std_total_rewards": np.std(total_rewards, axis=0),
        "mean_R_E_counts": np.mean(R_missedEthical_counts, axis=0),
        "std_R_E_counts": np.std(R_missedEthical_counts, axis=0),
        "mean_R_N_counts": np.mean(R_nonEthical_counts, axis=0),
        "std_R_N_counts": np.std(R_nonEthical_counts, axis=0),
        "mean_suboptimal_counts": np.mean(suboptimal_counts, axis=0),
        "std_suboptimal_counts": np.std(suboptimal_counts, axis=0),
        "mean_donation_full_time": np.mean(donation_full_times),
        "std_donation_full_time": np.std(donation_full_times),
        "times_not_full": times_not_full
    }

    # Return raw metrics as well for saving to .npz files
    raw_metrics = {
        "survival_times": survival_times,
        "total_rewards": total_rewards,
        "R_missedEthical_counts": R_missedEthical_counts,
        "R_nonEthical_counts": R_nonEthical_counts,
        "suboptimal_counts": suboptimal_counts,
        "donation_full_times": donation_full_times
    }

    return aggregated_results, raw_metrics


def log_simulation_results(results: dict):
    """
    Log the aggregated results of the simulations.
    """
    for i, (mean, std) in enumerate(zip(results["mean_survival_times"], results["std_survival_times"])):
        logger.info(f"Agent {i} - Mean Time to Survival: {mean:.2f}, Std: {std:.2f}")

    for i, times in enumerate(results["times_not_survived"]):
        logger.info(f"Agent {i} - Times not Survived: {times}")

    for i, (mean, std) in enumerate(zip(results["mean_total_rewards"], results["std_total_rewards"])):
        logger.info(f"Agent {i} - Mean Total Reward: {mean:.2f}, Std: {std:.2f}")

    logger.info(
        f"Donation Box - Mean Time to Full: {results['mean_donation_full_time']:.2f}, Std: {results['std_donation_full_time']:.2f}")
    logger.info(f"Donation Box - Times not Full: {results['times_not_full']}")

    for i, (mean, std) in enumerate(zip(results["mean_R_E_counts"], results["std_R_E_counts"])):
        logger.info(f"Agent {i} - Mean R'_E Actions: {mean:.2f}, Std: {std:.2f}")

    for i, (mean, std) in enumerate(zip(results["mean_R_N_counts"], results["std_R_N_counts"])):
        logger.info(f"Agent {i} - Mean R'_N Actions: {mean:.2f}, Std: {std:.2f}")

    for i, (mean, std) in enumerate(zip(results["mean_suboptimal_counts"], results["std_suboptimal_counts"])):
        logger.info(f"Agent {i} - Mean Suboptimal Actions: {mean:.2f}, Std: {std:.2f}")


def run_simulations(env: gym.Env, agents: list, n_sims: int = 100):
    """
    Run multiple simulations and aggregate the results.
    """
    results = []
    for _ in range(n_sims):
        result = run_simulation(env, agents)
        results.append(result)
    return results


def evaluate_policies_across_seeds(base_directory_path: str, seeds: range, n_sims: int):
    all_results = []

    for seed in seeds:
        directory_path = os.path.join(base_directory_path, f"2500_50000_{seed}_(1)")
        if not os.path.exists(directory_path):
            logger.warning(f"Directory {directory_path} does not exist. Skipping seed {seed}.")
            continue

        logger.info(f"Configuring environment for seed {seed}...")
        env = configure_environment("scalarised")

        logger.info(f"Loading trained agents for seed {seed}...")
        agents = load_agents(directory_path, "PPO")

        logger.info(f"Running simulations for seed {seed}...")
        seed_results = run_simulations(env, agents, n_sims)
        # if you want a display of results for each seed
        aggregated_results, raw_metrics = aggregate_simulation_results(seed_results, len(agents))
        log_simulation_results(aggregated_results)
        all_results.extend(seed_results)
        # Save raw metrics to .npz file
        save_path = os.path.join(directory_path, f"metrics_{seed}.npz")
        np.savez(save_path, **raw_metrics)

    if all_results:
        aggregated_results, _ = aggregate_simulation_results(all_results, len(agents))
        log_simulation_results(aggregated_results)
    else:
        logger.error("No valid results to aggregate and log.")


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained policies.")
    parser.add_argument("--directory_path", type=str, default="src/StoreNuria/policies/PPOseed",
                        help="Directory path for saving models.")
    parser.add_argument("--n_sims", type=int, default=100, help="Number of simulations to run.")
    return parser.parse_args()


def main():
    """
    Main function to configure the environment, load agents, and run simulations.
    """
    args = parse_args()
    base_directory_path = args.directory_path
    # if you want only to run a seed, specify a range of one number eg range(1, 2)
    seeds = range(1, 21)
    n_sims = args.n_sims
    evaluate_policies_across_seeds(base_directory_path, seeds, n_sims)


if __name__ == "__main__":
    main()
