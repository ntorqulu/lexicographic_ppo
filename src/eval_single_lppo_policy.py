import matplotlib.pyplot as plt
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
    individual_rewards = np.zeros(len(agents))
    ethical_rewards = np.zeros(len(agents))
    R_missedEthical_counts = np.zeros(len(agents))
    R_nonEthical_counts = np.zeros(len(agents))
    suboptimal_counts = np.zeros(len(agents))

    while not done:
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
        obs, rewards, done, info = env.step(actions)
        done = all(done)
        # print(f"Rewards: {rewards}")
        for i in range(len(agents)):
            # np.dot with weights [1, 10]
            total_rewards[i] += np.dot(rewards[i], [1, 10])
            individual_rewards[i] += rewards[i][0]
            ethical_rewards[i] += rewards[i][1] * 10
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
        "individual_rewards": individual_rewards,
        "ethical_rewards": ethical_rewards,
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
    individual_rewards = np.array([result["individual_rewards"] for result in results])
    ethical_rewards = np.array([result["ethical_rewards"] for result in results])
    R_missedEthical_counts = np.array([result["R_missedEthical_counts"] for result in results])
    R_nonEthical_counts = np.array([result["R_nonEthical_counts"] for result in results])
    suboptimal_counts = np.array([result["suboptimal_counts"] for result in results])
    donation_full_times = np.array([result["donation_box_full"] for result in results])

    times_not_survived = np.sum(survival_times == -1, axis=0)
    times_not_full = np.sum(donation_full_times == -1)

    return {
        "mean_survival_times": np.mean(survival_times, axis=0),
        "std_survival_times": np.std(survival_times, axis=0),
        "times_not_survived": times_not_survived,
        "mean_total_rewards": np.mean(total_rewards, axis=0),
        "std_total_rewards": np.std(total_rewards, axis=0),
        "mean_individual_rewards": np.mean(individual_rewards, axis=0),
        "std_individual_rewards": np.std(individual_rewards, axis=0),
        "mean_ethical_rewards": np.mean(ethical_rewards, axis=0),
        "std_ethical_rewards": np.std(ethical_rewards, axis=0),
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

    for i, (mean, std) in enumerate(zip(results["mean_individual_rewards"], results["std_individual_rewards"])):
        logger.info(f"Agent {i} - Mean Individual Reward: {mean:.2f}, Std: {std:.2f}")

    for i, (mean, std) in enumerate(zip(results["mean_ethical_rewards"], results["std_ethical_rewards"])):
        logger.info(f"Agent {i} - Mean Ethical Reward: {mean:.2f}, Std: {std:.2f}")

    logger.info(
        f"Donation Box - Mean Time to Full: {results['mean_donation_full_time']:.2f}, Std: {results['std_donation_full_time']:.2f}")
    logger.info(f"Donation Box - Times not Full: {results['times_not_full']}")

    for i, (mean, std) in enumerate(zip(results["mean_R_E_counts"], results["std_R_E_counts"])):
        logger.info(f"Agent {i} - Mean R'_E Actions: {mean:.2f}, Std: {std:.2f}")

    for i, (mean, std) in enumerate(zip(results["mean_R_N_counts"], results["std_R_N_counts"])):
        logger.info(f"Agent {i} - Mean R'_N Actions: {mean:.2f}, Std: {std:.2f}")

    for i, (mean, std) in enumerate(zip(results["mean_suboptimal_counts"], results["std_suboptimal_counts"])):
        logger.info(f"Agent {i} - Mean Suboptimal Actions: {mean:.2f}, Std: {std:.2f}")


def plot_single_result(metric_name, agent_ids, means, stds, ylabel):
    """
    Plot a single result metric with error bars.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(agent_ids, means, yerr=stds, capsize=5)
    plt.title(f"Mean {metric_name}")
    plt.ylabel(ylabel)
    plt.show()


def plot_simulation_results(results: dict):
    """
    Plot the aggregated results of the simulations.
    """
    n_agents = len(results["mean_survival_times"])
    agent_ids = [f"Agent {i}" for i in range(n_agents)]

    plot_single_result("Total Rewards", agent_ids, results["mean_total_rewards"], results["std_total_rewards"],
                       "Reward")
    plot_single_result("Individual Rewards", agent_ids, results["mean_individual_rewards"],
                       results["std_individual_rewards"],
                       "Reward")
    plot_single_result("Ehical Rewards", agent_ids, results["mean_ethical_rewards"], results["std_ethical_rewards"],
                       "Reward")
    plot_single_result("Time to Survival", agent_ids, results["mean_survival_times"], results["std_survival_times"],
                       "Time")
    plot_single_result("R'_E Actions", agent_ids, results["mean_R_E_counts"], results["std_R_E_counts"], "Count")
    plot_single_result("R'_N Actions", agent_ids, results["mean_R_N_counts"], results["std_R_N_counts"], "Count")
    plot_single_result("Suboptimal Actions", agent_ids, results["mean_suboptimal_counts"],
                       results["std_suboptimal_counts"], "Count")

    plt.figure(figsize=(8, 6))
    plt.bar(["Donation Box"], [results["mean_donation_full_time"]], yerr=[results["std_donation_full_time"]], capsize=5)
    plt.title("Donation Box - Mean Time to Full")
    plt.ylabel("Time")
    plt.show()


def run_simulations(env: gym.Env, agents: list, n_sims: int = 100):
    """
    Run multiple simulations and aggregate the results.
    """
    env.toggleTrack(True)
    env.toggleStash(True)

    results = []
    for _ in range(n_sims):
        result = run_simulation(env, agents)
        results.append(result)

    aggregated_results = aggregate_simulation_results(results, len(agents))
    log_simulation_results(aggregated_results)
    plot_simulation_results(aggregated_results)
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
    parser.add_argument("--n_sims", type=int, default=100, help="Number of simulations to run.")
    return parser.parse_args()


def main():
    """
    Main function to configure the environment, load agents, and run simulations.
    """
    args = parse_args()
    directory_path = args.directory_path
    logger.info("Configuring environment...")
    env = configure_environment("vectorial")

    logger.info("Loading trained agents...")
    agents = load_agents(directory_path, "LPPO")

    logger.info("Running simulations...")
    run_simulations(env, agents, n_sims=args.n_sims)


if __name__ == "__main__":
    main()
