import os
import numpy as np
import logging
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_metrics(directory_path):
    metrics = {
        'survival_times': [],
        'total_rewards': [],
        'R_missedEthical_counts': [],
        'R_nonEthical_counts': [],
        'suboptimal_counts': [],
        'donation_full_times': []
    }

    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".npz"):
                data = np.load(os.path.join(root, filename))
                for key in metrics.keys():
                    if key in data:
                        metrics[key].append(data[key])
                    else:
                        logger.warning(f"Key {key} not found in {filename}")

    for key in metrics.keys():
        metrics[key] = np.concatenate(metrics[key], axis=0)
        print(f"Metrics for {key}: {metrics[key].shape}")

    return metrics


def perform_t_tests(ppo_metrics, lppo_metrics):
    results = {}
    for key in ppo_metrics.keys():
        if ppo_metrics[key].ndim == 2:  # If the metric is 2D
            for i in range(ppo_metrics[key].shape[1]):
                t_stat, p_value = stats.ttest_ind(ppo_metrics[key][:, i], lppo_metrics[key][:, i])
                logger.info(f"T-test for {key} column {i}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
                results[f"{key}_column_{i}"] = (t_stat, p_value)
        else:  # If the metric is 1D
            t_stat, p_value = stats.ttest_ind(ppo_metrics[key], lppo_metrics[key])
            logger.info(f"T-test for {key}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
            results[key] = (t_stat, p_value)
    return results


def main():
    ppo_directory = "src/StoreNuria/policies/PPOseed"
    lppo_directory = "src/StoreNuria/policies/LPPOSafetySeed"

    logger.info("Reading PPO metrics...")
    ppo_metrics = read_metrics(ppo_directory)

    logger.info("Reading LPPO metrics...")
    lppo_metrics = read_metrics(lppo_directory)

    logger.info("Performing t-tests between PPO and LPPO metrics...")
    t_test_results = perform_t_tests(ppo_metrics, lppo_metrics)

    # Optionally, save the t-test results to a file
    np.savez("t_test_results.npz", **t_test_results)


if __name__ == "__main__":
    main()
