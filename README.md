# Lexicographic Proximal Policy Optimization for Multi-Agent Environments

This repository contains the implementation of Lexicographic Proximal Policy Optimization (LPPO) for the [MultiAgentEthicalGatheringGame](https://github.com/maymac00/MultiAgentEthicalGatheringGame/tree/master) environment. The repository is structured to facilitate easy training, evaluation, and visualization of the policies.

## Table of Contents

- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
- [Setup](#setup)
- [License](#license)

## Installation

To set up the environment and install the required packages, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ntorqulu/lexicographic_ppo.git
   cd lexicographic_ppo
    ```
2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure

The repository is structured as follows:

```
lexicographic_ppo/
│
├── src/
│   ├── __init__.py
│   ├── ActionSelection.py
│   ├── agent.py
│   ├── callbacks.py
│   ├── eval_average_rewards_boxplot.py
│   ├── eval_seed_lppo_policy.py
│   ├── eval_seed_ppo_policy.py
│   ├── eval_single_lppo_policy.py
│   ├── eval_single_ppo_policy.py
│   ├── LPPO.py
│   ├── PPO.py
│   ├── lr_schedules.py
│   ├── t-test.py
│   ├── train.py
│   ├── TrainingParameters.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── memory.py
│   │   └── misc.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.json
│
├── requirements.txt
├── README.md
└── setup.py
```

## Usage

### Training

You can train an agent using either LPPO or PPO by running the `train.py` script located in the `src` directory. You can 
pass the configuration parameters either via a JSON file or command-line arguments.

#### Command-Line Arguments

The `train.py` script accepts the following command-line arguments:
    ```
    python src/train.py --we_reward0 1 --we_reward1 10 --tot_steps 15000000 --execution_class LPPO --seed 1 --prioritize_performance_over_safety True
    ```
Where:
- `--we_reward0` and `--we_reward1` are the weights for the rewards of the two agents.
- `--tot_steps` is the total number of training steps.
- `--execution_class` is the class to be executed (LPPO or PPO).
- `--seed` is the seed for reproducibility.
- `--prioritize_performance_over_safety` is a boolean flag to prioritize performance over safety. By default, it is set to `False`.

#### Configuration File
You can also specify a configuration file to provide the parameters:
    ```
    python src/train.py --config config/config.json
    ```
The configuration file should be in JSON format and contain the following parameters:
    ```
    {
        "we_reward0": 1,
        "we_reward1": 10,
        "tot_steps": 15000000,
        "execution_class": "LPPO",
        "seed": 1,
        "prioritize_performance_over_safety": true
    }
    ```

### Evaluation
The evaluation scripts are located in the `src` directory. They all start by `eval`. 
You can evaluate trained policies using the provided scripts.
1. To evaluate the average rewards of the policies, run the `python src/eval_average_rewards_boxplot.py` script.
2. To evaluate a single LPPO policy, run the `python src/eval_single_lppo_policy.py` script. You can specify the number of simulations to run with the --n_sims argument. By default, it is set to 100 simulations. You can visualize the execution of the policy by enabling the --render argument.
3. To evaluate a single PPO policy, run the `python src/eval_single_ppo_policy.py` script. You can specify the number of simulations to run with the --n_sims argument. By default, it is set to 100 simulations. You can visualize the execution of the policy by enabling the --render argument.
4. To evaluate 20 seeds of LPPO policy, run the `python src/eval_seed_lppo_policy.py` script. You can specify the number of simulations to run for each seed with the --n_sims argument.
5. To evaluate 20 seeds of PPO policy, run the `python src/eval_seed_ppo_policy.py` script. You can specify the number of simulations to run for each seed with the --n_sims argument.
6. To perform a t-test between LPPO and PPO policies, run the `python src/t-test.py` script. It will generate a .npz file containing the results of the t-test.
7. To visualize the tensorboard logs, run the following command:
    ```
    tensorboard --logdir=src/StoreNuria/tensorboard/folder_name
    ```
    Where `folder_name` is the name of the folder containing the tensorboard logs. You can find the folder name in the `src/StoreNuria/tensorboard` directory. It can be `LPPOPerformanceSeed`, `LPPOSafetySeed` or `PPOseed`.
    Then, open a browser and navigate to `http://localhost:6006/`.

### Visualization
You can visualize the execution of the policies by enabling the `--render` argument in the evaluation scripts. The visualization will show the agents' movements and the rewards they receive.

## Setup
To install the package, you can use the `setup.py` file. This will install the package and its dependencies.
To install the package, run:
    ```
    pip install .
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



