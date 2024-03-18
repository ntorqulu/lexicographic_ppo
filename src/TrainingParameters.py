from dataclasses import dataclass


@dataclass
class TrainingParameters:
    # dataclass attributes created in __init__ and address by self
    env_name: str = "MultiAgentEthicalGathering-v1"  # Name of the environment
    seed: int = 1  # Seed for the environment
    folder: str = "example_data"  # Folder to save the model
    n_agents: int = 2  # Number of agents in the environment
    reward_size: int = 2  # Size of the reward vector
    tot_steps: int = 15000  # Total number of steps
    batch_size: int = 2500  # Batch size
    max_steps: int = 500  # Maximum number of steps per episode
    clip: float = 0.2  # PPO clip parameter
    gamma: float = 0.8  # Discount factor
    learning_rate: float = 3e-4  # TODO: just one learning rate for both actor and critic?
    n_epochs: int = 10  # Number of epochs per update
    n_cpus = 8  # Number of cpus to use
    n_envs: int = 5
    h_layers_size: int = 16  # TODO: from IPPO, SIZE OF THE HIDDEN LAYERS
