from dataclasses import dataclass


@dataclass
class TrainingParameters:
    # dataclass attributes created in __init__ and address by self
    env_name: str = "MultiAgentEthicalGathering-v1"  # Name of the environment
    tag: str = "tiny"  # Tag for the environment
    seed: int = 1  # Seed for the environment
    th_deterministic: bool = True  # Whether to use deterministic PyTorch operations
    save_dir: str = "example_data"  # Folder to save the model
    n_agents: int = 2  # Number of agents in the environment
    reward_size: int = 2  # Size of the reward vector
    tot_steps: int = 15000  # Total number of steps
    batch_size: int = 2500  # Batch size
    max_steps: int = 500  # Maximum number of steps per episode
    learning_rate: float = 1e-3  # TODO: just one learning rate for both actor and critic?
    n_cpus = 8  # Number of cpus to use
    h_layers_size: int = 16  # TODO: from IPPO, SIZE OF THE HIDDEN LAYERS
    # from lexico
    discount: float = 0.99
    kl_weight: float = 1.0
    kl_target: float = 0.025
