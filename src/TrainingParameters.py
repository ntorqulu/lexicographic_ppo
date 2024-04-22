from dataclasses import dataclass


@dataclass
class TrainingParameters:
    # dataclass attributes created in __init__ and address by self
    env_name: str = "MultiAgentEthicalGathering-v1"  # Name of the environment
    tag: str = "tiny"  # Tag for the environment
    seed: int = 1  # Seed for the environment
    th_deterministic: bool = True  # Whether to use deterministic PyTorch operations
    save_dir: str = "StoreNuria"  # Folder to save the model
    n_agents: int = 2  # Number of agents in the environment
    reward_size: int = 2  # Size of the reward vector
    tot_steps: int = 2500  # Total number of steps
    batch_size: int = 2500  # Batch size
    max_steps: int = 500  # Maximum number of steps per episode
    max_grad_norm: float = 1.0
    n_epochs: int = 10  # Number of epochs
    learning_rate: float = 0.001  # TODO: just one learning rate for both actor and critic?
    clip = 0.2
    entropy_coef: float = 0.04
    gae_lambda: float = 0.95
    gamma = 0.8
    v_coef: float = 0.5  # Coefficient for the value loss
    n_cpus = 8  # Number of cpus to use
    h_layers_size: int = 16  # TODO: from IPPO, SIZE OF THE HIDDEN LAYERS
    # from lexico
    discount: float = 0.95
    kl_weight: float = 1.0
    kl_target: float = 0.025
    we_reward0: float = 1
    we_reward1: float = 2.6
