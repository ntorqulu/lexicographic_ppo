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
    tot_steps: int = 25000  # Total number of steps
    batch_size: int = 2500  # Batch size
    max_steps: int = 500  # Maximum number of steps per episode
    max_grad_norm: float = 1.0
    n_epochs: int = 10  # Number of epochs
    critic_times: int = 1  # Number of critic updates per epoch
    actor_lr: float = 0.0003
    critic_lr: float = 0.001
    clip: float = 0.2
    ent_coef: float = 0.04
    gae_lambda: float = 0.95
    gamma: float = 0.8
    v_coef: float = 0.5  # Coefficient for the value loss
    n_cpus: int = 1  # Number of cpus to use
    cuda: bool = False
    h_size: int = 128
    h_layers: int = 2
    # from multireward
    we_reward0: float = 1
    we_reward1: float = 10
    execution_class: str = "LexicoPPO"  # Default to lmorlPPO
    # control param
    eval_mode: bool = False
    critic_times: int = 1
    beta_values: list = None
    eta_values: list = None
