from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingParameters:
    """
    A dataclass to store the training parameters for the MultiAgentEthicalGathering environment.
    """
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
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping
    n_epochs: int = 10  # Number of epochs
    critic_times: int = 1  # Number of critic updates per epoch
    actor_lr: float = 0.0003  # Learning rate for the actor
    critic_lr: float = 0.001  # Learning rate for the critic
    anneal_lr: bool = True  # Whether to anneal the learning rate
    anneal_entropy: bool = True  # Whether to anneal the entropy coefficient
    concavity_entropy: float = 1.8  # Concavity for entropy annealing
    clip: float = 0.2  # Clipping parameter for PPO
    ent_coef: float = 0.04  # Coefficient for the entropy term
    gae_lambda: float = 0.95  # Lambda for Generalized Advantage Estimation
    gamma: float = 0.8  # Discount factor
    v_coef: float = 0.5  # Coefficient for the value loss
    n_cpus: int = 1  # Number of CPUs to use
    cuda: bool = False  # Whether to use CUDA
    h_size: int = 128  # Size of hidden layers
    h_layers: int = 2  # Number of hidden layers
    # from multireward
    we_reward0: float = 1  # Weight for the first reward
    we_reward1: float = 10  # Weight for the second reward
    execution_class: str = "LPPO"  # Execution class, default to LPPO
    # control param
    eval_mode: bool = False  # Whether to run in evaluation mode
    beta_values: Optional[List[float]] = field(default_factory=lambda: [1, 0.5])  # Beta values for safety
    eta_value: float = 2.5  # Eta value for safety
    prioritize_performance_over_safety: bool = False  # Whether to prioritize performance over safety

    def __post_init__(self):
        """
        Post-initialization method for validation and default settings.
        """
        if self.beta_values is None:
            self.beta_values = [2, 1]
        if not (0 < self.gamma <= 1):
            raise ValueError("Gamma must be between 0 and 1.")
        if not (0 < self.gae_lambda <= 1):
            raise ValueError("GAE lambda must be between 0 and 1.")
        if not (0 < self.clip <= 1):
            raise ValueError("Clip parameter must be between 0 and 1.")
        if not self.we_reward0 >= 0:
            raise ValueError("we_reward0 must be non-negative.")
        if not self.we_reward1 >= 0:
            raise ValueError("we_reward1 must be non-negative.")
        if self.n_agents <= 0:
            raise ValueError("Number of agents must be positive.")
        if self.tot_steps <= 0:
            raise ValueError("Total number of steps must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if self.max_steps <= 0:
            raise ValueError("Max steps per episode must be positive.")
        if self.n_epochs <= 0:
            raise ValueError("Number of epochs must be positive.")
        if self.critic_times <= 0:
            raise ValueError("Number of critic updates per epoch must be positive.")
        if self.actor_lr <= 0:
            raise ValueError("Actor learning rate must be positive.")
        if self.critic_lr <= 0:
            raise ValueError("Critic learning rate must be positive.")
        if self.max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be positive.")
        if self.concavity_entropy <= 0:
            raise ValueError("Concavity entropy must be positive.")
        if self.ent_coef <= 0:
            raise ValueError("Entropy coefficient must be positive.")
        if self.v_coef <= 0:
            raise ValueError("Value coefficient must be positive.")
        if self.n_cpus <= 0:
            raise ValueError("Number of CPUs must be positive.")
        if self.h_size <= 0:
            raise ValueError("Hidden layer size must be positive.")
        if self.h_layers <= 0:
            raise ValueError("Number of hidden layers must be positive.")
