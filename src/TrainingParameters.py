from dataclasses import dataclass


@dataclass
class TrainingParameters:
    # dataclass attributes created in __init__ and address by self
    verbose: bool = False  # If True, print the training progress
    tb_log: bool = True  # If True, log the training progress to tensorboard
    tag: str = "tiny"  # Tag for the tensorboard log
    env_name: str = "MultiAgentEthicalGathering-v1"  # Name of the environment
    seed: int = 1  # Seed for the environment
    max_steps: int = 500  # Maximum number of steps per episode
    n_agents: int = 2  # Number of agents in the environment
    reward_size: int = 2  # Size of the reward vector
    n_steps: int = 2500  # Number of steps per update
    tot_steps: int = 15000  # Total number of steps
    save_dir: str = "example_data"  # Directory to save the model
    clip: float = 0.2  # PPO clip parameter
    gamma: float = 0.8  # Discount factor
    gae_lambda: float = 0.95 # GAE lambda parameter
    ent_coef: float = 0.04 # Entropy coefficient
    v_coef: float = 0.5 # Value function coefficient
    learning_rate: float = 3e-4 # TODO: just one learning rate for both actor and critic?
    # actor_lr: float = 3e-4 # Actor learning rate
    # critic_lr: float = 1e-3 # Critic learning rate
    anneal_lr: bool = True # If True, anneal the learning rate
    n_epochs: int = 10 # Number of epochs per update
    batch_size: int = 2500 # Batch size
    parallelize: bool = False # If True, parallelize the environment
    n_envs: int = 5
    h_size: int = 128 # TODO: from IPPO, SIZE OF THE HIDDEN LAYERS
    load: str = None # Path to load the model
    anneal_entropy: bool = True # If True, anneal the entropy coefficient
    concavity_entropy: float = 3.5 # Concavity of the entropy coefficient
    clip_vloss: bool = True # If True, clip the value loss