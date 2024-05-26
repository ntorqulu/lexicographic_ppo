from copy import deepcopy
from .misc import *


@th.jit.script
def compute_gae(b_values: Tensor, value_: Tensor, b_rewards: Tensor, b_dones: Tensor, gamma: float, gae_lambda: float):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        b_values (Tensor): Values of the states.
        value_ (Tensor): Value of the next state.
        b_rewards (Tensor): Rewards obtained.
        b_dones (Tensor): Done flags indicating end of episodes.
        gamma (float): Discount factor.
        gae_lambda (float): GAE lambda parameter.

    Returns:
        Tuple[Tensor, Tensor]: Returns and advantages tensors.
    """
    # Concatenate b_values and value_ along dimension 0
    values_ = th.cat((b_values[1:], value_.unsqueeze(0)), dim=0)
    gamma = gamma * (1 - b_dones)
    deltas = b_rewards + gamma * values_ - b_values
    advantages = th.zeros_like(b_values)
    last_gaelambda = th.zeros_like(b_values[0])
    # Compute advantages backwards in time
    for t in range(advantages.shape[0] - 1, -1, -1):
        last_gaelambda = advantages[t] = deltas[t] + gamma[t] * gae_lambda * last_gaelambda

    # Compute returns
    returns = advantages + b_values
    return returns, advantages


class Buffer:
    """
    Experience buffer for storing and sampling experiences for reinforcement learning.
    """
    def __init__(self, o_size: int, reward_size: int, size: int, max_steps: int, gamma: float, gae_lambda: float,
                 device: th.device):
        """
        Initialize the buffer.

        Args:
            o_size (int): Size of the observation space.
            reward_size (int): Size of the reward space.
            size (int): Total buffer size.
            max_steps (int): Maximum steps per episode.
            gamma (float): Discount factor.
            gae_lambda (float): GAE lambda parameter.
            device (th.device): Device to store tensors on.
        """
        self.size = size
        self.max_steps = max_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Assuming discrete actions with size 1
        a_size = 1

        # Initialize buffers
        self.b_observations = th.zeros((self.size, o_size)).to(device)
        self.b_actions = th.zeros((self.size, a_size)).to(device)
        self.b_logprobs = th.zeros(self.size, dtype=th.float32).to(device)
        self.b_rewards = th.zeros((self.size, reward_size), dtype=th.float32).to(device)
        self.b_values = deepcopy(self.b_rewards)
        self.b_dones = deepcopy(self.b_rewards)
        self.idx = 0

    def store(self, observation, action, logprob, reward, value, done):
        """
        Store a single step experience in the buffer.

        Args:
            observation (Tensor): Observation from the environment.
            action (Tensor): Action taken.
            logprob (Tensor): Log probability of the action.
            reward (Tensor): Reward received.
            value (Tensor): Value of the state.
            done (Tensor): Done flag.
        """
        self.b_observations[self.idx] = observation
        self.b_actions[self.idx] = action
        self.b_logprobs[self.idx] = logprob
        self.b_rewards[self.idx] = reward
        self.b_values[self.idx] = value
        self.b_dones[self.idx] = done
        self.idx += 1

    def compute_mc(self, value_):
        """
         Compute Monte Carlo returns and advantages using GAE.

         Args:
             value_ (Tensor): Value of the next state.
         """
        self.returns, self.advantages = compute_gae(self.b_values, value_, self.b_rewards, self.b_dones, self.gamma,
                                                    self.gae_lambda)

    def sample(self):
        """
        Sample experiences from the buffer.

        Returns:
            dict: Dictionary containing sampled experiences.
        """
        n_episodes = int(self.size / self.max_steps)

        return {
            'observations': self.b_observations.reshape((n_episodes, self.max_steps, -1)),
            'actions': self.b_actions.reshape((n_episodes, self.max_steps, -1)),
            'logprobs': self.b_logprobs.reshape((n_episodes, self.max_steps)),
            'values': self.b_values.reshape((n_episodes, self.max_steps, -1)),
            'returns': self.returns.reshape((n_episodes, self.max_steps, -1)),
            'advantages': self.advantages.reshape((n_episodes, self.max_steps, -1)),
        }

    def clear(self):
        """
        Clear the buffer.
        """
        self.idx = 0

    def detach(self):
        """
        Detach tensors from the computation graph.
        """
        self.b_observations = self.b_observations.detach()
        self.b_actions = self.b_actions.detach()
        self.b_logprobs = self.b_logprobs.detach()
        self.b_rewards = self.b_rewards.detach()
        self.b_values = self.b_values.detach()
        self.b_dones = self.b_dones.detach()
