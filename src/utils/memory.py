import torch as th
from copy import deepcopy

from src.utils.misc import Tensor


@th.jit.script
def compute_gae(b_values: Tensor, value_: Tensor, b_rewards: Tensor, b_dones: Tensor, gamma: float, gae_lambda: float):
    values_ = th.cat((b_values[1:], value_))
    b_dones = b_dones.unsqueeze(-1).repeat(1, values_.shape[-1])
    gamma = gamma * (1 - b_dones)
    deltas = b_rewards + gamma * values_ - b_values
    advantages = th.zeros_like(b_values)
    last_gaelambda = th.zeros_like(b_values[0])
    for t in range(advantages.shape[0] - 1, -1, -1):
        last_gaelambda = advantages[t] = deltas[t] + gamma[t] * gae_lambda * last_gaelambda

    returns = advantages + b_values
    return returns, advantages


class Buffer:
    def __init__(self, o_size: int, size: int, max_steps: int, gamma: float, gae_lambda: float, reward_size: int, device: th.device):
        """
        Initialize the Buffer object.

        Args:
            o_size (int): Size of the observation space.
            size (int): Maximum capacity of the buffer.
            max_steps (int): Maximum number of steps per episode.
            device (torch.device): Device where tensors will be stored.
        """
        self.size = size
        self.max_steps = max_steps  # All episodes have the same length, which is 500
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_size = reward_size

        # Take agents' observation space; discrete actions have size 1
        a_size = 1

        # Initialize tensors for storing observations, actions, rewards, next observations, and dones
        self.b_observations = th.zeros((size, o_size)).to(device)
        self.b_actions = th.zeros((size, a_size)).to(device)
        self.b_log_probs = th.zeros(self.size, dtype=th.float32).to(device)
        self.b_rewards = th.zeros((size, reward_size), dtype=th.float32).to(device)
        self.b_values = deepcopy(self.b_rewards)
        self.b_dones = deepcopy(self.b_log_probs)
        self.idx = 0

    def store(self, observation, action, logprob, reward, value, done):
        """
        Store a transition (observation, action, reward, next observation, done) in the buffer.

        Args:
            obs (torch.Tensor): Current observation.
            action (torch.Tensor): Action taken.
            reward (float): Reward received.
            next_obs (torch.Tensor): Next observation.
            s_value (float): Value of the state.
            logprob (float): Log probability of the action.
            done (bool): Whether the episode is done.
        """
        self.b_observations[self.idx] = observation
        self.b_actions[self.idx] = action
        self.b_log_probs[self.idx] = logprob
        self.b_rewards[self.idx] = reward
        self.b_values[self.idx] = value
        self.b_dones[self.idx] = done
        self.idx += 1

    def sample(self):
        """
        Sample trajectories from the buffer.
"""
        n_episodes = self.size // self.max_steps
        return {
            'observations': self.b_observations.reshape(self.size, -1),
            'actions': self.b_actions.reshape(self.size, -1),
            'log_probs': self.b_log_probs.reshape(self.size),
            'values': self.b_values.reshape((self.size, -1)),
            'returns': self.returns.reshape((self.size, -1)),
            'advantages': self.advantages.reshape((self.size, -1)),

        }

    def detach(self):
        """
        Detach tensors from computation graph.
        """
        self.b_observations = self.b_observations.detach()
        self.b_actions = self.b_actions.detach()
        self.b_rewards = self.b_rewards.detach()
        self.b_log_probs = self.b_log_probs.detach()
        self.b_values = self.b_values.detach()
        self.b_dones = self.b_dones.detach()

    def clear(self):
        """
        Clear the buffer.
        """
        self.idx = 0

    def compute_mc(self, value_):
        value_ = value_.view(1, -1)
        self.returns, self.advantages = compute_gae(self.b_values, value_, self.b_rewards, self.b_dones, self.gamma,
                                                    self.gae_lambda)
