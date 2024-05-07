import torch as th
from copy import deepcopy

from .misc import *


@th.jit.script
def compute_gae(b_values: Tensor, value_: Tensor, b_rewards: Tensor, b_dones: Tensor, gamma: float, gae_lambda: float):
    # b_values shape has to be (2500, 1) or (2500, 2)
    # value_ shape has to be (1,) or (2,), will be reshaped to (1, 1) or (1, 2)
    # b_rewards shape has to be (2500, 1)
    # b_dones shape has to be (2500, 1)
    # concat b_values and value_ along dim=0, resulting in shape (2500, 1)
    values_ = th.cat((b_values[1:], value_.unsqueeze(0)), dim=0)
    gamma = gamma * (1 - b_dones)
    deltas = b_rewards + gamma * values_ - b_values
    advantages = th.zeros_like(b_values)
    last_gaelambda = th.zeros_like(b_values[0])
    for t in range(advantages.shape[0] - 1, -1, -1):
        last_gaelambda = advantages[t] = deltas[t] + gamma[t] * gae_lambda * last_gaelambda

    returns = advantages + b_values
    # returns shape has to be (2500, 1) and advantages shape has to be (2500, 1)
    return returns, advantages


class Buffer:
    def __init__(self, o_size: int, reward_size: int, size: int, max_steps: int, gamma: float, gae_lambda: float,
                 device: th.device):
        self.size = size

        # Assuming all episodes last for max_steps steps; otherwise fix sampling
        self.max_steps = max_steps

        # Take agents' observation space; discrete actions have size 1
        a_size = 1

        self.b_observations = th.zeros((self.size, o_size)).to(device)
        self.b_actions = th.zeros((self.size, a_size)).to(device)
        self.b_logprobs = th.zeros(self.size, dtype=th.float32).to(device)
        self.b_rewards = th.zeros((self.size, reward_size), dtype=th.float32).to(device)
        self.b_values = deepcopy(self.b_rewards)
        self.b_dones = deepcopy(self.b_rewards)
        self.idx = 0

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.device = device

    def store(self, observation, action, logprob, reward, value, done):
        self.b_observations[self.idx] = observation
        self.b_actions[self.idx] = action
        self.b_logprobs[self.idx] = logprob
        self.b_rewards[self.idx] = reward
        self.b_values[self.idx] = value
        self.b_dones[self.idx] = done
        self.idx += 1

    def compute_mc(self, value_):
        self.returns, self.advantages = compute_gae(self.b_values, value_, self.b_rewards, self.b_dones, self.gamma,
                                                    self.gae_lambda)

    def sample(self):
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
        self.idx = 0

    def detach(self):
        self.b_observations = self.b_observations.detach()
        self.b_actions = self.b_actions.detach()
        self.b_logprobs = self.b_logprobs.detach()
        self.b_rewards = self.b_rewards.detach()
        self.b_values = self.b_values.detach()
        self.b_dones = self.b_dones.detach()
