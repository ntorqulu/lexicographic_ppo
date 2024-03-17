import torch as th
from copy import deepcopy


class Buffer:
    def __init__(self, o_size: int, size: int, max_steps: int, device: th.device):
        self.size = size

        # Assuming all episodes last for max_steps steps; otherwise fix sampling
        self.max_steps = max_steps

        # Take agents' observation space; discrete actions have size 1
        a_size = 1

        self.b_observations = th.zeros((self.size, o_size)).to(device)
        self.b_actions = th.zeros((self.size, a_size)).to(device)
        self.b_rewards = th.zeros(self.size, dtype=th.float32).to(device)
        self.next_observations = deepcopy(self.b_observations)
        self.b_dones = deepcopy(self.b_rewards)
        self.idx = 0

        self.device = device

    def store(self, obs, action, reward, next_obs, done):
        self.b_observations[self.idx] = obs
        self.b_actions[self.idx] = action
        self.b_rewards[self.idx] = reward
        self.next_observations[self.idx] = next_obs
        self.b_dones[self.idx] = done
        self.idx += 1

    def sample(self):
        n_episodes = int(self.size / self.max_steps)

        return {
            'observations': self.b_observations.reshape((n_episodes, self.max_steps, -1)),
            'actions': self.b_actions.reshape((n_episodes, self.max_steps, -1)),
            'rewards': self.b_rewards.reshape((n_episodes, self.max_steps)),
            'next_observations': self.next_observations.reshape((n_episodes, self.max_steps, -1)),
            'dones': self.b_dones.reshape((n_episodes, self.max_steps))
        }

    def detach(self):
        self.b_observations = self.b_observations.detach()
        self.b_actions = self.b_actions.detach()
        self.b_rewards = self.b_rewards.detach()
        self.next_observations = self.next_observations.detach()
        self.b_dones = self.b_dones.detach()

    def clear(self):
        self.idx = 0
