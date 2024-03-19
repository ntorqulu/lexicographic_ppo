import torch as th
from copy import deepcopy


class Buffer:
    def __init__(self, o_size: int, size: int, max_steps: int, device: th.device):
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

        # Take agents' observation space; discrete actions have size 1
        a_size = 1

        # Initialize tensors for storing observations, actions, rewards, next observations, and dones
        self.b_observations = th.zeros((self.size, o_size), device=device)
        self.b_actions = th.zeros((self.size, a_size), device=device)
        self.b_rewards = th.zeros(self.size, dtype=th.float32, device=device)
        self.next_observations = deepcopy(self.b_observations)
        self.b_dones = deepcopy(self.b_rewards)
        self.idx = 0

    def store(self, obs, action, reward, next_obs, done):
        """
        Store a transition (observation, action, reward, next observation, done) in the buffer.

        Args:
            obs (torch.Tensor): Current observation.
            action (torch.Tensor): Action taken.
            reward (float): Reward received.
            next_obs (torch.Tensor): Next observation.
            done (bool): Whether the episode is done.
        """
        self.b_observations[self.idx] = obs
        self.b_actions[self.idx] = action
        self.b_rewards[self.idx] = reward
        self.next_observations[self.idx] = next_obs
        self.b_dones[self.idx] = done
        self.idx += 1

    def sample(self):
        """
        Sample trajectories from the buffer.

        Returns:
            dict: A dictionary containing trajectories.
                'observations': Tensor of shape (n_episodes, max_steps, observation_size).
                'actions': Tensor of shape (n_episodes, max_steps, 1).
                'rewards': Tensor of shape (n_episodes, max_steps).
                'next_observations': Tensor of shape (n_episodes, max_steps, observation_size).
                'dones': Tensor of shape (n_episodes, max_steps).
        """
        n_episodes = int(self.size / self.max_steps)

        return {
            'observations': self.b_observations.reshape((n_episodes, self.max_steps, -1)),
            'actions': self.b_actions.reshape((n_episodes, self.max_steps, -1)),
            'rewards': self.b_rewards.reshape((n_episodes, self.max_steps)),
            'next_observations': self.next_observations.reshape((n_episodes, self.max_steps, -1)),
            'dones': self.b_dones.reshape((n_episodes, self.max_steps))
        }

    def detach(self):
        """
        Detach tensors from computation graph.
        """
        self.b_observations = self.b_observations.detach()
        self.b_actions = self.b_actions.detach()
        self.b_rewards = self.b_rewards.detach()
        self.next_observations = self.next_observations.detach()
        self.b_dones = self.b_dones.detach()

    def clear(self):
        """
        Clear the buffer.
        """
        self.idx = 0
