import collections
import logging
import warnings

import torch.nn as nn

from agent import make_network, Agent
from utils.misc import *
from utils.memory import Buffer
from pathlib import Path

# The MA environment does not follow the gym SA scheme, so it raises lots of warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def _array_to_dict_tensor(agents: List[int], data: Array, device: th.device, astype: Type = th.float32) -> Dict:
    # Check if the provided device is already the current device
    is_same_device = (device == th.cuda.current_device()) if device.type == 'cuda' else (device == th.device('cpu'))

    if is_same_device:
        return {k: th.as_tensor(d, dtype=astype) for k, d in zip(agents, data)}
    else:
        return {k: th.as_tensor(d, dtype=astype).to(device) for k, d in zip(agents, data)}


class LexicoPPO:
    def __init__(self, train_params, env):
        # Create logger with the name of the class
        self.logger = logging.getLogger(self.__class__.__name__)
        # Set level of the logger to DEBUG (for now)
        self.logger.setLevel(logging.DEBUG)
        # Add a console handler to the logger if it doesn't have one
        if len(self.logger.handlers) == 0:
            print("Adding console handler")
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # Action-space and observation-space sizes
        self.env = env
        self.o_size = env.observation_space.sample().shape[0]
        self.a_size = env.action_space.n  # number of actions
        self.h_size = train_params.h_layers_size
        self.reward_size = train_params.reward_size

        # Attributes
        self.r_agents = range(train_params.n_agents)
        self.run_metrics = None
        self.update_metrics = {}
        self.sim_metrics = {}
        self.folder = train_params.folder
        self.eval_mode = False
        self.seed = train_params.seed

        # Behaviours
        self.learning_rate = train_params.learning_rate

        # Training parameters
        self.tot_steps = train_params.tot_steps  # total steps done in training
        self.max_ep_length = train_params.max_steps  # max number of steps per episode is always 500
        self.batch_size = train_params.batch_size

        # Torch init
        self.device = set_torch(n_cpus=train_params.n_cpus, cuda=False)

        # Actor-critic
        self.n_updates = None
        self.buffer = None
        self.agents, self.buffer = {}, {}

        # Initialize agents
        for k in self.r_agents:
            # Create an Agent object for each agent in the environment
            # It contains the actor and critic networks, and the optimizers
            self.agents[k] = Agent(
                actor=make_network(network_purpose='policy', in_size=self.o_size, hidden_size=self.h_size,
                                   out_size=self.a_size).to(self.device),
                critic=make_network(network_purpose='prediction', in_size=self.o_size, hidden_size=self.h_size,
                                    out_size=self.reward_size).to(self.device),
                learning_rate=train_params.learning_rate  # same learning rate for both actor and critic
            )
            # TODO: use custom buffer or ReplayBuffer?
            self.buffer[k] = Buffer(self.o_size, self.batch_size, self.max_ep_length, self.device)

        self.t = 0  # total number of frames observed
        self.discount = 0.99
        self.mu = [0.0 for _ in range(self.reward_size - 1)]
        self.j = [0.0 for _ in range(self.reward_size - 1)]
        self.recent_losses = [collections.deque(maxlen=50) for i in range(self.reward_size)]

        self.beta = list(reversed(range(1, self.reward_size + 1)))
        self.eta = [1e-3 * eta for eta in list(reversed(range(1, self.reward_size + 1)))]

        self.kl_weight = 1.0
        self.kl_target = 0.025

    def environment_reset(self):
        non_tensor_observation, info = self.env.reset()
        observation = _array_to_dict_tensor(self.r_agents, non_tensor_observation, self.device)
        return observation

    def environment_setup(self):
        if self.env is None:
            raise ValueError("Environment is not set")
        obs, info = self.env.reset()
        # log the obs and info
        self.logger.info(f"Observation: {obs}")
        self.logger.info(f"Info: {info}")
        # TODO: check environment compatible with the agent
        self.o_size = self.env.observation_space.sample().shape[0]
        self.a_size = self.env.action_space.n
        # log the action and observation space
        self.logger.info(f"Action space: {self.a_size}")
        self.logger.info(f"Observation space: {self.o_size}")

    # Always reset and set_Agents to none for now
    def train(self):
        self.environment_setup()
        # set seed for training
        set_seeds(self.seed)

        # 6 updates, each one consisting of 5 complete trajectories
        # tot_steps = 15000, batch_size = 2500
        # n_updates = 15000 // 2500 = 6
        # max_ep_length = 500
        # num_trajectories = 2500 // 500 = 5
        self.n_updates = self.tot_steps // self.batch_size
        self.logger.info(f"Total number of updates: {self.n_updates}")
        self.logger.info(f"Start training")
        for update in range(1, self.n_updates + 1):
            self.logger.info(f"Update {update}/{self.n_updates}")
            print("rollout")
            self.rollout()  # collect trajectories
            self.update()  # update the policy
        self._finish_training()

    def rollout(self):
        observation = self.environment_reset()
        print("environment reset for the rollout")
        self.logger.info(f"Observation: {observation}")
        # set actions to 0 for all agents
        action = {k: 0 for k in self.r_agents}
        env_action, ep_reward = [np.zeros(2) for _ in range(2)]
        for step in range(self.batch_size):
            with th.no_grad():
                for k in self.r_agents:
                    env_action[k], action[k] = self.agents[k].actor.get_action(observation[k])
            non_tensor_next_observation, reward, done, info = self.env.step(action)
            reward = _array_to_dict_tensor(self.r_agents, reward, self.device)
            done = _array_to_dict_tensor(self.r_agents, done, self.device)
            next_observation = _array_to_dict_tensor(self.r_agents, non_tensor_next_observation, self.device)
            # TODO: check how to store rewards to set a preference
            """
            if mode == 1:
                # set first component of the reward positive and the second negative
                reward = [[reward[k][0], -reward[k][1]] for k in self.r_agents]
            elif mode == 2:
                # set first component of the reward negative and the second positive
                reward = [[-reward[k][0], reward[k][1]] for k in self.r_agents]
            # if mode == 3, let them unchanged
            """
            # For now, let rewards unchanged
            # save the collected experience into the buffer
            if not self.eval_mode:
                for k in self.r_agents:
                    self.buffer[k].store(
                        observation[k],
                        action[k],
                        reward[k],
                        next_observation[k],
                        done[k]
                    )
            # end simulation
            if all(list(done.values())):
                observation = self.environment_reset()

    def update(self):
        for k in self.r_agents:
            # reset index of the buffer
            self.buffer[k].clear()
            # sample the buffer
            batch = self.buffer[k].sample()
            # TODO: we don't do multiple epoch for the same batch
            # for each agent, update the actor, the critic and the lagrange multipliers
            # update the actor
            self.update_actor(batch, k)
            # update the critic
            self.update_critic(batch, k)
            # update lagrange multipliers
            self.update_lagrange(batch, k)

    def _finish_training(self):
        self.logger.info(f"Training finished")
        # save the model
        self.save_experiment_data()

    def update_actor(self, batch, k):
        self.agents[k].actor.train()
        actor_loss = self.compute_loss(batch, k)
        self.agents[k].a_optimizer.zero_grad()
        actor_loss.backward()
        self.agents[k].a_optimizer.step()
        # eval

    def update_critic(self, batch, k):
        self.agents[k].critic.train()
        predictions = self.agents[k].critic(batch['observations'])
        with th.no_grad():
            rewards_expanded = batch['rewards'].unsqueeze(-1)
            dones_expanded = batch['dones'].unsqueeze(-1)
            target = rewards_expanded + (
                        self.discount * self.agents[k].critic(batch['next_observations']) * dones_expanded)
        critic_loss = nn.MSELoss()(predictions, target)
        self.agents[k].c_optimizer.zero_grad()
        critic_loss.backward()
        self.agents[k].c_optimizer.step()
        # eval

    def update_lagrange(self, batch, k):
        for i in range(self.reward_size - 1):
            self.j[i] = -th.tensor(list(self.recent_losses[i])[25:]).mean()

        # update the lagrange multipliers
        r = self.reward_size - 1
        for i in range(r):
            if len(self.recent_losses[i]) > 0:
                self.mu[i] += self.eta[i] * (self.j[i] - (-self.recent_losses[i][-1]))
            else:
                self.mu[i] += self.eta[i] * self.j[i]
            self.mu[i] = max(0.0, self.mu[i])

    def compute_loss(self, batch, k):
        first_order = []
        for i in range(self.reward_size - 1):  # remember that reward_size is 2
            # it only enters one time, for i = 0
            w = self.beta[i] + self.mu[i] * sum(self.beta[j] for j in range(i + 1, self.reward_size))
            # computes only one weight, j takes value 1
            first_order.append(w)
        first_order.append(self.beta[self.reward_size - 1])
        first_order_weights = th.tensor(first_order)

        # compute ppo loss
        with th.no_grad():
            baseline = self.agents[k].critic(batch['observations'])
            rewards_expanded = batch['rewards'].unsqueeze(-1)
            dones_expanded = batch['dones'].unsqueeze(-1)
            outcome = rewards_expanded + (
                        self.discount * self.agents[k].critic(batch['next_observations']) * (1 - dones_expanded))
            advantage = (outcome - baseline).detach()
            old_log_probs = self.agents[k].actor.get_log_probs(batch['observations'], batch['actions']).detach()
        new_log_probs = self.agents[k].actor.get_log_probs(batch['observations'], batch['actions'])
        ratios = th.exp(new_log_probs - old_log_probs).to(self.device)
        first_order_weighted_average = th.sum(first_order_weights * advantage[:, 0:self.reward_size], dim=1).to(
            self.device)
        kl_penalty = (new_log_probs - old_log_probs).to(self.device)
        loss = -(ratios * first_order_weighted_average - self.kl_weight * kl_penalty).mean().to(self.device)
        # append the loss to the recent_losses
        for i in range(self.reward_size):
            self.recent_losses[i].append(-(ratios * advantage[:, i]).mean())
        # check for nans and infs
        if th.isnan(loss) or th.isinf(loss):
            print("Loss is nan or inf")
            return "oops"
        # Update KL weight term
        if kl_penalty.mean() < self.kl_target / 1.5:
            self.kl_weight *= 0.5
        elif kl_penalty.mean() > self.kl_target * 1.5:
            self.kl_weight *= 2.0
        return loss

    def save_experiment_data(self, folder=None):
        if folder is None:
            folder = Path(self.folder) / "model"
        else:
            folder = Path(folder) / "model"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Model folder created at {folder}")
        except Exception as e:
            self.logger.error(f"Error creating model folder: {e}")
            return
        for k in self.r_agents:
            try:
                th.save(self.agents[k].actor, folder / f"actor_{k}.pth")
                th.save(self.agents[k].critic, folder / f"critic_{k}.pth")
                self.agents[k].save_dir = str(folder)
                self.logger.info(f"Model for agent {k} saved at {folder}")
            except Exception as e:
                self.logger.error(f"Error saving model for agent {k}: {e}")
                return
