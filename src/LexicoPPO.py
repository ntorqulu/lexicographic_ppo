import collections
import copy
import json
import logging
import os
import time
import warnings
from typing import List, Type, Dict

import torch.nn as nn

from agent import make_network, Agent
from src.callbacks import Callback, UpdateCallback
from utils.misc import *
from utils.memory import Buffer

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
    """
        LexicoPPO class implements the Lexicographic variant of Proximal Policy Optimization (PPO) algorithm for
        multi-agent reinforcement learning.
        """

    callbacks: List[Callback] = []

    @staticmethod
    def actors_from_file(folder, dev='cpu', eval=True):
        """
        Creates the actors from the folder's model, and returns them set to eval mode.
        It is assumed that the model is a SoftmaxActor from file agent.py which only has hidden layers and an output layer.
        :return:
        """
        # Load the args from the folder
        with open(folder + "/config.json", "r") as f:
            args = argparse.Namespace(**json.load(f))
            # Load the model
            agents = []
            for k in range(args.n_agents):
                model = th.load(folder + f"/actor_{k}.pth")
                o_size = model["line1.weight"].shape[1]
                a_size = model["line2.weight"].shape[0]
                actor = make_network(network_purpose='policy', in_size=o_size, hidden_size=args.h_layers_size,
                                     out_size=a_size, eval_mode=eval).to(dev)
                actor.load_state_dict(model)

                agents.append(actor)
            return agents

    @staticmethod
    def agents_from_file(folder, dev='cpu', eval=True):
        """
        Creates the agents from the folder's model, and returns them set to eval mode.
        It is assumed that the model is a SoftmaxActor from file agent.py which only has hidden layers and an output layer.
        :return:
        """
        # Load the args from the folder
        with open(folder + "/config.json", "r") as f:
            args = argparse.Namespace(**json.load(f))
            # Load the model
            agents = []
            for k in range(args.n_agents):
                model = th.load(folder + f"/actor_{k}.pth")
                o_size = model["hidden.0.weight"].shape[1]
                a_size = model["output.weight"].shape[0]
                actor = make_network(network_purpose='policy', in_size=o_size, hidden_size=args.h_layers_size,
                                     out_size=a_size, eval=eval).to(dev)
                actor.load_state_dict(model)
                critic = make_network(network_purpose='prediction', in_size=o_size, hidden_size=args.h_size,
                                      out_size=args.reward_size, eval=eval).to(dev)
                critic.load_state_dict(th.load(folder + f"/critic_{k}.pth"))

                agents.append(Agent(actor, critic, args.learning_rate))
            return agents

    def __init__(self, train_params, env):
        """
        Initialize the LexicoPPO agent with training parameters and environment.

        Args:
            train_params (argparse.Namespace): Training parameters.
            env: Multi-agent environment.

        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        if len(self.logger.handlers) == 0:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.init_args = train_params

        self.env = env
        self.tag = train_params.tag
        self.o_size = env.observation_space.sample().shape[0]
        self.a_size = env.action_space.n  # number of actions
        self.h_size = train_params.h_layers_size
        self.reward_size = train_params.reward_size
        self.r_agents = range(train_params.n_agents)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_name = f"{train_params.env_name}_{train_params.seed}_{timestamp}_{np.random.randint(0, 100)}"
        self.save_dir = train_params.save_dir
        self.eval_mode = False
        self.seed = train_params.seed
        self.tot_steps = train_params.tot_steps  # total steps done in training
        self.max_ep_length = train_params.max_steps  # max number of steps per episode is always 500
        self.batch_size = train_params.batch_size
        self.device = set_torch(n_cpus=train_params.n_cpus, cuda=False)

        # metrics
        self.run_metrics = None
        self.update_metrics = {}
        self.sim_metrics = {}

        # Behaviours
        self.learning_rate = train_params.learning_rate

        # Actor-critic
        self.n_updates = None
        self.agents = {}
        self.buffer = {}

        # Lexico params
        self.discount = 0.99
        self.mu = [0.0 for _ in range(self.reward_size - 1)]
        self.j = [0.0 for _ in range(self.reward_size - 1)]
        self.recent_losses = [collections.deque(maxlen=50) for i in range(self.reward_size)]

        self.beta = list(reversed(range(1, self.reward_size + 1)))
        self.eta = [1e-3 * eta for eta in list(reversed(range(1, self.reward_size + 1)))]

        self.kl_weight = 1.0
        self.kl_target = 0.025

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

    def environment_reset(self):
        non_tensor_observation, info = self.env.reset()
        observation = _array_to_dict_tensor(self.r_agents, non_tensor_observation, self.device)
        return observation

    def environment_setup(self):
        if self.env is None:
            raise ValueError("Environment is not set")
        self.o_size = self.env.observation_space.sample().shape[0]
        self.a_size = self.env.action_space.n

    # Always reset and set_Agents to none for now
    def train(self):
        self.environment_setup()
        # set seed for training
        set_seeds(self.seed)

        # Reset run metrics
        self.run_metrics = {
            'global_steps': 0,  # total steps done in training
            'global_episodes': 0,  # total episodes done in training
            'start_time': time.time(),  # start time of the training
            'avg_episode_rewards': collections.deque(maxlen=500),  # average episode rewards
            'agent_performance': {},
            'mean_loss': collections.deque(maxlen=500),
        }

        # 6 updates, each one consisting of 5 complete trajectories
        # tot_steps = 15000, batch_size = 2500
        # n_updates = 15000 // 2500 = 6
        # max_ep_length = 500
        # num_trajectories = 2500 // 500 = 5
        self.n_updates = self.tot_steps // self.batch_size

        # log relevant information about the training
        self.logger.info(f"Training {self.run_name}")
        self.logger.info("----------------TRAINING----------------")
        self.logger.info(f"Environment: {self.env}")
        self.logger.info(f"Number of agents: {len(self.r_agents)}")
        self.logger.info(f"Reward size: {self.reward_size}")
        self.logger.info(f"Total steps: {self.tot_steps}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Max steps per episode: {self.max_ep_length}")
        self.logger.info(f"Number of updates: {self.n_updates}")
        self.logger.info(f"Number of hidden units: {self.h_size}")
        self.logger.info(f"---------------------------------------")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Discount factor: {self.discount}")
        self.logger.info(f"---------------------------------------")
        self.logger.info(f"KL weight: {self.kl_weight}")
        self.logger.info(f"KL target: {self.kl_target}")
        self.logger.info(f"---------------------------------------")
        self.logger.info(f"Mu: {self.mu}")
        self.logger.info(f"Beta: {self.beta}")
        self.logger.info(f"Eta: {self.eta}")
        self.logger.info(f"---------------------------------------")
        self.logger.info(f"Seed: {self.seed}")

        for update in range(1, self.n_updates + 1):
            self.logger.info(f"Update {update}/{self.n_updates}")
            self.run_metrics["sim_start_time"] = time.time()
            self.rollout()  # collect trajectories
            self.update()  # update the policy
        self._finish_training()

    def rollout(self):
        sim_metrics = {"reward_per_agent": np.zeros(len(self.r_agents))}
        observation = self.environment_reset()
        # set actions to 0 for all agents
        action = {k: 0 for k in self.r_agents}
        env_action, ep_reward = [np.zeros(len(self.r_agents)) for _ in range(2)]
        for step in range(self.batch_size):
            self.run_metrics['global_steps'] += 1
            with th.no_grad():
                for k in self.r_agents:
                    action[k] = self.agents[k].actor.get_action(observation[k])
            non_tensor_next_observation, reward, done, info = self.env.step(action)
            ep_reward += reward
            reward = _array_to_dict_tensor(self.r_agents, reward, self.device)
            done = _array_to_dict_tensor(self.r_agents, done, self.device)
            next_observation = _array_to_dict_tensor(self.r_agents, non_tensor_next_observation, self.device)
            action = _array_to_dict_tensor(self.r_agents, action, self.device)
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
                self.run_metrics['global_episodes'] += 1
                sim_metrics["reward_per_agent"] += ep_reward
                ep_reward = np.zeros(len(self.r_agents))
                observation = self.environment_reset()
        # rewards are averaged over the number of trajectories done -> 5
        # average reward per agent per trajectory
        sim_metrics["reward_per_agent"] /= (self.batch_size / self.max_ep_length)
        self.run_metrics["avg_episode_rewards"].append(sim_metrics["reward_per_agent"].mean())
        for k in self.r_agents:
            self.run_metrics["agent_performance"][f"Agent_{k}/Reward"] = sim_metrics["reward_per_agent"][k].mean()
        return np.array(
            self.run_metrics["agent_performance"][f"Agent_{self.r_agents[k]}/Reward"] for k in self.r_agents)

    def update(self):
        update_metrics = {}
        for k in self.r_agents:
            # reset index of the buffer
            self.buffer[k].clear()
            # sample the buffer
            batch = self.buffer[k].sample()
            # TODO: we don't do multiple epoch for the same batch
            # for each agent, update the actor, the critic and the lagrange multipliers
            # update the actor
            self.update_actor(batch, k, update_metrics)
            # update the critic
            self.update_critic(batch, k, update_metrics)
            # update lagrange multipliers
            self.update_lagrange(k, update_metrics)

            # Run callbacks
            for c in LexicoPPO.callbacks:
                if issubclass(type(c), UpdateCallback):
                    c.after_update()

    def _finish_training(self):
        # Log relevant data from training
        self.logger.info(f"Training finished in {time.time() - self.run_metrics['start_time']} seconds")
        self.logger.info(f"Average reward: {np.mean(self.run_metrics['avg_episode_rewards'])}")
        self.logger.info(f"Number of episodes: {self.run_metrics['global_episodes']}")
        self.logger.info(f"Number of updates: {self.n_updates}")
        # save the model
        self.save_experiment_data()

    def update_actor(self, batch, k, update_metrics):
        self.agents[k].actor.train()
        actor_loss = self.compute_loss(batch, k, update_metrics)
        update_metrics[f"Agent_{k}/Actor Loss"] = actor_loss.detach()
        self.agents[k].a_optimizer.zero_grad()
        actor_loss.backward()
        self.agents[k].a_optimizer.step()
        self.agents[k].actor.eval()

    def update_critic(self, batch, k, update_metrics):
        self.agents[k].critic.train()
        predictions = self.agents[k].critic(batch['observations'])
        with th.no_grad():
            rewards_expanded = batch['rewards'].unsqueeze(-1)
            dones_expanded = batch['dones'].unsqueeze(-1)
            target = rewards_expanded + (
                    self.discount * self.agents[k].critic(batch['next_observations']) * (1 - dones_expanded))
        critic_loss = nn.MSELoss()(predictions, target).to(self.device)
        self.agents[k].c_optimizer.zero_grad()
        critic_loss.backward()
        self.agents[k].c_optimizer.step()
        self.agents[k].critic.eval()

    def update_lagrange(self, k, update_metrics):
        for i in range(self.reward_size - 1):
            self.j[i] = -th.tensor(list(self.recent_losses[i])[25:]).mean()
        # update the lagrange multipliers
        r = self.reward_size - 1
        for i in range(r):
            self.mu[i] += self.eta[i] * (self.j[i] - (-self.recent_losses[i][-1]))
            self.mu[i] = max(0.0, self.mu[i])
        update_metrics[f"Agent_{k}/Mu"] = self.mu

    def compute_loss(self, batch, k, update_metrics):
        first_order = []
        for i in range(self.reward_size - 1):  # remember that reward_size is 2
            # it only enters one time, for i = 0
            w = self.beta[i] + self.mu[i] * sum(self.beta[j] for j in range(i + 1, self.reward_size))
            # computes only one weight, j takes value 1
            first_order.append(w)
        first_order.append(self.beta[self.reward_size - 1])
        update_metrics[f"Agent_{k}/First Order"] = first_order
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

    def save_experiment_data(self, folder=None, ckpt=False):
        config = self.init_args
        # Create new folder in to save the model using tag, n_steps, tot_steps and seed as name
        if folder is None:
            folder = f"{config.save_dir}/{config.tag}/{config.tot_steps}_{config.batch_size // config.max_steps}_{config.seed}"

        # Check if folder's config file is the same as the current config
        def diff_config(path):
            if os.path.exists(path):
                tries = 0
                while tries < 5:
                    try:
                        with open(path + "/config.json", "r") as f:
                            old_config = json.load(f)
                        if old_config != vars(config):
                            return True
                        return False
                    except FileNotFoundError as e:
                        tries += 1
                        time.sleep(1)
            return False

        num = 1
        if not ckpt:
            _folder = copy.copy(folder)
            while diff_config(_folder):
                # append a number to the folder name
                _folder = folder + "_(" + str(num) + ")"
                num += 1
            folder = _folder
        else:
            folder = folder + "_ckpt"

        if not os.path.exists(folder):
            # Try-catch for concurrency issues
            tries = 0
            while tries < 5:
                try:
                    os.makedirs(folder)
                    break
                except FileExistsError as e:
                    tries += 1
                    time.sleep(1)

        print(f"Saving model in {folder}")
        self.folder = folder
        setattr(config, "saved_dir", folder)
        setattr(self, "saved_dir", folder)

        # Save the model
        for k in self.r_agents:
            th.save(self.agents[k].actor.state_dict(), folder + f"/actor_{k}.pth")
            th.save(self.agents[k].critic.state_dict(), folder + f"/critic_{k}.pth")
            self.agents[k].save_dir = folder

        # Save the args as a json file
        with open(folder + "/config.json", "w") as f:
            json.dump(vars(config), f, indent=4)
        return folder

    def addCallbacks(self, callbacks, private=False):
        if isinstance(callbacks, list):
            for c in callbacks:
                if not issubclass(type(c), Callback):
                    raise TypeError("Element of class ", type(c).__name__, " not a subclass from Callback")
                c.ppo = self
                c.initiate()
            LexicoPPO.callbacks = callbacks
        elif isinstance(callbacks, Callback):
            callbacks.ppo = self
            callbacks.initiate()
            LexicoPPO.callbacks.append(callbacks)
        else:
            raise TypeError("Callbacks must be a Callback subclass or a list of Callback subclasses")
