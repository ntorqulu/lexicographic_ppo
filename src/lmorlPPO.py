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
from callbacks import Callback, UpdateCallback
from utils.misc import _array_to_dict_tensor
from utils.misc import *
from utils.memory import Buffer

# The MA environment does not follow the gym SA scheme, so it raises lots of warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


class lmorlPPO:
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
                o_size = model["line1.weight"].shape[1]
                a_size = model["line2.weight"].shape[0]
                actor = make_network(network_purpose='policy', in_size=o_size, hidden_size=args.h_layers_size,
                                     out_size=a_size, eval_mode=eval).to(dev)
                actor.load_state_dict(model)
                critic = make_network(network_purpose='prediction', in_size=o_size, hidden_size=args.h_size,
                                      out_size=args.reward_size, eval_mode=eval).to(dev)
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
            ch.setLevel(logging.DEBUG)
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
        self.we_reward0 = train_params.we_reward0
        self.we_reward1 = train_params.we_reward1
        self.r_agents = range(train_params.n_agents)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_name = f"{train_params.env_name}_{train_params.seed}_{timestamp}_{np.random.randint(0, 100)}"
        self.save_dir = train_params.save_dir
        self.eval_mode = False
        self.seed = train_params.seed
        self.th_deterministic = train_params.th_deterministic
        self.tot_steps = train_params.tot_steps  # total steps done in training
        self.max_ep_length = train_params.max_steps  # max number of steps per episode is always 500
        self.batch_size = train_params.batch_size
        self.n_epochs = train_params.n_epochs
        self.device = set_torch(n_cpus=train_params.n_cpus, cuda=False)

        # metrics
        self.run_metrics = None
        self.update_metrics = {}
        self.sim_metrics = {}

        # Behaviours
        self.learning_rate = train_params.learning_rate
        self.clip = train_params.clip
        self.v_coef = train_params.v_coef
        self.entropy_coef = train_params.entropy_coef
        self.max_grad_norm = train_params.max_grad_norm
        self.discount = train_params.discount
        self.gae_lambda = train_params.gae_lambda
        self.gamma = train_params.gamma

        # Actor-critic
        self.n_updates = None
        self.agents = {}
        self.buffer = {}

        # Lexico params, one for each agent
        self.discount = train_params.discount
        self.mu = [[0.0 for _ in range(self.reward_size - 1)] for _ in self.r_agents]
        self.j = [[0.0 for _ in range(self.reward_size - 1)] for _ in self.r_agents]
        self.recent_losses = [[collections.deque(maxlen=50) for _ in range(self.reward_size)] for _ in self.r_agents]

        self.beta = [list(reversed(range(1, self.reward_size + 1))) for _ in self.r_agents]
        self.eta = [[1e-3 * eta for eta in list(reversed(range(1, self.reward_size + 1)))] for _ in self.r_agents]

        #KL penalty
        self.kl_weight = train_params.kl_weight
        self.kl_target = train_params.kl_target

        # Initialize agents
        for k in self.r_agents:
            # Create an Agent object for each agent in the environment
            # It contains the actor and critic networks, and the optimizers
            self.agents[k] = Agent(
                actor=make_network(network_purpose='policy', in_size=self.o_size, hidden_size=self.h_size,
                                   out_size=self.a_size).to(self.device),
                critic=make_network(network_purpose='prediction', in_size=self.o_size, hidden_size=self.h_size,
                                    out_size=2).to(self.device),
                learning_rate=train_params.learning_rate  # same learning rate for both actor and critic
            )
            self.buffer[k] = Buffer(self.o_size, self.batch_size, self.max_ep_length, self.gamma,
                                    self.gae_lambda, 2, self.device)

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
        set_seeds(self.seed, self.th_deterministic)

        # Reset run metrics
        self.run_metrics = {
            'global_steps': 0,  # total steps done in training
            'global_episodes': 0,  # total episodes done in training
            'start_time': time.time(),  # start time of the training
            'avg_episode_rewards': collections.deque(maxlen=500),
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
        self.logger.info(f"Clip: {self.clip}")
        self.logger.info(f"---------------------------------------")
        self.logger.info(f"Value coefficient: {self.v_coef}")
        self.logger.info(f"---------------------------------------")
        self.logger.info(f"Seed: {self.seed}")

        for update in range(1, self.n_updates + 1):
            self.logger.info(f"Update {update}/{self.n_updates}")
            self.run_metrics["sim_start_time"] = time.time()
            self.rollout()  # collect trajectories
            self.update()  # update the policy
        self._finish_training()

    def rollout(self):
        sim_metrics = {"reward_per_agent": np.zeros(len(self.r_agents)),
                       "separate_rewards_per_agent": [np.zeros(2) for _ in range(len(self.r_agents))]}
        observation = self.environment_reset()  # tensor
        # set actions, logprobs and value functions to 0 for all agents
        action, logprob, s_value = [{k: 0 for k in self.r_agents} for _ in range(3)]
        env_action, ep_reward = [np.zeros(2) for _ in range(2)]
        ep_separate_rewards_per_agent = [np.zeros(2) for _ in range(len(self.r_agents))]

        for step in range(self.batch_size):
            self.run_metrics['global_steps'] += 1
            with th.no_grad():
                for k in self.r_agents:
                    # logprob of each action choosen
                    env_action[k], action[k], logprob[k], _, = self.agents[k].actor.get_action(observation[k])
                    if not self.eval_mode:
                        # s_value is a tensor of size 2, one for each reward
                        s_value[k] = self.agents[k].critic(observation[k])
            non_tensor_next_observation, reward, done, info = self.env.step(env_action)
            reward = np.array([ag.r_vec for ag in self.env.agents.values()])
            # add weights to each component of the reward (r0, r1)
            r0 = reward[:, 0] * self.we_reward0
            r1 = reward[:, 1] * self.we_reward1

            ep_reward += r0 + r1  # size 2, where first component is the reward of agent 0 and second of agent 1

            # Update separate rewards per agent
            for i, (r0_i, r1_i) in enumerate(zip(r0, r1)):
                ep_separate_rewards_per_agent[i] += np.array([r0_i, r1_i])

            reward = _array_to_dict_tensor(self.r_agents, reward, self.device)
            done = _array_to_dict_tensor(self.r_agents, done, self.device)

            # save the collected experience into the buffer
            if not self.eval_mode:
                for k in self.r_agents:
                    self.buffer[k].store(
                        observation[k],
                        action[k],
                        logprob[k],
                        reward[k],
                        s_value[k],
                        done[k]
                    )
            # end simulation
            if all(list(done.values())):
                self.run_metrics['global_episodes'] += 1
                sim_metrics["reward_per_agent"] += ep_reward
                ep_reward = np.zeros(len(self.r_agents))
                for i, r in enumerate(ep_separate_rewards_per_agent):
                    sim_metrics["separate_rewards_per_agent"][i] += r
                    ep_separate_rewards_per_agent[i] = np.zeros(2)
                observation = self.environment_reset()

        # Calculate the average reward per agent per trajectory
        sim_metrics["reward_per_agent"] /= (self.batch_size / self.max_ep_length)
        self.run_metrics["avg_episode_rewards"].append(sim_metrics["reward_per_agent"].mean())

        # Update agent performance metrics
        for k in self.r_agents:
            self.run_metrics["agent_performance"][f"Agent_{k}/Reward"] = sim_metrics["reward_per_agent"][k].mean()

            # Visualize separate rewards for each agent
            agent_rewards = sim_metrics["separate_rewards_per_agent"][k]
            agent_rewards /= (self.batch_size / self.max_ep_length)
            self.run_metrics["agent_performance"][f"Agent_{k}/Reward_0"] = agent_rewards[0].mean()
            self.run_metrics["agent_performance"][f"Agent_{k}/Reward_1"] = agent_rewards[1].mean()

        return np.array(
            [self.run_metrics["agent_performance"][f"Agent_{self.r_agents[k]}/Reward"] for k in self.r_agents])

    def update(self):
        for c in lmorlPPO.callbacks:
            if issubclass(type(c), UpdateCallback):
                c.before_update()
        update_metrics = {}

        with th.no_grad():
            for k in self.r_agents:
                value_ = self.agents[k].critic(self.environment_reset()[k])
                self.buffer[k].compute_mc(value_.reshape(-1))
        for k in self.r_agents:
            # reset index of the buffer
            self.buffer[k].clear()
            # sample the buffer
            batch = self.buffer[k].sample()
            # for each agent, update the actor, the critic and the lagrange multipliers
            # update the actor
            for epoch in range(self.n_epochs):
                actor_loss = self.update_actor(batch, k, update_metrics)
            # update the critic
            for epoch in range(self.n_epochs):
                critic_loss = self.update_critic(batch, k, update_metrics)

            loss = actor_loss + critic_loss
            update_metrics[f"Agent_{k}/Loss"] = loss.detach().cpu()
        self.update_metrics = update_metrics
        mean_loss = np.array([self.update_metrics[f"Agent_{k}/Loss"] for k in self.r_agents]).mean()
        self.run_metrics["mean_loss"].append(mean_loss)

        # Run callbacks
        for c in lmorlPPO.callbacks:
            if issubclass(type(c), UpdateCallback):
                c.after_update()
        return update_metrics

    def update_actor(self, batch, k, update_metrics):
        self.agents[k].actor.train()
        actor_loss = self.compute_loss(batch, k, update_metrics)
        self.agents[k].a_optimizer.zero_grad()
        actor_loss.backward()
        self.agents[k].a_optimizer.step()
        self.agents[k].actor.eval()
        return actor_loss

    def update_critic(self, batch, k, update_metrics):

        # 2. Train the critic
        self.agents[k].critic.train()

        # 3. Compute expected values with new critic policy
        predictions = self.agents[k].critic(batch['observations'])

        self.logger.debug(f"Shape of 'predictions': {predictions.shape}")
        self.logger.debug(f"Shape of 'observations': {batch['observations'].shape}")

        # 5. Compute the loss
        critic_loss = 0.5 * ((predictions - batch['returns']) ** 2).mean(dim=0)
        self.logger.debug(f"Shape of 'critic_loss': {critic_loss.shape}")
        # 6. Update the critic
        update_metrics[f"Agent_{k}/Critic Loss_0"] = critic_loss[0].detach()
        update_metrics[f"Agent_{k}/Critic Loss_1"] = critic_loss[1].detach()
        critic_loss = critic_loss * self.v_coef
        critic_loss = critic_loss.mean() # TODO: mean? [2]
        self.agents[k].c_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.agents[k].critic.parameters(), self.max_grad_norm)
        self.agents[k].c_optimizer.step()

        # 7. Set the critic to eval mode
        self.agents[k].critic.eval()
        return critic_loss

    def update_lagrange(self, k):
        for i in range(self.reward_size - 1):
            self.j[k][i] = -th.tensor(list(self.recent_losses[k][i][25:])).mean()
            # update the lagrange multipliers
        r = self.reward_size - 1
        for i in range(r):
            self.mu[k][i] += self.eta[k][i] * (self.j[k][i] - (-self.recent_losses[k][i][-1]))
            self.mu[k][i] = max(0.0, self.mu[k][i])

    def compute_loss(self, batch, k, update_metrics):
        # 1. Compute weights only for the first order terms -> c_i in the paper
        first_order = []
        for i in range(self.reward_size - 1):  # remember that reward_size is 2
            # it only enters one time, for i = 0
            w = self.beta[k][i] + self.mu[k][i] * sum(self.beta[k][j] for j in range(i + 1, self.reward_size))
            # computes only one weight, j takes value 1
            first_order.append(w)
        first_order.append(self.beta[k][self.reward_size - 1])
        # update_metrics[f"Agent_{k}/First Order"] = first_order
        first_order_weights = th.tensor(first_order)
        # 2. Compute updated log probabilities and ratio
        _, _, logprob, entropy = self.agents[k].actor.get_action(batch['observations'],
                                                                 batch['actions'])
        logratio = logprob - batch['log_probs']  # size [num batches, 500]
        self.logger.debug(f"Agent {k} Log Ratio: {logratio}")
        self.logger.debug(f"Agent {k} Log Ratio shape: {logratio.shape}")
        ratio = logratio.exp()  # size [num batches, 500]
        self.logger.debug(f"Agent {k} Ratio: {ratio}")
        self.logger.debug(f"Agent {k} Ratio shape: {ratio.shape}")
        update_metrics[f"Agent_{k}/Ratio"] = ratio.mean().detach()

        # 3. Compute the advantage
        mb_advantages = batch['advantages']  # Size([num batches, 500, 2])
        advantage = normalize(mb_advantages)

        first_order_weighted_advantages = th.sum(first_order_weights * advantage[:, 0:self.reward_size], dim=1)

        # 6. Compute the loss
        self.logger.debug(f"Agent {k} Ratio shape: {ratio.shape}")

        actor_nonClip_loss = first_order_weighted_advantages * ratio  # Size([2500, 2])
        self.logger.debug(f"Agent {k} actor_loss shape: {actor_nonClip_loss.shape}")
        actor_clip_loss = first_order_weighted_advantages * th.clamp(ratio, 1 - self.clip, 1 + self.clip)  # Size([2500, 2])
        self.logger.debug(f"Agent {k} actor_clip_loss shape: {actor_clip_loss.shape}")
        # Calculate clip fraction and mean between columns (r0 and r1). Mean between batches
        actor_loss = th.min(actor_nonClip_loss, actor_clip_loss).mean(dim=0)  # Size([2])
        self.logger.debug(f"Agent {k} actor_loss shape: {actor_loss.shape}")
        # mean for each reward (columns) and then mean of the means
        actor_loss_reduced = -actor_loss  # size [1]
        update_metrics[f"Agent_{k}/Actor Loss"] = actor_loss_reduced.detach()
        self.logger.debug(f"Agent {k} actor_loss_reduced shape: {actor_loss_reduced.shape}")
        for i in range(self.reward_size):
            actor_loss = ratio * advantage[:, i]
            actor_clip_loss = advantage[:, i] * th.clamp(ratio, 1 - self.clip, 1 + self.clip)
            actor_loss = th.min(actor_loss, actor_clip_loss)
            self.recent_losses[k][i].append(-actor_loss.mean())
        return actor_loss_reduced

    def _finish_training(self):
        # Log relevant data from training
        self.logger.info(f"Training finished in {time.time() - self.run_metrics['start_time']} seconds")
        self.logger.info(f"Average reward: {np.mean(self.run_metrics['avg_episode_rewards'])}")
        self.logger.info(f"Number of episodes: {self.run_metrics['global_episodes']}")
        self.logger.info(f"Number of updates: {self.n_updates}")
        # save the model
        self.save_experiment_data()

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
            lmorlPPO.callbacks = callbacks
        elif isinstance(callbacks, Callback):
            callbacks.ppo = self
            callbacks.initiate()
            lmorlPPO.callbacks.append(callbacks)
        else:
            raise TypeError("Callbacks must be a Callback subclass or a list of Callback subclasses")
