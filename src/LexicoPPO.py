import collections

from agent import make_network, Agent
from src.utils.misc import *
from torch.distributions import Categorical


class LexicoPPO:
    def __init__(self, train_params, env):
        self.batch_size = train_params.batch_size

        # Action-space and observation-space sizes
        self.env = env
        self.o_size = env.observation_space.sample().shape[0]
        self.a_size = env.action_space.n  # number of actions
        self.h_size = train_params.h_size
        self.h_layers = train_params.h_layers
        self.reward_size = train_params.reward_size

        # Attributes
        self.r_agents = range(train_params.n_agents)
        self.run_metrics = None
        self.update_metrics = {}
        self.sim_metrics = {}
        self.folder = None
        self.eval_mode = False
        self.seed = train_params.seed

        # Behaviours
        self.actor_lr = train_params.actor_lr
        self.critic_lr = train_params.critic_lr
        self.lr_scheduler = None
        self.private_callbacks = []

        # Training parameters
        self.tot_steps = train_params.tot_steps # total steps done in training
        self.max_ep_length = train_params.max_steps # max number of steps per episode is always 500
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
                learning_rate=train_params.learning_rate
            )
            # TODO: use custom buffer or ReplayBuffer?
            self.buffer[k] = Buffer(self.o_size, self.n_steps, self.max_steps, self.gamma,
                                    self.gae_lambda, self.device)

        self.t = 0  # total number of frames observed
        self.discount = 0.99
        self.mu = [0.0 for _ in range(self.reward_size - 1)]
        self.j = [0.0 for _ in range(self.reward_size - 1)]
        self.recent_losses = [collections.deque(maxlen=50) for _ in range(self.reward_size)]

        self.beta = list(reversed(range(1, self.reward_size + 1)))
        self.eta = [1e-3 * eta for eta in list(reversed(range(1, self.reward_size + 1)))]

        self.kl_weight = 1.0
        self.kl_target = 0.025

    # TODO: act(self, state) is implemented for each actor in agent.py

    def environment_setup(self):
        # TODO: check environment compatible with the agent
        self.o_size = self.env.observation_space.sample().shape[0]
        self.a_size = self.env.action_space.n

    # Always reset and set_Agents to none for now
    def train(self):
        self.environment_setup()
        set_seeds(self.seed)

        # 6 updates, each one consisting of 5 complete trajectories
        # tot_steps = 15000, batch_size = 2500
        # n_updates = 15000 // 2500 = 6
        # max_ep_length = 500
        # num_trajectories = 2500 // 500 = 5
        self.n_updates = self.tot_steps // self.batch_size
        for update in range(1, self.n_updates + 1):
            self.rollout() # collect trajectories
            self.update() # update the policy
        self._finish_training()

    def rollout(self):
        observation = self.environment_reset()
        # set actions to 0 for all agents
        action = [0 for _ in self.r_agents]
        env_action = [0 for _ in self.r_agents]
        for step in range(self.max_ep_length):
            with th.no_grad():
                for k in self.r_agents:
                    env_action[k] = self.agents[k].actor.get_action(observation[k])
            non_tensor_next_observation, reward, done, info = self.env.step(env_action)
            next_observation = _array_to_dict_tensor(self.r_agents, non_tensor_next_observation, self.device)
            if mode == 1:
                # set first component of the reward positive and the second negative
                reward = [[reward[k][0], -reward[k][1]] for k in self.r_agents]
            else:
                # set first component of the reward negative and the second positive
                reward = [[-reward[k][0], reward[k][1]] for k in self.r_agents]

    def update(self):
        pass

    def _finish_training(self):
        pass


