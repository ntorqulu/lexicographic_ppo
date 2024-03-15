import collections

from agent import make_network, Agent
from src.utils.misc import *


class LexicoPPO:
    def __init__(self, train_params, env):
        self.batch_size = train_params.batch_size

        # Action-space and observation-space sizes
        self.o_size = env.observation_space.sample().shape[0]
        self.a_size = env.action_space.n # number of actions
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

        # Behaviours
        self.actor_lr = train_params.actor_lr
        self.critic_lr = train_params.critic_lr
        self.lr_scheduler = None
        self.private_callbacks = []

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
                actor=make_network(network_purpose='policy', in_size=self.o_size, hidden_size=self.h_size, out_size=self.a_size).to(self.device),
                critic=make_network(network_purpose='prediction', in_size=self.o_size, hidden_size=self.h_size, out_size=self.reward_size).to(self.device),
                learning_rate=train_params.learning_rate
            )
            self.buffer[k] = Buffer(self.o_size, self.n_steps, self.max_steps, self.gamma,
                                    self.gae_lambda, self.device)

        self.t = 0 # total number of frames observed
        self.discount = 0.99
        self.mu = [0.0 for _ in range(self.reward_size - 1)]
        self.j = [0.0 for _ in range(self.reward_size - 1)]
        self.recent_losses = [collections.deque(maxlen=50) for _ in range(self.reward_size)]

        self.beta = list(reversed(range(1, self.reward_size + 1)))
        self.eta = [1e-3 * eta for eta in list(reversed(range(1, self.reward_size + 1)))]

        self.kl_weight = 1.0
        self.kl_target = 0.025




    def train(self):
        pass