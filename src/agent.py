import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch as th

ACTIONS = [0, 1, 2, 3, 4, 5, 6]


class Agent:
    # Agent class, contains the actor and critic networks, and the optimizers
    def __init__(self, actor, critic, learning_rate):
        self.actor = actor
        self.critic = critic
        # TODO: just one learning rate for both actor and critic?
        self.a_optimizer = optim.Adam(list(self.actor.parameters()), lr=learning_rate * 0.01)
        self.c_optimizer = optim.Adam(list(self.critic.parameters()), lr=learning_rate)
        self.device = next(self.actor.parameters()).device
        assert self.device == next(self.critic.parameters()).device
        self.save_dir = None

    def predict(self, x):
        return self.actor.predict(x)


# TODO: not hidden layers, make it more generic to accept any number of layers?
class DNN(nn.Module):
    # Simple feedforward neural network, for critic
    def __init__(self, in_size, out_size, hidden_size=16):
        super(DNN, self).__init__()
        self.out_size = out_size
        self.line1 = nn.Linear(in_size, hidden_size, bias=True)
        self.line2 = nn.Linear(hidden_size, out_size, bias=True)

    def forward(self, x):
        x = F.relu(self.line1(x))
        x = self.line2(x)
        return x


class PolicyDNN(nn.Module):

    # Policy network, for actor
    def __init__(self, in_size, action_size, hidden_size=16, eval_mode=False):
        super(PolicyDNN, self).__init__()
        self.line1 = nn.Linear(in_size, hidden_size, bias=True)
        self.line2 = nn.Linear(hidden_size, action_size, bias=True)
        self.eval_mode = eval_mode

    def forward(self, x):
        x = F.relu(self.line1(x))
        x = self.line2(x)
        x = F.softmax(x, dim=-1)
        return x

    def get_action(self, x):
        # Get action from the policy network
        action = Categorical(self.forward(x)).sample()
        env_action = ACTIONS[action.item()]
        return env_action, action

    def get_log_probs(self, states, actions):
        # get probability distribution over actions
        probs = self.forward(states)
        log_probs = th.log(th.gather(probs, 1, actions.to(th.int64)))
        return th.sum(log_probs, dim=1)

    def predict(self, x):
        if not self.eval_mode:
            raise ValueError("Cannot predict in training mode")
            # Check if it's a tensor
        if not isinstance(x, th.Tensor):
            x = th.tensor(x, dtype=th.float32)
        with th.no_grad():
            action = Categorical(self.forward(x)).sample()
            env_action = ACTIONS[action.item()]
        return env_action


def make_network(network_purpose, in_size, hidden_size, out_size, eval_mode=False):
    # Create a neural network
    assert network_purpose in ['policy', 'prediction']
    if network_purpose == 'policy':
        net = PolicyDNN(in_size, out_size, hidden_size, eval_mode)

    if network_purpose == 'prediction':
        net = DNN(in_size, out_size, hidden_size)
    return net
