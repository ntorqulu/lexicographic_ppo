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
        self.a_optimizer = optim.Adam(list(self.actor.parameters()), lr=learning_rate * 0.3)
        self.c_optimizer = optim.Adam(list(self.critic.parameters()), lr=learning_rate)
        self.device = next(self.actor.parameters()).device
        assert self.device == next(self.critic.parameters()).device
        self.save_dir = None

    def predict(self, x):
        return self.actor.predict(x)


def Linear(input_dim, output_dim, act_fn='leaky_relu', init_weight_uniform=True):
    """
    Creat a linear layer.

    Parameters
    ----------
    input_dim : int
        The input dimension.
    output_dim : int
        The output dimension.
    act_fn : str
        The activation function.
    init_weight_uniform : bool
        Whether uniformly sample initial weights.
    """
    gain = th.nn.init.calculate_gain(act_fn)
    fc = th.nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(fc.weight, gain=gain)
    else:
        nn.init.orthogonal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc


# TODO: not hidden layers, make it more generic to accept any number of layers?
class DNN(nn.Module):
    # Simple feedforward neural network, for critic
    def __init__(self, in_size, out_size, hidden_size=16):
        super(DNN, self).__init__()
        self.hidden = [None] * 2
        self.hidden[0] = Linear(in_size, hidden_size, act_fn='tanh')
        for i in range(1, len(self.hidden)):
            self.hidden[i] = Linear(hidden_size, hidden_size, act_fn='tanh')
        self.hidden = nn.ModuleList(self.hidden)
        self.output = Linear(hidden_size, out_size, act_fn='linear')

    def forward(self, x):
        for i in range(len(self.hidden)):
            l = self.hidden[i]
            x = F.leaky_relu(l(x))
        return self.output(x)


class PolicyDNN(nn.Module):

    # Policy network, for actor
    def __init__(self, in_size, action_size, hidden_size=16, eval_mode=False):
        super(PolicyDNN, self).__init__()
        self.hidden = [None] * 16
        self.hidden[0] = Linear(in_size, hidden_size, act_fn='tanh')
        for i in range(1, len(self.hidden)):
            self.hidden[i] = Linear(hidden_size, hidden_size, act_fn='tanh')
        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(hidden_size, action_size)
        self.eval_mode = eval_mode

    def forward(self, x):
        for i in range(len(self.hidden)):
            l = self.hidden[i]
            x = th.tanh(l(x))
        return F.softmax(self.output(x), dim=-1)

    def get_action(self, x, action=None):
        prob = self.forward(x)
        env_action, action, logprob, entropy = self.get_action_data(prob, action)
        return env_action, action, logprob.gather(-1, action.to(th.int64)).squeeze(), entropy

    def get_action_data(self, prob, action=None):
        env_action = None
        if action is None:
            action = Categorical(prob).sample()
            action = th.tensor(action)
            env_action = ACTIONS[action.item()]
        logprob = th.log(prob)
        entropy = (prob * logprob).sum(-1)
        return env_action, action, logprob, entropy

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
