import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch as th

from ActionSelection import SoftmaxActionSelection

ACTIONS = [0, 1, 2, 3, 4, 5, 6]


class Agent:
    def __init__(self, actor, critic, actor_lr, critic_lr, filter=None):
        self.actor = actor
        self.critic = critic
        self.a_optimizer = optim.Adam(list(self.actor.parameters()), lr=actor_lr, eps=1e-5)
        self.c_optimizer = optim.Adam(list(self.critic.parameters()), lr=critic_lr, eps=1e-5)
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


class SoftmaxActor(nn.Module):
    action_selection = SoftmaxActionSelection(ACTIONS)

    def __init__(self, o_size: int, a_size: int, h_size: int, h_layers: int, eval=False):
        super().__init__()
        self.hidden = [None] * h_layers
        self.hidden[0] = Linear(o_size, h_size, act_fn='tanh')
        for i in range(1, len(self.hidden)):
            self.hidden[i] = Linear(h_size, h_size, act_fn='tanh')

        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(h_size, a_size)
        self.eval_mode = eval

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
            action, env_action = self.select_action(np.array(prob, dtype='float64').squeeze())
            action = th.tensor(action)

        logprob = th.log(prob)
        entropy = -(prob * logprob).sum(-1)
        return env_action, action, logprob, entropy

    def predict(self, x):
        if not self.eval_mode:
            raise ValueError("Cannot predict in training mode")
        # Check if its a tensor
        if not isinstance(x, th.Tensor):
            x = th.tensor(x, dtype=th.float32)
        with th.no_grad():
            prob = self.forward(x)
            action, env_action = self.select_action(np.array(prob, dtype='float64').squeeze())
        return env_action

    def select_action(self, probs):
        if self.eval_mode:
            return SoftmaxActor.action_selection.action_selection(probs)
        else:
            return SoftmaxActor.action_selection.action_selection(probs)


class Critic(nn.Module):
    def __init__(self, o_size: int, reward_size: int, h_size: int, h_layers: int):
        super().__init__()
        self.hidden = [None] * h_layers
        self.hidden[0] = Linear(o_size, h_size, act_fn='tanh')
        for i in range(1, len(self.hidden)):
            self.hidden[i] = Linear(h_size, h_size, act_fn='tanh')
        self.hidden = nn.ModuleList(self.hidden)
        self.output = Linear(h_size, reward_size, act_fn='linear')

    def forward(self, x):
        for i in range(len(self.hidden)):
            l = self.hidden[i]
            x = F.leaky_relu(l(x))
        return self.output(x)