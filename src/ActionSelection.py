import numpy as np


class SoftmaxActionSelection:
    def __init__(self, env_action_space):
        self.env_action_space = env_action_space
        self.action_space = range(len(env_action_space))

    def action_selection(self, probs):
        try:
            action = np.random.multinomial(1, probs).argmax()
        except ValueError as e:
            probs /= probs.sum()
            action = np.random.multinomial(1, probs).argmax()
        assert action in self.action_space, f"Action {action} not in action space {self.action_space}"
        return action, self.env_action_space[action]
