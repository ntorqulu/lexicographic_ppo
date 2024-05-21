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


class FilterSoftmaxActionSelection(SoftmaxActionSelection):
    def __init__(self, env_action_space, threshold=0.1):
        super().__init__(env_action_space)
        self.threshold = threshold

    def action_selection(self, probs):
        higher = probs[probs > self.threshold]
        if len(higher) != len(probs):
            # Softmax it
            softmax_values = np.exp(higher) / np.exp(higher).sum(axis=0)
            cont = 0
            for i, p in enumerate(probs):
                if p > self.threshold:
                    probs[i] = softmax_values[cont]
                    cont += 1
                else:
                    probs[i] = 0
        return super().action_selection(probs)
