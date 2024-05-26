import numpy as np


class SoftmaxActionSelection:
    """
    Base class for action selection using softmax probabilities.
    """
    def __init__(self, env_action_space):
        self.env_action_space = env_action_space
        self.action_space = range(len(env_action_space))

    def action_selection(self, probs):
        """
        Select an action based on the given probabilities using the softmax approach.

        Args:
            probs (list or np.ndarray): The list of action probabilities.

        Returns:
            tuple: The selected action index and the corresponding action from the environment's action space.
        """
        try:
            # Select an action based on the probabilities
            action = np.random.multinomial(1, probs).argmax()
        except ValueError as e:
            # Normalize the probabilities if they do not sum to 1
            probs /= probs.sum()
            action = np.random.multinomial(1, probs).argmax()
        # Ensure the selected action is within the valid action space
        assert action in self.action_space, f"Action {action} not in action space {self.action_space}"
        return action, self.env_action_space[action]


class FilterSoftmaxActionSelection(SoftmaxActionSelection):
    def __init__(self, env_action_space, threshold=0.1):
        """
        Initialize the action selector with a filtering threshold.

        Args:
            env_action_space (list): The list of actions available in the environment.
            threshold (float): The threshold for filtering low-probability actions.
        """
        super().__init__(env_action_space)
        self.threshold = threshold

    def action_selection(self, probs):
        """
        Select an action based on the given probabilities using the softmax approach
        and filter out probabilities below a certain threshold.

        Args:
            probs (list or np.ndarray): The list of action probabilities.

        Returns:
            tuple: The selected action index and the corresponding action from the environment's action space.
        """
        # Filter out probabilities below the threshold
        higher = probs[probs > self.threshold]
        if len(higher) != len(probs):
            # Apply softmax to the filtered probabilities
            softmax_values = np.exp(higher) / np.exp(higher).sum(axis=0)
            cont = 0
            for i, p in enumerate(probs):
                if p > self.threshold:
                    probs[i] = softmax_values[cont]
                    cont += 1
                else:
                    probs[i] = 0
        # Call the base class's action selection method
        return super().action_selection(probs)
