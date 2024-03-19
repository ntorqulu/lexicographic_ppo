from abc import ABC, abstractmethod

import numpy as np


class Callback(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.ppo = None

    def initiate(self):
        """
           Method to overload in case you need self.ppo for the constructor. This is needed since you do not have
           access to qlearn instance on Callback.__init__"""
        pass


class UpdateCallback(Callback):
    def __init__(self, ppo):
        self.ppo = ppo
        self.update_metrics = None

    @abstractmethod
    def after_update(self):
        pass

    @abstractmethod
    def before_update(self):
        pass


class PrintAverageReward(UpdateCallback):

    def __init__(self, ppo, n=100):
        super().__init__(ppo)
        self.n = n

    def after_update(self):
        if self.ppo.run_metrics["global_episodes"] % self.n == 0:
            print(f"Average Reward: {np.array(self.ppo.run_metrics['avg_episode_rewards']).mean()}")

    def before_update(self):
        pass
