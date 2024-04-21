import threading
import time
import torch as th
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
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


class TensorBoardLogging(UpdateCallback):
    def __init__(self, ppo, log_dir, f=1):
        super().__init__(ppo)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.freq = f  # Frequency of logging
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(self.ppo.init_args).items()])),
        )
        self.semaphore = threading.Semaphore(1)

    def before_update(self):
        pass

    def after_update(self):
        # TODO: Add a way to log the parameters of the agents individually
        with self.semaphore:
            if self.ppo.run_metrics["global_episodes"] % self.freq == 0:
                th.set_num_threads(1)
                # Log metrics from run metrics (avg reward), update metrics, and ppo parameters (e.g. entropy, lr)
                self.writer.add_scalar("Training/Avg Reward", np.array(self.ppo.run_metrics["avg_episode_rewards"]).mean(),
                                       self.ppo.run_metrics["global_steps"])
                self.writer.add_scalar("Training/LR", self.ppo.learning_rate, self.ppo.run_metrics["global_steps"])
                self.writer.add_scalar("Training/SPS",
                                       self.ppo.tot_steps / (time.time() - self.ppo.run_metrics["start_time"]),
                                       self.ppo.run_metrics["global_steps"])

                for k, v in self.ppo.run_metrics["agent_performance"].items():
                    self.writer.add_scalar(k, v, self.ppo.run_metrics["global_steps"])

                for key, value in self.ppo.update_metrics.items():
                    if isinstance(value, list):
                        self.writer.add_scalar(key, np.array(value).mean(), self.ppo.run_metrics["global_steps"])
                    else:
                        self.writer.add_scalar(key, value, self.ppo.run_metrics["global_steps"])
