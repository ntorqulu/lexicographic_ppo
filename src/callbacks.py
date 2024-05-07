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


# Printing Wrappers:
class PrintAverageReward(UpdateCallback):

    def __init__(self, ppo, n=100, show_time=False):
        super().__init__(ppo)
        self.n = n
        self.show_time = show_time
        self.t0 = time.time()

    def after_update(self):
        if self.ppo.run_metrics["ep_count"] % self.n == 0:
            s = ""
            s += f"Average Reward: {np.array(self.ppo.run_metrics['avg_reward']).mean()}"
            if self.show_time:
                s += f"\t | SPS: {self.ppo.max_steps * self.n / (time.time() - self.t0)}"
                self.t0 = time.time()
            print(s)

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
            if self.ppo.run_metrics["ep_count"] % self.freq == 0:
                th.set_num_threads(1)
                # Log metrics from run metrics (avg reward), update metrics, and ppo parameters (e.g. entropy, lr)
                self.writer.add_scalar("Training/Avg Reward", np.array(self.ppo.run_metrics["avg_reward"]).mean(),
                                       self.ppo.run_metrics["global_step"])
                # if ppo is LPPO, add avg_episode_rewards_0 and avg_episode_rewards_1
                if hasattr(self.ppo, "avg_episode_rewards_0"):
                    self.writer.add_scalar("Training/Avg Reward 0", np.array(self.ppo.avg_episode_rewards_0).mean(),
                                               self.ppo.run_metrics["global_step"])
                    self.writer.add_scalar("Training/Avg Reward 1", np.array(self.ppo.avg_episode_rewards_1).mean(),
                                               self.ppo.run_metrics["global_step"])
                self.writer.add_scalar("Training/Entropy coef", self.ppo.entropy_value,
                                       self.ppo.run_metrics["global_step"])
                self.writer.add_scalar("Training/Actor LR", self.ppo.actor_lr, self.ppo.run_metrics["global_step"])
                self.writer.add_scalar("Training/Critic LR", self.ppo.critic_lr, self.ppo.run_metrics["global_step"])
                self.writer.add_scalar("Training/SPS",
                                       self.ppo.batch_size / (time.time() - self.ppo.run_metrics["sim_start_time"]),
                                       self.ppo.run_metrics["global_step"])
                self.writer.add_scalar("Training/Mean loss across agents",
                                       np.array(self.ppo.run_metrics["mean_loss"]).mean(),
                                       self.ppo.run_metrics["global_step"])

                for k, v in self.ppo.run_metrics["agent_performance"].items():
                    self.writer.add_scalar(k, v, self.ppo.run_metrics["global_step"])

                for key, value in self.ppo.update_metrics.items():
                    if isinstance(value, list):
                        self.writer.add_scalar(key, np.array(value).mean(), self.ppo.run_metrics["global_step"])
                    else:
                        self.writer.add_scalar(key, value, self.ppo.run_metrics["global_step"])
