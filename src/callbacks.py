import threading
import time
import torch as th
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Callback(ABC):
    """
    Abstract base class for callbacks used in the PPO and LPPO training process.
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.ppo = None

    def initiate(self):
        """
        Method to be overloaded if access to the PPO/LPPO instance is needed during construction.
        """
        pass


class UpdateCallback(Callback):
    """
    Base class for callbacks that need to perform actions before and after PPO updates.
    """
    def __init__(self, ppo):
        """
        Initialize the update callback with the PPO/LPPO instance.

        Args:
            ppo: PPO or LPPO instance.
        """
        self.ppo = ppo
        self.update_metrics = None

    @abstractmethod
    def after_update(self):
        pass

    @abstractmethod
    def before_update(self):
        pass


class AnnealEntropy(UpdateCallback):
    """
    Callback for annealing the entropy coefficient during training.
    """
    def __init__(self, ppo, base_value=1.0, final_value=0.1, concavity=3.5, type="linear_concave"):
        """
        Initialize the entropy annealing callback.

        Args:
            ppo: PPO/LPPO instance.
            base_value (float): Initial entropy coefficient.
            final_value (float): Final entropy coefficient.
            concavity (float): Concavity parameter for the annealing curve.
            type (str): Type of annealing ("linear_concave" or "linear").
        """
        super().__init__(ppo)
        self.concavity = concavity
        self.base_value = base_value
        self.final_value = final_value
        self.type = type

    def before_update(self):
        pass

    def after_update(self):
        if self.type == "linear_concave":
            update = self.ppo.run_metrics["global_step"] / self.ppo.batch_size
            normalized_update = (update - 1.0) / self.ppo.n_updates
            complementary_update = 1 - normalized_update
            decay_step = normalized_update ** self.concavity / (
                    normalized_update ** self.concavity + complementary_update ** self.concavity)
            frac = (self.base_value - self.final_value) * (1 - decay_step) + self.final_value
            self.ppo.entropy_value = frac * self.ppo.init_args.ent_coef
        elif self.type == "linear":
            update = self.ppo.run_metrics["global_step"] / self.ppo.n_steps
            frac = 1.0 - (update - 1.0) / self.ppo.n_updates
            self.ppo.entropy_value = frac * self.ppo.init_args.ent_coef


# Printing Wrappers:
class PrintAverageReward(UpdateCallback):
    """
    Callback for printing the average reward after a specified number of episodes.
    """
    def __init__(self, ppo, n=100, show_time=False):
        """
        Initialize the print average reward callback.

        Args:
            ppo: PPO/LPPO instance.
            n (int): Number of episodes after which to print the average reward.
            show_time (bool): Whether to show the time taken for the episodes.
        """
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
    """
    Callback for logging training metrics to TensorBoard.
    """
    def __init__(self, ppo, log_dir, f=1):
        """
        Initialize the TensorBoard logging callback.

        Args:
            ppo: PPO instance.
            log_dir (str): Directory to save TensorBoard logs.
            f (int): Frequency of logging.
        """
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
