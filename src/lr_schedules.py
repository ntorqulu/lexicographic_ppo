class LRScheduler:
    """
    Base class for learning rate schedulers.
    """
    def __init__(self, ppo):
        """
        Initialize the learning rate scheduler with a PPO instance.

        Args:
            ppo: PPO/LPPO instance.
        """
        self.ppo = ppo

    def step(self):
        """
        Update the learning rates. Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class DefaultPPOAnnealing(LRScheduler):
    """
    Default learning rate scheduler that anneals the learning rates linearly.
    """
    def __init__(self, ppo):
        """
        Initialize the default PPO annealing scheduler.

        Args:
            ppo: PPO instance.
        """
        super().__init__(ppo)

    def step(self):
        """
        Update the learning rates for the actor and critic networks linearly.
        """
        update = self.ppo.run_metrics["global_step"] / self.ppo.n_steps
        frac = 1.0 - (update - 1.0) / self.ppo.n_updates

        # Update learning rates for the PPO instance
        self.ppo.actor_lr = frac * self.ppo.init_args.actor_lr
        self.ppo.critic_lr = frac * self.ppo.init_args.critic_lr

        # Update learning rates for all agents
        for ag in self.ppo.agents.values():
            ag.a_optimizer.param_groups[0]["lr"] = frac * self.ppo.init_args.actor_lr
            ag.c_optimizer.param_groups[0]["lr"] = frac * self.ppo.init_args.critic_lr


class IndependentPPOAnnealing(LRScheduler):
    """
    Independent learning rate scheduler that allows different initial learning rates for each agent.
    """
    def __init__(self, ppo, initial_lr_rates_per_agent: dict):
        """
        Initialize the independent PPO annealing scheduler.

        Args:
            ppo: PPO instance.
            initial_lr_rates_per_agent (dict): Dictionary with initial learning rates for each agent.
        """
        super().__init__(ppo)
        self.initial_lr_rates_per_agent = initial_lr_rates_per_agent

        # Validate the initial learning rates dictionary
        assert sorted(list(initial_lr_rates_per_agent.keys())) == sorted(list(self.ppo.r_agents)), \
            "initial_lr_rates_per_agent keys must match ppo.r_agents keys"
        for k in initial_lr_rates_per_agent.keys():
            assert "actor_lr" in initial_lr_rates_per_agent[k], \
                "initial_lr_rates_per_agent must have an 'actor_lr' key for each agent"
            assert "critic_lr" in initial_lr_rates_per_agent[k], \
                "initial_lr_rates_per_agent must have a 'critic_lr' key for each agent"

    def step(self):
        """
        Update the learning rates for each agent independently.
        """
        update = self.ppo.run_metrics["global_step"] / self.ppo.batch_size
        frac = 1.0 - (update - 1.0) / self.ppo.n_updates

        # Update learning rates for each agent independently
        for k in self.initial_lr_rates_per_agent.keys():
            self.ppo.agents[k].a_optimizer.param_groups[0]["lr"] = frac * self.initial_lr_rates_per_agent[k]["actor_lr"]
            self.ppo.agents[k].c_optimizer.param_groups[0]["lr"] = frac * self.initial_lr_rates_per_agent[k]["critic_lr"]