class LRScheduler:
    def __init__(self, ppo):
        self.ppo = ppo

    def step(self):
        raise NotImplementedError("Subclasses should implement this method.")


class DefaultPPOAnnealing(LRScheduler):
    def __init__(self, ppo):
        super().__init__(ppo)

    def step(self):
        update = self.ppo.run_metrics["global_step"] / self.ppo.n_steps
        frac = 1.0 - (update - 1.0) / self.ppo.n_updates
        self.ppo.actor_lr = frac * self.ppo.init_args.actor_lr
        self.ppo.critic_lr = frac * self.ppo.init_args.critic_lr

        for ag in self.ppo.agents.values():
            ag.a_optimizer.param_groups[0]["lr"] = frac * self.ppo.init_args.actor_lr
            ag.c_optimizer.param_groups[0]["lr"] = frac * self.ppo.init_args.critic_lr


class IndependentPPOAnnealing(LRScheduler):
    def __init__(self, ppo, initial_lr_rates_per_agent: dict):
        super().__init__(ppo)
        self.initial_lr_rates_per_agent = initial_lr_rates_per_agent
        # Check dict is valid
        assert sorted(list(initial_lr_rates_per_agent.keys())) == sorted(list(self.ppo.r_agents)), \
            "initial_lr_rates_per_agent keys must match ppo.r_agents keys"
        for k in initial_lr_rates_per_agent.keys():
            assert "actor_lr" in initial_lr_rates_per_agent[k], \
                "initial_lr_rates_per_agent must have an 'actor_lr' key for each agent"
            assert "critic_lr" in initial_lr_rates_per_agent[k], \
                "initial_lr_rates_per_agent must have a 'critic_lr' key for each agent"

    def step(self):
        update = self.ppo.run_metrics["global_step"] / self.ppo.batch_size
        frac = 1.0 - (update - 1.0) / self.ppo.n_updates

        # TODO: Think about how to log this

        for k in self.initial_lr_rates_per_agent.keys():
            self.ppo.agents[k].a_optimizer.param_groups[0]["lr"] = frac * self.initial_lr_rates_per_agent[k]["actor_lr"]
            self.ppo.agents[k].c_optimizer.param_groups[0]["lr"] = frac * self.initial_lr_rates_per_agent[k]["critic_lr"]