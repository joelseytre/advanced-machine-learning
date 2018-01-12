"""
This is the machinnery that runs your agent in an environment.

This is not intented to be modified during the practical.
"""


class Runner:
    def __init__(self, environment, agent, verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose

    def step(self):
        observation = self.environment.observe()
        action = self.agent.act(observation)
        reward = self.environment.act(action)
        self.agent.reward(observation, action, reward)
        return (observation, action, reward)

    def loop(self, iterations):
        cumul_reward = 0.0
        for i in range(1, iterations+1):
            (obs, act, rew) = self.step()
            cumul_reward += rew
            if self.verbose:
                print("Simulation step {}:".format(i))
                print(" ->       observation: {}".format(obs))
                print(" ->            action: {}".format(act))
                print(" ->            reward: {}".format(rew))
                print(" -> cumulative reward: {}".format(cumul_reward))
        return cumul_reward

def iter_or_loopcall(o, count):
    if callable(o):
        return [ o() for _ in range(count) ]
    else:
        # must be iterable
        return list(iter(o))


class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """

    def __init__(self, env_maker, agent_maker, count, verbose=False):
        self.environments = iter_or_loopcall(env_maker, count)
        self.agents = iter_or_loopcall(agent_maker, count)
        assert(len(self.agents) == len(self.environments))
        self.verbose = verbose

    def step(self):
        observations = [ env.observe() for env in self.environments ]
        actions = [ agent.act(obs) for (agent, obs) in zip(self.agents, observations) ]
        rewards = [ env.act(action) for (env, action) in zip(self.environments, actions) ]
        for (agent, obs, action, reward) in zip(self.agents, observations, actions, rewards):
            agent.reward(obs, action, reward)
        return sum(rewards)/len(rewards)

    def loop(self, iterations):
        cum_avg_reward = 0.0
        for i in range(1, iterations+1):
            avg_reward = self.step()
            cum_avg_reward += avg_reward
            if self.verbose:
                print("Simulation step {}:".format(i))
                print(" ->            average reward: {}".format(avg_reward))
                print(" -> cumulative average reward: {}".format(cum_avg_reward))
        return cum_avg_reward
