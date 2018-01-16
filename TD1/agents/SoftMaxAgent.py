import numpy as np
import math


class SoftMaxAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.tau = 0.1
        self.num_arms = 10
        self.avg_perf = np.zeros((self.num_arms,))
        self.arms_pulled = np.zeros((self.num_arms,))
        self.exponentials = np.ones((self.num_arms,))
        self.best_arm = -1

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        return np.random.choice(range(0, 10), p=self.exponentials / sum(self.exponentials))

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """

        # recalculate avg perf for the arm pulled
        self.avg_perf[action] = (self.arms_pulled[action]*self.avg_perf[action] + reward) / (self.arms_pulled[action] + 1)

        # recalculate the linked exponential
        self.exponentials[action] = math.exp(self.avg_perf[action] / self.tau)

        # add 1 to number of times pulled
        self.arms_pulled[action] += 1
        pass