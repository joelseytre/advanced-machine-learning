import numpy as np
import math


class UCBAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.num_arms = 10
        self.avg_perf = np.zeros((self.num_arms,))
        self.arms_pulled = np.zeros((self.num_arms,))
        self.B = np.zeros((self.num_arms,))

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        return np.argmax(self.B)

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        total_arms_pulled = sum(self.arms_pulled)

        ones = np.ones((self.num_arms,))

        self.B = self.avg_perf + np.sqrt(2*np.log(np.maximum(total_arms_pulled, ones)) / np.maximum(self.arms_pulled, ones))

        # recalculate avg perf for the arm pulled
        self.avg_perf[action] = (self.arms_pulled[action]*self.avg_perf[action] + reward) / (self.arms_pulled[action] + 1)

        # add 1 to number of times pulled
        self.arms_pulled[action] += 1
        pass