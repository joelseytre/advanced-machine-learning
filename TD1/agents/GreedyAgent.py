import numpy as np

class GreedyAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.num_arms = 10
        self.epsilon = 0.1
        self.avg_perf = np.zeros((self.num_arms,))
        self.current = np.zeros((self.num_arms,))
        self.exponentials = np.ones((self.num_arms,))
        self.best_arm = -1

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        return self.best_arm

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        exploratory_length = self.expl_length_per_agent * self.num_arms
        if sum(self.current) <= exploratory_length:
            self.best_arm = np.argmax(self.avg_perf)

        self.avg_perf[action] = (self.current[action]*self.avg_perf[action] + reward) / (self.current[action] + 1)

        self.current[action] += 1
        # if sum(self.current) == 100:
        #     print(self.avg_perf)
        # elif (sum(self.current) <= 150) and (sum(self.current) >= 140):
        #     print(action)
        # pass