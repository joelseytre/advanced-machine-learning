import numpy as np


class UpdatingGreedyAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.num_arms = 10
        self.epsilon = 0.1
        self.expl_length_per_agent = 10
        self.avg_perf = np.zeros((self.num_arms,))
        self.arms_pulled = np.zeros((self.num_arms,))
        self.best_arm = -1

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        total_arms_pulled = sum(self.arms_pulled)
        exploratory_length = self.expl_length_per_agent * self.num_arms
        if total_arms_pulled < exploratory_length:
            return int(total_arms_pulled // self.num_arms)
        else:
            rand = np.random.rand()
            if rand < self.epsilon:
                return np.random.randint(0, 9)
            else:
                return self.best_arm

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        exploratory_length = self.expl_length_per_agent * self.num_arms
        if sum(self.arms_pulled) >= exploratory_length:
            self.best_arm = np.argmax(self.avg_perf)

        self.avg_perf[action] = (self.arms_pulled[action]*self.avg_perf[action] + reward) / (self.arms_pulled[action] + 1)

        self.arms_pulled[action] += 1
        pass