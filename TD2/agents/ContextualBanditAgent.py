import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""

ACT_UP    = 1
ACT_DOWN  = 2
ACT_LEFT  = 3
ACT_RIGHT = 4
ACT_TORCH_UP    = 5
ACT_TORCH_DOWN  = 6
ACT_TORCH_LEFT  = 7
ACT_TORCH_RIGHT = 8


class ContextualBanditAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.num_arms = 8
        self.avg_perf = np.zeros((1, self.num_arms))
        self.arms_pulled = np.zeros((1, self.num_arms))
        self.B = np.zeros((1, self.num_arms))
        self.states_hash = [self.generate_hash([[0, 0], False, False, True])]

    def generate_hash(self, observation):
        h = hash(str(observation[0][0]) + str(observation[0][1]) + str(observation[1] and observation[3] > 0))
        return h

    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        """

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        h = self.generate_hash(observation)
        if h not in self.states_hash:
            self.states_hash += [h]
            self.avg_perf = np.append(self.avg_perf, [[0]*self.num_arms], axis=0)
            self.arms_pulled = np.append(self.arms_pulled, [[0]*self.num_arms], axis=0)
            self.B = np.append(self.B, [[0]*self.num_arms], axis=0)
        state = self.states_hash.index(h)

        return np.argmax(self.B[state, :])

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        h = self.generate_hash(observation)
        if h not in self.states_hash:
            self.states_hash += [h]
            self.avg_perf = np.append(self.avg_perf, [np.zeros(self.num_arms, )])
            self.arms_pulled = np.append(self.arms_pulled, [np.zeros(self.num_arms, )])
            self.B = np.append(self.B, [np.zeros(self.num_arms, )])
        state = self.states_hash.index(h)

        ones = np.ones((self.num_arms,))

        total_arms_pulled = np.sum(self.arms_pulled[state, :])

        self.B[state, :] = self.avg_perf[state, :] + (np.sqrt(2*np.log(np.maximum(total_arms_pulled, ones))
                                                              / np.maximum(self.arms_pulled[state, :], 0.2*ones)))

        # recalculate avg perf for the arm pulled
        self.avg_perf[state, action] = (self.arms_pulled[state, action]*self.avg_perf[state, action] + reward) \
                                       / (self.arms_pulled[state, action] + 1)

        # add 1 to number of times pulled
        self.arms_pulled[state, action] += 1
