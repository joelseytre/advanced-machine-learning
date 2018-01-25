import numpy as np
from random import *

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


class QAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.print_score_last_100 = True

        # settings: learning rate, discount factor, initial scalar value for Q
        self.lr = 0.1
        self.discount = 0.6
        self.Q_init = 0
        self.previous_position = [-1, -1]
        self.more_previous_position = [-1, -1]

        # for contextual bandig aka epsilon-greedy part
        self.epsilon = 0.1
        self.num_arms = 4
        self.avg_perf = np.zeros((1, self.num_arms))
        self.arms_pulled = np.zeros((1, self.num_arms))
        self.best_arm = [1]


        # for Q-learning
        self.total_reward = 0
        self.previous_state = -1
        self.previous_action = -1
        self.previous_reward = -1000000
        self.learning_period = 900
        self.count_episodes = 0
        self.Q = self.Q_init*np.ones((1, self.num_arms))
        self.states_hash = [self.generate_hash([[0, 0], True, False, True])]

    @staticmethod
    def generate_hash(observation):
        if observation[1] and observation[3]>0:
            h = hash("kill him")
        else:
            h = hash(str(observation[0][0])+str(observation[0][1]))
        return h

    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        """
        self.count_episodes += 1
        if self.count_episodes % 100 == 1:
            self.total_reward = 0
        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        # if self.count_episodes ==1:
        #     print("\naction")
        #     print(observation)

        h = self.generate_hash(observation)

        # exploring
        if self.count_episodes <= self.learning_period:
            if h not in self.states_hash:
                # epsilon-greedy
                self.avg_perf = np.append(self.avg_perf, [[-3] * self.num_arms], axis=0)
                self.arms_pulled = np.append(self.arms_pulled, [[0] * self.num_arms], axis=0)
                self.best_arm += [-1]

                # Q-learning
                self.states_hash += [h]
                self.Q = np.append(self.Q, [[self.Q_init] * self.num_arms], axis=0)
            state = self.states_hash.index(h)
            if sum(self.arms_pulled[state, :]) < 4:
                return sum(self.arms_pulled[state, :]) + 1
            else:
                rand = np.random.rand()
                if rand < self.epsilon:
                    return np.random.randint(1, 5)
                else:
                    return np.argmax(self.Q[state, :]) + 1

        # scoring points!
        else:
            if h not in self.states_hash:
                print("WTF! State not seen in exploratory period!!!\nReturning random number...")
                print("BTW: game number: %i" % self.count_episodes)
                print("BTW: observation: %s" % str(observation))
                self.states_hash += [h]
                self.Q = np.append(self.Q, [[self.Q_init] * self.num_arms], axis=0)
                return np.random.randint(1, 5)
            else:
                state = self.states_hash.index(h)
                if state == 0:
                    return np.random.randint(5, 9)
                return np.argmax(self.Q[state, :]) + 1

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        h = self.generate_hash(observation)
        state = self.states_hash.index(h)
        if self.count_episodes <= self.learning_period:
            action = int(action)
            # update the epsilon-greedy approach:
            self.best_arm[state] = int(np.argmax(self.avg_perf[state, :])) + 1
            self.avg_perf[state, action - 1] = (self.arms_pulled[state, action - 1] * self.avg_perf[state, action - 1] + reward) \
                                           / (self.arms_pulled[state, action - 1] + 1)
            self.arms_pulled[state, action - 1] += 1

            # learn Q
            if self.previous_state != -1:
                # if the game finishes => update Q for current state / action
                if reward == 100 or reward == -10:
                    self.Q[state, action - 1] = (1 - self.lr) * self.Q[state, action - 1] + self.lr * reward
                # in any case, we update Q for the previous state / action
                self.Q[self.previous_state, self.previous_action - 1] = (1 - self.lr) * self.Q[self.previous_state, self.previous_action - 1]\
                                                        + self.lr * (self.previous_reward + self.discount * np.max(self.Q[state, :]))
        else:
            self.total_reward += reward
        # if self.print_score_last_100:
        #     if self.count_episodes == 1000 and (reward == 100 or reward == -10):
        #         print("%i Total reward (past 100): %i" % (self.count_episodes // 100, int(self.total_reward)))

        self.previous_state = state
        self.previous_reward = reward
        self.previous_action = action
        self.more_previous_position = self.previous_position
        self.previous_position = observation[0]
