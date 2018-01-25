import numpy as np
import operator

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
        self.print_score_last_100 = False

        # settings: learning rate, discount factor, initial scalar value for Q
        self.lr_increment = 0.005
        self.i_lr = 0.2
        self.f_lr = 0.0
        self.i_discount = 0.5
        self.f_discount = 0.5
        self.i_epsilon = 0.2
        self.f_epsilon = 0.1

        self.Q_init = 10
        self.last_direction = -1
        self.previous_position = [-1, -1]
        self.lr = self.i_lr
        self.discount = self.i_discount

        # for contextual bandig aka epsilon-greedy part
        self.epsilon = self.i_epsilon
        self.num_arms = 4


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
            h = hash(str(observation[0][0])+str(observation[0][1])+str(observation[1] and observation[3]>0))
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
        self.discount += (self.f_discount - self.i_discount)/900
        self.lr = self.i_lr + self.count_episodes*(self.f_lr - self.i_lr)/900
        self.epsilon += (self.f_epsilon - self.i_epsilon)/900

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        # if self.count_episodes ==1:
        #     print("\naction")
        #     print(observation)

        h = self.generate_hash(observation)
        if self.previous_position != observation[0]:
            if self.previous_action == 1:
                reverse_action = 2
            elif self.previous_action == 2:
                reverse_action = 1
            elif self.previous_action == 3:
                reverse_action = 4
            elif self.previous_action == 4:
                reverse_action = 3
            else:
                reverse_action = -1
        else:
            reverse_action = self.previous_action
        reverse_action = -1
        # exploring
        if h not in self.states_hash:
            self.states_hash += [h]
            self.Q = np.append(self.Q, [[self.Q_init] * self.num_arms], axis=0)
        state = self.states_hash.index(h)
        rand = np.random.rand()
        if state == 0:
            fire = np.random.randint(5, 9)
            if fire == reverse_action:
                fire = 13 - fire
            return fire
        if self.count_episodes <= self.learning_period:
            if rand < self.epsilon:
                move = np.random.randint(1, 5)
                if move == reverse_action:
                    move = 5 - move
                return move
        # if wumpus

        move = np.argmax(self.Q[state, :]) + 1
        # if move == reverse_action:
        #     move = 5 - move
        return move

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        h = self.generate_hash(observation)
        state = self.states_hash.index(h)
        shooted = False
        previous_shooted = False
        if action > 4:
            shooted = True
        if self.previous_action > 4:
            previous_shooted = True
        if self.count_episodes <= self.learning_period:
            # learn Q
            if self.previous_state != -1:
                # if the game finishes => update Q for current state / action
                if reward == 100 or reward == -10:
                    if not shooted:
                        self.Q[state, action - 1] = (1 - self.lr) * self.Q[state, action - 1] + self.lr * reward
                # in any case, we update Q for the previous state / action
                if not previous_shooted:
                    self.Q[self.previous_state, self.previous_action - 1] = (1 - self.lr) * self.Q[self.previous_state, self.previous_action - 1]\
                                                        + self.lr * (self.previous_reward + self.discount * np.max(self.Q[state, :]))
        else:
            self.total_reward += reward
        if self.print_score_last_100:
            if self.count_episodes == 1000 and (reward == 100 or reward == -10):
                print("%i Total reward (past 100): %i" % (self.count_episodes // 100, int(self.total_reward)))

        self.previous_state = state
        self.previous_reward = reward
        self.previous_action = action
        self.previous_position = observation[0]
        self.lr += self.lr_increment
