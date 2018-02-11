import numpy as np

class DiscreteAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.epsilon = 0.1
        self.lr = 0.5
        self.discount = 0.8

        # p <=> x and k <=> vx
        self.p = 10
        self.k = 10

        self.W_init = 5

        self.W = self.W_init * np.ones((self.p+1, self.k+1, 2))

        self.S = np.zeros((self.p+1, self.k+1, 2))

        self.previous_observation = None
        self.previous_action = None
        self.previous_reward = None
        self.count_episodes = 0
        self.count_steps = 0
        self.avg_steps = 0
        self.count_victories = 0
        self.learning_period = 180

        self.action_coeff = 0.1
        self.action_multiplier = 1.6
        self.time_since_last_victory = -1
        self.action_multiplier_limit = 4

        np.set_printoptions(suppress=True, precision=1)

    def phi(self, observation):
        steps = np.array([self.S[1, 0, 0] - self.S[0, 0, 0], self.S[0, 1, 1] - self.S[0, 0, 1]])
        temp = np.exp(-np.square((np.array(observation)-self.S)))
        phi = temp[:, :, 0]*temp[:, :, 1]
        return phi

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change when starting a new
        episode of the same game, but your initial location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """

        # initialize S
        if self.S[0, 0, 0] == 0:
            for i in range(0, self.p + 1):
                for j in range(0, self.k + 1):
                    self.S[i, j, :] = [x_range[0]+(float(i*(x_range[1]-x_range[0]))/float(self.p)),
                                       -20 + (float(j*40)/float(self.k))]
        self.count_episodes += 1
        self.count_steps = 0
        self.time_since_last_victory += 1

        if self.time_since_last_victory >= self.action_multiplier_limit:
            self.action_coeff *= self.action_multiplier
            self.time_since_last_victory = 0

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        if self.count_episodes <= self.learning_period:
            rand = np.random.rand()
            if rand < self.epsilon:
                return (2*np.random.randint(0, 2)-1) * self.action_coeff
        phi = self.phi(observation)
        Q = [np.sum(self.W[:, :, a] * phi) for a in range(0, 2)]
        choice = 2*np.argmax(Q) - 1
        choice = self.action_coeff * choice
        return choice

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """

        action = action / self.action_coeff

        # skip 1st step
        if self.previous_observation is not None:
            previous_phi = self.phi(self.previous_observation)
            learning = self.lr * previous_phi / np.max(previous_phi)

            current_phi = self.phi(observation)
            current_W = [np.sum(((self.W[:, :, a] * current_phi) / np.sum(current_phi))) for a in range(0, 2)]

            # if the game finishes => update Q for current state / action
            if reward > 0:
                current_learning = self.lr * current_phi / np.max(current_phi)
                self.W[:, :, int(0.5*(action + 1))] = (1 - current_learning) * self.W[:, :, int(0.5*(action + 1))] + current_learning * reward
            # in any case, we update Q for the previous state / action
            self.W[:, :, int(0.5*(self.previous_action+1))] \
                = (1-learning) * self.W[:, :, int(0.5*(self.previous_action+1))] \
                + learning * (self.previous_reward + self.discount * np.max(current_W))
        self.previous_action = action
        self.previous_reward = reward
        self.previous_observation = observation
        self.count_steps += 1
        # print(reward)
        if reward > 0:
            self.time_since_last_victory = -1
            if self.count_episodes > self.learning_period:
                self.count_victories += 1
                self.avg_steps += self.count_steps
            # print(self.W)
            print("Solved episode %s (steps: %s)"
                      % (self.count_episodes, self.count_steps))
            if self.count_episodes == 200:
                print("Test time finished (%i / 20 games), avg step %.1f => score: %s"
                      % (self.count_victories,
                         float(self.avg_steps) / float(self.count_victories),
                         float(self.count_victories)*50-0.1*self.avg_steps))