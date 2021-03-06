import numpy as np

np.random.seed(42)

class QAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.epsilon = 0.1
        self.lr = 0.5
        self.lr_psi = 0.5
        self.discount = 0.8

        # p <=> x and k <=> vx
        self.p = 10
        self.k = 10

        self.sigma = 1

        self.T_init = 0.5
        self.psi_init = 5

        self.T = self.T_init * np.ones((self.p+1, self.k+1))
        # self.T = self.T_init*(2*np.random.rand(self.p+1,self.k+1)-1)
        self.psi = self.psi_init * np.ones((self.p+1, self.k+1))

        self.S = np.zeros((self.p+1, self.k+1, 2))

        self.previous_observation = None
        self.previous_action = None
        self.previous_reward = None
        self.count_episodes = 0
        self.count_steps = 0
        self.avg_steps = 0
        self.count_victories = 0
        self.learning_period = 180

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

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """

        phi = self.phi(observation)
        mu = np.sum(self.T * phi)
        choice = np.random.normal(mu, self.sigma)
        return choice

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """

        # skip 1st step
        if self.previous_observation is not None:
            previous_phi = self.phi(self.previous_observation)
            current_phi = self.phi(observation)

            # in any case, we update Q for the previous state / action
            diff2 = self.lr_psi * (
                self.previous_reward
                + self.discount * np.sum(self.psi * current_phi)
                - np.sum(self.psi * previous_phi)) * previous_phi
            self.psi -= diff2
            self.T -= self.lr * (
                self.previous_reward
                + self.discount * np.sum(self.psi * current_phi)
                - np.sum(self.psi * previous_phi)) * (self.previous_action - self.T)/self.sigma
            # if the game finishes => update Q for current state / action
            if reward > 0:
                diff1 = self.lr_psi * (reward - np.sum(self.psi * current_phi)) * current_phi
                self.psi -= diff1

                self.T -= self.lr * (reward - np.sum(self.psi * current_phi)) * (action - self.T)/self.sigma
        self.previous_action = action
        self.previous_reward = reward
        self.previous_observation = observation
        self.count_steps += 1
        print(self.T)
        if reward > 0:
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