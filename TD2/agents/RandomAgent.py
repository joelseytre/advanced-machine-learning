import numpy as np


class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        """
        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """
        return np.random.randint(1,9)

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass
