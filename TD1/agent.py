import numpy as np
from agents.GreedyAgent import GreedyAgent
from agents.UpdatingGreedyAgent import UpdatingGreedyAgent
from agents.SoftMaxAgent import SoftMaxAgent
from agents.UCBAgent import UCBAgent

"""
Contains the definition of the agent that will run in an
environment.
"""

# Choose which Agent is run for scoring
# Agent = GreedyAgent
Agent = UpdatingGreedyAgent
# Agent = SoftMaxAgent
# Agent = UCBAgent
