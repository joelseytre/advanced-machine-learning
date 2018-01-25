from agents.ContextualBanditAgent import ContextualBanditAgent
from agents.QAgent1 import QAgent as QAgent1
from agents.QAgent2 import QAgent as QAgent2
from agents.RandomAgent import RandomAgent

# Agent = RandomAgent
# Agent = ContextualBanditAgent
# Agent = QAgent1
Agent = QAgent2


# python main.py --ngames 1000 --niter 100 --batch 200