from agents.ContextualBanditAgent import ContextualBanditAgent
from agents.QAgent import QAgent
from agents.RandomAgent import RandomAgent

# Agent = RandomAgent
# Agent = ContextualBanditAgent
Agent = QAgent


# python main.py --ngames 1000 --niter 100 --batch 200