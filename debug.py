import sys
sys.path.insert(0, 'src')
from core.agent import PrimeAgent, AgentConfig
agent = PrimeAgent(config=AgentConfig(verbose=False, auto_save=False))
summary = agent.run('Train a small transformer for reasoning')
print(summary)