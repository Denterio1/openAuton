
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.agent import PrimeAgent, AgentConfig


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.agent import PrimeAgent

if __name__ == "__main__":
    agent = PrimeAgent(config=AgentConfig(auto_save=True, verbose=True))
    agent.run("Train a small transformer for reasoning")
    print(agent.stats())