from game import Game
from agent import QLearningAgent
from trainer import Trainer

import pprint

def main() -> None:
    game = Game()
    agent = QLearningAgent()
    agent.epsilon = 0.3

    trainer = Trainer(game, agent)

    trainer.train(episodes = 1000)

    # pprint.pprint(agent.q_table)

    trainer.print_policy()

if __name__ == "__main__":
    main()
