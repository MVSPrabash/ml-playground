from agent import QLearningAgent
from game import Game, State, Action


class Trainer:
    def __init__(self, game: Game, agent: QLearningAgent):
        self.game = game
        self.agent = agent

    def train(self, episodes: int) -> None:
        for _ in range(episodes):
            self.train_episode()

    def train_episode(self) -> None:
        state: State = self.game.reset()
        done: bool = False

        while not done:
            action: Action = self.agent.choose_action(state)

            next_state, reward, done = self.game.step(action)

            self.agent.update(state, action, reward, next_state, done)

            state = next_state

