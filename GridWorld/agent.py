from game import State, Action
import random

class QLearningAgent:
    alpha: float
    gamma: float
    epsilon: float

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1
    ):
        self.q_table: dict[State, dict[Action, float]] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = tuple(Action)


    def choose_action(
        self,
        state: State
    ) -> Action:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        q_values: dict[Action, float] = self.get_q_values(state)
        
        max_val = max(q_values.values())
        best_actions = [act for act, val in q_values.items() if val == max_val]
        return random.choice(best_actions)


    def update(
        self,
        state: State,
        action: Action,
        reward: int,
        next_state: State,
        done: bool
    ) -> None:
        q_values = self.get_q_values(state)

        if done:
            target: float = reward
        else:
            next_q_values = self.get_q_values(next_state)
            max_future_q = max(next_q_values.values())
            target: float = reward if done else reward + self.gamma * max_future_q

        q_values[action] += self.alpha * (target - q_values[action])


    def get_q_values(self, state: State) -> dict[Action, float]:
        if state not in self.q_table:
            self.q_table[state] = {
                Action.UP: 0.0,
                Action.DOWN: 0.0,
                Action.RIGHT: 0.0,
                Action.LEFT:0.0
            }

        return self.q_table[state]

