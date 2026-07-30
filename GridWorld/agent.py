from game import State, Action

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


    def choose_action(self, state: State) -> Action:...


    def update(self) -> None:...


    def get_q_values(self, state: State) -> dict[Action, float]:
        if state not in self.q_table:
            self.q_table[state] = {
                Action.UP: 0.0,
                Action.DOWN: 0.0,
                Action.RIGHT: 0.0,
                Action.LEFT:0.0
            }

        return self.q_table[state]

