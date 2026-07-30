from enum import IntEnum
from dataclasses import dataclass

class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass(frozen = True)
class State:
    row: int
    col: int


class Game:
    def __init__(self, board_file: str = 'board.txt'):
        self.board: list[list[str]] = self._load_board(board_file)

        self.rows = len(self.board)
        self.cols = len(self.board[0])

        self.start, self.goal = self._find_start_and_goal()

        self.player = self.start


    def test(self):
        print("start:", self.start, "goal:", self.goal, "player:", self.player)
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                print(self.board[row][col], end=' ')
            print()


    def reset(self) -> State:
        self.player = self.start
        return self.get_state()


    def step(self, action: Action) -> tuple[State, int, bool]:
        pos = self._move(action)

        if not (0 <= pos.row < self.rows and 0 <= pos.col < self.cols):
            return self.player, -5, False

        if self.board[pos.row][pos.col] == 'X':
            return self.player, -5, False

        # change the physical position
        if pos != self.goal:
            self.board[self.player.row][self.player.col], self.board[pos.row][pos.col] = self.board[pos.row][pos.col], self.board[self.player.row][self.player.col]
        else:
            self.board[self.player.row][self.player.col] = '.'
            self.board[pos.row][pos.col] = 'S'

        self.player = pos

        if (self.player == self.goal):
            return self.player, 100, True
        
        return self.player, -1, False


    def get_state(self) -> State:
        return self.player


    def _load_board(self, board_file) -> list[list[str]]:
        board = []

        with open(board_file, 'r') as file:
            for line in file:
                row = line.strip().split()
                board.append(row)

        return board

    def _find_start_and_goal(self) -> tuple[State, State]:
        start = None
        goal = None

        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                if (self.board[row][col] == 'S'):
                    start = State(row, col)
                elif (self.board[row][col] == 'G'):
                    goal = State(row, col)

        if start is None or goal is None:
            raise ValueError("board must contain both 'S' and 'G'")

        return start, goal


    def _move(self, action: Action) -> State:
        row = self.player.row
        col = self.player.col

        if action == Action.UP:
            row -= 1
        elif action == Action.DOWN:
            row += 1
        elif action == Action.RIGHT:
            col += 1
        elif action == Action.LEFT:
            col -= 1


        return State(row, col)

## TEST ##

game: Game = Game()
print("Grid World")

print("Startup test")
game.test()

done = False
while not done:
    userAction: str = input("> ")

    state: State
    reward: int = 0
    
    if userAction.upper() == 'UP':
        state, reward, done = game.step(Action.UP)
    elif userAction.upper() == 'DOWN':
        state, reward, done = game.step(Action.DOWN)
    elif userAction.upper() == 'RIGHT':
        state, reward, done = game.step(Action.RIGHT)
    elif userAction.upper() == 'LEFT':
        state, reward, done = game.step(Action.LEFT)
    else:
        print("Action not valid")

    print('state:', state)
    print('reward:', reward)

    # board
    for row in range(game.rows):
        for col in range(game.cols):
            print(game.board[row][col], end=' ')
        print()
    
