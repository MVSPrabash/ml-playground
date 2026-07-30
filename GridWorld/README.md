# Grid World agent
An AI agent that learns and finds a path to the goal

## Design
The project is divied into 2 modules, The game & The agent
The agent interacts with the envirnoment exclusively through the API.

## The Game : Grid World
### Game API
The Game exposes an API.
Send an action via the API and receive new state.

#### What's state?
player's current position on the board
state = (row, col)

## The Agent
### Q value for each action
1. Move = -1
2. Goal = +100
3. Wall = -5
