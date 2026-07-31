# Grid World agent
An AI agent that learns to navigate a grid and finds a path to the goal using Q-learning.

## Design
The project is divied into 3 modules
1. **Game** - The GridWorld environment
2. **Agent** - Learns an optimal policy using Q-table
3. **Trainer** - Orchestrates the interaction between the Game and the Agent

The agent interacts with the envirnoment exclusively through the API.
The Trainer coordinates the learning process.

---

## The Game (Grid World)

### Game API
The Game exposes an API.

```python
state = game.reset()

next_state, reward, done = game.step(action)
```

The environment receives an action and returns:
- `next_state`
- `reward`
- `done`

#### What's a state?
The State represents player's current position on the board
```python
State(row, col)
```

### Reward Function
| Event | Reward |
|-------|-------:|
| Valid move | -1 |
| Reach goal | +100 |
| Hit wall or boundary | -5 |

## The Agent
The Agent learns using a **Q-table**, which stores the expected value of taking an action at a given state.
```python
Q(state, action)
```

The Agent is responsible for:
- choosing an action
- updating Q-table from experience

---

## The Trainer
The Trainer controls the learning loop

For each episode:
1. Reset the environment.
2. Ask the Agent to choose an action.
3. Execute the action using the Game API.
4. Receive `(next_state, reward, done)`.
5. Send the result `(next_state, reward)` to the Agent so it can update its Q-table.
6. Repeat until the episode ends.
