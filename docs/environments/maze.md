# Maze Environment

![MazeEnv Example](/assets/maze-env-example.gif)

## Overview

The Maze environment is a multi-agent grid world environment with a maze layout where agents navigate through to reach flags. It provides a flexible framework for testing pathfinding and navigation algorithms in gridworld settings.

## Environment Details

| Attribute             | Description                                              |
| --------------------- | -------------------------------------------------------- |
| Action Space          | `Discrete(5)` (STAY, UP, RIGHT, DOWN, LEFT)              |
| Observation Space     | `Box` of shape `(2, height, width)` or `(height, width)` |
| Observation Encoding  | See [Observation Modes](#observation-modes)              |
| Reward                | Configurable via `reward_config`                         |
| Number of Agents      | Configurable (default: 1)                                |
| Termination Condition | All agents reach flag or agent hits wall                 |
| Truncation Steps      | Configurable (default via `max_episode_steps`)           |
| Creation              | `gymnasium.make("multigrid-maze-v0")`                    |

## Observation Modes

The environment supports two observation modes:

### Tensor Mode (Default)
- **Shape**: `(2, height, width)` 
- **Description**: 2D grid with two channels
  - **Channel 0**: Static objects (walls, flags)
  - **Channel 1**: Agent positions
- **Encoding**:
  - `0`: background
  - `1`: agent
  - `2`: flag
  - `3`: wall

### Map Mode
- **Shape**: `(height, width)`
- **Description**: Single-channel 2D grid combining all objects
- **Encoding**: Same as tensor mode but flattened to single channel

## Actions

The environment uses 5 discrete actions:
- `0`: STAY - Agent remains in current position
- `1`: UP - Move up
- `2`: RIGHT - Move right  
- `3`: DOWN - Move down
- `4`: LEFT - Move left

All agents' actions must be supplied as a list of integers in the `step()` method.

## Rewards

The reward system is configurable via the `reward_config` parameter:

- **Flag Reward**: `+1.0` (default) for reaching the flag
- **Step Penalty**: `-0.01` (default) for each step taken
- **Wall Penalty**: `0.0` (default) for hitting walls

These values can be customized using the `RewardConfig` class.

## Termination

The episode terminates when:
- All agents reach their respective flags, OR
- An agent hits a wall (if wall penalty is configured as absorbing)

## Configuration

### Layout Configuration

The maze layout can be configured using the `LayoutConfig` class:

```python
layout_config = {
    "width": 10,                           # Grid width
    "height": 10,                          # Grid height  
    "flag_positions": [(9, 9)],            # List of flag positions
    "init_agent_positions": [(5, 5)],      # List of initial agent positions
    "wall_positions": [],                  # List of wall positions
}
```

### Reward Configuration

```python
reward_config = {
    "flag_reward": 1.0,                    # Reward for reaching flag
    "wall_penalty_ratio": 0.0,             # Penalty for hitting walls
    "step_penalty_ratio": 0.01,            # Penalty per step
}
```

## Usage Example

```python
import gymnasium as gym
import gym_multigrid

env = gym.make(
    "multigrid-maze-v0",
    max_episode_steps=100,
    kwargs={
        "num_agents": 1,
        "layout_config": {
            "width": 10,
            "height": 10,
            "flag_positions": [(9, 9)],
            "init_agent_positions": [(5, 5)],
            "wall_positions": [],
        },
        "reward_config": {
            "flag_reward": 1.0,
            "wall_penalty_ratio": 0.0,
            "step_penalty_ratio": 0.01,
        },
        "observation_mode": "tensor",
        "render_mode": "rgb_array",
    },
)

# Reset environment
obs, info = env.reset()

# Take actions
actions = [1]  # Move up
obs, reward, terminated, truncated, info = env.step(actions)
```

## Advanced Features

### Dynamic Layout Updates

The layout can be updated every time the environment is reset using the `reset()` method's `options` parameter:

```python
new_layout = {
    "width": 15,
    "height": 15,
    "flag_positions": [(14, 14)],
    "init_agent_positions": [(1, 1)],
    "wall_positions": [(5, 5), (6, 6), (7, 7)],
}

obs, info = env.reset(options={"layout_config": new_layout})
```

### Rendering

The environment supports two rendering modes:
- **Human**: Visual rendering for human observation
- **RGB Array**: Returns the environment state as a 3D numpy array with RGB values

```python
# Render as RGB array
rgb_array = env.render()

# Render for human viewing
env.render_mode = "human"
env.render()
```