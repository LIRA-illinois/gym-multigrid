# Rooms Environment

![RoomsEnv Example](/assets/rooms-env-example.gif)

## Overview

The Rooms environment features separate rooms where agents navigate to reach a goal while avoiding obstacles like lava and holes. This environment is ideal for testing navigation algorithms in complex, multi-room layouts with various hazards.

## Environment Details

| Attribute             | Description                                       |
| --------------------- | ------------------------------------------------- |
| Action Space          | `Discrete(5)` (STAY, LEFT, DOWN, RIGHT, UP)       |
| Observation Space     | Varies by observation mode                        |
| Observation Encoding  | See [Observation Modes](#observation-modes)       |
| Reward                | Configurable step penalty + object rewards        |
| Number of Agents      | Configurable                                      |
| Termination Condition | Agent reaches goal or steps on absorbing obstacle |
| Truncation Steps      | Configurable (default via `max_episode_steps`)    |
| Creation              | `gymnasium.make("gym_multigrid/RoomsEnv-v0")`     |

## Observation Modes

The environment supports multiple observation modes to accommodate different learning approaches:

### Positional Mode
- **Space**: `Box(shape=(4,), dtype=float64)` 
- **Description**: Agent and goal positions as normalized coordinates
- **Contents**:
  - `[0-1]`: Agent's x, y position (scaled to [0,1])
  - `[2-3]`: Goal's x, y position (scaled to [0,1])

### Positional Dictionary Mode
- **Space**: `Dict` with `"obs"` and `"desired_goal"` keys
- **Description**: Positions of all objects as dictionary
- **Contents**:
  - `"obs"`: Agent, lava, and hole positions `(N, 2)` where N = num_agents + num_lavas + num_holes
  - `"desired_goal"`: Goal position `(2,)` 

### Tensor Mode
- **Space**: `Box(shape=(width, height), dtype=int64)`
- **Description**: 2D grid where each cell contains object type encoding
- **Encoding**:
  - `0`: empty
  - `1`: wall  
  - `2`: agent
  - `3`: goal
  - `4`: lava
  - `5`: hole
  - `6`: floor
  - `7`: door
  - `8`: key
  - `9`: ball
  - `10`: box

### Vectorized Tensor Mode
- **Space**: `Box(shape=(width * height,), dtype=int64)`
- **Description**: Flattened version of tensor mode
- **Encoding**: Same as tensor mode but as 1D array

## Actions

The environment uses 5 discrete actions from the `NavigationActions` class:
- `0`: STAY - Agent remains in current position
- `1`: LEFT - Move left
- `2`: DOWN - Move down  
- `3`: RIGHT - Move right
- `4`: UP - Move up

## Rewards

The reward system consists of:
- **Step Penalty**: Configurable penalty per step (default: `-0.01`)
- **Object Rewards**: Defined per object in spawn configurations
  - **Goal**: Typically `+1.0` for reaching
  - **Lava**: Configurable penalty (e.g., `-1.0`)
  - **Holes**: Configurable penalty (e.g., `-0.5`)

## Termination

Episodes terminate when:
- Agent reaches a goal (if `absorbing=True`)
- Agent steps on lava (if `absorbing=True`)
- Agent steps on hole (if `absorbing=True`)

The `absorbing` attribute in object configurations controls whether contact ends the episode.

## Configuration

### Layout Configuration

Define the room layout using ASCII maps and spawn configurations:

```python
layout_config = {
    "field_map": [
        "#############",
        "#     #     #", 
        "#     #     #",
        "#           #",
        "#     #     #",
        "#     #     #",
        "## ####     #",
        "#     ### ###",
        "#     #     #",
        "#     #     #", 
        "#           #",
        "#     #     #",
        "#############",
    ],
    "spawn_configs": [
        {
            "agent": (9, 3),
            "goal": {"pos": (3, 9), "reward": 1.0, "absorbing": True},
            "lavas": [
                {"pos": (5, 5), "reward": -1.0, "absorbing": True},
                {"pos": (7, 7), "reward": -1.0, "absorbing": True},
            ],
            "holes": [
                {"pos": (4, 4), "reward": -0.5, "absorbing": False},
            ],
        }
    ],
}
```

### Object Configuration

Each object type supports detailed configuration:

```python
object_config = {
    "pos": (x, y),                    # Fixed position
    "reward": 1.0,                    # Reward for interaction
    "absorbing": True,                # Whether contact terminates episode
    "random_init_range": (            # Random initialization area
        (min_x, min_y),              # Top-left corner
        (width, height)               # Area dimensions
    ),
}
```

### Reward Configuration

```python
reward_config = {
    "step_penalty": 0.01,             # Penalty per step
    "sum_reward": True,               # Whether to sum rewards across agents
}
```

## Usage Example

```python
import gymnasium as gym
import gym_multigrid

env = gym.make(
    "gym_multigrid/RoomsEnv-v0",
    max_episode_steps=100,
    kwargs={
        "spawn_type": 0,
        "layout_config": {
            "field_map": [
                "#############",
                "#     #     #",
                "#     #     #", 
                "#           #",
                "#     #     #",
                "#     #     #",
                "#############",
            ],
            "spawn_configs": [
                {
                    "agent": (3, 3),
                    "goal": {"pos": (9, 3), "reward": 1.0, "absorbing": True},
                    "lavas": [
                        {"pos": (6, 3), "reward": -1.0, "absorbing": True},
                    ],
                }
            ],
        },
        "reward_config": {
            "step_penalty": 0.01,
            "sum_reward": True,
        },
        "observation_mode": "tensor",
        "render_mode": "rgb_array",
    },
)

# Reset environment
obs, info = env.reset()

# Take actions
actions = [3]  # Move right
obs, reward, terminated, truncated, info = env.step(actions)
```

## Advanced Features

### Random Object Placement

Objects can be randomly placed within specified ranges:

```python
spawn_config = {
    "agent": None,  # Will be randomly placed
    "goal": {
        "random_init_range": ((1, 1), (5, 5)),  # Random in 5x5 area
        "reward": 1.0,
        "absorbing": True,
    },
    "lavas": [
        {
            "random_init_range": ((6, 6), (3, 3)),  # Random in 3x3 area
            "reward": -1.0,
            "absorbing": True,
        }
    ],
}
```

### Multiple Layouts

The environment supports multiple spawn configurations chosen by the `spawn_type` parameter:

```python
# Multiple spawn configurations for different agents
spawn_configs = [
    {
        "agent": (1, 1),
        "goal": {"pos": (5, 5), "reward": 1.0, "absorbing": True},
    },
    {
        "agent": (10, 10), 
        "goal": {"pos": (5, 5), "reward": 1.0, "absorbing": True},  # Shared goal
    },
]
```

### Rendering

The environment supports standard rendering modes:
- **Human**: Visual rendering for human observation
- **RGB Array**: Returns environment state as RGB array

```python
# Render as RGB array
rgb_array = env.render()

# Render for human viewing  
env.render_mode = "human"
env.render()
```