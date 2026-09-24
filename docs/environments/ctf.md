# Capture the Flag (CtF)

![2 v 2 CtF Game](/assets/ctf-env-example.gif)

## Overview

The CtF game is a simple grid-world environment with discrete state and action spaces, but includes complex adversarial dynamics.

## Game Dynamics

In our CtF game, if a pair of blue and red agents are next to each other in the blue territory, then the red agent is killed with 75\% probability (and vice versa in the red territory).
The game ends when either agent captures its opponent's flag or all blue agents are defeated.
The 12 $\times$ 12 state space is fully observable, and there are 5 discrete actions for an agent: stay, up, right, down, and left.
The field objects consist of the $m$ friendly agents, $n$ enemy agents, 2 flags, and 96 territories (48 for each agent), and 48 obstacles (4 at the center and 44 surrounding the the territories).
The observation of a state is a 12 $\times$ 12 $\times$ 3 tensor, where the first layer represents a map of the territories and obstacles, the second layer represent a map of the agents and flags, and the third layer represents status of the agents whether an agent is dead or alive.
This observation tensor is used as the input to the RL algorithms.

At every timestep, the blue agent is rewarded -0.01 to encourage it to reach the red flag faster.
The blue agent is rewarded 1 by capturing the red flag and -1 by having the blue flag captured by the red agent.
Furthermore, the blue agent is rewarded 0.25 by killing the red agent and -0.25 by being killed by the red agent.

## M vs. N CtF Game

The default CtF game.
| Attribute             | Description                                                                          |
| --------------------- | ------------------------------------------------------------------------------------ |
| Action Space          | `MultiDiscrete(5)`                                                                   |
| Observation Space     | `positional`, `map`, `flattened`, `pos_map`, `pos_map_flattened`, `tensor`           |
| Observation Encoding  | See [Observation Options](#observation-options)                                      |
| Reward                | `flag_reward`, `battle_reward_ratio`, `obstacle_penalty_ratio`, `step_penalty_ratio` |
| Number of Agents      | `m` blue (friendly, controlled) agents, `n` red (enemy, uncontrolled) agents         |
| Termination Condition | When all the blue agents are killed, or either of the flags is captured              |
| Truncation Steps      | `100`                                                                                |
| Creation              | Directly import the class                                                            |

## 1 vs. 1 CtF Game

Wrapper of M vs. N CtF game when $M = 1$ and $N = 1$.
| Attribute             | Description                                                                          |
| --------------------- | ------------------------------------------------------------------------------------ |
| Action Space          | `Discrete(5)`                                                                        |
| Observation Space     | `positional`, `map`, `flattened`, `pos_map`, `pos_map_flattened`, `tensor`           |
| Observation Encoding  | See [Observation Options](#observation-options)                                      |
| Reward                | `flag_reward`, `battle_reward_ratio`, `obstacle_penalty_ratio`, `step_penalty_ratio` |
| Termination Condition | When the blue agent is killed, or either of the flags is captured                    |
| Number of Agents      | 1 blue (friendly, controlled) agent, 1 red (enemy, uncontrolled) agent               |
| Truncation Steps      | `100`                                                                                |
| Creation              | Directly import the class                                                            |

## Observation Options

There are five options for the observation: `positional`, `map`, `flattened`, `pos_map`, `pos_map_flattened`, `tensor`.

### `positional`

The `positional` observation provides dictionary of object positions in the map ($2 \times \{\text{number of objects}\}$) in addition to binary flags of agents being alive or dead, whose size is $m+n$ in M vs N CtF game while $1$ for the 1v1 game.

### `map`

The `map` observation is an encoded map which is `np.NDArray` of the size of the map.

### `flattened`

The `flattened` observation is a flattened vector version of `positional`.

### `pos_map`

The `pos_map` observation is a dictionary observation of the agent and flag positions, encoded map of the static objects (obstacles, blue & red territories) with the dead/alive flags.

### `pos_map_flattened`

The `pos_map_flattened` observation is a flattened vector version of `pos_map`.

### `tensor`

The `tensor` observation is a feature tensor whose size is map height $\times$ map width $\times$ 3, where the first layer represents a map of the territories and obstacles, the second layer represent a map of the agents and flags, and the third layer represents status of the agents whether an agent is dead or alive.

## Configuration

The CtF environment can be configured with various parameters:

### Core Parameters

- `map_path`: Path to the map file defining the battlefield layout
- `num_blue_agents`: Number of blue (friendly) agents (default: 2)
- `num_red_agents`: Number of red (enemy) agents (default: 2)
- `battle_range`: Range within which battles can occur (default: 1)
- `territory_adv_rate`: Probability of winning battles in own territory (default: 0.75)

### Reward Parameters

- `flag_reward`: Reward for capturing enemy flag (default: 1.0)
- `battle_reward_ratio`: Reward ratio for winning battles (default: 0.25)
- `obstacle_penalty_ratio`: Penalty ratio for hitting obstacles (default: 0.0)
- `step_penalty_ratio`: Penalty ratio per step (default: 0.01)

### Enemy Policy Configuration

Enemy agents can use different policies from the built-in heuristic policies:

```python
# Single policy for all enemies
enemy_policies = "fight"  # or "capture", "patrol", "patrol_fight", "roomba", "rw"

# Different policies for each enemy
enemy_policies = ["fight", "capture", "patrol"]

# Using policy classes directly
from gym_multigrid.policy.ctf.heuristic import FightPolicy, CapturePolicy
enemy_policies = [FightPolicy, CapturePolicy]

# Policy configurations
enemy_policy_kwargs = {
    "randomness": 0.25,
    "avoided_objects": ["obstacle", "blue_agent", "red_agent"]
}

# Individual configurations for each policy
enemy_policy_kwargs = [
    {"randomness": 0.1, "avoided_objects": ["obstacle"]},
    {"randomness": 0.3, "avoided_objects": ["obstacle", "blue_agent"]},
]
```

## Usage Examples

### Basic M vs N CtF Game

```python
from gym_multigrid.envs.ctf import CtfMvNEnv

env = CtfMvNEnv(
    map_path="path/to/ctf_map.txt",
    num_blue_agents=2,
    num_red_agents=2,
    enemy_policies="fight",  # Use fighting strategy
    observation_option="tensor",
    render_mode="rgb_array",
)

# Reset environment
obs, info = env.reset()

# Take actions for blue agents
blue_actions = [1, 2]  # Actions for each blue agent
obs, reward, terminated, truncated, info = env.step(blue_actions)
```

### 1 vs 1 CtF Game

```python
from gym_multigrid.envs.ctf import Ctf1v1Env

env = Ctf1v1Env(
    map_path="path/to/ctf_map.txt",
    enemy_policies="patrol_fight",  # Use patrol-fight strategy
    enemy_policy_kwargs={"randomness": 0.1},  # Low randomness for consistent behavior
    observation_option="positional",
    flag_reward=1.0,
    battle_reward_ratio=0.25,
    step_penalty_ratio=0.01,
)

# Reset environment
obs, info = env.reset()

# Take action for single blue agent
blue_action = 3  # Single action
obs, reward, terminated, truncated, info = env.step(blue_action)
```

### Custom Configuration

```python
env = CtfMvNEnv(
    map_path="custom_map.txt",
    num_blue_agents=3,
    num_red_agents=2,
    enemy_policies=["fight", "capture"],  # Mix of fight and capture strategies
    enemy_policy_kwargs=[
        {"randomness": 0.2, "avoided_objects": ["obstacle"]},
        {"randomness": 0.1, "avoided_objects": ["obstacle", "blue_agent"]},
    ],
    battle_range=1.5,
    territory_adv_rate=0.8,
    flag_reward=2.0,
    battle_reward_ratio=0.3,
    step_penalty_ratio=0.02,
    observation_option="pos_map",
    max_steps=200,
)
```

## Map Format

CtF maps are text files defining the battlefield layout. The map should include:
- Blue and red territories
- Blue and red flag positions
- Obstacles
- Open spaces

## Available Policies

The environment includes several built-in enemy policies that can be used for red (enemy) agents. These policies are defined in the `heuristic.py` module and provide different strategic behaviors:

### Built-in Heuristic Policies

#### `"rw"` (Random Walk)
- **Class**: `RwPolicy`
- **Behavior**: Random movement in any direction
- **Use Case**: Baseline policy for comparison, unpredictable movement

#### `"fight"` (Fight Policy)
- **Class**: `FightPolicy`
- **Behavior**: Seeks out and attacks opponent agents using A* pathfinding
- **Strategy**: Takes shortest path to nearest enemy agent
- **Parameters**:
  - `randomness`: Probability of random action (default: 0.25)
  - `avoided_objects`: Objects to avoid in pathfinding (default: obstacles and other agents)

#### `"capture"` (Capture Policy)
- **Class**: `CapturePolicy`
- **Behavior**: Attempts to capture the opponent's flag using optimal pathfinding
- **Strategy**: Takes shortest path to enemy flag
- **Parameters**:
  - `randomness`: Probability of random action (default: 0.25)
  - `avoided_objects`: Objects to avoid in pathfinding

#### `"patrol"` (Patrol Policy)
- **Class**: `PatrolPolicy`
- **Behavior**: Patrols along the border between blue and red territories
- **Strategy**: Moves along territory boundaries to defend area
- **Parameters**:
  - `randomness`: Probability of random action (default: 0.25)
  - `avoided_objects`: Objects to avoid in pathfinding

#### `"patrol_fight"` (Patrol Fight Policy)
- **Class**: `PatrolFightPolicy`
- **Behavior**: Patrols territory borders, switches to attack mode when enemies enter territory
- **Strategy**: Combines patrol and fight behaviors for defensive play
- **Parameters**:
  - `randomness`: Probability of random action (default: 0.25)
  - `avoided_objects`: Objects to avoid in pathfinding

#### `"roomba"` (Roomba Policy)
- **Class**: `RoombaPolicy`
- **Behavior**: Exploration-based movement with simple behavioral rules
- **Strategy**: 
  - Scans for flags within range and moves toward them
  - Avoids enemies in enemy territory, chases in own territory
  - Random exploration with direction persistence
- **Parameters**:
  - `enemy_range`: Detection range for enemies (default: 4)
  - `flag_range`: Detection range for flags (default: 5)
  - `randomness`: Probability of changing direction (default: 0.15)

### Policy Configuration

Policies can be configured with various parameters:

```python
# Single policy for all enemies
enemy_policies = "fight"

# Multiple policies with configurations
enemy_policies = ["fight", "capture", "patrol"]
enemy_policy_kwargs = [
    {"randomness": 0.1, "avoided_objects": ["obstacle"]},
    {"randomness": 0.3, "avoided_objects": ["obstacle", "blue_agent"]},
    {"randomness": 0.2, "avoided_objects": ["obstacle", "red_agent"]},
]
```

### Common Policy Parameters

All policies support these common parameters:
- `randomness`: Float between 0-1 controlling random action probability
- `avoided_objects`: List of object types to avoid in pathfinding
- `field_map`: Environment map (automatically set by environment)
- `action_set`: Available actions (automatically set by environment)
- `random_generator`: Random number generator (automatically set by environment)

### Custom Policies

Custom policies can be implemented by extending the `CtfPolicy` base class:

```python
from gym_multigrid.policy.ctf.heuristic import CtfPolicy

class CustomPolicy(CtfPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "custom"
    
    def act(self, observation, curr_pos):
        # Implement custom behavior
        return action
```

## Rendering

The environment supports two rendering modes:
- **Human**: Visual rendering for human observation
- **RGB Array**: Returns environment state as RGB array

```python
# Render as RGB array
rgb_array = env.render()

# Render for human viewing
env.render_mode = "human"
env.render()
```
