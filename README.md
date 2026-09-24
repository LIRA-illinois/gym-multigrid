# gym-multigrid

This repo is intended to be a lightweight, multi-agent, gridworld environment. It was originally based on this [multigrid environment](https://github.com/ArnaudFickinger/gym-multigrid), but has since been heavily modified and developed beyond the scope of the original environment. Please cite this respository as well as the original repository if you use this environment in any of your projects:

```
@misc{multigrid,
  title = {Tran Research Group Gymnasium Gridworld Environment},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LIRA-illinois/gym-multigrid}},
}
```

## Getting started

The library is available on PyPI, so you can simply run `pip install gym-multigrid` to add the package to your dependency!

Alternatively, if you would like to install specific branch version of the package (for example, `dev` branch), run:

```
 pip install git+https://github.com/LIRA-illinois/gym-multigrid.git@dev
```

To modify or contribute to this project, install gym-multigrid from source.
This repo uses [poetry](https://python-poetry.org/docs/) library dependency management.
To install the dependencies for this project run:

```
git clone https://github.com/LIRA-illinois/gym-multigrid.git
cd gym-multigrid
poetry install
```

## Included environments

### Capture-the-Flag (CtF)

![2 v 2 CtF Game](./assets/ctf-env-example.gif)

Agents move around to capture enemy's flag.
The details are [here](./docs/environments/ctf.md).

### Collect Game

![Collect Game Respawn](./assets/collect-game-respawn.gif)

| Attribute             | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| Action Space          | `Discrete(4)`                                              |
| Observation Space     | `np.array` of shape `grid.width x grid.height`             |
| Observation Encoding  | `(OBJECT_IDX, COLOR_IDX, STATE)`                           |
| Reward                | `(0, 1)`                                                   |
| Number of Agents      | `2`                                                        |
| Termination Condition | `None`                                                     |
| Truncation Steps      | `50`                                                       |
| Creation              | `gymnasium.make("multigrid-collect-respawn-clustered-v0")` |

Agents move around the grid to collect objects. The object respawns in a random location after being collected.

### Rooms

![RoomsEnv Example](./assets/rooms-env-example.gif)

Agents move around the grid to reach a goal while avoiding lava and holes.
The details are [here](./docs/environments/rooms.md).

### Prey Predator

![Prey Predator Example](./assets/prey-pred-env-example.gif)

Predators chase and catch preys moving within designated areas.
The details are [here](./docs/environments/prey-predator.md).

### Maze

![MazeEnv Example](./assets/maze-env-example.gif)
Agents navigate through a maze to reach a goal.
The details are [here](./docs/environments/maze.md).

### Wildfire

![WildfireEnv Example](./assets/wildfire-env-example.gif)

| Attribute               | Description                                                         |
| ----------------------- | ------------------------------------------------------------------- |
| Actions                 | `Discrete`                                                          |
| Agent Action Space      | `Discrete(5)`                                                       |
| Observations            | `Discrete`                                                          |
| Observability           | `Fully observable`                                                  |
| Agent Observation Space | `Box([0,...],[1,...],(shape depends on number of agents,),float64)` |
| States                  | `Discrete`                                                          |
| State Space             | `Box([0,...],[1,...],(shape depends on number of agents,),float64)` |
| Agents                  | `Cooperative or Non-cooperative or Group`                           |
| Number of Agents        | `>=1`                                                               |
| Termination Condition   | `No trees on fire exist`                                            |
| Truncation Steps        | `>=1`                                                               |
| Creation                | `gymnasium.make("wildfire-v0")`                                     |

Agents move over trees on fire to dump fire retardant. Initial fire is randomly located. Agents can be cooperative (shared reward) or non-cooperative (individual/group rewards). A non-cooperative agent preferentially protects a region of selfish interest within the grid. Above GIF contains two groups of agents with their selfish regions shown in same color.

## Extending multigrid

Please see this [guide](https://docs.google.com/document/d/13bCjSzRvLkdGWx7er67VQwF87pJmRIkDR41fm6iMToI/edit?usp=sharing) for creating a custom multigrid environment. See [CONTRIBUTING.md](https://github.com/Tran-Research-Group/gym-multigrid/blob/main/CONTRIBUTING.md) for our code guidelines if you are interested in adding your environment to this repo.
