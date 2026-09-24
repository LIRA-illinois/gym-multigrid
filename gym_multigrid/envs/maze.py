from dataclasses import dataclass
from typing import Any, Final, Literal, TypedDict, cast

import numpy as np
from gymnasium import Space, spaces
from numpy.typing import NDArray

from gym_multigrid.core.agent import Actions, Agent, MazeActions
from gym_multigrid.core.constants import NAV_DIR_TO_VEC
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Flag, Obstacle, WorldObj
from gym_multigrid.core.world import MazeWorld, World
from gym_multigrid.multigrid import (
    DEFAULT_FULL_OBS_ENV_PARTIAL_OBS_CONFIG,
    GridConfig,
    MultiGridEnv,
    ObservationMode,
    PartialObsConfig,
    RenderingConfig,
)
from gym_multigrid.typing import Position


class LayoutConfig(TypedDict):
    width: int
    height: int
    flag_positions: list[Position]
    init_agent_positions: list[Position]
    wall_positions: list[Position]


@dataclass
class Layout:
    width: int
    height: int
    flag_positions: list[Position]
    init_agent_positions: list[Position]
    wall_positions: list[Position]

    def generate_static_obs(
        self, obj_to_idx: dict[str, int] = MazeWorld.OBJECT_TO_IDX
    ) -> NDArray:
        static_obs: NDArray[np.int64] = np.zeros(
            (self.height, self.width), dtype=np.int64
        )
        for i, j in self.wall_positions:
            static_obs[i, j] = obj_to_idx["wall"]

        for i, j in self.flag_positions:
            static_obs[i, j] = obj_to_idx["flag"]

        self.static_obs: NDArray[np.int64] = static_obs

        return static_obs


class RewardConfig(TypedDict):
    flag_reward: float
    wall_penalty_ratio: float
    step_penalty_ratio: float


@dataclass
class Reward:
    flag_reward: float
    wall_penalty_ratio: float
    step_penalty_ratio: float


class ResetOptions(TypedDict):
    layout_config: LayoutConfig


class TensorObservationMode(ObservationMode["MazeEnv", spaces.Box, NDArray[np.int64]]):
    def observation_space(self, env: "MazeEnv") -> spaces.Box:
        return spaces.Box(
            low=0,
            high=len(env.world.OBJECT_TO_IDX) - 1,
            shape=(2, env.height, env.width),
            dtype=np.int64,
        )

    def create_observation(self, env: "MazeEnv") -> NDArray[np.int64]:
        observation: NDArray[np.int64] = np.zeros(
            (2, env.height, env.width), dtype=np.int64
        )

        observation[0, :, :] = env.layout.static_obs
        for agent in env.agents:
            if agent.pos is not None:
                observation[1, agent.pos[0], agent.pos[1]] = MazeWorld.OBJECT_TO_IDX[
                    "agent"
                ]
            else:
                pass

        return observation


class MapObservationMode(ObservationMode["MazeEnv", spaces.Box, NDArray[np.int64]]):
    def observation_space(self, env: "MazeEnv") -> spaces.Box:
        return spaces.Box(
            low=0,
            high=len(MazeWorld.OBJECT_TO_IDX) - 1,
            shape=(env.height, env.width),
            dtype=np.int64,
        )

    def create_observation(self, env: "MazeEnv") -> NDArray[np.int64]:
        observation: NDArray[np.int64] = np.zeros(
            (env.height, env.width), dtype=np.int64
        )
        observation[:, :] = env.layout.static_obs
        for agent in env.agents:
            if agent.pos is not None:
                observation[agent.pos[0], agent.pos[1]] = MazeWorld.OBJECT_TO_IDX[
                    "agent"
                ]
            else:
                pass

        return observation


DEFAULT_LAYOUT_CONFIG: LayoutConfig = {
    "width": 10,
    "height": 10,
    "flag_positions": [(9, 9)],
    "init_agent_positions": [(5, 5)],
    "wall_positions": [],
}

DEFAULT_REWARD_CONFIG: RewardConfig = {
    "flag_reward": 1.0,
    "wall_penalty_ratio": 0.0,
    "step_penalty_ratio": 0.01,
}


class MazeEnv(MultiGridEnv[NDArray[np.int64]]):
    """
    Multi-agent grid world environment with a maze layout to navigate through to reach the flags.

    Observation
    -----------
    The observation is a 2D grid with two channels:
    - The first channel represents the static objects in the environment (e.g., walls and flags).
    - The second channel represents the agent's position.
    - Encoding of the objects:
        - 0: background
        - 1: agent
        - 2: flag
        - 3: wall
    - The default observation shape is (2, 10, 10).

    Actions
    -------
    - 5 discrete actions (STAY, UP, RIGHT, DOWN, LEFT)
    - All agents' actions have to be supplied as a list of integers in `step()` method.

    Rewards
    -------
    - +1 for reaching the flag
    - -0.01 * 1 for each step
    - 0 * 1 for hitting the wall
    - These values can be configured using the `reward_config` parameter in the constructor.

    Termination
    -----------
    - The episode terminates when all agents reach the flag or a agent hits the wall.

    Rendering
    ---------
    - The environment can be rendered in two modes: human and rgb_array.
    - In rgb_array mode, the environment is rendered as a 3D numpy array with RGB values.

    Note
    ----
    - The layout can be updated every time the environment is reset using the `reset()` method's `options` parameter.
        - `options={"layout_config": layout_config}`

    Example
    -------
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
    ```
    """

    # Update metadata of the parent class
    observation_modes: dict[
        str, type[ObservationMode["MazeEnv", spaces.Box, NDArray[np.int64]]]
    ] = {
        "tensor": TensorObservationMode,
        "map": MapObservationMode,
    }

    def __init__(
        self,
        num_agents: int = 1,
        layout_config: LayoutConfig = DEFAULT_LAYOUT_CONFIG,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
        observation_mode: Literal["tensor"] = "tensor",
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ):
        """
        Initialize a new single agent maze environment

        Parameters
        ----------
        num_agents : int
            Number of agents in the environment
        layout_config : LayoutConfig = DEFAULT_LAYOUT_CONFIG
            Configuration of the layout of the environment.
            The default layout is a 10x10 grid with a flag at the bottom right corner
            and an agent at the center.
            This configuration can be updated using `reset()` method's `options` parameter as
            `options={"layout_config": layout_config}`.
        reward_config : RewardConfig = {
            "flag_reward": 1.0,
            "wall_penalty_ratio": 0.0,
            "step_penalty_ratio": 0.01,
        }
            Configuration of the reward function.
            The default reward function gives a reward of 1.0 for reaching the flag,
            no penalty for hitting the wall, and a penalty of 0.01 * 1.0 for each step.
        observation_mode : Literal["tensor"] = "tensor"
            Observation mode of the environment. The default observation mode is "tensor".
        render_mode : Literal["human", "rgb_array"] = "rgb_array"
            Render mode of the environment. The default render mode is "rgb_array".

        Example
        --------
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
        ```
        """

        world: Final[World] = MazeWorld
        action_set: type[Actions] = MazeActions

        self.layout_config_dict: LayoutConfig = layout_config
        self.layout = Layout(**self.layout_config_dict)
        self.layout.generate_static_obs()

        self.observation_mode = self.observation_modes[observation_mode]()

        self.reward = Reward(**reward_config)

        agents: list[Agent] = [
            Agent(
                world,
                index=i,
                color="blue",
                view_size=None,
                actions=action_set,
                dir_to_vec=NAV_DIR_TO_VEC,
                type="agent",
            )
            for i in range(num_agents)
        ]

        grid_config: Final[GridConfig] = GridConfig(
            height=self.layout.height,
            width=self.layout.width,
            actions_set=action_set,
            world=world,
        )

        rendering_config: RenderingConfig = RenderingConfig(
            render_mode=render_mode, uncached_object_types=["agent"]
        )

        partial_obs_config: PartialObsConfig = DEFAULT_FULL_OBS_ENV_PARTIAL_OBS_CONFIG

        super().__init__(
            agents=agents,
            **dict(grid_config),
            **dict(rendering_config),
            **dict(partial_obs_config),
        )

    def _set_observation_space(self) -> Space:
        return self.observation_mode.observation_space(self)

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height, self.world)

        for i, j in self.layout.wall_positions:
            self.put_obj(
                Obstacle(
                    self.world,
                    reward=-self.reward.wall_penalty_ratio * self.reward.flag_reward,
                ),
                i,
                j,
            )

        for flag_idx, (i, j) in enumerate(self.layout.flag_positions):
            self.put_obj(
                Flag(self.world, index=flag_idx, color="red", bg_color=None), i, j
            )

        self.init_grid: Grid = self.grid.copy()

        agents_init_pos: list[Position] = self.layout.init_agent_positions
        if len(agents_init_pos) == 0:
            agents_init_pos = [(-1, -1)] * len(self.agents)
        elif len(agents_init_pos) == 1:
            agents_init_pos = agents_init_pos * len(self.agents)
        elif len(agents_init_pos) == len(self.agents):
            pass
        else:
            raise ValueError(
                f"Number of agents {len(self.agents)} and number of initial agent positions {len(agents_init_pos)} do not match: {len(self.agents)} != {len(agents_init_pos)}"
            )

        for agent, init_pos in zip(self.agents, agents_init_pos):
            agent.reset()
            self.place_agent(agent, pos=init_pos)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if options is not None:
            if "layout_config" in options:
                self.layout_config_dict = self.layout_config_dict.copy()
                self.layout_config_dict.update(options["layout_config"])
                self.layout = Layout(**self.layout_config_dict)
                self.layout.generate_static_obs()
        else:
            pass

        self._reset_gym(seed=seed)
        self._gen_grid(self.width, self.height)

        self.agent_traj: list[list[Position]] = [[agent.pos for agent in self.agents]]
        self.rewards: list[float] = []

        obs = self._get_obs()
        info: dict[str, float] = self._get_info()

        return obs, info

    def _get_obs(self) -> NDArray[np.int64]:
        return self.observation_mode.create_observation(self)

    def _get_info(self) -> dict[str, float]:
        info = {}
        return info

    def _move_agent(self, action: int, agent: Agent) -> None:
        next_pos: Position

        assert agent.pos is not None

        action_set: type[MazeActions] = cast(type[MazeActions], self.actions)
        match action:
            case action_set.STAY:
                next_pos = agent.pos
            case action_set.LEFT:
                next_pos = agent.west_pos(in_tuple=True)
            case action_set.DOWN:
                next_pos = agent.south_pos(in_tuple=True)
            case action_set.RIGHT:
                next_pos = agent.east_pos(in_tuple=True)
            case action_set.UP:
                next_pos = agent.north_pos(in_tuple=True)
            case _:
                raise ValueError(f"Invalid action: {action}")

        if (
            next_pos[0] < 0
            or next_pos[1] < 0
            or next_pos[0] >= self.height
            or next_pos[1] >= self.width
        ):
            pass  # Do nothing
        else:
            next_cell: WorldObj | None = self.grid.get(*next_pos)

            if next_cell is None:
                agent.move(next_pos, self.grid, self.init_grid)
            elif next_cell.can_overlap():
                agent.move(
                    next_pos,
                    self.grid,
                    self.init_grid,
                )
            else:
                pass

    def _move_agents(self, actions: list[int]) -> None:
        # Move agent
        order: list[int] = np.random.permutation(len(self.agents)).tolist()

        for i in order:
            self._move_agent(actions[i], self.agents[i])

    def _is_agent_on_obj(self, agent_loc: Position, obj: list[Position]) -> bool:
        on_obj: bool = False

        for obj_loc in obj:
            if agent_loc[0] == obj_loc[0] and agent_loc[1] == obj_loc[1]:
                on_obj = True
                break
            else:
                pass

        return on_obj

    def step(
        self, action: NDArray[np.int64] | np.int64
    ) -> tuple[NDArray[np.int64], float, bool, bool, dict[str, Any]]:
        actions: list[int] = np.array([action]).flatten().tolist()

        self._move_agents(actions)

        assert self.agents[0].pos is not None

        terminated: bool = False
        truncated: bool = False

        flag_reward: float = self.reward.flag_reward
        wall_penalty: float = flag_reward * self.reward.wall_penalty_ratio
        step_penalty: float = flag_reward * self.reward.step_penalty_ratio
        reward: float = 0.0

        all_agents_on_flag: bool = True
        for agent in self.agents:
            if not self._is_agent_on_obj(agent.pos, self.layout.flag_positions):
                all_agents_on_flag = False
            else:
                reward += flag_reward

        agent_on_wall: bool = False

        if wall_penalty != 0:
            for agent in self.agents:
                if self._is_agent_on_obj(agent.pos, self.layout.wall_positions):
                    agent_on_wall = True
                    reward -= wall_penalty
                else:
                    pass
        else:
            pass

        terminated: bool = all_agents_on_flag or agent_on_wall

        reward -= step_penalty

        self.agent_traj.append([agent.pos for agent in self.agents])
        self.rewards.append(reward)

        observation = self._get_obs()
        info: dict[str, float] = self._get_info()

        return observation, reward, terminated, truncated, info
