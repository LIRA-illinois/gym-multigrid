from typing import Any, Literal, TypedDict

import numpy as np
import torch
from gymnasium import spaces
from numpy.typing import NDArray
from pydantic import BaseModel

from gym_multigrid.core.agent import Agent, GridActions, NavigationActions
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal, Hole, Lava, Wall
from gym_multigrid.core.world import RoomsWorld
from gym_multigrid.multigrid import (
    GridConfig,
    MultiGridEnv,
    ObservationMode,
    PartialObsConfig,
    RenderingConfig,
)
from gym_multigrid.typing import Position, Size


class PositionalObs(ObservationMode["RoomsEnv", spaces.Box, NDArray[np.float64]]):
    """
    Observation mode that returns the agent's position and goal position as a vector.

    The object locations are scaled to [0, 1] by the grid width and height.
    The observation vector contains:
    - Agent's x, y positions
    - Goal's x, y positions
    - Lavas' x, y positions
    - Holes' x, y positions

    """

    def observation_space(self, env: "RoomsEnv") -> spaces.Box:
        return spaces.Box(
            low=0,
            high=1,
            shape=(
                len(env.agents)
                + len(env.layout_config.spawn_configs[0].lavas)
                + len(env.layout_config.spawn_configs[0].holes)
                + 1,  # Goal position
                2,
            ),
            dtype=np.float64,
        )

    def create_observation(self, env: "RoomsEnv") -> NDArray[np.float64]:
        obs = np.array(
            [env.agents[0].pos] + [env.goal_pos] + env.lava_pos + env.hole_pos,
            dtype=np.float64,
        )
        obs = obs / np.maximum(env.width, env.height)

        return obs


class PositionalDictObs(
    ObservationMode["RoomsEnv", spaces.Dict, dict[str, NDArray[np.float64]]]
):
    """
    Observation mode that returns the agent's position and goal position as a dictionary.

    The object locations are scaled to [0, 1] by the grid width and height.
    The observation dictionary contains:
    - "obs": The agent, lava, and hole positions scaled to [0, 1].
    - "desired_goal": The goal position scaled to [0, 1].

    The observation space is a Dict with:
    - "obs": Box with shape (N, 2) where N is the number of agents, lava, and holes.
      Each entry is a 2D position scaled to [0, 1].
    - "desired_goal": Box with shape (2,) representing the goal position scaled to [0, 1].
    The agent's position is at index 0, followed by lava positions and hole positions.
    The goal position is at index 0 and 1 in the "desired_goal" array
    """

    def observation_space(self, env: "RoomsEnv") -> spaces.Dict:
        return spaces.Dict(
            {
                "obs": spaces.Box(
                    low=0,
                    high=1,
                    shape=(
                        len(env.agents)
                        + len(env.layout_config.spawn_configs[0].lavas)
                        + len(env.layout_config.spawn_configs[0].holes),
                        2,
                    ),
                    dtype=np.float64,
                ),
                "desired_goal": spaces.Box(
                    low=0,
                    high=1,
                    shape=(2,),
                    dtype=np.float64,
                ),
            }
        )

    def create_observation(self, env: "RoomsEnv") -> dict[str, NDArray[np.float64]]:
        # Scale the agent's position and goal position to [0, 1] by the grid width and height
        grid_size = np.maximum(env.width, env.height)

        return {
            "obs": np.array(
                [env.agents[0].pos] + env.lava_pos + env.hole_pos, dtype=np.float64
            )
            / grid_size,
            "desired_goal": np.array(env.goal_pos, dtype=np.float64) / grid_size,
        }


class TensorObs(ObservationMode["RoomsEnv", spaces.Box, NDArray[np.int64]]):
    """
    Observation mode that returns the agent, goal, and obstacles (lava, holes) as a grid tensor.

    The observation is a 2D grid where:
    - Each cell contains an integer representing the object type:
        - "empty": 0,
        - "wall": 1,
        - "agent": 2,
        - "goal": 3,
        - "lava": 4,
        - "hole": 5,
        - "floor": 6,
        - "door": 7,
        - "key": 8,
        - "ball": 9,
        - "box": 10,

    The observation space is a Box with shape (width, height) and dtype int64,
    with values in the range [0, 10].
    """

    def observation_space(self, env: "RoomsEnv") -> spaces.Box:
        return spaces.Box(
            low=0,
            high=10,
            shape=(env.width, env.height),
            dtype=np.int64,
        )

    def create_observation(self, env: "RoomsEnv") -> NDArray[np.int64]:
        obs = np.zeros((env.width, env.height), dtype=np.int64)
        obs[:, :] = self.static_obs
        for agent in env.agents:
            obs[agent.pos[1], agent.pos[0]] = env.world.OBJECT_TO_IDX["agent"]

        return obs

    def save_static_obs(
        self, env: "RoomsEnv", options: dict[str, Any] | None = None
    ) -> None:
        static_obs: NDArray[np.int64] = (
            np.ones((env.width, env.height), dtype=np.int64)
            * env.world.OBJECT_TO_IDX["empty"]
        )
        for y, row in enumerate(env.layout_config.field_map):
            for x, cell in enumerate(row):
                if cell == "#":
                    static_obs[y, x] = env.world.OBJECT_TO_IDX["wall"]
                else:
                    pass
        for lava in env.lava_pos:
            static_obs[lava[1], lava[0]] = env.world.OBJECT_TO_IDX["lava"]
        for hole in env.hole_pos:
            static_obs[hole[1], hole[0]] = env.world.OBJECT_TO_IDX["hole"]
        static_obs[env.goal_pos[1], env.goal_pos[0]] = env.world.OBJECT_TO_IDX["goal"]
        self.static_obs = static_obs


class VectorizedTensorObs(TensorObs):
    """
    Observation mode that returns the agent, goal, and obstacles (lava, holes) as a flattened grid tensor.

    The observation is a 1D array where:
    - Each cell contains an integer representing the object type same as in TensorObs.

    The observation space is a Box with shape (width * height,) and dtype int64,
    with values in the range [0, 10].
    """

    def observation_space(self, env: "RoomsEnv") -> spaces.Box:
        return spaces.Box(
            low=0,
            high=10,
            shape=(env.width * env.height,),
            dtype=np.int64,
        )

    def create_observation(self, env: "RoomsEnv") -> NDArray[np.int64]:
        obs = super().create_observation(env)
        return obs.flatten()


class ObjConfig(BaseModel):
    pos: Position | tuple[Position, Size] | None = (
        None  # Default position indicating no specific position
    )
    reward: float = 0.0
    absorbing: bool = True


class ObjConfigDict(TypedDict, total=False):
    pos: Position | tuple[Position, Size] | None
    reward: float
    absorbing: bool


class SpawnConfig(BaseModel):
    """
    Configuration for spawning objects in the environment.

    Attributes
    ----------
    agent: Position | tuple[Position, Size] | None
        The position or size of the agent object.
    goal: ObjConfig
        The configuration for the goal object.
    lavas: list[ObjConfig]
        A list of configurations for lava objects.
    holes: list[ObjConfig]
        A list of configurations for hole objects.
    cleared_doorways: list[Position]
        A list of positions for cleared doorways, if any.
        When specified, there will be no negative reward objects placed next to these doorways.
    """

    agent: Position | tuple[Position, Size] | None = None
    goal: ObjConfig  # List of goal objects, if any
    lavas: list[ObjConfig] = []  # List of lava objects, if any
    holes: list[ObjConfig] = []  # List of hole objects, if any
    waypoints: list[ObjConfig] = []  # List of way point objects, if any
    cleared_doorways: list[Position] = []  # List of cleared doorways, if any
    cleared_doorway_neighbors: list[Position] = []  # Neighbors of cleared doorways

    model_config = {"arbitrary_types_allowed": True}

    # Store the neighbor cells of the cleared doorways during initialization
    def model_post_init(self, context: Any) -> None:
        self.cleared_doorway_neighbors = [
            (pos[0] + dx, pos[1] + dy)
            for pos in self.cleared_doorways
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]

    def pos_in_doorway_neighbors(self, env: MultiGridEnv, pos: Position) -> bool:
        """
        Check if the given position is in the cleared doorway neighbors.

        Parameters
        ----------
        pos: Position
            The position to check.

        Returns
        -------
        bool
            True if the position is in the cleared doorway neighbors, False otherwise.
        """
        return pos in self.cleared_doorway_neighbors


class SpawnConfigDict(TypedDict, total=False):
    agent: Position | tuple[Position, Size] | None
    goal: ObjConfigDict
    lavas: list[ObjConfigDict]
    holes: list[ObjConfigDict]
    waypoints: list[ObjConfigDict]
    cleared_doorways: list[Position]


class LayoutConfig(BaseModel):
    field_map: list[str]
    spawn_configs: list[SpawnConfig]


class LayoutConfigDict(TypedDict):
    field_map: list[str]
    spawn_configs: list[SpawnConfigDict]


class RewardConfig(BaseModel):
    step_penalty: float = 0.01
    sum_reward: bool = True


class RewardConfigDict(TypedDict):
    step_penalty: float
    sum_reward: bool


class ObjInitDict(TypedDict):
    pos: Position | None
    top: Position | None
    size: Size | None


class RoomsEnvInitDict(TypedDict, total=False):
    spawn_type: int
    layout_config: LayoutConfigDict
    state_representation: str
    reward_config: RewardConfigDict
    tile_size: int
    render_mode: Literal["human", "rgb_array"]


class RoomsEnvInit(BaseModel):
    spawn_type: int = 0
    layout_config: LayoutConfig = LayoutConfig(
        field_map=[
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
        spawn_configs=[],
    )
    state_representation: str = "tensor"
    reward_config: RewardConfig = RewardConfig(
        step_penalty=0.01,
        sum_reward=True,
    )
    tile_size: int = 32
    render_mode: Literal["human", "rgb_array"] = "rgb_array"


class RoomsEnv(
    MultiGridEnv[
        NDArray[np.int64] | NDArray[np.float64] | dict[str, NDArray[np.float64]]
    ]
):
    """
    Environment with separate rooms, where agents navigate to a goal while avoiding obstacles like lava and holes.

    Observation
    -----------
    There are multiple observation modes available:
    - "positional": Returns the agent's position and goal position as a vector.
    - "positional_dict": Returns the agent's position, lava positions, and hole positions
      as a dictionary with keys "obs" and "desired_goal".
    - "tensor": Returns a 2D grid tensor where each cell contains an integer representing
      the object type (e.g., empty, wall, agent, goal, lava, hole).
    - "vectorized_tensor": Returns a flattened 1D array of the grid tensor.
    The observation space is defined based on the selected observation mode.

    Action
    -------
    The action space is defined by the `NavigationActions` class, which includes actions:
    0. Stay
    1. Left
    2. Down
    3. Right
    4. Up

    Rewards
    -------
    The environment provides a step penalty defined in the `RewardConfig` class.
    The agent receives a reward for reaching the goal and may receive penalties for stepping on lava or holes.
    The object-based rewards are defined in `LayoutConfig.spawn_configs`.

    Termination
    -----------
    The episode terminates when the agent reaches the goal or steps on a lava or hole.
    The termination condition can be changed by modifying the `absorbing` attribute of the goal, lava, and hole objects in the `LayoutConfig.spawn_configs`.

    Rendering
    ---------
    - The environment can be rendered in two modes: human and rgb_array.
    - In rgb_array mode, the environment is rendered as a 3D numpy array with RGB values.

    Note
    -------
    - The objects can be randomly initialized within specified ranges using the `random_init_range` attribute in the `ObjConfig` class.

    Example
    -------
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
                        },
                        {
                            "agent": (11, 1),
                            "goal": {"pos": (7, 9), "reward": 1.0, "absorbing": True},
                        },
                        {
                            "agent": (9, 3),
                            "goal": {"pos": (9, 9), "reward": 1.0, "absorbing": True},
                        },
                        {
                            "agent": (3, 9),
                            "goal": {"pos": (9, 4), "reward": 1.0, "absorbing": True},
                            "lavas": [
                                {"pos": (8, 4), "reward": 0, "absorbing": False},
                                {"pos": (9, 2), "reward": 0, "absorbing": False},
                                {"pos": (11, 1), "reward": 0, "absorbing": False},
                                {"pos": (5, 3), "reward": 0, "absorbing": False},
                                {"pos": (3, 5), "reward": 0, "absorbing": False},
                                {"pos": (3, 2), "reward": 0, "absorbing": False},
                                {"pos": (5, 9), "reward": -1, "absorbing": True},
                                {"pos": (3, 8), "reward": -1, "absorbing": True},
                                {"pos": (2, 11), "reward": -1, "absorbing": True},
                                {"pos": (10, 8), "reward": -1, "absorbing": True},
                                {"pos": (8, 9), "reward": -1, "absorbing": True},
                                {"pos": (7, 11), "reward": -1, "absorbing": True},
                            ],
                            "holes": [
                                {"pos": (7, 3), "reward": 0, "absorbing": False},
                                {"pos": (10, 5), "reward": 0, "absorbing": False},
                                {"pos": (8, 6), "reward": 0, "absorbing": False},
                                {"pos": (4, 4), "reward": -1, "absorbing": True},
                                {"pos": (2, 3), "reward": -1, "absorbing": True},
                                {"pos": (1, 1), "reward": -1, "absorbing": True},
                                {"pos": (2, 7), "reward": 0, "absorbing": False},
                                {"pos": (1, 9), "reward": 0, "absorbing": False},
                                {"pos": (4, 10), "reward": 0, "absorbing": False},
                                {"pos": (7, 8), "reward": -1, "absorbing": True},
                                {"pos": (9, 10), "reward": -1, "absorbing": True},
                                {"pos": (11, 11), "reward": -1, "absorbing": True},
                            ],
                        },
                    ],
                },
                "state_representation": "vectorized_tensor",
                "reward_config": {
                    "step_penalty": 0.01,
                    "sum_reward": True,
                },
                "tile_size": 32,
                "render_mode": "rgb_array",
            },
        )
    ```

    """

    observation_modes: dict[
        str,
        type[ObservationMode["RoomsEnv", spaces.Box, NDArray[np.int64]]]
        | type[ObservationMode["RoomsEnv", spaces.Box, NDArray[np.float64]]]
        | type[
            ObservationMode["RoomsEnv", spaces.Dict, dict[str, NDArray[np.float64]]]
        ],
    ] = {
        "positional": PositionalObs,
        "positional_dict": PositionalDictObs,
        "tensor": TensorObs,
        "vectorized_tensor": VectorizedTensorObs,
    }
    layout_config: LayoutConfig

    def __init__(
        self,
        spawn_type: int = 0,
        layout_config: LayoutConfigDict | LayoutConfig = {
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
                },
                {
                    "agent": (11, 1),
                    "goal": {"pos": (7, 9), "reward": 1.0, "absorbing": True},
                },
                {
                    "agent": (9, 3),
                    "goal": {"pos": (9, 9), "reward": 1.0, "absorbing": True},
                },
                {
                    "agent": (3, 9),
                    "goal": {"pos": (9, 4), "reward": 1.0, "absorbing": True},
                    "lavas": [
                        {"pos": (8, 4), "reward": 0, "absorbing": False},
                        {"pos": (9, 2), "reward": 0, "absorbing": False},
                        {"pos": (11, 1), "reward": 0, "absorbing": False},
                        {"pos": (5, 3), "reward": 0, "absorbing": False},
                        {"pos": (3, 5), "reward": 0, "absorbing": False},
                        {"pos": (3, 2), "reward": 0, "absorbing": False},
                        {"pos": (5, 9), "reward": -1, "absorbing": True},
                        {"pos": (3, 8), "reward": -1, "absorbing": True},
                        {"pos": (2, 11), "reward": -1, "absorbing": True},
                        {"pos": (10, 8), "reward": -1, "absorbing": True},
                        {"pos": (8, 9), "reward": -1, "absorbing": True},
                        {"pos": (7, 11), "reward": -1, "absorbing": True},
                    ],
                    "holes": [
                        {"pos": (7, 3), "reward": 0, "absorbing": False},
                        {"pos": (10, 5), "reward": 0, "absorbing": False},
                        {"pos": (8, 6), "reward": 0, "absorbing": False},
                        {"pos": (4, 4), "reward": -1, "absorbing": True},
                        {"pos": (2, 3), "reward": -1, "absorbing": True},
                        {"pos": (1, 1), "reward": -1, "absorbing": True},
                        {"pos": (2, 7), "reward": 0, "absorbing": False},
                        {"pos": (1, 9), "reward": 0, "absorbing": False},
                        {"pos": (4, 10), "reward": 0, "absorbing": False},
                        {"pos": (7, 8), "reward": -1, "absorbing": True},
                        {"pos": (9, 10), "reward": -1, "absorbing": True},
                        {"pos": (11, 11), "reward": -1, "absorbing": True},
                    ],
                    "cleared_doorways": [(2, 6), (6, 3), (9, 7), (6, 10)],
                },
            ],
        },
        state_representation: str = "tensor",
        allow_stay: bool = True,
        reward_config: RewardConfigDict | RewardConfig = {
            "step_penalty": 0.01,
            "sum_reward": True,
        },
        tile_size: int = 32,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Initialize a new capture the flag environment.

        Parameters
        ----------
        spawn_type: int
            The array index of the spawn configuration to use from `layout_config.spawn_configs`.
            This determines the initial positions of the agent, goal, lava, and holes.
        layout_config: LayoutConfigDict
            The layout configuration for the environment.
        state_representation: str
            The state representation mode to use. Options are:
            - "positional": Returns the agent's position and goal position as a vector.
            - "positional_dict": Returns the agent's position, lava positions, and hole positions as a dictionary.
            - "tensor": Returns a 2D grid tensor where each cell contains an integer representing the object type.
            - "vectorized_tensor": Returns a flattened 1D array of the grid tensor.
        allow_stay: bool
            Whether to allow the agent to take the "stay" action.
        reward_config: RewardConfigDict
            The reward configuration for the environment.
        tile_size: int
            The size of each tile in the grid for rendering.
        render_mode: Literal["human", "rgb_array"]
            The rendering mode to use. Options are:
            - "human": Renders the environment for human consumption (e.g., using Pygame).
            - "rgb_array": Returns a RGB array representation of the environment.
        """
        ### fundamental parameters
        self.spawn_type: int = spawn_type
        # spawn_configs: list[SpawnConfig] = [
        #     SpawnConfig(**conf) for conf in layout_config["spawn_configs"]
        # ]
        if isinstance(layout_config, dict):
            self.layout_config: LayoutConfig = LayoutConfig.model_validate(
                layout_config
            )
        else:
            self.layout_config = layout_config

        if isinstance(reward_config, dict):
            self.reward_config: RewardConfig = RewardConfig.model_validate(
                reward_config
            )
        else:
            self.reward_config = reward_config

        grid_size: tuple[int, int] = (
            len(self.layout_config.field_map[0]),
            len(self.layout_config.field_map),
        )
        self.state_representation = self.observation_modes[state_representation]()

        width, height = grid_size
        world = RoomsWorld
        self.allow_stay: bool = allow_stay
        actions_set = NavigationActions if allow_stay else GridActions

        # NOTE: currently only one agent is supported
        agents = [
            Agent(
                world,
                color="blue",
                bg_color=None,
                actions=actions_set,
                type="agent",
            )
        ]

        grid_config = GridConfig(
            grid_size=None,
            width=width,
            height=height,
            world=world,
            actions_set=actions_set,
        )
        render_config = RenderingConfig(tile_size=tile_size, render_mode=render_mode)
        partial_obs_config = PartialObsConfig()

        super().__init__(
            agents=agents,
            **dict(grid_config),
            **render_config.model_dump(),
            **dict(partial_obs_config),
        )

    def _set_observation_space(self) -> spaces.Box | spaces.Dict:
        observation_space = self.state_representation.observation_space(self)

        return observation_space

    def _parse_init_pos(
        self, init_pos: Position | tuple[Position, Size] | None
    ) -> ObjInitDict:
        """
        Parse the initial position for the agent.

        Parameters
        ----------
        init_pos: Position | tuple[Position, Size] | None
            The initial position for the agent.

        Returns
        -------
        obj_init_dict: ObjInitDict
        A dictionary containing the parsed position, top left corner, and size for the agent's initial position.
        The dictionary contains the following keys:
        - pos: Position | None
           - The position to place the agent.
        - top: Position | None
            - The top left corner of the range to position to place the agent.
        - size: Size | None
           - The size of the range to position the agent.

        """
        pos: Position | None
        size: Size | None
        top: Position | None
        match init_pos:
            case None:
                pos = None
                top = None
                size = None
            case (x, y) if isinstance(x, int) and isinstance(y, int):
                pos = (x, y)
                top = None
                size = None
            case (a, b) if isinstance(a, tuple) and isinstance(b, tuple):
                pos = None
                top = a
                size = b
            case _:
                raise ValueError(
                    f"Invalid initial position: {init_pos}. ",
                    "Expected None, a tuple of (x, y), or a tuple of (top, size)",
                )

        obj_init_dict: ObjInitDict = {
            "pos": pos,
            "top": top,
            "size": size,
        }

        return obj_init_dict

    def _gen_grid(self, width: int, height: int):
        # Create the grid
        self.grid = Grid(width, height, self.world)

        # Translate the maze structure into the grid
        for y, row in enumerate(self.layout_config.field_map):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.grid.set(x, y, Wall(self.world, type="wall", color="grey"))
                else:
                    pass

        # place goal
        goal = self.layout_config.spawn_configs[self.spawn_type].goal
        goal_obj = Goal(
            self.world,
            0,
            color="green",
            reward=goal.reward,
            absorbing=goal.absorbing,
        )
        self.place_object(goal_obj, **self._parse_init_pos(goal.pos))
        self.goal_pos: Position = goal_obj.pos

        # Place Waypoints
        self.waypoint_pos: list[Position] = []
        for waypoint in self.layout_config.spawn_configs[self.spawn_type].waypoints:
            waypoint_obj = Goal(
                self.world,
                color="dark_grey",
                reward=waypoint.reward,
                absorbing=waypoint.absorbing,
            )
            self.place_object(waypoint_obj, **self._parse_init_pos(waypoint.pos))
            self.waypoint_pos.append(waypoint_obj.pos)

        self.lava_pos: list[Position] = []
        self.hole_pos: list[Position] = []

        # place lavas
        for i, lava in enumerate(
            self.layout_config.spawn_configs[self.spawn_type].lavas
        ):
            lava_obj = Lava(
                self.world,
                color="red",
                reward=lava.reward,
                absorbing=lava.absorbing,
            )
            if lava.reward < 0:
                self.place_object(
                    lava_obj,
                    **self._parse_init_pos(lava.pos),
                    reject_fn=self.layout_config.spawn_configs[
                        self.spawn_type
                    ].pos_in_doorway_neighbors,
                )
            else:
                self.place_object(lava_obj, **self._parse_init_pos(lava.pos))
            self.lava_pos.append(lava_obj.pos)

        # place holes
        for i, hole in enumerate(
            self.layout_config.spawn_configs[self.spawn_type].holes
        ):
            hole_obj = Hole(
                self.world,
                color="purple",
                bg_color=None,
                reward=hole.reward,
                absorbing=hole.absorbing,
            )
            if hole.reward < 0:
                self.place_object(
                    hole_obj,
                    **self._parse_init_pos(hole.pos),
                    reject_fn=self.layout_config.spawn_configs[
                        self.spawn_type
                    ].pos_in_doorway_neighbors,
                )
            else:
                self.place_object(hole_obj, **self._parse_init_pos(hole.pos))
            self.hole_pos.append(hole_obj.pos)

    def _reset_agents(self):
        """
        Reset the agents' positions in the grid.
        If random_init_pos is True, randomly select a position from the grid.
        """
        for agent in self.agents:
            agent_pos = self.layout_config.spawn_configs[self.spawn_type].agent

            self.place_agent(
                agent, **self._parse_init_pos(agent_pos), reset_agent_status=True
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if isinstance(options, dict):
            self.spawn_type = options.get("spawn_type", self.spawn_type)
            self.layout_config.spawn_configs = [
                SpawnConfig(**conf)
                for conf in options.get(
                    "spawn_configs",
                    [
                        config.model_dump()
                        for config in self.layout_config.spawn_configs
                    ],
                )
            ]

        super().reset(seed=seed, options=options)
        self.state_representation.save_static_obs(self)

        ### NOTE: NOT MULTIAGENT SETTING
        observations = self._get_obs()
        info = {"is_success": False}

        return observations, info

    def step(
        self, action: np.int64 | NDArray[np.int64]
    ) -> tuple[
        NDArray[np.int64] | NDArray[np.float64] | dict[str, NDArray[np.float64]],
        NDArray[np.float64] | float,
        bool,
        bool,
        dict[str, Any],
    ]:
        self.step_count += 1

        ### NOTE: MULTIAGENT SETTING NOT IMPLEMENTED
        assert action.size == 1, "Only one agent is supported in this environment."
        actions: list[int] = np.array([action], dtype=np.int64).flatten().tolist()
        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        info = {"is_success": False}
        terminated: bool = False
        for i in order:
            agent: Agent = self.agents[i]
            assert isinstance(agent, Agent)
            if agent.terminated or agent.paused or not agent.started:
                continue

            # Get the current agent position
            curr_pos: Position = agent.pos

            # Rotate left
            self.actions: type[NavigationActions]
            fwd_pos: Position
            match actions[i]:
                case self.actions.LEFT:
                    fwd_pos = agent.west_pos(in_tuple=True)
                case self.actions.RIGHT:
                    fwd_pos = agent.east_pos(in_tuple=True)
                case self.actions.UP:
                    fwd_pos = agent.north_pos(in_tuple=True)
                case self.actions.DOWN:
                    fwd_pos = agent.south_pos(in_tuple=True)
                case self.actions.STAY:
                    fwd_pos = curr_pos
                case _:
                    raise ValueError(
                        f"Unknown action: {actions[i]}. Expected one of {self.actions}"
                    )

            info["is_success"] = False
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is not None:
                if fwd_cell.can_overlap():
                    agent.move(fwd_pos, self.grid, self.init_grid)
                    rewards[i] = fwd_cell.reward
                    if fwd_cell.absorbing:
                        terminated = True
                        if isinstance(fwd_cell, Goal):
                            info["is_success"] = True
                        else:
                            pass
                else:
                    pass
            else:
                agent.move(fwd_pos, self.grid, self.init_grid)

        if self.reward_config.sum_reward:
            rewards = np.sum(rewards)
        else:
            pass
        rewards -= self.reward_config.step_penalty

        ### NOTE: not multiagent setting
        truncated: bool = False

        observations = self._get_obs()

        return observations, rewards, terminated, truncated, info

    def _get_obs(self):
        return self.state_representation.create_observation(self)

    def get_rewards_heatmap(self, extractor: torch.nn.Module, eigenvectors: np.ndarray):
        raise NotImplementedError(
            "get_rewards_heatmap is not implemented for RoomsEnv. "
            "Please implement this method in your subclass."
        )
        assert self.state_representation in [
            "vectorized_tensor",
            "tensor",
        ], f"Unsupported state representation: {self.state_representation}"

        # Environment indices
        empty_idx = 1
        goal_idx = 8
        agent_idx = 10
        wall_idx = 2

        # Get base state
        state, _ = self.reset()
        agent_pos = np.where(state == agent_idx)
        state[agent_pos] = empty_idx
        self.close()

        if self.state_representation != "tensor":
            state = state.reshape(self.width, self.height, -1)

        mask = (state != wall_idx) & (state != goal_idx)
        non_mask = ~mask

        heatmaps = []
        grid_shape = (self.width, self.height, 1)
        for n in range(eigenvectors.shape[0]):
            eig = eigenvectors[n]
            reward_map = np.full(grid_shape, fill_value=np.nan)

            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    current_idx = (i, j, 0)
                    current_val = state[current_idx]

                    if current_val == wall_idx or current_val == goal_idx:
                        reward_map[current_idx] = 0.0
                    else:
                        # Copy and manipulate state
                        state_copy = np.copy(state)
                        state_copy[current_idx] = agent_idx
                        with torch.no_grad():
                            feature, _ = extractor(state_copy)
                        feature = feature.cpu().numpy().squeeze(0)

                        reward = np.dot(eig, feature)
                        reward_map[current_idx] = reward

            # reward_map = # normalize between -1 to 1
            pos_mask = np.logical_and(mask, (reward_map > 0))
            neg_mask = np.logical_and(mask, (reward_map < 0))

            # Normalize positive values to [0, 1]
            if np.any(pos_mask):
                pos_max, pos_min = (
                    reward_map[pos_mask].max(),
                    reward_map[pos_mask].min(),
                )
                if pos_max != pos_min:
                    reward_map[pos_mask] = (reward_map[pos_mask] - pos_min) / (
                        pos_max - pos_min + 1e-4
                    )

            # Normalize negative values to [-1, 0]
            if np.any(neg_mask):
                neg_max, neg_min = (
                    reward_map[neg_mask].max(),
                    reward_map[neg_mask].min(),
                )
                if neg_max != neg_min:
                    reward_map[neg_mask] = (reward_map[neg_mask] - neg_min) / (
                        neg_max - neg_min + 1e-4
                    ) - 1.0

            # Set all other entries (walls, empty) to 0
            reward_map = reward_map.reshape(self.width, self.height, -1)
            reward_map = self.reward_map_to_rgb(reward_map, mask)

            # set color theme as blue and red (blue = -1 and red = 1)
            # set wall color at value 0 and goal idx as 1
            heatmaps.append(reward_map)

        return heatmaps

    def reward_map_to_rgb(self, reward_map: np.ndarray, mask) -> np.ndarray:
        rgb_img = np.zeros((self.width, self.height, 3), dtype=np.float64)

        pos_mask = np.logical_and(mask, (reward_map > 0))
        neg_mask = np.logical_and(mask, (reward_map < 0))

        # Blue for negative: map [-1, 0] → [1, 0]
        rgb_img[neg_mask[:, :, 0], 2] = -reward_map[neg_mask]  # blue channel

        # Red for positive: map [0, 1] → [0, 1]
        rgb_img[pos_mask[:, :, 0], 0] = reward_map[pos_mask]  # red channel

        # rgb_img.flatten()[mask] to grey
        rgb_img[~mask[:, :, 0], :] = 0.5

        return rgb_img
