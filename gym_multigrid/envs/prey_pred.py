import enum
import warnings
from typing import Any, Literal, Type, TypedDict

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from gym_multigrid.core.agent import Agent, NavigationActions
from gym_multigrid.core.constants import NAV_DIR_TO_VEC, PREY_PRED_COLORS
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Floor, WorldObj
from gym_multigrid.core.world import World
from gym_multigrid.multigrid import (
    DEFAULT_FULL_OBS_ENV_PARTIAL_OBS_CONFIG,
    GridConfig,
    MultiGridEnv,
    PartialObsConfig,
    RenderingConfig,
)
from gym_multigrid.policy import AgentPolicy
from gym_multigrid.policy.prey_pred import PREY_PRED_POLICIES
from gym_multigrid.policy.prey_pred.utils import a_star
from gym_multigrid.typing import Position

PreyPredWorld = World(
    encode_dim=2,
    normalize_obs=1,
    COLORS=PREY_PRED_COLORS,
    OBJECT_TO_IDX={
        "empty": 0,
        "wall": 1,
        "prey_area": 2,
        "predator": 3,
        "easy_prey": 4,
        "hard_prey": 5,
        "dead_prey": 6,
    },
)


class ObservationConfig(BaseModel):
    """
    Configuration for the observation of the environment.

    Attributes
    ----------
    encode_prey_areas : bool
        Whether to encode prey areas in the observation.
        If True, prey areas are encoded as a separate layer in the observation.
        If False, prey areas are not encoded in the observation.
    """

    encode_prey_areas: bool = True


class ObservationConfigDict(TypedDict):
    encode_prey_areas: bool


class PredatorConfig(BaseModel):
    """
    Configuration for the predator agent in the environment.

    Attributes
    ----------
    init_pos : tuple[int, int]
        The initial position of the predator agent.
    policy_type : Literal["teammate", "ego"]
        The type of policy used by the predator agent.
        "ego" means the predator is controlled by the user, while "teammate" means it is controlled by the environment.
    target_preys : list[int]
        The list of prey indices that this predator is targeting.
    color : str
        The color of the predator agent.
    """

    init_pos: tuple[int, int]
    policy_type: Literal["teammate", "ego"]
    target_preys: list[int]
    color: str


class PredatorConfigDict(TypedDict):
    init_pos: tuple[int, int]
    policy_type: Literal["teammate", "ego"]
    target_preys: list[int]
    color: str


class PreyType(BaseModel):
    """
    Configuration for a type of prey object in the environment.

    Attributes
    ----------
    name : str
        The name of the prey type.
    policy : Literal["random"]
        The policy used by the prey agent.
    capture_reward : float
        The reward ego to the predator for capturing a prey of this type.
    num_required_preds_capture : Literal[1, 2, 3, 4]
        The number of neighboring predators required to capture a prey of this type.
    num_required_preds_fix : Literal[1, 2, 3]
        The number of neighboring predators required to fix a prey of this type (up to a maximum of 4).
        The value should be less than or equal to `num_required_preds_capture`.
        When `num_required_preds_fix` is 1, the prey is fixed immediately after being captured.
    """

    name: str
    policy: Literal["random"]
    capture_reward: float
    num_required_preds_capture: int
    num_required_preds_fix: int


class PreyTypeDict(TypedDict):
    """
    Configuration for a type of prey object in the environment.

    Attributes
    ----------
    name : str
        The name of the prey type.
    policy : str
        The policy used by the prey agent.
    capture_reward : float
        The reward ego to the predator for capturing a prey of this type.
    num_required_preds_capture : int
        The number of neighboring predators required to capture a prey of this type.
    num_required_preds_fix : int
        The number of neighboring predators required to fix a prey of this type (up to a maximum of 4).
        The value should be less than or equal to `num_required_preds_capture`.
        When `num_required_preds_fix` is 1, the prey is fixed immediately after being captured.
    """

    name: str
    policy: Literal["random"]
    capture_reward: float
    num_required_preds_capture: int
    num_required_preds_fix: int


class PreyConfig(BaseModel):
    """
    Configuration for the prey agent in the environment.

    Attributes
    ----------
    type : str
        The type of the prey agent.
    territory_dims : tuple[int, int]
        The dimensions of the prey's territory.
    territory_left_top_corner : tuple[int, int]
        The left top corner of the prey's territory.
    color : str
        The color of the prey agent.
    """

    type: str
    territory_dims: tuple[int, int]
    territory_left_top_corner: tuple[int, int]
    color: str


class PreyConfigDict(TypedDict):
    """
    Configuration for the prey agaent in the environment.

    Attributes
    ----------

    """

    type: str
    territory_dims: tuple[int, int]
    territory_left_top_corner: tuple[int, int]
    color: str


class ResetOptions(TypedDict, total=False):
    agent_pos_list: list[tuple[int, int]]


class Prey(Agent):
    def __init__(
        self,
        prey_config: PreyConfig,
        prey_type: PreyType,
        world: World,
        index: int,
        view_size: int | None = None,
        policy_dict: dict[str, Type[AgentPolicy]] = PREY_PRED_POLICIES,
    ):
        self.prey_config: PreyConfig = prey_config
        self.prey_type: PreyType = prey_type
        self.policy_class: Type[AgentPolicy] = policy_dict[self.prey_type.policy]

        self.neighbor_pos_offsets: NDArray[np.int_] = np.array(
            [[-1, 0], [1, 0], [0, -1], [0, 1]]
        )
        self.neighbor_pos: NDArray[np.int_] = np.zeros((4, 2), dtype=np.int_)

        super().__init__(
            world=world,
            index=index,
            actions=NavigationActions,
            color=prey_config.color,
            bg_color="light_grey",
            type=prey_type.name,
            view_size=view_size,
            dir_to_vec=NAV_DIR_TO_VEC,
        )

    def get_init_pos(self) -> tuple[int, int]:
        # Randomly select a position within the prey territory
        territory_dims: tuple[int, int] = self.prey_config.territory_dims
        territory_left_top_corner: tuple[int, int] = (
            self.prey_config.territory_left_top_corner
        )

        x: int = self.policy.random_generator.integers(
            territory_left_top_corner[0],
            territory_left_top_corner[0] + territory_dims[0],
        )
        y: int = self.policy.random_generator.integers(
            territory_left_top_corner[1],
            territory_left_top_corner[1] + territory_dims[1],
        )

        return x, y

    def reset(self, env_generator: np.random.Generator) -> None:
        super().reset()
        self.policy: AgentPolicy = self.policy_class(
            action_set=self.actions,
            random_generator=env_generator,
        )
        if self.pos is not None:
            self.neighbor_pos = self.pos + self.neighbor_pos_offsets
        else:
            self.neighbor_pos = np.zeros((4, 2), dtype=np.int_)

        self.captured_by: list[int] = []

    def act(self, observation: NDArray, options: dict[str, Any]) -> int:
        return self.policy.act(observation, options)

    @property
    def pos(self) -> Position:
        return (self._pos[0], self._pos[1])

    @pos.setter
    def pos(self, pos: Position) -> None:
        self._pos = pos
        if pos is not None:
            self.neighbor_pos = pos + self.neighbor_pos_offsets

    def check_pos_in_territory(self, pos: tuple[int, int]) -> bool:
        """
        Check if the given position is within the prey territory.

        Parameters
        ----------
        pos : tuple[int, int]
            The position to check.

        Returns
        -------
        in_territory : bool
            True if the position is within the prey territory, False otherwise.
        """
        territory_dims: tuple[int, int] = self.prey_config.territory_dims
        territory_left_top_corner: tuple[int, int] = (
            self.prey_config.territory_left_top_corner
        )

        in_territory: bool = (
            territory_left_top_corner[0]
            <= pos[0]
            < territory_left_top_corner[0] + territory_dims[0]
            and territory_left_top_corner[1]
            <= pos[1]
            < territory_left_top_corner[1] + territory_dims[1]
        )

        return in_territory

    def check_fix_condition(self, grid: Grid) -> bool:
        """
        Check if the prey is in a fixable condition.

        Parameters
        ----------
        grid : Grid
            The grid of the environment.

        Returns
        -------
        fixable : bool
            True if the prey is in a fixable condition, False otherwise.
        """
        num_required_preds_fix: int = self.prey_type.num_required_preds_fix
        num_neighbor_preds: int = 0

        for neighbor_pos in self.neighbor_pos:
            cell: None | WorldObj = grid.get(*neighbor_pos)
            if cell is not None and cell.type == "predator":
                num_neighbor_preds += 1
            else:
                pass

        fixable: bool = num_neighbor_preds >= num_required_preds_fix

        return fixable

    def check_capture_condition(self, grid: Grid) -> tuple[bool, list[int]]:
        """
        Check if the prey is in a capturable condition.

        Parameters
        ----------
        grid : Grid
            The grid of the environment.

        Returns
        -------
        capturable : bool
            True if the prey is in a capturable condition, False otherwise.
        """
        num_required_preds_capture: int = self.prey_type.num_required_preds_capture
        num_neighbor_preds: int = 0

        pred_ids: list[int] = []

        for neighbor_pos in self.neighbor_pos:
            cell: None | WorldObj = grid.get(*neighbor_pos)
            if isinstance(cell, Predator):
                num_neighbor_preds += 1
                pred_ids.append(cell.index)
            else:
                pass

        capturable: bool = num_neighbor_preds >= num_required_preds_capture
        pred_ids = pred_ids if capturable else []

        return capturable, pred_ids

    def pos_in_neighbor(self, pos: tuple[int, int]) -> bool:
        """
        Check if the given position is in the prey's neighborhood.

        Parameters
        ----------
        pos : tuple[int, int]
            The position to check.

        Returns
        -------
        in_neighbor : bool
            True if the position is in the prey's neighborhood, False otherwise.
        """
        in_neighbor: bool = any(
            np.all(pos == neighbor_pos) for neighbor_pos in self.neighbor_pos
        )

        return in_neighbor


class Predator(Agent):
    def __init__(
        self,
        world: World,
        index: int,
        pred_config: PredatorConfig,
        view_size: int | None = None,
    ):
        self.pred_config: PredatorConfig = pred_config

        super().__init__(
            world=world,
            index=index,
            actions=NavigationActions,
            color=pred_config.color,
            bg_color="white",
            type="predator",
            view_size=view_size,
            dir_to_vec=NAV_DIR_TO_VEC,
        )

    def get_init_pos(self) -> tuple[int, int]:
        return self.pred_config.init_pos

    def reset(self, env_generator: np.random.Generator) -> None:
        super().reset()

    @property
    def target_preys(self) -> list[int]:
        return self.pred_config.target_preys


DEFAULT_OBSERVATION_CONFIG: ObservationConfigDict = {
    "encode_prey_areas": True,
    # remove_dead_agents=True,
    # encode_dead_agents_as="wall",
}

DEFAULT_PREDATOR_CONFIGS: list[PredatorConfigDict] = [
    {
        "init_pos": (6, 6),
        "policy_type": "ego",
        "color": "red",
        "target_preys": [],
    },
    {
        "init_pos": (7, 7),
        "policy_type": "teammate",
        "color": "orange",
        "target_preys": [],
    },
    {
        "init_pos": (8, 8),
        "policy_type": "teammate",
        "color": "yellow",
        "target_preys": [],
    },
]

DEFAULT_PREY_CONFIGS: list[PreyConfigDict] = [
    {
        "type": "easy_prey",
        "territory_dims": (4, 4),
        "territory_left_top_corner": (2, 2),
        "color": "yellow",
    },
    {
        "type": "easy_prey",
        "territory_dims": (4, 4),
        "territory_left_top_corner": (9, 2),
        "color": "yellow",
    },
    {
        "type": "hard_prey",
        "territory_dims": (4, 4),
        "territory_left_top_corner": (2, 9),
        "color": "red",
    },
    {
        "type": "hard_prey",
        "territory_dims": (4, 4),
        "territory_left_top_corner": (9, 9),
        "color": "red",
    },
]
DEFAULT_PREY_TYPES: list[PreyTypeDict] = [
    {
        "name": "easy_prey",
        "policy": "random",
        "capture_reward": 1.0,
        "num_required_preds_capture": 1,
        "num_required_preds_fix": 1,
    },
    {
        "name": "hard_prey",
        "policy": "random",
        "capture_reward": 1.0,
        "num_required_preds_capture": 2,
        "num_required_preds_fix": 1,
    },
]


class PreyPredEnv(MultiGridEnv[NDArray[np.int_]]):
    """
    Environment in which the predator must catch the prey.

    Observation
    -----------
    - Observation is a 3D array of shape (2, width, height).
    - The first layer is the static objects (walls, prey territories).
    - The second layer is the dynamic objects (agents).
    - Encoding:
        - empty: 0
        - wall: 1
        - prey_area: 2
        - predator: 3
        - easy_prey: 4
        - hard_prey: 5

    Actions
    -------
    - 5 discrete actions:
        - 0: stay
        - 1: move left
        - 2: move down
        - 3: move right
        - 4: move up
    - Predator agents movements have to be supplied by the user via `step()`.
    - Prey agents movements are generated by the environment. Defaulted as random.

    Preys
    -----
    - Preys have territories in which they can move.
    - Preys can be of two types: easy and hard.
    - Easy preys require 1 predator to capture and 0 predator to fix (immediately captured).
    - Hard preys require 2 predators to capture and 1 predator to fix.

    Predators
    ---------
    - Predators can capture preys by moving to their position.
    - Predators can fix preys by moving to their position when the prey is captured.
    - Predators can be of two types: ego and teammate.
    - Ego predators are controlled by the user and can capture any prey.
    - Teammate predators are controlled by the environment and can only capture preys designated to them.

    Rewards
    -------
    - Predators receive a reward for capturing preys.
    - The reward is equal to the `capture_reward` of the prey type.

    Termination
    -----------
    - The episode ends when all preys are captured.

    Example
    -------
    ```python
    import gymnasium as gym
    import gym_multigrid

    env = gym.make(
        "multigrid-prey-pred-v0",
        max_episode_steps=100,
        observation_config={
            "encode_prey_areas": True,
        },
        pred_configs=[
            {
                "init_pos": (6, 6),
                "policy_type": "ego",
                "color": "red",
                "target_preys": [],
            },
            {
                "init_pos": (6, 7),
                "policy_type": "teammate",
                "color": "blue",
                "target_preys": [],
            },
        ],
        prey_configs=[
            {
                "type": "easy_prey",
                "territory_dims": (4, 4),
                "territory_left_top_corner": (2, 2),
                "color": "yellow",
            },
            {
                "type": "hard_prey",
                "territory_dims": (4, 4),
                "territory_left_top_corner": (9, 9),
                "color": "red",
            },
        ],
        prey_types=[
            {
                "name": "easy_prey",
                "policy": "random",
                "capture_reward": 1.0,
                "num_required_preds_capture": 1,
                "num_required_preds_fix": 1,
            },
            {
                "name": "hard_prey",
                "policy": "random",
                "capture_reward": 1.0,
                "num_required_preds_capture": 2,
                "num_required_preds_fix": 1,
            },
        ],
    )
    ```
    """

    agents: list[Prey | Predator]  # List of agents in the environment

    def __init__(
        self,
        max_episode_steps: int = 300,
        observation_config: ObservationConfigDict = DEFAULT_OBSERVATION_CONFIG,
        pred_configs: list[PredatorConfigDict] = DEFAULT_PREDATOR_CONFIGS,
        prey_configs: list[PreyConfigDict] = DEFAULT_PREY_CONFIGS,
        prey_types: list[PreyTypeDict] = DEFAULT_PREY_TYPES,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        verbose: bool = False,
    ):
        self.max_episode_steps: int = max_episode_steps
        self.verbose: bool = verbose

        world = PreyPredWorld

        self.observation_config: ObservationConfig = ObservationConfig(
            **observation_config
        )
        self.pred_configs: list[PredatorConfig] = [
            PredatorConfig(**pred_config) for pred_config in pred_configs
        ]
        self.prey_configs: list[PreyConfig] = [
            PreyConfig(**prey_config) for prey_config in prey_configs
        ]
        self.prey_types: list[PreyType] = [
            PreyType(**prey_type) for prey_type in prey_types
        ]

        self.num_preys: int = len(prey_configs)
        self.num_preds: int = len(pred_configs)

        grid_config: GridConfig = GridConfig(
            grid_size=15,
            actions_set=NavigationActions,
            world=world,
        )

        rendering_config: RenderingConfig = RenderingConfig(
            render_mode=render_mode,
            uncached_object_types=["predator"]
            + [prey_type.name for prey_type in self.prey_types],
        )

        partial_obs_config: PartialObsConfig = DEFAULT_FULL_OBS_ENV_PARTIAL_OBS_CONFIG
        agents: list[Predator | Prey] = self._gen_agents(
            self.pred_configs, self.prey_configs, self.prey_types, world
        )
        super().__init__(
            agents=agents,
            **dict(grid_config),
            **dict(rendering_config),
            **dict(partial_obs_config),
        )

    def _gen_agents(
        self,
        pred_configs: list[PredatorConfig],
        prey_configs: list[PreyConfig],
        prey_types: list[PreyType],
        world: World,
    ) -> list[Predator | Prey]:
        """
        Generate the agents for the environment.

        Parameters
        ----------
        pred_configs : list[PredatorConfig]
            The configuration for the predator agents in the environment.
        prey_configs : list[PreyConfig]
            The configuration for the prey agents in the environment.
        prey_types : list[PreyType]
            The configuration for the prey types in the environment.
        """
        agents: list[Predator | Prey] = []

        for i, pred_config in enumerate(pred_configs):
            predator: Predator = Predator(
                world=world,
                index=i,
                pred_config=pred_config,
                view_size=None,
            )
            agents.append(predator)

        for j, prey_config in enumerate(prey_configs):
            prey_type: PreyType = next(
                prey_type
                for prey_type in prey_types
                if prey_type.name == prey_config.type
            )
            prey: Prey = Prey(
                prey_config=prey_config,
                prey_type=prey_type,
                world=world,
                index=j,
                view_size=None,
            )
            agents.append(prey)

        return agents

    def _set_action_space(self) -> tuple[spaces.Space, int | np.integer]:
        self.ac_dim: int | np.integer
        if self.num_preds == 1:
            action_space = spaces.Discrete(len(self.actions))
            ac_dim = action_space.n
        else:
            action_space = spaces.Box(
                low=0,
                high=len(self.actions) - 1,
                shape=(self.num_preds,),
                dtype=np.int64,
            )
            ac_dim = action_space.shape[0]

        return action_space, ac_dim

    def _set_observation_space(self) -> spaces.Box:
        observation_space = spaces.Box(
            low=0,
            high=len(self.world.OBJECT_TO_IDX) - 1,
            shape=(2, self.width, self.height),
            dtype=np.int_,
        )
        return observation_space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.int_], dict[str, Any]]:
        self.step_count: int = 0

        self._reset_gym(seed=seed)
        self._gen_grid(self.width, self.height)

        # Reset the agents. If agent_pos_list is provided, use it to reset the agents
        agent_pos_list: list[tuple[int, int]] | None = None
        if options is not None:
            agent_pos_list = options.get("agent_pos_list", None)
        else:
            pass
        self._reset_agents(agent_pos_list=agent_pos_list)

        obs: NDArray[np.int_] = self._get_obs()
        info: dict[str, Any] = self._get_info()

        return obs, info

    def _get_obs(self) -> NDArray[np.int_]:
        """
        Get the observation of the environment.

        Returns
        -------
        observation : NDArray[np.int_]
            The observation of the environment.
            1st layer: static objects
            2nd layer: dynamic objects (agents)
        """

        obs: NDArray[np.int_] = np.zeros((2, self.width, self.height), dtype=np.int_)

        obs[0, :, :] = self.static_obs

        # Add agent ids to the observation
        for agent in self.agents:
            if not agent.terminated:
                obs[1, agent.pos[1], agent.pos[0]] = self.world.OBJECT_TO_IDX[
                    agent.type
                ]
            else:
                pass

        return obs

    def _get_info(self) -> dict[str, Any]:
        output_dict: dict[str, Any] = {}

        # Get captured prey info
        captured_preys: list[dict[str, Any]] = [
            {
                "prey_id": prey.index,
                "prey_type": prey.type,
                "captured_by": prey.captured_by,
            }
            for prey in self.prey_agents
        ]

        output_dict["captured_preys"] = captured_preys

        return output_dict

    def _gen_grid(self, width: int, height: int) -> None:
        """
        Generate the grid for the environment.

        Parameters
        ----------
        width : int
            The width of the grid.
        height : int
            The height of the grid.
        """
        self.grid = Grid(width, height, self.world)

        # Put white floors everywhere
        self.grid.rect_filled(
            0, 0, width, height, Floor(world=self.world, color="white", type="empty")
        )

        # Put walls
        self.grid.wall_rect(0, 0, width, height)

        static_obs: NDArray[np.int_] = np.zeros(
            (self.width, self.height), dtype=np.int_
        )
        # Place prey territories
        for prey_config in self.prey_configs:
            left_top_corner: tuple[int, int] = prey_config.territory_left_top_corner
            territory_dims: tuple[int, int] = prey_config.territory_dims

            self.grid.rect_filled(
                *left_top_corner,
                *territory_dims,
                Floor(world=self.world, color="light_grey", type="prey_area"),
            )

        # Add static objects ids to the observation
        for i in range(self.width):
            for j in range(self.height):
                cell: None | WorldObj = self.grid.get(i, j)
                if cell is None:
                    static_obs[j, i] = self.world.OBJECT_TO_IDX["empty"]
                elif (
                    cell.type == "prey_area"
                    and not self.observation_config.encode_prey_areas
                ):
                    static_obs[j, i] = self.world.OBJECT_TO_IDX["empty"]
                else:
                    static_obs[j, i] = self.world.OBJECT_TO_IDX[cell.type]

        self.static_obs: NDArray[np.int_] = static_obs

        self.init_grid: Grid = self.grid.copy()

    def _reset_agents(
        self, agent_pos_list: list[tuple[int, int]] | None = None
    ) -> None:
        """
        Reset the agents in the environment.
        """
        for agent in self.agents:
            agent.reset(self.np_random)

        use_agent_pos_list: bool = agent_pos_list is not None
        if agent_pos_list is not None:
            if len(agent_pos_list) != len(self.agents):
                warnings.warn(
                    f"""Invalid agent position list: Expected {len(self.agents)} positions, got {len(agent_pos_list)}.
                    Resetting agents to their initial positions.""",
                    UserWarning,
                )
                use_agent_pos_list = False
            else:
                pass

        for agent in self.agents:
            agent_pos: tuple[int, int] = (
                agent.get_init_pos()
                if not use_agent_pos_list or not isinstance(agent_pos_list, list)
                else agent_pos_list.pop(0)
            )
            self.place_agent(agent, agent_pos)

    def step(
        self, action: NDArray[np.int_] | np.int_
    ) -> tuple[NDArray[np.int_], float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Parameters
        ----------
        action : int
            The action to take.

        Returns
        -------
        observation : NDArray[np.int_]
            The observation of the environment.
        reward : float
            The reward for the action.
        terminated: bool
            Whether the episode has terminated.
        terminated: bool
            Whether the episode has terminated.
        info : dict[str, Any]
            Additional information about the environment.
        """
        terminated: bool = False
        reward: float = 0

        actions: list[int] = np.array(action).flatten().astype(np.int_).tolist()
        prey_actions = [
            agent.act(self._get_obs(), {})
            for agent in self.agents[self.num_preds :]
            if isinstance(agent, Prey)
        ]
        all_actions: list[int] = actions + prey_actions
        # Order to apply the actions to the agents
        order: NDArray[np.int_] = self.np_random.permutation(len(self.agents))

        for i in order:
            agent: Prey | Predator = self.agents[i]
            assert isinstance(agent, (Prey, Predator))
            act: int = all_actions[i]

            if agent.terminated:
                continue
            else:
                next_pos: tuple[int, int] = self._get_next_pos(agent, act)
                next_cell: None | WorldObj = self.grid.get(*next_pos)

                if isinstance(next_cell, WorldObj) and not next_cell.can_overlap():
                    continue
                else:
                    if isinstance(agent, Prey) and (
                        agent.check_fix_condition(self.grid)
                        or not agent.check_pos_in_territory(next_pos)
                    ):
                        continue
                    else:
                        pass

                    # Move agent
                    self.grid.set(*next_pos, agent)
                    self.grid.set(*agent.pos, self.init_grid.get(*agent.pos))
                    # Change the dir of the agent
                    if agent.pos != next_pos:
                        dir_vec: NDArray[np.int_] = np.array(next_pos) - np.array(
                            agent.pos
                        )
                        agent.dir = agent.vec2dir(dir_vec)
                    agent.pos = next_pos

                    # Update the agent's bg_color
                    init_grid_cell: WorldObj | None = self.init_grid.get(*agent.pos)
                    if init_grid_cell is not None:
                        agent.bg_color = init_grid_cell.color

                    # Determine rewards and remove captured preys from the grid
                    for prey in self.agents[self.num_preds :]:
                        assert isinstance(prey, Prey)
                        if prey.terminated:
                            continue
                        else:
                            prey_captured: bool
                            pred_ids: list[int]
                            prey_captured, pred_ids = prey.check_capture_condition(
                                self.grid
                            )
                            if prey_captured:
                                captured_by_designated_preds: bool = True

                                for pred_id in pred_ids:
                                    pred = self.agents[pred_id]
                                    assert isinstance(pred, Predator)
                                    if pred.target_preys[prey.index] == 1:
                                        continue
                                    else:
                                        captured_by_designated_preds = False
                                        break

                                if captured_by_designated_preds:
                                    prey.terminated = True
                                    prey.captured_by = pred_ids
                                    reward += prey.prey_type.capture_reward
                                    self.grid.set(
                                        *prey.pos, self.init_grid.get(*prey.pos)
                                    )
                                else:
                                    pass
                            else:
                                continue

        # Terminate the episode if all the prey are captured
        terminated = all(prey.terminated for prey in self.agents[self.num_preds :])
        truncated = self.step_count >= self.max_episode_steps

        self.step_count += 1

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_next_pos(self, agent: Predator | Prey, action: int) -> tuple[int, int]:
        # Check that self.actions has the required enum variables

        self.actions: Type[NavigationActions]

        match action:
            case self.actions.STAY:
                next_pos = agent.pos
            case self.actions.LEFT:
                next_pos = agent.west_pos(in_tuple=True)
            case self.actions.RIGHT:
                next_pos = agent.east_pos(in_tuple=True)
            case self.actions.UP:
                next_pos = agent.north_pos(in_tuple=True)
            case self.actions.DOWN:
                next_pos = agent.south_pos(in_tuple=True)
            case _:
                raise ValueError(f"Invalid action: {action}")

        return next_pos

    def _move_prey(self, prey: Prey, next_pos: tuple[int, int]) -> None:
        """
        Move the prey agent to the next position.

        Parameters
        ----------
        prey : Prey
            The prey agent to move.
        next_pos : tuple[int, int]
            The next position to move the agent to.
        """
        next_cell: None | WorldObj = self.grid.get(*next_pos)

        agent_moves: bool = False

    def _move_pred(self, pred: Predator, next_pos: tuple[int, int]) -> float:
        """
        Move the predator agent to the next position.

        Parameters
        ----------
        pred : Predator
            The predator agent to move.
        next_pos : tuple[int, int]
            The next position to move the agent to.

        Returns
        -------
        reward : float
            The reward for the action.
        """
        raise NotImplementedError(
            "Predator movement logic is not implemented. Use `step()` to move the predator agents."
        )

    @property
    def prey_agents(self) -> list[Prey]:
        return [
            agent for agent in self.agents[self.num_preds :] if isinstance(agent, Prey)
        ]

    @property
    def pred_agents(self) -> list[Predator]:
        return [
            agent
            for agent in self.agents[: self.num_preds]
            if isinstance(agent, Predator)
        ]


class BasePolicy:
    def __init__(
        self,
        action_set: Type[enum.IntEnum] = NavigationActions,
        dir_to_vec: list[NDArray[np.int_]] = NAV_DIR_TO_VEC,
        random_generator: np.random.Generator | None = None,
    ):
        self.dir_to_vec: list[NDArray[np.int_]] = dir_to_vec
        self.action_set: Type[enum.IntEnum] = action_set
        self.random_generator: np.random.Generator = (
            random_generator
            if random_generator is not None
            else np.random.default_rng()
        )

    def act(self, action_option: dict[str, Any]) -> int:
        """
        Choose an action for the predator agent.

        Parameters
        ----------
        observation : NDArray[np.int_]
            The observation of the environment.
        options : GreedyPredatorActionOption
            Additional options for the action.

        Returns
        -------
        action : int
            The action to take.
        """
        ...

    def vec2dir(self, vec: NDArray[np.int_]) -> int:
        """
        Convert a vector to a direction.

        Parameters
        ----------
        vec : NDArray[np.int_]
            The vector to convert.

        Returns
        -------
        dir : int
            The direction corresponding to the vector.
        """
        for dir, dir_vec in enumerate(self.dir_to_vec):
            if np.array_equal(dir_vec, vec):
                return dir
        raise ValueError(f"Invalid vector: {vec}")


class GreedyPredatorActionOptionDict(TypedDict):
    current_pos: tuple[int, int]
    preys: list[Prey]
    grid: Grid


class GreedyPredatorActionOption(BaseModel):
    current_pos: tuple[int, int]
    preys: list[Prey]
    grid: Grid
    model_config = ConfigDict(arbitrary_types_allowed=True)


class GreedyPredatorPolicy(BasePolicy):
    def __init__(
        self,
        target_list: list[Literal[0, 1]],
        random_prob: float = 0.1,
        action_set: Type[enum.IntEnum] = NavigationActions,
        dir_to_vec: list[NDArray[np.int_]] = NAV_DIR_TO_VEC,
        random_generator: np.random.Generator | None = None,
    ):
        self.target_list: list[Literal[0, 1]] = target_list
        self.random_prob: float = random_prob
        super().__init__(
            action_set=action_set,
            dir_to_vec=dir_to_vec,
            random_generator=random_generator,
        )

    def act(self, action_option: dict[str, Any]) -> int:
        """
        Choose an action for the predator agent.

        Parameters
        ----------
        observation : NDArray[np.int_]
            The observation of the environment.
        options : GreedyPredatorActionOption
            Additional options for the action.

        Returns
        -------
        action : int
            The action to take.
        """
        checked_action_option: GreedyPredatorActionOption = GreedyPredatorActionOption(
            **action_option
        )
        current_pos: tuple[int, int] = checked_action_option.current_pos
        preys: list[Prey] = checked_action_option.preys
        grid: Grid = checked_action_option.grid

        target_prey: None | Prey = None

        for prey, target_candidate in zip(preys, self.target_list):
            if target_candidate == 1 and not prey.terminated:
                target_prey = prey
                break
            else:
                pass

        # If there is no target prey, choose a random action
        # Else, choose the greedy action to move towards the target prey with random probability `random_prob`
        act_randomly: bool = (
            False
            if target_prey is not None
            and (
                target_prey.pos_in_neighbor(current_pos)
                or self.random_generator.random() >= self.random_prob
            )
            else True
        )

        action: int

        match act_randomly:
            case True:
                action = self.random_generator.integers(0, len(self.action_set))
            case False:
                assert target_prey is not None, "Target prey should not be None"
                path: list[tuple[int, int]] = a_star(current_pos, target_prey.pos, grid)
                next_pos: tuple[int, int] = path[1] if len(path) > 1 else current_pos
                dir_vec: NDArray[np.int_] = np.array(next_pos) - np.array(current_pos)
                action = self.vec2dir(dir_vec)

        return action
