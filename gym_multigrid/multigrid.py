import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    SupportsFloat,
    Type,
    TypeVar,
)

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.spaces.space import Space
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from gym_multigrid.core.agent import Actions, Agent, AgentT, DefaultActions
from gym_multigrid.core.constants import OBJECT_TO_STR, TILE_PIXELS
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Door, WorldObj
from gym_multigrid.core.world import DefaultWorld, World
from gym_multigrid.typing import Position, Size
from gym_multigrid.utils.window import Window

EnvType = TypeVar("EnvType", bound="MultiGridEnv")
SpaceType = TypeVar("SpaceType", bound=Space)


class ObservationMode(Generic[EnvType, SpaceType, ObsType], ABC):
    static_obs: ObsType

    @abstractmethod
    def observation_space(self, env: EnvType) -> SpaceType: ...

    """
    Define the observation space of the environment.

    Parameters
    ----------
    env : gym.Env[ObsType, ActType]
        The environment

    Returns
    -------
    observation_space: gym.Space
        The observation space of the environment
    """

    @abstractmethod
    def create_observation(self, env: EnvType) -> ObsType: ...

    """
    Create an observation from the environment.

    Parameters
    ----------
    env : gym.Env[ObsType, ActType]
        The environment

    Returns
    -------
    observation: ObsType
        The observation
    """

    def save_static_obs(
        self, env: EnvType, options: dict[str, Any] | None = None
    ) -> None:
        """
        Save the static observation of the environment.
        This is used to save the observation for later use.
        """
        pass


class GridConfig(BaseModel):
    grid_size: int | None = None
    width: int | None = None
    height: int | None = None
    world: World = Field(default=DefaultWorld)
    actions_set: Type[Actions] = Field(default=DefaultActions)


class RenderingConfig(BaseModel):
    """
    Attributes
    ----------
    render_mode : Literal["human", "rgb_array"] = "rgb_array"
        Rendering mode
    uncached_object_types : list[str] = []
        List of object types that should not be cached in the rendering cache
    close_window : bool = False
        Whether to close the rendering window
    tile_size : int = TILE_PIXELS
        Size of the tiles in the rendering
    """

    render_mode: Literal["human", "rgb_array"] = "rgb_array"
    uncached_object_types: list[str] = []
    close_window: bool = False
    tile_size: int = TILE_PIXELS


class PartialObsConfig(BaseModel):
    partial_obs: bool = False
    agent_view_size: int | None = None
    see_through_walls: bool = False
    highlight_visible_cells: bool = False


DEFAULT_FULL_OBS_ENV_PARTIAL_OBS_CONFIG: PartialObsConfig = PartialObsConfig(
    partial_obs=False,
    agent_view_size=None,
    see_through_walls=False,
    highlight_visible_cells=False,
)


class MultiGridEnv(gym.Env[ObsType, np.int64 | NDArray[np.int64]]):
    """
    2D grid world game environment
    """

    # Setup and env properties
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "video.frames_per_second": 10,
        "observation_modes": {},  # type: dict[str, ObservationMode]
    }
    grid: Grid

    def __init__(
        self,
        agents: list[AgentT],
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        world: World = DefaultWorld,
        actions_set: Type[Actions] = DefaultActions,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        uncached_object_types: list[str] = [],
        close_window: bool = False,
        tile_size: int = TILE_PIXELS,
        partial_obs: bool = False,
        agent_view_size: int | None = None,
        see_through_walls: bool = False,
        highlight_visible_cells: bool = False,
        max_steps: int | None = None,
    ) -> None:
        """
        Initialize a new grid world environment

        Parameters
        ----------
        agents : list[gym_multigrid.core.agent.Agent]
            List of agents in the environment
        grid_size : int | None = None
            Size of the grid (if square).
            If None, width and height must be set.
        width : int | None = None
            Width of the grid
        height : int | None = None
            Height of the grid
        world : World = DefaultWorld
            World object that defines the objects in the environment
        actions_set : Type[Actions] = DefaultActions
            Actions available to the agents
        render_mode : Literal["human", "rgb_array"] = "rgb_array"
            Rendering mode
        uncached_object_types : list[str] = []
            List of object types that should not be cached in the rendering cache
        close_window : bool = False
            Whether to close the rendering window
        tile_size : int = TILE_PIXELS
            Size of the tiles in the rendering
        partial_obs : bool = False
            Whether agents have partial or full observation.
            If True, the agent's observation is a square view area centered on the agent, specified by agent_view_size.
        agent_view_size : int | None = None
            Size of the square view area centered on the agent
        see_through_walls : bool = False
            Whether agents can see through walls
        highlight_visible_cells : bool = False
            Whether to highlight the cells visible to the agent
        max_steps : int | None = None
            Maximum number of steps per episode.
            If `None`, `truncated` returned by the `step` method will always be False.
        """
        self.agents = agents
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.uncached_object_types: list[str] = uncached_object_types
        self.close_window: bool = close_window
        self.tile_size: int = tile_size

        # Does the agents have partial or full observation?
        self.partial_obs: bool = partial_obs
        self.see_through_walls: bool = see_through_walls and self.partial_obs
        self.highlight_visible_cells: bool = highlight_visible_cells
        if self.partial_obs and agent_view_size is None:
            warnings.warn(
                "Partial observation is enabled but agent_view_size is not set. Defaulting to 7.",
                UserWarning,
            )
            agent_view_size = 7
        else:
            pass

        self.agent_view_size = agent_view_size

        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size
        else:
            assert width != None and height != None

        self.width: int = width
        self.height: int = height

        # Action enumeration for this environment
        self.actions: Type[Actions] = actions_set

        # Actions are discrete integer values

        self.world = world

        self.action_space, self.ac_dim = self._set_action_space()
        self.observation_space = self._set_observation_space()

        if self.observation_space is spaces.Box:
            self.ob_dim = np.prod(self.observation_space.shape)
        else:
            pass

        # Range of possible rewards
        self.reward_range: tuple[int, int] = (0, 1)

        # Window to use for human rendering mode
        self.window: Window | None = None

        # Environment configuration
        self.max_steps: int | None = max_steps
        if self.max_steps is not None:
            warnings.warn(
                """
                `max_steps` will be deprecated in the base class in the future.
                Please use `gymnasium.wrappers.TimeLimit` instead to limit the number of steps in an episode.
                If you want to keep using `max_steps` for some purpose, please implement it in your own child classes.
                """,
                DeprecationWarning,
            )
        else:
            pass

        # Define the empty grid. _gen_grid is supposed to fill this up
        self.grid = Grid(width, height, world)

    def _set_action_space(self) -> tuple[spaces.Space, int | np.integer]:
        self.ac_dim: int | np.integer
        if len(self.agents) == 1:
            action_space = spaces.Discrete(len(self.actions))
            ac_dim = action_space.n
        else:
            action_space = spaces.Box(
                low=0,
                high=len(self.actions) - 1,
                shape=(len(self.agents),),
                dtype=np.int64,
            )
            ac_dim = action_space.shape[0]

        return action_space, ac_dim

    def _set_observation_space(self) -> spaces.Space:
        if self.partial_obs:
            assert self.agent_view_size is not None
            observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.agent_view_size,
                    self.agent_view_size,
                    self.world.encode_dim,
                ),
                dtype=np.int_,
            )

        else:
            observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.width, self.height, self.world.encode_dim),
                dtype=np.int_,
            )

        return observation_space

    def _reset_gym(self, seed: int | None = None) -> None:
        super().reset(seed=seed)

    def _reset_agents(self) -> None:
        """
        Reset the agents to their initial positions
        """

        # for a in self.agents:
        #     a.reset()
        #     self.place_agent(a)

        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        # It is recommended to use the random number generator self.np_random
        # that is provided by the environment’s base class, gymnasium.Env.
        # If you only use this RNG, you do not need to worry much about seeding,
        # but you need to remember to call ``super().reset(seed=seed)`` to make
        # sure that gymnasium.Env correctly seeds the RNG
        self._reset_gym(seed=seed)

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        # if state is given in options, then use it to generate the grid with given state as the initial state
        self._gen_grid(self.width, self.height)
        self.init_grid: Grid = self.grid.copy()

        # Agent status should be reset inside self._gen_grid
        self._reset_agents()

        # Step count since episode start
        self.step_count: int = 0

        # Return first observation
        # if self.partial_obs:
        #     obs = self.gen_obs()
        # else:
        #     obs = [
        #         self.grid.encode_for_agents(agent_pos=self.agents[i].pos)
        #         for i in range(len(self.agents))
        #     ]
        # obs = [self.world.normalize_obs * ob for ob in obs]
        # info = self._get_info()
        # return obs, info

        return None, None  # type: ignore

    def _get_obs(self) -> ObsType:
        """
        Get the current observation of the environment.
        This method is used to get the observation after reset or step.
        """
        ...

    def _get_info(self):
        return {}

    @property
    def steps_remaining(self):
        assert self.max_steps is not None, (
            "steps_remaining is only available if max_steps is set"
        )
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        str = ""

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                # if i == self.agent_pos[0] and j == self.agent_pos[1]:
                # str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                # continue

                c = self.grid.get(i, j)

                if c == None:
                    str += "  "
                    continue

                if isinstance(c, Door):
                    if c.is_open:
                        str += "__"
                    elif c.is_locked:
                        str += "L" + c.color[0].upper()
                    else:
                        str += "D" + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += "\n"

        return str

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height, self.world)
        assert False, "_gen_grid needs to be implemented by each environment"

    # Env step logic
    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_build(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_special_moves(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_switch(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _reward(self, current_agent, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        assert isinstance(self.max_steps, int)
        rewards[current_agent] += reward - 0.9 * (self.step_count / self.max_steps)
        return rewards

    def place_obj(
        self,
        obj: WorldObj,
        top: Position | None = None,
        size: Size | None = None,
        reject_fn: Callable[["MultiGridEnv", Position], bool] | None = None,
        max_tries: float = math.inf,
    ):
        """
        Place an object at an empty position in the grid

        Parameters
        ----------
        obj : WorldObj
            The object to place in the grid
        top : Position | None = None
            The top-left position of the rectangle where to place the object.
            If None, the whole grid will be used.
        size : Size | None = None
            The size of the rectangle where to place the object (width, height).
            If None, the whole grid will be used.
        reject_fn : Callable[["MultiGridEnv", Position], bool] | None = None
            A function that takes the environment and a position as input and returns True if the position should
            be rejected for placing the object. If None, no filtering is applied.
        max_tries : float = math.inf
            Maximum number of tries to place the object at a random position.

        Returns
        -------
        pos : Position
            The position where the object was placed in the grid.
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos: Position = (
                self._rand_int(top[0], min(top[0] + size[0] - 1, self.grid.width - 1)),
                self._rand_int(top[1], min(top[1] + size[1] - 1, self.grid.height - 1)),
            )

            # Don't place the object on top of another object
            if self.grid.get(*pos):
                continue
            # Check if there is a filtering criterion
            elif reject_fn and reject_fn(self, pos):
                continue
            else:
                break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.pos = pos

        return pos

    def put_obj(self, obj: WorldObj, i: int, j: int):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.pos = (i, j)

    def place_agent(
        self,
        agent: Agent,
        pos: Position | None = None,
        top: Position | None = None,
        size: Size | None = None,
        rand_dir: bool = False,
        max_tries: float = math.inf,
        reset_agent_status: bool = False,
    ) -> Position:
        """
        Set the agent's starting point at an empty position in the grid and reset the agent's state

        Parameters
        ----------
        agent : Agent
            The agent to place in the grid
        pos : Position | None = None
            The position to place the agent at. If None, a random position will be chosen.
        top : Position | None = None
            The top-left position of the rectangle where to place the agent.
            If None, the whole grid will be used.
        size : Size | None = None
            The size of the rectangle where to place the agent.
            If None, the whole grid will be used.
        rand_dir : bool = False
            Whether to randomly set the agent's direction.
            If False, the agent will face to direction 3.
        max_tries : float = math.inf
            Maximum number of tries to place the agent at a random position.
        reset_agent_status : bool = False
            Whether to reset the agent's status (e.g., position, direction).
            If False, the agent's position and direction will not be reset.
        """
        if reset_agent_status:
            agent.reset()
        else:
            pass

        if pos is not None and pos != (-1, -1):
            agent.pos = pos
            self.put_obj(agent, i=pos[0], j=pos[1])
        else:
            agent.pos = None
            pos = self.place_obj(agent, top, size, max_tries=max_tries)
            agent.pos = pos
            agent.init_pos = pos

        if rand_dir:
            agent.dir = self._rand_int(0, 3)
        else:
            agent.dir = 3

        agent.init_dir = agent.dir

        return pos

    def place_object(
        self,
        obj: WorldObj,
        pos: Position | None = None,
        top: Position | None = None,
        size: Size | None = None,
        reject_fn: Callable[["MultiGridEnv", Position], bool] | None = None,
        max_tries: float = math.inf,
        reset_obj_status: bool = False,
    ) -> Position:
        """
        Place an object at an empty position in the grid.

        Parameters
        ----------
        obj : WorldObj
            The object to place in the grid.
        pos : Position | None = None
            The position to place the object at. If None, a random position will be chosen.
        top : Position | None = None
            The top-left position of the rectangle where to place the object.
            If None, the whole grid will be used.
        size : Size | None = None
            The size of the rectangle where to place the object.
            If None, the whole grid will be used.
        reject_fn : Callable[["MultiGridEnv", Position], bool] | None = None
            A function that takes the environment and a position as input and returns True if the position should
            be rejected for placing the object. If None, no filtering is applied.
        max_tries : float = math.inf
            Maximum number of tries to place the object at a random position.
        reset_obj_status : bool = False
            Whether to reset the object's status (e.g., position).
        """
        if reset_obj_status:
            obj.reset()
        else:
            pass

        if pos is not None and pos != (-1, -1):
            obj.pos = pos
            self.put_obj(obj, i=pos[0], j=pos[1])
        else:
            pos = self.place_obj(obj, top, size, reject_fn, max_tries=max_tries)
            obj.pos = pos
            obj.init_pos = pos

        return pos

    def agent_sees(self, a, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """
        raise NotImplementedError(
            "agent_sees is not implemented in the base class. "
            "Please implement it in your own environment."
        )
        coordinates = a.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid, _ = self.grid.decode(obs["image"])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Example method showing potential implementation of the step method.
        Implement this method in your own environment.
        """
        self.step_count += 1

        raise NotImplementedError(
            "step is not implemented in the base class. "
            "Please implement it in your own environment."
        )
        # order = np.random.permutation(len(actions))

        # rewards = np.zeros(len(actions))
        # terminated = False
        # truncated = False
        # for i in order:
        #     if (
        #         self.agents[i].terminated
        #         or self.agents[i].paused
        #         or not self.agents[i].started
        #         or actions[i] == self.actions.still
        #     ):
        #         continue

        #     # Get the position in front of the agent
        #     fwd_pos = self.agents[i].front_pos

        #     # Get the contents of the cell in front of the agent
        #     fwd_cell = self.grid.get(*fwd_pos)

        #     # Rotate left
        #     if actions[i] == self.actions.LEFT:
        #         self.agents[i].dir -= 1
        #         if self.agents[i].dir < 0:
        #             self.agents[i].dir += 4

        #     # Rotate right
        #     elif actions[i] == self.actions.RIGHT:
        #         self.agents[i].dir = (self.agents[i].dir + 1) % 4

        #     # Move forward
        #     elif actions[i] == self.actions.forward:
        #         if fwd_cell is not None:
        #             if fwd_cell.type == "goal":
        #                 terminated = True
        #                 rewards = self._reward(i, rewards, 1)
        #             elif fwd_cell.type == "switch":
        #                 self._handle_switch(i, rewards, fwd_pos, fwd_cell)
        #         elif fwd_cell is None or fwd_cell.can_overlap():
        #             self.grid.set(*fwd_pos, self.agents[i])
        #             self.grid.set(*self.agents[i].pos, None)
        #             self.agents[i].pos = fwd_pos
        #         self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

        #     elif "build" in self.actions.available and actions[i] == self.actions.build:
        #         self._handle_build(i, rewards, fwd_pos, fwd_cell)

        #     # Pick up an object
        #     elif actions[i] == self.actions.pickup:
        #         self._handle_pickup(i, rewards, fwd_pos, fwd_cell)

        #     # Drop an object
        #     elif actions[i] == self.actions.drop:
        #         self._handle_drop(i, rewards, fwd_pos, fwd_cell)

        #     # Toggle/activate an object
        #     elif actions[i] == self.actions.toggle:
        #         if fwd_cell:
        #             fwd_cell.toggle(self, fwd_pos)

        #     # Done action (not used by default)
        #     elif actions[i] == self.actions.done:
        #         pass

        #     else:
        #         assert False, "unknown action"

        # if self.max_steps is not None and self.step_count >= self.max_steps:
        #     truncated = True

        # if self.partial_obs:
        #     obs = self.gen_obs()
        # else:
        #     obs = [
        #         self.grid.encode_for_agents(agent_pos=self.agents[i].pos)
        #         for i in range(len(actions))
        #     ]

        # obs = [self.world.normalize_obs * ob for ob in obs]
        # info = self._get_info()
        # return obs, rewards, terminated, truncated, info

    # Agent obs logic
    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agents.
        This method also outputs a visibility mask telling us which grid
        cells the agents can actually see.
        """

        grids = []
        vis_masks = []

        for a in self.agents:
            topX, topY, botX, botY = a.get_view_exts()

            grid = self.grid.slice(topX, topY, a.view_size, a.view_size)

            for i in range(a.dir + 1):
                grid = grid.rotate_left()

            # Process occluders and visibility
            # Note that this incurs some performance cost
            if self.partial_obs and not self.see_through_walls:
                if a.view_size is None:
                    raise ValueError(
                        "Agent view size must be set for partial observation"
                    )
                vis_mask = grid.process_vis(
                    agent_pos=(a.view_size // 2, a.view_size - 1)
                )
            else:
                vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

            grids.append(grid)
            vis_masks.append(vis_mask)

        return grids, vis_masks

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grids, vis_masks = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        obs = [
            grid.encode_for_agents(
                self.world, [grid.width // 2, grid.height - 1], vis_mask
            )
            for grid, vis_mask in zip(grids, vis_masks)
        ]

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS // 2):
        """
        Render an agent observation for visualization
        """
        raise NotImplementedError(
            "get_obs_render is not implemented in the base class. "
            "Please implement it in your own environment."
        )

        grid, vis_mask = self.grid.decode(obs)

        # Render the whole grid
        img = grid.render(self.world, tile_size, highlight_mask=vis_mask)

        return img

    # Randomizing
    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high, endpoint=True)

    def _rand_float(self, low: float, high: float) -> float:
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """
        Generate random boolean value
        """

        return self.np_random.choice([True, False])

    def _rand_elem(self, iterable: Iterable) -> object:
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst) - 1)
        return lst[idx]

    def _rand_subset(self, iterable: Iterable, num_elems: int) -> list:
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_pos(self, x_low: int, x_high: int, y_low: int, y_high: int) -> Position:
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.integers(x_low, x_high),
            self.np_random.integers(y_low, y_high),
        )

    # Env rendering
    def render(self):
        """
        Render the whole-grid human view
        """

        if self.close_window:
            if self.window:
                self.window.close()
            return

        if self.render_mode == "human" and not self.window:
            self.window = Window("gym_multigrid")
            self.window.show(block=False)

        highlight_masks = {}
        if self.highlight_visible_cells:
            # Compute which cells are visible to the agent
            _, vis_masks = self.gen_obs_grid()

            highlight_masks = {
                (i, j): [] for i in range(self.width) for j in range(self.height)
            }

            for i, a in enumerate(self.agents):
                # Compute the world coordinates of the bottom-left corner
                # of the agent's view area
                if a.view_size is None:
                    raise ValueError(
                        "Agent view size must be set for highlighting visible cells"
                    )
                f_vec = a.dir_vec
                r_vec = a.right_vec
                top_left = (
                    a.pos + f_vec * (a.view_size - 1) - r_vec * (a.view_size // 2)
                )

                # Mask of which cells to highlight

                # For each cell in the visibility mask
                for vis_j in range(0, a.view_size):
                    for vis_i in range(0, a.view_size):
                        # If this cell is not visible, don't highlight it
                        if not vis_masks[i][vis_i, vis_j]:
                            continue

                        # Compute the world coordinates of this cell
                        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                        if abs_i < 0 or abs_i >= self.width:
                            continue
                        if abs_j < 0 or abs_j >= self.height:
                            continue

                        # Mark this cell to be highlighted
                        highlight_masks[abs_i, abs_j].append(i)

        # Render the whole grid
        img = self.grid.render(
            self.tile_size,
            highlight_masks=highlight_masks if self.highlight_visible_cells else None,
            uncached_object_types=self.uncached_object_types,
        )

        if self.render_mode == "human":
            if self.window is None:
                self.window = Window("gym_multigrid")
                self.window.show(block=False)
            self.window.show_img(img)

        return img
