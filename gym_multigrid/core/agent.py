import enum
import math
from typing import Any, Literal, Optional, Type, TypeAlias, TypeVar

import numpy as np
from gym_multigrid.core.constants import DIR_TO_VEC
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import WorldObj
from gym_multigrid.core.world import World
from gym_multigrid.policy.base import AgentPolicy
from gym_multigrid.typing_utils import Position
from gym_multigrid.utils.rendering import (
    fill_coords,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)
from numpy import ndarray
from numpy.typing import NDArray

Actions: TypeAlias = enum.IntEnum
AgentT = TypeVar("AgentT", bound="Agent", covariant=True)


class DefaultActions(enum.IntEnum):
    """Set of DefaultActions

    Parameters
    ----------
    enum : IntEnum
        Base class for creating enumerated constants that are also subclasses of int.
    """

    STILL = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    PICKUP = 4
    DROP = 5
    TOGGLE = 6
    DONE = 7


class GridActions(enum.IntEnum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class CollectActions(enum.IntEnum):
    """Set of actions available for the agents in Collect Game environment

    Parameters
    ----------
    enum : IntEnum
        Base class for creating enumerated constants that are also subclasses of int.
    """

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class WildfireActions(enum.IntEnum):
    """Set of actions available for the agents in Wildfire environment

    Parameters
    ----------
    enum : IntEnum
        Base class for creating enumerated constants that are also subclasses of int.
    """

    STILL = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


class CtfActions(enum.IntEnum):
    STAY = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3
    UP = 4


class FRActions(enum.IntEnum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class MazeActions(enum.IntEnum):
    STAY = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3
    UP = 4


class NavigationActions(enum.IntEnum):
    STAY = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3
    UP = 4


class LBFActions(enum.IntEnum):
    # matches the order from original LBF action set
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    LOAD = 5


class AlternativeNavigationActions(enum.IntEnum):
    # matches the order from original LBF action set
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class SimpleNavigationActions(enum.IntEnum):
    # matches the order from original LBF action set
    STAY = 0
    LEFT = 1
    RIGHT = 2


class Agent(WorldObj):
    """Defines the class for an agent in the environment"""

    def __init__(
        self,
        world: World,
        index: int = 0,
        view_size: int | None = None,
        actions: Type[Actions] = DefaultActions,
        dir_to_vec: list[NDArray] = DIR_TO_VEC,
        color: str | None = None,
        bg_color: str | None = None,
        type: str = "agent",
    ) -> None:
        """Initialize the agent object

        Parameters
        ----------
        world : World
            the world within which grid is situated
        index : int, optional
            a number useful to identify the instantiated agent, by default 0
        view_size : int, optional
            the side length of agent's obs grid, if partial observability holds, by default 7
        actions : Type[Actions], optional
            set of actions available to the agent, by default DefaultActions
        dir_to_vec : list[NDArray], optional
            map of agent direction indices to vectors, by default DIR_TO_VEC
        color : str | None, optional
            color of the agent, by default None
        bg_color : str | None, optional
            background color of the tile containing agent, by default None
        type : str, optional
            type of the world object, by default "agent"
        """
        if color is None:
            color = world.IDX_TO_COLOR[index]
        else:
            pass

        super().__init__(world, type, color, bg_color)
        self.dir: int = 0
        self.init_dir: int = 0
        self.index = index
        self.view_size = view_size
        self.carrying = None
        self.terminated: bool = False
        self.started: bool = True
        self.paused: bool = False
        self.collided: bool = False
        self.actions = actions
        self.world = world
        self.dir_to_vec = dir_to_vec

    def reset(self, options: dict[str, Any] | None = None) -> None:
        """
        Reset the agent to its initial state
        The reset attributes of the agent are:
        - pos: None
        - dir: None
        - init_dir: None
        - carrying: None
        - terminated: False
        - started: True
        - paused: False
        - collided: False
        """
        super().reset()
        self.dir = 0
        self.init_dir = 0
        self.carrying = None
        self.terminated = False
        self.started = True
        self.paused = False
        self.collided = False

    def render(self, img) -> None:
        """Render the agent at its current position

        Parameters
        ----------
        img : NDArray
            the image to render the agent in
        """
        c = self.world.COLORS[self.color]
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        # Rotate the agent based on its direction
        assert self.dir is not None

        dir_vec: NDArray[np.int_] = self.dir_to_vec[self.dir]
        orientation: int
        if np.array_equal(dir_vec, np.array([1, 0])):
            orientation = 0
        elif np.array_equal(dir_vec, np.array([0, 1])):
            orientation = 1
        elif np.array_equal(dir_vec, np.array([-1, 0])):
            orientation = 2
        elif np.array_equal(dir_vec, np.array([0, -1])):
            orientation = 3
        else:
            raise ValueError("Invalid direction vector")

        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * orientation)
        fill_coords(
            img, tri_fn, c, self.world.COLORS[self.bg_color] if self.bg_color else None
        )

    def encode(self, current_agent: bool = False) -> tuple[int, ...]:
        """Encode a description of this object as a 3-tuple of integers

        Parameters
        ----------
        current_agent : bool, optional
            whether the agent is the current agent, by default False
        """
        if self.world.encode_dim == 3:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                self.dir,
            )
        elif self.carrying:
            if current_agent:
                return (
                    self.world.OBJECT_TO_IDX[self.type],
                    self.world.COLOR_TO_IDX[self.color],
                    self.world.OBJECT_TO_IDX[self.carrying.type],
                    self.world.COLOR_TO_IDX[self.carrying.color],
                    self.dir,
                    1,
                )
            else:
                return (
                    self.world.OBJECT_TO_IDX[self.type],
                    self.world.COLOR_TO_IDX[self.color],
                    self.world.OBJECT_TO_IDX[self.carrying.type],
                    self.world.COLOR_TO_IDX[self.carrying.color],
                    self.dir,
                    0,
                )

        else:
            if current_agent:
                return (
                    self.world.OBJECT_TO_IDX[self.type],
                    self.world.COLOR_TO_IDX[self.color],
                    0,
                    0,
                    self.dir,
                    1,
                )
            else:
                return (
                    self.world.OBJECT_TO_IDX[self.type],
                    self.world.COLOR_TO_IDX[self.color],
                    0,
                    0,
                    self.dir,
                    0,
                )

    def move(
        self,
        next_pos: Position,
        grid: Grid,
        init_grid: Grid | None = None,
        dummy_move: bool = False,
        bg_color: str | None = None,
    ) -> None:
        """Move the agent to a new position

        Parameters
        ----------
        next_pos : Position
            the position to move the agent to
        grid : Grid
            the grid to move the agent in
        init_grid : Grid | None = None
            the initial grid before agent is moved, by default None
        dummy_move : bool = False
            whether the move is a dummy move, by default False
        bg_color : str | None = None
            the background color of the tile containing agent, by default None
        """
        if self.pos is not None:
            direction = np.array(next_pos) - np.array(self.pos)

            for i, vec in enumerate(self.dir_to_vec):
                if np.array_equal(vec, direction):
                    self.dir = i
                    break

            if init_grid is not None:
                grid.set(*self.pos, init_grid.get(*self.pos))
        else:
            pass

        if dummy_move:
            pass
        else:
            self.pos = next_pos

        grid.set(*self.pos, self)

        self.bg_color = bg_color

        if init_grid is not None:
            obj = init_grid.get(*self.pos)
            if isinstance(obj, WorldObj):
                self.bg_color = obj.bg_color

    @property
    def dir_vec(self) -> ndarray:
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.

        Returns
        -------
        NDArray
            the direction vector for forward movement in the current orientation of the agent
        """

        assert self.dir >= 0 and self.dir < len(self.dir_to_vec)
        return self.dir_to_vec[self.dir]

    @property
    def right_vec(self) -> ndarray:
        """
        Get the vector pointing to the right of the agent.

        Returns
        -------
        NDArray
            the vector pointing to the right of the agent
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent

        Returns
        -------
        NDArray
            the position of the cell that is right in front of the agent
        """

        return self.pos + self.dir_vec

    def can_view(
        self,
        x: int,
        y: int,
        obs_type: Literal["directional", "symmetrical"] = "directional",
    ) -> bool:
        """
        check if a grid position is visible to the agent by view distance
        does not check occlusion by other objects

        Parameters
        ----------
        x : int
            x-coordinate in the grid
        y : int
            y-coordinate in the grid

        Returns
        -------
        bool
            whether the grid position is visible to the agent
        """

        view_x, view_y = self.get_view_coords(x, y, obs_type)
        return bool((0 <= view_x < self.view_size) and (0 <= view_y < self.view_size))

    def get_view_coords(
        self,
        i: int,
        j: int,
        obs_type: Literal["directional", "symmetrical"] = "directional",
    ):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.

        Parameters
        ----------
        i : int
            x-coordinate in the grid
        j : int
            y-coordinate in the grid

        Returns
        -------
        tuple
            the coordinates of the grid in the agent's view
        """
        agent_x, agent_y = self.pos

        match obs_type:
            case "symmetrical":
                # (vx, vy) is the object's position in an agent's local frame
                vx = i - agent_x + self.view_size // 2
                vy = j - agent_y + self.view_size // 2

            case "directional":
                dx, dy = self.dir_vec
                rx, ry = self.right_vec

                # Compute the absolute coordinates of the top-left view corner
                hs = self.view_size // 2
                tx = agent_x + (dx * (self.view_size - 1)) - (rx * hs)
                ty = agent_y + (dy * (self.view_size - 1)) - (ry * hs)

                lx = i - tx
                ly = j - ty

                # Project the coordinates of the object relative to the top-left
                # corner onto the agent's own coordinate system
                vx = rx * lx + ry * ly
                vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(
        self, obs_type: Literal["directional", "symmetrical"] = "directional"
    ):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set

        Returns
        -------
        tuple
            the extents of the square set of tiles visible to the agent
        """

        assert self.view_size is not None
        match obs_type:
            case "directional":
                match self.dir:
                    # Facing right
                    case 0:
                        top_x = self.pos[0]
                        top_y = self.pos[1] - self.view_size // 2
                    # facing down
                    case 1:
                        top_x = self.pos[0] - self.view_size // 2
                        top_y = self.pos[1]
                    # Facing left
                    case 2:
                        top_x = self.pos[0] - self.view_size + 1
                        top_y = self.pos[1] - self.view_size // 2
                    # Facing up
                    case 3:
                        top_x = self.pos[0] - self.view_size // 2
                        top_y = self.pos[1] - self.view_size + 1
                    case _:
                        assert False, "invalid agent direction"

            case "symmetrical":
                top_x = self.pos[0] - self.view_size // 2
                top_y = self.pos[1] - self.view_size // 2

            case _:
                raise NotImplementedError("Invalid obs_type")

        bot_x = top_x + self.view_size
        bot_y = top_y + self.view_size

        return (top_x, top_y, bot_x, bot_y)

    def dir2vec(
        self, direction: int, in_tuple: bool = False
    ) -> NDArray[np.int_] | tuple[int, int]:
        """
        Get the vector corresponding to the given direction
        """
        return (
            self.dir_to_vec[direction]
            if not in_tuple
            else tuple(self.dir_to_vec[direction])
        )

    def vec2dir(self, vec: NDArray[np.int_]) -> int:
        """
        Get the direction index corresponding to the given vector
        """
        for i, d in enumerate(self.dir_to_vec):
            if np.array_equal(d, vec):
                return i

        raise ValueError("Invalid direction vector")


class LBFAgent(Agent):
    def __init__(
        self,
        world: World,
        index: int,
        color: str = "red",
        view_size: Optional[int] = None,
        init_pos: Optional[tuple[int, int]] = None,
        init_grid: Optional[Grid] = None,
        level: int = 1,
    ) -> None:

        self.init_pos = init_pos
        self.init_grid = init_grid
        self.level = level

        self.reward: float = 0.0
        self.t_first_hit_goal: int = -1
        self.neighbor_pos_offsets: NDArray[np.int_] = np.array(
            [[-1, 0], [1, 0], [0, -1], [0, 1]]
        )
        self.neighbor_pos: NDArray[np.int_] = np.zeros((4, 2), dtype=np.int_)

        # an agent may have multiple current goal states
        self.task_goals: dict[int, NDArray] = {}

        super().__init__(
            world=world,
            index=index,
            actions=LBFActions,
            color=color,
            type="agent",
            view_size=view_size,
            dir_to_vec=DIR_TO_VEC,
        )

    def move(
        self,
        next_pos: Position,
        grid: Grid,
        init_grid: Grid | None = None,
        dummy_move: bool = False,
        bg_color: str | None = None,
        current_task: int | None = None,
        t: int | None = None,
    ):
        super().move(
            next_pos,
            grid,
            init_grid,
            dummy_move,
            bg_color,
        )

        if (
            self.t_first_hit_goal == -1
            and current_task is not None
            and t is not None
            and self.in_goal_set(current_task)
        ):
            self.t_first_hit_goal = t

    @property
    def pos(self) -> Position:
        return (self._pos[0], self._pos[1])

    @pos.setter
    def pos(self, pos: Position) -> None:
        self._pos = pos
        if pos is not None:
            self.neighbor_pos = pos + self.neighbor_pos_offsets

    def add_goal_pos(self, pos: Position, room_idx: int) -> None:
        pos = np.array([pos])

        if room_idx not in self.task_goals:
            self.task_goals[room_idx] = pos

        elif not np.any(np.all(pos == self.task_goals[room_idx], axis=1)):
            self.task_goals[room_idx] = np.vstack((self.task_goals[room_idx], pos))

    def in_goal_set(self, current_task: int, pos: Position = None) -> bool:
        if pos is None:
            pos = self.pos

        goal_positions = self.task_goals.get(current_task, None)

        if goal_positions is None:
            return False
        else:
            return np.any(np.all(pos == goal_positions, axis=1))

    def reset(self, init_pos: tuple[int, int], level: int = 1) -> None:
        super().reset()
        if self.pos is not None:
            self.neighbor_pos = self.pos + self.neighbor_pos_offsets
        else:
            self.neighbor_pos = np.zeros((4, 2), dtype=np.int_)

        self.level = level
        self.init_pos = init_pos
        self.reward: float = 0.0
        self.t_first_hit_goal = -1

    def encode(self, current_agent: bool = False) -> tuple[int]:
        """Encode a description of this object as a 3-tuple of integers

        Parameters
        ----------
        current_agent : bool, optional
            whether the agent is the current agent, by default False
        """
        if self.world.encode_dim == 3:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                self.level,
            )

    def render(self, img: NDArray[np.uint8]) -> None:
        fill_coords(
            img,
            point_in_rect(0.15, 0.85, 0.15, 0.85),
            color=self.world.COLORS[self.color],
            bg_color=self.bg_color,
        )

        self._render_object_info(img, info=("index", "level"))


class PolicyAgent(Agent):
    """
    Agent with a policy that determines its actions
    """

    def __init__(
        self,
        policy: AgentPolicy,
        world: World,
        index: int = 0,
        view_size: int | None = None,
        actions: type[Actions] = DefaultActions,
        dir_to_vec: list[NDArray] = DIR_TO_VEC,
        color: str | None = None,
        bg_color: str | None = None,
        type: str = "agent",
    ) -> None:
        """Initialize the PolicyAgent object

        Parameters
        ----------
        policy : AgentPolicy
            the policy that determines the agent's actions
        world : World
            the world within which grid is situated
        index : int, optional
            a number useful to identify the instantiated agent, by default 0
        view_size : int, optional
            the size of agent view, if partial observability holds, by default 7
        actions : type[Actions], optional
            set of actions available to the agent, by default DefaultActions
        dir_to_vec : list[NDArray], optional
            map of agent direction indices to vectors, by default DIR_TO_VEC
        color : str | None, optional
            color of the agent, by default None
        bg_color : str | None, optional
            background color of the tile containing agent, by default None
        type : str, optional
            type of the world object, by default "agent"
        """
        super().__init__(
            world, index, view_size, actions, dir_to_vec, color, bg_color, type
        )
        self.policy: AgentPolicy = policy

    def reset(self, options: dict[str, Any] | None = None) -> None:
        super().reset()
        self.policy.reset()
