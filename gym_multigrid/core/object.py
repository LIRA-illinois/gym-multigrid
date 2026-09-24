from typing import Any, Final, Literal, overload

import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.constants import STATE_IDX_TO_COLOR_WILDFIRE
from gym_multigrid.core.world import World
from gym_multigrid.typing import Position
from gym_multigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
    point_in_star,
)


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(
        self,
        world: World,
        type: str = "base",
        color: str = "grey",
        bg_color: str | None = None,
        reward: float = 0.0,
        absorbing: bool = False,
    ):
        """Create a WorldObj object

        Parameters
        ----------
        world : World
            the world in which the object exists
        type : str, optional
            type of the object, by default "base"
        color : str, optional
            color of the object, by default "grey"
        bg_color : str | None, optional
            background color of the tile containing object, by default None
        """
        assert type in world.OBJECT_TO_IDX, type
        assert color in world.COLOR_TO_IDX, color
        self.type: str = type
        self.color: str = color
        self.init_color: Final[str] = color
        self.contains = None
        self.world = world
        self.bg_color: str | None = bg_color
        self.init_bg_color: Final[str | None] = bg_color
        self.reward: float = reward
        self.absorbing: bool = absorbing

        # Initial position of the object
        self.init_pos: Position = (-1, -1)

        # Current position of the object
        self._pos: Position = (-1, -1)

    @property
    def init_pos_undefined(self) -> bool:
        """Check if the initial position of the object is undefined"""
        return self.init_pos == (-1, -1)

    @property
    def pos_undefined(self) -> bool:
        """Check if the current position of the object is undefined"""
        return self.pos == (-1, -1)

    @property
    def pos(self) -> Position:
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    @overload
    def west_pos(self, in_tuple: Literal[True]) -> Position: ...
    @overload
    def west_pos(self, in_tuple: Literal[False] = False) -> NDArray[np.int_]: ...
    def west_pos(self, in_tuple: bool = False) -> NDArray[np.int_] | Position:
        """
        Get the position of the cell to the left of the object

        Returns
        -------
        NDArray
            the position of the cell to the left of the object
        """
        if self.pos is None:
            raise ValueError("Agent position is not set")
        else:
            delta: NDArray[np.int_] = np.array([-1, 0])  # Move left in the grid
            next_pos: NDArray[np.int_] = np.array(self.pos) + delta

            match in_tuple:
                case True:
                    return (next_pos[0], next_pos[1])
                case False:
                    return next_pos

    @overload
    def east_pos(self, in_tuple: Literal[True]) -> Position: ...
    @overload
    def east_pos(self, in_tuple: Literal[False] = False) -> NDArray[np.int_]: ...
    def east_pos(self, in_tuple: bool = False) -> NDArray[np.int_] | Position:
        """
        Get the position of the cell to the right of the agent

        Returns
        -------
        NDArray
            the position of the cell to the right of the agent
        """
        if self.pos is None:
            raise ValueError("Agent position is not set")
        else:
            delta: NDArray[np.int_] = np.array([1, 0])  # Move right in the grid
            next_pos: NDArray[np.int_] = np.array(self.pos) + delta

            match in_tuple:
                case True:
                    return (next_pos[0], next_pos[1])
                case False:
                    return next_pos

    @overload
    def north_pos(self, in_tuple: Literal[True]) -> Position: ...
    @overload
    def north_pos(self, in_tuple: Literal[False] = False) -> NDArray[np.int_]: ...
    def north_pos(self, in_tuple: bool = False) -> NDArray[np.int_] | Position:
        """
        Get the position of the cell above the agent

        Returns
        -------
        NDArray
            the position of the cell above the agent
        """
        if self.pos is None:
            raise ValueError("Agent position is not set")
        else:
            delta: NDArray[np.int_] = np.array([0, -1])  # Move up in the grid
            next_pos: NDArray[np.int_] = np.array(self.pos) + delta

            match in_tuple:
                case True:
                    return (next_pos[0], next_pos[1])
                case False:
                    return next_pos

    @overload
    def south_pos(self, in_tuple: Literal[True]) -> Position: ...
    @overload
    def south_pos(self, in_tuple: Literal[False] = False) -> NDArray[np.int_]: ...
    def south_pos(self, in_tuple: bool = False) -> NDArray[np.int_] | Position:
        """
        Get the position of the cell below the agent

        Returns
        -------
        NDArray
            the position of the cell below the agent
        """
        if self.pos is None:
            raise ValueError("Agent position is not set")
        else:
            delta: NDArray[np.int_] = np.array([0, 1])  # Move down in the grid
            next_pos: NDArray[np.int_] = np.array(self.pos) + delta

            match in_tuple:
                case True:
                    return (next_pos[0], next_pos[1])
                case False:
                    return next_pos

    def get_all_neighbor_pos(self) -> dict[str, NDArray[np.int_]]:
        """get all of the neighboring positions"""
        return {
            "left": self.west_pos(),
            "right": self.east_pos(),
            "up": self.north_pos(),
            "down": self.south_pos(),
        }

    def reset(self, options: dict[str, Any] | None = None) -> None:
        """
        Reset the object to its initial state.
        This method can be called before the start of every episode.
        """
        self.pos = self.init_pos
        self.contains = None
        self.color = self.init_color
        self.bg_color = self.init_bg_color

    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self) -> bool:
        """Can the agent pick this up?"""
        return False

    def can_contain(self) -> bool:
        """Can this contain another object?"""
        return False

    def see_behind(self) -> bool:
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos: Position) -> bool:
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self, current_agent: bool = False) -> tuple[int, ...]:
        """Encode the a description of this object as a 3-tuple of integers"""
        if self.world.encode_dim == 3:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                0,
            )
        else:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                0,
                0,
                0,
                0,
            )

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int):
        assert False, "not implemented"

    def render(self, img: NDArray[np.uint8]) -> None:
        """Draw this object with the given renderer"""
        raise NotImplementedError


class ObjectGoal(WorldObj):
    def __init__(
        self,
        world: World,
        index: int,
        target_type: str = "ball",
        reward: float = 1,
        color: int | None = None,
    ):
        if color is None:
            super().__init__(world, "objgoal", world.IDX_TO_COLOR[index])
        else:
            super().__init__(world, "objgoal", world.IDX_TO_COLOR[color])
        self.target_type: str = target_type
        self.index: int = index
        self.reward: float = reward

    def can_overlap(self):
        return False

    def render(self, img: NDArray[np.uint8]):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Goal(WorldObj):
    def __init__(
        self,
        world: World,
        index: int | None = None,
        reward: float = 1,
        color: str | None = None,
        absorbing: bool = False,
    ):
        if color is None and index is not None:
            super().__init__(
                world, "goal", world.IDX_TO_COLOR[index], absorbing=absorbing
            )
        elif color is not None:
            super().__init__(world, "goal", color=color, absorbing=absorbing)
        else:
            super().__init__(world, "goal", "green", absorbing=absorbing)
        self.index = index
        self.reward = reward

    def can_overlap(self):
        return True

    def render(self, img: NDArray[np.uint8]):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Switch(WorldObj):
    def __init__(self, world: World):
        super().__init__(world, "switch", world.IDX_TO_COLOR[0])

    def can_overlap(self):
        return True

    def render(self, img: NDArray[np.uint8]):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, world: World, color: str = "blue", type: str = "floor"):
        super().__init__(world, type, color, color)

    def can_overlap(self) -> bool:
        return True

    def render(self, img: NDArray[np.uint8]):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Zone(Floor):
    """
    Alias for Floor
    """

    def __init__(self, world: World, color: str = "blue", type: str = "zone"):
        super().__init__(world, color, type)


class Lava(WorldObj):
    def __init__(
        self,
        world: World,
        color: str = "red",
        type: str = "lava",
        reward: float = 0,
        absorbing: bool = False,
    ):
        super().__init__(
            world, type, color, bg_color=None, reward=reward, absorbing=absorbing
        )

    def can_overlap(self):
        return True

    def render(self, img: NDArray[np.uint8]):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):
    def __init__(self, world: World, type: str = "wall", color: str = "grey"):
        super().__init__(world, type, color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Trap(WorldObj):
    def __init__(
        self,
        world: World,
        color: str = "purple",
        reward: float = -1,
        absorbing: bool = False,
    ):
        super().__init__(
            world, "trap", color, reward=reward, bg_color=None, absorbing=absorbing
        )

    def can_overlap(self):
        return True

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Hole(WorldObj):
    def __init__(
        self,
        world: World,
        color: str = "black",
        bg_color: str | None = "purple",
        reward: float = -1,
        absorbing: bool = False,
    ):
        super().__init__(
            world, "hole", color, reward=reward, bg_color=bg_color, absorbing=absorbing
        )

    def can_overlap(self):
        return True

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(
            img,
            point_in_circle(0.5, 0.5, 0.31),
            self.world.COLORS[self.color],
            self.world.COLORS[self.bg_color] if self.bg_color else None,
        )


class Star(WorldObj):
    def __init__(self, world: World, color: str = "yellow", reward: float = 1):
        super().__init__(world, "star", color, reward=reward)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(
            img,
            point_in_star(0.5, 0.5, 0.31, 5),
            self.world.COLORS[self.color],
        )


class Obstacle(WorldObj):
    def __init__(
        self,
        world: World,
        reward: float = 0,
        can_see_through: bool = True,
        color: str = "grey",
    ):
        super().__init__(world, "obstacle", color, reward=reward)
        self.can_see_through = can_see_through

    def see_behind(self):
        return self.can_see_through

    def can_overlap(self):
        return True if self.reward != 0 else False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.world.COLORS[self.color])


class Door(WorldObj):
    def __init__(
        self,
        world: World,
        color: str,
        is_open: bool = False,
        is_locked: bool = False,
    ):
        super().__init__(world, "door", color)
        self.is_open: bool = is_open
        self.is_locked: bool = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos: Position) -> bool:
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self, current_agent: bool = False):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        state: int
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1
        else:
            raise ValueError("Invalid door state")

        return (
            self.world.OBJECT_TO_IDX[self.type],
            self.world.COLOR_TO_IDX[self.color],
            state,
            0,
            0,
            0,
        )

    def render(self, img):
        c = self.world.COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    def __init__(self, world: World, color: str = "blue"):
        super(Key, self).__init__(world, "key", color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = self.world.COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    def __init__(self, world: World, index: int = 0, reward: float = 2):
        super().__init__(world, "ball", world.IDX_TO_COLOR[index])
        self.index = index
        self.reward = reward

    def can_pickup(self):
        return True

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.world.COLORS[self.color])


class Box(WorldObj):
    def __init__(self, world: World, color: str, contains=None):
        super(Box, self).__init__(world, "box", color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = self.world.COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos: Position):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True


class Flag(WorldObj):
    def __init__(
        self,
        world: World,
        index: int,
        type: str = "flag",
        color: str = "blue",
        bg_color: str | None = "light_blue",
    ):
        super().__init__(world, type, color, bg_color)
        self.index: int = index

    def can_pickup(self):
        return True

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(
            img,
            point_in_circle(0.5, 0.5, 0.31),
            self.world.COLORS[self.color],
            self.world.COLORS[self.bg_color] if self.bg_color else None,
        )


class Tree(WorldObj):
    def __init__(
        self,
        world: World,
        tree_state_idx: int = 0,
        region: str = "common",
    ):
        super().__init__(world, "tree", STATE_IDX_TO_COLOR_WILDFIRE[tree_state_idx])
        self.state = tree_state_idx
        self.agent_above = False
        self.region = region

    def can_overlap(self):
        return True

    def encode(self, current_agent: bool = False):
        return (
            self.world.OBJECT_TO_IDX[self.type],
            self.world.COLOR_TO_IDX[self.color],
            self.state,
        )

    def render(self, img):
        c = self.world.COLORS[self.color]

        fill_coords(img, point_in_rect(0, 1, 0, 1), c)


class AgentGoal(WorldObj):
    def __init__(
        self,
        world: World,
        type: str = "goal",
        color: str = "green",
        bg_color: str | None = None,
    ):
        super().__init__(world, type, color, bg_color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.world.COLORS[self.color])


class SimpleDoor(WorldObj):
    def __init__(
        self,
        world: World,
        color: str = "light_grey",
        type: str = "door",
    ):
        super().__init__(world, type=type, color=color)

        self.locked: bool = True

    def is_open(self) -> bool:
        return self.locked

    def can_overlap(self) -> bool:
        return not self.locked

    def open(self) -> None:
        self.locked = False
        self.color = "grey"

    def close(self) -> None:
        self.locked = True
        self.color = self.init_color

    def see_behind(self) -> bool:
        return False

    def render(self, img):
        c = self.world.COLORS[self.color]

        # Door frame and door
        fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
        fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
        fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

        # Draw door handle
        fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)
