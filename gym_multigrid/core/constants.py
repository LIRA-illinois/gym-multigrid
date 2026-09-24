import numpy as np
from numpy.typing import NDArray

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    "red": np.array([228, 3, 3]),
    "orange": np.array([255, 140, 0]),
    "yellow": np.array([255, 237, 0]),
    "green": np.array([0, 128, 38]),
    "blue": np.array([0, 77, 255]),
    "purple": np.array([117, 7, 135]),
    "brown": np.array([120, 79, 23]),
    "grey": np.array([100, 100, 100]),
    "light_red": np.array([234, 153, 153]),
    "light_blue": np.array([90, 170, 223]),
    "black": np.array([0, 0, 0]),
    "white": np.array([255, 255, 255]),
    "medium_red": np.array([231, 80, 80]),
}

# Accessible colors for colorblind people from https://www.nature.com/articles/nmeth.1618
ACCESSIBLE_COLORS: dict[str, NDArray[np.uint8]] = {
    "red": np.array([213, 94, 0]),
    "orange": np.array([230, 159, 0]),
    "yellow": np.array([240, 228, 66]),
    "green": np.array([0, 158, 115]),
    "blue": np.array([0, 114, 178]),
    "sky_blue": np.array([86, 180, 233]),
    "purple": np.array([204, 121, 167]),
    "white": np.array([255, 255, 255]),
    "black": np.array([0, 0, 0]),
    "grey": np.array([127, 127, 127]),
    "light_grey": np.array([199, 199, 199]),
    "dark_grey": np.array([77, 77, 77]),
}

CTF_COLORS: dict[str, NDArray[np.uint8]] = {
    "red": np.array([228, 3, 3]),
    "orange": np.array([255, 140, 0]),
    "yellow": np.array([255, 237, 0]),
    "green": np.array([0, 128, 38]),
    "blue": np.array([0, 77, 255]),
    "purple": np.array([117, 7, 135]),
    "brown": np.array([120, 79, 23]),
    "grey": np.array([100, 100, 100]),
    "light_red": np.array([255, 228, 225]),
    "light_blue": np.array([240, 248, 255]),
    "white": np.array([255, 250, 250]),
    "red_grey": np.array([170, 152, 169]),
    "blue_grey": np.array([140, 146, 172]),
}

MAZE_COLORS: dict[str, NDArray[np.uint8]] = {
    "red": np.array([228, 3, 3]),
    "orange": np.array([255, 140, 0]),
    "yellow": np.array([255, 237, 0]),
    "green": np.array([0, 128, 38]),
    "blue": np.array([0, 77, 255]),
    "purple": np.array([117, 7, 135]),
    "brown": np.array([120, 79, 23]),
    "grey": np.array([100, 100, 100]),
    "light_red": np.array([255, 228, 225]),
    "light_blue": np.array([240, 248, 255]),
    "white": np.array([255, 250, 250]),
}

LABYRINTH_COLORS: dict[str, NDArray[np.uint8]] = {
    "red": np.array([228, 3, 3]),
    "orange": np.array([255, 140, 0]),
    "yellow": np.array([255, 237, 0]),
    "green": np.array([0, 128, 38]),
    "blue": np.array([0, 77, 255]),
    "purple": np.array([117, 7, 135]),
    "brown": np.array([120, 79, 23]),
    "grey": np.array([100, 100, 100]),
    "light_red": np.array([255, 228, 225]),
    "light_blue": np.array([240, 248, 255]),
    "white": np.array([255, 250, 250]),
    "light_grey": np.array([200, 200, 200]),
}

PREY_PRED_COLORS: dict[str, NDArray[np.uint8]] = {
    "red": np.array([228, 3, 3]),
    "orange": np.array([255, 140, 0]),
    "yellow": np.array([255, 237, 0]),
    "green": np.array([0, 128, 38]),
    "blue": np.array([0, 77, 255]),
    "purple": np.array([117, 7, 135]),
    "brown": np.array([120, 79, 23]),
    "grey": np.array([100, 100, 100]),
    "light_red": np.array([255, 228, 225]),
    "light_blue": np.array([240, 248, 255]),
    "white": np.array([255, 250, 250]),
    "light_grey": np.array([200, 200, 200]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX: "dict[str, int]" = {key: i for i, key in enumerate(COLORS.keys())}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}

# Map of state names to integers
STATE_TO_IDX_WILDFIRE = {
    "healthy": 0,
    "on fire": 1,
    "burnt": 2,
}

# Map of state idx to color
STATE_IDX_TO_COLOR_WILDFIRE = {
    0: "green",
    1: "orange",
    2: "brown",
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

NAV_DIR_TO_VEC: list[NDArray[np.int_]] = [
    # Stay
    np.array((0, 0)),
    # Left
    np.array((-1, 0)),
    # Down
    np.array((0, 1)),
    # Right
    np.array((1, 0)),
    # Up
    np.array((0, -1)),
]

# Map of object types to short string
OBJECT_TO_STR = {
    "wall": "x",
    "floor": "F",
    "door": "D",
    "key": "K",
    "ball": "o",
    "box": "B",
    "goal": "G",
    "lava": "V",
    "agent": "a",
    "tree": "T",
}

# Short string for opened door
OPENED_DOOR_IDS = "_"

# Map agent's direction to short string
AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
