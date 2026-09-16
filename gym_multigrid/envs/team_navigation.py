from ast import literal_eval
from collections import defaultdict
from copy import copy
from itertools import combinations, product
from math import prod
from os.path import dirname, join
from typing import Any, Literal, Optional
from warnings import warn

# import gurobipy as gp
import numpy as np
import pandas as pd
import yaml
from cv2 import INTER_CUBIC, putText, resize

# from gurobipy import GRB
from gymnasium import spaces
from numpy.random._generator import Generator
from numpy.typing import NDArray

from gym_multigrid.core.agent import (
    AlternativeNavigationActions,
    LBFAgent,
    SimpleNavigationActions,
)
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Detector, DetectorGroup, Goal, Wall, WorldObj
from gym_multigrid.core.world import TeamNavigationWorld
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.typing_utils import Position
from gym_multigrid.utils.rendering import (
    FontConfig,
)
from gym_multigrid.utils.subtasks import NavigationTaskCatalog

RENDER_TEXT_CONFIG = {
    "fontFace": FontConfig.fontFace,
    "fontScale": 0.4,
    "color": (0, 0, 0),
    "thickness": 1,
    "lineType": FontConfig.lineType,
}

HEADER_TEXT_CONFIG = RENDER_TEXT_CONFIG | {"thickness": 2}


class TeamNavigationEnv(MultiGridEnv):
    """
    Environment in which the agents have to reach goals. Supports::
        - multiple rooms and a hierarchical representation of "tasks" in the environment
        - pure navigation tasks with assigned and non-assigned goals
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    world = TeamNavigationWorld
    avail_action_sets = {
        "AlternativeNavigationActions": AlternativeNavigationActions,
        "SimpleNavigationActions": SimpleNavigationActions,
    }

    class RewardConfig:
        def __init__(
            self,
            simultaneous_goal_reward_type: Literal[
                "terminal_sparse_all_simultaneous",
                "terminal_shaped_hit_times",
                "dense_shaped_hit_times",
                "terminal_shaped_largest_group_arrival",
                "per_step_arrival_group_size",
            ]
            | None = None,
            time_penalty: float = -0.01,
            partial_arrival_penalty: float = -0.05,
        ) -> None:
            """
            all signs for rewarding events assume to be handled in their definition since we only use the "+=" operator in the code below for simplicity and easier design
            values are <=0 for "penalties" and >= 0 for "rewards"

            Parameters
            ----------
            num_agents : int
            all_agents_reach_goal_proportion_per_agent : float, optional
                _description_, by default 2.0
            """
            self.simultaneous_goal_reward_type = simultaneous_goal_reward_type
            self.time_penalty = time_penalty
            self.partial_arrival_penalty = partial_arrival_penalty
            self.task_completed_bonus = 1.0

            self.base_hit_reward = 1.0
            self.detection_penalty = -0.025

    class TransitionProbs:
        def __init__(
            self,
            p_chosen_move: float,
            actions: AlternativeNavigationActions | SimpleNavigationActions,
        ) -> None:
            # define events that can happen (support) and their probabilities
            # based on Gym "Frozen Lake" environment. If an agent intends to move in a direction, the env may cause them to move in that direction or in either perpendicular direction.
            # define the base probability for each action here
            self._p_chosen_move = p_chosen_move
            self._slide_prob = (1 - p_chosen_move) / 2

            if actions == AlternativeNavigationActions:
                self.trans = {
                    actions.STAY: {
                        "possible_action": [actions.STAY],
                        "prob": [1.0],
                    },
                    actions.LEFT: {
                        "possible_action": [
                            actions.LEFT,
                            actions.UP,
                            actions.DOWN,
                        ],
                        "prob": [
                            self._p_chosen_move,
                            self._slide_prob,
                            self._slide_prob,
                        ],
                    },
                    actions.RIGHT: {
                        "possible_action": [
                            actions.RIGHT,
                            actions.UP,
                            actions.DOWN,
                        ],
                        "prob": [
                            self._p_chosen_move,
                            self._slide_prob,
                            self._slide_prob,
                        ],
                    },
                    actions.UP: {
                        "possible_action": [
                            actions.UP,
                            actions.LEFT,
                            actions.RIGHT,
                        ],
                        "prob": [
                            self._p_chosen_move,
                            self._slide_prob,
                            self._slide_prob,
                        ],
                    },
                    actions.DOWN: {
                        "possible_action": [
                            actions.DOWN,
                            actions.LEFT,
                            actions.RIGHT,
                        ],
                        "prob": [
                            self._p_chosen_move,
                            self._slide_prob,
                            self._slide_prob,
                        ],
                    },
                }
            elif actions == SimpleNavigationActions:
                self.trans = {
                    actions.STAY: {
                        "possible_action": [actions.STAY],
                        "prob": [1.0],
                    },
                    actions.LEFT: {
                        "possible_action": [
                            actions.LEFT,
                        ],
                        "prob": [
                            self._p_chosen_move,
                        ],
                    },
                    actions.RIGHT: {
                        "possible_action": [
                            actions.RIGHT,
                        ],
                        "prob": [
                            self._p_chosen_move,
                        ],
                    },
                }

        def get_stochastic_action(self, action: int, np_random: Generator) -> int:
            # handles environment randomness as it affects the agent's actual movement
            # for internal env use only
            # EX: agent takes "UP", but slips so ends up moving "RIGHT"
            # in this case, we replace "UP" with "RIGHT" when doing move_agent() and other internal step() methods
            return int(
                np_random.choice(
                    self.trans[action]["possible_action"], p=self.trans[action]["prob"]
                )
            )

    def __init__(
        self,
        map_name: Optional[str] = None,
        width: Optional[int] = 10,
        height: Optional[int] = 10,
        n_agents: int = 4,
        sight: int = 2,
        state_type: Literal[
            "multigrid_flattened", "simple_navigation_features"
        ] = "simple_navigation_features",
        obs_type: Literal[
            "multigrid_flattened", "simple_navigation_features"
        ] = "simple_navigation_features",
        goal_type: Literal["simultaneous_arrival"] | None = None,
        observe_other_agents: bool = True,
        chosen_move_prob: float = 1.0,
        agent_0_delay_prob: float = 0.0,
        highlight_visible_cells: bool = False,
        reward_config: dict[str, float | bool] = {
            "simultaneous_goal_reward_type": None,
        },
        episode_limit: int | None = None,
        navigation_tasks: list[dict[str, Any]]
        | dict[int, dict[str, Any]]
        | None = None,
        navigation_task_catalog: NavigationTaskCatalog | None = None,
    ) -> None:
        """
        Initialize the env.
        """
        self._map_name = map_name or ""
        self.hallway = "_hall" in self._map_name

        self.num_agents = n_agents
        if not 0.0 <= agent_0_delay_prob <= 1.0:
            raise ValueError("agent_0_delay_prob must be between 0.0 and 1.0.")
        self.agent_0_delay_prob = agent_0_delay_prob
        self._delayed_agents = np.zeros(self.num_agents, dtype=bool)
        self.reward_config = self.RewardConfig(**reward_config)
        self.goal_type = goal_type

        # only use episode_limit for internal class logic,
        # do NOT use for episode truncation (use standard gymnasium wrapper for that)
        self._episode_limit = episode_limit
        self._time_scale = max((self._episode_limit or 1) - 1, 1)

        """
        if self.goal_type == "simultaneous_arrival":
            # solve for the reward scaling to ensure it is in [0, 1]
            # TODO check scratch.py for a n example implementation of this scaling
            # you need to solve an optimization problem to find it, which is funny :P
            self.max_reward_simultaneous_arrival = (
                self._get_max_reward_simultaneous_arrival()
            )
        """

        # multi-room support
        self.field_map: pd.DataFrame | None = None
        self.current_task = 0
        self.room_coords: dict[int, tuple] = {}
        self.num_rooms: int
        self.room_has_goals: dict[int, bool] = {0: False}

        if map_name is not None:
            if width is not None or height is not None:
                warn("(height, width) and field map provided, using field map size.")

            self.field_map = self._load_field_map(map_name)
            height, width = self.field_map.shape

        else:
            # add 2 b/c of the outer wall that automatically spawns
            # when no map is specified
            self.num_rooms = 1
            width = width + 2
            height = height + 2

            self.room_coords = {
                0: {
                    "x_limits": (0, width),
                    "y_limits": (0, height),
                }
            }

        # obs options for this specific env
        self.state_type = state_type
        self.env_obs_type = obs_type
        match self.env_obs_type:
            # in both cases, sight is like the "radius" of the square obs centered on the agent
            # this is here b/c multigrid and the original LBF obs handle it a little differently
            case "multigrid_flattened":
                agent_view_size = 2 * sight + 1
                self.observe_other_agents = observe_other_agents
            case "simple_navigation_features":
                agent_view_size = None
            case _:
                raise NotImplementedError(
                    f"Observation type {obs_type!r} is not implemented."
                )

        # if observe_other_agents=False, agents cannot see the other agents and those cells replaced with empty spaces
        self._spawn_attempts: int = 1000

        # initial encoding for objects in the observation
        if navigation_task_catalog is not None and navigation_tasks is not None:
            raise ValueError(
                "Pass either navigation_tasks or navigation_task_catalog, not both."
            )
        if navigation_tasks is not None:
            navigation_task_catalog = NavigationTaskCatalog.from_configs(
                navigation_tasks,
                num_agents=self.num_agents,
                width=width,
                height=height,
            )
        self.navigation_task_catalog = navigation_task_catalog
        self._configured_goal_positions = self._get_configured_goal_positions()

        # init agents
        agents = [
            LBFAgent(
                world=self.world,
                index=i,
                view_size=agent_view_size,
            )
            for i in range(self.num_agents)
        ]

        super().__init__(
            width=width,
            height=height,
            world=self.world,
            see_through_walls=False,
            agents=agents,
            actions_set=SimpleNavigationActions
            if self.hallway
            else AlternativeNavigationActions,
            render_mode="rgb_array",
            obs_type="symmetrical",
            agent_view_size=agent_view_size,
            highlight_visible_cells=highlight_visible_cells,
        )

        # objects that disappear when a given room is completed
        self.room_despawn_objects: dict[int, list]

        # stochastic transition dynamics
        self.transition_prob = self.TransitionProbs(
            chosen_move_prob, actions=self.actions
        )

        self.active_task_state = 0

    def _get_configured_goal_positions(self) -> tuple[Position, ...]:
        if self.navigation_task_catalog:
            positions = {
                tuple(position)
                for transition in self.navigation_task_catalog
                for position in self.navigation_task_catalog[transition].goal_positions
            }
            return tuple(sorted(positions))

        if self.field_map is None:
            return ()

        positions = []
        for y, row in self.field_map.iterrows():
            for x, cell in row.items():
                if (isinstance(cell, str) and cell == "g") or (
                    isinstance(cell, tuple) and cell[0] == "g"
                ):
                    positions.append((x, y))
        return tuple(positions)

    def _get_max_reward_simultaneous_arrival(self):
        # there might be an analytical formula for this,
        # but I don't feel like solving the problem manually to get it
        # max_{t_i} \sum_{(i, j) \in E (pairs of agents)} |t_i - t_j|
        # s.t. 0 \leq t_i < T_{max}
        # this should have a binary solution where floor(n_agents / 2) agents reach at t=0 and the rest reach at T_max, but there might be edge cases where that isn't the case
        agents = np.arange(0, self.num_agents)
        agent_combos = list(combinations(agents, r=2))

        # init the model
        env = gp.Env()
        env.setParam("OutputFlag", 0)
        model = gp.Model(env=env)

        # build decision vars
        hit_times = defaultdict(int)

        # absolute value is non-linear, so need to linearize with aux variables
        # for gurobi to work
        expr_vars = defaultdict(int)
        # aux variable for the absolute value itself
        abs_vars = defaultdict(int)

        for i in range(self.num_agents):
            hit_times[i] = model.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=self._episode_limit - 1,
                name=f"hit_time_{i}",
            )

        for i, combo in enumerate(agent_combos):
            expr_vars[combo] = model.addVar(lb=-GRB.INFINITY, name=f"expr_var_{i}")
            abs_vars[i] = model.addVar(name=f"abs_var_{i}")
        model.update()

        # build constraints
        for i, combo in enumerate(agent_combos):
            model.addConstr(
                expr_vars[combo] == (hit_times[combo[0]] - hit_times[combo[1]])
            )
            model.addConstr(abs_vars[i] == gp.abs_(expr_vars[combo]))
        model.update()

        # build objective
        obj = 0
        for i, combo in enumerate(agent_combos):
            obj += abs_vars[i]
            # print(combo[0], combo[1])
        model.setObjective(obj, GRB.MAXIMIZE)
        model.update()

        # solve
        model.optimize()

        # print("optimal objective value")
        # print(model.ObjVal)
        # print("largest-spread hit times")
        # for i, time in hit_times.items():
        #     print(i, time.X)

        return model.ObjVal

    # grid generation
    def _load_field_map(self, map_name: str) -> pd.DataFrame:
        # read the room layout yaml file to compose the rooms into a cohesive env
        map_dir = join(dirname(__file__), "maps", "team_navigation", map_name)
        config_path = join(map_dir, "config.yaml")
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # grab all the rooms arranged in a grid
        rooms = defaultdict(list)
        self.room_coords = {}
        x_min, y_min = 0, 0
        room_idx = 0
        for j, row in enumerate(config["room_layout"]):
            for i, room_config in enumerate(row):
                room_load_path = join(map_dir, f"{room_config}.csv")
                room = pd.read_csv(room_load_path, header=None).astype(object)
                rooms[j].append(room)

                x_min, x_max = x_min, x_min + room.shape[1]
                y_min, y_max = y_min, y_min + room.shape[0]

                self.room_coords[room_idx] = {
                    "x_limits": (x_min, x_max),
                    "y_limits": (y_min, y_max),
                }
                self.room_has_goals[room_idx] = False

                x_min = x_max
                room_idx += 1

            y_min = y_max

        # concat all the rooms into a single env
        rows = [pd.concat(rooms[i], axis=1, ignore_index=True) for i in rooms]
        field_map = pd.concat(rows, axis=0, ignore_index=True)
        self.num_rooms = room_idx

        # astype(object) allows literal_eval to convert strings to tuples where needed
        for y, row in field_map.iterrows():
            for x in row.index:
                cell = field_map.loc[y, x]
                if pd.isnull(cell):
                    continue
                elif isinstance(cell, str) and len(cell) > 1:
                    # insert quotes so ast sees the object encoding as a valid string
                    cell = f'{cell[0:1]}"{cell[1:2]}"{cell[2:]}'
                    # convert to tuple data types from strings
                    field_map.at[y, x] = literal_eval(cell)
                else:
                    pass

        return field_map

    def _gen_grid(
        self,
        width: int,
        height: int,
        start_task: int = 0,
        navigation_task_transition: tuple[int, int] | None = None,
    ) -> None:
        # Create a blank grid for this episode
        self.grid = Grid(width, height, self.world)

        # remove all objects in room_despawn_objects[room_idx] when that room is completed
        self.room_despawn_objects: dict[int, list] = defaultdict(list)
        self.detector_groups: dict[int, set[Position]] = {}

        # place objects from field_map
        if self.field_map is not None:
            obj_place = ["w", "D"] if self.navigation_task_catalog else ["w", "g", "D"]
            self._parse_field_map(obj_place=obj_place)
        else:
            # add outer wall to stop agents from going off the edge of the env
            self.grid.wall_rect(x=0, y=0, w=self.width, h=self.height)

        if self.navigation_task_catalog:
            transition = navigation_task_transition or (start_task, start_task)
            self._place_navigation_task_goals(*transition)

        # objects spawned before init_grid is initialized will respawn after an agent steps on them and leaves that cell
        # need separate logic to modify self.init_grid to despawn those objects if desired
        self.init_grid: Grid = self.grid.copy()

        # spawn agents
        if self.navigation_task_catalog:
            transition = navigation_task_transition or (start_task, start_task)
            self._spawn_navigation_agents(*transition)
        else:
            self._spawn_agents()

        # If a start_task was provided prior to grid generation, apply adjustments
        # via helper that encapsulates the logic for starting mid-episode.
        if start_task > 0 and not self.navigation_task_catalog:
            self._apply_start_task_adjustments(start_task)

    def activate_navigation_task(self, from_state: int, to_state: int) -> None:
        """Switch the active catalog task without resetting the episode clock."""
        if not self.navigation_task_catalog:
            raise RuntimeError("No state-keyed navigation task catalog is configured.")
        self.navigation_task_catalog.task_for(from_state, to_state)

        for obj in self.room_despawn_objects.get(self.active_task_state, []):
            self.despawn_object(obj)
        for agent in self.agents:
            if agent.pos is not None:
                self.despawn_object(agent)
            agent.t_first_hit_goal = -1

        self._place_navigation_task_goals(from_state, to_state)
        self._spawn_navigation_agents(from_state, to_state)

    def _place_navigation_task_goals(self, from_state: int, to_state: int) -> None:
        if self.navigation_task_catalog is None:
            raise ValueError("No navigation task catalog is configured.")
        task = self.navigation_task_catalog.task_for(from_state, to_state)

        self.active_task_state = to_state
        self.current_task = to_state
        self.room_despawn_objects[to_state] = []
        for agent, goal_position in zip(self.agents, task.goal_positions):
            goal = Goal(
                self.world,
                color="dark_yellow",
                assigned_agent_index=agent.index,
            )
            self.place_object(goal, pos=goal_position)
            if hasattr(self, "init_grid"):
                self.init_grid.set(*goal_position, goal)
            agent.room_goals = {to_state: np.asarray([goal_position], dtype=np.int_)}
            self.room_despawn_objects[to_state].append(goal)

    def _spawn_navigation_agents(self, from_state: int, to_state: int) -> None:
        if self.navigation_task_catalog is None:
            raise ValueError("No navigation task catalog is configured.")
        task = self.navigation_task_catalog.task_for(from_state, to_state)

        joint_positions = task.init_state_dist.states[
            self.np_random.choice(
                len(task.init_state_dist.states), p=task.init_state_dist.probs
            )
        ]
        if len(joint_positions) != self.num_agents:
            raise ValueError(
                f"Navigation task {from_state}->{to_state} must define one spawn position per agent."
            )

        for agent, position in zip(self.agents, joint_positions):
            if not self._check_valid_pos(position, spawn=True):
                raise ValueError(
                    f"Navigation task {from_state}->{to_state} has an invalid spawn position {position}."
                )
            agent.reset(init_pos=position)
            self.place_agent(agent, pos=position, init_grid=self.init_grid)

    def _parse_field_map(self, obj_place: Optional[list] = None) -> dict:
        """
        obj_place: list of object types to place
        """
        num_spawned_objects = defaultdict(int)

        detector_groups: dict[int, list] = defaultdict(list)

        for y, row in self.field_map.iterrows():
            for x in row.index:
                cell = row[x]
                room_idx = self._get_object_room((x, y))

                if pd.isnull(cell):
                    # empty cells
                    continue

                if isinstance(cell, tuple):
                    obj_type, *obj_args = cell

                    # spawn goals, doors, and walls first so they're in init_grid
                    # do agents after init_grid is defined
                    if obj_type in obj_place:
                        match obj_type:
                            case "a":
                                # place agents
                                agent_idx = obj_args[0]
                                for agent in self.agents:
                                    if agent.index == agent_idx:
                                        agent.reset(init_pos=(x, y))
                                        self.place_agent(
                                            agent, pos=(x, y), init_grid=self.init_grid
                                        )
                                        num_spawned_objects["a"] += 1
                                        break

                            case "g":
                                # place goals + assign to agents
                                assigned_agent_idx = obj_args[0]
                                obj = Goal(
                                    self.world,
                                    color="dark_yellow",
                                    assigned_agent_index=assigned_agent_idx,
                                )
                                self.place_object(obj, pos=(x, y))
                                for agent in self.agents:
                                    if agent.index == assigned_agent_idx:
                                        agent.add_goal_pos((x, y), room_idx)
                                self.room_has_goals[room_idx] = True
                                self.room_despawn_objects[room_idx].append(obj)

                            case "D":
                                detector_group_idx = obj_args[0]
                                pos = (x, y)
                                detector_groups[detector_group_idx].append(pos)

                elif isinstance(cell, str):
                    match cell:
                        case "g":
                            obj = Goal(self.world, color="dark_yellow")
                            self.place_object(obj, pos=(x, y))

                            # each goal assigned to all agents
                            for agent in self.agents:
                                agent.add_goal_pos((x, y), room_idx)
                            self.room_has_goals[room_idx] = True
                            self.room_despawn_objects[room_idx].append(obj)

                        case "w":
                            obj = Wall(self.world, type="wall", color="grey")
                            self.place_object(obj, pos=(x, y))

                else:
                    raise NotImplementedError("invalid object in field map config file")

        self._place_detectors(detector_groups)

        return num_spawned_objects

    def _place_detectors(self, detector_groups: dict[int, list[Position]]) -> None:
        """Create detectors for each configured group of sensor tiles.

        Each detector covers the set of positions assigned to its group and fires
        probabilistically when an agent occupies any of those cells. This keeps the
        detector logic lightweight while still matching the stochastic detection
        pattern used elsewhere in the multigrid codebase.
        """

        for group_idx, positions in sorted(detector_groups.items()):
            unique_positions = set(tuple(pos) for pos in positions)
            detector_color = "blue" if group_idx == 0 else "purple"

            # TODO set the detection probs correctly
            detectors = []

            for pos in unique_positions:
                obj = Detector(
                    world=self.world,
                    visual_detect_prob=1.0,
                    radio_detect_prob=1.0,
                    color=detector_color,
                )
                self.place_object(obj, pos=pos)
                detectors.append(obj)

            self.detector_groups[group_idx] = DetectorGroup(detectors)

    @property
    def _detected_agents(self):
        """Return True if any configured detector probabilistically detects an agent."""
        detected_agents = []

        for _, group in self.detector_groups.items():
            detected_agents += group.detect_agents(
                agents=self.agents,
                comms_val=1.0,
                random_generator=self.np_random,
            )

        return detected_agents

    def _apply_start_task_adjustments(self, start_task: int) -> None:
        """Apply adjustments to the grid and agents so the environment appears
        as if earlier rooms were already completed.
        """
        # Despawn objects, update counters for previous rooms
        self._despawn_previous_room_objects(start_task)

        # Place agents on the goals/waypoints of the most recent completed room
        self._place_agents_for_start_task(start_task)

    def _despawn_previous_room_objects(self, start_task: int) -> None:
        # Also despawn room-specific objects (doors, waypoints)
        for room_idx in range(start_task):
            for obj in list(self.room_despawn_objects.get(room_idx, [])):
                try:
                    self.despawn_object(obj)
                except Exception:
                    pass

    def _place_agents_for_start_task(self, start_task: int) -> None:
        # Choose most-recent completed room
        completed_rooms = [
            r
            for r in range(self.num_rooms)
            if (r < start_task and self.room_has_goals.get(r, False))
        ]
        if len(completed_rooms) > 0:
            prev_room = max(completed_rooms)
        else:
            prev_room = start_task - 1

        for agent in self.agents:
            if prev_room in agent.room_goals and len(agent.room_goals[prev_room]) > 0:
                goal_pos = tuple(agent.room_goals[prev_room][0])
                try:
                    if agent.pos is not None:
                        self.despawn_object(agent)
                except Exception:
                    pass

                agent.reset(init_pos=goal_pos)
                self.place_agent(agent, pos=goal_pos, init_grid=self.init_grid)

    def _get_object_room(self, pos: Position) -> int:
        x, y = pos
        for room_idx, room_coords in self.room_coords.items():
            x_min, x_max = room_coords["x_limits"]
            y_min, y_max = room_coords["y_limits"]

            if (x_min <= x < x_max) and (y_min <= y < y_max):
                return room_idx

    def _spawn_agents(
        self,
    ) -> None:

        num_spawned_agents = 0

        if self.field_map is not None:
            # parse field map to spawn any agents defined there
            num_spawned_agents = self._parse_field_map(obj_place=["a"])["a"]

        if num_spawned_agents < self.num_agents:
            for agent in self.agents:
                attempts = 0
                while attempts < self._spawn_attempts:
                    # make sure the agents spawn in the room associated w/ the current task
                    x_min, x_max = self.room_coords[self.current_task]["x_limits"]

                    if self.hallway:
                        # hardcode each agent's starting y position to place it in the right hallway
                        # this doesn't quite work if you want to have rooms above and below each other, but
                        # it works if you just have rooms to the left and right of each other
                        y_min = 2 * agent.index + 1
                        # do + 1 here since np random excludes the high value from its choice
                        y_max = y_min + 1
                    else:
                        y_min, y_max = self.room_coords[self.current_task]["y_limits"]

                    pos = (
                        self.np_random.integers(x_min, x_max),
                        self.np_random.integers(y_min, y_max),
                    )

                    if self._check_valid_pos(pos, spawn=True):
                        agent.reset(init_pos=pos)
                        self.place_agent(agent, pos=pos, init_grid=self.init_grid)
                        break

                    attempts += 1

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[NDArray[np.int_], dict[str, Any]]:
        # despawn any agents from the previous episode
        self._reset_gym(seed=seed)

        # used to render actions
        self._pre_step_actions: NDArray = np.full(self.num_agents, None)
        self._t_render = "Start"

        self._t = 0

        # current room / task
        self.current_task = 0
        self.active_task_state = 0
        navigation_task_transition = None
        if options is not None and "navigation_task_transition" in options:
            navigation_task_transition = tuple(
                int(state) for state in options["navigation_task_transition"]
            )
        if options is not None and "navigation_task_state" in options:
            self.current_task = int(options["navigation_task_state"])
            self.active_task_state = self.current_task
        elif options is not None and "hl_start_state" in options:
            self.current_task = int(options["hl_start_state"])
            self.active_task_state = self.current_task

        if navigation_task_transition is not None:
            self.current_task = navigation_task_transition[1]
            self.active_task_state = self.current_task
        elif self.navigation_task_catalog:
            navigation_task_transition = self.navigation_task_catalog.first_transition(
                self.current_task
            )
            self.current_task = navigation_task_transition[1]
            self.active_task_state = self.current_task

        # generate new env layout
        self._gen_grid(
            self.width,
            self.height,
            start_task=self.current_task,
            navigation_task_transition=navigation_task_transition,
        )
        self._sample_agent_delays()

        obs: NDArray[np.int_] = self.obs
        info: dict[str, Any] = self._get_info()

        return obs, info

    # step
    def step(
        self,
        action: NDArray[np.int_] | np.int_,
    ) -> tuple[NDArray[np.int_], float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Parameters
        ----------
        action :
            The action to take.

        Returns
        -------
        observation : NDArray[np.int_]
            The observation of the environment.
        reward : float
            The reward for the action.
        terminated: bool
            Whether the episode has terminated.
        truncated: bool
            Whether the episode has been truncated (out of time).
        info : dict[str, Any]
            Additional information about the environment.
        """
        terminated: bool = False
        for a in self.agents:
            a.reward = 0.0

        actions: list[int] = np.array(action).flatten().astype(np.int_).tolist()
        next_positions = defaultdict(list)

        # check if actions are valid, replace with STAY if not valid
        for agent, action in zip(self.agents, actions):
            if self._delayed_agents[agent.index]:
                action = self.actions.STAY

            # get stochastic action
            action = self.transition_prob.get_stochastic_action(
                action,
                self.np_random,
            )

            # check if the action is valid
            valid_actions = self._get_valid_actions(agent)

            if action not in valid_actions:
                action = self.actions.STAY

            next_pos: tuple[int, int] = self._get_next_pos(agent, action)
            next_positions[next_pos].append(agent)

        # move agents
        self._move_agents(next_positions)

        # check if the task was completed
        task_completed = self._update_task()

        # check if the episode terminates for any reason
        terminated = self._terminated(task_completed)

        # truncated handled by TimeLimit wrapper
        truncated = False

        self._sample_agent_delays()
        obs: NDArray[np.int_] = self.obs

        # add up cumulative rewards for this step
        reward: float = 0.0

        # per-step time penalty
        # reward += self.reward_config.time_penalty
        # reward += self._reward_partial_arrival()

        # if task_completed:
        #     reward += self.reward_config.task_completed_bonus

        if self.goal_type == "simultaneous_arrival":
            reward += self._simultaneous_arrival_reward(terminated)

        # check if agents are detected, update rewards if they are
        # have it only be assigned to the agent that is
        ## doesn't really matter b/c it's summed at the end over all agents, but helps w/ scaling
        # for agent in self._detected_agents:
        #     agent.reward += self.reward_config.detection_penalty

        agent_rewards = float(np.sum([a.reward for a in self.agents]))
        reward += agent_rewards

        info = self._get_info(task_completed=task_completed)

        self._t += 1

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def _reward_partial_arrival(self) -> float:
        n_agents_at_goal = sum(agent.t_first_hit_goal != -1 for agent in self.agents)

        # only penalizes if part of the team arrives at different times
        if 0 < n_agents_at_goal < self.num_agents:
            return self.reward_config.partial_arrival_penalty

        return 0.0

    def _simultaneous_arrival_reward(self, terminated: bool):
        if (
            self.reward_config.simultaneous_goal_reward_type
            == "terminal_sparse_all_simultaneous"
        ):
            return float(terminated)

        if (
            self.reward_config.simultaneous_goal_reward_type
            == "terminal_shaped_largest_group_arrival"
            and (self._t == self._episode_limit - 1 or terminated)
        ):
            # reward based on the number of agents in the largest "group" that arrives simultaneously
            hit_times = [
                agent.t_first_hit_goal
                for agent in self.agents
                if agent.t_first_hit_goal != -1
            ]
            if not hit_times:
                return 0.0

            simultaneous_reach_counts = [
                hit_times.count(hit_time) for hit_time in set(hit_times)
            ]
            return max(simultaneous_reach_counts) / self.num_agents

        if self.reward_config.simultaneous_goal_reward_type in [
            "terminal_shaped_hit_times",
            "dense_shaped_hit_times",
        ]:
            # get reward / penalty for simultaneous arrival
            hit_reward: float = 0.0
            not_hit_reward: float = 0.0

            if (
                self.reward_config.simultaneous_goal_reward_type
                == "terminal_shaped_hit_times"
                and ((self._t == self._episode_limit - 1) or terminated)
            ):
                # reward for the agents that arrived at their goals
                # penalty for arriving at different times
                agents_hit = set([a for a in self.agents if a.t_first_hit_goal != -1])
                if len(agents_hit) > 0:
                    # get all unique pairs of agents, then sum over them
                    agent_combos = list(combinations(agents_hit, r=2))
                    for combo in agent_combos:
                        # use self._episode_limit - 1 b/c agents cannot spawn on top of their goals
                        hit_reward += np.abs(
                            combo[0].t_first_hit_goal - combo[1].t_first_hit_goal
                        )

                    hit_reward /= self.max_reward_simultaneous_arrival

                # terminal penalty for agents that did not arrive
                agents_not_hit = set(self.agents) - agents_hit
                not_hit_reward = len(agents_not_hit) / len(self.agents)

            elif (
                self.reward_config.simultaneous_goal_reward_type
                == "dense_shaped_hit_times"
            ):
                agents_hit_goal_prev = set(
                    [a for a in self.agents if 0 <= a.t_first_hit_goal < self._t]
                )
                agents_hit_goal_curr = set(
                    [a for a in self.agents if a.t_first_hit_goal == self._t]
                )

                # compute rewards
                if len(agents_hit_goal_prev) > 0 and len(agents_hit_goal_curr) > 0:
                    agent_combos = list(
                        product(agents_hit_goal_prev, agents_hit_goal_curr)
                    )
                    for combo in agent_combos:
                        hit_reward += np.abs(
                            combo[0].t_first_hit_goal - combo[1].t_first_hit_goal
                        )
                hit_reward /= self.max_reward_simultaneous_arrival

                # add the terminal penalty for agents that don't hit the goals
                agents_hit_goal_prev |= agents_hit_goal_curr
                if self._t == self._episode_limit - 1:
                    agents_not_hit = set(self.agents) - agents_hit_goal_prev
                    not_hit_reward = len(agents_not_hit) / len(self.agents)

            reward = -1 * (0.5 * hit_reward + 0.5 * not_hit_reward)
            return reward

        if (
            self.reward_config.simultaneous_goal_reward_type
            == "per_step_arrival_group_size"
        ):
            # get n of agents that arrive at this time
            # the info in this set could also be constructed by taking the previous state + the state after moving the agents, and figuring out which agents just arrived at their goal states, this was just an easier implementation :P
            arrived_agents = [
                agent for agent in self.agents if agent.t_first_hit_goal == self._t
            ]

            # exponential might cause numerical issues, really blows up around 20 agents or so
            # return 2 ** len(arrived_agents) / (2**self.num_agents)
            return len(arrived_agents) ** 2 / (self.num_agents**2)

        return 0.0

    def _move_agents(self, next_positions: dict[Position, list]) -> None:
        # if two or more agents try to move to the same position they all fail and stay at their current position
        for next_pos, agents in next_positions.items():
            # make sure only one agent will arrive at the cell
            if len(agents) == 1 and self._check_valid_pos(next_pos):
                # do movements for non colliding players
                agent = agents[0]

                # Move agent
                agent.move(
                    next_pos=next_pos,
                    grid=self.grid,
                    init_grid=self.init_grid,
                    current_task=self.current_task,
                    t=copy(self._t),
                )

    def _update_task(self) -> bool:
        # task only successfully completed if all agents reach goal at the same time
        simultaneous_goal_reached = all(
            agent.t_first_hit_goal == self._t for agent in self.agents
        )

        if simultaneous_goal_reached:
            for obj in self.room_despawn_objects[self.active_task_state]:
                self.despawn_object(obj)

        return simultaneous_goal_reached

        # TODO figure out how you want to do this w/ possibly branching task graphs
        ## probably don't do this update in this method, but do it as part of the env's reset method or something, pass the current task into this env class from the outer wrapper
        # # move on to the next room if it exists
        # room_completed = True
        # if self.num_rooms > 1:
        #     self.current_task += 1

    def _get_next_pos(self, agent: LBFAgent, action: int) -> tuple[int, int]:
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

        # convert from np ints to ints
        if isinstance(next_pos[0], np.int_):
            next_pos = tuple(map(int, next_pos))
        return next_pos

    def _terminated(self, task_completed: bool) -> bool:
        # episode can terminate for different reasons
        # task was completed
        # all agents reached the goals (absorbing states), so they can't do anything else and the episode is essentially over
        if task_completed or all(
            [agent.in_goal_set(self.current_task) for agent in self.agents]
        ):
            return True

        return False

        # TODO implement this for the HL stuff
        ## for training the HL policy want the option to run each subtask in isolation
        # if self.navigation_task_catalog:
        #     return (
        #         task_completed
        #         and self.active_task_state == self.navigation_task_catalog.last_state()
        #     )

        # Terminate the episode if all rooms have been completed
        # project is completed when you reach the end of the final room
        # TODO needs to be updated to handle non-sequential rooms
        # if self.num_rooms > 1:
        #     return task_completed and self.current_task == self.num_rooms

    def _get_info(self, task_completed: bool = False) -> dict:
        # step info
        info = {}

        agents_hit = [agent for agent in self.agents if agent.t_first_hit_goal != -1]
        info["sum_goal_hit_time_difference"] = sum(
            abs(agent_a.t_first_hit_goal - agent_b.t_first_hit_goal)
            for agent_a, agent_b in combinations(agents_hit, r=2)
        )

        # TODO this doesn't quite work for the multi-task case, need to figure that out
        info["task_completed"] = task_completed
        info["navigation_task_state"] = self.active_task_state

        return info

    # state
    @property
    def state(self) -> NDArray[tuple[int]] | NDArray:
        # define as a class attribute so you can access the state using env.state
        # no matter how many layers of wrappers are around this env
        # return a state of size (n_state_features)

        # use multigrid's basic state for now, come back to this later
        match self.state_type:
            case "simple_navigation_features":
                agent_features = np.concatenate(
                    [
                        self._get_navigation_agent_features(i)
                        for i in range(self.num_agents)
                    ]
                )
                state = np.concatenate(
                    (
                        agent_features,
                        np.concatenate(
                            [
                                self._get_navigation_goal_positions(i)
                                for i in range(self.num_agents)
                            ]
                        ),
                        # self._get_elapsed_time_obs(),
                    )
                )

            case "multigrid_flattened":
                # NOTE: compared to original, currently does NOT have the coordinates of other objects in the ego agent's frame, so that could reduce training performance
                state = self.grid.encode()
                state = state.flatten()

                if self.goal_type == "simultaneous_arrival":
                    first_hit_times = np.array(
                        [
                            self._get_first_hit_time_obs(agent.index)
                            for agent in self.agents
                        ]
                    )
                    state = np.concatenate([state, first_hit_times.flatten()])

            case _:
                raise NotImplementedError

        state = np.concatenate([state, self._delayed_agents.astype(np.float32)])

        return state

    def _get_state_size(self) -> int:
        """standard function to interface with EPyMARL training loop,
        returns the flattened size of the global state."""
        match self.state_type:
            case "multigrid_flattened":
                state_size = prod(self.state.shape)

            case "simple_navigation_features":
                state_size = self.state.shape[0]

            case _:
                raise NotImplementedError

        return state_size

    # obs
    @property
    def obs(self) -> NDArray:
        """get the team's joint observation

        Returns
        -------
        NDArray
            obs of size (num_agents, num_obs_features

        """
        # return an obs of size (num_agents, num_obs_features)
        # use multigrid's basic obs for now, come back to this later
        match self.env_obs_type:
            case "simple_navigation_features":
                obs = np.vstack(
                    [self._get_navigation_features(i) for i in range(self.num_agents)]
                )

            case "multigrid_flattened":
                # one issues w/ this obs is that agents can see when goals disappear (i.e., by another agent reaching it), so that leaks some information that I don't want the agents to have
                # I need to force them to rely on comms to solve this task, and if they don't have to use comms, they won't
                obs_list = self.gen_obs(
                    observe_other_agents=self.observe_other_agents,
                )

                for i, obs in enumerate(obs_list):
                    obs = obs.flatten()

                    # add scaled first hit times to the obs
                    # adding this here b/c multigrid doesn't easily support changing an object's encoding dimension
                    if self.goal_type == "simultaneous_arrival":
                        obs = np.concatenate([obs, self._get_first_hit_time_obs(i)])
                    obs_list[i] = obs

                obs = np.vstack(obs_list)

            case _:
                raise NotImplementedError
        return obs

    def _get_navigation_features(self, agent_idx: int) -> NDArray[np.float32]:
        return np.concatenate(
            (
                self._get_navigation_agent_features(agent_idx),
                self._get_navigation_goal_positions(agent_idx),
                np.array([self._delayed_agents[agent_idx]], dtype=np.float32),
            )
        ).astype(np.float32)

    def _sample_agent_delays(self) -> None:
        self._delayed_agents.fill(False)
        if self.num_agents > 1:
            self._delayed_agents[0] = self.np_random.random() < self.agent_0_delay_prob

    def _get_navigation_agent_features(self, agent_idx: int) -> NDArray[np.float32]:
        coordinate_scale = self._get_navigation_coordinate_scale()
        agent_position = np.asarray(self.agents[agent_idx].pos, dtype=np.float32)
        agent_position = agent_position / coordinate_scale
        return agent_position

        """
        hit_time = self.agents[agent_idx].t_first_hit_goal
        # binary representation of whether the agent has hit a goal state yet or not
        has_hit = float(hit_time >= 0)
        if hit_time >= 0:
            normalized_hit_time = np.clip(hit_time / self._time_scale, 0.0, 1.0)
        else:
            normalized_hit_time = 0.0

        return np.concatenate(
            (
                agent_position,
                np.array([has_hit, normalized_hit_time], dtype=np.float32),
            )
        ).astype(np.float32)
        """

    def _get_navigation_goal_positions(self, agent_idx: int) -> NDArray[np.float32]:
        coordinate_scale = self._get_navigation_coordinate_scale()
        goal_positions = np.asarray(
            self.agents[agent_idx].room_goals.get(self.current_task, []),
            dtype=np.float32,
        ).reshape(-1, 2)
        if len(goal_positions) == 0:
            return np.zeros(2, dtype=np.float32)
        return goal_positions[0] / coordinate_scale

    def _get_elapsed_time_obs(self) -> NDArray[np.float32]:
        return np.array(
            [np.clip(self._t / self._time_scale, 0.0, 1.0)], dtype=np.float32
        )

    def _get_navigation_coordinate_scale(self) -> NDArray[np.float32]:
        """Scale coordinates against the walkable inner grid dimensions."""
        return np.array(
            [max(self.width - 2, 1), max(self.height - 2, 1)], dtype=np.float32
        )

    # obs helpers
    def _get_first_hit_time_obs(self, agent_idx: int):
        if self.agents[agent_idx].t_first_hit_goal > -1:
            first_hit_time_obs = np.array(
                [self.agents[agent_idx].t_first_hit_goal / self._episode_limit]
            )
        else:
            first_hit_time_obs = np.array([self.agents[agent_idx].t_first_hit_goal])

        return first_hit_time_obs

    def _get_obs_size(self) -> int:
        """standard function to interface with EPyMARL training loop, returns the flattened size of a single agent's observation."""
        match self.env_obs_type:
            case "multigrid_flattened":
                obs_size: int = self.observation_space.shape[1]

                if self.goal_type == "simultaneous_arrival":
                    # for the first hitting time
                    obs_size += 1

            case "simple_navigation_features":
                obs_size = self.observation_space.shape[1]

            case _:
                raise NotImplementedError

        return obs_size

    def _set_observation_space(self) -> spaces.Space:
        match self.env_obs_type:
            case "multigrid_flattened":
                # agent obs is a square with radius of self.view_size centered on the agent
                team_obs_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.num_agents,
                        self.world.encode_dim
                        * self.agent_view_size
                        * self.agent_view_size,
                    ),
                    dtype=np.int_,
                )

            case "simple_navigation_features":
                team_obs_space = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(
                        self.num_agents,
                        5,
                    ),
                    dtype=np.float32,
                )

            case _:
                raise NotImplementedError

        return team_obs_space

    # actions
    @property
    def avail_actions(self) -> list[list[int]]:
        # added this method to interface with PYMARL training loop
        # returns list of agent lists, where each agent's list has binary values representing
        # available actions
        _avail_actions_team = []
        for agent in self.agents:
            if (
                agent.in_goal_set(self.current_task)
                or self._delayed_agents[agent.index]
            ):
                # prevent agents from moving if they are in their goal state or delayed
                avail_actions_dict = {
                    action.name: action == self.actions.STAY for action in self.actions
                }
            else:
                # all actions are available
                avail_actions_dict = {action.name: True for action in self.actions}

            _avail_actions_team.append(list(avail_actions_dict.values()))
        return _avail_actions_team

    def _get_valid_actions(self, agent: LBFAgent) -> list[int]:
        # handle actions that cause the agent to collide w/ a non-overlappable objects
        valid = []
        for action in self.actions:
            if action == self.actions.STAY or self._check_valid_pos(
                self._get_next_pos(agent, action.value)
            ):
                valid.append(action.value)

        return valid

    def _set_action_space(self) -> tuple[spaces.Space, int]:
        env_agent_action_space = spaces.Discrete(len(self.actions))
        action_space = [env_agent_action_space] * len(self.agents)
        ac_dim = len(self.actions)

        # if self.team_bandwidth_allocation_action:
        #     comms_action_space = spaces.Box(low=0, high=1)
        #     action_space.append(comms_action_space)
        #     ac_dim = ac_dim + 1

        # convert from list of spaces to gymnasium space
        action_space = spaces.Tuple(action_space)

        return action_space, ac_dim

    # rendering
    def render(self):
        img = super().render()

        # render actions in a separate image that we then append to base env image
        # determine info image width and create a blank white image
        sidebar_width = max(12 * self.tile_size, self.num_agents * 2 * self.tile_size)
        x_text = 5
        line_height = int(self.tile_size * 0.5)

        # ensure pre_step_actions is a 1D array
        if len(self._pre_step_actions.shape) > 1:
            self._pre_step_actions = self._pre_step_actions.flatten()

        action_panel_height = (len(self._pre_step_actions) + 3) * line_height
        info_img = 255 * np.ones(
            (action_panel_height, sidebar_width, 3), dtype=img.dtype
        )

        # header with basic info
        time_header = f"t : {self._t_render}"
        y_text = line_height
        putText(info_img, time_header, (x_text, y_text), **HEADER_TEXT_CONFIG)

        # draw each agent's action as text stacked vertically
        # extra spacing between different sections of text
        y_text += line_height
        action_header = "Agent : Pre-step action"
        putText(info_img, action_header, (x_text, y_text), **HEADER_TEXT_CONFIG)

        # start_y set below header to avoid overlap
        for i, action in enumerate(self._pre_step_actions):
            # convert from int to action name if not none
            if action is not None:
                action = self.actions(action).name.title()

            text = f"{i} : {action}"
            y_text += line_height
            putText(info_img, text, (x_text, y_text), **RENDER_TEXT_CONFIG)

        # place observations directly below the action summary in the sidebar
        observation_img = self._render_agent_observations(sidebar_width)
        sidebar = np.concatenate([info_img, observation_img], axis=0)
        if img.shape[0] < sidebar.shape[0]:
            padded_img = 255 * np.ones(
                (sidebar.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype
            )
            padded_img[: img.shape[0]] = img
            img = padded_img
        elif sidebar.shape[0] < img.shape[0]:
            padded_sidebar = 255 * np.ones(
                (img.shape[0], sidebar.shape[1], sidebar.shape[2]),
                dtype=sidebar.dtype,
            )
            padded_sidebar[: sidebar.shape[0]] = sidebar
            sidebar = padded_sidebar
        img = np.concatenate([img, sidebar], axis=1)

        # upscale until at least 360p
        upscale_mult = 1
        min_target_dims = (360, 640)
        while (upscale_mult * img.shape[0] < min_target_dims[0]) or (
            upscale_mult * img.shape[1] < min_target_dims[1]
        ):
            upscale_mult += 1

        new_dims = (
            img.shape[1] * upscale_mult,
            img.shape[0] * upscale_mult,
        )
        img = resize(img, new_dims, interpolation=INTER_CUBIC)

        return img

    def _render_agent_observations(self, width: int) -> NDArray[np.uint8]:
        """Render the information available to each agent."""
        if self.env_obs_type == "multigrid_flattened":
            grids, _ = self.gen_obs_grid()
            view_size = self.agent_view_size or grids[0].width
            margin = 8
            tile_size = max(
                8,
                min(
                    self.tile_size // 2,
                    max(
                        (width - (self.num_agents + 1) * margin)
                        // (self.num_agents * view_size),
                        8,
                    ),
                ),
            )
            local_images = []
            observe_other_agents = getattr(self, "observe_other_agents", True)
            for agent, grid in zip(self.agents, grids):
                if not observe_other_agents:
                    center = (grid.width // 2, grid.height // 2)
                    for x in range(grid.width):
                        for y in range(grid.height):
                            obj = grid.get(x, y)
                            if (
                                obj is not None
                                and obj.type == "agent"
                                and (x, y) != center
                            ):
                                grid.set(x, y, None)
                local_images.append(grid.render(tile_size))

            line_height = int(self.tile_size * 0.5)
            header_height = 2 * self.tile_size
            panel_height = header_height + max(image.shape[0] for image in local_images)
            panel = 255 * np.ones((panel_height, width, 3), dtype=np.uint8)
            putText(
                panel,
                "Agent observations (local view)",
                (5, self.tile_size),
                **HEADER_TEXT_CONFIG,
            )
            y_offset = header_height
            for index, local_image in enumerate(local_images):
                x_offset = margin + index * (local_image.shape[1] + margin)
                putText(
                    panel,
                    f"Agent {index}",
                    (x_offset, y_offset - 2),
                    **RENDER_TEXT_CONFIG,
                )
                panel[
                    y_offset : y_offset + local_image.shape[0],
                    x_offset : x_offset + local_image.shape[1],
                ] = local_image
            return panel

        observations = np.asarray(self.obs)
        coordinate_scale = self._get_navigation_coordinate_scale()
        component_labels = ["agent_x", "agent_y", "goal_x", "goal_y", "delay"]
        line_height = int(self.tile_size * 0.5)
        header_height = 2 * self.tile_size
        panel_height = header_height + (len(component_labels) + 1) * line_height
        panel = 255 * np.ones((panel_height, width, 3), dtype=np.uint8)

        putText(
            panel,
            "Agent observations (feature vectors)",
            (5, self.tile_size),
            **HEADER_TEXT_CONFIG,
        )

        card_width = width // self.num_agents
        for index, observation in enumerate(observations):
            x_offset = index * card_width + 5
            y_text = header_height
            putText(
                panel,
                f"Agent {index}",
                (x_offset, y_text),
                **RENDER_TEXT_CONFIG,
            )
            display_observation = observation.copy()
            display_observation[:-1] = display_observation[:-1] * np.tile(
                coordinate_scale, 2
            )
            for label, value in zip(component_labels, display_observation):
                y_text += line_height
                rendered_value = (
                    "ON"
                    if label == "delay" and value > 0.5
                    else "OFF"
                    if label == "delay"
                    else f"{value:.0f}"
                )
                putText(
                    panel,
                    f"{label}: {rendered_value}",
                    (x_offset, y_text),
                    **RENDER_TEXT_CONFIG,
                )
        return panel

    @property
    def t_render(self) -> str:
        return self._t_render

    @t_render.setter
    def t_render(self, t) -> None:
        self._t_render = t

    # helper methods
    def _get_neighborhood(
        self,
        row: int,
        col: int,
        radius: int = 1,
        ignore_diag: bool = False,
        return_object_type: Literal["agent"] | None = None,
    ) -> list[WorldObj] | NDArray:
        # neighborhood not same thing as adjacent, it's more general
        # get global coords to use
        x_min, x_max = max(row - radius, 0), min(row + radius + 1, self.width)
        y_min, y_max = max(col - radius, 0), min(col + radius + 1, self.height)

        if return_object_type is not None:
            objects = []
            # directly get the objects of the specified type from the grid
            for i in range(x_min, x_max):
                for j in range(y_min, y_max):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type == return_object_type:
                        objects.append(cell)

            return objects

        grid = self.grid.encode()

        if ignore_diag:
            # get object encodings in a plus-shape centered on (row, col)
            grids = []
            grids.append(grid[x_min:x_max, col, :])
            grids.append(grid[row, y_min:y_max, :])
            return np.concatenate(grids)

        # get object encodings in a square centered on (row, col)
        return grid[x_min:x_max, y_min:y_max, :]

    def _check_valid_pos(self, pos: tuple[int, int], spawn: bool = False) -> bool:

        cell = self.grid.get(*pos)

        if spawn:
            # only allow agents to spawn on empty spaces
            # spawning on other objects causes those objects to permanently despawn
            return cell is None

        else:
            return cell is None or cell.can_overlap()
