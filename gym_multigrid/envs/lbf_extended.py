from ast import literal_eval
from collections import defaultdict
from copy import copy
from itertools import combinations, product
from math import prod
from os.path import dirname, join
from typing import Any, Literal, Optional
from warnings import warn

import gurobipy as gp
import numpy as np
import pandas as pd
import yaml
from cv2 import INTER_CUBIC, putText, resize
from gurobipy import GRB
from gymnasium import spaces
from numpy.random._generator import Generator
from numpy.typing import NDArray

from gym_multigrid.core.agent import AlternativeNavigationActions, LBFActions, LBFAgent
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal, Wall, WorldObj
from gym_multigrid.core.world import LBFWorld, World
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.typing_utils import Position
from gym_multigrid.utils.rendering import (
    FontConfig,
    fill_coords,
    point_in_circle,
)

RENDER_TEXT_CONFIG = {
    "fontFace": FontConfig.fontFace,
    "fontScale": 0.4,
    "color": (0, 0, 0),
    "thickness": 1,
    "lineType": FontConfig.lineType,
}

HEADER_TEXT_CONFIG = RENDER_TEXT_CONFIG | {"thickness": 2}


class Fruit(WorldObj):
    def __init__(
        self,
        world: World,
        level: int,
        color: str = "green",
    ) -> None:
        self.level = level

        self.neighbor_pos_offsets: NDArray[np.int_] = np.array(
            [[-1, 0], [1, 0], [0, -1], [0, 1]]
        )
        self.neighbor_pos: NDArray[np.int_] = np.zeros((4, 2), dtype=np.int_)

        super().__init__(
            world=world,
            color=color,
            type="fruit",
        )

    @property
    def pos(self) -> Position:
        return (self._pos[0], self._pos[1])

    @pos.setter
    def pos(self, pos: Position) -> None:
        self._pos = pos
        if pos is not None:
            self.neighbor_pos = pos + self.neighbor_pos_offsets

    def can_pickup(self) -> bool:
        return True

    def check_load_condition(self, grid: Grid, loading_agents: list[LBFAgent]) -> bool:
        """
        Check if the fruit was collected by its surrounding agents.

        Parameters
        ----------
        grid : Grid
            The grid of the environment.
        loading_agents : list[LBFAgent]
            list of all agents taking the load action in the env

        Returns
        -------
        curr_fruit_loading_agents: list[LBFAgent]
            agents that attempted to load this fruit
        load_success : bool
            True if the fruit was loaded by the agents in curr_fruit_loading_agents
        tot_agent_levels: int
            total level of the agents that contributed to loading this fruit
        """
        tot_agent_levels: int = 0
        curr_fruit_loading_agents: list[LBFAgent] = []

        for neighbor_pos in self.neighbor_pos:
            cell = grid.get(*neighbor_pos)
            if isinstance(cell, LBFAgent) and cell in loading_agents:
                tot_agent_levels += cell.level
                curr_fruit_loading_agents.append(cell)
            else:
                pass

        load_success: bool = tot_agent_levels >= self.level

        return curr_fruit_loading_agents, load_success, tot_agent_levels

    def render(self, img) -> None:
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.world.COLORS[self.color])
        self._render_object_info(img, info=("level",))

    def reset(self) -> None:
        super().reset()
        if self.pos is not None:
            self.neighbor_pos = self.pos + self.neighbor_pos_offsets
        else:
            self.neighbor_pos = np.zeros((4, 2), dtype=np.int_)

    def encode(self, current_agent: bool = False) -> tuple[int, ...]:
        """Encode the a description of this object as a 3-tuple of integers"""
        if self.world.encode_dim == 3:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                self.level,
            )
        else:
            return (
                self.world.OBJECT_TO_IDX[self.type],
                self.world.COLOR_TO_IDX[self.color],
                self.level,
                0,
                0,
                0,
            )


class LBFExtendedEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect fruit. Extends original LBF by supporting:
        - multiple rooms and a hierarchical representation of "tasks" in the environment
        - pure navigation and mixed navigation + fruit collection tasks
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    world = LBFWorld
    avail_action_sets = {
        "LBFActions": LBFActions,
        "LBFNavigationActions": AlternativeNavigationActions,
    }

    class RewardConfig:
        def __init__(
            self,
            num_agents: int,
            normalize_fruit_reward: bool = True,
            failed_load_penalty: float = 0.0,
            agent_reach_goal_mult: float = 0.5,
            agent_leave_goal_mult: float = -0.7,
            all_agents_reach_goal_proportion_per_agent: float = 2.0,
            simultaneous_goal_reward_type: Literal["terminal", "during_episode"]
            | None = None,
            # move_mult: float = 0.0,
            # max_dense_reach_goal_proportion: float,
        ) -> None:
            """
            all signs for rewarding events assume to be handled in their definition since we only use the "+=" operator in the code below for simplicity and easier design
            values are <=0 for "penalties" and >= 0 for "rewards"

            Parameters
            ----------
            num_agents : int
            normalize_fruit_reward : bool, optional
            failed_load_penalty : float, optional
                value <= 0, by default 0.0
            agent_reach_goal_mult : float, optional
                value >= 0 for "rewards", by default 0.5
            agent_leave_goal_mult : float, optional
                <= 0, by default -0.7
            all_agents_reach_goal_proportion_per_agent : float, optional
                _description_, by default 2.0
            """
            # does NOT consider fruit loading rewards since those are based on fruit levels

            # this setup worked well for 3 agents
            # did not work for 6
            # the gap between reach and leave may not be large enough and may get blown out
            # movement_reward=0.0,
            # agent_reach_goal_reward=0.9,
            # agent_leave_goal_reward=-1.0,
            # all_agents_at_goal_reward=10.0,

            self.simultaneous_goal_reward_type = simultaneous_goal_reward_type

            self.normalize_fruit_reward = normalize_fruit_reward
            self.failed_load_penalty: float = failed_load_penalty

            # overall scaling for all rewards in the env
            # use to prevent rewards from getting to large
            # and producing too-large gradients for learning
            self.base_goal_reward = 1.0

            # individual agent rewards, do not multiply by n_agents

            # reach goal serves as the base reward that all others are derived from
            # using reward multipliers. This helps design a reward function that can work for
            # different team sizes
            self.agent_reach_goal_reward: float = (
                self.base_goal_reward * agent_reach_goal_mult
            )

            self.agent_leave_goal_reward: float = (
                self.base_goal_reward * agent_leave_goal_mult
            )

            # reward for entire team, needs to be multiplied by number of agents
            # to work for different team sizes
            self.all_agents_at_goal_reward: float = (
                self.agent_reach_goal_reward
                * all_agents_reach_goal_proportion_per_agent
                * num_agents
            )

            # self.movement_reward: float = self.base_reward * move_mult

            # max_dense_reward needs to be set small enough so agents can't just
            # wander around to get large total reward. They would do that if the
            # discounted summed reward from the "dense" component of R(s, a)
            # is larger than the final reward they get from the team-level goal.
            # It also needs to be small enough so during learning, they
            # have an incentive get the "reach individual goal" reward.
            # self.max_dense_reward: float = (
            #     self.agent_reach_goal_reward * max_dense_reach_goal_proportion
            # )
            # self.max_dense_reward_per_agent: float = self.max_dense_reward / num_agents

    class TransitionProbs:
        def __init__(
            self,
            p_chosen_move: float,
            actions: LBFActions | AlternativeNavigationActions,
        ) -> None:
            # define events that can happen (support) and their probabilities
            # based on Gym "Frozen Lake" environment. If an agent intends to move in a direction, the env may cause them to move in that direction or in either perpendicular direction.
            # define the base probability for each action here
            self._p_chosen_move = p_chosen_move
            self._slide_prob = (1 - p_chosen_move) / 2

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
            if actions == LBFActions:
                self.trans[actions.LOAD] = {
                    "possible_action": [actions.LOAD],
                    "prob": [1.0],
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
        min_agent_level: int = 1,
        max_agent_level: int = 2,
        max_num_fruit: int = 2,
        max_num_fruit_per_room: Optional[int] = None,
        min_fruit_level: int = 1,
        max_fruit_level: Optional[int] = None,
        force_coop: bool = False,
        observe_agent_levels: bool = True,
        state_type: Literal["original", "multigrid_flattened"] = "original",
        obs_type: Literal["original", "multigrid_flattened"] = "original",
        goal_type: Literal["simultaneous_arrival"] | None = None,
        observe_other_agents: bool = True,
        chosen_move_prob: float = 1.0,
        highlight_visible_cells: bool = False,
        asymmetric_fruit_obs: Literal["smacv2", "different_sensors"] | None = None,
        smacv2_fruit_obs_prob: float = 1.0,
        reward_config: dict[str, float | bool] = {
            "agent_reach_goal_mult": 0.5,
            "agent_leave_goal_mult": -0.7,
            "all_agents_reach_goal_proportion_per_agent": 2.0,
            "failed_load_penalty": 0.0,
            "normalize_fruit_reward": True,
            "simultaneous_goal_reward_type": None,
        },
        num_agents_fruit_obs: int | None = None,
        action_set: Literal["LBFActions", "LBFNavigationActions"] = "LBFActions",
        episode_limit: int | None = None,
    ) -> None:
        """
        Initialize the LBFGameEnv.

        Parameters
        ----------
        map_name:
            name of the map to load (yaml file)
            if None, just 1 room w/ some width and height

        size : int
            Size of grid if square. Default 19
        num_fruit : list[int]
            Number of fruit of each type present in environment.
        agents_index : list[int]
            Colour index for each agent.
        fruits_index : list[int]
            Colour index for each fruit type.
        fruits_reward : list[float]
            Reward given for collecting each fruit type.
        use_mdp: bool, whether to use the project MDP or not
        state_type: Literal["original", "multigrid_flattened"] = "original"
            format for the state, orignal breaks when using Goal objects since they were not in the original LBF env
        obs_type: Literal["original", "multigrid"] = "original"
            format for the observation, orignal breaks when using Goal objects since they were not in the original LBF env
        """
        self.num_agents = n_agents
        self.reward_config = self.RewardConfig(
            num_agents=self.num_agents, **reward_config
        )
        self.num_agents_fruit_obs = num_agents_fruit_obs
        self.goal_type = goal_type

        # only use episode_limit for internal class logic,
        # do NOT use for episode truncation (use standard gymnasium wrapper for that)
        self._episode_limit = episode_limit

        if self.goal_type == "simultaneous_arrival":
            # solve for the reward scaling to ensure it is in [0, 1]
            # TODO check scratch.py for a n example implementation of this scaling
            # you need to solve an optimization problem to find it, which is funny :P
            self.max_reward_simultaneous_arrival = (
                self._get_max_reward_simultaneous_arrival()
            )

        # multi-room support
        self.field_map: pd.DataFrame | None = None
        self.current_task = 0
        self.room_coords: dict[int, tuple]
        self.num_rooms: int
        self.room_has_goals: dict[int, bool] = {0: False}

        if map is not None:
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

        # hierarchical model of environment project the tasks that comprise it
        self.max_num_fruit_per_room = max_num_fruit_per_room

        # initialize per-room fruit counters
        self.num_fruit_per_room: dict[int, int] = defaultdict(int)

        # determine max number of fruit to spawn overall. If a per-room limit
        # is provided, use it to cap the total possible fruits across all rooms.
        if max_num_fruit_per_room is not None:
            # ensure integer multiplication
            self.max_num_fruit = int(max_num_fruit_per_room) * int(self.num_rooms)
        else:
            self.max_num_fruit = max_num_fruit

        self._num_fruit_spawned: int = 0

        # obs options for this specific env
        self.state_type = state_type
        self.env_obs_type = obs_type
        match self.env_obs_type:
            # in both cases, sight is like the "radius" of the square obs centered on the agent
            # this is here b/c multigrid and the original LBF obs handle it a little differently
            case "original":
                agent_view_size = sight
            case "multigrid_flattened":
                agent_view_size = 2 * sight + 1

        # if observe_other_agents=False, agents cannot see the other agents and those cells replaced with empty spaces
        self.observe_other_agents = observe_other_agents
        self.observe_agent_levels = observe_agent_levels

        self.asymmetric_fruit_obs = asymmetric_fruit_obs
        self.smacv2_fruit_obs_prob = float(smacv2_fruit_obs_prob)
        if not 0.0 <= self.smacv2_fruit_obs_prob <= 1.0:
            raise ValueError("fruit_obs_prob must lie in [0, 1]")

        self.force_coop = force_coop
        self.min_agent_levels = np.array([min_agent_level] * self.num_agents)
        self.max_agent_levels = np.array([max_agent_level] * self.num_agents)

        self.min_fruit_level = min_fruit_level
        self.max_fruit_level = max_fruit_level
        self._spawn_attempts: int = 1000

        # initial encoding for objects in the observation
        self.init_object_obs = np.array([-1, -1, 0]).reshape(1, -1)

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
            actions_set=self.avail_action_sets[action_set],
            render_mode="rgb_array",
            obs_type="symmetrical",
            agent_view_size=agent_view_size,
            highlight_visible_cells=highlight_visible_cells,
        )

        # use same approach as the gymma wrapper in EPYMARL
        # from the learning agent's perspective, all actions are always available
        avail_actions_dict = {action.name: True for action in self.actions}
        self._avail_actions = [
            list(avail_actions_dict.values()) for agent in self.agents
        ]

        # optional env config, allows deterministic design of env with a config file, does not support random spawning of objects
        self.waypoint_pos: list[Position]

        # objects that disappear when each room is completed
        self.room_despawn_objects: dict[int, list]

        # fruit tracking
        self.num_fruit_per_room: dict[int, int]
        self.num_fruit_collected_per_room: dict[int, int]

        # stochastic transition dynamics
        self.transition_prob = self.TransitionProbs(
            chosen_move_prob, actions=self.actions
        )

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
        map_dir = join(dirname(__file__), "maps", "lbf_extended", map_name)
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

    def _gen_grid(self, width: int, height: int, start_task: int = 0) -> None:
        # Create a blank grid for this episode
        self.grid = Grid(width, height, self.world)

        # remove all objects in room_despawn_objects[room_idx] when that room is completed
        self.room_despawn_objects: dict[int, list] = defaultdict(list)

        # place objects from field_map
        if self.field_map is not None:
            self._parse_field_map(obj_place=["w", "g", "d"])
        else:
            # add outer wall to stop agents from going off the edge of the env
            self.grid.wall_rect(x=0, y=0, w=self.width, h=self.height)

        # objects spawned before init_grid is initialized will respawn after an agent steps on them and leaves that cell
        # need separate logic to modify self.init_grid to despawn those objects if desired
        self.init_grid: Grid = self.grid.copy()

        # spawn agents
        self._spawn_agents(self.min_agent_levels, self.max_agent_levels)

        # spawn fruit
        if self.max_fruit_level is not None:
            max_fruit_level = self.max_fruit_level * np.ones(self.max_num_fruit)
        else:
            # sum of 3 highest-level agents
            agent_levels = sorted([agent.level for agent in self.agents])
            max_fruit_level = sum(agent_levels[:3])

        self._num_fruit_spawned = self._spawn_fruit(
            min_levels=self.min_fruit_level * np.ones(self.max_num_fruit),
            max_levels=max_fruit_level * np.ones(self.max_num_fruit),
        )

        # If a start_task was provided prior to grid generation, apply adjustments
        # via helper that encapsulates the logic for starting mid-episode.
        if start_task > 0:
            self._apply_start_task_adjustments(start_task)

    def _parse_field_map(self, obj_place: Optional[list] = None) -> dict:
        """
        obj_place: list of object types to place
        """
        num_spawned_objects = defaultdict(int)

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
                    # do agents and fruit after init_grid is defined
                    if obj_type in obj_place:
                        match obj_type.lower():
                            case "a":
                                # place agents
                                level, agent_idx = obj_args[0], obj_args[1]
                                for agent in self.agents:
                                    if agent.index == agent_idx:
                                        agent.reset(init_pos=(x, y), level=level)
                                        self.place_agent(
                                            agent, pos=(x, y), init_grid=self.init_grid
                                        )
                                        num_spawned_objects["a"] += 1
                                        break

                            case "f":
                                # place fruit
                                level = obj_args[0]

                                obj = Fruit(
                                    world=self.world,
                                    level=level,
                                )
                                self.place_object(obj, pos=(x, y))
                                self.num_fruit_per_room[room_idx] += 1
                                num_spawned_objects["f"] += 1

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

                elif isinstance(cell, str):
                    obj_type = cell
                    match obj_type.lower():
                        case "g":
                            obj = Goal(self.world)
                            self.place_object(obj, pos=(x, y))

                            # each goal assigned to all agents
                            for agent in self.agents:
                                agent.add_goal_pos((x, y), room_idx)
                            self.room_has_goals[room_idx] = True
                            self.room_despawn_objects[room_idx].append(obj)

                        case "w":
                            obj = Wall(self.world, type="wall", color="grey")
                            self.place_object(obj, pos=(x, y))

                        case "d":
                            obj = Wall(self.world, type="wall", color="grey")
                            self.place_object(obj, pos=(x, y))
                            self.room_despawn_objects[room_idx].append(obj)

                else:
                    raise NotImplementedError("invalid object in field map config file")

        return num_spawned_objects

    def _apply_start_task_adjustments(self, start_task: int) -> None:
        """Apply adjustments to the grid and agents so the environment appears
        as if earlier rooms were already completed.
        """
        # Despawn fruits and update counters for previous rooms
        self._despawn_previous_room_fruits_and_objects(start_task)

        # Place agents on the goals/waypoints of the most recent completed room
        self._place_agents_for_start_task(start_task)

    def _despawn_previous_room_fruits_and_objects(self, start_task: int) -> None:
        grid_enc = self.grid.encode()
        fruit_idx = self.world.OBJECT_TO_IDX["fruit"]
        for i in range(grid_enc.shape[0]):
            for j in range(grid_enc.shape[1]):
                if grid_enc[i, j, 0] == fruit_idx:
                    room_idx = self._get_object_room((i, j))
                    if room_idx < start_task:
                        obj = self.grid.get(i, j)
                        if obj is not None:
                            self.despawn_object(obj)
                            if self.num_fruit_per_room.get(room_idx, 0) > 0:
                                self.num_fruit_per_room[room_idx] -= 1
                            self.num_fruit_collected_per_room[room_idx] = (
                                self.num_fruit_per_room.get(room_idx, 0)
                            )
                            self.all_room_fruit_collected[room_idx] = True

        # Also despawn room-specific objects (doors, waypoints)
        for room_idx in range(start_task):
            for obj in list(self.room_despawn_objects.get(room_idx, [])):
                try:
                    self.despawn_object(obj)
                except Exception:
                    pass

    def _place_agents_for_start_task(self, start_task: int) -> None:
        # Choose most-recent completed room with goals
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

                agent.reset(init_pos=goal_pos, level=agent.level)
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
        min_agent_levels: NDArray,
        max_agent_levels: NDArray,
    ) -> None:

        num_spawned_agents = 0

        if self.field_map is not None:
            # parse field map to spawn any agents defined there
            num_spawned_agents = self._parse_field_map(obj_place=["a"])["a"]

        if num_spawned_agents < self.num_agents:
            # permute agent levels
            agent_permutation = self.np_random.permutation(self.num_agents)
            min_agent_levels = min_agent_levels[agent_permutation]
            max_agent_levels = max_agent_levels[agent_permutation]

            for agent, min_agent_level, max_agent_level in zip(
                self.agents, min_agent_levels, max_agent_levels
            ):
                attempts = 0
                while attempts < self._spawn_attempts:
                    # make sure the agents spawn in the room associated w/ the current task
                    x_min, x_max = self.room_coords[self.current_task]["x_limits"]
                    y_min, y_max = self.room_coords[self.current_task]["y_limits"]

                    pos = (
                        self.np_random.integers(x_min, x_max),
                        self.np_random.integers(y_min, y_max),
                    )

                    if self._check_valid_pos(pos, spawn=True):
                        level = self.np_random.integers(
                            min_agent_level, max_agent_level + 1
                        )
                        agent.reset(init_pos=pos, level=level)
                        self.place_agent(agent, pos=pos, init_grid=self.init_grid)
                        break

                    attempts += 1

    def _spawn_fruit(
        self,
        min_levels: NDArray,
        max_levels: NDArray,
    ) -> int:
        """
        Returns
        -------
        int
            number of fruit spawned in the environment, may be less than max_num_fruit
        """
        num_spawned_fruit = 0

        if self.field_map is not None:
            # parse field map to spawn any agents defined there
            num_manually_placed_fruit = self._parse_field_map(obj_place=["f"])["f"]

            # if any manually placed fruit, skip rest of random fruit placement logic
            if num_manually_placed_fruit > 0:
                num_spawned_fruit = num_manually_placed_fruit
                self.max_num_fruit = num_manually_placed_fruit
                return num_spawned_fruit

        if num_spawned_fruit < self.max_num_fruit:
            attempts = 0
            min_levels = max_levels if self.force_coop else min_levels

            # permute fruit levels
            fruit_permutation = self.np_random.permutation(self.max_num_fruit)
            min_levels = min_levels[fruit_permutation]
            max_levels = max_levels[fruit_permutation]

            while (
                num_spawned_fruit < self.max_num_fruit
                and attempts < self._spawn_attempts
            ):
                attempts += 1

                # offset by 2 here to avoid avoid spawning a high-level fruit along the wall or in the corner
                # where there may not be enough available sides for the agents to surround and load it
                pos = (
                    self.np_random.integers(1, self.width - 1),
                    self.np_random.integers(1, self.height - 1),
                )

                # check if any fruit in the neighborhood
                radius_obs = self._get_neighborhood(*pos, radius=1)[:, :, 0]
                plus_obs = self._get_neighborhood(*pos, radius=2, ignore_diag=True)[
                    :, 0
                ]

                # ensure the per-room fruit limit is not exceeded (if specified)
                room_idx = self._get_object_room(pos)
                if (self.max_num_fruit_per_room is not None) and (
                    self.num_fruit_per_room.get(room_idx, 0)
                    >= self.max_num_fruit_per_room
                ):
                    continue

                # fruit cannot spawn if:
                # next to a wall (helps prevent generation of un-solvable tasks, e.g. level 4 fruit w/ 2 sides blocked by a wall and only level 1 agents available)
                # next to or near another fruit (helps prevent generation of un-solvable tasks, helps space out the fruit)
                # on a space another object already occupies (prevent spawning on goals, walls, agents, etc.)
                grid_enc = self.grid.encode()
                if (
                    np.any(radius_obs == self.world.OBJECT_TO_IDX["wall"])
                    or np.any(radius_obs == self.world.OBJECT_TO_IDX["fruit"])
                    or np.any(plus_obs == self.world.OBJECT_TO_IDX["fruit"])
                    or (grid_enc[*pos, 0] != self.world.OBJECT_TO_IDX["empty"])
                ):
                    continue

                self.place_object(
                    Fruit(
                        world=self.world,
                        level=self.np_random.integers(
                            min_levels[num_spawned_fruit],
                            max_levels[num_spawned_fruit] + 1,
                        ),
                    ),
                    pos=pos,
                )

                # update room counters
                self.num_fruit_per_room[room_idx] += 1
                num_spawned_fruit += 1

            return num_spawned_fruit

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[NDArray[np.int_], dict[str, Any]]:
        # despawn any agents or fruit from the previous episode
        self._reset_gym(seed=seed)

        # used to render actions
        self._pre_step_actions: NDArray = np.full(self.num_agents, None)
        self._t_render = "Start"

        self._t = 0

        self._fruit_obs_state: dict[tuple[int, int], dict[str, Any]] = defaultdict(list)

        # reset fruit tracking
        self.num_fruit_per_room: dict[int, int] = defaultdict(int)
        self.num_fruit_collected_per_room = {
            room_idx: 0 for room_idx in range(self.num_rooms)
        }
        self.all_room_fruit_collected = [False] * self.num_rooms

        # current room / task
        if options is not None and "hl_start_state" in options:
            self.current_task = options["hl_start_state"]

        for room_idx in range(self.current_task):
            self.all_room_fruit_collected[room_idx] = True

        # generate new env layout
        self._gen_grid(self.width, self.height, start_task=self.current_task)

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
        moving_agents = defaultdict(list)
        loading_agents = set()

        # check if actions are valid, replace with STAY if not valid
        for agent, action in zip(self.agents, actions):
            # get stochastic action
            action = self.transition_prob.get_stochastic_action(
                action,
                self.np_random,
            )

            # check if the action is valid
            valid_actions = self._get_valid_actions(agent)
            if action not in valid_actions:
                action = self.actions.STAY

            # setup for env transition
            if actions == LBFActions and action == self.actions.LOAD:
                loading_agents.add(agent)

            elif action in [
                self.actions.UP,
                self.actions.DOWN,
                self.actions.LEFT,
                self.actions.RIGHT,
            ]:
                next_pos: tuple[int, int] = self._get_next_pos(agent, action)
                moving_agents[next_pos].append(agent)

        # move agents
        self._move_agents(moving_agents)

        # process the loadings
        self._load_fruit(loading_agents)

        # update waypoints
        room_completed = self._update_room()

        # check if entire project is complete
        terminated = self._terminated(room_completed)

        # truncated handled by TimeLimit wrapper
        truncated = False

        obs: NDArray[np.int_] = self.obs

        # add up cumulative rewards for this step
        reward: float = 0.0

        if self.goal_type == "simultaneous_arrival":
            reward += self._simultaneous_arrival_reward(terminated)

        agent_rewards = float(np.sum([a.reward for a in self.agents]))
        # print("agent_rewards", agent_rewards)
        reward += agent_rewards

        info = self._get_info(terminated=terminated, room_completed=room_completed)

        self._t += 1

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def _simultaneous_arrival_reward(self, terminated: bool):
        # get reward / penalty for simultaneous arrival
        hit_reward: float = 0.0
        not_hit_reward: float = 0.0

        if self.reward_config.simultaneous_goal_reward_type == "terminal" and (
            (self._t == self._episode_limit - 1) or terminated
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

        elif self.reward_config.simultaneous_goal_reward_type == "during_episode":
            agents_hit_goal_prev = set(
                [a for a in self.agents if 0 <= a.t_first_hit_goal < self._t]
            )
            agents_hit_goal_curr = set(
                [a for a in self.agents if a.t_first_hit_goal == self._t]
            )

            # compute rewards
            if len(agents_hit_goal_prev) > 0 and len(agents_hit_goal_curr) > 0:
                agent_combos = list(product(agents_hit_goal_prev, agents_hit_goal_curr))
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

    def _move_agents(self, moving_agents: dict[Position, list]) -> None:
        # if two or more agents try to move to the same position they all fail and stay at their current position
        for next_pos, agents in moving_agents.items():
            # make sure only one agent will arrive at the cell
            if len(agents) == 1 and self._check_valid_pos(next_pos):
                # do movements for non colliding players
                agent = agents[0]

                # run goal logic if there are goals in the current room
                self._single_goal_reward_logic(agent, next_pos)

                # Move agent
                agent.move(
                    next_pos=next_pos,
                    grid=self.grid,
                    init_grid=self.init_grid,
                    current_task=self.current_task,
                    t=copy(self._t),
                )

    def _single_goal_reward_logic(self, agent: LBFAgent, next_pos: Position) -> None:
        # only enable goal rewards + penalties if all fruit has been collected in the room

        if self.room_has_goals[self.current_task] and (
            self.max_num_fruit == 0 or self.all_room_fruit_collected[self.current_task]
        ):
            # if not at goal and reach goal, get a reward
            if (not agent.in_goal_set(self.current_task)) and agent.in_goal_set(
                self.current_task, pos=next_pos
            ):
                agent.reward += self.reward_config.agent_reach_goal_reward

            # if agent at goal and next pos is not a goal, get penalty
            if agent.in_goal_set(self.current_task) and not agent.in_goal_set(
                self.current_task, pos=next_pos
            ):
                agent.reward += self.reward_config.agent_leave_goal_reward

    def _update_room(self) -> bool:
        room_completed = False

        # if there are goals in the room, reaching goals after collecting all fruit completes this room
        if self.room_has_goals[self.current_task]:
            reached_room_goal: list[bool] = [
                agent.in_goal_set(self.current_task) for agent in self.agents
            ]
            if (
                all(reached_room_goal)
                and self.all_room_fruit_collected[self.current_task]
            ):
                # print(
                #     f"All agents reached goals for room {self.current_task}. Moving to next room."
                # )
                for obj in self.room_despawn_objects[self.current_task]:
                    self.despawn_object(obj)

                # move on to the next room if it exists
                room_completed = True
                if self.num_rooms > 1:
                    self.current_task += 1

        # otherwise collecting all fruit completes this room
        elif self.all_room_fruit_collected[self.current_task]:
            for obj in self.room_despawn_objects[self.current_task]:
                self.despawn_object(obj)

            # move on to the next room if it exists
            room_completed = True
            if self.num_rooms > 1:
                self.current_task += 1

        return room_completed

    def _load_fruit(self, loading_agents: set) -> None:
        loading_agents_tmp = loading_agents.copy()

        while loading_agents_tmp:
            agent = loading_agents_tmp.pop()
            for neighbor_pos in agent.neighbor_pos:
                cell = self.grid.get(*neighbor_pos)
                if isinstance(cell, Fruit):
                    curr_fruit_loading_agents, load_success, tot_agent_levels = (
                        cell.check_load_condition(self.grid, list(loading_agents))
                    )

                    if load_success:
                        fruit_level = cell.level
                        # the fruit was loaded and each player that helped load scores points
                        for a in curr_fruit_loading_agents:
                            a.reward += float(a.level * fruit_level)

                            if self.reward_config.normalize_fruit_reward:
                                a.reward = a.reward / float(
                                    tot_agent_levels * self._num_fruit_spawned
                                )

                        # despawn the fruit from the env
                        self.grid.set(*cell.pos, None)
                        cell.pos = np.array([-1, -1])
                        self.num_fruit_collected_per_room[self.current_task] += 1

                    else:
                        for a in curr_fruit_loading_agents:
                            a.reward += self.reward_config.failed_load_penalty

                    # remove these agents so they are not checked again
                    loading_agents_tmp -= set(curr_fruit_loading_agents)

                    # print(
                    #     f"Agents {[agent.index for agent in adj_agents]} with levels {adj_agent_levels} collected fruit at {fruit_pos} with level {fruit_level}"
                    # )

        if (
            self.num_fruit_collected_per_room[self.current_task]
            == self.num_fruit_per_room[self.current_task]
        ):
            self.all_room_fruit_collected[self.current_task] = True

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

            # make sure this is checked last to support LBFActions and LBFNavigationActions
            case self.actions.LOAD:
                next_pos = agent.pos

            # case _:
            #     raise ValueError(f"Invalid action: {action}")

        # convert from np ints to ints
        if isinstance(next_pos[0], np.int_):
            next_pos = tuple(map(int, next_pos))
        return next_pos

    def _terminated(self, room_completed: bool) -> bool:
        # Terminate the episode if all rooms have been completed
        # project is completed when you reach the end of the final room
        # TODO needs to be updated to handle non-sequential rooms
        if self.num_rooms > 1:
            return self.current_task == self.num_rooms
        else:
            return room_completed

    def _get_info(self, terminated: bool = False, room_completed: bool = False) -> dict:
        # step info
        info = {}

        # Agents only succeed at the full "project" if terminated = True
        # info["project_completed"] = terminated

        info["task_completed"] = room_completed

        return info

    # state
    @property
    def state(self) -> NDArray[tuple[int]] | NDArray:
        # define as a class attribute so you can access the state using env.state
        # no matter how many layers of wrappers are around this env
        # return a state of size (n_state_features)

        # use multigrid's basic state for now, come back to this later
        match self.state_type:
            case "original":
                # reproduce the original LBF "state" by concatenating all agent obs into a long vector
                # NOTE: technically not a "state" from an RL theory perspective since
                # there may be info not observed by any agent given their limited obs range
                obs = self.obs
                state = np.concatenate(obs)

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

        return state

    def _get_state_size(self) -> int:
        """standard function to interface with EPyMARL training loop,
        returns the flattened size of the global state."""
        match self.state_type:
            case "original":
                state_size: int = self._get_obs_size() * self.num_agents

            case "multigrid_flattened":
                state_size = prod(self.state.shape)

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
            case "original":
                # same as _make_gym_obs from original LBF
                obs = self._get_original_obs()

            case "multigrid_flattened":
                # dict where each key is (agent_idx, fruit_pos) and value is bool
                fruit_obs_mask = self._compute_fruit_obs_decisions()

                obs_list = self.gen_obs(
                    observe_other_agents=self.observe_other_agents,
                    fruit_obs_mask=fruit_obs_mask,
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

    # original lbf's obs
    def _get_original_obs(self) -> NDArray:
        joint_obs = np.vstack(
            [self._get_original_agent_obs(agent) for agent in self.agents]
        )

        """
            # not using this, low priority to implement
            # if self._grid_observation:
            #     layers = self._make_global_grid_arrays()
            #     agents_bounds = [
            #         self._get_agent_grid_bounds(*player.position) for player in self.players
            #     ]
            #     nobs = tuple(
            #         [
            #             layers[:, start_x:end_x, start_y:end_y]
            #             for start_x, end_x, start_y, end_y in agents_bounds
            #         ]
            #     )
            # else:
            #   nobs = tuple([self._make_obs_array(obs) for obs in observations])
        """

        return joint_obs

    def _get_original_agent_obs(
        self,
        agent: LBFAgent,
    ) -> NDArray:
        # get the agent's local obs
        radius_obs = self._get_neighborhood(*agent.pos, radius=self.agent_view_size)

        fruit_positions = np.vstack(
            np.where(radius_obs[:, :, 0] == self.world.OBJECT_TO_IDX["fruit"])
        ).T
        fruit_mask = np.ones(len(fruit_positions), dtype=bool)

        fruit_obs = self._get_original_object_obs(
            agent_radius_obs=radius_obs,
            obj_type="fruit",
            num_objects=self.max_num_fruit,
            include_mask=fruit_mask,
        )

        agent_obs = self._get_original_object_obs(
            agent_radius_obs=radius_obs,
            obj_type="agent",
            num_objects=self.num_agents,
            ego_agent=agent,
        )

        obs = np.concatenate((fruit_obs, agent_obs)).flatten()

        return obs

    def _get_original_object_obs(
        self,
        agent_radius_obs: NDArray,
        obj_type: Literal["agent", "fruit"],
        num_objects: int,
        ego_agent: Optional[LBFAgent] = None,
        include_mask: Optional[NDArray[np.bool_]] = None,
    ) -> NDArray:
        obj_obs = np.repeat(self.init_object_obs, repeats=num_objects, axis=0)

        obj_positions = np.vstack(
            np.where(agent_radius_obs[:, :, 0] == self.world.OBJECT_TO_IDX[obj_type])
        ).T

        # ego agent is first in its observations of the agents
        if ego_agent is not None:
            # ego agent's position in its local frame
            y, x = self._transform_to_neighborhood_original(
                origin=ego_agent.pos, sight=self.agent_view_size, pos=ego_agent.pos
            )
            obj_obs[0, :] = np.array([y, x, ego_agent.level])

            # remove ego_agent to avoid double counting
            rows_remove = np.argwhere(np.all(obj_positions == np.array([y, x]), axis=1))
            obj_positions = np.delete(obj_positions, rows_remove, axis=0)

        if include_mask is None:
            include_mask = np.ones(len(obj_positions), dtype=bool)

        for i, (y, x) in enumerate(obj_positions):
            if not include_mask[i]:
                continue
            level = agent_radius_obs[y, x, 2]
            obj_obs[i, :] = np.array([y, x, level])

        # remove agent levels from the obs
        if (not self.observe_agent_levels) and (obj_type == "agent"):
            obj_obs = obj_obs[:, 0:2]

        return obj_obs

    # updated obs
    def _compute_fruit_obs_decisions(
        self,
    ) -> dict[tuple[int, tuple[int, int]], bool] | None:
        match self.asymmetric_fruit_obs:
            case None:
                return None
            case "smacv2":
                return self._fruit_obs_smacv2()
            case "different_sensors":
                return self._fruit_obs_different_sensors()

    def _fruit_obs_smacv2(
        self,
    ):
        # loop over the fruit positions and check if each agent can view it
        fruit_positions = [
            obj.pos
            for obj in self.grid.grid
            if hasattr(obj, "type") and obj.type == "fruit"
        ]

        fruit_agents: dict[tuple[int, int], list[LBFAgent]] = defaultdict(list)
        for pos in fruit_positions:
            for agent in self.agents:
                if agent.can_view(*pos, obs_type=self.obs_type):
                    fruit_agents[pos].append(agent)

        # choose which agent gets to see the fruit based on order of observation
        # key = (fruit_pos, agent), val = visibility bool
        # decisions: dict[tuple[int, tuple[int, int]], bool] = {}
        for fruit_pos, agents in fruit_agents.items():
            # pick the first agent that sees each fruit in an episode
            if len(self._fruit_obs_state[fruit_pos]) == 0:
                if len(agents) == 1:
                    self._fruit_obs_state[fruit_pos] += agents
                else:
                    # if multiple agents see it at the same time, pick a random agent to be the one that sees it
                    self._fruit_obs_state[fruit_pos].append(
                        agents[self.np_random.integers(low=0, high=len(agents))]
                    )

            # assign fruit obs to the rest of the agents that are within obs range of the fruit based on probability
            for agent in agents:
                if (
                    agent not in self._fruit_obs_state[fruit_pos]
                    and self.np_random.random() < self.smacv2_fruit_obs_prob
                ):
                    self._fruit_obs_state[fruit_pos].append(agent)

        agent_fruit_obs_mask = {}
        for fruit_pos in fruit_positions:
            for agent in self._fruit_obs_state[fruit_pos]:
                fruit_pos_in_agent_frame = agent.get_view_coords(
                    *fruit_pos, obs_type=self.obs_type
                )
                agent_fruit_obs_mask |= {(agent, fruit_pos_in_agent_frame): True}

        return agent_fruit_obs_mask

    def _fruit_obs_different_sensors(self):
        # for each fruit, choose N agents at random and give them the ability to see the fruits
        # the other agents cannot see the fruits, but might be able to interact with them in other ways
        fruit_positions = [
            obj.pos
            for obj in self.grid.grid
            if hasattr(obj, "type") and obj.type == "fruit"
        ]

        agent_fruit_obs_mask = {}
        for fruit_pos in fruit_positions:
            if self.num_agents_fruit_obs > 0:
                indices = self.np_random.choice(
                    self.num_agents, size=self.num_agents_fruit_obs, replace=False
                )
            else:
                indices = []

            obs_agents = [self.agents[i] for i in indices]

            for agent in obs_agents:
                if len(self._fruit_obs_state[fruit_pos]) < self.num_agents_fruit_obs:
                    self._fruit_obs_state[fruit_pos].append(agent)

                fruit_pos_in_agent_frame = agent.get_view_coords(
                    *fruit_pos, obs_type=self.obs_type
                )
                agent_fruit_obs_mask |= {(agent, fruit_pos_in_agent_frame): True}

        return agent_fruit_obs_mask

    def _transform_to_neighborhood_original(
        self, center: tuple[int, int], sight: int, pos: tuple[int, int]
    ) -> tuple[int, int]:
        """
        # this matches the original lbf's implementation, but
        # I don't get the min(sight, center) thing
        """
        return (
            pos[0] - center[0] + min(sight, center[0]),
            pos[1] - center[1] + min(sight, center[1]),
        )

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
            case "original":
                obs_size: int = self.observation_space.shape[1]
            case "multigrid_flattened":
                obs_size: int = self.observation_space.shape[1]

                if self.goal_type == "simultaneous_arrival":
                    # for the first hitting time
                    obs_size += 1

            case _:
                raise NotImplementedError

        return obs_size

    def _set_observation_space(self) -> spaces.Space:
        match self.env_obs_type:
            case "original":
                # get obs space for a single agent
                agent_levels = sorted(self.max_agent_levels)
                max_fruit_level = (
                    self.max_fruit_level
                    if self.max_fruit_level is not None
                    else sum(agent_levels[:3])
                )
                max_pos = (self.height - 1, self.width - 1)

                # fruit obs bounds
                max_fruit_obs_single = np.array([[*max_pos, max_fruit_level]])

                min_fruit_obs = np.repeat(
                    self.init_object_obs, repeats=self.max_num_fruit, axis=0
                ).flatten()
                max_fruit_obs = np.repeat(
                    max_fruit_obs_single,
                    repeats=self.max_num_fruit,
                    axis=0,
                ).flatten()

                # agent obs bounds
                min_agent_obs_single = self.init_object_obs
                max_agent_obs_single = np.array([[*max_pos, max(agent_levels)]])

                if not self.observe_agent_levels:
                    min_agent_obs_single = min_agent_obs_single[:, 0:2]
                    max_agent_obs_single = max_agent_obs_single[:, 0:2]

                min_agent_obs = np.repeat(
                    min_agent_obs_single, repeats=self.num_agents, axis=0
                ).flatten()
                max_agent_obs = np.repeat(
                    max_agent_obs_single,
                    repeats=self.num_agents,
                    axis=0,
                ).flatten()

                # min and max total obs (single agent)
                min_obs_single = np.concat([min_fruit_obs, min_agent_obs]).reshape(
                    1, -1
                )
                max_obs_single = np.concat([max_fruit_obs, max_agent_obs]).reshape(
                    1, -1
                )

                # following original env
                # team_obs_space = spaces.Tuple(
                #     tuple([obs_space] * len(self.agents))
                # )

                # get joint obs space for the team
                team_obs_space = spaces.Box(
                    low=np.repeat(min_obs_single, repeats=self.num_agents, axis=0),
                    high=np.repeat(max_obs_single, repeats=self.num_agents, axis=0),
                    dtype=np.int_,
                )

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

            case _:
                raise NotImplementedError

        return team_obs_space

    # actions
    @property
    def avail_actions(self) -> list[list[int]]:
        # added this method to interface with PYMARL training loop
        # returns list of agent lists, where each agent's list has binary values representing
        # available actions
        # based on the MAIC paper's implementation of LBF with some minor cleanup
        # https://github.com/mansicer/MAIC/blob/main/src/envs/lbforaging/foraging.py
        return self._avail_actions

    def _get_valid_actions(self, agent: LBFAgent) -> list[int]:
        # handle actions that cause the agent to collide w/ a non-overlappable objects
        valid = []
        for action in self.actions:
            # non-moving actions are automatically valid
            next_pos = self._get_next_pos(agent, action.value)

            if self._check_valid_action(agent, action):
                if action.name in [
                    "STAY",
                    "LOAD",
                ]:
                    valid.append(action.value)
                elif self._check_valid_pos(next_pos):
                    valid.append(action.value)

        return valid

    def _check_valid_action(
        self, agent: LBFAgent, action: LBFActions | AlternativeNavigationActions
    ) -> bool:
        match action:
            case self.actions.STAY:
                return True

            # ensure agents do not go beyond the env's border
            # only matters if there is no wall around the border
            case self.actions.UP:
                return agent.pos[1] > 0

            case self.actions.DOWN:
                return agent.pos[1] < self.height - 1

            case self.actions.LEFT:
                return agent.pos[0] > 0

            case self.actions.RIGHT:
                return agent.pos[0] < self.width - 1

            # make sure this is checked last to support LBFActions and LBFNavigationActions
            case self.actions.LOAD:
                return self._adjacent_fruit(agent) > 0

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
        info_img = 255 * np.ones((img.shape[0], 5 * self.tile_size, 3), dtype=img.dtype)
        x_text = 5
        line_height = int(self.tile_size * 0.5)

        # header with basic info
        time_header = f"t : {self._t_render}"
        y_text = line_height
        putText(info_img, time_header, (x_text, y_text), **HEADER_TEXT_CONFIG)

        # draw each agent's action as text stacked vertically
        # extra spacing between different sections of text
        y_text += line_height
        action_header = "Agent : Pre-step action"
        putText(info_img, action_header, (x_text, y_text), **HEADER_TEXT_CONFIG)

        # ensure pre_step_actions is a 1D array
        if len(self._pre_step_actions.shape) > 1:
            self._pre_step_actions = self._pre_step_actions.flatten()

        # start_y set below header to avoid overlap
        for i, action in enumerate(self._pre_step_actions):
            # convert from int to action name if not none
            if action is not None:
                action = self.actions(action).name.title()

            text = f"{i} : {action}"
            y_text += line_height
            putText(info_img, text, (x_text, y_text), **RENDER_TEXT_CONFIG)

        # print visibility of the fruit by the agents
        y_text += line_height
        y_text += line_height
        fruit_obs_header = "Fruit : Obs. Agents"
        putText(info_img, fruit_obs_header, (x_text, y_text), **HEADER_TEXT_CONFIG)

        for i, (k, v) in enumerate(self._fruit_obs_state.items()):
            agents = [a.index for a in v]
            agents.sort()
            text = f"{int(k[0]), int(k[1])} : {agents}"

            y_text += line_height
            putText(info_img, text, (x_text, y_text), **RENDER_TEXT_CONFIG)

        # append info image to the right of env image
        img = np.concatenate([img, info_img], axis=1)

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

        "example text is here, this is my example text"
        return img

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
        return_object_type: Literal["fruit", "agent"] | None = None,
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

    def _adjacent_fruit(self, agent: LBFAgent) -> bool:
        """
        original logic for adjacent_fruit takes (x, y) as input and tells you the sum around it. Fruit encoded as a 1 in original lbf, so the logic below equivalent to "if any positions around me have a fruit in them, I can take the load action"

        Parameters
        ----------
        agent : LBFAgent

        Returns
        -------
        bool: true if agent is next to a piece of fruit
        """
        fruit_neighbor: list[bool] = [False] * len(agent.neighbor_pos)
        for i, neighbor_pos in enumerate(agent.neighbor_pos):
            if isinstance(self.grid.get(*neighbor_pos), Fruit):
                fruit_neighbor[i] = True

        return any(fruit_neighbor)

    def _check_valid_pos(self, pos: tuple[int, int], spawn: bool = False) -> bool:

        cell = self.grid.get(*pos)
        if not spawn:
            return cell is None or cell.can_overlap()

        else:
            # do not allow agents to spawn on top of other objects, can cause those objects to permanently despawn
            return cell is None
