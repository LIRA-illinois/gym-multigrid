# pylint: disable=line-too-long, dangerous-default-value
"""Defines the WildfireEnv class, which simulates dynamics of unmanned aerial vehicles (UAVs) fighting a spreading wildfire"""

import random
from collections import OrderedDict
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, Dict, Discrete

from gym_multigrid.core.agent import Actions, Agent, WildfireActions
from gym_multigrid.core.constants import (
    COLORS,
    STATE_IDX_TO_COLOR_WILDFIRE,
    STATE_TO_IDX_WILDFIRE,
    TILE_PIXELS,
)
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Tree
from gym_multigrid.core.world import WildfireWorld
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.utils.misc import (
    get_initial_fire_coordinates,
    render_agent_tiles,
)
from gym_multigrid.utils.window import Window


class WildfireEnv(MultiGridEnv):
    """Grid environment which simulates dynamics of unmanned aerial vehicles (UAVs) fighting a spreading wildfire"""

    helper_grid: Grid

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.99,
        delta_beta: float = 0,
        size: int = 17,
        num_agents: int = 2,
        agent_start_positions: tuple[tuple[int, int], ...] = ((1, 1), (15, 15)),
        agent_colors: tuple[str, ...] = ("red", "blue"),
        agent_groups: tuple[tuple[int, ...], ...] | None = None,
        agent_view_size: int = 10,
        initial_fire_size: int = 1,
        max_steps: int = 100,
        partial_obs: bool = False,
        actions_set: type[Actions] = WildfireActions,
        render_mode: str = "rgb_array",
        render_selfish_region_boundaries: bool = False,
        cooperative_reward: bool = False,
        altruism_weight: float = 0.2,
        log_selfish_region_metrics: bool = False,
        selfish_region_xmin: list[int] | None = None,
        selfish_region_xmax: list[int] | None = None,
        selfish_region_ymin: list[int] | None = None,
        selfish_region_ymax: list[int] | None = None,
    ):
        """Create a WildfireEnv environment

        Parameters
        ----------
        alpha : float, optional
            parameter for the wildfire dynamics model, by default 0.05
        beta : float, optional
            parameter for the wildfire dynamics model, by default 0.99
        delta_beta : float, optional
            parameter for the wildfire dynamics model, by default 0
        size : int, optional
            side of the square gridworld, by default 17
        num_agents : int, optional
            number of UAV agents, by default 2
        agent_start_positions : tuple[tuple[int,int]], optional
            tuple of tuples containing the start positions of the agents, in order of agent index. By default ((1, 1), (15, 15))
        agent_colors : tuple[str,str], optional
            tuple of strings of color names of all agents (or groups if agents are grouped) in order of increasing index. All agents in a group have the same color. The strings should be keys in the COLORS dictionary in constants.py. Only applicable if cooperative_reward is False. Fully cooperative agents have light_blue color by default. By default self-interested agents have red and blue colors
        agent_groups : tuple[tuple], optional
            tuple of tuples containing the indices (in ascending order) of agents in each group. Only applicable if cooperative_reward is False. By default None
        agent_view_size : int, optional
            side of the square region visible to an agent with partial observability, by default 10. Only applicable if partial_obs is True
        initial_fire_size : int, optional
            side of the square shaped initial fire region, by default 1
        max_steps : int, optional
            maximum number of steps in an episode, by default 100
        partial_obs : bool, optional
            whether agents have partial observability, by default False
        actions_set : WildfireActions, optional
            action space of the agents. All agents have the same action space. By default WildfireActions.
        render_mode : str, optional
            mode of rendering the environment, by default "rgb_array"
        render_selfish_region_boundaries : bool, optional
            whether to render boundaries of selfish regions, by default False
        cooperative_reward : bool, optional
            whether the agents use a cooperative reward, by default False. If True, the agents are fully cooperative and receive the same reward.
        altruism_weight : float, optional
            parameter to control altruism in the Markov game reward functions, by default 0.2. Only applicable if cooperative_reward is False. Should be in the range [0,1).
        log_selfish_region_metrics : bool, optional
            whether to log metrics related to trees in selfish regions, by default False
        selfish_region_xmin : list, optional
            list containing x-coordinates of the left boundaries of the regions of selfish interest for the agents (or groups if the agents are grouped. All agents in a group have same region of selfish interest). Regions of selfish interest are rectangular. List elements are in order of agent (or group) indices. Only applicable if cooperative_reward is False. By default None.
        selfish_region_xmax : list, optional
            list containing x-coordinates of the right boundaries of the regions of selfish interest for the agents (or groups if the agents are grouped. All agents in a group have same region of selfish interest). Regions of selfish interest are rectangular. List elements are in order of agent (or group) indices. Only applicable if cooperative_reward is False. By default None.
        selfish_region_ymin : list, optional
            list containing y-coordinates of the top boundaries of the regions of selfish interest for the agents (or groups if the agents are grouped. All agents in a group have same region of selfish interest). Regions of selfish interest are rectangular. List elements are in order of agent (or group) indices. Only applicable if cooperative_reward is False. By default None.
        selfish_region_ymax : list, optional
            list containing y-coordinates of the bottom boundaries of the regions of selfish interest for the agents (or groups if the agents are grouped. All agents in a group have same region of selfish interest). Regions of selfish interest are rectangular. List elements are in order of agent (or group) indices. Only applicable if cooperative_reward is False. By default None.
        """
        self.alpha = alpha
        self.beta = beta
        self.delta_beta = delta_beta
        self.num_agents = num_agents
        self.agent_start_positions = agent_start_positions
        self.agent_colors = agent_colors
        self.agent_groups = agent_groups
        if agent_groups:
            self.idx_to_group = {}
            for i, group in enumerate(agent_groups):
                for agent_index in group:
                    self.idx_to_group[agent_index] = i
        # observation vector of each agent is concatenation of obs_depth number of one-hot encodings, see paper for details. len(STATE_IDX_TO_COLOR_WILDFIRE) = the number of tree states
        self.obs_depth = self.num_agents + len(STATE_IDX_TO_COLOR_WILDFIRE)
        self.max_steps = max_steps
        self.world = WildfireWorld
        self.grid_size = size
        self.grid_size_without_walls = size - 2
        self.initial_fire_size = initial_fire_size
        self.burnt_trees = 0
        self.unburnt_trees = []
        self.trees_on_fire = 0
        self.cooperative_reward = cooperative_reward
        self.altruism_weight = altruism_weight
        if altruism_weight < 0 or altruism_weight >= 1:
            raise ValueError("Altruism weight should be in the range [0,1).")
        self.render_selfish_region_boundaries = render_selfish_region_boundaries
        self.log_selfish_region_metrics = log_selfish_region_metrics
        if self.log_selfish_region_metrics:
            # initialize attributes for logging metrics related to trees in selfish regions
            self.selfish_xmin = np.array(selfish_region_xmin)
            self.selfish_xmax = np.array(selfish_region_xmax)
            self.selfish_ymin = np.array(selfish_region_ymin)
            self.selfish_ymax = np.array(selfish_region_ymax)
            # raise error if xmin > xmax or ymin > ymax
            if np.any(self.selfish_xmin > self.selfish_xmax) or np.any(
                self.selfish_ymin > self.selfish_ymax
            ):
                raise ValueError(
                    "Invalid selfish region. xmin should be less than or equal to xmax and ymin should be less than or equal to ymax for every selfish region."
                )
            self.selfish_region_trees_on_fire = np.zeros(len(self.selfish_xmin))
            self.selfish_region_burnt_trees = np.zeros(len(self.selfish_xmin))
            self.selfish_region_size = (
                self.selfish_xmax
                - self.selfish_xmin
                + np.ones(len(selfish_region_xmin))
            ) * (
                self.selfish_ymax
                - self.selfish_ymin
                + np.ones(len(selfish_region_ymin))
            )

        if self.cooperative_reward:
            # initialize cooperative agents
            agents = [
                Agent(
                    world=self.world,
                    index=i,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color="light_blue",
                )
                for i in range(self.num_agents)
            ]
        elif self.agent_groups:
            # initialize self-interested agents with different colors
            agents = [
                Agent(
                    world=self.world,
                    index=i,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color=self.agent_colors[self.idx_to_group[i]],
                )
                for i in range(self.num_agents)
            ]
        else:
            # initialize self-interested agents with different colors
            agents = [
                Agent(
                    world=self.world,
                    index=i,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color=self.agent_colors[i],
                )
                for i in range(self.num_agents)
            ]

        super().__init__(
            agents=agents,
            grid_size=size,
            max_steps=max_steps,
            partial_obs=partial_obs,
            agent_view_size=agent_view_size,
            actions_set=actions_set,
            world=self.world,
            render_mode=render_mode,
        )
        self.observation_space: Box | Dict = self._set_observation_space()
        self.action_space = Dict(
            {f"{a.index}": Discrete(len(self.actions)) for a in self.agents}
        )

    def _set_observation_space(self) -> Dict:
        """Set the observation space for each agent in the environment. All agents possess the same observation space

        Returns
        -------
        observation_space : dict
            dictionary where each key is an agent's index and the value is the observation space
            for that agent
        """
        # observation vector of agent is the concatenation of obs_depth number of one-hot encodings where each encoding has grid_size_without_walls number of elements valued either 0 or 1. Additionally, the observation vector contains the normalized time step at the end
        low = np.full(self.obs_depth * ((self.grid_size_without_walls + 1) ** 2) + 1, 0)
        high = np.full(
            self.obs_depth * ((self.grid_size_without_walls + 1) ** 2) + 1, 1
        )
        observation_space = Dict(
            {
                f"{a.index}": Box(
                    low=low,
                    high=high,
                    dtype=np.float64,
                )
                for a in self.agents
            }
        )

        return observation_space

    def _gen_grid(self, width: int, height: int, state: npt.NDArray | None = None):
        """Generate the grid for the environment

        Parameters
        ----------
        width : int
            width of the grid
        height : int
            height of the grid
        state : ndarray, optional
            specifies the initial state of the environment, by default None.
            If none, it is chosen uniformly at random from the assumed initial state distribution
        """
        self.grid = Grid(width, height, self.world)

        # generate the walls of the grid
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        agent_start_pos = []
        if state is not None:
            # store positions of agents and trees on fire as per the specified initial state.
            state = state[:-1].reshape(
                (
                    self.obs_depth + 1,
                    self.grid_size,
                    self.grid_size,
                ),
            )
            initial_fire = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if state[1, j, i]:
                        # i and j are swapped because the y-coordinate specifies the row, while the x-coordinate specifies the column.
                        initial_fire.append((i, j))
                    for o in self.agents:
                        index = o.index
                        if state[len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + index, j, i]:
                            # i and j are swapped because the y-coordinate specifies the row, while the x-coordinate specifies the column.
                            agent_start_pos.append((i, j))

        else:
            # choose location of initial fire uniformly at random
            if self.initial_fire_size % 2 == 0:
                # for even sized initial fires, choose location of top left corner of fire region uniformly at random
                top_left_corner = (
                    random.randint(
                        1,
                        self.grid_size_without_walls - (self.initial_fire_size),
                    ),
                    random.randint(
                        1,
                        self.grid_size_without_walls - (self.initial_fire_size),
                    ),
                )
                initial_fire = get_initial_fire_coordinates(
                    *top_left_corner,
                    self.grid_size,
                    self.initial_fire_size,
                )
            else:
                # for odd sized initial fires, choose location of center of fire region uniformly at random
                fire_square_center = (
                    random.randint(
                        1 + ((self.initial_fire_size - 1) // 2),
                        self.grid_size_without_walls
                        - ((self.initial_fire_size - 1) // 2),
                    ),
                    random.randint(
                        1 + ((self.initial_fire_size - 1) // 2),
                        self.grid_size_without_walls
                        - ((self.initial_fire_size - 1) // 2),
                    ),
                )
                initial_fire = get_initial_fire_coordinates(
                    *fire_square_center,
                    self.grid_size,
                    self.initial_fire_size,
                )
            # agent_start_pos is specified during environment initialization
            agent_start_pos = self.agent_start_positions

        for pos in initial_fire:
            region = "common"
            if self.log_selfish_region_metrics:
                # update count of trees on fire in selfish regions
                if self.agent_groups:
                    for i, _ in enumerate(self.agent_groups):
                        if self.in_selfish_region(pos[0], pos[1], i):
                            # selfish region is identified by the lowest index among indices of the corresponding group of selfish agents
                            region = f"{i}"
                            self.selfish_region_trees_on_fire[i] += 1
                            break
                else:
                    for a in self.agents:
                        if self.in_selfish_region(pos[0], pos[1], a.index):
                            # selfish region is identified by the lowest index among indices of the corresponding selfish agent
                            region = f"{a.index}"
                            self.selfish_region_trees_on_fire[a.index] += 1
                            break
            # insert tree on fire in grid
            self.put_obj(
                Tree(self.world, STATE_TO_IDX_WILDFIRE["on fire"], region=region),
                int(pos[0]),
                int(pos[1]),
            )

        # update counts of trees on fire and healthy trees
        self.trees_on_fire += self.initial_fire_size**2
        num_healthy_trees = self.grid_size_without_walls**2 - len(initial_fire)

        # insert healthy tree in grid
        for _ in range(num_healthy_trees):
            tree_obj = Tree(self.world, STATE_TO_IDX_WILDFIRE["healthy"])
            self.place_obj(tree_obj)
            if self.log_selfish_region_metrics:
                # check if tree is in a selfish region, and update region attribute of tree if it is
                if self.agent_groups:
                    for i, _ in enumerate(self.agent_groups):
                        if self.in_selfish_region(
                            *(tree_obj.pos),
                            i,  # pylint: disable=not-an-iterable
                        ):
                            tree_obj.region = f"{i}"
                            break
                else:
                    for a in self.agents:
                        if self.in_selfish_region(
                            *(tree_obj.pos),
                            a.index,  # pylint: disable=not-an-iterable
                        ):
                            tree_obj.region = f"{a.index}"
                            break

        # helper grid is a work around for grid being unable to store multiple objects at a single cell. It does not contain agents
        self.helper_grid = self.grid.copy()

        # create list of unburnt trees. initial state does not have burnt trees.
        for c in self.helper_grid.grid:
            if c is not None and c.type == "tree":
                self.unburnt_trees.append(c)

        # insert agents in grid
        for i, a in enumerate(self.agents):
            self.place_agent(a, pos=agent_start_pos[i])
            self.helper_grid.get(*agent_start_pos[i]).agent_above = True

    def _get_obs(self) -> list[npt.NDArray]:
        """Get observation vectors of all agents in the environment.

        Returns
        -------
        agent_obs: list(ndarray)
            list of agent observations where the element at i^th list index is the observation vector for the agent with index i.
        """
        # initialize list of observation vector of each agent
        agent_obs = [
            np.zeros(
                (
                    self.obs_depth,
                    self.grid_size_without_walls + 1,
                    self.grid_size_without_walls + 1,
                ),
                dtype=np.float64,
            )
            for _ in range(self.num_agents)
        ]

        # update walls and tree states in agent observations
        for obj in self.helper_grid.grid:
            if obj is None:
                continue
            i, j = obj.pos
            for a in self.agents:
                # convert to agent centered coordinates
                nc = [i - a.pos[0], j - a.pos[1]]
                # wrap around to get agent centered toroidal coordinates
                if nc[0] < 0:
                    nc[0] += self.grid_size_without_walls + 1
                if nc[1] < 0:
                    nc[1] += self.grid_size_without_walls + 1
                # update agent's observation. # switch x and y coordinates because the y-coordinate specifies the row, while the x-coordinate specifies the column
                if obj.type == "tree":
                    agent_obs[a.index][obj.state, nc[1], nc[0]] = 1
                elif obj.type == "wall":
                    agent_obs[a.index][
                        len(STATE_IDX_TO_COLOR_WILDFIRE), nc[1], nc[0]
                    ] = 1

        # for each agent, update other agents' positions in agent observations
        for a in self.agents:
            for o in self.agents:
                if o.index != a.index:
                    idx = o.index - int(np.heaviside(o.index - a.index, 0))
                    # convert to agent centered coordinates
                    nc = [
                        o.pos[0] - a.pos[0],
                        o.pos[1] - a.pos[1],
                    ]
                    # wrap around to get agent centered toroidal coordinates
                    if nc[0] < 0:
                        nc[0] += self.grid_size_without_walls + 1
                    if nc[1] < 0:
                        nc[1] += self.grid_size_without_walls + 1
                    agent_obs[a.index][
                        len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + idx,
                        nc[1],
                        nc[0],
                    ] = 1

        # flatten, and append normalized time step at the end of, each agent observation
        for a in self.agents:
            agent_obs[a.index] = np.append(
                agent_obs[a.index].flatten(),
                np.array(self.step_count / self.max_steps, dtype=np.float64),
            )
        return agent_obs

    def get_state(self) -> npt.NDArray:
        """Get the state representation of the environment.

        Returns
        -------
        ndarray
            state representation of the environment
        """
        # initialize array to store state vector
        s = np.zeros(
            (
                self.obs_depth + 1,
                self.grid_size,
                self.grid_size,
            ),
            dtype=np.float64,
        )

        # update tree states and walls in state representation
        for o in self.helper_grid.grid:
            # switch x and y coordinates because the y-coordinate specifies the row, while the x-coordinate specifies the column
            if o.type == "tree":
                s[o.state, o.pos[1], o.pos[0]] = 1
            if o.type == "wall":
                s[len(STATE_IDX_TO_COLOR_WILDFIRE), o.pos[1], o.pos[0]] = 1

        # update agent positions in state representation
        for a in self.agents:
            s[
                len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + a.index,
                a.pos[1],
                a.pos[0],
            ] = 1

        # flatten, and append normalized time step at the end of, state representation
        s = np.append(
            s.flatten(),
            np.array(self.step_count / self.max_steps, dtype=np.float64),
        )
        return s

    def get_state_interpretation(
        self, state: npt.NDArray, print_interpretation: bool = True
    ) -> tuple[list[tuple[int, int]], float]:
        """Get human readable interpretation of the state of the environment

        Parameters
        ----------
        state : ndarray
            state representation of the environment
        print_interpretation : bool, optional
           whether to print the interpretation of the state, by default True. Human readable interpretation refers to printing the positions of trees on fire, agents, and the time step.

        Returns
        -------
        trees_on_fire : list[tuple[int,int]]
            list of tuples containing position coordinates (x,y) of trees on fire.
        time_step : float
            normalized time step of the episode at which time the state was recorded
        """
        time_step = state[-1]
        state = state[:-1].reshape(
            (
                self.obs_depth + 1,
                self.grid_size,
                self.grid_size,
            ),
        )
        if print_interpretation:
            print("-------------------------------------------------------------")
            print("State interpretation:")
        trees_on_fire = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if state[1, j, i] == 1:
                    if print_interpretation:
                        print(f"Tree at position {(i, j)} is on fire.")
                    trees_on_fire.append((i, j))
                for o in self.agents:
                    index = o.index
                    if state[len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + index, j, i] == 1:
                        if print_interpretation:
                            print(f"Agent {o.index} is at position {(i, j)}.")
        if print_interpretation:
            print(f"Time step: {time_step}")
            print("-------------------------------------------------------------")
        return trees_on_fire, time_step

    def construct_state(
        self,
        trees_on_fire: list[tuple[int, int]],
        agent_pos: list[tuple[int, int]],
        time_step: int,
    ) -> npt.NDArray:
        """Construct the state representation vector of the environment for given positions of trees on fire and agents

        Parameters
        ----------
        trees_on_fire : list
            list of tuples containing position coordinates (x,y) of trees on fire.
        agent_pos : list
            list of tuples containing position coordinates (x,y) of agents, in order of agent index.
        time_step : int
            normalized time step of the episode at which time the state was recorded.

        Returns
        -------
        state : ndarray
            state representation of the environment
        """
        state = np.zeros(
            (
                self.obs_depth + 1,
                self.grid_size,
                self.grid_size,
            ),
            dtype=np.float64,
        )
        # update tree states and walls in state representation. There are no burnt trees in the state
        state[0, :, :] = 1
        for pos in trees_on_fire:
            state[1, pos[1], pos[0]] = 1
            state[0, pos[1], pos[0]] = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (
                    i == 0
                    or i == self.grid_size - 1
                    or j == 0
                    or j == self.grid_size - 1
                ):
                    state[len(STATE_IDX_TO_COLOR_WILDFIRE), j, i] = 1
                    state[0, j, i] = 0

        # update agent positions in state representation
        for i, pos in enumerate(agent_pos):
            state[
                len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + i,
                pos[1],
                pos[0],
            ] = 1

        # flatten, and append normalized time step at the end of, state representation
        state = np.append(
            state.flatten(),
            np.array(time_step / self.max_steps, dtype=np.float64),
        )
        return state

    def reset(
        self, seed: int | None = None, options: dict[str, npt.NDArray] | None = None
    ) -> tuple[OrderedDict[str, npt.NDArray], dict[str, int]]:
        """Reset the state of the environment

        Parameters
        ----------
        seed : int, optional
            seed for random number generator, by default None
        options : dict, optional
            dictionary containing additional option for resetting the environment to a specified initial state. By default None. The dictionary key must be "state" and value must be a numpy array representing the state. If options = none, initial state is chosen uniformly at random from initial state distribution.

        Returns
        -------
        obs : OrderedDict
            dictionary where each key is the agent index and the value is the observation vector for that agent.
        info : dict
            dictionary containing additional information about the environment.
            Here, it contains the number of burnt trees in the environment after reset
        """
        # reset environment attributes
        self.burnt_trees = 0
        self.trees_on_fire = 0
        self.unburnt_trees = []
        if self.log_selfish_region_metrics:
            self.selfish_region_trees_on_fire = np.zeros(len(self.selfish_xmin))
            self.selfish_region_burnt_trees = np.zeros(len(self.selfish_xmin))

        # parse options
        super().reset(seed=seed)

        # get agent observations
        agent_obs = self._get_obs()
        obs = OrderedDict({f"{a.index}": agent_obs[a.index] for a in self.agents})

        # create info dictionary
        info = {"burnt trees": self.burnt_trees}
        return obs, info

    def move_agent(self, i: int, next_pos: tuple[int, int]):
        """Move agent to a new position in the grid

        Parameters
        ----------
        i : int
            index of agent to be moved
        next_pos : tuple[int, int]
            coordinates of new position
        """
        # add agent to grid in new position
        self.grid.set(*next_pos, self.agents[i])

        # get tree in agent's old position from helper grid and add tree to grid
        tree = self.helper_grid.get(*self.agents[i].pos)
        tree.agent_above = False
        self.grid.set(*self.agents[i].pos, tree)

        # update attributes
        next_tree = self.helper_grid.get(*next_pos)
        next_tree.agent_above = True
        self.agents[i].pos = next_pos

    def neighbors_on_fire(self, tree_pos: tuple[int, int]) -> int:
        """Get the number of neighboring trees on fire for a given tree.
           Neighbors are adjacent trees in the cardinal directions. A tree can have at most 4 neighbors

        Parameters
        ----------
        tree_pos : tuple[int, int]
            position coordinates of the tree whose neighbors are to be checked

        Returns
        -------
        num : int
            the number of neighboring trees on fire
        """
        num = 0
        relative_pos = [
            np.array([1, 0]),
            np.array([-1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
        ]
        for r in relative_pos:
            neighbor_pos = tree_pos + r
            if neighbor_pos[0] >= 0 and neighbor_pos[0] < self.helper_grid.width:
                if neighbor_pos[1] >= 0 and neighbor_pos[1] < self.helper_grid.height:
                    o = self.helper_grid.get(*neighbor_pos)
                    if o is not None and o.type == "tree":
                        if o.state == 1:
                            num += 1
        return num

    def in_selfish_region(self, i: int, j: int, region_index: int) -> bool:
        """Check if given tree is in region of selfish interest with given index

        Parameters
        ----------
        i : int
            x-coordinate of tree position
        j : int
            y-coordinate of tree position
        region_index : int
            index of selfish region. Same as index of corresponding selfish agent (or group of selfish agents).

        Returns
        -------
        bool
            True, if tree is in the region of selfish interest. Otherwise, False
        """
        return (
            i >= self.selfish_xmin[region_index]
            and i <= self.selfish_xmax[region_index]
            and j >= self.selfish_ymin[region_index]
            and j <= self.selfish_ymax[region_index]
        )

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        OrderedDict[str, npt.NDArray], dict[str, int], bool, dict[str, dict[str, int]]
    ]:
        """Take a step in the environment. Wildfire dynamics are propagated by one time step, and agents move according to their actions.

        Parameters
        ----------
        actions : dict
            dictionary containing actions for each agent. The key is the agent index and the value is the action.

        Returns
        -------
        next_obs : OrderedDict
            dictionary where each key is the agent index and the value is the new observation vector (ndarray) for that agent after the environment step.
        rewards : dict
            dictionary where each key is the agent index and the value is the reward given to that agent after the environment step.
        terminated : bool
            True, if the episode is terminated, otherwise, False. Episode is terminated if there are zero trees on fire.
        truncated : bool
            True, if the episode is truncated, otherwise, False. Episode is done if maximum number of steps is reached.
        info : dict
            dictionary where each key is the agent index and the value is an info dictionary containing additional information about the environment. Here, each agent's info dictionary contains the same information, viz., the number of burnt trees.
        """
        self.step_count += 1
        actions = list(actions.values())
        terminated = False
        truncated = False

        # Move agents sequentially, in random order
        order = np.random.permutation(len(actions))
        for i in order:
            if actions[i] == self.actions.STILL:
                continue
            if actions[i] == self.actions.NORTH:
                next_pos = self.agents[i].north_pos()
                next_cell = self.grid.get(*next_pos)
                if next_cell is None or next_cell.can_overlap():
                    self.move_agent(i, next_pos)
            elif actions[i] == self.actions.SOUTH:
                next_pos = self.agents[i].south_pos()
                next_cell = self.grid.get(*next_pos)
                if next_cell is None or next_cell.can_overlap():
                    self.move_agent(i, next_pos)
            elif actions[i] == self.actions.EAST:
                next_pos = self.agents[i].east_pos()
                next_cell = self.grid.get(*next_pos)
                if next_cell is None or next_cell.can_overlap():
                    self.move_agent(i, next_pos)
            elif actions[i] == self.actions.WEST:
                next_pos = self.agents[i].west_pos()
                next_cell = self.grid.get(*next_pos)
                if next_cell is None or next_cell.can_overlap():
                    self.move_agent(i, next_pos)

        # propagate wildfire dynamics by one time step
        # initialize lists to store trees transitioning to on fire and burnt state in the current time step
        trees_to_fire_state = []
        # ensure num_trees_to_fire_state_sr is always defined to avoid UnboundLocalError in reward computation
        num_trees_to_fire_state_sr = {}
        if self.log_selfish_region_metrics:
            num_trees_to_fire_state_sr = {
                f"{i}": 0 for i, _ in enumerate(self.selfish_xmin)
            }
        trees_to_burnt_state = []
        # loop over all unburnt trees. Burnt trees remain burnt
        for c in self.unburnt_trees:
            if c.state == 0:
                pos = np.array(c.pos)
                # transition from healthy to on fire with probability 1 - (1 - alpha)^n
                if np.random.rand() < 1 - (1 - self.alpha) ** self.neighbors_on_fire(
                    pos
                ):
                    # update relevant attributes and lists
                    trees_to_fire_state.append(c)
                    self.trees_on_fire += 1
                    if self.log_selfish_region_metrics:
                        if c.region != "common":
                            self.selfish_region_trees_on_fire[int(c.region)] += 1
                            num_trees_to_fire_state_sr[c.region] += 1
            if c.state == 1:
                # transition from on fire to burnt with probability 1 - beta + delta_beta * agent_above
                if np.random.rand() < 1 - self.beta + c.agent_above * self.delta_beta:
                    # update relevant attributes and list
                    trees_to_burnt_state.append(c)
                    self.burnt_trees += 1
                    self.trees_on_fire -= 1
                    if self.log_selfish_region_metrics:
                        if c.region != "common":
                            self.selfish_region_burnt_trees[int(c.region)] += 1
                            self.selfish_region_trees_on_fire[int(c.region)] -= 1

        # update tree objects in helper grid and grid. This update is done after the loop to avoid affecting
        # the transition probabilities of trees later in the loop due to trees that have already transitioned earlier in the loop.
        for c in trees_to_fire_state:
            c.state = 1
            c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]
            # update tree in grid
            o = self.grid.get(c.pos[0], c.pos[1])
            if o.type == "tree":
                o.state = 1
                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]
        for c in trees_to_burnt_state:
            self.unburnt_trees.remove(c)
            c.state = 2
            c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]
            # update tree in grid
            o = self.grid.get(c.pos[0], c.pos[1])
            if o.type == "tree":
                o.state = 2
                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]

        # check if episode is done
        if self.trees_on_fire == 0:
            terminated = True
            rewards = {f"{a.index}": 0 for a in self.agents}
        elif self.step_count >= self.max_steps:
            truncated = True
            rewards = {f"{a.index}": 0 for a in self.agents}
        else:
            # compute agent rewards
            agent_rewards = np.zeros(self.num_agents)
            if self.cooperative_reward:
                agent_rewards -= 0.5 * len(trees_to_fire_state)
            if not self.cooperative_reward:
                for a in self.agents:
                    own_fires = num_trees_to_fire_state_sr.get(f"{a.index}", 0)
                    agent_rewards[a.index] -= 0.5 * (
                        own_fires
                        + self.altruism_weight * (len(trees_to_fire_state) - own_fires)
                    )
            # agent rewards dictionary
            rewards = {f"{a.index}": agent_rewards[a.index] for a in self.agents}

        # get agent observations after the environment step
        agent_obs = self._get_obs()
        next_obs = OrderedDict({f"{a.index}": agent_obs[a.index] for a in self.agents})

        # info dictionary
        info = {"burnt trees": self.burnt_trees}
        infos = {f"{a.index}": info for a in self.agents}

        return next_obs, rewards, terminated, truncated, infos

    def render(
        self, close: bool = False, highlight: bool = False, tile_size: int = TILE_PIXELS
    ) -> npt.NDArray:
        """Render the whole-grid human view

        Parameters
        ----------
        close : bool, optional
            close the rendering window, by default False. Only applicable if render_mode is "human"
        highlight : bool, optional
            highlight the cells visible to the agent, by default False
        tile_size : int, optional
            size of each tile in pixels, by default TILE_PIXELS (defined in constants.py)

        Returns
        -------
        img : ndarray
            image of the grid
        """
        if close:
            if self.window:
                self.window.close()
            return

        if self.render_mode == "human" and not self.window:
            self.window = Window("gym_multigrid")
            self.window.show(block=False)

        if highlight:
            # Compute which cells are visible to the agent
            _, vis_masks = self.gen_obs_grid()

            highlight_masks = {
                (i, j): [] for i in range(self.width) for j in range(self.height)
            }

            for i, a in enumerate(self.agents):
                # Compute the world coordinates of the bottom-left corner
                # of the agent's view area
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

        # Render the grid
        if self.render_selfish_region_boundaries:
            # include selfish region boundaries in the render
            colors = [COLORS[color] for color in self.agent_colors]
            img = self.grid.render(
                tile_size,
                highlight_masks=highlight_masks if highlight else None,
                uncached_object_types=self.uncached_object_types,
                x_min=self.selfish_xmin,
                y_min=self.selfish_ymin,
                x_max=self.selfish_xmax,
                y_max=self.selfish_ymax,
                colors=colors,
            )
        else:
            img = self.grid.render(
                tile_size,
                highlight_masks=highlight_masks if highlight else None,
                uncached_object_types=self.uncached_object_types,
            )

        # Re-render the tiles containing agents to include trees below agent
        if self.render_selfish_region_boundaries:
            # include selfish region boundaries in the render
            for a in self.agents:
                img = render_agent_tiles(
                    img,
                    a,
                    self.helper_grid,
                    self.world,
                    x_min=self.selfish_xmin,
                    y_min=self.selfish_ymin,
                    x_max=self.selfish_xmax,
                    y_max=self.selfish_ymax,
                    colors=colors,
                )
        else:
            for a in self.agents:
                img = render_agent_tiles(img, a, self.helper_grid, self.world)

        if self.render_mode == "human":
            self.window.show_img(img)

        return img
