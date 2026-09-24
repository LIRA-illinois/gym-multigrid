import pdb
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional, TypeAlias, TypedDict

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from gym_multigrid.core.agent import NAV_DIR_TO_VEC, Actions, Agent, NavigationActions
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import AgentGoal, Wall, WorldObj, Zone
from gym_multigrid.core.object import SimpleDoor as Door
from gym_multigrid.core.world import LabyrinthWorld, World
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.typing import Position
from gym_multigrid.utils.subtasks import (
    EnvObjectGroup,
    HLMDPConfig,
    ObjectGroup,
    ObjGroupT,
    PositionDist,
    StateData,
    SubtaskData,
)

# HLMDP interface
state_data_tuple = (
    StateData(
        idx=0,
        outgoing_init_state_dist=PositionDist(
            probs=(1.0,),
            states=((1, 3), (1, 6)),
        ),
    ),
    StateData(
        idx=1,
        outgoing_init_state_dist=PositionDist(probs=(1.0,), states=((3, 3), (3, 6))),
    ),
)


subtask_data_tuple = (
    SubtaskData(
        edge=(0, 1),
        idx=0,
        final_state=((3, 3), (3, 6)),
        termination_condition="reach_assigned_final_state",
    ),
    SubtaskData(
        edge=(1, 2),
        idx=1,
        final_state=((5, 3), (5, 6)),
        termination_condition="reach_assigned_final_state",
    ),
)


hlmdp_config = HLMDPConfig(
    state_data_tuple=state_data_tuple, subtask_data_tuple=subtask_data_tuple
)


# Classes used to interface with the env
class EnvInfo(TypedDict):
    """info about the environment used for PyMARL training"""

    state_shape: int
    obs_shape: int
    n_actions: int
    n_agents: int


class StepInfo(TypedDict):
    """info used in the env's step function"""

    success: bool


@dataclass
class RewardConfig:
    movement_reward: float
    agent_reach_goal_reward: float
    agent_leave_goal_reward: float
    all_agents_at_goal_reward: float


# Env config
reward_config = RewardConfig(
    movement_reward=-0.02,
    agent_reach_goal_reward=0.2,
    agent_leave_goal_reward=-0.3,
    all_agents_at_goal_reward=1.0,
)


Observation: TypeAlias = dict[str, NDArray[np.int_]] | NDArray[np.int_]


class LabyrinthEnv(MultiGridEnv):
    metadata = {"render_fps": 10, "render_modes": ["human", "rgb_array"]}

    # setup and env properties
    def __init__(
        self,
        height: int = 10,
        width: int = 7,
        num_agents: int = 2,
        p_intended_movement: float = 0.95,
        actions_set: type[Actions] = NavigationActions,
        subtask_idx: int = 0,
        world: World = LabyrinthWorld,
        hlmdp_config: HLMDPConfig = hlmdp_config,
        observation_option: Literal["goal"] = "goal",
        obs_type: Literal["dict", "array", "array_scaled"] = "array_scaled",
        reward_config: RewardConfig = reward_config,
        agent_dir_to_vec: list[NDArray[np.int_]] = NAV_DIR_TO_VEC,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Constructor for the LabyrinthEnv class.

        Parameters
        ----------
        height : int = 9
            Height of the grid.
        width : int = 10
            Width of the grid.
        num_agents : list[int] = [0, 1, 2]
            indices dxs of agents in the environment.
        p_intended_movement : float = 0.95
            Probability of the intended movement.
            Should be in the range [0, 1].
        subtask_idx: int = 0
            The current subtask index.
        hlmdp_config: HLMDPConfig = hlmdp_config
            Defines the structure of the leader's high-level MDP
        observation_option : Literal["goal"] = "goal"
            Observation option.
            - "goal": The observation includes agent positions and position of the assigned goal.
        actions_set : type[Actions] = NavigationActions
            Set of actions for the agents.
            By default, there are five actions: "stay", "up", "right", "down", and "left".
        agent_dir_to_vec : list[NDArray[np.int_]] = NAV_DIR_TO_VEC
            Direction vectors for the agents.
            The length of the list should be equal to the number of actions in the actions set.
        reward_config: RewardConfig
            Configuration for conditions that cause the reward function to output non-zero reward
        world : World = LabyrinthWorld
            World for the environment.
        render_mode : Literal["human", "rgb_array"] = "rgb_array"
            Render mode for the environment.
        """
        self.num_agents: int = num_agents
        self.p_intended_movement: float = p_intended_movement
        self.subtask_idx: int = subtask_idx
        self.hlmdp_config: HLMDPConfig = hlmdp_config
        self.reward_config: RewardConfig = reward_config

        # observation config
        self.observation_option: Literal["goal"] = observation_option
        self.obs_type: Literal["dict", "array", "array_scaled"] = obs_type

        # agent config
        agent_view_size: int | None = None
        agents: list[Agent] = [
            Agent(
                world=world,
                index=i,
                view_size=agent_view_size,
                actions=actions_set,
                dir_to_vec=agent_dir_to_vec,
            )
            for i in range(num_agents)
        ]

        # goal config
        self.goals: tuple[tuple[int, int], ...] = ()

        uncached_object_types: list[str] = ["agent"]

        super().__init__(
            agents=agents,
            width=width,
            height=height,
            actions_set=actions_set,
            world=world,
            render_mode=render_mode,
            uncached_object_types=uncached_object_types,
        )

        # define the action space
        self.action_space = spaces.MultiDiscrete(
            [len(self.actions) for _ in range(self.num_agents)]
        )

        # define available objects in this environment
        self.object_options: dict[str, WorldObj] = {
            "goal": AgentGoal,
            "door": Door,
            "zone": Zone,
            "wall": Wall,
        }

    def _set_observation_space(self) -> spaces.Box:
        max_x: int = self.width - 1
        max_y: int = self.height - 1

        if self.observation_option == "goal":
            goal_indices: list[int] = [1 for _ in range(self.num_agents)]
        else:
            raise ValueError(f"Invalid observation option: {self.observation_option}")

        if self.obs_type == "dict":
            observation_space = spaces.Dict(
                {
                    str(i): spaces.Box(
                        low=np.zeros(2 * (num_goals + 1)),
                        high=np.array([max_x, max_y] * (num_goals + 1)),
                        dtype=np.int_,
                    )
                    for i, num_goals in enumerate(goal_indices)
                }
            )

        elif self.obs_type in ["array", "array_scaled"]:
            obs_shape = (self.num_agents, 4)
            if self.obs_type == "array":
                max_val = np.max((max_x, max_y))
            else:
                max_val = 1

            observation_space = spaces.Box(
                low=np.zeros(obs_shape),
                high=max_val * np.ones(obs_shape),
                dtype=np.float64,
            )

        return observation_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[NDArray[np.int_], StepInfo]:
        super().reset(seed=seed, options=options)

        obs: Observation = self.get_obs()
        info: StepInfo = self._get_step_info()

        return obs, info

    def _gen_grid(self, width, height) -> None:
        self.grid = Grid(width, height, self.world)

        # make a list of all the objects you want to spawn in the env
        env_object_config = [
            # surrounding wall
            EnvObjectGroup(
                obj_type="wall",
                group_index=0,
                pos=((0, 0, 7, 10),),
                color="grey",
                spawned_subtask_indices=(0, 1),
                fill_mode="empty",
            ),
            # middle walls
            EnvObjectGroup(
                obj_type="wall",
                group_index=1,
                pos=((3, 1),),
                color="grey",
                spawned_subtask_indices=(0, 1),
                fill_mode="empty",
            ),
            EnvObjectGroup(
                obj_type="wall",
                group_index=2,
                pos=((3, 8),),
                color="grey",
                spawned_subtask_indices=(0, 1),
                fill_mode="empty",
            ),
            EnvObjectGroup(
                obj_type="wall",
                group_index=2,
                pos=((3, 4, 1, 2),),
                color="grey",
                spawned_subtask_indices=(0, 1),
                fill_mode="empty",
            ),
        ]

        # add the goals for the current subtask
        for subtask_idx, data in self.hlmdp_config.subtask_data.items():
            if subtask_idx == self.subtask_idx:
                env_object_config.append(
                    EnvObjectGroup(
                        obj_type="goal",
                        group_index=subtask_idx,
                        pos=data.final_state,
                        color="green",
                        spawned_subtask_indices=(subtask_idx),
                        fill_mode="empty",
                    )
                )

        obj_group_dict: dict[str, dict[int, ObjGroupT]] = {}

        # Place objects
        for obj_group_config in env_object_config:
            obj_type: str = obj_group_config.obj_type
            group_index: int = obj_group_config.group_index

            if obj_type not in obj_group_dict:
                obj_group_dict[obj_type] = {}
            else:
                pass

            obj_group_dict[obj_type][group_index] = ObjectGroup(
                object_options=self.object_options, **asdict(obj_group_config)
            )
            obj_group_dict[obj_type][group_index].put_objects(self.grid, self.world)

        self.obj_group_dict = obj_group_dict

        # init_grid has to be defined before the agents are placed
        self.init_grid: Grid = self.grid.copy()

        # Place the agents
        # pick the init_pos based on the subtask (assuming it is an init pos and not a distribution to sample from)
        init_pos = self.hlmdp_config.subtask_data[
            self.subtask_idx
        ].init_state_dist.states

        assert len(self.agents) == len(init_pos)
        for agent, pos in zip(self.agents, init_pos):
            self.place_agent(agent, pos)

    def get_obs(self) -> Observation:
        if self.obs_type == "dict":
            obs: dict[str, Any] = {}

        elif self.obs_type in ["array", "array_scaled"]:
            obs: NDArray[np.int_] = np.zeros((self.num_agents, 4), dtype=np.float64)

        else:
            obs = None

        for agent in self.agents:
            # agent_obs: [agent_x, agent_y, assigned_goal_x, assigned_goal_y]
            agent_obs = np.array(agent.pos, dtype=np.float64)

            # agent obs needs to be an np array here b/c final state is already an array

            if self.observation_option == "goal":
                goal_state = self.hlmdp_config.subtask_data[
                    self.subtask_idx
                ].final_state[agent.index]
                agent_obs = np.append(agent_obs, goal_state)

            if self.obs_type == "dict":
                obs[str(agent.index)] = agent_obs.flatten()

            elif self.obs_type in ["array", "array_scaled"]:
                obs[agent.index, :] = agent_obs.flatten()

        if self.obs_type == "array_scaled":
            # subtract 2 b/c we assume an outer wall around the env
            max_x: int = self.width - 2
            max_y: int = self.height - 2
            obs_scaled = obs / np.array([max_x, max_y, max_x, max_y], dtype=np.float64)
            return obs_scaled

        return obs

    def get_env_info(self) -> EnvInfo:
        # obs_shape should only be the shape of a single agent
        # self.observation_space.shape = (n_agents, dim_1_size, dim_2_size, ...)
        obs_shape = int(np.prod(self.observation_space.shape[1:]))

        env_info: EnvInfo = {
            "state_shape": self.width * self.height * self.world.encode_dim,
            "obs_shape": obs_shape,
            "n_actions": len(self.actions),
            "n_agents": self.num_agents,
        }

        return env_info

    def get_avail_actions(self, verbose=False) -> list[list[bool]]:
        """gets available actions for each agent

        Returns
        -------
        avail_actions: list[list[bool]]
            available actions for each agent
        """
        avail_actions = []
        for agent in self.agents:
            avail_actions.append(self._get_avail_actions_agent(agent))

        if verbose:
            self.print_avail_actions_str(avail_actions)

        return avail_actions

    def print_avail_actions_str(self, avail_actions):
        if self.actions == NavigationActions:
            actions = [a.name for a in NavigationActions]

            print("Available actions")
            for agent_idx, avail_actions_agent in enumerate(avail_actions):
                action_str = ""
                for i, avail in enumerate(avail_actions_agent):
                    if avail:
                        action_str += f"{actions[i]},  "

                print(f"Agent: {agent_idx} --- {action_str}")

    def _get_avail_actions_agent(self, agent: Agent) -> list[bool]:
        if self.actions == NavigationActions:
            avail_actions_tmp: dict[int, bool] = {}

            # populate with filler data
            for action in NavigationActions:
                avail_actions_tmp[action.value] = True

            # set the values in avail_actions
            # you should be able to set the desired order of the neighbor positions here
            neighbor_positions: dict[str, NDArray[np.int_]] = (
                agent.get_all_neighbor_pos()
            )

            for direction, pos in neighbor_positions.items():
                neighbor_cell = self.grid.get(*pos)
                if (neighbor_cell is None) or (neighbor_cell.can_overlap()):
                    continue
                else:
                    avail_actions_tmp[NavigationActions[direction].value] = False

            # turn avail_actions into a list with the ordering of the actions same as in NavigationActions
            avail_actions: list[bool] = list(avail_actions_tmp.values())

        else:
            raise NotImplementedError(
                "This action space is not implemented in _get_avail_actions"
            )

        return avail_actions

    # general env step logic
    def step(
        self,
        actions: NDArray[np.int_],
    ) -> tuple[Observation, float, bool, bool, StepInfo]:
        self.step_count += 1

        # transition from s_t to s_{t+1}
        curr_state, next_state = self._move_agents(actions=actions)

        # get reward for transition (s_t, a_t, s_{t+1})
        reward: float = self._reward(
            curr_state=curr_state, actions=actions, next_state=next_state
        )

        # get observation from being in s_{t+1}
        obs: Observation = self.get_obs()

        terminated: bool = self._terminated(next_state)

        # truncated is handled by a Gymnasium wrapper
        truncated: bool = False

        info: StepInfo = self._get_step_info(terminated=terminated)

        return obs, reward, terminated, truncated, info

    # agent movement
    def _move_agents(
        self, actions: list[int]
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """
        Move agents based on the chosen action and environment randomness

        Parameters
        ----------
        actions : list[int]
            Actions to take.

        Returns
        -------
        curr_state : NDArray[np.int_]
            (x, y) positions of the agents at time t (before the state transition).
        next_state : NDArray[np.int_]
            (x, y) positions of the agents at time t+1 (after the state transition).
        """
        # Randomly generate the order of the agents
        agent_indices: list[int] = list(range(self.num_agents))
        self.np_random.shuffle(agent_indices)

        # placeholder for the (x, y) positions of all the agents
        ## assumes no other state information matters for the reward calculation
        curr_state: NDArray[np.int_] = np.zeros((self.num_agents, 2))
        next_state: NDArray[np.int_] = np.zeros((self.num_agents, 2))

        for i in agent_indices:
            curr_state[i, :] = self.agents[i].pos
            self._move_agent(action=actions[i], agent=self.agents[i])

            # set next_state_list[:, i] to be the next (x, y) position of agent i
            next_state[i, :] = self.agents[i].pos

        return curr_state, next_state

    def _move_agent(self, action: int, agent: Agent) -> None:
        """
        Move agents based on the action.

        Parameters
        ----------
        action : int
            Action chosen by the agent.
        agent : Agent
            Agent to move.
        """
        ################
        # pick out the next position the agent would go to based on its action if there were no randomness in the env's transition function
        ## this logic would not be necessary if there were action masking based on avail_actions that would prevent an agent from selecting an action like "walk into the wall", but I guess that isn't standard in gymnasium envs
        chosen_next_state: Position
        assert agent.pos is not None
        chosen_next_state = agent.pos + agent.dir_to_vec[action]

        # define a flag to tell if the agent will move or not
        chosen_next_state_in_available_pos: bool = False
        available_pos: list[Position] = self._get_available_pos(agent=agent)
        for pos in available_pos:
            if np.array_equal(pos, chosen_next_state):
                chosen_next_state_in_available_pos = True
                break
            else:
                pass

        ################
        # define agent movement logic
        # agent moves to a new state
        if chosen_next_state_in_available_pos:
            action_probs: list[float] = []

            # assign probabilities to each possible action
            # assigns equal probability to all other available next positions
            # EX: if the agent chose "up", and P(s, "up", s_up) = 0.97, then
            # P(s, "right", s_right) = P(s, "down", s_down) = P(s, "left", s_left) =
            # (1 - 0.97) / 3 = 0.1
            for pos in available_pos:
                action_probs.append(
                    self.p_intended_movement
                    if np.array_equal(pos, chosen_next_state)
                    else (1 - self.p_intended_movement) / (len(available_pos) - 1)
                )

            # normalize the action probabilities
            action_probs = np.array(action_probs) / np.sum(action_probs)
            avail_pos_indices: list[int] = list(range(len(available_pos)))

            # chose the next position based on the randomly chosen action
            next_state_idx = self.np_random.choice(avail_pos_indices, p=action_probs)
            next_state = available_pos[next_state_idx]

            # make the agent move
            next_cell: WorldObj | None = self.grid.get(*next_state)
            if next_cell is None:
                agent.move(next_state, self.grid, self.init_grid, bg_color=None)
            elif next_cell.can_overlap():
                agent.move(
                    next_state, self.grid, self.init_grid, bg_color=next_cell.bg_color
                )
            else:
                raise ValueError(
                    f"Invalid action f{action} and position f{next_state} for agent {agent.index}. \
                        Available positions: {available_pos}"
                )

        # agent stays in its current state
        else:
            pass

    def _get_available_pos(self, agent: Agent) -> list[Position]:
        possible_pos: list[Position] = []

        for direction in agent.dir_to_vec:
            next_state: Position = agent.pos + direction

            if (
                next_state[0] < 0
                or next_state[1] < 0
                or next_state[0] >= self.width
                or next_state[1] >= self.height
            ):
                pass
            else:
                next_cell: WorldObj | None = self.grid.get(*next_state)

                if self.grid.get(*next_state) is None:
                    possible_pos.append(next_state)
                elif next_cell.can_overlap():
                    possible_pos.append(next_state)
                else:
                    pass

        return possible_pos

    # reward function
    def _reward(
        self,
        curr_state: NDArray[np.int_],
        actions: NDArray[np.int_],
        next_state: NDArray[np.int_],
    ) -> float:
        reward: float = 0.0

        reward += self._reward_movement(actions)

        final_state = np.array(
            self.hlmdp_config.subtask_data[self.subtask_idx].final_state
        )
        reward += self._reward_reach_goal(curr_state, next_state, final_state)
        reward += self._reward_leave_goal(curr_state, next_state, final_state)
        reward += self._reward_all_at_goal(next_state, final_state)

        return reward

    def _reward_movement(self, actions: NDArray[np.int_]) -> float:
        """Cost incurred by each action that is not "stay"
        Parameters
        ----------
        actions : NDArray[np.int_]
            agent actions

        Returns
        -------
        float
            reward
        """

        reward = 0
        reward += self.reward_config.movement_reward * np.sum(
            actions != self.actions.STAY
        )

        return reward

    def _reward_reach_goal(
        self,
        curr_state: NDArray[np.int_],
        next_state: NDArray[np.int_],
        final_state: NDArray[np.int_],
    ) -> float:
        """Reward for an agent reaching its assigned final goal state

        Parameters
        ----------
        curr_state : NDArray[np.int_]
            current state
        next_state : NDArray[np.int_]
            next state
        final_state : NDArray[np.int_]
            final assigned state for the agents for this subtask

        Returns
        -------
        float
            reward
        """
        reward = 0

        n_agents_reach_goal = 0

        for agent in self.agents:
            # agent's next state is its goal state AND
            # agent is not currently in its goal state
            if (
                np.array_equal(next_state[agent.index, :], final_state[agent.index, :])
            ) and (
                not np.array_equal(
                    curr_state[agent.index, :], final_state[agent.index, :]
                )
            ):
                # print(f"Agent {agent.index} reached its goal state")
                n_agents_reach_goal += 1

        reward += n_agents_reach_goal * self.reward_config.agent_reach_goal_reward

        return reward

    def _reward_leave_goal(
        self,
        curr_state: NDArray[np.int_],
        next_state: NDArray[np.int_],
        final_state: NDArray[np.int_],
    ) -> float:
        """Reward for an agent leaving its assigned final goal state

        Parameters
        ----------
        curr_state : NDArray[np.int_]
            current state
        next_state : NDArray[np.int_]
            next state
        final_state : NDArray[np.int_]
            final assigned state for the agents for this subtask

        Returns
        -------
        float
            reward
        """
        reward = 0

        n_agents_leave_goal = 0

        for agent in self.agents:
            # agent's current state is its goal state AND
            # agent's next state is not its goal state
            if (
                np.array_equal(curr_state[agent.index, :], final_state[agent.index, :])
            ) and (
                not np.array_equal(
                    next_state[agent.index, :], final_state[agent.index, :]
                )
            ):
                # print(f"Agent {agent.index} left its goal state")
                n_agents_leave_goal += 1

        reward += n_agents_leave_goal * self.reward_config.agent_leave_goal_reward

        return reward

    def _reward_all_at_goal(
        self, next_state: NDArray[np.int_], final_state: NDArray[np.int_]
    ) -> float:
        """Reward the team for all being at their final assigned states

        Parameters
        ----------
        next_state : NDArray[np.int_]
            next state
        final_state : NDArray[np.int_]
            final assigned state for the agents for this subtask

        Returns
        -------
        float
            reward
        """
        reward = 0

        reward += (
            np.array_equal(next_state, final_state)
            * self.reward_config.all_agents_at_goal_reward
        )

        return reward

    # termination function
    def _terminated(self, next_state: NDArray[np.int_]) -> bool:
        """
        Returns
        -------
        bool
            terminated tells whether the env is terminated or not
        """
        terminated = self._agents_reached_terminal_goal(next_state)

        return terminated

    # def _agents_detected(self) -> bool:
    #     detected: bool = False
    #     for detector in self.detectors:
    #         if detector.detect_agents(self.agents, self.obj_group_dict, self.np_random):
    #             detected = True
    #         else:
    #             pass

    #     return detected

    def _agents_reached_terminal_goal(self, next_state: NDArray[np.int_]) -> bool:
        final_state = self.hlmdp_config.subtask_data[self.subtask_idx].final_state
        return np.array_equal(next_state, final_state)

    # step info
    def _get_step_info(self, terminated=False) -> StepInfo:
        """get info to be returned in the step function"""
        step_info: StepInfo = {"success": terminated}

        return step_info
