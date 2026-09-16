from typing import Literal, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from networkx.drawing.nx_agraph import graphviz_layout
from numpy.typing import NDArray

from gym_multigrid.core.constants import COLORS


class MDPAgent:
    # simple agent class for a simple MDP
    def __init__(self, init_state: int) -> None:
        self._state = init_state
        self._prev_state = init_state

    def reset(self, init_state: int):
        self._state = init_state
        self._prev_state = init_state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def prev_state(self):
        return self._prev_state

    @prev_state.setter
    def prev_state(self, value):
        self._prev_state = value


class ProjectMDP(Env):
    # currently only hard coded for LBF env
    """Using terminology from the project scheduling literature, this MDP represents a project which consists of multiple tasks with ordering (precedence) constraints, pre-defined transitions, and state-dependent action spaces."""

    def __init__(
        self,
        num_rooms: int = 2,
        msg_budget_per_agent: list[int] | None = None,
        task_type: Literal["atomic", "composed"] = "composed",
        transitions: list[tuple[int, int]] | None = None,
        states: list[int] | None = None,
        initial_state: int = 0,
        goal_states: list[int] | None = None,
        add_post_goal_state: bool = True,
    ):
        super().__init__()

        self.agent = MDPAgent(init_state=0)
        self.tasks: list[tuple]
        self.init_state: int
        self.goal_states: set[int]
        self.fail_state: int
        self.state_space: NDArray[np.int_]
        self.successor_map: dict[tuple[int, tuple], int]
        self.msg_budget_per_agent = msg_budget_per_agent or [1]

        self._build_env(
            num_rooms=num_rooms,
            task_type=task_type,
            transitions=transitions,
            states=states,
            initial_state=initial_state,
            goal_states=goal_states,
            add_post_goal_state=add_post_goal_state,
        )

        self.task_completed: bool = False

        # stuff for MDP rendering
        self.graph: Optional[nx.Graph] = None
        # scale colors to be in [0, 1] for rendering
        self.colors = {k: v / 255 for k, v in COLORS.items()}
        self.node_colors: list
        self.edge_widths: dict = {
            "normal": 1,
            "highlight": 2.5,
        }

        # n_waypoints = n_agents is a mathematically different task from !=
        # you want to use a pandas df here for easier bookkeeping
        # self.observation_space: spaces.Discrete
        # self.action_space: spaces.Discrete

        # self.df_state: pd.DataFrame
        # df_task

    def _build_env(
        self,
        num_rooms: int,
        task_type: Literal["atomic", "composed"],
        transitions: list[tuple[int, int]] | None,
        states: list[int] | None,
        initial_state: int,
        goal_states: list[int] | None,
        add_post_goal_state: bool = True,
    ):
        """
        assume 1 set of waypoints per room
        set of atomic tasks that can advance the state in the MDP
         - clear a room of fruit
         - all agents reach a waypoint for the current room
         - composed task = "current room cleared of fruit" and "all agents reach the current room's waypoint" are True
        """

        if transitions is None:
            if (num_rooms, task_type) != (2, "composed"):
                raise NotImplementedError
            transitions = [(0, 1), (1, 2)]
            states = [0, 1, 2]
            initial_state = 0
            goal_states = [2]

        self.tasks = [
            (int(source), int(destination)) for source, destination in transitions
        ]
        graph_states = set(states or ())
        graph_states.update(state for edge in self.tasks for state in edge)
        if initial_state not in graph_states:
            raise ValueError(
                f"Initial state {initial_state} is not in the MDP state set."
            )

        self.init_state = initial_state
        # initial goal states (before optionally adding a post-goal state)
        initial_goal_states = set(goal_states or [max(graph_states)])

        # Optionally insert one additional state after the current goal state
        # and make that the MDP's goal state. This is useful for testing
        # transitions that occur after the 'legacy' goal.
        if add_post_goal_state:
            post_state = max(graph_states) + 1
            graph_states.add(post_state)
            # add transitions from each initial goal to the new post-goal state
            for g in initial_goal_states:
                self.tasks.append((g, post_state))

            self.goal_states = {post_state}
        else:
            self.goal_states = initial_goal_states
        if not self.goal_states <= graph_states:
            raise ValueError("All goal states must be present in the MDP state set.")
        self.fail_state = max(graph_states) + 1
        self.state_space = sorted(graph_states) + [self.fail_state]

        self.goal_state = next(iter(self.goal_states))

        # include "stay" task for self-transition of absorbing states
        for state in [*self.goal_states, self.fail_state]:
            self.tasks.append((state, state))

        self.observation_space = spaces.Discrete(n=len(self.state_space))

        # Action space: destination state and communication budget.
        self.n_tasks = len(self.tasks)
        self.action_space = spaces.Box(
            low=np.array([0, 0.0]),
            high=np.array([self.n_tasks - 1, 1.0]),
            dtype=np.float32,
        )

        # transition probs
        self._transition_probs: pd.DataFrame
        transition_probs: list[dict] = []

        # init probs as None until we have real data
        self.successor_map = {}
        for edge in self.tasks:
            curr_state, chosen_next_state = edge
            if curr_state == chosen_next_state:
                # add dummy actions for self-transition for goal state and fail state
                # dummy action for absorbing states always has a comms val of 0 since it isn't a real task
                action = (chosen_next_state, 0.0)

                self.successor_map[(curr_state, action)] = chosen_next_state
                transition_probs.append(
                    {
                        "state": curr_state,
                        "state_type": self._get_state_type(curr_state),
                        "action": action,
                        "next_state": chosen_next_state,
                        "prob": 1.0,
                    }
                )

            else:
                for budget in self.msg_budget_per_agent:
                    action = (chosen_next_state, budget)

                    next_states = [chosen_next_state, self.fail_state]
                    for next_state in next_states:
                        self.successor_map[(curr_state, action)] = chosen_next_state
                        transition_probs.append(
                            {
                                "state": curr_state,
                                "state_type": self._get_state_type(curr_state),
                                "action": action,
                                "next_state": next_state,
                                "prob": None,
                            }
                        )

        self._transition_probs = pd.DataFrame.from_records(transition_probs)

    def _get_state_type(self, state: int) -> Literal["goal", "fail", "normal"]:
        if state in self.goal_states:
            return "goal"
        if state == self.fail_state:
            return "fail"
        return "normal"

    def step(
        self,
        action: dict,
        task_completed: bool,
        project_failed: bool,
    ) -> tuple[int, float, bool, bool, dict]:

        # Due to how time steps work in the PYMARL runner, need to set task_completed
        # so it is seen in the state getter to populate pre_transition_data to be used to select actions
        self.task_completed = False

        if project_failed:
            next_state = self.fail_state

        elif task_completed:
            self.task_completed = True
            action_tuple = self._get_action_tuple(action)
            # take action given by the hl agent
            next_state = self.successor_map[(self.agent.state, action_tuple)]

        else:
            # task still in progress
            next_state = self.agent.state

        # update agent state
        self.agent.prev_state = self.agent.state
        self.agent.state = next_state

        # Determine reward and termination
        terminated = self.agent.state in self.goal_states
        project_failed = self.agent.state == self.fail_state

        # reward = 1.0 if terminated else (-0.01 if failed else 0.0)
        reward = 0.0

        # truncated handled by TimeLimit wrapper on this MDP
        # or the low-level env in a hierarchical setup
        truncated = False

        obs = self.state
        env_info: dict = {"project_failed": project_failed}

        return (
            obs,
            reward,
            terminated,
            truncated,
            env_info,
        )

    def _get_action_tuple(self, action: dict) -> tuple:
        chosen_next_state = action["chosen_next_state"]
        msg_budget_raw = action["comms_budget"]

        # Discretize comms value to nearest level
        # only used if sampling from the action space for development purposes
        message_budget: float = self.msg_budget_per_agent[
            np.argmin(np.abs(np.array(self.msg_budget_per_agent) - msg_budget_raw))
        ]

        action_tuple = (chosen_next_state, message_budget)
        return action_tuple

    def get_action_tuple(self, action: dict) -> tuple:
        """Return the canonical discrete representation of an HL action."""
        return self._get_action_tuple(action)

    def reset(
        self, seed: Optional[int] = None, options: dict = None
    ) -> tuple[NDArray, dict]:
        """
        Reset MDP to initial state following Gymnasium API.

        Parameters
        ----------
        seed : Optional[int]
            Random seed for reproducibility
        options : dict, optional
            Additional options for starting state, e.g. {'hl_start_state': 1}

        Returns
        -------
        tuple[int, dict]
            (observation, info) where observation is the initial MDP state
        """
        super().reset(seed=seed)
        start_state = self.init_state
        if options is not None and "hl_start_state" in options:
            start_state = int(options["hl_start_state"])

        self.agent.reset(start_state)
        self.task_completed = False

        obs: NDArray = self.state
        info: dict = {"hl_start_state": self.agent.state}

        return obs, info

    @property
    def state(self) -> NDArray:
        return np.array([self.agent.state, self.task_completed])

    @property
    def transition_probs(self):
        return self._transition_probs

    @transition_probs.setter
    def transition_probs(self, df_data: pd.DataFrame):
        # the trans agenda is here bwahaha >:D
        df_trans = self._transition_probs

        for _, row in df_data.iterrows():
            # task success rate
            df_trans.loc[
                (df_trans.state == row.hl_start_state)
                & (df_trans.action == (row.hl_task[1], row.msg_budget_per_agent))
                & (df_trans.next_state == row.hl_task[1]),
                "prob",
            ] = 1.0 - row.test_task_completed_mean

            # fail rate
            df_trans.loc[
                (df_trans.state == row.hl_start_state)
                & (df_trans.action == (row.hl_task[1], row.msg_budget_per_agent))
                & (df_trans.next_state != row.hl_task[1]),
                "prob",
            ] = row.test_task_completed_mean

    def get_env_info(self):
        """standard function to interface with EPyMARL training loop"""
        env_info = {
            "state_shape": self._get_state_size(),
            # "obs_shape": self._get_obs_size(),
            # "n_actions": len(self.actions),
            # "n_agents": len(self.agents),
        }
        return env_info

    def _get_state_size(self) -> int:
        """standard function to interface with EPyMARL training loop,
        returns the flattened size of the global state."""
        # size of agent.shape
        state_size = int(np.prod(self.state.shape))
        return state_size

    def render(
        self,
        action: Optional[dict] = None,
        img_shape: tuple[float] = (6, 3),
        dpi: int = 200,
    ):
        # make an image of the MDP using networkX to show the nodes + available edges between them
        # only set up the MDP graph once during training
        if self.graph is None:
            self.graph = nx.MultiDiGraph()

            for state in self.state_space:
                state_row = self._transition_probs.loc[
                    self._transition_probs.state == state
                ].iloc[0]

                self.graph.add_node(state, **{"state_type": state_row.state_type})

                # get all outgoing edges for this state
                df_edge = self._transition_probs.loc[
                    (self._transition_probs.state == state)
                ]

                for _, row in df_edge.iterrows():
                    edge_action = tuple(row.action)
                    if row.next_state == self.fail_state:
                        # Failure does not depend on which successor task was
                        # selected. Show one failure edge per comms budget.
                        edge_action = (self.fail_state, edge_action[1])
                        if self.graph.has_edge(
                            row.state,
                            row.next_state,
                            key=edge_action[1],
                        ):
                            continue
                    self.graph.add_edge(
                        row.state,
                        row.next_state,
                        key=edge_action[1],
                        action=edge_action,
                    )

            self.node_colors: list = []

            for node in self.graph.nodes:
                if self.graph.nodes[node]["state_type"] == "normal":
                    self.node_colors.append(self.colors["light_grey"])
                elif self.graph.nodes[node]["state_type"] == "fail":
                    self.node_colors.append(self.colors["red"])
                elif self.graph.nodes[node]["state_type"] == "goal":
                    self.node_colors.append(self.colors["yellow"])

        # add an outline to the agent's current state
        node_outline_colors: list = [self.colors["white"]] * len(self.graph.nodes)
        node_outline_widths: list = [self.edge_widths["normal"]] * len(self.graph.nodes)
        for i, node in enumerate(self.graph.nodes):
            if node == self.agent.state:
                node_outline_colors[i] = self.colors["black"]
                node_outline_widths[i] = self.edge_widths["highlight"]
                break

        # highlight chosen action
        edge_outline_colors: list = ["gray"] * len(self.graph.edges)
        edge_outline_widths: list = [self.edge_widths["normal"]] * len(self.graph.edges)
        if action is not None:
            action_tuple = self._get_action_tuple(action)
            for i, (*edge, attrs) in enumerate(self.graph.edges(keys=True, data=True)):
                is_selected_action = attrs["action"] == action_tuple
                is_selected_failure = (
                    edge[1] == self.fail_state and attrs["action"][1] == action_tuple[1]
                )
                if edge[0] == self.agent.state and (
                    is_selected_action or is_selected_failure
                ):
                    if edge[1] == self.fail_state:
                        edge_outline_colors[i] = "red"
                    else:
                        edge_outline_colors[i] = "green"
                    edge_outline_widths[i] = self.edge_widths["highlight"]

        fig, ax = plt.subplots(figsize=img_shape, dpi=dpi)

        # render the graph
        self._draw_labeled_multigraph(
            ax=ax,
            G=self.graph,
            edge_label="action",
            node_outline_colors=node_outline_colors,
            node_outline_widths=node_outline_widths,
            edge_outline_colors=edge_outline_colors,
            edge_outline_widths=edge_outline_widths,
        )

        img: NDArray = self._fig_to_array(fig)

        return img

    def _draw_labeled_multigraph(
        self,
        ax,
        G,
        edge_label: str,
        node_outline_colors: list,
        node_outline_widths: list,
        edge_outline_colors: list,
        edge_outline_widths: list,
    ):
        """Draw the MDP with routed, individually curved edges.

        The failure state is intentionally excluded from the layout graph. A
        direct edge from every task state to a common failure state otherwise
        makes Graphviz route long edges through the task graph.
        """
        layout_graph = nx.DiGraph()
        layout_graph.add_nodes_from(G.nodes)
        layout_graph.add_edges_from(
            (source, target)
            for source, target, _key, _attrs in G.edges(keys=True, data=True)
            if target != self.fail_state
        )

        # "dot" and Grankdir give nice left-to-right graphs for DAGs
        pos = graphviz_layout(
            layout_graph,
            prog="dot",
            args="-Grankdir=LR -Gnodesep=2.0 -Granksep=2.2",
        )

        # Center the failure sink inside the branching portion of this task
        # graph. The post-goal tail (4 -> 5) remains to the right.
        task_positions = np.asarray(
            [pos[node] for node in layout_graph if node != self.fail_state],
            dtype=float,
        )
        x_min, y_min = task_positions.min(axis=0)
        x_max, y_max = task_positions.max(axis=0)
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        failure_anchor_states = [state for state in (0, 1, 2, 3) if state in pos]
        if len(failure_anchor_states) < 2:
            failure_anchor_states = [
                state for state in layout_graph if state != self.fail_state
            ]
        failure_anchor_positions = np.asarray(
            [pos[state] for state in failure_anchor_states],
            dtype=float,
        )
        pos[self.fail_state] = tuple(failure_anchor_positions.mean(axis=0))

        edge_data = list(G.edges(keys=True, data=True))
        edge_groups: dict[tuple[int, int], list[tuple]] = {}
        for edge in edge_data:
            edge_groups.setdefault((edge[0], edge[1]), []).append(edge)

        edge_curvatures: dict[tuple[int, int, int], float] = {}
        for (source, target), edges in edge_groups.items():
            if source == target:
                radii = [0.35] * len(edges)
            elif target == self.fail_state:
                # Failure arcs fan symmetrically into the central sink.
                center = 0.0
                spacing = 0.24
                radii = [
                    center + spacing * (index - (len(edges) - 1) / 2)
                    for index in range(len(edges))
                ]
            elif len(edges) == 1:
                radii = [0.0]
            else:
                spacing = 0.26
                radii = [
                    spacing * (index - (len(edges) - 1) / 2)
                    for index in range(len(edges))
                ]

            for edge, radius in zip(edges, radii):
                edge_curvatures[edge[:3]] = radius

        # draw nodes + labels
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=self.node_colors,
            edgecolors=node_outline_colors,
            linewidths=node_outline_widths,
            ax=ax,
        )
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

        # Draw each edge separately so its curvature is not selected by the
        # edge key position in one shared connectionstyle list.
        for edge_index, (source, target, key, attrs) in enumerate(edge_data):
            edge = (source, target, key)
            connectionstyle = f"arc3,rad={edge_curvatures[edge]}"
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[edge],
                edge_color=edge_outline_colors[edge_index],
                width=edge_outline_widths[edge_index],
                style="dashed" if target == self.fail_state else "solid",
                connectionstyle=connectionstyle,
                arrows=True,
                arrowsize=14,
                ax=ax,
            )
            destination, budget = attrs[edge_label]
            budget_text = (
                f"{budget:g}" if isinstance(budget, (int, float)) else str(budget)
            )
            label = f"a=({destination}, {budget_text})"
            label_bbox = {
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
                "pad": 0.2,
            }
            if source == target:
                label_offset = 0.18 * y_span
                if target == self.fail_state:
                    label_offset *= -1
                ax.text(
                    pos[source][0],
                    pos[source][1] + label_offset,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                    bbox=label_bbox,
                    zorder=3,
                )
            else:
                edge_length = np.linalg.norm(
                    np.asarray(pos[target]) - np.asarray(pos[source])
                )
                label_pos = 0.36 if edge_length < 0.25 * x_span else 0.5
                nx.draw_networkx_edge_labels(
                    G,
                    pos,
                    {edge: label},
                    connectionstyle=connectionstyle,
                    label_pos=label_pos,
                    font_color="black",
                    font_size=6,
                    bbox=label_bbox,
                    ax=ax,
                )

        # image formatting
        handles = [
            Line2D(
                [0],
                [0],
                color="white",
                marker="o",
                markerfacecolor=self.colors["yellow"],
                markersize=10,
                label="Project Success",
            ),
            Line2D(
                [0],
                [0],
                color="white",
                marker="o",
                markerfacecolor=self.colors["red"],
                markersize=10,
                label="Project Failure",
            ),
            # Circle((0, 0), radius=0.05, color=self.colors["yellow"], label="Project Success"),
            # Circle((0, 0), radius=0.05, color=self.colors["red"], label="Project Failure"),
            Line2D(
                [0],
                [0],
                color="green",
                lw=self.edge_widths["highlight"],
                label="Task Success",
            ),
            Line2D(
                [0],
                [0],
                color="red",
                lw=self.edge_widths["highlight"],
                label="Task Failure",
            ),
        ]

        ax.legend(
            handles=handles,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
        ax.set_axis_off()
        ax.figure.subplots_adjust(left=0.03, right=0.76, top=0.97, bottom=0.03)

    def _fig_to_array(self, fig: Figure) -> NDArray:
        """
        Convert matplotlib figure to numpy array (faster, in-memory method).

        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure object

        Returns
        -------
        np.ndarray
            Image array with shape (height, width, 3) in RGB format
        """
        # Render figure to RGBA buffer
        fig.canvas.draw()

        # Get pixel buffer from canvas
        buf = fig.canvas.buffer_rgba()

        # Get figure dimensions
        w, h = fig.canvas.get_width_height()
        plt.close()

        # Reshape to (height, width, 4) for RGBA, then drop the alpha channel
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)

        arr = arr[:, :, :3]

        return arr
