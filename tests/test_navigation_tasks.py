import numpy as np
import pytest
from gym_multigrid.envs.mdp import ProjectMDP
from gym_multigrid.utils.subtasks import NavigationTaskCatalog


def _task(task_id, from_state, to_state, goals, init_positions=None):
    return {
        "id": task_id,
        "from_state": from_state,
        "to_state": to_state,
        "goal_positions": goals,
        "init_state_dist": {
            "states": [
                [
                    *(init_positions or [(1, 1), (2, 1), (3, 1)]),
                ]
            ],
            "probs": [1.0],
        },
    }


def test_navigation_catalog_distinguishes_branching_transitions():
    catalog = NavigationTaskCatalog.from_configs(
        [
            _task("0_to_1", 0, 1, [(1, 2), (2, 2), (3, 2)]),
            _task(
                "1_to_2",
                1,
                2,
                [(1, 3), (2, 3), (3, 3)],
                init_positions=[(1, 2), (2, 2), (3, 2)],
            ),
        ],
        num_agents=3,
        width=5,
        height=5,
    )

    assert catalog.task_for(0, 1).task_id == "0_to_1"
    assert catalog.task_for(1, 2).task_id == "1_to_2"


def test_navigation_catalog_rejects_disconnected_successor_spawn():
    with pytest.raises(ValueError, match="must start from the predecessor"):
        NavigationTaskCatalog.from_configs(
            [
                _task("0_to_1", 0, 1, [(1, 2), (2, 2), (3, 2)]),
                _task("1_to_2", 1, 2, [(1, 4), (2, 4), (3, 4)]),
            ],
            num_agents=3,
            width=5,
            height=5,
        )


def test_project_mdp_supports_branching_and_merging_transitions():
    mdp = ProjectMDP(
        msg_budget_per_agent=[0, 1],
        states=[0, 1, 2, 3],
        transitions=[(0, 1), (0, 2), (1, 3), (2, 3)],
        initial_state=0,
        goal_states=[3],
    )

    assert (0, (1, 0)) in mdp.successor_map
    assert (0, (2, 1)) in mdp.successor_map
    assert mdp.goal_states == {3}
    assert np.array_equal(mdp.reset()[0], np.array([0, False]))
