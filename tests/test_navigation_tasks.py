from pathlib import Path

import numpy as np
import pytest
import yaml
from gym_multigrid.envs.mdp import ProjectMDP
from gym_multigrid.envs.team_navigation import TeamNavigationEnv
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


def test_delay_agents_are_uniformly_sampled_for_episode():
    env = TeamNavigationEnv(
        width=8,
        height=8,
        n_agents=4,
        n_agents_delay=2,
        agent_delay_prob=1.0,
    )

    try:
        selected_sets = set()
        for seed in range(20):
            env.reset(seed=seed)
            selected = tuple(np.flatnonzero(env._delay_candidate_agents))
            assert len(selected) == 2
            assert tuple(np.flatnonzero(env._delayed_agents)) == selected
            selected_sets.add(selected)
            env.step(np.full(4, int(env.actions.STAY), dtype=int))
            assert tuple(np.flatnonzero(env._delay_candidate_agents)) == selected

        assert len(selected_sets) > 1
        assert {agent for selected in selected_sets for agent in selected} == set(
            range(4)
        )
    finally:
        env.close()


def test_delay_states_are_sampled_independently():
    env = TeamNavigationEnv(
        width=8,
        height=8,
        n_agents=4,
        n_agents_delay=4,
        agent_delay_prob=0.5,
    )

    try:
        env.reset(seed=0)
        sampled_states = []
        for _ in range(32):
            env._sample_agent_delays()
            sampled_states.append(env._delayed_agents.copy())

        assert any(state.any() and not state.all() for state in sampled_states)
    finally:
        env.close()


def test_team_delay_probability_targets_any_candidate_delay():
    n_agents = 6
    n_candidates = 4
    team_probability = 0.1
    env = TeamNavigationEnv(
        width=8,
        height=8,
        n_agents=n_agents,
        n_agents_delay=n_candidates,
        team_delay_prob=team_probability,
    )

    try:
        any_delay_probability = 1 - (1 - env.agent_delay_prob) ** n_candidates
        assert any_delay_probability == pytest.approx(team_probability)

        env.reset(seed=0)
        assert env._delay_candidate_agents.sum() == n_candidates
    finally:
        env.close()


def test_delay_status_changes_during_episode_and_is_subset_of_team():
    env = TeamNavigationEnv(
        width=8,
        height=8,
        n_agents=3,
        n_agents_delay=1,
        team_delay_prob=0.5,
    )

    try:
        env.reset(seed=101)
        candidate = tuple(np.flatnonzero(env._delay_candidate_agents))
        assert len(candidate) == 1

        observed_states = set()
        for step_index in range(200):
            state = tuple(np.flatnonzero(env._delayed_agents))
            observed_states.add(state)
            assert len(state) <= 1
            assert set(state).issubset(candidate)
            env._sample_agent_delays()

        assert len(observed_states) >= 2
    finally:
        env.close()


def test_nonzero_team_delay_probability_requires_candidates():
    with pytest.raises(ValueError, match="team_delay_prob must be 0"):
        TeamNavigationEnv(
            width=8,
            height=8,
            n_agents=4,
            n_agents_delay=0,
            team_delay_prob=0.1,
        )


def test_team_and_individual_delay_probabilities_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either team_delay_prob"):
        TeamNavigationEnv(
            width=8,
            height=8,
            n_agents=4,
            agent_delay_prob=0.1,
            team_delay_prob=0.1,
        )


@pytest.mark.parametrize("n_agents_delay", [-1, 5])
def test_delay_agent_count_must_fit_team_size(n_agents_delay):
    with pytest.raises(ValueError, match="n_agents_delay"):
        TeamNavigationEnv(
            width=8,
            height=8,
            n_agents=4,
            n_agents_delay=n_agents_delay,
        )


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


def test_learned_hallway_catalog_only_configures_root_spawn():
    config_path = (
        Path(__file__).parents[1]
        / "gym_multigrid"
        / "envs"
        / "maps"
        / "team_navigation"
        / "navigation_hallway_learned.yaml"
    )
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)

    catalog = NavigationTaskCatalog.from_configs(
        config,
        num_agents=3,
        width=8,
        height=7,
    )

    assert catalog.task_for(0, 1).init_state_dist is not None
    assert all(
        catalog.task_for(from_state, to_state).init_state_dist is None
        for from_state, to_state in [(1, 2), (2, 3), (3, 4)]
    )


def test_two_goal_per_agent_branching_config_samples_initial_states_uniformly():
    submodule_root = Path(__file__).resolve().parents[1]
    config_path = (
        submodule_root
        / "gym_multigrid"
        / "envs"
        / "maps"
        / "team_navigation"
        / "navigation_branching_two_goals_small_hall.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    task_0_to_1 = config["tasks"][0]

    catalog = NavigationTaskCatalog.from_configs(
        config["tasks"],
        num_agents=3,
        width=8,
        height=10,
    )
    task = catalog.task_for(0, 1)
    assert len(task.goal_position_options_by_agent) == 3
    assert all(
        len(options) == 2
        for options in task.goal_position_options_by_agent
    )
    assert task.has_multiple_goal_options
    assert {
        (task_edge["from_state"], task_edge["to_state"]):
        task_edge["goal_positions"]
        for task_edge in config["tasks"]
    } == {
        (0, 1): [[[4, 1], [4, 2]], [[4, 4], [4, 5]], [[4, 7], [4, 8]]],
        (0, 2): [[[5, 1], [5, 2]], [[5, 4], [5, 5]], [[5, 7], [5, 8]]],
        (1, 3): [[[1, 1], [1, 2]], [[1, 4], [1, 5]], [[1, 7], [1, 8]]],
        (2, 3): [[[1, 1], [1, 2]], [[1, 4], [1, 5]], [[1, 7], [1, 8]]],
        (3, 4): [[[6, 1], [6, 2]], [[6, 4], [6, 5]], [[6, 7], [6, 8]]],
    }

    env = TeamNavigationEnv(
        map_name="3ga_1r_two_level_small_hall",
        n_agents=3,
        navigation_tasks=config["tasks"],
    )
    support_counts = {}
    try:
        _, _ = env.reset(
            seed=2026,
            options={
                "navigation_task_transition": [0, 1],
                "navigation_init_state_dist": task_0_to_1["init_state_dist"],
            },
        )
        for _ in range(4000):
            _, _ = env.reset(
                options={
                    "navigation_task_transition": [0, 1],
                    "navigation_init_state_dist": task_0_to_1["init_state_dist"],
                },
            )
            support = tuple(tuple(agent.pos) for agent in env.agents)
            support_counts[support] = support_counts.get(support, 0) + 1
    finally:
        env.close()

    assert len(support_counts) == 8
    proportions = [count / 4000 for count in support_counts.values()]
    assert all(0.105 <= proportion <= 0.145 for proportion in proportions)


def test_observation_feature_labels_scale_with_goal_count():
    env = TeamNavigationEnv(
        width=8,
        height=10,
        n_agents=3,
    )
    try:
        labels = env._observation_feature_labels(42)
        assert len(labels) == 42
        assert labels[:3] == [
            "agent_x",
            "agent_y",
            "delay",
        ]
        assert labels[3:9] == [
            "g0_x",
            "g0_y",
            "g1_x",
            "g1_y",
            "g2_x",
            "g2_y",
        ]
        assert env._observation_feature_source_indices(42)[:3] == [
            0,
            1,
            41,
        ]
        assert env._observation_feature_source_indices(42)[3:] == list(range(2, 41))
        assert env._format_observation_feature_value(2, 0.0, 42, "delay") == "OFF"
        assert env._format_observation_feature_value(2, 1.0, 42, "delay") == "ON"
        assert env._format_observation_feature_value(
            0, 0.5, 42, "agent_x"
        ) == str(int(round(0.5 * env._coordinate_scale[0])))
        assert env._format_observation_feature_value(
            3, 0.5, 42, "g0_x"
        ) == str(int(round(0.5 * env._coordinate_scale[0])))
    finally:
        env.close()


def test_successor_reset_uses_learned_spawn_distribution():
    catalog = NavigationTaskCatalog.from_configs(
        [
            {
                "id": "0_to_1",
                "from_state": 0,
                "to_state": 1,
                "goal_positions": [[4, 1], [4, 3], [4, 5]],
                "init_state_dist": {
                    "states": [[[1, 1], [1, 3], [1, 5]]],
                    "probs": [1.0],
                },
            },
            {
                "id": "1_to_2",
                "from_state": 1,
                "to_state": 2,
                "goal_positions": [[5, 1], [5, 3], [5, 5]],
            },
        ],
        num_agents=3,
        width=8,
        height=7,
    )
    env = TeamNavigationEnv(
        width=8,
        height=7,
        n_agents=3,
        navigation_task_catalog=catalog,
        hallway=True,
    )
    assert env.observation_space.shape == (3, 5)
    learned_spawn_dist = {
        "states": [[[4, 1], [4, 3], [4, 5]]],
        "probs": [1.0],
    }

    try:
        _, info = env.reset(
            options={
                "navigation_task_transition": [1, 2],
                "navigation_init_state_dist": learned_spawn_dist,
            }
        )
        assert info["final_state"] == (
            (4, 1),
            (4, 3),
            (4, 5),
        )

        with pytest.raises(ValueError, match="has no initial state distribution"):
            env.reset(options={"navigation_task_transition": [1, 2]})
    finally:
        env.close()


def test_one_to_one_goal_assignment_accepts_any_agent_permutation():
    goal_positions = [(2, 2), (3, 2), (4, 2)]
    catalog = NavigationTaskCatalog.from_configs(
        [
            {
                "from_state": 0,
                "to_state": 1,
                "goal_positions": goal_positions,
                "goal_assignment": "one_to_one",
                "init_state_dist": {
                    "states": [[[3, 3], [2, 3], [4, 3]]],
                    "probs": [1.0],
                },
            },
            {
                "from_state": 1,
                "to_state": 2,
                "goal_positions": [(2, 3), (3, 3), (4, 3)],
                "init_state_dist": {
                    "states": [[[3, 2], [2, 2], [4, 2]]],
                    "probs": [1.0],
                },
            },
        ],
        num_agents=3,
        width=9,
        height=9,
    )
    env = TeamNavigationEnv(
        map_name="3ga_1r_small",
        n_agents=3,
        navigation_task_catalog=catalog,
    )

    try:
        obs, _ = env.reset(options={"navigation_task_transition": [0, 1]})
        assert env.observation_space.shape == (3, 9)
        assert obs.shape == (3, 9)

        _, _, terminated, _, info = env.step(np.full(3, int(env.actions.UP), dtype=int))

        assert terminated
        assert info["task_completed"]
        assert set(info["final_state"]) == set(goal_positions)
        assert len(set(info["final_state"])) == 3
        assert info["final_state"] == ((3, 2), (2, 2), (4, 2))
    finally:
        env.close()


def test_one_to_one_successor_spawn_must_be_a_goal_permutation():
    with pytest.raises(ValueError, match="one-to-one assignments"):
        NavigationTaskCatalog.from_configs(
            [
                {
                    "from_state": 0,
                    "to_state": 1,
                    "goal_positions": [(2, 2), (3, 2), (4, 2)],
                    "goal_assignment": "one_to_one",
                },
                {
                    "from_state": 1,
                    "to_state": 2,
                    "goal_positions": [(2, 3), (3, 3), (4, 3)],
                    "init_state_dist": {
                        "states": [[[2, 2], [2, 2], [4, 2]]],
                        "probs": [1.0],
                    },
                },
            ],
            num_agents=3,
            width=9,
            height=9,
        )


def test_project_mdp_supports_branching_and_merging_transitions():
    mdp = ProjectMDP(
        msg_budget_per_agent=[0, 1],
        states=[0, 1, 2, 3],
        transitions=[(0, 1), (0, 2), (1, 3), (2, 3)],
        initial_state=0,
        goal_states=[3],
        add_post_goal_state=False,
    )

    assert (0, (1, 0)) in mdp.successor_map
    assert (0, (2, 1)) in mdp.successor_map
    assert mdp.goal_states == {3}
    assert np.array_equal(mdp.reset()[0], np.array([0, False]))
