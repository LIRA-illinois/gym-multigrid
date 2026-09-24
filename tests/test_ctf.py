import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gym_multigrid.envs.ctf import Ctf1v1Env, CtfMvNEnv
from gym_multigrid.utils.map import load_text_map


def test_ctf_pos_map() -> None:
    map_path: str = "tests/assets/board_wall.txt"

    env = Ctf1v1Env(
        map_path=map_path, render_mode="human", observation_option="pos_map"
    )
    obs, _ = env.reset()
    frames = [env.render()]

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break

    os.makedirs("tests/out/animations", exist_ok=True)
    imageio.mimsave("tests/out/animations/ctf_pos_map.gif", frames, duration=0.5)
    assert os.path.exists("tests/out/animations/ctf_pos_map.gif")


def test_ctf_pos_map_flattened() -> None:
    map_path: str = "tests/assets/board_wall.txt"

    env = Ctf1v1Env(
        map_path=map_path, render_mode="human", observation_option="pos_map_flattened"
    )
    obs, _ = env.reset()
    frames = [env.render()]

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())

    os.makedirs("tests/out/animations", exist_ok=True)
    imageio.mimsave(
        "tests/out/animations/ctf_pos_map_flattened.gif", frames, duration=0.5
    )
    assert os.path.exists("tests/out/animations/ctf_pos_map_flattened.gif")


@pytest.mark.parametrize(
    "enemy_policy, num_blue_agents, num_red_agents",
    [
        ("fight", 2, 1),
        ("roomba", 2, 1),
        ("fight", 1, 2),
        ("roomba", 1, 2),
    ],
)
def test_ctf_mvn_enemy_policies(
    enemy_policy: str, num_blue_agents: int, num_red_agents: int
) -> None:
    map_path: str = "tests/assets/board_wall.txt"

    env = CtfMvNEnv(
        map_path=map_path,
        render_mode="human",
        observation_option="tensor",
        num_blue_agents=num_blue_agents,
        num_red_agents=num_red_agents,
        enemy_policies=enemy_policy,
    )
    obs, _ = env.reset()
    frames = [env.render()]

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break

    os.makedirs("tests/out/animations", exist_ok=True)
    imageio.mimsave("tests/out/animations/ctf_tensor.gif", frames, duration=0.5)
    assert terminated or truncated


# TODO: might be good idea to include seeding test for other environments
def test_ctf_random_seeding() -> None:
    map_path: str = "tests/assets/board.txt"
    env = Ctf1v1Env(
        map_path=map_path, render_mode="human", observation_option="flattened"
    )
    env.reset(seed=1)
    array1 = env.np_random.random(10)
    env.reset(seed=1)
    array2 = env.np_random.random(10)

    np.testing.assert_allclose(array1, array2)


# MvN CtF test
def test_ctf_mvn_human() -> None:
    map_path: str = "tests/assets/board.txt"
    env = CtfMvNEnv(
        num_blue_agents=2,
        num_red_agents=2,
        map_path=map_path,
        render_mode="human",
        observation_option="flattened",
    )
    obs, _ = env.reset()
    env.render()

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()


def test_ctf_mvn_rgb() -> None:
    map_path: str = "tests/assets/board.txt"
    env = CtfMvNEnv(
        num_blue_agents=2,
        num_red_agents=2,
        map_path=map_path,
        render_mode="rgb_array",
        observation_option="flattened",
    )
    obs, _ = env.reset()
    frames = [env.render()]
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())

    os.makedirs("tests/out/animations", exist_ok=True)
    imageio.mimsave("tests/out/animations/ctf_mvn.gif", frames, duration=0.5)

    assert os.path.exists("tests/out/animations/ctf_mvn.gif")
