import os

import gymnasium as gym
import imageio
import numpy as np
import pytest
import yaml
from numpy.typing import NDArray

import gym_multigrid
from gym_multigrid.core.agent import NavigationActions


@pytest.fixture
def rooms_env():
    """Fixture for creating and cleaning up the multigrid-rooms-v0 environment."""
    env = gym.make("multigrid-rooms-v0")
    yield env
    env.close()


def test_rooms_init(rooms_env: gym.Env) -> None:
    """Test CollectGameRooms initialization"""
    assert rooms_env is not None


def test_rooms_render(rooms_env: gym.Env) -> None:
    """Test RoomsEnv render"""
    image_path = "tests/out/plots/test_rooms_render.png"
    rooms_env.reset()
    img = rooms_env.render()

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    imageio.imsave(image_path, img)
    assert os.path.exists(image_path)


def test_rooms_render_lava():
    """Test RoomsEnv render with lava"""
    env = gym.make("multigrid-rooms-v0", spawn_type=3, state_representation="tensor")
    image_path = "tests/out/plots/test_rooms_render_lava.png"
    target_obs: NDArray[np.int64] = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 1],
            [1, 0, 0, 4, 0, 0, 1, 0, 0, 4, 0, 0, 1],
            [1, 0, 5, 0, 0, 4, 0, 5, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 5, 0, 1, 0, 4, 3, 0, 0, 1],
            [1, 0, 0, 4, 0, 0, 1, 0, 0, 0, 5, 0, 1],
            [1, 1, 0, 1, 1, 1, 1, 0, 5, 0, 0, 0, 1],
            [1, 0, 5, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 4, 0, 0, 1, 5, 0, 0, 4, 0, 1],
            [1, 5, 0, 2, 0, 4, 1, 0, 4, 0, 0, 0, 1],
            [1, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 1],
            [1, 0, 4, 0, 0, 0, 1, 4, 0, 0, 0, 5, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    obs, _ = env.reset()
    img = env.render()
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    imageio.imsave(image_path, img)
    assert os.path.exists(image_path)
    assert np.array_equal(obs, target_obs)

    new_obs, _, _, _, _ = env.step(np.array(NavigationActions.STAY))
    assert np.array_equal(new_obs, obs)

    env.close()


def test_rooms_reset(rooms_env: gym.Env) -> None:
    """Test RoomsEnv reset. Should return initial observation and info."""
    obs, info = rooms_env.reset()
    assert obs is not None
    assert isinstance(info, dict)
    assert info == {
        "is_success": False,
    }


@pytest.mark.parametrize(
    ["reset_options", "action", "target_reward", "target_info"],
    [
        (
            {
                "spawn_configs": [
                    {"agent": (2, 3), "goal": {"pos": (3, 3), "reward": 1.0}}
                ]
            },
            NavigationActions.RIGHT,
            0.99,
            {"is_success": True},
        ),
        (
            {
                "spawn_configs": [
                    {"agent": (2, 3), "goal": {"pos": (3, 3), "reward": 1.0}}
                ]
            },
            NavigationActions.LEFT,
            -0.01,
            {"is_success": False},
        ),
        (
            {
                "spawn_configs": [
                    {"agent": (2, 3), "goal": {"pos": (3, 3), "reward": 1.0}}
                ]
            },
            NavigationActions.UP,
            -0.01,
            {"is_success": False},
        ),
        (
            {
                "spawn_configs": [
                    {"agent": (2, 3), "goal": {"pos": (3, 3), "reward": 1.0}}
                ]
            },
            NavigationActions.DOWN,
            -0.01,
            {"is_success": False},
        ),
    ],
)
def test_rooms_step(
    rooms_env: gym.Env,
    reset_options,
    action,
    target_reward,
    target_info,
) -> None:
    """Test RoomsEnv step. Should return obs, reward, terminated, truncated, info."""
    obs, info = rooms_env.reset(options=reset_options)
    next_obs, reward, terminated, truncated, next_info = rooms_env.step(
        np.array(action)
    )
    assert next_obs is not None
    assert isinstance(reward, (int, float, np.integer, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(next_info, dict)
    assert reward == target_reward
    assert next_info == target_info


def test_rooms_episode() -> None:
    """Test RoomsEnv episode. Should complete an episode without errors."""
    animation_path = "tests/out/animations/test_rooms_episode.gif"
    rooms_env = gym.make("multigrid-rooms-v0", spawn_type=3)
    terminated: bool = False
    rooms_env.reset()
    frames = [rooms_env.render()]

    for _ in range(50):
        action = rooms_env.action_space.sample()
        obs, reward, terminated, truncated, info = rooms_env.step(action)
        frames.append(rooms_env.render())
        if terminated or truncated:
            break

    os.makedirs(os.path.dirname(animation_path), exist_ok=True)
    imageio.mimsave(animation_path, frames, duration=0.1)
    assert os.path.exists(animation_path)


def test_random_spawn() -> None:
    """Test RoomsEnv with random spawn."""
    layout_config_path: str = "tests/configs/fouroom/random.yaml"
    with open(layout_config_path, "r") as f:
        layout_config = yaml.safe_load(f)["layout_config"]
    env = gym.make(
        "multigrid-rooms-v0",
        layout_config=layout_config,
        state_representation="positional_dict",
    )
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(info, dict)
    assert info == {"is_success": False}

    image = env.render()
    image_path = "tests/out/plots/test_rooms_random_spawn.png"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    imageio.imsave(image_path, image)  # type: ignore
    assert os.path.exists(image_path)
