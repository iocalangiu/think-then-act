"""
Integration test: real FetchPickAndPlace-v3 + ObservationHarness + env setup
helpers + dense reward, run against actual MuJoCo physics.

Needs mujoco/gymnasium-robotics, which aren't installed locally — run via
`modal run tests/run_integration.py` (CPU, no GPU needed for this file).
"""

import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("gymnasium_robotics")
import numpy as np

import gymnasium_robotics  # noqa: F401  — registers Fetch* envs

from think_then_act.env.setup import init_random_episode, setup_env
from think_then_act.env.wrapper import ObservationHarness
from think_then_act.reward.dense_reward import compute_dense_reward

pytestmark = pytest.mark.integration

TABLE_CENTRE = np.array([1.30, 0.75])
TABLE_RADIUS = 0.20


def _make_env(max_steps=10):
    env = gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_steps)
    env = ObservationHarness(env)
    setup_env(env)
    return env


def test_reset_captures_initial_frame_and_state():
    env = _make_env()
    try:
        env.reset(seed=0)
        assert len(env.episode_log) == 1
        entry = env.episode_log[0]
        assert entry["frame"].ndim == 3 and entry["frame"].shape[-1] == 3
        assert len(entry["observation"]) == 25
        assert len(entry["achieved_goal"]) == 3
        assert len(entry["desired_goal"]) == 3
    finally:
        env.close()


def test_init_random_episode_places_block_and_target_on_table():
    env = _make_env()
    try:
        env.reset(seed=1)
        rng = np.random.default_rng(1)
        obs, ok = init_random_episode(env, rng)

        assert ok
        achieved_xy = np.array(obs["achieved_goal"][:2])
        desired_xy = np.array(obs["desired_goal"][:2])

        assert np.linalg.norm(achieved_xy - TABLE_CENTRE) <= TABLE_RADIUS + 1e-6
        assert np.linalg.norm(desired_xy - TABLE_CENTRE) <= TABLE_RADIUS + 1e-6
        # init_random_episode retries until block/target are >0.10m apart.
        assert np.linalg.norm(achieved_xy - desired_xy) > 0.05
    finally:
        env.close()


def test_rollout_produces_finite_dense_rewards():
    env = _make_env(max_steps=5)
    try:
        env.reset(seed=2)
        for _ in range(5):
            obs, _, terminated, truncated, info = env.step(env.action_space.sample())
            reward, breakdown = compute_dense_reward(
                obs=obs["observation"],
                achieved_goal=obs["achieved_goal"],
                desired_goal=obs["desired_goal"],
                info=info,
            )
            assert np.isfinite(reward)
            assert isinstance(breakdown["is_success"], bool)
            if terminated or truncated:
                break
    finally:
        env.close()
