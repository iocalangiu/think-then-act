"""
Integration test: SubgoalConditionedEnv against real FetchPickAndPlace-v3 +
MuJoCo physics. Needs mujoco/gymnasium_robotics — run via
`modal run tests/run_integration.py`.
"""

import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("gymnasium_robotics")
np = pytest.importorskip("numpy")

import gymnasium_robotics  # noqa: F401

from think_then_act.env.setup import setup_env
from think_then_act.env.wrapper import ObservationHarness
from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS
from think_then_act.training.subgoal_env import SubgoalConditionedEnv
from think_then_act.training.subgoal_features import SUBGOAL_OBS_DIM

pytestmark = pytest.mark.integration


def _make_base_env(max_steps=10):
    env = gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_steps)
    env = ObservationHarness(env)
    setup_env(env)
    return env


def test_reset_and_step_shapes_for_every_subgoal():
    for subgoal in SUBGOAL_LABELS:
        base = _make_base_env()
        env  = SubgoalConditionedEnv(base, subgoal=subgoal, max_episode_steps=5)
        try:
            flat_obs, info = env.reset(seed=0)

            assert flat_obs.shape == (SUBGOAL_OBS_DIM,)
            assert info["subgoal"] == subgoal
            assert 0.0 <= info["collision_prob"] <= 1.0

            flat_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert flat_obs.shape == (SUBGOAL_OBS_DIM,)
            assert np.isfinite(reward)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert "done" in info
        finally:
            env.close()


def test_reset_randomizes_block_position_every_episode():
    """
    Any training loop calling env.reset() internally (SB3's .learn() did;
    the from-scratch LowLevelGRPOTrainer's collect_rollouts does too) needs
    reset() itself to vary the scene by default, or a whole run would happen
    on one fixed block/target position.
    """
    base = _make_base_env()
    env  = SubgoalConditionedEnv(base, subgoal="align_xy", max_episode_steps=5)
    try:
        block_positions = []
        for _ in range(5):
            _, info = env.reset()
            block_positions.append(np.array(base.episode_log[-1]["achieved_goal"]))

        # At least two of five random resets should land on different XY —
        # essentially certain given the disk-sampling in init_random_episode.
        distinct = {tuple(np.round(p[:2], 3)) for p in block_positions}
        assert len(distinct) > 1
    finally:
        env.close()


def test_reset_with_same_rng_reproduces_the_same_scene():
    """
    LowLevelGRPOTrainer.collect_rollouts needs multiple rollouts in a group
    to share the identical randomized scene (same seed passed as a freshly
    constructed rng each time) so within-group reward variance reflects
    different sampled actions, not different starting conditions — this is
    the mechanism that makes that possible.
    """
    base = _make_base_env()
    env  = SubgoalConditionedEnv(base, subgoal="align_xy", max_episode_steps=5)
    try:
        env.reset(rng=np.random.default_rng(42))
        block_a = np.array(base.episode_log[-1]["achieved_goal"])

        env.reset(rng=np.random.default_rng(42))
        block_b = np.array(base.episode_log[-1]["achieved_goal"])

        env.reset(rng=np.random.default_rng(43))
        block_c = np.array(base.episode_log[-1]["achieved_goal"])

        np.testing.assert_allclose(block_a, block_b)
        assert not np.allclose(block_a[:2], block_c[:2])
    finally:
        env.close()


def test_episode_truncates_at_max_steps_if_subgoal_never_completes():
    base = _make_base_env(max_steps=50)
    # move_to_target essentially never completes in 3 random steps.
    env = SubgoalConditionedEnv(base, subgoal="move_to_target", max_episode_steps=3)
    try:
        env.reset(seed=1)

        n_steps = 0
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            n_steps += 1
            if terminated or truncated:
                break

        assert n_steps <= 3
        assert truncated or terminated
    finally:
        env.close()


def test_set_subgoal_changes_active_subgoal_and_reported_info():
    base = _make_base_env()
    env  = SubgoalConditionedEnv(base, subgoal="align_xy", max_episode_steps=5)
    try:
        env.reset(seed=2)

        env.set_subgoal("lift")
        assert env.subgoal == "lift"

        _, _, _, _, info = env.step(env.action_space.sample())
        assert info["subgoal"] == "lift"
    finally:
        env.close()
