"""
Unit tests for training.fetch_skills — no mujoco/gymnasium needed, same as
subgoal_reward.py/subgoal_features.py (pure Python/numpy glue); a fake
base_env stands in for the real one.
"""

import numpy as np
import pytest

from think_then_act.training.fetch_skills import build_fetch_skills
from think_then_act.reward.subgoal_reward import compute_subgoal_reward
from think_then_act.training.subgoal_features import build_subgoal_observation


def _make_obs():
    return {
        "observation": np.arange(25, dtype=np.float32) * 0.01,
        "achieved_goal": np.array([1.0, 0.75, 0.425], dtype=np.float32),
        "desired_goal": np.array([1.1, 0.80, 0.425], dtype=np.float32),
    }


class _FakeBaseEnv:
    def __init__(self, frame):
        self._frame = frame

    def last_frame(self):
        return self._frame


class _FakePolicy:
    def act(self, obs, deterministic=True):
        return np.zeros(4, dtype=np.float32)


class _CountingCollisionModel:
    def __init__(self, prob=0.5):
        self.prob = prob
        self.n_calls = 0

    def predict_proba(self, frame):
        self.n_calls += 1
        return self.prob


class _CountingPoseModel:
    def __init__(self, pos=(9.0, 9.0, 9.0)):
        self.pos = np.array(pos, dtype=np.float32)
        self.n_calls = 0

    def predict_position(self, frame):
        self.n_calls += 1
        return self.pos


def test_build_fetch_skills_rejects_unknown_subgoal():
    with pytest.raises(ValueError):
        build_fetch_skills({"not_a_subgoal": _FakePolicy()})


def test_build_obs_matches_build_subgoal_observation_directly():
    skills = build_fetch_skills({"align_xy": _FakePolicy()}, max_steps=30)
    obs = _make_obs()
    base_env = _FakeBaseEnv(frame=np.zeros((4, 4, 3), dtype=np.uint8))

    got = skills["align_xy"].build_obs(obs, base_env)
    expected = build_subgoal_observation(
        obs["observation"], obs["achieved_goal"], obs["desired_goal"],
        "align_xy", 0.0,   # no collision_model -> 0.0
    )
    np.testing.assert_array_equal(got, expected)


def test_reward_and_done_matches_compute_subgoal_reward_directly():
    skills = build_fetch_skills({"move_to_target": _FakePolicy()})
    obs = _make_obs()
    base_env = _FakeBaseEnv(frame=np.zeros((4, 4, 3), dtype=np.uint8))

    reward, done = skills["move_to_target"].reward_and_done(obs, base_env)
    expected_reward, expected_breakdown = compute_subgoal_reward(
        "move_to_target", obs["observation"], obs["achieved_goal"], obs["desired_goal"],
    )
    assert reward == pytest.approx(expected_reward)
    assert done == expected_breakdown["done"]


def test_collision_prob_memoized_per_frame_not_per_call():
    collision_model = _CountingCollisionModel(prob=0.7)
    skills = build_fetch_skills({"descend": _FakePolicy()}, collision_model=collision_model)
    obs = _make_obs()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    base_env = _FakeBaseEnv(frame=frame)

    skills["descend"].build_obs(obs, base_env)
    skills["descend"].reward_and_done(obs, base_env)
    assert collision_model.n_calls == 1   # same frame -> memoized, not recomputed

    base_env._frame = np.ones((4, 4, 3), dtype=np.uint8)   # new frame
    skills["descend"].build_obs(obs, base_env)
    assert collision_model.n_calls == 2   # new frame -> recomputed once


def test_collision_prob_feeds_into_descend_reward():
    collision_model = _CountingCollisionModel(prob=0.8)
    skills = build_fetch_skills({"descend": _FakePolicy()}, collision_model=collision_model)
    obs = _make_obs()
    base_env = _FakeBaseEnv(frame=np.zeros((4, 4, 3), dtype=np.uint8))

    reward, _ = skills["descend"].reward_and_done(obs, base_env)
    expected_reward, _ = compute_subgoal_reward(
        "descend", obs["observation"], obs["achieved_goal"], obs["desired_goal"],
        collision_prob=0.8,
    )
    assert reward == pytest.approx(expected_reward)


def test_build_obs_uses_ground_truth_achieved_goal_when_no_pose_model():
    skills = build_fetch_skills({"align_xy": _FakePolicy()}, max_steps=30)
    obs = _make_obs()
    base_env = _FakeBaseEnv(frame=np.zeros((4, 4, 3), dtype=np.uint8))

    got = skills["align_xy"].build_obs(obs, base_env)
    expected = build_subgoal_observation(
        obs["observation"], obs["achieved_goal"], obs["desired_goal"],
        "align_xy", 0.0,
    )
    np.testing.assert_array_equal(got, expected)


def test_build_obs_uses_pose_model_estimate_when_set():
    pose_model = _CountingPoseModel(pos=(9.0, 9.0, 9.0))
    skills = build_fetch_skills({"align_xy": _FakePolicy()}, pose_model=pose_model, max_steps=30)
    obs = _make_obs()
    base_env = _FakeBaseEnv(frame=np.zeros((4, 4, 3), dtype=np.uint8))

    got = skills["align_xy"].build_obs(obs, base_env)
    expected = build_subgoal_observation(
        obs["observation"], pose_model.pos, obs["desired_goal"],
        "align_xy", 0.0,
    )
    np.testing.assert_array_equal(got, expected)
    # Deliberately NOT equal to the ground-truth-achieved_goal version —
    # confirms the pose model's (very different) estimate actually made it
    # into the observation, not silently ignored.
    ground_truth_obs = build_subgoal_observation(
        obs["observation"], obs["achieved_goal"], obs["desired_goal"],
        "align_xy", 0.0,
    )
    assert not np.array_equal(got, ground_truth_obs)


def test_reward_and_done_stays_on_ground_truth_regardless_of_pose_model():
    """The whole point of keeping reward/done privileged: a pose model
    reporting a wildly wrong position must not change success semantics."""
    pose_model = _CountingPoseModel(pos=(9.0, 9.0, 9.0))
    skills = build_fetch_skills({"move_to_target": _FakePolicy()}, pose_model=pose_model)
    obs = _make_obs()
    base_env = _FakeBaseEnv(frame=np.zeros((4, 4, 3), dtype=np.uint8))

    reward, done = skills["move_to_target"].reward_and_done(obs, base_env)
    expected_reward, expected_breakdown = compute_subgoal_reward(
        "move_to_target", obs["observation"], obs["achieved_goal"], obs["desired_goal"],
    )
    assert reward == pytest.approx(expected_reward)
    assert done == expected_breakdown["done"]


def test_pose_estimate_memoized_per_frame_not_per_call():
    pose_model = _CountingPoseModel()
    skills = build_fetch_skills({"align_xy": _FakePolicy()}, pose_model=pose_model)
    obs = _make_obs()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    base_env = _FakeBaseEnv(frame=frame)

    skills["align_xy"].build_obs(obs, base_env)
    skills["align_xy"].build_obs(obs, base_env)
    assert pose_model.n_calls == 1   # same frame -> memoized, not recomputed

    base_env._frame = np.ones((4, 4, 3), dtype=np.uint8)   # new frame
    skills["align_xy"].build_obs(obs, base_env)
    assert pose_model.n_calls == 2   # new frame -> recomputed once


def test_build_fetch_skills_returns_one_skill_per_policy_with_correct_max_steps():
    policies = {"align_xy": _FakePolicy(), "lift": _FakePolicy()}
    skills = build_fetch_skills(policies, max_steps=15)

    assert set(skills.keys()) == {"align_xy", "lift"}
    assert skills["align_xy"].max_steps == 15
    assert skills["lift"].policy is policies["lift"]
