"""
Unit tests for hrl.skill_env — a fake, gymnasium-free toy env + fake
skills. Deliberately proves this module needs no mujoco/gymnasium: unlike
most env-adjacent code in this project, SkillEnv itself is plain Python
glue and is fully exercised here without a live sim.
"""

import pytest

from think_then_act.hrl.skill_env import Skill, SkillEnv


class _FakeEnv:
    """
    1D toy env: state is a float, action is a 1-element sequence added to
    state each step. Native reward = -abs(state) (a stand-in for "some
    task-level cost"). info["is_success"] flips True once abs(state)>=5 —
    the WHOLE-TASK completion signal, independent of any one skill's own
    goal.
    """

    def __init__(self):
        self.state = 0.0
        self.step_count = 0

    def reset(self, **kwargs):
        self.state = 0.0
        self.step_count = 0
        return {"value": self.state}, {}

    def step(self, action):
        self.state += float(action[0])
        self.step_count += 1
        reward = -abs(self.state)
        terminated = False
        truncated = self.step_count >= 1000   # not hit in these tests
        info = {"is_success": abs(self.state) >= 5.0}
        return {"value": self.state}, reward, terminated, truncated, info


class _FixedActionPolicy:
    def __init__(self, action):
        self._action = action

    def act(self, obs, deterministic=True):
        return self._action


def _increment_skill(threshold=3.0, max_steps=10) -> Skill:
    return Skill(
        name="increment",
        policy=_FixedActionPolicy([1.0]),
        build_obs=lambda obs, env: obs,
        reward_and_done=lambda obs, env: (0.0, obs["value"] >= threshold),
        max_steps=max_steps,
    )


def _never_done_skill(name="increment_never_done", max_steps=6) -> Skill:
    return Skill(
        name=name,
        policy=_FixedActionPolicy([1.0]),
        build_obs=lambda obs, env: obs,
        reward_and_done=lambda obs, env: (0.0, False),
        max_steps=max_steps,
    )


def test_step_runs_skill_until_its_own_done_condition():
    env = _FakeEnv()
    sk_env = SkillEnv(env, {"increment": _increment_skill(threshold=3.0)})
    sk_env.reset()

    obs, reward, done, truncated, info = sk_env.step("increment")

    assert info["skill_success"] is True
    assert info["skill_steps"] == 3   # state goes 1 -> 2 -> 3, done at 3
    assert obs["value"] == 3.0


def test_step_stops_at_max_steps_when_skill_never_finishes():
    env = _FakeEnv()
    sk_env = SkillEnv(env, {"never": _never_done_skill(max_steps=4)})
    sk_env.reset()

    obs, reward, done, truncated, info = sk_env.step("never")

    assert info["skill_success"] is False
    assert info["skill_steps"] == 4
    assert obs["value"] == 4.0


def test_default_high_level_reward_sums_base_env_native_reward():
    env = _FakeEnv()
    sk_env = SkillEnv(env, {"increment": _increment_skill(threshold=3.0)})
    sk_env.reset()

    _, reward, _, _, _ = sk_env.step("increment")

    # state sequence 1,2,3 -> native rewards -1,-2,-3 -> sum -6
    assert reward == pytest.approx(-6.0)


def test_custom_high_level_reward_fn_overrides_default():
    env = _FakeEnv()
    sk_env = SkillEnv(
        env, {"increment": _increment_skill(threshold=3.0)},
        high_level_reward_fn=lambda **kwargs: 42.0,
    )
    sk_env.reset()

    _, reward, _, _, _ = sk_env.step("increment")
    assert reward == 42.0


def test_task_done_defaults_to_info_is_success():
    env = _FakeEnv()
    # never-done skill so the loop runs the full 6 steps (state reaches 6,
    # crossing the is_success threshold of 5 before the skill call ends).
    sk_env = SkillEnv(env, {"never": _never_done_skill(max_steps=6)})
    sk_env.reset()

    _, _, task_done, _, info = sk_env.step("never")
    assert info["is_success"] is True
    assert task_done is True


def test_custom_task_done_fn_overrides_default():
    env = _FakeEnv()
    sk_env = SkillEnv(
        env, {"never": _never_done_skill(max_steps=6)},
        task_done_fn=lambda obs, info: False,
    )
    sk_env.reset()

    _, _, task_done, _, info = sk_env.step("never")
    assert info["is_success"] is True   # default signal still true...
    assert task_done is False            # ...but the override wins


def test_max_skill_calls_truncates_high_level_episode():
    env = _FakeEnv()
    sk_env = SkillEnv(env, {"increment": _increment_skill(threshold=3.0)}, max_skill_calls=2)
    sk_env.reset()

    sk_env.step("increment")                      # call 1: state 0 -> 3
    _, _, task_done, truncated, _ = sk_env.step("increment")   # call 2: already >=3, 1 step -> state 4

    assert task_done is False    # abs(state)=4 < is_success threshold of 5
    assert truncated is True     # but the call budget (2) is exhausted


def test_step_accepts_both_string_and_index_skill_choice():
    env_a, env_b = _FakeEnv(), _FakeEnv()
    skill = _increment_skill(threshold=3.0)
    sk_env_a = SkillEnv(env_a, {"increment": skill})
    sk_env_b = SkillEnv(env_b, {"increment": skill})
    sk_env_a.reset()
    sk_env_b.reset()

    obs_a, reward_a, *_ = sk_env_a.step("increment")
    obs_b, reward_b, *_ = sk_env_b.step(0)

    assert obs_a == obs_b
    assert reward_a == reward_b


def test_unknown_skill_name_raises():
    env = SkillEnv(_FakeEnv(), {"increment": _increment_skill()})
    env.reset()
    with pytest.raises(ValueError):
        env.step("not_a_real_skill")


def test_no_skills_raises_at_construction():
    with pytest.raises(ValueError):
        SkillEnv(_FakeEnv(), {})


def test_reset_resets_skill_call_count():
    env = _FakeEnv()
    sk_env = SkillEnv(env, {"increment": _increment_skill(threshold=3.0)}, max_skill_calls=1)
    sk_env.reset()
    _, _, _, truncated, _ = sk_env.step("increment")
    assert truncated is True   # budget of 1 exhausted

    sk_env.reset()
    assert sk_env._skill_call_count == 0
