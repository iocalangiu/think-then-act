"""
think_then_act.hrl.skill_env

Generic Semi-MDP (options-framework) environment wrapper: turns "pick a
skill" into ONE transition for a high-level agent, running a low-level
policy to completion/timeout underneath. This is the reusable piece —
deliberately has NO import of gymnasium, MuJoCo, or anything Fetch/
subgoal-reward specific — just plain Python, duck-typed against
`env.step()`/`env.reset()`. Everything project-specific is supplied by the
CALLER as a `Skill`: a name plus three small callbacks (build the
low-level policy's input, run its reward/termination check, and the
policy object itself). See training/fetch_skills.py for this project's
adapter that builds a `dict[str, Skill]` from the existing subgoal
policies/rewards; a different project — even one not using gymnasium at
all — reuses this file as-is and writes its own equivalent adapter.

Background (Sutton, Precup & Singh, 1999, "Between MDPs and semi-MDPs"):
a "skill"/"option" is a (initiation set, policy, termination condition)
triple; executing one takes a variable number of base-env steps and
collapses to one transition at the higher level of temporal abstraction.
That's exactly what this wrapper implements — nothing here is specific to
robotics, VLMs, or this project's subgoal vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Skill:
    """
    One entry in the skill registry SkillEnv is given.

    name         : identifier the high-level agent selects by (also usable
                   as an int index into the registry — see SkillEnv.step).
    policy       : anything with `.act(obs_vector, deterministic=True) ->
                   action` (e.g. this project's SubgoalGaussianPolicy).
    build_obs    : (raw_env_obs, base_env) -> the low-level policy's input
                   vector. Takes base_env too so project-specific closures
                   can pull extra context from it (e.g. a rendered frame
                   for a collision predictor) without SkillEnv needing to
                   know that's happening.
    reward_and_done: (raw_env_obs, base_env) -> (reward: float, done: bool).
                   `done` means THIS SKILL's own goal was reached (e.g.
                   "gripper is now above the block"), independent of
                   whether the overall task is done — that's judged
                   separately, via SkillEnv's task_done_fn.
    max_steps    : hard cap on how many base-env steps one invocation of
                   this skill may run before being cut off (skill_success
                   will be False if this is hit without reward_and_done
                   ever reporting done).
    """
    name: str
    policy: Any
    build_obs: Callable[[Any, Any], Any]
    reward_and_done: Callable[[Any, Any], tuple]
    max_steps: int


def _default_task_done(obs: Any, info: dict) -> bool:
    """Standard gymnasium-robotics goal-env convention. Override via
    task_done_fn for envs that don't follow it."""
    return bool(info.get("is_success", False))


def _default_high_level_reward(
    skill_name: str, base_env_rewards: list, skill_success: bool, obs: Any, info: dict,
) -> float:
    """
    Sum of the BASE env's own native per-step rewards accumulated while
    the skill ran. Deliberately does NOT use the skill's own internal
    reward_and_done reward — that's shaped for training the skill in
    isolation and its scale/meaning varies per skill (this project's
    align_xy reward is -distance, lift's is height-above-table; summing
    those across a high-level trajectory that picks different skills isn't
    a consistent unit). The base env's native reward is the only signal
    every skill shares a common scale for, which is what makes this a
    sensible project-agnostic default. Override via high_level_reward_fn
    for anything more specific (e.g. a whole-task dense reward delta).
    """
    return float(sum(base_env_rewards))


class SkillEnv:
    """
    Wraps a base env (anything with `.reset(**kwargs) -> (obs, info)` and
    `.step(action) -> (obs, reward, terminated, truncated, info)` — the
    gymnasium step/reset signature, but duck-typed, not a gym.Wrapper
    subclass, so this has no hard gymnasium dependency) so a high-level
    agent's `step(skill)` call runs that skill's policy for up to
    `skill.max_steps` base-env steps and returns ONE (obs, reward,
    terminated, truncated, info) transition — the Semi-MDP "options"
    abstraction. The high-level agent never sees the intermediate
    low-level steps directly (they're summarized in `info` for
    logging/debugging).

    high_level_reward_fn(skill_name, base_env_rewards, skill_success, obs,
    info) -> float: defaults to summing the base env's native reward
    (see _default_high_level_reward for why). Pass your own to score
    high-level transitions differently (e.g. a whole-task reward delta).

    task_done_fn(obs, info) -> bool: decides whether the OVERALL task (not
    just this skill) is finished. Defaults to the gymnasium-robotics
    `info["is_success"]` convention.

    max_skill_calls: safety cap on the high-level episode length — without
    it, a high-level agent that never manages to finish the task (or keeps
    picking a skill that never succeeds) would run forever.
    """

    def __init__(
        self,
        env: Any,
        skills: dict,
        high_level_reward_fn: Optional[Callable] = None,
        task_done_fn: Optional[Callable] = None,
        max_skill_calls: int = 20,
    ) -> None:
        if not skills:
            raise ValueError("SkillEnv needs at least one skill")
        self.env = env
        self.skills = skills
        self.skill_names = list(skills.keys())
        self.high_level_reward_fn = high_level_reward_fn or _default_high_level_reward
        self.task_done_fn = task_done_fn or _default_task_done
        self.max_skill_calls = max_skill_calls

        self._current_obs = None
        self._skill_call_count = 0

    def _resolve_skill_name(self, skill_choice) -> str:
        if isinstance(skill_choice, str):
            if skill_choice not in self.skills:
                raise ValueError(f"Unknown skill {skill_choice!r}; must be one of {self.skill_names}")
            return skill_choice
        return self.skill_names[int(skill_choice)]

    def reset(self, **kwargs) -> tuple:
        self._skill_call_count = 0
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        return obs, info

    def step(self, skill_choice) -> tuple:
        """
        skill_choice: a skill name (str) or an index into the registry
        (int) — string names are the primary interface (e.g. a VLM's
        decoded text output), the int form exists for plain gym-API-style
        callers (e.g. a random policy over range(len(skill_names))).
        """
        name = self._resolve_skill_name(skill_choice)
        skill = self.skills[name]

        obs = self._current_obs
        base_env_rewards = []
        skill_success = False
        env_terminated = env_truncated = False
        info = {}
        steps_run = 0

        for steps_run in range(1, skill.max_steps + 1):
            obs_vec = skill.build_obs(obs, self.env)
            action = skill.policy.act(obs_vec, deterministic=True)
            obs, env_reward, env_terminated, env_truncated, info = self.env.step(action)
            base_env_rewards.append(float(env_reward))

            _, skill_done = skill.reward_and_done(obs, self.env)
            if skill_done:
                skill_success = True
                break
            if env_terminated or env_truncated:
                break

        self._current_obs = obs
        self._skill_call_count += 1

        high_level_reward = self.high_level_reward_fn(
            skill_name=name, base_env_rewards=base_env_rewards,
            skill_success=skill_success, obs=obs, info=info,
        )

        task_done = self.task_done_fn(obs, info) or env_terminated
        budget_exhausted = self._skill_call_count >= self.max_skill_calls
        high_level_truncated = bool(env_truncated or budget_exhausted)

        info = dict(info, skill=name, skill_success=skill_success, skill_steps=steps_run)
        return obs, high_level_reward, bool(task_done), high_level_truncated, info

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
