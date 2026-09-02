"""
think_then_act.training.subgoal_env

Gym wrapper for the low-level, subgoal-conditioned controller (see memory:
hierarchical_architecture.md). Wraps an ObservationHarness-wrapped
FetchPickAndPlace-v3 env; the active subgoal is fixed per instance
(set_subgoal() changes it between episodes — SkillEnv, not yet built, is
what will actually switch subgoals mid-training-run).

Needs mujoco/gymnasium — not installed locally, so this module (like
env/wrapper.py) is integration-tested only, via `modal run tests/run_integration.py`.
The observation-assembly and reward math it calls into
(training/subgoal_features.py, reward/subgoal_reward.py) are pure functions
with their own local unit tests.

Design note — no HER here: subgoal_reward.py's rewards are already DENSE
(smooth per-step signal), not sparse. HER specifically compensates for
sparse/hard-exploration reward by relabeling failed trajectories against
whatever goal they actually reached; it doesn't add much on top of an
already-dense, well-shaped reward, and forcing every subgoal (including
close_gripper/release, which are about finger aperture, not a 3D point)
into one shared HER-relabelable goal space would need awkward encoding
tricks. HER remains a valid thing to add later if dense reward alone turns
out to be insufficient.

Trained by training/low_level_grpo.py — a from-scratch GRPO trainer (same
group-relative-advantage algorithm as training/grpo_trainer.py, the VLM's
trainer), not stable-baselines3 (tried and removed — see bugs_and_fixes
memory, 2026-07-11: SB3's torch/gymnasium requirements repeatedly conflicted
with this project's pins).
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from think_then_act.env.block_randomization import get_block_dims, get_perceived_block_dims
from think_then_act.env.setup import init_episode_before_subgoal, grip_contact_forces
from think_then_act.perception.block_pose_predictor import BlockPosePredictor
from think_then_act.perception.collision_predictor import CollisionPredictor
from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS, DEFAULT_WEIGHTS, compute_subgoal_reward
from think_then_act.training.subgoal_features import (
    build_subgoal_observation, obs_dim_for_subgoal, sanitize_observation_for_perception,
)


class SubgoalConditionedEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        subgoal: str,
        collision_model: CollisionPredictor = None,
        pose_model: BlockPosePredictor = None,
        align_xy_policy=None,   # optional trained align_xy actor — only used when
                                 # subgoal="descend", to start each episode from
                                 # wherever align_xy actually leaves the arm instead
                                 # of a fresh scattered reset. See env/setup.py's
                                 # init_episode_before_subgoal/_run_align_xy_until_done.
        max_episode_steps: int = 30,
        randomize_each_episode: bool = True,
        randomize_block_size: bool = False,   # opt-in — see env/setup.py's
                                 # init_random_episode docstring. Defaults
                                 # False so every EXISTING training run
                                 # (every subgoal) stays byte-for-byte
                                 # unaffected; only close_gripper's retrain
                                 # opts in for now (hierarchical_architecture
                                 # memory, Stage 4).
        size_range: tuple = None,   # optional (min, max) curriculum override —
                                 # forwarded to init_random_episode's own
                                 # size_range param (see its docstring).
                                 # None (default) = block_randomization.py's
                                 # own full [0.01, 0.08] range. Ignored
                                 # unless randomize_block_size is also True.
        width_range: tuple = None,   # optional PER-AXIS overrides (take
        length_range: tuple = None,  # precedence over size_range for that
        height_range: tuple = None,  # one axis) — see init_random_episode's
                                 # docstring. e.g. pinning width_range/
                                 # length_range to (0.05,0.05) while giving
                                 # height_range=(0.07,0.07) records/evaluates
                                 # a non-cube shape instead of a uniform cube.
        close_gripper_done_streak: int = 3,   # consecutive steps close_gripper's
                                 # OWN done condition must hold before the
                                 # episode actually terminates — found
                                 # 2026-08-13 via verify_close_gripper_grasp.py's
                                 # lift-test (1/10 seeds): a single MuJoCo
                                 # contact-solver force SPIKE (a momentary
                                 # settling transient, not a real sustained
                                 # grip) can satisfy closedness/d_xy/d_z for
                                 # exactly one step, firing `done` even
                                 # though the block was never actually held
                                 # (confirmed via video: gripper rose 25cm,
                                 # block never left the table). Same class of
                                 # bug as the original width-based proxy's
                                 # exploits (close-on-nothing, hover-and-
                                 # claim), just a different edge case — this
                                 # requires the grip to be genuinely stable
                                 # for a few consecutive steps, not just
                                 # true at one single instant.
        align_xy_done_streak: int = 3,   # consecutive steps align_xy's d_xy
                                 # threshold must hold before the episode
                                 # actually terminates — added 2026-09-01
                                 # after a real-UR3e rollout showed the
                                 # (at the time) checkpoint oscillating in a
                                 # stable limit cycle near the target rather
                                 # than settling: since align_xy_threshold
                                 # was only ever checked for ONE instant,
                                 # nothing in training ever required the
                                 # policy to actually STOP there, only pass
                                 # through it once — same "stop AND stay,"
                                 # not just "touch," requirement as
                                 # close_gripper_done_streak, generalized to
                                 # a genuinely different failure mode (an
                                 # oscillating policy, not a contact-force
                                 # spike). See also align_xy_stillness_weight
                                 # (subgoal_reward.py) for the complementary
                                 # reward-side half of this fix.
    ) -> None:
        super().__init__(env)
        self.collision_model       = collision_model
        self.pose_model            = pose_model
        self.align_xy_policy       = align_xy_policy
        self.max_episode_steps     = max_episode_steps
        self.randomize_each_episode = randomize_each_episode
        self.randomize_block_size  = randomize_block_size
        self.size_range            = size_range
        self.width_range           = width_range
        self.length_range          = length_range
        self.height_range          = height_range
        self.close_gripper_done_streak = close_gripper_done_streak
        self.align_xy_done_streak = align_xy_done_streak
        self._done_streak = 0
        # Default RNG used when reset() isn't given an explicit override —
        # advances every call, so repeated resets get genuinely different
        # scenes rather than one fixed block/target position for the whole
        # training run.
        self._episode_rng = np.random.default_rng()
        self._step_count = 0
        self.set_subgoal(subgoal)   # also sets self.observation_space (see set_subgoal)
        self.action_space = env.action_space

    def set_subgoal(self, subgoal: str) -> None:
        if subgoal not in SUBGOAL_LABELS:
            raise ValueError(f"Unknown subgoal {subgoal!r}; must be one of {SUBGOAL_LABELS}")
        self.subgoal = subgoal
        # Keeps observation_space in sync if a caller ever switches TO/FROM
        # close_gripper on a live instance (not done anywhere today — every
        # current caller either constructs with subgoal="close_gripper"
        # directly, or switches only among the other 5, same-width
        # subgoals — but leaving this stale would be a silent shape-mismatch
        # footgun the moment that changes).
        obs_dim = obs_dim_for_subgoal(subgoal)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _collision_prob(self) -> float:
        if self.collision_model is None:
            return 0.0
        return self.collision_model.predict_proba(self.env.last_frame())

    def _grip_force(self) -> dict:
        # Unlike collision_model/pose_model, this is a direct MuJoCo contact
        # query, not a learned model — no None-means-unchanged branch needed,
        # always cheap enough to compute regardless of self.subgoal (only
        # close_gripper's reward actually uses it, same as collision_prob
        # being computed unconditionally even though only descend uses it).
        return grip_contact_forces(self.env)

    def _gate_done(self, raw_done: bool) -> bool:
        # Requires the current subgoal's own done condition to hold for N
        # CONSECUTIVE steps before treating the episode as genuinely
        # finished — generalized 2026-09-01 from the close_gripper-only
        # _gate_close_gripper_done (see close_gripper_done_streak's
        # __init__ comment: a single-step MuJoCo contact-force spike can
        # satisfy closedness/d_xy/d_z for exactly one step without a real,
        # stable grasp) to also cover align_xy (see align_xy_done_streak's
        # comment: an oscillating policy can touch align_xy_threshold for
        # one instant without ever actually stopping there). Every OTHER
        # subgoal's done still reflects compute_subgoal_reward's own raw
        # value unchanged (streak requirement of 1 == no gating).
        streak_required = {
            "close_gripper": self.close_gripper_done_streak,
            "align_xy": self.align_xy_done_streak,
        }.get(self.subgoal, 1)
        if streak_required <= 1:
            return raw_done
        self._done_streak = self._done_streak + 1 if raw_done else 0
        return self._done_streak >= streak_required

    def _reward_block_dims(self) -> tuple:
        # (block_half_height, block_width) to pass into compute_subgoal_reward
        # — (None, None) unless randomize_block_size is actually on, so
        # every non-opted-in subgoal stays on reward/subgoal_reward.py's
        # documented None-means-unchanged fallback EXACTLY (e.g.
        # release_open_threshold=0.08 flat, not derived from the default
        # cube's width, which would give a numerically DIFFERENT 0.07).
        # get_block_dims itself would happily fall back to the default
        # cube's geometry even with randomization off — the gate has to be
        # here, not there, or every subgoal would silently switch onto the
        # derived-constants path the moment this file was touched, whether
        # or not size randomization was ever requested.
        if not self.randomize_block_size:
            return None, None
        dims = get_block_dims(self.env.unwrapped.model)
        return dims["height"] / 2.0, dims["width"]

    def _perceived_block_dims(self) -> np.ndarray:
        # Ground truth + per-episode noise (see env/block_randomization.py)
        # — ONLY meaningful for close_gripper's observation (see
        # subgoal_features.build_subgoal_observation, which ignores this for
        # every other subgoal regardless), but cheap enough to always
        # compute, same as _grip_force/_true_block_dims above.
        return get_perceived_block_dims(self.env.unwrapped.model)

    def _perceived_block_pos(self, obs: dict) -> np.ndarray:
        """
        Ground truth (obs["achieved_goal"]) when no pose_model is set — same
        None-means-unchanged-behavior convention as collision_model. Reward/
        done (compute_subgoal_reward, called separately in reset()/step())
        deliberately keeps using obs["achieved_goal"] directly, NOT this —
        only the policy's observation input switches to the learned
        estimate. See perception/block_pose_predictor.py's docstring for why.
        """
        if self.pose_model is None:
            return obs["achieved_goal"]
        return self.pose_model.predict_position(self.env.last_frame())

    def _flat_obs(self, obs: dict, collision_prob: float, perceived_block_pos: np.ndarray) -> np.ndarray:
        # obs["observation"] itself bakes in the TRUE block position twice
        # more (indices 3:6, 6:9 — see sanitize_observation_for_perception's
        # docstring) — swapping only achieved_goal above would leave those
        # privileged copies fully intact. Only sanitize when pose_model is
        # actually active; when it's None, achieved_goal already IS ground
        # truth, so there's nothing to fix and this stays a no-op (byte-
        # identical behavior to before pose_model existed).
        observation = obs["observation"]
        if self.pose_model is not None:
            observation = sanitize_observation_for_perception(observation, perceived_block_pos)
        return build_subgoal_observation(
            observation, perceived_block_pos, obs["desired_goal"],
            self.subgoal, collision_prob, block_dims=self._perceived_block_dims(),
        )

    def reset(self, *, rng: np.random.Generator = None, **kwargs):
        """
        rng: optional override for the block/target randomization draw. Pass
        a freshly-seeded np.random.default_rng(state_seed) here (recreated
        with the SAME seed) to make multiple rollouts share the identical
        randomized scene — the same pattern grpo_trainer.py's VLM
        collect_rollouts uses so within-group reward variance reflects
        different sampled actions, not different starting conditions.
        Falls back to this instance's own auto-advancing RNG if omitted.
        """
        obs, info = self.env.reset(**kwargs)
        if self.randomize_each_episode:
            active_rng = rng if rng is not None else self._episode_rng
            # For close_gripper/lift/move_to_target/release (subgoals that
            # only make sense once an earlier one already succeeded), this
            # fast-forwards a scripted oracle to the matching handoff state
            # instead of always starting fresh/ungrasped — see
            # env/setup.py's init_episode_before_subgoal docstring for why
            # (2026-07-14: `lift` had a literally constant reward, and
            # `close_gripper` learned to crash into the table — both because
            # the block was never actually held). align_xy is unaffected —
            # this is exactly init_random_episode for it. descend runs the
            # trained align_xy_policy forward (when given — see __init__)
            # instead of a fresh reset, for the same reason (2026-07-24:
            # descend never trained on an already-aligned start, and froze
            # when actually deployed right after align_xy).
            obs, ok = init_episode_before_subgoal(
                self.env, active_rng, self.subgoal, align_xy_policy=self.align_xy_policy,
                randomize_block_size=self.randomize_block_size, size_range=self.size_range,
                width_range=self.width_range, length_range=self.length_range, height_range=self.height_range,
            )
            if not ok:
                obs, info = self.env.reset(**kwargs)   # fallback, matches other scripts' pattern
        self._step_count = 0
        self._done_streak = 0
        collision_prob = self._collision_prob()
        perceived_block_pos = self._perceived_block_pos(obs)
        block_half_height, block_width = self._reward_block_dims()
        # Compute the breakdown at reset too (previously only step() did) so
        # callers can read e.g. close_gripper's initial d_grip_block —
        # "how far off was the starting position" — not just its value
        # after a full episode of the policy acting. Same collision_prob-
        # key-collision guard as step()'s merge below. block_half_height/
        # block_width feed lift/release/close_gripper only when
        # randomize_block_size is on (see _reward_block_dims) — None
        # (harmless no-op, old fixed-constant behavior) otherwise.
        _, breakdown = compute_subgoal_reward(
            self.subgoal, obs["observation"], obs["achieved_goal"], obs["desired_goal"],
            collision_prob=collision_prob, grip_force=self._grip_force(),
            block_half_height=block_half_height, block_width=block_width,
        )
        breakdown["done"] = self._gate_done(breakdown["done"])
        flat_obs = self._flat_obs(obs, collision_prob, perceived_block_pos)
        info = dict(info, subgoal=self.subgoal, **breakdown)
        info["collision_prob"] = collision_prob
        # Raw positions (not just the derived d_grip_block breakdown field)
        # so callers building a per-step trajectory (e.g.
        # record_subgoal_video.py) can see exactly where the block/gripper
        # were at each step, not just the scalar distance between them.
        # block_pos stays ground truth (matches what reward/done actually
        # used above); perceived_block_pos is what the POLICY's observation
        # actually contains — the two only differ once pose_model is set.
        info["block_pos"] = obs["achieved_goal"].tolist()
        info["perceived_block_pos"] = np.asarray(perceived_block_pos, dtype=np.float32).tolist()
        info["grip_pos"]  = obs["observation"][0:3].tolist()
        info["target_pos"] = np.asarray(obs["desired_goal"], dtype=np.float32).tolist()
        return flat_obs, info

    def step(self, action):
        obs, _env_reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        collision_prob = self._collision_prob()
        perceived_block_pos = self._perceived_block_pos(obs)
        block_half_height, block_width = self._reward_block_dims()

        reward, breakdown = compute_subgoal_reward(
            self.subgoal, obs["observation"], obs["achieved_goal"], obs["desired_goal"],
            collision_prob=collision_prob, action=action, grip_force=self._grip_force(),
            block_half_height=block_half_height, block_width=block_width,
        )
        breakdown["done"] = self._gate_done(breakdown["done"])
        subgoal_done = breakdown["done"]
        timed_out    = self._step_count >= self.max_episode_steps
        # Closing the fingers physically shoves the block (contact
        # dynamics) — found 2026-07-14 via video: without this, the arm
        # can drift away afterward instead of recovering, burning the rest
        # of the episode's step budget on a rollout that's already failed.
        # Cutting it short here is a TRUNCATION, not a termination — the
        # subgoal didn't succeed, so PPO's GAE should bootstrap from the
        # critic at this point rather than treating it as a true zero-
        # continuation terminal state (see rollout_workers.py's
        # terminated-vs-truncated handling).
        drifted_too_far = (
            self.subgoal == "close_gripper"
            and breakdown.get("d_grip_block", 0.0) > DEFAULT_WEIGHTS.close_gripper_drift_limit
        )

        flat_obs = self._flat_obs(obs, collision_prob, perceived_block_pos)
        # NOT dict(info, collision_prob=..., **breakdown) — breakdown already
        # has its own "collision_prob" key for the `descend` subgoal
        # (reward_descend reports it back for logging), which would collide
        # with the explicit kwarg above and raise "multiple values for
        # keyword argument". Merge breakdown first, then set collision_prob
        # unconditionally (same value breakdown already has when present).
        info = dict(info, subgoal=self.subgoal, **breakdown)
        info["collision_prob"]      = collision_prob
        info["drifted_too_far"]     = drifted_too_far
        info["block_pos"]           = obs["achieved_goal"].tolist()
        info["perceived_block_pos"] = np.asarray(perceived_block_pos, dtype=np.float32).tolist()
        info["grip_pos"]            = obs["observation"][0:3].tolist()
        info["target_pos"]          = np.asarray(obs["desired_goal"], dtype=np.float32).tolist()

        return (
            flat_obs,
            reward,
            bool(subgoal_done or terminated),
            bool(truncated or timed_out or drifted_too_far),
            info,
        )
