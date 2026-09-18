"""
think_then_act.training.singularity_force_env

Additive gym.Wrapper stacking on top of an already-constructed
SubgoalConditionedEnv (training/subgoal_env.py), adding two new observation
channels for the meta-RL experiment (see memory: meta_rl_sim2real_direction,
ur3e_sim2real):

  1. Commanded-vs-achieved Cartesian displacement discrepancy (3 floats,
     signed xyz, metres) — the "did the world do what I asked" signal,
     standing in for a real UR3e singularity's symptom.
  2. Contact-force onset features (2 floats: rising-edge flag, decayed
     trace since onset) — deliberately NOT raw force magnitude, since the
     MuJoCo Fetch gripper and the real UR3e's gripper are unrelated
     geometries (confirmed mismatch); onset/timing transfers across that
     mismatch, magnitude does not.

Both channels are injected by the SAME wrapper as the perturbations that
make them non-trivial (env/action_force_perturbation.py) — sim physics
alone barely produces a commanded != achieved discrepancy on its own, so
the discrepancy feature would be near-zero noise without the paired
perturbation.

Off by default: include_discrepancy_obs/include_force_obs default True but
enable_singularity_perturbation/enable_force_perturbation default False, so
constructing this wrapper with no perturbation still trains a wider-input
policy against otherwise-unperturbed dynamics (ablation cell C in the
project's protocol); nothing existing constructs this class, so
obs_dim_for_subgoal() and every current training script are unaffected
regardless.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from think_then_act.env.setup import grip_contact_forces
from think_then_act.env.action_force_perturbation import (
    sample_singularity_perturbation, apply_singularity_perturbation,
    sample_force_perturbation, apply_force_perturbation,
)
from think_then_act.training.subgoal_features import obs_dim_for_subgoal


class SingularityForceAugmentedEnv(gym.Wrapper):
    EXTRA_DISCREPANCY_DIM = 3
    EXTRA_FORCE_DIM = 2

    def __init__(
        self,
        env,
        include_discrepancy_obs: bool = True,
        include_force_obs: bool = True,
        action_scale_m: float = 0.05,   # metres per unit action — matches the
                                         # real-robot constant confirmed at
                                         # hardware/move_arm/run_subgoal_real.py:122
                                         # (also env/oracle.py, env/setup.py's
                                         # own "moves at most 0.05m" comment).
                                         # Only used to convert the policy's
                                         # normalized (-1,1) action into metres
                                         # for the discrepancy/perturbation math
                                         # below — does NOT change how the
                                         # wrapped env itself interprets action.
        enable_singularity_perturbation: bool = False,
        enable_force_perturbation: bool = False,
        singularity_kwargs: dict | None = None,   # forwarded to sample_singularity_perturbation
        force_kwargs: dict | None = None,         # forwarded to sample_force_perturbation
        force_onset_z_threshold: float = 3.0,    # z-scored against this EPISODE's own
                                                  # no-contact baseline window, not an
                                                  # absolute Newton value — necessary
                                                  # because force_kwargs' scale/noise/
                                                  # offset are themselves randomized
                                                  # per episode, so no fixed absolute
                                                  # threshold would stay meaningful.
        force_baseline_window: int = 3,   # steps used to estimate this episode's own
                                           # no-contact force noise floor
        onset_trace_decay: float = 0.7,
    ) -> None:
        super().__init__(env)
        self.include_discrepancy_obs = include_discrepancy_obs
        self.include_force_obs = include_force_obs
        self.action_scale_m = action_scale_m
        self.enable_singularity_perturbation = enable_singularity_perturbation
        self.enable_force_perturbation = enable_force_perturbation
        self.singularity_kwargs = singularity_kwargs or {}
        self.force_kwargs = force_kwargs or {}
        self.force_onset_z_threshold = force_onset_z_threshold
        self.force_baseline_window = force_baseline_window
        self.onset_trace_decay = onset_trace_decay

        extra = (self.EXTRA_DISCREPANCY_DIM if include_discrepancy_obs else 0) \
            + (self.EXTRA_FORCE_DIM if include_force_obs else 0)
        base_dim = env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_dim + extra,), dtype=np.float32
        )

        self._prev_grip_pos = None
        self._step_idx = 0
        self._force_baseline_samples: list = []
        self._onset_step = None

    @staticmethod
    def obs_dim_for(subgoal: str, include_discrepancy_obs: bool = True, include_force_obs: bool = True) -> int:
        """
        obs_dim_for_subgoal(subgoal) + this wrapper's extra dims — for
        constructing a policy/critic network sized to match this wrapper's
        widened observation_space without hardcoding the extra-dim count in
        more than one place. Does not touch obs_dim_for_subgoal itself.
        """
        extra = (SingularityForceAugmentedEnv.EXTRA_DISCREPANCY_DIM if include_discrepancy_obs else 0) \
            + (SingularityForceAugmentedEnv.EXTRA_FORCE_DIM if include_force_obs else 0)
        return obs_dim_for_subgoal(subgoal) + extra

    def obs_dim(self) -> int:
        return self.observation_space.shape[0]

    def reset(self, *, rng: np.random.Generator = None, **kwargs):
        flat_obs, info = self.env.reset(rng=rng, **kwargs)
        model = self.env.unwrapped.model

        # SubgoalConditionedEnv.reset() resolves rng (the explicit override
        # if given, else its own auto-advancing _episode_rng) into
        # self.env._noise_rng before returning — read that BACK rather than
        # re-deriving which rng "would have" been used, so this wrapper's
        # perturbation draws come from the exact same generator instance
        # SubgoalConditionedEnv itself just used for this episode's own
        # randomization. Deliberately reaches into a private attribute of
        # the wrapped env; if subgoal_env.py ever renames/removes
        # _noise_rng, this needs updating alongside it.
        active_rng = self.env._noise_rng
        sample_singularity_perturbation(
            model, active_rng, enable=self.enable_singularity_perturbation, **self.singularity_kwargs
        )
        sample_force_perturbation(
            model, active_rng, enable=self.enable_force_perturbation, **self.force_kwargs
        )

        self._prev_grip_pos = np.array(info["grip_pos"], dtype=np.float64)
        self._step_idx = 0
        self._force_baseline_samples = []
        self._onset_step = None

        onset_flag, onset_trace = self._force_onset_features({"left": 0.0, "right": 0.0})
        augmented = self._augment(flat_obs, discrepancy=np.zeros(3), onset_flag=onset_flag, onset_trace=onset_trace)
        return augmented, info

    def step(self, action):
        model = self.env.unwrapped.model
        commanded_delta_m = np.asarray(action[:3], dtype=np.float64) * self.action_scale_m
        perturbed_delta_m = apply_singularity_perturbation(model, self._step_idx, commanded_delta_m)

        perturbed_action = np.array(action, dtype=np.float32, copy=True)
        perturbed_action[:3] = (perturbed_delta_m / self.action_scale_m).astype(np.float32)

        flat_obs, reward, terminated, truncated, info = self.env.step(perturbed_action)
        self._step_idx += 1

        achieved_grip_pos = np.array(info["grip_pos"], dtype=np.float64)
        achieved_delta_m = achieved_grip_pos - self._prev_grip_pos
        self._prev_grip_pos = achieved_grip_pos
        # ORIGINAL commanded delta, not the perturbed one — matches what a
        # real controller always knows it asked for vs. what it actually got;
        # it never sees its own perturbation, only the outcome.
        discrepancy = commanded_delta_m - achieved_delta_m

        raw_forces = grip_contact_forces(self.env)
        perturbed_forces = apply_force_perturbation(model, None, raw_forces)
        onset_flag, onset_trace = self._force_onset_features(perturbed_forces)

        info = dict(info)
        info["discrepancy_xyz"] = discrepancy.tolist()
        info["raw_grip_force"] = raw_forces
        info["perturbed_grip_force"] = perturbed_forces
        info["force_onset_flag"] = onset_flag

        augmented = self._augment(flat_obs, discrepancy, onset_flag, onset_trace)
        return augmented, reward, terminated, truncated, info

    def _augment(self, flat_obs, discrepancy: np.ndarray, onset_flag: float, onset_trace: float) -> np.ndarray:
        """
        Assembles the final observation from ALREADY-COMPUTED discrepancy/
        onset features — never calls _force_onset_features itself, since
        that function advances per-episode state (baseline window, onset
        latch) and must be called exactly once per step (reset()/step()
        each call it themselves, once, so they can also surface onset_flag
        in `info` without a second, state-mutating call).
        """
        parts = [np.asarray(flat_obs, dtype=np.float32)]
        if self.include_discrepancy_obs:
            parts.append(np.asarray(discrepancy, dtype=np.float32))
        if self.include_force_obs:
            parts.append(np.array([onset_flag, onset_trace], dtype=np.float32))
        return np.concatenate(parts)

    def _force_onset_features(self, raw_forces: dict) -> tuple:
        total = raw_forces["left"] + raw_forces["right"]
        if len(self._force_baseline_samples) < self.force_baseline_window:
            self._force_baseline_samples.append(total)
            return 0.0, 0.0

        baseline_mean = float(np.mean(self._force_baseline_samples))
        baseline_std = float(np.std(self._force_baseline_samples)) + 1e-6
        z = (total - baseline_mean) / baseline_std

        onset_flag = 0.0
        if z >= self.force_onset_z_threshold and self._onset_step is None:
            self._onset_step = self._step_idx
            onset_flag = 1.0

        onset_trace = 0.0
        if self._onset_step is not None:
            onset_trace = self.onset_trace_decay ** max(0, self._step_idx - self._onset_step)

        return onset_flag, onset_trace
