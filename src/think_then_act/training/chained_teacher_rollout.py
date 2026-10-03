"""
think_then_act.training.chained_teacher_rollout

Collects behavioral-cloning demonstrations for a SINGLE recurrent policy
spanning align_xy -> descend as one continuous episode (hidden state
carried across the handoff, not reset at a subgoal boundary) — see memory:
meta_rl_sim2real_direction. Two frozen, already-trained MLP teachers
(align_xy's and descend's own SubgoalGaussianPolicy checkpoints) provide
the supervised action label; the recurrent student sees the full widened
observation from training/singularity_force_env.py's
SingularityForceAugmentedEnv.

Needs mujoco/gymnasium (drives a live env) — integration-tested only via
`modal run`, same split as training/rollout_workers.py. The pure
safe-regime logic this depends on (training/safe_regime.py) is unit-tested
separately without mujoco.

Why this works with NO observation-schema change: align_xy and descend
already share the identical 29-dim relative-frame observation
(RELATIVE_OBS_SUBGOALS, training/subgoal_features.py) — deliberately, from
the 2026-09-01 sim2real coordinate-frame fix. So `env.set_subgoal(...)`
(already a public method on SubgoalConditionedEnv, used elsewhere to
switch a live instance between same-width subgoals) can flip the active
phase on the SAME env instance mid-episode without touching
observation_space at all — no per-phase padding/masking machinery needed.

Why the wrapped env's own terminated flag is deliberately ignored here:
SubgoalConditionedEnv.step() sets terminated=True the moment align_xy's OWN
gated done condition fires (by design, for standalone align_xy training/
eval) — but here we want to CONTINUE the same physics episode into descend,
not stop. So this module drives its own phase-switch and stopping logic
directly from `info["done"]` (align_xy's completion) and the new
safe-regime gate (descend's completion), and only falls back to the
wrapped env's truncated flag / the step budget as a hard ceiling.

Base-observation slicing: grip_contact_forces()'s discrepancy/force-onset
augmentation (SingularityForceAugmentedEnv) is a strict np.concatenate
appending EXTRA dims after the base subgoal observation — never a
transformation of it — so `augmented_obs[:BASE_DIM]` is byte-identical to
what SubgoalConditionedEnv itself would have returned unaugmented. That's
what each frozen teacher acts on; the student sees the full augmented_obs.
"""

from __future__ import annotations
import numpy as np

from think_then_act.training.subgoal_features import obs_dim_for_subgoal
from think_then_act.training.safe_regime import SafeRegimeGate

# align_xy and descend share RELATIVE_OBS_DIM — asserted, not assumed, since
# everything below depends on this staying true.
BASE_DIM = obs_dim_for_subgoal("align_xy")
assert BASE_DIM == obs_dim_for_subgoal("descend"), (
    "chained_teacher_rollout assumes align_xy and descend share one observation "
    "width (RELATIVE_OBS_SUBGOALS) — if that ever changes, this module's "
    "same-width env.set_subgoal() handoff needs rework."
)


def run_chained_align_descend_episode(
    env, align_xy_actor, descend_actor, seed: int,
    max_steps: int, d_xy_limit: float = 0.01, d_z_limit: float = 0.02,
    safe_regime_streak: int = 3,
) -> dict:
    """
    env: a SingularityForceAugmentedEnv wrapping a SubgoalConditionedEnv
    constructed with subgoal="align_xy" (perturbation should be OFF for BC
    data collection — see training/behavioral_cloning.py's module docstring
    for why: a teacher never trained under perturbation is not a valid
    label source once things are actually going wrong).
    align_xy_actor/descend_actor: frozen SubgoalGaussianPolicy instances
    (eval mode), each subgoal's already-trained, UNTOUCHED checkpoint.

    Returns a dict shaped for training/behavioral_cloning.py's
    BehavioralCloningTrainer.fit():
        {"obs": (T, obs_dim) float32, "teacher_action": (T, action_dim) float32,
         "success": bool, "n_steps": int, "switched_at_step": int | None,
         "final_d_xy": float | None, "final_d_z": float | None}
    """
    rng = np.random.default_rng(seed)
    obs, info = env.reset(rng=rng)
    env.env.set_subgoal("align_xy")   # env.env: the wrapped SubgoalConditionedEnv —
                                       # explicit, in case a caller reused an instance
                                       # left mid-descend from a previous episode.

    gate = SafeRegimeGate(streak_required=safe_regime_streak, d_xy_limit=d_xy_limit, d_z_limit=d_z_limit)
    obs_list, action_list = [], []
    active_actor = align_xy_actor
    switched_at_step = None
    success = False
    hit_step_ceiling = False
    any_force_onset_during_descend = False
    best_streak_during_descend = 0
    final_d_xy = final_d_z = None

    for step_idx in range(max_steps):
        base_obs = obs[:BASE_DIM]
        teacher_action = active_actor.act(base_obs, deterministic=True)

        obs_list.append(np.asarray(obs, dtype=np.float32))
        action_list.append(np.asarray(teacher_action, dtype=np.float32))

        obs, reward, terminated, truncated, info = env.step(teacher_action)

        if active_actor is align_xy_actor and info.get("done", False):
            env.env.set_subgoal("descend")
            active_actor = descend_actor
            gate.reset()
            switched_at_step = step_idx + 1

        if active_actor is descend_actor:
            final_d_xy = info.get("d_xy")
            final_d_z = info.get("d_z")
            if info.get("force_onset_flag", 0.0):
                any_force_onset_during_descend = True
            if gate.update(info.get("d_xy", float("inf")), info.get("d_z", float("inf")),
                            info.get("force_onset_flag", 0.0)):
                success = True
                break
            best_streak_during_descend = max(best_streak_during_descend, gate.streak)

        if truncated:
            break
    else:
        # Loop ran to max_steps without success/truncated/break firing —
        # the step-count ceiling itself is what ended this episode, distinct
        # from the wrapped env's own `truncated` flag (which fires at ITS OWN
        # max_episode_steps — normally the same number, but worth telling
        # apart when diagnosing WHY an episode failed).
        hit_step_ceiling = True

    return {
        "obs": np.stack(obs_list).astype(np.float32),
        "teacher_action": np.stack(action_list).astype(np.float32),
        "success": success,
        "n_steps": len(obs_list),
        "switched_at_step": switched_at_step,
        "hit_step_ceiling": hit_step_ceiling,
        "any_force_onset_during_descend": any_force_onset_during_descend,
        "best_streak_during_descend": best_streak_during_descend,
        "final_d_xy": float(final_d_xy) if final_d_xy is not None else None,
        "final_d_z": float(final_d_z) if final_d_z is not None else None,
    }


def run_recurrent_student_episode(
    env, student_actor, seed: int,
    max_steps: int, d_xy_limit: float = 0.01, d_z_limit: float = 0.02,
    safe_regime_streak: int = 3,
) -> dict:
    """
    Same phase-switch/stopping structure as run_chained_align_descend_episode,
    but driven by ONE trained SubgoalRecurrentPolicy instead of two frozen
    teachers, with hidden state carried continuously across the align_xy->
    descend switch (never reset at the handoff — that continuity is the
    entire point of training this as one recurrent policy rather than two).

    The phase switch itself is still driven by the ENVIRONMENT's own
    info["done"] — compute_subgoal_reward evaluates "is align_xy done" from
    the current self.subgoal + geometry alone, independent of which actor
    produced the action that got here, so the exact same switch condition
    used during teacher-driven BC collection applies unchanged at eval time.

    Use this to check whether BC actually worked end-to-end (does the
    trained student alone reach the safe regime, unassisted by either
    teacher) before deciding whether to proceed to RL fine-tuning.

    Returns the same shape as run_chained_align_descend_episode, minus
    "teacher_action" (there's no teacher here) plus "phase_at_stop" ("align_xy"
    or "descend" — did it even get through the handoff at all).
    """
    rng = np.random.default_rng(seed)
    obs, info = env.reset(rng=rng)
    env.env.set_subgoal("align_xy")

    gate = SafeRegimeGate(streak_required=safe_regime_streak, d_xy_limit=d_xy_limit, d_z_limit=d_z_limit)
    hidden_state = None
    phase = "align_xy"
    switched_at_step = None
    success = False
    final_d_xy = final_d_z = None
    n_steps = 0

    for step_idx in range(max_steps):
        action, hidden_state = student_actor.act(obs, hidden_state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        n_steps = step_idx + 1

        if phase == "align_xy" and info.get("done", False):
            env.env.set_subgoal("descend")
            phase = "descend"
            gate.reset()
            switched_at_step = step_idx + 1

        if phase == "descend":
            final_d_xy = info.get("d_xy")
            final_d_z = info.get("d_z")
            if gate.update(info.get("d_xy", float("inf")), info.get("d_z", float("inf")),
                            info.get("force_onset_flag", 0.0)):
                success = True
                break

        if truncated:
            break

    return {
        "success": success,
        "n_steps": n_steps,
        "switched_at_step": switched_at_step,
        "phase_at_stop": phase,
        "final_d_xy": float(final_d_xy) if final_d_xy is not None else None,
        "final_d_z": float(final_d_z) if final_d_z is not None else None,
    }


def compute_episode_weight(episode: dict, weight_dim: int = 1, floor: float = 0.05) -> float:
    """
    Raw (un-normalized) importance weight for one successful demonstration,
    proportional to how large a correction its align_xy phase needed along
    `weight_dim` (default: action index 1, the lateral/dim-1 axis a bias
    check found systematically under-represented among successful chained
    episodes — see debug_success_filter_bias.py's 2026-09-18 finding:
    failures skew toward larger |dim-1| corrections than successes, 0.56
    vs. 0.36 mean, so the success filter itself thins out exactly the
    harder examples the student needs to see enough of).

    Computed as mean(|teacher_action[:align_xy_steps, weight_dim]|) — same
    metric that check used — floored at `floor` so a near-zero-correction
    episode still gets SOME weight rather than being multiplied out of the
    dataset entirely (it's still a valid, if easy, demonstration).

    collect_successful_demonstrations normalizes these to mean 1.0 across
    the whole returned dataset, so the overall loss scale (and effective
    learning rate) stays comparable to unweighted training — only the
    RELATIVE emphasis between easy and hard episodes changes.
    """
    align_xy_steps = episode["switched_at_step"] or episode["n_steps"]
    if align_xy_steps <= 0:
        return floor
    raw = float(np.mean(np.abs(episode["teacher_action"][:align_xy_steps, weight_dim])))
    return max(raw, floor)


def collect_successful_demonstrations(
    env, align_xy_actor, descend_actor, seeds: list,
    max_steps: int, d_xy_limit: float = 0.01, d_z_limit: float = 0.02,
    safe_regime_streak: int = 3, weight_dim: int = 1,
) -> dict:
    """
    Runs one chained episode per seed, keeps only the ones that reach the
    safe regime (episode-level success filtering — a failed episode as a
    whole is a bad demonstration and is dropped entirely; there's no way to
    salvage individual "good" steps from it — see module docstring). Never
    switched to descend at all (align_xy itself never finished) counts as
    a failure too.

    Each kept demonstration gets a "weight" field (see compute_episode_weight)
    — the success filter is itself biased toward easier (smaller
    correction) scenes, so harder survivors are upweighted to compensate,
    normalized so the dataset's MEAN weight is 1.0 (comparable overall loss
    scale to unweighted training). Pass weight_dim=None to disable this
    entirely (every demonstration gets weight=1.0 — old behavior).

    Returns {"demonstrations": list[dict] (only successful episodes, each
    shaped as run_chained_align_descend_episode's return value plus
    "weight"), "n_attempted": int, "n_successful": int, "success_rate": float}
    — the success_rate is itself a real diagnostic about how reliable the
    existing align_xy/descend MLPs are when chained, worth reporting
    regardless of whether BC training proceeds.
    """
    demonstrations = []
    for seed in seeds:
        episode = run_chained_align_descend_episode(
            env, align_xy_actor, descend_actor, seed, max_steps,
            d_xy_limit, d_z_limit, safe_regime_streak,
        )
        if episode["success"]:
            demonstrations.append(episode)

    if weight_dim is not None and demonstrations:
        raw_weights = [compute_episode_weight(ep, weight_dim) for ep in demonstrations]
        mean_raw = float(np.mean(raw_weights))
        for ep, raw in zip(demonstrations, raw_weights):
            ep["weight"] = raw / mean_raw if mean_raw > 0 else 1.0
    else:
        for ep in demonstrations:
            ep["weight"] = 1.0

    n_attempted = len(seeds)
    n_successful = len(demonstrations)
    return {
        "demonstrations": demonstrations,
        "n_attempted": n_attempted,
        "n_successful": n_successful,
        "success_rate": float(n_successful) / n_attempted if n_attempted else 0.0,
    }
