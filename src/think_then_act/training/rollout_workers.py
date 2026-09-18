"""
think_then_act.training.rollout_workers

Parallel episode collection for training/low_level_ppo.py. Needs
mujoco/gymnasium in every worker process — not installed locally, so, like
subgoal_env.py and low_level_grpo.py's collect_rollouts, this module is
integration-tested only (via `modal run`), never imported at module scope
by anything that needs to run without a live env.

Why a process pool, and why persistent across iterations: observed
2026-07-14, collect_s dominates >99% of a GRPO training iteration's wall
time (~55s collection vs ~0.1s gradient update) — the rollouts are fully
independent MuJoCo episodes, embarrassingly parallel, and the update step
is cheap enough that there's nothing to gain from true async
actor-learner overlap (nothing to hide collection behind). make_pool is
called ONCE per subgoal (low_level_ppo.py's _ensure_pool), not once per
iteration: each worker's persistent env pays MuJoCo model load + osmesa GL
context creation ONE time, not once per training iteration — paying that
cost every iteration would eat most of the parallelism's benefit. Spawn
(not fork) context deliberately: MuJoCo's C bindings and the osmesa GL
context are not fork-safe.
"""

from __future__ import annotations

import numpy as np

_WORKER_ENV        = None
_WORKER_MAX_STEPS  = None


def _worker_init(env_kwargs: dict) -> None:
    """multiprocessing initializer — runs once per worker process (or once
    in the current process, for the serial no-pool path)."""
    global _WORKER_ENV, _WORKER_MAX_STEPS
    import os
    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import torch

    from think_then_act.env.setup import setup_env
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.perception.block_pose_predictor import BlockPosePredictor
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import obs_dim_for_subgoal

    collision_model = None
    ckpt = env_kwargs.get("collision_ckpt")
    if ckpt:
        collision_model = CollisionPredictor()
        collision_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        collision_model.eval()

    # Same never-pickle-the-live-module treatment as collision_model above —
    # each worker loads its own copy from the checkpoint PATH.
    pose_model = None
    pose_ckpt = env_kwargs.get("pose_ckpt")
    if pose_ckpt:
        pose_model = BlockPosePredictor()
        pose_model.load_state_dict(torch.load(pose_ckpt, map_location="cpu"))
        pose_model.eval()

    # Only relevant for subgoal="descend" (env/setup.py's
    # init_episode_before_subgoal ignores it otherwise) — only load it for
    # descend's own worker pool, not the other five subgoals' pools.
    align_xy_policy = None
    align_xy_ckpt = env_kwargs.get("align_xy_ckpt")
    if align_xy_ckpt and env_kwargs.get("subgoal") == "descend":
        align_xy_policy = SubgoalGaussianPolicy(obs_dim=obs_dim_for_subgoal("align_xy"))
        align_xy_ckpt_data = torch.load(align_xy_ckpt, map_location="cpu")
        align_xy_policy.load_state_dict(
            align_xy_ckpt_data["actor"] if isinstance(align_xy_ckpt_data, dict) and "actor" in align_xy_ckpt_data
            else align_xy_ckpt_data
        )
        align_xy_policy.eval()

    max_episode_steps = env_kwargs["max_episode_steps"]
    # +250, not *2: env/setup.py's init_episode_before_subgoal can spend up
    # to max_setup_steps=200 running the scripted oracle (for close_gripper/
    # lift/move_to_target/release's pre-subgoal setup) BEFORE the actual
    # max_episode_steps-step episode even starts — the raw env's own
    # TimeLimit has to have headroom for both, on top of the same one
    # underlying env instance, or the setup phase gets silently truncated
    # and falls back to the old (broken) fresh/ungrasped reset.
    base = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                  max_episode_steps=max_episode_steps + 250)
    )
    setup_env(base)
    env_extra_kwargs = {}
    if "done_streak" in env_kwargs:
        env_extra_kwargs["close_gripper_done_streak"] = env_kwargs["done_streak"]
    env = SubgoalConditionedEnv(
        base, subgoal=env_kwargs["subgoal"], collision_model=collision_model,
        pose_model=pose_model, align_xy_policy=align_xy_policy, max_episode_steps=max_episode_steps,
        randomize_block_size=env_kwargs.get("randomize_block_size", False),
        size_range=env_kwargs.get("size_range"),
        pos_noise_std=env_kwargs.get("pos_noise_std", 0.0),
        **env_extra_kwargs,
    )
    _WORKER_ENV       = env
    _WORKER_MAX_STEPS = max_episode_steps


def _build_models(actor_state, critic_state, obs_dim, action_dim, hidden_dim):
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy, SubgoalValueNetwork

    actor = SubgoalGaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    actor.load_state_dict(actor_state)
    actor.eval()
    critic = SubgoalValueNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim)
    critic.load_state_dict(critic_state)
    critic.eval()
    return actor, critic


def _run_episode(actor, critic, seed: int) -> dict:
    """
    Runs ONE episode against the process-local persistent env (_WORKER_ENV)
    with the given (already-loaded) actor/critic. Stores old_log_prob and
    value at COLLECTION time — PPO's clipped ratio needs the FIXED
    collection-time log-prob to compare later epochs' recomputed log-prob
    against, and GAE needs the collection-time value estimates too (see
    low_level_ppo.py's compute_gae).
    """
    import torch

    env = _WORKER_ENV
    rng = np.random.default_rng(seed)
    obs, info = env.reset(rng=rng)
    # Only present in info for close_gripper (see subgoal_reward.py) — how
    # far the setup phase actually landed from the block, BEFORE the
    # policy has taken a single action. Distinct from final_d_grip_block
    # below (which reflects the policy's behavior over the whole episode).
    initial_d_grip_block = info.get("d_grip_block")
    if initial_d_grip_block is not None:
        initial_d_grip_block = float(initial_d_grip_block)
    # d_xy/d_z: present for align_xy/descend (see reward_align_xy/
    # reward_descend's breakdown) — added to check whether descend's
    # ISOLATED training reset (a plain fresh reset, per
    # env/setup.py's "align_xy/descend need no setup") actually ever
    # starts already-xy-aligned the way the real chained rollout always
    # does (descend only ever runs right after align_xy in practice).
    # If initial_d_xy is rarely small here, that's a training/deployment
    # distribution mismatch, the same class of bug already found for
    # close_gripper/lift/move_to_target/release on 2026-07-14 — descend
    # was assumed exempt since align_xy IS always first, but descend
    # itself never is.
    initial_d_xy = info.get("d_xy")
    if initial_d_xy is not None:
        initial_d_xy = float(initial_d_xy)
    initial_d_z = info.get("d_z")
    if initial_d_z is not None:
        initial_d_z = float(initial_d_z)

    steps = []
    terminated = truncated = False
    final_d_grip_block   = None   # only present in info for close_gripper (see subgoal_reward.py)
    final_closedness     = None   # ditto — how peaked/on-target the finger width ended up
    final_translation_norm = None # ditto — ||dx,dy,dz|| of the LAST action taken
    final_d_xy = None   # only present in info for align_xy/descend
    final_d_z  = None   # only present in info for descend
    with torch.no_grad():
        for _ in range(_WORKER_MAX_STEPS):
            obs_arr = np.asarray(obs, dtype=np.float32)
            obs_t = torch.from_numpy(obs_arr).unsqueeze(0)
            action_t, raw_sample_t, log_prob_t, _ = actor.sample(obs_t)
            value_t = critic(obs_t)
            action = action_t.squeeze(0).numpy()

            next_obs, reward, terminated, truncated, info = env.step(action)
            if "d_grip_block" in info:
                final_d_grip_block = float(info["d_grip_block"])
            if "closedness" in info:
                final_closedness = float(info["closedness"])
            if "translation_norm" in info:
                final_translation_norm = float(info["translation_norm"])
            if "d_xy" in info:
                final_d_xy = float(info["d_xy"])
            if "d_z" in info:
                final_d_z = float(info["d_z"])
            steps.append({
                "obs"         : obs_arr,
                "raw_sample"  : raw_sample_t.squeeze(0).numpy(),
                "old_log_prob": float(log_prob_t.item()),
                "value"       : float(value_t.item()),
                "reward"      : float(reward),
            })
            obs = next_obs
            if terminated or truncated:
                break

        if terminated:
            # Subgoal actually achieved — no real continuation to bootstrap.
            bootstrap_value = 0.0
        else:
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            bootstrap_value = float(critic(obs_t).item())

    return {
        "steps"                : steps,
        "bootstrap_value"      : bootstrap_value,
        "total_reward"         : float(sum(s["reward"] for s in steps)),
        "n_steps"              : len(steps),
        "initial_d_grip_block" : initial_d_grip_block,
        "final_d_grip_block"   : final_d_grip_block,
        "final_closedness"       : final_closedness,
        "final_translation_norm" : final_translation_norm,
        "initial_d_xy" : initial_d_xy,
        "final_d_xy"   : final_d_xy,
        "initial_d_z"  : initial_d_z,
        "final_d_z"    : final_d_z,
    }


def _collect_one(task: tuple) -> dict:
    """Pool worker entry point. Must be module-level (picklable target for
    multiprocessing's spawn context)."""
    actor_state, critic_state, obs_dim, action_dim, hidden_dim, seed = task
    actor, critic = _build_models(actor_state, critic_state, obs_dim, action_dim, hidden_dim)
    return _run_episode(actor, critic, seed)


def collect_serial(actor_state, critic_state, obs_dim, action_dim, hidden_dim,
                    seeds: list, env_kwargs: dict) -> list:
    """No pool — builds one persistent env in the current process (reused
    across calls). Used when config.n_workers <= 1."""
    if _WORKER_ENV is None:
        _worker_init(env_kwargs)
    actor, critic = _build_models(actor_state, critic_state, obs_dim, action_dim, hidden_dim)
    return [_run_episode(actor, critic, seed) for seed in seeds]


def make_pool(env_kwargs: dict, n_workers: int):
    """One-time-per-subgoal setup — see module docstring for why this is
    called once, not once per iteration."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    return ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(env_kwargs,))


def close_pool(pool) -> None:
    pool.close()
    pool.join()


def collect_with_pool(pool, actor_state, critic_state, obs_dim, action_dim, hidden_dim,
                       seeds: list) -> list:
    tasks = [
        (actor_state, critic_state, obs_dim, action_dim, hidden_dim, seed)
        for seed in seeds
    ]
    return pool.map(_collect_one, tasks)


# ----------------------------------------------------------------------
# Recurrent-policy variants, for training/low_level_ppo_recurrent.py.
# Deliberately SEPARATE globals/functions from everything above (never
# reusing _WORKER_ENV/_WORKER_MAX_STEPS or calling _worker_init/_run_episode)
# so there is zero shared-mutable-state risk even if both trainers are ever
# imported into the same process, and the existing PPO trainer's rollout
# path stays byte-for-byte unchanged. See training/low_level_ppo_recurrent.py
# and training/singularity_force_env.py for what these build on.
# ----------------------------------------------------------------------

_WORKER_ENV_RNN       = None
_WORKER_MAX_STEPS_RNN = None

# env_kwargs keys forwarded straight into SingularityForceAugmentedEnv's
# constructor by _worker_init_recurrent — every one of these defaults to
# that class's own "off/unchanged" default, so omitting all of them from
# env_kwargs entirely reproduces a plain augmented-but-unperturbed env
# (discrepancy/force obs included, no perturbation) — see that class's
# docstring for the full off-by-default reasoning.
_SINGULARITY_FORCE_ENV_KWARGS = {
    "include_discrepancy_obs", "include_force_obs", "action_scale_m",
    "enable_singularity_perturbation", "enable_force_perturbation",
    "singularity_kwargs", "force_kwargs", "force_onset_z_threshold",
    "force_baseline_window", "onset_trace_decay",
}


def _worker_init_recurrent(env_kwargs: dict) -> None:
    """
    Recurrent-trainer counterpart of _worker_init — same env construction
    (ObservationHarness -> setup_env -> SubgoalConditionedEnv, same
    collision/pose/align_xy checkpoint-path loading), with
    SingularityForceAugmentedEnv stacked on top. Duplicates _worker_init's
    loading logic rather than factoring it out, since factoring would mean
    editing _worker_init — accepted tradeoff, same as
    RecurrentLowLevelPPOTrainer standing alone rather than subclassing
    LowLevelPPOTrainer (see low_level_ppo_recurrent.py's docstring).
    """
    global _WORKER_ENV_RNN, _WORKER_MAX_STEPS_RNN
    import os
    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import torch

    from think_then_act.env.setup import setup_env
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.perception.block_pose_predictor import BlockPosePredictor
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import obs_dim_for_subgoal
    from think_then_act.training.singularity_force_env import SingularityForceAugmentedEnv

    collision_model = None
    ckpt = env_kwargs.get("collision_ckpt")
    if ckpt:
        collision_model = CollisionPredictor()
        collision_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        collision_model.eval()

    pose_model = None
    pose_ckpt = env_kwargs.get("pose_ckpt")
    if pose_ckpt:
        pose_model = BlockPosePredictor()
        pose_model.load_state_dict(torch.load(pose_ckpt, map_location="cpu"))
        pose_model.eval()

    align_xy_policy = None
    align_xy_ckpt = env_kwargs.get("align_xy_ckpt")
    if align_xy_ckpt and env_kwargs.get("subgoal") == "descend":
        align_xy_policy = SubgoalGaussianPolicy(obs_dim=obs_dim_for_subgoal("align_xy"))
        align_xy_ckpt_data = torch.load(align_xy_ckpt, map_location="cpu")
        align_xy_policy.load_state_dict(
            align_xy_ckpt_data["actor"] if isinstance(align_xy_ckpt_data, dict) and "actor" in align_xy_ckpt_data
            else align_xy_ckpt_data
        )
        align_xy_policy.eval()

    max_episode_steps = env_kwargs["max_episode_steps"]
    base = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                  max_episode_steps=max_episode_steps + 250)
    )
    setup_env(base)
    env_extra_kwargs = {}
    if "done_streak" in env_kwargs:
        env_extra_kwargs["close_gripper_done_streak"] = env_kwargs["done_streak"]
    env = SubgoalConditionedEnv(
        base, subgoal=env_kwargs["subgoal"], collision_model=collision_model,
        pose_model=pose_model, align_xy_policy=align_xy_policy, max_episode_steps=max_episode_steps,
        randomize_block_size=env_kwargs.get("randomize_block_size", False),
        size_range=env_kwargs.get("size_range"),
        pos_noise_std=env_kwargs.get("pos_noise_std", 0.0),
        **env_extra_kwargs,
    )
    sf_kwargs = {k: v for k, v in env_kwargs.items() if k in _SINGULARITY_FORCE_ENV_KWARGS}
    env = SingularityForceAugmentedEnv(env, **sf_kwargs)

    _WORKER_ENV_RNN       = env
    _WORKER_MAX_STEPS_RNN = max_episode_steps


def _build_models_recurrent(actor_state, critic_state, obs_dim, action_dim, hidden_dim, rnn_hidden_size):
    from think_then_act.policy.subgoal_recurrent_policy import SubgoalRecurrentPolicy, SubgoalRecurrentValueNetwork

    actor = SubgoalRecurrentPolicy(obs_dim=obs_dim, action_dim=action_dim,
                                    hidden_dim=hidden_dim, rnn_hidden_size=rnn_hidden_size)
    actor.load_state_dict(actor_state)
    actor.eval()
    critic = SubgoalRecurrentValueNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, rnn_hidden_size=rnn_hidden_size)
    critic.load_state_dict(critic_state)
    critic.eval()
    return actor, critic


def _run_episode_recurrent(actor, critic, seed: int) -> dict:
    """
    Same shape/rationale as _run_episode, plus: hidden state is threaded
    through actor.sample()/critic() each step and reset to None (zeros) at
    the start of every episode — whole-episode BPTT, no cross-episode
    carry (see low_level_ppo_recurrent.py's design note on this scope
    choice) — and per-step discrepancy/force-onset diagnostics from
    SingularityForceAugmentedEnv's info dict are accumulated for the
    ablation protocol's mean_discrepancy_norm / had_force_onset /
    recovery_steps metrics (see low_level_ppo_recurrent.py's ppo_step).
    """
    import torch
    from think_then_act.env.action_force_perturbation import get_singularity_perturbation_state

    env = _WORKER_ENV_RNN
    rng = np.random.default_rng(seed)
    obs, info = env.reset(rng=rng)
    initial_d_grip_block = info.get("d_grip_block")
    if initial_d_grip_block is not None:
        initial_d_grip_block = float(initial_d_grip_block)
    initial_d_xy = info.get("d_xy")
    if initial_d_xy is not None:
        initial_d_xy = float(initial_d_xy)
    initial_d_z = info.get("d_z")
    if initial_d_z is not None:
        initial_d_z = float(initial_d_z)

    steps = []
    terminated = truncated = False
    final_d_grip_block     = None
    final_closedness       = None
    final_translation_norm = None
    final_d_xy = None
    final_d_z  = None
    discrepancy_norms = []
    had_force_onset = False
    actor_h = None
    critic_h = None
    with torch.no_grad():
        for _ in range(_WORKER_MAX_STEPS_RNN):
            obs_arr = np.asarray(obs, dtype=np.float32)
            obs_t = torch.from_numpy(obs_arr).unsqueeze(0)
            action_t, raw_sample_t, log_prob_t, _, actor_h = actor.sample(obs_t, actor_h)
            value_t, critic_h = critic(obs_t, critic_h)
            action = action_t.squeeze(0).numpy()

            next_obs, reward, terminated, truncated, info = env.step(action)
            if "d_grip_block" in info:
                final_d_grip_block = float(info["d_grip_block"])
            if "closedness" in info:
                final_closedness = float(info["closedness"])
            if "translation_norm" in info:
                final_translation_norm = float(info["translation_norm"])
            if "d_xy" in info:
                final_d_xy = float(info["d_xy"])
            if "d_z" in info:
                final_d_z = float(info["d_z"])
            if "discrepancy_xyz" in info:
                discrepancy_norms.append(float(np.linalg.norm(info["discrepancy_xyz"])))
            if info.get("force_onset_flag", 0.0):
                had_force_onset = True
            steps.append({
                "obs"         : obs_arr,
                "raw_sample"  : raw_sample_t.squeeze(0).numpy(),
                "old_log_prob": float(log_prob_t.item()),
                "value"       : float(value_t.item()),
                "reward"      : float(reward),
            })
            obs = next_obs
            if terminated or truncated:
                break

        if terminated:
            bootstrap_value = 0.0
        else:
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            bootstrap_value_t, _ = critic(obs_t, critic_h)
            bootstrap_value = float(bootstrap_value_t.item())

    # Recovery-time diagnostic: steps between the sampled singularity
    # perturbation's onset and discrepancy_norms first dropping back under
    # a small threshold — None if perturbation was never enabled this
    # episode, or discrepancy never recovers within the episode. Heuristic
    # threshold (0.01m), well below action_scale_m's 0.05m default — see
    # low_level_ppo_recurrent.py's risk notes.
    recovery_steps = None
    singularity_state = get_singularity_perturbation_state(env.unwrapped.model)
    if singularity_state is not None and singularity_state.get("enable", False) and discrepancy_norms:
        onset_step = singularity_state["onset_step"]
        recovery_threshold_m = 0.01
        for t in range(onset_step, len(discrepancy_norms)):
            if discrepancy_norms[t] < recovery_threshold_m:
                recovery_steps = t - onset_step
                break

    return {
        "steps"                : steps,
        "bootstrap_value"      : bootstrap_value,
        "total_reward"         : float(sum(s["reward"] for s in steps)),
        "n_steps"              : len(steps),
        "initial_d_grip_block" : initial_d_grip_block,
        "final_d_grip_block"   : final_d_grip_block,
        "final_closedness"       : final_closedness,
        "final_translation_norm" : final_translation_norm,
        "initial_d_xy" : initial_d_xy,
        "final_d_xy"   : final_d_xy,
        "initial_d_z"  : initial_d_z,
        "final_d_z"    : final_d_z,
        "mean_discrepancy_norm": float(np.mean(discrepancy_norms)) if discrepancy_norms else None,
        "had_force_onset"      : had_force_onset,
        "recovery_steps"       : recovery_steps,
    }


def _collect_one_recurrent(task: tuple) -> dict:
    """Pool worker entry point. Must be module-level (picklable target for
    multiprocessing's spawn context)."""
    actor_state, critic_state, obs_dim, action_dim, hidden_dim, rnn_hidden_size, seed = task
    actor, critic = _build_models_recurrent(actor_state, critic_state, obs_dim, action_dim, hidden_dim, rnn_hidden_size)
    return _run_episode_recurrent(actor, critic, seed)


def collect_serial_recurrent(actor_state, critic_state, obs_dim, action_dim, hidden_dim, rnn_hidden_size,
                              seeds: list, env_kwargs: dict) -> list:
    """No pool — builds one persistent env in the current process (reused
    across calls). Used when config.n_workers <= 1."""
    if _WORKER_ENV_RNN is None:
        _worker_init_recurrent(env_kwargs)
    actor, critic = _build_models_recurrent(actor_state, critic_state, obs_dim, action_dim, hidden_dim, rnn_hidden_size)
    return [_run_episode_recurrent(actor, critic, seed) for seed in seeds]


def make_pool_recurrent(env_kwargs: dict, n_workers: int):
    """One-time-per-subgoal setup, recurrent-trainer counterpart of
    make_pool — same spawn-context rationale (MuJoCo/osmesa are not
    fork-safe)."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    return ctx.Pool(processes=n_workers, initializer=_worker_init_recurrent, initargs=(env_kwargs,))


def collect_with_pool_recurrent(pool, actor_state, critic_state, obs_dim, action_dim, hidden_dim, rnn_hidden_size,
                                 seeds: list) -> list:
    tasks = [
        (actor_state, critic_state, obs_dim, action_dim, hidden_dim, rnn_hidden_size, seed)
        for seed in seeds
    ]
    return pool.map(_collect_one_recurrent, tasks)
