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
