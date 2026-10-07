"""
diagnose_elevated_vs_table_holding.py

Checks a specific hypothesis raised 2026-10-02 about why completion_rate
on an ELEVATED target is higher than on a table-height one for the same
checkpoint (see the BC Architecture Scaling artifact's generalization
callout: MSE 90.0% elevated vs. 46.3% table-height), and why the
"joints churn while the gripper sits still" pattern (scripts/diagnose_
drag_mechanism.py) shows up on table-height successes but wasn't checked
on elevated ones: once a block is AT a table-height target, the target's
own z equals the block's resting z, so gravity holds it there for free —
nothing physically requires the gripper to stay engaged, so a policy that
disengages/churns after success pays no PHYSICAL price, only (as of this
session) no reward price either. An elevated target has no such free
ride: if grip force or support lapses, the block falls and genuine
success is lost immediately — so a policy succeeding at an elevated
target is physically forced to stay continuously, stably engaged, not
just rewarded for it.

Traces one genuine-success elevated-target episode (reusing eval_elevated_
target.py's own elevation mechanism and held-out seed range, 200_000+ep)
against the same mse_n2000_seed0 BC checkpoint used throughout, printing
the same per-step signals as diagnose_drag_mechanism.py (contact force,
height, action) for the window AFTER genuine success is first reached —
directly comparable to that script's already-captured table-height trace
(seed=100000, t=33 onward: contact force oscillating/dropping to 0,
dz~=-0.85 to -0.92, grip_z flatlined at 0.4156).

Run with:
    modal run scripts/diagnose_elevated_vs_table_holding.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=300)
def diagnose(
    ckpt_path: str = "checkpoints/bc_scaling/mse_n2000_seed0_smw0.0_k5.pt",
    n_eval_episodes: int = 30,
    max_steps: int = 100,
    elevation_min: float = 0.10,
    elevation_max: float = 0.30,
    block_resting_z: float = 0.425,
    lift_threshold: float = 0.02,
) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, grip_contact_forces
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type="mse")
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))

    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps)
    setup_env(env)

    traced = False
    for ep in range(n_eval_episodes):
        seed = 200_000 + ep   # same elevated-target held-out seed range as eval_elevated_target.py
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        obs, setup_ok = init_random_episode(env, rng)
        if not setup_ok:
            continue
        elevation = float(rng.uniform(elevation_min, elevation_max))
        env.unwrapped.goal[2] += elevation
        obs, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
        if terminated or truncated:
            continue

        hidden_state = None
        success = False
        ever_lifted_and_gripped = False
        trace = []
        for t in range(max_steps):
            flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
            action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            forces = grip_contact_forces(env)
            height_above_resting = float(obs["achieved_goal"][2]) - block_resting_z
            genuine_now = min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold
            if genuine_now:
                ever_lifted_and_gripped = True
            if info.get("is_success", False):
                success = True
            trace.append({
                "t": t, "action": [round(float(a), 3) for a in action],
                "left_force": round(forces["left"], 2), "right_force": round(forces["right"], 2),
                "height_above_resting": round(height_above_resting, 4),
                "genuine_now": genuine_now,
            })
            if terminated or truncated:
                break

        genuine = success and ever_lifted_and_gripped
        if genuine and not traced:
            traced = True
            print(f"\n=== ELEVATED episode seed={seed}  elevation={elevation:.3f}  label=genuine ===", flush=True)
            first_genuine_t = next(r["t"] for r in trace if r["genuine_now"])
            print(f"  first genuine grasp+lift at t={first_genuine_t}", flush=True)
            n_zero_contact_after = sum(
                1 for r in trace if r["t"] >= first_genuine_t and max(r["left_force"], r["right_force"]) == 0.0
            )
            print(f"  steps with ZERO contact force after first genuine grasp: "
                  f"{n_zero_contact_after}/{len(trace) - first_genuine_t}", flush=True)
            for row in trace[first_genuine_t:]:
                print(f"  t={row['t']:3d}  action={row['action']}  "
                      f"contact(L/R)={row['left_force']}/{row['right_force']}  "
                      f"height_above_resting={row['height_above_resting']}  genuine_now={row['genuine_now']}",
                      flush=True)
            break

    env.close()
    return {"traced": traced}


@app.local_entrypoint()
def main():
    diagnose.remote()
