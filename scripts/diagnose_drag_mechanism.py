"""
diagnose_drag_mechanism.py

Checks a specific mechanistic hypothesis for why the flat MSE BC policy's
rollouts sometimes drag/slide the block along the table instead of lifting
it cleanly: that the policy keeps commanding a downward (or otherwise
blocked) displacement that the table physically prevents, and that
unrealized push manifests as real contact pressure (and, per
env/setup.py's randomize_gripper_start docstring, re-synced mocap targets
each step rather than classic unbounded windup) rather than as actual
motion — which could explain both symptoms raised: dragging (downward
pressure increases friction/coupling with the table during lateral motion)
and "joints moving while the gripper doesn't" (the arm keeps getting
driven toward a target the table is blocking).

Rolls the checkpoint out over the same 30 fixed table-height eval seeds
used throughout this project (seed=100_000+ep), classifies each episode as
genuine success / raw-success-only ("slide") / neither using the project's
standard two-finger-contact + lift verification, and for slide episodes
(plus one genuine-success episode as a control) logs per-step: commanded
action, gripper position, block position + its step-to-step xy delta,
finger-block contact force, and whether the gripper/fingers are touching
the table (env.setup.has_contact(env, "table0")) — directly testable: does
a block-drag step coincide with table contact and a persistent
non-positive commanded dz, i.e. "pushing down into something that's
already stopped it"?

Run with:
    modal run scripts/diagnose_drag_mechanism.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=2.0,
              volumes={MODEL_CACHE_DIR: model_volume}, timeout=600)
def diagnose(
    ckpt_path: str = "checkpoints/bc_scaling/mse_n2000_seed0_smw0.0_k5.pt",
    n_eval_episodes: int = 30,
    max_steps: int = 100,
    lift_threshold: float = 0.02,
    block_resting_z: float = 0.425,
    drag_xy_threshold: float = 0.004,   # metres/step of block xy motion counted as "dragging"
    max_traced_slide_episodes: int = 3,
    use_blocked_descent_guard: bool = False,   # 2026-10-02: confirmed via mocap_gap_z telemetry
                                  # that a commanded descent the table physically blocks doesn't
                                  # vanish -- it sits as a standing mocap-vs-actual setpoint error
                                  # (up to ~0.045m observed) that persists for dozens of steps, in
                                  # both drag AND genuine-success episodes. This closed-loop guard
                                  # needs no raw mujoco access: if last step's commanded dz<0
                                  # produced ~no real grip_z change, the command was blocked, so
                                  # suppress (not re-issue) it this step rather than let it keep
                                  # standing as unrealized pressure.
    blocked_dz_threshold: float = 0.001,   # metres of realized |grip_z change| below which last
                                  # step's negative-dz command is considered "blocked"
) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    import mujoco
    from think_then_act.env.setup import setup_env, init_random_episode, grip_contact_forces, get_contact_geoms
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type="mse")
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))
    print(f"loaded {ckpt_path}", flush=True)

    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps)
    setup_env(env)
    raw = env.unwrapped
    print(f"mujoco nmocap={raw.model.nmocap}", flush=True)

    _GRIPPER_BODIES = {"robot0:gripper_link", "robot0:l_gripper_finger_link", "robot0:r_gripper_finger_link"}

    def gripper_table_contact(env) -> bool:
        """
        True only if a GRIPPER body (not the block, object0) is touching
        table0 — the generic has_contact(env, "table0") helper also fires
        whenever the block is resting on the table (always true), which is
        useless here. This is the specific signal the drag hypothesis
        needs: is the gripper itself pressed against the table surface.
        """
        for n1, n2 in get_contact_geoms(env):
            pair = {n1, n2}
            if "table0" in pair and (pair & _GRIPPER_BODIES):
                return True
        return False

    def mocap_gap(env) -> float:
        """Z-distance between the mocap target and the gripper body's actual
        position — directly tests whether a commanded-but-blocked displacement
        is sitting there as a persistent setpoint error, independent of the
        reset-each-step behavior noted in env/setup.py's
        randomize_gripper_start docstring."""
        raw = env.unwrapped
        if raw.model.nmocap < 1:
            return float("nan")
        grip_body_id = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link")
        return float(raw.data.mocap_pos[0][2] - raw.data.xpos[grip_body_id][2])

    episode_classes = {"genuine": [], "slide": [], "neither": []}
    per_episode_stats = []
    traced_slides = 0
    traced_genuine = False

    for ep in range(n_eval_episodes):
        seed = 100_000 + ep
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        obs, setup_ok = init_random_episode(env, rng)
        if not setup_ok:
            continue

        hidden_state = None
        success = False
        ever_lifted_and_gripped = False
        prev_block_xy = np.asarray(obs["achieved_goal"][:2], dtype=np.float64)
        prev_grip_z = float(obs["observation"][2])
        prev_action_dz = 0.0
        n_suppressed = 0

        trace = []   # only populated if this episode ends up worth printing
        for t in range(max_steps):
            flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
            action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)

            if use_blocked_descent_guard:
                # Pinned-in-z (near-zero realized movement last step) AND the
                # policy wants MORE descent right now -> suppress, regardless
                # of whether last step's command was itself already a
                # suppression (a forced dz=0 can realize ~0 movement too,
                # which must still count as "pinned" for this step's check —
                # requiring the PREVIOUS action to have been negative, as an
                # earlier version of this guard did, let the policy alternate
                # suppressed/full-negative every other step and never
                # actually stopped the standing pressure).
                realized_dz = float(obs["observation"][2]) - prev_grip_z
                is_pinned = abs(realized_dz) < blocked_dz_threshold
                if is_pinned and action[2] < 0.0:
                    action = np.array(action, dtype=np.float32, copy=True)
                    action[2] = 0.0
                    n_suppressed += 1
            prev_grip_z = float(obs["observation"][2])
            prev_action_dz = float(action[2])

            obs, reward, terminated, truncated, info = env.step(action)

            grip_pos = np.asarray(obs["observation"][:3], dtype=np.float64)
            block_xy = np.asarray(obs["achieved_goal"][:2], dtype=np.float64)
            block_xy_delta = float(np.linalg.norm(block_xy - prev_block_xy))
            prev_block_xy = block_xy

            forces = grip_contact_forces(env)
            genuine_grip_now = min(forces["left"], forces["right"]) > 0.0
            height_above_resting = float(obs["achieved_goal"][2]) - block_resting_z
            if genuine_grip_now and height_above_resting > lift_threshold:
                ever_lifted_and_gripped = True
            table_contact = gripper_table_contact(env)
            gap = mocap_gap(env)

            trace.append({
                "t": t, "action": [round(float(a), 3) for a in action],
                "grip_z": round(float(grip_pos[2]), 4),
                "block_xy_delta": round(block_xy_delta, 5),
                "left_force": round(forces["left"], 2), "right_force": round(forces["right"], 2),
                "table_contact": table_contact,
                "mocap_gap_z": round(gap, 5),
                "height_above_resting": round(height_above_resting, 4),
            })

            if info.get("is_success", False):
                success = True
            if terminated or truncated:
                break

        genuine = success and ever_lifted_and_gripped
        label = "genuine" if genuine else ("slide" if success else "neither")
        episode_classes[label].append(seed)
        max_abs_gap = max((abs(row["mocap_gap_z"]) for row in trace), default=0.0)
        n_drag_steps = sum(1 for row in trace if row["block_xy_delta"] > drag_xy_threshold)
        per_episode_stats.append({
            "seed": seed, "label": label, "n_suppressed": n_suppressed,
            "max_abs_mocap_gap_z": round(max_abs_gap, 5), "n_drag_steps": n_drag_steps,
        })

        should_trace = (label == "slide" and traced_slides < max_traced_slide_episodes) or \
                        (label == "genuine" and not traced_genuine)
        if should_trace:
            if label == "slide":
                traced_slides += 1
            else:
                traced_genuine = True
            print(f"\n=== episode seed={seed}  label={label} ===", flush=True)
            # Only print steps with non-trivial block motion, or any table
            # contact — the interesting moments, not the whole trace.
            for row in trace:
                if row["block_xy_delta"] > drag_xy_threshold or row["table_contact"]:
                    flag = "DRAG" if row["block_xy_delta"] > drag_xy_threshold else "    "
                    print(f"  t={row['t']:3d} {flag}  action={row['action']}  grip_z={row['grip_z']}  "
                          f"block_xy_delta={row['block_xy_delta']}  "
                          f"contact(L/R)={row['left_force']}/{row['right_force']}  "
                          f"gripper_table_contact={row['table_contact']}  mocap_gap_z={row['mocap_gap_z']}  "
                          f"height={row['height_above_resting']}",
                          flush=True)

    env.close()
    print(f"\n=== Summary (use_blocked_descent_guard={use_blocked_descent_guard}) ===", flush=True)
    for label, seeds in episode_classes.items():
        print(f"  {label:10s}: {len(seeds)}/{n_eval_episodes}  seeds={seeds}", flush=True)
    total_drag_steps = sum(s["n_drag_steps"] for s in per_episode_stats)
    total_suppressed = sum(s["n_suppressed"] for s in per_episode_stats)
    max_gap_overall = max((s["max_abs_mocap_gap_z"] for s in per_episode_stats), default=0.0)
    print(f"  total_drag_steps={total_drag_steps}  total_suppressed_dz_commands={total_suppressed}  "
          f"max |mocap_gap_z| across all episodes={max_gap_overall:.5f}", flush=True)

    return {
        "episode_classes": {label: seeds for label, seeds in episode_classes.items()},
        "per_episode_stats": per_episode_stats,
        "total_drag_steps": total_drag_steps,
        "total_suppressed_dz_commands": total_suppressed,
        "max_abs_mocap_gap_z": max_gap_overall,
    }


@app.local_entrypoint()
def main(ckpt_path: str = "checkpoints/bc_scaling/mse_n2000_seed0_smw0.0_k5.pt",
          use_blocked_descent_guard: bool = False):
    result = diagnose.remote(ckpt_path=ckpt_path, use_blocked_descent_guard=use_blocked_descent_guard)
    print("\n", result)
