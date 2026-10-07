"""
render_mdn_rollout_video.py

Renders an MDN rollout video + a full per-step trace (chosen component,
gripper action value, contact force, block height) through the whole
episode — specifically to inspect the close-and-lift window directly,
rather than guess from aggregate stats. Built in response to the question
"maybe the gripper dim can't be modeled with a single Gaussian" — a per-step
trace shows whether grip gets stuck at a mushy middle value (consistent
with that hypothesis) or genuinely reaches the extremes but something else
goes wrong (inconsistent with it).

Run with:
    modal run scripts/multi-cube-experiment/render_mdn_rollout_video.py --seed 0
    modal run scripts/multi-cube-experiment/render_mdn_rollout_video.py --seed 5 --decode-mode weighted_mean
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=2.0,
              volumes={MODEL_CACHE_DIR: model_volume}, timeout=300)
def render_mdn_rollout_video(
    ckpt_path: str = "checkpoints/mdn_inspect_argmax.pt",
    seed: int = 0,
    decode_mode: str = "argmax",
    max_steps: int = 100,
    block_resting_z: float = 0.425,
    out_path: str = "",   # "" -> auto-named per seed/decode_mode, so
                                  # multiple seeds don't silently overwrite
                                  # each other (confirmed happened, 2026-10-02
                                  # — all 6 earlier seed runs wrote the same
                                  # fixed path, only the last survived)
    render_video: bool = True,   # False for a cheap scan (trace only, no
                                  # ffmpeg/frame-saving) across many seeds
) -> dict:
    if not out_path:
        out_path = f"demonstrations/videos/mdn_rollout_seed{seed}_{decode_mode}.mp4"
    import os
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import torch.nn.functional as F

    from think_then_act.env.setup import setup_env, init_random_episode, save_video, grip_contact_forces
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.policy.flat_bc_heads import MDNPolicy

    mdn = MDNPolicy(obs_dim=FLAT_OBS_DIM, decode_mode=decode_mode)
    full_ckpt_path = os.path.join(MODEL_CACHE_DIR, ckpt_path)
    ckpt = torch.load(full_ckpt_path, map_location="cpu")
    mdn.load_state_dict(ckpt["actor"])
    mdn.eval()

    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps, render_mode="rgb_array")
    setup_env(env)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    obs, setup_ok = init_random_episode(env, rng)

    frames = [env.render()] if render_video else None
    hidden_state = None
    trace = []
    success = False
    for t in range(max_steps):
        flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])

        # Also grab the raw pi/means for this step (not just the decoded action)
        with torch.no_grad():
            obs_t = torch.from_numpy(flat_obs.astype(np.float32)).unsqueeze(0)
            h_in = hidden_state if hidden_state is not None else torch.zeros(1, 1, mdn.rnn_hidden_size)
            pi_logits, means, log_stds, _ = mdn.forward(obs_t, h_in)
            pi = F.softmax(pi_logits, dim=-1)[0].numpy()
            best_k = int(np.argmax(pi))

        action, hidden_state = mdn.act(flat_obs, hidden_state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if render_video:
            frames.append(env.render())

        block_z = float(obs["achieved_goal"][2])
        height_above_resting = block_z - block_resting_z
        forces = grip_contact_forces(env)
        two_finger = min(forces["left"], forces["right"])

        trace.append({
            "t": t, "grip_action": float(action[3]), "best_k": best_k, "best_pi": float(pi[best_k]),
            "height_above_resting": round(height_above_resting, 4),
            "contact_force_min": round(two_finger, 4),
        })
        if info.get("is_success", False):
            success = True
        if terminated or truncated:
            break

    env.close()

    max_contact = max((r["contact_force_min"] for r in trace), default=0.0)

    if render_video:
        full_out_path = os.path.join(MODEL_CACHE_DIR, out_path)
        os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
        save_video(frames, full_out_path, fps=20)
        model_volume.commit()
        print(f"\n{'step':>4} {'grip_action':>12} {'best_k':>7} {'best_pi':>8} {'height':>8} {'contact_f':>10}")
        for r in trace:
            print(f"{r['t']:>4} {r['grip_action']:>+12.3f} {r['best_k']:>7} {r['best_pi']:>8.3f} "
                  f"{r['height_above_resting']:>+8.4f} {r['contact_force_min']:>10.4f}")
        print(f"\nsuccess={success}  max_contact={max_contact:.4f}  video saved -> {full_out_path}")
    else:
        print(f"seed={seed:>3}  success={success}  max_contact={max_contact:.4f}")

    return {"success": success, "trace": trace, "out_path": out_path if render_video else None,
            "max_contact": max_contact}


@app.local_entrypoint()
def main(seed: int = 0, decode_mode: str = "argmax", render_video: bool = True):
    result = render_mdn_rollout_video.remote(seed=seed, decode_mode=decode_mode, render_video=render_video)
    print(f"\nDone: success={result['success']}  max_contact={result['max_contact']:.4f}")
