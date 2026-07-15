"""
record_full_rollout.py

End-to-end visual sanity check across ALL SIX subgoal controllers chained
together on ONE live episode: align_xy -> descend -> close_gripper -> lift
-> move_to_target -> release. Unlike record_subgoal_video.py (one subgoal,
in isolation, with an oracle-constructed starting state), this runs each
trained policy in sequence on whatever state the PREVIOUS policy actually
left the arm in — no oracle involved anywhere — so it's the real test of
whether the six controllers compose into a working pick-and-place, not just
whether each one works in isolation.

Reuses hrl/skill_env.py's Skill abstraction (via training/fetch_skills.py's
build_fetch_skills) for build_obs/reward_and_done per subgoal, but drives
the loop directly here instead of going through SkillEnv, since this needs
a frame captured after every individual base-env step (for the video),
which SkillEnv's one-transition-per-skill-call abstraction doesn't expose.

Each subgoal runs for up to --max-steps-per-subgoal env steps or until its
own reward_and_done reports done, whichever comes first, then the NEXT
subgoal's policy takes over from wherever that left the arm — even if the
previous one didn't actually finish (its skill_success is still reported
in the summary printed at the end, so a failure to complete is visible,
not silently masked).

Output video has the 6 subgoal names listed top-right, always black; the
one currently executing is rendered bold (larger + stroke-outlined — the
default bitmap font has no real bold variant) so the highlighted label
visibly changes as the video plays.

Run with:
    modal run scripts/record_full_rollout.py
    modal run scripts/record_full_rollout.py --seed 3
    modal run scripts/record_full_rollout.py --max-steps-per-subgoal 40 --algo ppo --use-best
    modal run scripts/record_full_rollout.py --algo grpo --no-use-best   # latest GRPO checkpoints

Output (on the volume, under /model-cache/subgoal_videos/):
    full_rollout_seed{seed}{_ppo}.mp4

Download with:
    python3 -m modal volume get rl-harness-model-cache subgoal_videos/ ./artifacts/subgoal_videos/
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=900,
)
def record_full_rollout(
    seed: int = 0,
    max_steps_per_subgoal: int = 30,
    algo: str = "ppo",       # "grpo" or "ppo" — determines checkpoint filename pattern
    use_best: bool = True,   # per-subgoal best-by-completion-rate checkpoint rather than
                              # latest/final — matters because training can regress after
                              # peaking (see training/checkpoints.py)
) -> dict:
    import os
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, save_video, init_random_episode
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS
    from think_then_act.training.checkpoints import resolve_subgoal_checkpoint
    from think_then_act.training.fetch_skills import build_fetch_skills
    from think_then_act.training.subgoal_features import SUBGOAL_OBS_DIM

    print("\n" + "=" * 60)
    print(f"  FULL CHAINED ROLLOUT  seed={seed}  algo={algo}  use_best={use_best}")
    print("=" * 60)

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")

    # ------------------------------------------------------------------
    # Load one trained actor per subgoal. Fails fast (naming exactly which
    # subgoal) rather than silently substituting an untrained policy for a
    # missing checkpoint.
    # ------------------------------------------------------------------
    policies = {}
    for subgoal in SUBGOAL_LABELS:
        ckpt_path = resolve_subgoal_checkpoint(ckpt_dir, subgoal, algo=algo, use_best=use_best)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        policy = SubgoalGaussianPolicy(obs_dim=SUBGOAL_OBS_DIM)
        # PPO checkpoints (low_level_ppo.py's save_checkpoint) are
        # {"actor": ..., "critic": ...}; GRPO checkpoints are a flat
        # state_dict, loaded as-is — same convention as record_subgoal_video.py.
        policy.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
        policy.eval()
        policies[subgoal] = policy
        print(f"  {subgoal:15s} <- {os.path.basename(ckpt_path)}")

    collision_model = None
    collision_ckpt = os.path.join(ckpt_dir, "collision_predictor.pt")
    if os.path.exists(collision_ckpt):
        collision_model = CollisionPredictor()
        collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
        collision_model.eval()

    skills = build_fetch_skills(policies, collision_model=collision_model,
                                 max_steps=max_steps_per_subgoal)

    # ------------------------------------------------------------------
    # One live env for the whole chained episode — no per-subgoal reset,
    # no oracle. +50 headroom on top of the worst case (every subgoal
    # burning its full budget) so the base env's own TimeLimit never cuts
    # a subgoal off mid-attempt.
    # ------------------------------------------------------------------
    base = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                  max_episode_steps=max_steps_per_subgoal * len(SUBGOAL_LABELS) + 50)
    )
    setup_env(base)
    base.reset()
    rng = np.random.default_rng(seed)
    obs, ok = init_random_episode(base, rng)
    if not ok:
        raise RuntimeError(f"init_random_episode failed for seed={seed}; try a different seed")

    # ------------------------------------------------------------------
    # Overlay: 6 subgoal names, top-right, always black; the one currently
    # executing rendered larger + stroke-outlined to fake bold (the
    # default bitmap font — no ttf files baked into rl_image — has no real
    # bold variant).
    # ------------------------------------------------------------------
    font_normal = ImageFont.load_default(size=15)
    font_bold   = ImageFont.load_default(size=18)

    def annotate(frame: np.ndarray, current: str) -> np.ndarray:
        img = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(img)
        margin, line_h, y = 10, 24, 10
        for name in SUBGOAL_LABELS:
            is_current = name == current
            font = font_bold if is_current else font_normal
            stroke = 1 if is_current else 0
            bbox = draw.textbbox((0, 0), name, font=font, stroke_width=stroke)
            x = img.width - margin - (bbox[2] - bbox[0])
            draw.text((x, y), name, font=font, fill=(0, 0, 0),
                      stroke_width=stroke, stroke_fill=(0, 0, 0))
            y += line_h
        return np.array(img)

    frames = [annotate(base.last_frame(), SUBGOAL_LABELS[0])]
    summary = []
    info = {}
    ended_early = False

    for subgoal in SUBGOAL_LABELS:
        skill = skills[subgoal]
        skill_success = False
        terminated = truncated = False
        steps_run = 0

        for steps_run in range(1, skill.max_steps + 1):
            obs_vec = skill.build_obs(obs, base)
            action = skill.policy.act(obs_vec, deterministic=True)
            obs, reward, terminated, truncated, info = base.step(action)
            frames.append(annotate(base.last_frame(), subgoal))

            _, skill_done = skill.reward_and_done(obs, base)
            if skill_done:
                skill_success = True
                break
            if terminated or truncated:
                break

        summary.append({"subgoal": subgoal, "steps": steps_run, "skill_success": skill_success})
        print(f"  {subgoal:15s} steps={steps_run:3d}  skill_done={skill_success}")

        if terminated or truncated:
            print(f"  base env ended during {subgoal} (terminated={terminated} truncated={truncated})")
            ended_early = True
            break

    base.close()

    out_dir = os.path.join(MODEL_CACHE_DIR, "subgoal_videos")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_ppo" if algo == "ppo" else ""
    out_path = os.path.join(out_dir, f"full_rollout_seed{seed}{suffix}.mp4")
    save_video(frames, out_path, fps=10)
    model_volume.commit()

    task_success = bool(info.get("is_success", False))
    print("\n" + "=" * 60)
    print(f"  task_success={task_success}  ended_early={ended_early}  n_frames={len(frames)}")
    print(f"  Saved -> {out_path}")
    print("=" * 60)

    return {
        "status": "PASS", "seed": seed, "summary": summary,
        "task_success": task_success, "ended_early": ended_early,
        "video_path": out_path, "n_frames": len(frames),
    }


@app.local_entrypoint()
def main(
    seed: int = 0, max_steps_per_subgoal: int = 30,
    algo: str = "ppo", use_best: bool = True,
):
    print(f"\nRecording full chained rollout (align_xy -> ... -> release), "
          f"seed={seed} algo={algo} use_best={use_best}...")
    result = record_full_rollout.remote(
        seed=seed, max_steps_per_subgoal=max_steps_per_subgoal,
        algo=algo, use_best=use_best,
    )
    print(f"\nDone. task_success={result['task_success']}  ended_early={result['ended_early']}")
    for s in result["summary"]:
        print(f"  {s['subgoal']:15s} steps={s['steps']:3d}  skill_done={s['skill_success']}")
    print(f"  -> {result['video_path']}")
    print(f"\nDownload with:")
    print(f"  python3 -m modal volume get --force rl-harness-model-cache "
          f"subgoal_videos/ ./artifacts/subgoal_videos/")
