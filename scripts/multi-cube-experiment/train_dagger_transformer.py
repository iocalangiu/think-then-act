"""
train_dagger_transformer.py

DAgger (Ross et al. 2011) for the transformer architecture's single-cube
skill -- a stable SUPERVISED alternative, chosen over retrying PPO after
real instability was observed there (every attempt at lr=3e-4 and lr=1e-4
collapsed). A real root cause WAS found and fixed afterward (a dropout-
induced log_prob inconsistency between sample()'s rollout-time path and
recompute_log_prob()'s training-time path -- see transformer_policy.py's
own docstring, confirmed directly: mean abs log_prob diff of 0.24 on
IDENTICAL weights/inputs before the fix, 0.000000 after), but the user
chose DAgger over a PPO retry regardless, so this script proceeds on that
basis rather than re-attempting PPO.

Expert = the already-strong mse+PPO checkpoint
(flat_task_ppo_gripper3d_v1_cont_best.pt, 70% completion) -- queried for
its own action at every state the STUDENT (transformer) visits while
rolling out in the real env, not states the expert itself would visit.
That is the actual point of DAgger over plain BC: it corrects the student
specifically where ITS OWN policy wanders (including mistakes plain BC
never learned to avoid), not just wherever the original demos happened to
go.

Each round:
  1. Roll out the CURRENT student in the real env -- its own actions
     actually step the episode, so the visited states reflect today's
     weights, not the expert's or the original teacher's.
  2. Separately run the EXPERT's own hidden state forward over that SAME
     observation sequence, in order -- the expert never acts in the env,
     it only "watches" the student's trajectory and reports what IT would
     have done at each visited state.
  3. Aggregate (obs, expert_action) pairs with every PRIOR round's data,
     not just the latest -- the defining feature of DAgger over a single
     relabel-and-retrain pass.
  4. Retrain the student via standard supervised BC on the full
     aggregated set.

Run with:
    modal run --detach scripts/train_dagger_transformer.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=4.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600 * 3)
def train_dagger_transformer(
    student_ckpt: str = "checkpoints/gripper3d_v1/transformer_n2000_seed0.pt",
    expert_ckpt: str = "checkpoints/flat_task_ppo_gripper3d_v1_cont_best.pt",
    n_rounds: int = 5,
    episodes_per_round: int = 30,
    max_episode_steps: int = 100,
    n_epochs_per_round: int = 10,
    lr: float = 3e-4,
    eval_episodes: int = 30,
    seed: int = 0,
    ckpt_name: str = "dagger_transformer_v1",
) -> dict:
    import os
    import json
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, grip_contact_forces, randomize_gripper_start_3d
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.flat_task_env import FlatTaskEnv
    from think_then_act.reward.flat_task_reward import FlatTaskWeights
    from think_then_act.training.flat_task_ppo import FlatTaskPPOTrainer, FlatTaskPPOConfig
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer

    print("\n" + "=" * 60)
    print("  DAGGER: transformer student, mse+PPO expert")
    print("=" * 60)

    torch.manual_seed(seed)

    student_cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type="transformer", lr=lr, n_epochs=1)
    student = MultiHeadBCTrainer(student_cfg)
    student.load_checkpoint(os.path.join(MODEL_CACHE_DIR, student_ckpt))
    print(f"  loaded student (transformer) from {student_ckpt}")

    expert_ppo_cfg = FlatTaskPPOConfig(obs_dim=FLAT_OBS_DIM, architecture="mse")
    expert_trainer = FlatTaskPPOTrainer(expert_ppo_cfg)
    expert_trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, expert_ckpt))
    expert = expert_trainer.actor
    expert.eval()
    print(f"  loaded expert (mse) from {expert_ckpt}")

    def make_env():
        base = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_episode_steps)
        setup_env(base)
        return FlatTaskEnv(base, weights=FlatTaskWeights(), randomize_pose_prob=1.0, pose_scheme="gripper_3d")

    def run_eval(actor, n_eval_episodes=eval_episodes):
        eval_env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_episode_steps)
        setup_env(eval_env)
        n_genuine = 0
        n_scored = 0
        for ep in range(n_eval_episodes):
            seed_ep = 200_000 + ep
            rng = np.random.default_rng(seed_ep)
            reset_obs, _ = eval_env.reset(seed=seed_ep)
            obs, pose_ok, _ = randomize_gripper_start_3d(eval_env, rng, reset_obs)
            if not pose_ok:
                continue
            obs, setup_ok = init_random_episode(eval_env, rng)
            if not setup_ok:
                continue
            n_scored += 1
            hidden_state = None
            success = False
            ever_lifted_and_gripped = False
            for _ in range(max_episode_steps):
                flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
                action, hidden_state = actor.act(flat_obs, hidden_state, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                height_above_resting = float(obs["achieved_goal"][2]) - 0.425
                forces = grip_contact_forces(eval_env)
                if min(forces["left"], forces["right"]) > 0.0 and height_above_resting > 0.02:
                    ever_lifted_and_gripped = True
                if info.get("is_success", False):
                    success = True
                if terminated or truncated:
                    break
            if success and ever_lifted_and_gripped:
                n_genuine += 1
        eval_env.close()
        return n_genuine / n_scored if n_scored else 0.0

    initial_rate = run_eval(student.actor)
    print(f"  [round 0 / BC-only] completion_rate={initial_rate:.1%}")

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")
    best_ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}_best.pt")
    student.save_checkpoint(best_ckpt_path)
    model_volume.commit()
    best_rate = initial_rate
    history = [{"round": 0, "completion_rate": initial_rate, "n_aggregated": 0}]

    aggregated_demos = []
    for round_i in range(1, n_rounds + 1):
        env = make_env()
        new_demos = []
        attempt = 0
        while len(new_demos) < episodes_per_round and attempt < episodes_per_round * 3:
            seed_ep = round_i * 1_000_000 + attempt
            attempt += 1
            rng = np.random.default_rng(seed_ep)
            obs, info = env.reset(rng=rng, seed=seed_ep)
            if not info.get("setup_ok", True):
                continue

            student_hidden = None
            expert_hidden = None
            obs_list, expert_action_list = [], []
            for _ in range(max_episode_steps):
                obs_arr = np.asarray(obs, dtype=np.float32)
                student_action, student_hidden = student.actor.act(obs_arr, student_hidden, deterministic=True)
                with torch.no_grad():
                    expert_action, expert_hidden = expert.act(obs_arr, expert_hidden, deterministic=True)
                obs_list.append(obs_arr)
                expert_action_list.append(np.asarray(expert_action, dtype=np.float32))
                next_obs, reward, terminated, truncated, step_info = env.step(student_action)
                obs = next_obs
                if terminated or truncated:
                    break
            new_demos.append({
                "obs": np.stack(obs_list).astype(np.float32),
                "teacher_action": np.stack(expert_action_list).astype(np.float32),
            })
        env.close()

        aggregated_demos.extend(new_demos)
        print(f"  [round {round_i}] collected {len(new_demos)} student rollouts (student's own actions "
              f"stepped the env, expert relabeled every visited state) -> aggregated pool now "
              f"{len(aggregated_demos)}")

        epoch_losses = []
        rng_train = np.random.default_rng(seed + round_i)
        for epoch in range(n_epochs_per_round):
            perm = rng_train.permutation(len(aggregated_demos))
            shuffled = [aggregated_demos[i] for i in perm]
            result = student.fit(shuffled)
            epoch_losses.append(result["epoch_losses"][0])
        print(f"  [round {round_i}] retrain losses: {[round(l, 4) for l in epoch_losses]}")

        rate = run_eval(student.actor)
        print(f"  [round {round_i}] completion_rate={rate:.1%}")
        history.append({"round": round_i, "completion_rate": rate, "n_aggregated": len(aggregated_demos),
                         "epoch_losses": epoch_losses})

        latest_ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}_latest.pt")
        student.save_checkpoint(latest_ckpt_path)
        model_volume.commit()
        if rate > best_rate:
            best_rate = rate
            student.save_checkpoint(best_ckpt_path)
            model_volume.commit()
            print(f"    [best] completion_rate={rate:.1%} -> {best_ckpt_path}")

        history_path = os.path.join(ckpt_dir, f"{ckpt_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=float)
        model_volume.commit()

    print(f"\n  Final: best completion_rate={best_rate:.1%}")
    return {"best_completion_rate": best_rate, "initial_rate": initial_rate, "history": history}


@app.local_entrypoint()
def main(
    student_ckpt: str = "checkpoints/gripper3d_v1/transformer_n2000_seed0.pt",
    expert_ckpt: str = "checkpoints/flat_task_ppo_gripper3d_v1_cont_best.pt",
    n_rounds: int = 5,
    episodes_per_round: int = 30,
    n_epochs_per_round: int = 10,
    lr: float = 3e-4,
    ckpt_name: str = "dagger_transformer_v1",
):
    result = train_dagger_transformer.remote(
        student_ckpt=student_ckpt, expert_ckpt=expert_ckpt, n_rounds=n_rounds,
        episodes_per_round=episodes_per_round, n_epochs_per_round=n_epochs_per_round,
        lr=lr, ckpt_name=ckpt_name,
    )
    print("\n", result)
