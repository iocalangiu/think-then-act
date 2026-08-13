# think-then-act

Training a robot arm to plan a sequence of actions using images of the environment.

The arm lives in the **FetchPickAndPlace-v3** MuJoCo environment. The task is to pick up a block and move it to a target location.

---
## How it works

<img width="670" height="326" alt="Screenshot 2026-07-14 at 16 33 47" src="https://github.com/user-attachments/assets/27d54f45-de89-4b1f-ad36-5344c00b4d6e" />

Hierarchical architecture (working on the low-level controller; high-level VLM not yet trained). A vision-language model reads the image + text prompt and picks a subgoal (e.g. "align_xy"), passed as a one-hot into a small MLP that outputs the actual continuous action (dx, dy, dz, grip). Both parts are trained in two stages: SFT to teach format/behavior, then RL to improve it — GRPO for the VLM, PPO for the MLP.

The low-level MLP was first trained with GRPO, but it kept collapsing partway through training. PPO worked instead, maybe because the reward is dense?

<img width="341" height="180" alt="image" src="https://github.com/user-attachments/assets/5102fe5c-ab35-4b86-86fc-1335ec60abe4" />

The `close_gripper` low-level policy was trained with randomized block sizes (1–8cm) so it
generalizes past one fixed cube. Below: gripping a 9cm-tall block — taller than anything seen
during training.

![block-size generalization demo](/Users/ioanacalangiu/Documents/GitHub/think-then-act/artifacts/close_gripper_size_videos/generalization_demo.gif)

Rollouts are run across 8 CPUs (8 MuJoCo episodes at once to collect data), then pause them while 1 core does the quick PPO update step, then repeat. PPO update step takes a split second, so it makes no sense to start rolling out in parallel.




```
advantage[i] = (reward[i] − mean(rewards)) / std(rewards)
loss = −mean(advantage[i] × log_prob[i])
```

Two memory constraints shaped the implementation:

- **Gradient accumulation per step**: rather than accumulating all computation graphs before a single backward pass (which OOMs on a 24 GB A10G with a 2B-parameter model and image tokens), gradients are backpropagated immediately after each rollout step.
- **Gradient checkpointing**: activations are recomputed during the backward pass instead of stored, trading compute for memory.

---

## Stack

| Component | Choice |
|-----------|--------|
| Simulator | MuJoCo 3.1.6 + gymnasium-robotics 1.3.1 |
| Environment | FetchPickAndPlace-v3 (headless OSMesa) |
| Policy | Qwen2-VL-2B-Instruct |
| RL algorithm | GRPO + LoRA (peft 0.12.0) |
| Compute | Modal serverless (A10G for training, T4 for eval) |

---

## Project structure

```
src/think_then_act/
  modal_app.py             — container image, Modal app, persistent volume
  env/wrapper.py           — gymnasium wrapper: captures RGB frames + episode log
  env/setup.py             — shared env setup: robot base shift, random block/target, video I/O
  policy/vlm_policy.py     — VLMPolicy: prompt builder, generator, response parser
  policy/model_loader.py   — shared Qwen2-VL + LoRA loading (base model, attach LoRA, load checkpoint)
  reward/dense_reward.py   — dense reward function (gripper distance, grasp, placement)
  training/grpo_trainer.py — GRPOTrainer: rollout collection, log-prob computation, gradient step

scripts/
  eval.py                  — evaluation harness: runs N episodes, saves rollout video
  run_train_m6.py          — full training run (50–100 iterations, checkpoints, interleaved eval)
  run_episode.py           — single-episode rollout + diagnostic dump
  sft_train.py             — SFT warm-start fine-tuning before GRPO
  generate_sft_data.py     — oracle-generated SFT training examples
  analyze_seeds.py         — classify seeds as GOOD/HARD via oracle rollouts
  compare_runs.py          — compare two grpo_m6c_metrics*.jsonl logs side by side
  milestones/              — early milestone scripts (M1–M5), kept for reference; superseded
                             by run_train_m6.py + eval.py
```

Installed as an editable package (`pip install -e .`) so the above import as `think_then_act.*` both locally and inside the Modal container image.

---

## Running

**Training (50 iterations):**
```bash
modal run --detach scripts/run_train_m6.py
```

**Evaluation against a checkpoint:**
```bash
modal run scripts/eval.py --checkpoint-path /model-cache/checkpoints/grpo_m6c_final
```

**Download rollout video:**
```bash
modal volume get rl-harness-model-cache eval_rollout.mp4 ./artifacts/eval_rollout.mp4
```

---

## Testing

**Unit tests** (pure logic — reward math, prompt/action parsing — no mujoco/torch, run locally):
```bash
pip install -e ".[dev]"
pytest
```

**Integration tests** (need mujoco/gymnasium-robotics, and for the GPU one, torch/peft — run inside the Modal container):
```bash
modal run tests/run_integration.py                # env + reward, real MuJoCo physics (CPU)
modal run tests/run_integration.py --gpu-tests     # + loads Qwen2-VL-2B on a GPU (costs more, run sparingly)
```
