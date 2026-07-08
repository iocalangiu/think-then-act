# think-then-act

Training a robot arm to plan a sequence of actions using images of the environment.

The arm lives in the **FetchPickAndPlace-v3** MuJoCo environment. The task is to pick up a block and move it to a target location.

This is work in progress with the idea to discover and understand differences in computational costs and accuracy between GRPO and PPO, as well as experimenting with different fine-tuning adapters.

---

## How it works

### Vision-Language Policy

The input image shows the arm, the block, and the target location (marked as a sphere). A VLM (**Qwen2-VL, 2B parameters**) receives the image and a structured system prompt, and is instructed to respond in a strict format:

```
<think>
Block is at [1.25, 0.75, 0.025]. Target is at [1.20, 0.90, 0.44].
Gripper is above and to the right. Priority: move left and down, keep gripper open.
</think>
<action>[-0.8, 0.3, -0.9, -1.0]</action>
```

The response is parsed by `_extract_think` and `_parse_action` into a 4-dimensional action vector `[dx, dy, dz, grip]` controlling the gripper end-effector.

### Dense Reward

Rather than using the environment's sparse reward (+1 for correct placement, 0 otherwise), I implemented a **dense reward function** that rewards correct sub-steps of the sequence — for example, moving the gripper toward the block, or closing the gripper once near it.

A structured reward function is possible here because the simulator provides ground-truth state.

### GRPO Training

The policy is fine-tuned using **GRPO** (Group Relative Policy Optimization) with LoRA adapters on the query and value projection weights. GRPO runs multiple rollouts from the same starting state, computes a reward for each, and increases the probability of higher-reward responses relative to the group:

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
modal volume get rl-harness-model-cache eval_rollout.mp4 ./eval_rollout.mp4
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
