"""
trainer.py

GRPOTrainer — Group Relative Policy Optimization for the VLM robot policy.

Algorithm (one iteration):
  For each of N_STATES initial states:
    1. Generate G=4 rollouts with stochastic VLM sampling (different actions).
    2. Execute each action in the env; compute dense_reward per rollout.
    3. Normalize rewards within the group → advantages.
  Gradient step:
    4. For each rollout, compute log probability of its response under the
       current (LoRA-fine-tuned) policy.
    5. Loss = -mean(advantage × log_prob) over all rollouts.
    6. Backprop through LoRA params only; clip gradients; Adam step.

The base VLM weights are frozen. Only the LoRA adapter is updated.

Why GRPO (not vanilla PPO)?
  - No separate value-function network needed.
  - Advantage is computed purely from reward comparisons within a group.
  - Works well for structured-text outputs (think + action tokens).
  - Same paradigm used in DeepSeek-R1 for reasoning traces.
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    # Model
    model_id     : str = "Qwen/Qwen2-VL-2B-Instruct"
    cache_dir    : str = "/model-cache"

    # LoRA adapter
    lora_rank    : int  = 8
    lora_alpha   : int  = 16
    lora_dropout : float = 0.05
    # Only query/value projection weights are updated — standard LoRA practice.
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # GRPO rollout
    group_size        : int = 4   # G: number of rollouts per state
    n_states          : int = 4   # states per training iteration
    max_episode_steps : int = 5   # safety cap; env truncation is the primary stop

    # Exploration noise added to reward during rollout collection.
    # Non-zero value ensures within-group reward variance even when the
    # policy mode-collapses to a single action (e.g. the system-prompt example).
    # Keeps think-section log_prob differences meaningful for gradient flow.
    # Set to 0.0 once the policy generates diverse actions on its own (M6+).
    reward_noise_std: float = 0.05

    # Environment randomisation (match SFT training distribution)
    randomize_env : bool  = True   # shift robot base + random block/target each episode

    # Optimisation
    lr            : float = 5e-5
    max_grad_norm : float = 1.0

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()
                if not isinstance(v, list)}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class GRPOTrainer:
    """
    Thin GRPO trainer wrapping Qwen2-VL-2B-Instruct + LoRA.

    Typical usage:
        trainer = GRPOTrainer(GRPOConfig())
        for iteration in range(N):
            metrics = trainer.train_iteration(env, iteration)
        trainer.save_checkpoint("/model-cache/checkpoints/iter_5")
    """

    def __init__(self, config: GRPOConfig) -> None:
        self.config = config
        self._setup_model()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from peft import get_peft_model, LoraConfig

        print(f"[GRPOTrainer] Loading base model from cache...")
        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=self.config.cache_dir,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            cache_dir=self.config.cache_dir,
        )

        print(f"[GRPOTrainer] Applying LoRA (rank={self.config.lora_rank})...")
        lora_cfg = LoraConfig(
            r                 = self.config.lora_rank,
            lora_alpha        = self.config.lora_alpha,
            target_modules    = self.config.lora_target_modules,
            lora_dropout      = self.config.lora_dropout,
            bias              = "none",
            task_type         = "CAUSAL_LM",
        )
        self.model = get_peft_model(self.base_model, lora_cfg)
        # enable_input_require_grads() must come before gradient_checkpointing_enable()
        # when using PEFT — otherwise frozen base weights break the backward graph.
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()
        self.model.print_trainable_parameters()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=self.config.lr)
        print("[GRPOTrainer] Ready.\n")

    # ------------------------------------------------------------------
    # Rollout collection (no gradient)
    # ------------------------------------------------------------------

    def collect_rollouts(self, env, state_seed: int) -> list[dict]:
        """
        Reset env to `state_seed`, then run group_size full-episode rollouts.

        Each rollout runs up to max_episode_steps steps (or until the env
        signals terminated/truncated).  Returns a list of rollout dicts:
            steps        : list of per-step dicts (frame, state_entry, response, …)
            total_reward : sum of dense rewards across the episode
            n_steps      : number of steps actually taken
        """
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from policy import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, VLMPolicy
        from reward import compute_dense_reward

        rollouts = []

        for g in range(self.config.group_size):
            # Unique CUDA seed per rollout; env.reset(seed=...) may reset torch RNG.
            torch.cuda.manual_seed(state_seed * 10_000 + g * 100)

            current_obs, _ = env.reset(seed=state_seed)

            if self.config.randomize_env:
                from env_utils import init_random_episode
                # Same rng seed for all rollouts in group → same block/target positions,
                # different action sequences (diversity comes from temperature sampling).
                rng = np.random.default_rng(state_seed)
                current_obs, ok = init_random_episode(env, rng)
                if not ok:
                    current_obs, _ = env.reset(seed=state_seed)  # fallback

            episode_steps  = []
            total_reward   = 0.0

            for step_num in range(self.config.max_episode_steps):
                frame = env.last_frame()   # current visual state (before this action)

                # Build VLM prompt from current observation.
                achieved  = [round(v, 4) for v in current_obs["achieved_goal"]]
                desired   = [round(v, 4) for v in current_obs["desired_goal"]]
                distance  = float(np.linalg.norm(
                    np.array(current_obs["desired_goal"])
                    - np.array(current_obs["achieved_goal"])
                ))
                user_text = USER_PROMPT_TEMPLATE.format(
                    achieved_goal=achieved, desired_goal=desired, distance=distance
                )
                # 224×224: ~64 visual tokens vs ~289 at 480×480; critical for grad memory.
                # Per-rollout pixel noise breaks within-group mode collapse: same visual
                # state → different tokens → different generation paths → diverse log_probs.
                noise_rng  = np.random.default_rng(state_seed * 10_000 + g * 100 + step_num)
                frame_noisy = np.clip(
                    frame.astype(np.int32) + noise_rng.integers(-8, 9, frame.shape),
                    0, 255,
                ).astype(np.uint8)
                pil_image = Image.fromarray(frame_noisy).resize((224, 224), Image.LANCZOS)
                messages  = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text",  "text": user_text},
                    ]},
                ]

                text_input     = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                img_inputs, vid_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text_input], images=img_inputs, videos=vid_inputs,
                    return_tensors="pt", padding=True,
                ).to("cuda")

                with torch.no_grad():
                    out_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=1.5,
                        top_p=0.95,
                    )

                gen_ids  = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
                response = self.processor.batch_decode(
                    gen_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                action, action_found = VLMPolicy._parse_action(response)
                if not action_found:
                    # Seeded random fallback keeps rollout diversity when tag is absent.
                    rng    = np.random.default_rng(state_seed * 10_000 + g * 100 + step_num)
                    action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)

                next_obs, _, terminated, truncated, info2 = env.step(action)
                step_reward, breakdown = compute_dense_reward(
                    obs           = next_obs["observation"],
                    achieved_goal = next_obs["achieved_goal"],
                    desired_goal  = next_obs["desired_goal"],
                    info          = info2,
                )
                total_reward += step_reward

                episode_steps.append({
                    "frame"       : np.array(pil_image),   # 224×224 uint8
                    "state_entry" : {                       # pre-action obs for prompt rebuild
                        "observation"  : np.array(current_obs["observation"]),
                        "achieved_goal": np.array(current_obs["achieved_goal"]),
                        "desired_goal" : np.array(current_obs["desired_goal"]),
                    },
                    "response"     : response,
                    "action"       : action.tolist(),
                    "action_parsed": action_found,
                    "step_reward"  : step_reward,
                })

                current_obs = next_obs
                if terminated or truncated:
                    break

            # Small reward noise ensures non-zero within-group variance while the policy
            # is mode-collapsed.  Set reward_noise_std=0 once actions genuinely differ.
            if self.config.reward_noise_std > 0:
                rng = np.random.default_rng(state_seed * 10_000 + g)
                total_reward += float(rng.normal(0.0, self.config.reward_noise_std))

            rollouts.append({
                "steps"        : episode_steps,
                "total_reward" : total_reward,
                "n_steps"      : len(episode_steps),
            })

        return rollouts

    # ------------------------------------------------------------------
    # Log probability (WITH gradient)
    # ------------------------------------------------------------------

    def compute_episode_log_prob(self, rollout: dict) -> "torch.Tensor":
        """
        Sum of per-step log_probs over a full episode rollout.

        Gradient accumulation is handled at the STEP level in grpo_step
        (backward called per step), so this method is not called directly —
        grpo_step iterates rollout["steps"] and calls _step_log_prob.
        This method exists for clarity / testing.
        """
        import torch
        total = torch.tensor(0.0, device="cuda")
        for step in rollout["steps"]:
            total = total + self._step_log_prob(step)
        return total

    def _step_log_prob(self, step: dict) -> "torch.Tensor":
        """
        Differentiable log probability of the response tokens for one step.

        Rebuilds inputs from the stored (frame, state_entry, response) rather
        than storing CUDA tensors in the rollout — prevents memory leaks between
        collection and the gradient phase.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from policy import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

        torch.cuda.empty_cache()   # free any lingering allocations before grad forward

        frame       = step["frame"]
        state_entry = step["state_entry"]
        response    = step["response"]

        # Rebuild prompt (identical to collection phase).
        achieved  = [round(v, 4) for v in state_entry["achieved_goal"]]
        desired   = [round(v, 4) for v in state_entry["desired_goal"]]
        distance  = float(np.linalg.norm(
            np.array(state_entry["desired_goal"])
            - np.array(state_entry["achieved_goal"])
        ))
        user_text  = USER_PROMPT_TEMPLATE.format(
            achieved_goal=achieved, desired_goal=desired, distance=distance
        )
        pil_image  = Image.fromarray(frame)
        messages   = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": user_text},
            ]},
        ]
        text_input     = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(messages)

        # Prompt-only encoding — to find exactly where response tokens start.
        prompt_enc = self.processor(
            text=[text_input], images=img_inputs, videos=vid_inputs,
            return_tensors="pt", padding=True,
        )
        prompt_len = prompt_enc["input_ids"].shape[1]
        del prompt_enc   # free CPU memory before CUDA allocation

        # Full sequence (prompt + response) — this is the differentiable forward pass.
        full_text  = text_input + response
        full_enc   = self.processor(
            text=[full_text], images=img_inputs, videos=vid_inputs,
            return_tensors="pt", padding=True,
        ).to("cuda")

        outputs = self.model(**full_enc, use_cache=False)   # use_cache must be False with gradient checkpointing
        logits  = outputs.logits                   # (1, seq_len, vocab_size)

        # Slice to response region only.
        # logits[:, t, :] predicts token at position t+1, so we shift by 1.
        resp_logits     = logits[:, prompt_len - 1 : -1, :]
        resp_token_ids  = full_enc["input_ids"][:, prompt_len:]

        log_probs       = F.log_softmax(resp_logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, resp_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs.sum(dim=-1).squeeze()

    # ------------------------------------------------------------------
    # GRPO gradient step
    # ------------------------------------------------------------------

    def grpo_step(self, rollout_groups: list[list[dict]]) -> dict:
        """
        One GRPO gradient update over all collected rollout groups.

        For each group:
          advantage[i] = (reward[i] - mean(rewards)) / std(rewards)

        Loss = -mean(advantage[i] * log_prob[i])  over all groups × rollouts.

        Gradient accumulation pattern: backward() is called immediately per
        rollout so only ONE computation graph lives in memory at a time.
        Accumulating a single large graph across all 16 rollouts would OOM
        on the 24 GB A10G for a 2B-param model with image tokens.
        """
        import torch

        self.optimizer.zero_grad()

        all_rewards    : list[float] = []
        all_advantages : list[float] = []
        loss_scalar    : float       = 0.0
        total_rollouts : int         = sum(len(g) for g in rollout_groups)

        for group in rollout_groups:
            rewards    = np.array([r["total_reward"] for r in group], dtype=np.float64)
            std        = rewards.std() + 1e-8
            advantages = (rewards - rewards.mean()) / std

            all_rewards.extend(rewards.tolist())
            all_advantages.extend(advantages.tolist())

            for rollout, adv in zip(group, advantages):
                adv_t = torch.tensor(float(adv), device="cuda", dtype=torch.float32)
                # Per-step backward: keeps only ONE step's computation graph in memory.
                # Summing step log_probs before backward would hold all graphs at once
                # and OOM on A10G for multi-step episodes with a 2B VLM.
                for step in rollout["steps"]:
                    step_lp   = self._step_log_prob(step)
                    step_loss = (-adv_t * step_lp) / total_rollouts
                    loss_scalar += float(step_loss.item())
                    step_loss.backward()
                    torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "loss"        : loss_scalar,
            "mean_reward" : float(np.mean(all_rewards)),
            "std_reward"  : float(np.std(all_rewards)),
            "mean_abs_adv": float(np.mean(np.abs(all_advantages))),
        }

    # ------------------------------------------------------------------
    # One full training iteration
    # ------------------------------------------------------------------

    def train_iteration(self, env, iteration: int) -> dict:
        """Collect rollouts for n_states states, then do one GRPO update."""
        import time

        seeds = [
            self.config.n_states * iteration + i
            for i in range(self.config.n_states)
        ]

        print(f"  Collecting {self.config.n_states} × {self.config.group_size}"
              f" rollouts (seeds={seeds})...")

        t_collect = time.time()
        rollout_groups = []
        for seed in seeds:
            group      = self.collect_rollouts(env, seed)
            rewards    = [round(r["total_reward"], 4) for r in group]
            steps      = [r["n_steps"] for r in group]
            n_parsed   = sum(1 for r in group for s in r["steps"] if s.get("action_parsed", False))
            n_total    = sum(r["n_steps"] for r in group)
            # Sample the last rollout's step-0 action to check for diversity
            step0_acts = [[round(v, 2) for v in r["steps"][0]["action"]] for r in group]
            print(f"    seed={seed}: rewards={rewards}  steps={steps}  parsed={n_parsed}/{n_total}")
            print(f"             step0 actions: {step0_acts}")
            rollout_groups.append(group)
        collect_s = time.time() - t_collect

        print(f"  Running GRPO gradient step...")
        t_update = time.time()
        metrics  = self.grpo_step(rollout_groups)
        update_s = time.time() - t_update

        metrics["iteration"] = iteration + 1
        metrics["collect_s"] = round(collect_s, 1)
        metrics["update_s"]  = round(update_s,  1)
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        print(f"[GRPOTrainer] LoRA checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        import torch
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, path, is_trainable=True)
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=self.config.lr)
        print(f"[GRPOTrainer] LoRA checkpoint loaded ← {path}")
