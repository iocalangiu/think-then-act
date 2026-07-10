"""
think_then_act.training.grpo_trainer

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
    # NF4 quantized base (QLoRA) — generation is memory-bandwidth-bound, so
    # this speeds up rollout collection on top of batching, not just VRAM.
    load_in_4bit : bool = True

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

    # Entropy regularization: loss += -entropy_coef * H(pi_theta), computed over
    # the response-token distribution at each generated position (mean per token).
    # Subtracting an entropy bonus from a loss-to-minimize is equivalent to adding
    # it to the maximized objective — pushes the policy away from a deterministic
    # collapse onto whatever action it first found, without a KL-to-reference term.
    # Set to 0.0 to disable. Too high → outputs drift toward incoherent/unparseable
    # text; too low → GRPO's existing mode-collapse tendency (see reward_noise_std
    # above, which was a workaround for the same symptom) goes unchecked.
    entropy_coef  : float = 0.01

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
        from think_then_act.policy.model_loader import load_base_model, attach_lora

        print(f"[GRPOTrainer] Loading base model from cache "
              f"(4bit={self.config.load_in_4bit})...")
        self.base_model, self.processor = load_base_model(
            self.config.model_id, cache_dir=self.config.cache_dir,
            load_in_4bit=self.config.load_in_4bit,
        )

        print(f"[GRPOTrainer] Applying LoRA (rank={self.config.lora_rank})...")
        self.model = attach_lora(
            self.base_model,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
        )
        self.model.print_trainable_parameters()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=self.config.lr)

        # Left-padding is required for batched generate(): it pads every row
        # to the same length by adding tokens on the LEFT, so every row's
        # real prompt content ends at the same column and one slice recovers
        # each row's newly generated tokens (see collect_rollouts). _step_log_prob
        # is unaffected — it always runs at batch size 1, where padding side
        # is a no-op.
        self.processor.tokenizer.padding_side = "left"

        print("[GRPOTrainer] Ready.\n")

    # ------------------------------------------------------------------
    # Rollout collection (no gradient)
    # ------------------------------------------------------------------

    def collect_rollouts(self, envs: list, state_seed: int) -> list[dict]:
        """
        Reset `group_size` independent envs to `state_seed`, then run all
        group_size rollouts in lockstep, batching every still-running
        rollout's prompt into a SINGLE model.generate() call per step_num
        instead of group_size sequential calls.

        Autoregressive decoding at batch=1 is memory-bandwidth-bound (loading
        weights per token dominates over compute), so a GPU spends most of
        its time idle doing group_size separate single-sequence decodes.
        Batching amortizes the weight load across rows — this is where
        iteration wall-clock was almost entirely going (collect_s >> update_s
        in the M6C run logs).

        Returns the same shape as before: a list of rollout dicts
            steps        : list of per-step dicts (frame, state_entry, response, …)
            total_reward : sum of dense rewards across the episode
            n_steps      : number of steps actually taken
        """
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from think_then_act.policy.vlm_policy import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, VLMPolicy
        from think_then_act.reward.dense_reward import compute_dense_reward
        from think_then_act.env.setup import init_random_episode

        G = self.config.group_size
        assert len(envs) == G, f"collect_rollouts needs group_size={G} envs, got {len(envs)}"

        # A single batched generate() call draws independent random samples
        # per row from one CUDA RNG stream, so within-group diversity no
        # longer needs a distinct manual_seed per rollout (that was only ever
        # a diversity mechanism, not a correctness requirement) — the
        # per-rollout pixel noise below already breaks input-level symmetry.
        torch.cuda.manual_seed(state_seed * 10_000)

        current_obs = []
        for env in envs:
            obs, _ = env.reset(seed=state_seed)
            if self.config.randomize_env:
                # Same rng seed for all rollouts in group → same block/target positions,
                # different action sequences (diversity comes from temperature sampling).
                rng = np.random.default_rng(state_seed)
                obs, ok = init_random_episode(env, rng)
                if not ok:
                    obs, _ = env.reset(seed=state_seed)  # fallback
            current_obs.append(obs)

        episode_steps  = [[] for _ in range(G)]
        done           = [False] * G
        logged_failure = [False] * G

        # eval() disables gradient checkpointing's forced use_cache=False (it
        # only kicks in when self.training=True), so generate() gets its
        # KV-cache back — quick_eval() already does this correctly, but
        # collect_rollouts never did, meaning every rollout-collection
        # generate() call in this project's history has been recomputing
        # attention over the whole growing sequence at each new token
        # instead of reusing cached keys/values. Must restore train() before
        # returning so grpo_step's backward pass still gets gradient
        # checkpointing's memory savings.
        self.model.eval()
        try:
            for step_num in range(self.config.max_episode_steps):
                active = [g for g in range(G) if not done[g]]
                if not active:
                    break

                # Build one batch entry per still-running rollout.
                pil_images, messages_batch = [], []
                for g in active:
                    frame       = envs[g].last_frame()   # current visual state (before this action)
                    obs_arr     = current_obs[g]["observation"]
                    gripper_pos = [round(v, 4) for v in obs_arr[0:3]]
                    achieved    = [round(v, 4) for v in current_obs[g]["achieved_goal"]]
                    desired     = [round(v, 4) for v in current_obs[g]["desired_goal"]]
                    user_text   = USER_PROMPT_TEMPLATE.format(
                        gripper_pos=gripper_pos, achieved_goal=achieved, desired_goal=desired,
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
                    pil_images.append(pil_image)
                    messages_batch.append([
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text",  "text": user_text},
                        ]},
                    ])

                text_inputs = [
                    self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                    for m in messages_batch
                ]
                img_inputs, vid_inputs = process_vision_info(messages_batch)
                inputs = self.processor(
                    text=text_inputs, images=img_inputs, videos=vid_inputs,
                    return_tensors="pt", padding=True,
                ).to("cuda")

                with torch.no_grad():
                    out_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        stop_strings=["</action>"],
                        tokenizer=self.processor.tokenizer,
                    )

                # Left-padding means every row's real prompt ends at the same
                # column (inputs["input_ids"].shape[1]), so this one slice
                # recovers each row's generated tokens regardless of that row's
                # own real prompt length.
                prompt_len = inputs["input_ids"].shape[1]
                gen_ids    = out_ids[:, prompt_len:]
                responses  = self.processor.batch_decode(
                    gen_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                pad_id = self.processor.tokenizer.pad_token_id
                for row, g in enumerate(active):
                    response     = responses[row]
                    n_gen_tokens = int((gen_ids[row] != pad_id).sum().item())

                    action, action_found = VLMPolicy._parse_action(response)
                    if not action_found:
                        if not logged_failure[g]:
                            logged_failure[g] = True
                            preview = response.replace("\n", " ")[:220]
                            print(f"      [parse FAIL] g={g} step={step_num} "
                                  f"tokens={n_gen_tokens}  |  {preview!r}")
                        # Seeded random fallback keeps rollout diversity when tag is absent.
                        rng    = np.random.default_rng(state_seed * 10_000 + g * 100 + step_num)
                        action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)

                    next_obs, _, terminated, truncated, info2 = envs[g].step(action)
                    step_reward, breakdown = compute_dense_reward(
                        obs           = next_obs["observation"],
                        achieved_goal = next_obs["achieved_goal"],
                        desired_goal  = next_obs["desired_goal"],
                        info          = info2,
                    )

                    episode_steps[g].append({
                        "frame"       : np.array(pil_images[row]),   # 224×224 uint8
                        "state_entry" : {                       # pre-action obs for prompt rebuild
                            "observation"  : np.array(current_obs[g]["observation"]),
                            "achieved_goal": np.array(current_obs[g]["achieved_goal"]),
                            "desired_goal" : np.array(current_obs[g]["desired_goal"]),
                        },
                        "response"     : response,
                        "action"       : action.tolist(),
                        "action_parsed": action_found,
                        "step_reward"  : step_reward,
                    })

                    current_obs[g] = next_obs
                    if terminated or truncated:
                        done[g] = True
        finally:
            self.model.train()

        rollouts = []
        for g in range(G):
            total_reward = sum(s["step_reward"] for s in episode_steps[g])
            # Small reward noise ensures non-zero within-group variance while the policy
            # is mode-collapsed.  Set reward_noise_std=0 once actions genuinely differ.
            if self.config.reward_noise_std > 0:
                rng = np.random.default_rng(state_seed * 10_000 + g)
                total_reward += float(rng.normal(0.0, self.config.reward_noise_std))

            rollouts.append({
                "steps"        : episode_steps[g],
                "total_reward" : total_reward,
                "n_steps"      : len(episode_steps[g]),
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
        grpo_step iterates rollout["steps"] and calls _step_log_prob_and_entropy.
        This method exists for clarity / testing.
        """
        import torch
        total = torch.tensor(0.0, device="cuda")
        for step in rollout["steps"]:
            log_prob, _entropy = self._step_log_prob_and_entropy(step)
            total = total + log_prob
        return total

    def _step_log_prob_and_entropy(self, step: dict) -> tuple:
        """
        Differentiable (log probability, mean token entropy) of the response
        tokens for one step. Both come from the same forward pass / logits —
        entropy is free once log_prob is computed, no extra model call.

        Rebuilds inputs from the stored (frame, state_entry, response) rather
        than storing CUDA tensors in the rollout — prevents memory leaks between
        collection and the gradient phase.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from think_then_act.policy.vlm_policy import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

        torch.cuda.empty_cache()   # free any lingering allocations before grad forward

        frame       = step["frame"]
        state_entry = step["state_entry"]
        response    = step["response"]

        # Rebuild prompt (identical to collection phase).
        obs_arr     = np.array(state_entry["observation"])
        gripper_pos = [round(v, 4) for v in obs_arr[0:3]]
        achieved    = [round(v, 4) for v in state_entry["achieved_goal"]]
        desired     = [round(v, 4) for v in state_entry["desired_goal"]]
        user_text   = USER_PROMPT_TEMPLATE.format(
            gripper_pos=gripper_pos, achieved_goal=achieved, desired_goal=desired,
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

        # Full-distribution entropy per response position (not just the sampled
        # token) — H = -sum_v p(v) log p(v), averaged over response length so it
        # doesn't scale with how many tokens the response happened to use.
        probs         = log_probs.exp()
        token_entropy = -(probs * log_probs).sum(dim=-1)
        mean_entropy  = token_entropy.mean(dim=-1).squeeze()

        return token_log_probs.sum(dim=-1).squeeze(), mean_entropy

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
        policy_loss_scalar : float   = 0.0
        entropy_sum    : float       = 0.0
        entropy_count  : int         = 0
        total_rollouts : int         = sum(len(g) for g in rollout_groups)
        beta           : float       = self.config.entropy_coef

        within_group_stds : list[float] = []

        for group in rollout_groups:
            rewards    = np.array([r["total_reward"] for r in group], dtype=np.float64)
            std        = rewards.std() + 1e-8
            advantages = (rewards - rewards.mean()) / std

            within_group_stds.append(float(rewards.std()))
            all_rewards.extend(rewards.tolist())
            all_advantages.extend(advantages.tolist())

            for rollout, adv in zip(group, advantages):
                adv_t = torch.tensor(float(adv), device="cuda", dtype=torch.float32)
                # Per-step backward: keeps only ONE step's computation graph in memory.
                # Summing step log_probs before backward would hold all graphs at once
                # and OOM on A10G for multi-step episodes with a 2B VLM.
                for step in rollout["steps"]:
                    step_lp, step_entropy = self._step_log_prob_and_entropy(step)
                    policy_loss  = (-adv_t * step_lp) / total_rollouts
                    # Subtracting the entropy bonus from the loss-to-minimize is
                    # equivalent to adding beta*H to the maximized objective.
                    step_loss    = policy_loss - (beta * step_entropy) / total_rollouts
                    policy_loss_scalar += float(policy_loss.item())
                    loss_scalar        += float(step_loss.item())
                    entropy_sum         += float(step_entropy.item())
                    entropy_count       += 1
                    step_loss.backward()
                    torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        n_parsed = sum(
            1 for group in rollout_groups for r in group
            for s in r["steps"] if s.get("action_parsed", False)
        )
        n_total = sum(
            r["n_steps"] for group in rollout_groups for r in group
        )

        return {
            "loss"              : loss_scalar,
            "policy_loss"       : policy_loss_scalar,
            "mean_entropy"      : entropy_sum / max(entropy_count, 1),
            "mean_reward"       : float(np.mean(all_rewards)),
            "std_reward"        : float(np.std(all_rewards)),
            "mean_abs_adv"      : float(np.mean(np.abs(all_advantages))),
            "avg_within_std"    : float(np.mean(within_group_stds)),
            "parse_rate"        : n_parsed / max(n_total, 1),
        }

    # ------------------------------------------------------------------
    # One full training iteration
    # ------------------------------------------------------------------

    def train_iteration(self, envs: list, iteration: int) -> dict:
        """
        Collect rollouts for n_states states, then do one GRPO update.

        envs must be a list of group_size independent, pre-configured
        environments (see collect_rollouts) — reused across iterations by
        the caller, reset internally on every call.
        """
        import time

        seeds = [
            self.config.n_states * iteration + i
            for i in range(self.config.n_states)
        ]

        print(f"  Collecting {self.config.n_states} × {self.config.group_size}"
              f" rollouts (seeds={seeds})...")

        t_collect = time.time()
        rollout_groups = []
        total_parsed = 0
        total_steps_all = 0
        for seed in seeds:
            group      = self.collect_rollouts(envs, seed)
            rewards    = [round(r["total_reward"], 4) for r in group]
            steps      = [r["n_steps"] for r in group]
            n_parsed   = sum(1 for r in group for s in r["steps"] if s.get("action_parsed", False))
            n_total    = sum(r["n_steps"] for r in group)
            step0_acts = [[round(v, 2) for v in r["steps"][0]["action"]] for r in group]
            rwd_arr    = np.array([r["total_reward"] for r in group])
            total_parsed    += n_parsed
            total_steps_all += n_total
            print(f"    seed={seed}: rewards={rewards}  steps={steps}  "
                  f"parsed={n_parsed}/{n_total}  within_std={rwd_arr.std():.3f}")
            print(f"             step0 actions: {step0_acts}")
            rollout_groups.append(group)
        collect_s = time.time() - t_collect

        print(f"  Running GRPO gradient step...")
        t_update = time.time()
        metrics  = self.grpo_step(rollout_groups)
        update_s = time.time() - t_update

        metrics["iteration"]  = iteration + 1
        metrics["collect_s"]  = round(collect_s, 1)
        metrics["update_s"]   = round(update_s,  1)
        metrics["parse_rate"] = total_parsed / max(total_steps_all, 1)
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
        from think_then_act.policy.model_loader import load_lora_checkpoint
        self.model = load_lora_checkpoint(self.base_model, path, trainable=True)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=self.config.lr)
        print(f"[GRPOTrainer] LoRA checkpoint loaded ← {path}")
