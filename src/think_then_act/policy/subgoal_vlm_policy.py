"""
think_then_act.policy.subgoal_vlm_policy

SubgoalVLMPolicy — the HIGH-LEVEL VLM policy actor: looks at a frame plus
the current state and picks which of the 6 subgoals (SUBGOAL_LABELS)
should run next, to be handed straight to hrl.skill_env.SkillEnv.step().
See memory: hierarchical_architecture.md.

This is a SEPARATE policy from policy.vlm_policy.VLMPolicy, not a
replacement edited in place — VLMPolicy predates the SkillEnv pivot and
outputs raw dx/dy/dz/grip bins (a flat, non-hierarchical controller);
scripts/generate_sft_data.py/sft_train.py train that one. This file and
scripts/generate_subgoal_sft_data.py/subgoal_sft_train.py are that
predecessor's replacement for the high-level role specifically.

Structured output format the policy must produce:
    <think>
    [natural language reasoning about the scene]
    </think>
    <action>subgoal_label</action>

Where subgoal_label is exactly one of SUBGOAL_LABELS — a bare category
token, not coordinates. Unlike VLMPolicy's 4x17-bin continuous action
space, there is nothing to discretize/decode here: SkillEnv.step(name)
consumes the label string directly, and Skill.build_obs already reads
achieved_goal/desired_goal straight from the env obs, not from anything
the VLM outputs (see training/fetch_skills.py) — so <action> only ever
needs to name a category.

Privileged numeric state (gripper/block/target positions) is given in the
prompt, same as VLMPolicy — this policy is trained to REASON about which
subgoal applies, not to localize the block from pixels. Swapping in a
detector-estimated achieved_goal/desired_goal later (in place of the
oracle-perfect ones) needs no change here, only to how the caller fills
USER_PROMPT_TEMPLATE.
"""

from __future__ import annotations

import re
import numpy as np
from PIL import Image

from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# System prompt shown to the model before every turn.
# STRATEGY mirrors training.subgoal_labeler.label_subgoal's own priority
# order exactly — that's the same rule the SFT labels were generated with,
# so the prompt and the training targets describe one consistent policy,
# not two independently-phrased ones.
SYSTEM_PROMPT = f"""\
You are the high-level controller for a 7-DOF Fetch robot arm performing a \
pick-and-place task. You do not move the arm directly — you choose which \
SUBGOAL a low-level controller should execute next.

TASK: Pick up the block on the table and move it to the target position \
(shown in the image as a small sphere).

VALID SUBGOALS — output exactly one of these words:
  {", ".join(SUBGOAL_LABELS)}

YOU MUST ALWAYS RESPOND IN THIS EXACT FORMAT:
<think>
[your reasoning about the current scene]
</think>
<action>subgoal_label</action>

Example: <action>align_xy</action>

STRATEGY — identify the current state and pick the matching subgoal:
  Not holding the block, not above it laterally       -> align_xy
  Not holding the block, aligned but still above it    -> descend
  Not holding the block, aligned and at grasp height   -> close_gripper
  Holding the block, still near table height           -> lift
  Holding the block, lifted, not yet at the target     -> move_to_target
  Holding the block, at the target                     -> release

Analyse the image carefully. Ground your reasoning in the actual gripper/\
block/target numbers given below, not just the picture."""

# User prompt template — filled in with live state values each step.
# `is_grasped` is stated as a GIVEN fact, not left for the model to derive
# from finger-width + geometry — added 2026-07-17 after a held-out accuracy
# eval showed close_gripper/lift recall collapsing to ~4% (vs ~58% for
# align_xy) while the model over-predicted "release" as a generic fallback.
# Those are exactly the subgoals whose correct choice hinges on already
# knowing whether the block is grasped; see
# training.subgoal_labeler.is_block_grasped's docstring for the full story.
#
# `table_height` (added 2026-07-20) is the same idea applied to `lift`:
# reward/subgoal_reward.py's reward_lift computes height_above_table =
# block_z - table_z, and that arithmetic was already being SHOWN in
# training targets (generate_subgoal_sft_data.py's "lift" template), but
# table_z itself was never stated anywhere in the model's own input — a
# held-out eval showed lift collapsing to 0% recall, with every true-lift
# example instead reasoning about distance-to-TARGET (whose position IS
# given) rather than height-above-TABLE (whose reference constant wasn't).
# Constant across every example (this env's table is fixed), but stating
# it explicitly gives the model something in its own context to point the
# subtraction at, instead of a value it can only have memorized.
USER_PROMPT_TEMPLATE = """\
Current state:
  Gripper position : {gripper_pos}
  Block position    : {achieved_goal}
  Target position   : {desired_goal}
  Block grasped     : {is_grasped}
  Table height      : {table_height}m

Observe the image carefully and respond in the required format."""


# ---------------------------------------------------------------------------
# SubgoalVLMPolicy
# ---------------------------------------------------------------------------
class SubgoalVLMPolicy:
    """
    Wraps Qwen2-VL-2B-Instruct for high-level subgoal selection.

    Load once, call act() many times:
        policy = SubgoalVLMPolicy(cache_dir="/model-cache")
        response, subgoal, think = policy.act(frame, state_entry)
        obs, reward, done, truncated, info = skill_env.step(subgoal)  # if subgoal is not None
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        cache_dir: str | None = None,
        lora_path: str | None = None,
        # 400, not 256: eval_subgoal_vlm.py already bumped its OWN generate()
        # call to 400 (worked-arithmetic templates run ~140-450 tokens, and
        # an early-training model prone to rambling needs real headroom
        # above that — a tight cap here silently shows up as a PARSE_FAIL,
        # not a wrong-but-parseable answer). That fix was only ever applied
        # at the eval script's call site, not here — every OTHER caller
        # (record_subgoal_demo.py) inherited the stale, now-too-small
        # default, confirmed 2026-07-20 via a live demo raw_response cut off
        # mid-word before ever reaching </think>.
        max_new_tokens: int = 400,
        device: str = "cuda",
    ) -> None:
        from think_then_act.policy.model_loader import load_base_model, load_lora_checkpoint

        print(f"[SubgoalVLMPolicy] Loading {model_id}...")
        print(f"[SubgoalVLMPolicy] Cache dir: {cache_dir or 'HuggingFace default'}")

        self.model, self.processor = load_base_model(model_id, cache_dir=cache_dir)

        if lora_path:
            print(f"[SubgoalVLMPolicy] Loading LoRA adapter from {lora_path}...")
            self.model = load_lora_checkpoint(self.model, lora_path)
            print("[SubgoalVLMPolicy] LoRA adapter loaded.")

        self.max_new_tokens = max_new_tokens
        self.device = device
        print("[SubgoalVLMPolicy] Model loaded and ready.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(
        self,
        frame: np.ndarray,
        state_entry: dict,
    ) -> tuple[str, str | None, str, bool, bool]:
        """
        Run one inference step.

        Args:
            frame       : (H, W, 3) uint8 RGB array from ObservationHarness.
            state_entry : dict with "observation"/"achieved_goal"/"desired_goal"
                          (e.g. a raw SkillEnv/base-env obs, or an episode_log entry).

        Returns:
            raw_response    : full model text (log/debug)
            subgoal         : one of SUBGOAL_LABELS, or None if unparseable/
                               unknown — callers must handle None themselves
                               (e.g. re-prompt, or fall back to a default
                               subgoal) since SkillEnv.step requires a valid
                               skill name.
            think_text      : text from <think>...</think>; empty str if tag missing
            think_tag_found : True if <think> tag was present in raw_response
            action_tag_found: True if <action> tag was present in raw_response
        """
        prompt = self._build_prompt(state_entry)
        # Resize to 224x224 -- MUST match subgoal_sft_train.py's compute_loss/
        # quick_eval exactly (same resize, same LANCZOS filter). The LoRA
        # adapter was trained exclusively on 224x224 inputs; feeding it the
        # raw render resolution (480x480) here would silently evaluate/run
        # it off-distribution from what it actually learned.
        pil_image = Image.fromarray(frame).resize((224, 224), Image.LANCZOS)
        raw_response = self._generate(pil_image, prompt)
        think_text, think_found = self._extract_think(raw_response)
        subgoal, action_found   = self._parse_action(raw_response)
        return raw_response, subgoal, think_text, think_found, action_found

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, state_entry: dict) -> str:
        from think_then_act.reward.subgoal_reward import DEFAULT_WEIGHTS
        from think_then_act.training.subgoal_labeler import is_block_grasped

        obs_arr     = np.array(state_entry["observation"])
        gripper_pos = [round(v, 4) for v in obs_arr[0:3]]
        achieved    = [round(v, 4) for v in state_entry["achieved_goal"]]
        desired     = [round(v, 4) for v in state_entry["desired_goal"]]
        # Derived here (not read off state_entry) from the SAME obs_arr this
        # function already has — the full 25-float observation includes the
        # finger-width slice is_block_grasped needs, so no caller changes
        # are required (record_subgoal_demo.py, eval_subgoal_vlm.py already
        # pass "observation" through unchanged).
        grasped = is_block_grasped(obs_arr, state_entry["achieved_goal"], state_entry["desired_goal"])
        return USER_PROMPT_TEMPLATE.format(
            gripper_pos=gripper_pos,
            achieved_goal=achieved,
            desired_goal=desired,
            is_grasped="yes" if grasped else "no",
            # Fixed constant (this env's table never moves) — not derived
            # from obs_arr, same DEFAULT_WEIGHTS.table_z the "lift" training
            # target's own worked arithmetic already uses (see
            # USER_PROMPT_TEMPLATE's comment for why it needs to be GIVEN,
            # not just used in the target text).
            table_height=DEFAULT_WEIGHTS.table_z,
        )

    def _generate(self, image: "Image.Image", user_text: str) -> str:
        """Format the Qwen2-VL chat template and run inference."""
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": user_text},
                ],
            },
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            # do_sample=True + low temperature, same reasoning as
            # VLMPolicy._generate: Qwen2-VL's default generation config sets
            # sampling params, and mixing those with do_sample=False warns.
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                # Stops the instant </action> closes instead of always
                # running to max_new_tokens — same trick eval_subgoal_vlm.py
                # already uses; this call site never had it.
                stop_strings=["</action>"],
                tokenizer=self.processor.tokenizer,
            )

        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    @staticmethod
    def _extract_think(text: str) -> tuple[str, bool]:
        """
        Pull out think content from the raw response.
        Returns (think_text, tag_was_found).

        Same Qwen2 chat-template quirk as VLMPolicy._extract_think:
        add_generation_prompt=True appends <think> to the PROMPT tokens, so
        the opening tag never appears in the decoded output — only
        </think> is visible. Handles both that case and full <think>...
        </think> (other models / explicit tags).
        """
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            return match.group(1).strip(), True

        if "</think>" in text:
            think_content = text.split("</think>")[0].strip()
            return think_content, True

        return "", False

    @staticmethod
    def _parse_action(text: str) -> tuple[str | None, bool]:
        """
        Extract the subgoal label from <action>subgoal_label</action>.
        Returns (subgoal_or_None, tag_was_found). `tag_was_found` is True
        only if a label was found AND it's a recognized SUBGOAL_LABELS
        member — an <action> tag containing garbage is treated the same as
        a missing tag, since either way there's no valid skill to run.
        """
        match = re.search(r"<action>\s*(\w+)\s*</action>", text)
        if not match:
            return None, False
        label = match.group(1)
        if label not in SUBGOAL_LABELS:
            return None, False
        return label, True
