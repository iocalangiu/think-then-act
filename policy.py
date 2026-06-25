"""
policy.py

VLMPolicy — the vision-language model policy actor for the robot arm.

Architecture:
  Input  : RGB frame (H x W x 3 uint8) + structured state dict
  Model  : Qwen/Qwen2-VL-2B-Instruct (2B params, fits in T4 16 GB VRAM)
  Output : raw text response containing <think> and <action> tags

The model is NOT expected to perform well out of the box — it has no
robotics training.  It exists here to verify the full inference pipeline
before RL training (Milestone 5) teaches it the task via reward signals.

Structured output format the policy must produce:
    <think>
    [natural language reasoning about the scene]
    </think>
    <action>[dx, dy, dz, grip]</action>

Where dx/dy/dz are end-effector delta movements in [-1, 1] and grip is
gripper command in [-1, 1] (negative = open, positive = close).
"""

from __future__ import annotations

import re
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# System prompt shown to the model before every turn.
# The concrete example at the end dramatically improves format compliance
# because the model can copy the structure rather than invent it.
SYSTEM_PROMPT = """\
You are a controller for a 7-DOF Fetch robot arm performing a pick-and-place task.

TASK: Pick up the block on the table and move it to the target position (shown in the image).

ACTION SPACE — your output controls the gripper end-effector:
  dx   : move left (−1.0) ↔ right (+1.0)
  dy   : move backward (−1.0) ↔ forward (+1.0)
  dz   : move down (−1.0) ↔ up (+1.0)
  grip : open gripper (−1.0) ↔ close gripper (+1.0)
All four values MUST be floats in [−1.0, 1.0].

YOU MUST ALWAYS RESPOND IN THIS EXACT FORMAT — no exceptions:
<think>
1. Where is the block right now?
2. Where is the target?
3. What should the arm do next to get the block to the target?
</think>
<action>[dx, dy, dz, grip]</action>

--- EXAMPLE A — arm above and to the right of block, needs to descend and align ---
<think>
Block is at [1.25, 0.75, 0.025] on the table surface. Target is at [1.20, 0.90, 0.44].
Gripper is currently above and to the right of the block. Distance: 0.48 m.
Priority: move left and down to align over the block, keep gripper open to receive it.
</think>
<action>[-0.8, 0.3, -0.9, -1.0]</action>

--- EXAMPLE B — gripper is directly above block, ready to grasp and lift ---
<think>
Block is at [1.31, 0.74, 0.025]. Target is at [1.28, 0.74, 0.40]. Distance: 0.38 m, mostly vertical.
Gripper is already aligned over the block. Close the gripper and lift sharply upward.
</think>
<action>[0.0, 0.0, 0.9, 1.0]</action>

--- EXAMPLE C — block grasped, carry it forward and slightly left to target ---
<think>
Block is at [1.30, 0.70, 0.15] — elevated, gripper is closed around it.
Target is at [1.18, 0.88, 0.42]. I need to move left (−dx), forward (+dy), and up (+dz).
</think>
<action>[-0.6, 0.7, 0.5, 1.0]</action>"""

# User prompt template — filled in with live state values each step.
USER_PROMPT_TEMPLATE = """\
Current state:
  Block position  (achieved_goal) : {achieved_goal}
  Target position (desired_goal)  : {desired_goal}
  Distance block→target            : {distance:.4f} m

Observe the image carefully and respond in the required format."""


# ---------------------------------------------------------------------------
# VLMPolicy
# ---------------------------------------------------------------------------
class VLMPolicy:
    """
    Wraps Qwen2-VL-2B-Instruct for robot arm control.

    Load once, call act() many times:
        policy = VLMPolicy(cache_dir="/model-cache")
        response, action, think = policy.act(frame, state_entry)
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        cache_dir: str | None = None,
        max_new_tokens: int = 256,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print(f"[VLMPolicy] Loading {model_id}...")
        print(f"[VLMPolicy] Cache dir: {cache_dir or 'HuggingFace default'}")

        load_kwargs: dict = {
            "torch_dtype" : torch.float16,  # fp16 halves VRAM usage vs fp32
            "device_map"  : "auto",          # lets accelerate place layers on GPU
        }
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, **load_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            **({} if cache_dir is None else {"cache_dir": cache_dir}),
        )
        self.max_new_tokens = max_new_tokens
        self.device = device
        print("[VLMPolicy] Model loaded and ready.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(
        self,
        frame: np.ndarray,
        state_entry: dict,
    ) -> tuple[str, np.ndarray, str, bool, bool]:
        """
        Run one inference step.

        Args:
            frame       : (H, W, 3) uint8 RGB array from ObservationHarness.
            state_entry : a single entry from episode_log.

        Returns:
            raw_response    : full model text (log/debug)
            action          : (4,) float32 clipped to [-1, 1]; zeros if tag missing
            think_text      : text from <think>...</think>; empty str if tag missing
            think_tag_found : True if <think> tag was present in raw_response
            action_tag_found: True if <action> tag was present in raw_response
        """
        prompt = self._build_prompt(state_entry)
        pil_image = Image.fromarray(frame)
        raw_response = self._generate(pil_image, prompt)
        think_text, think_found = self._extract_think(raw_response)
        action, action_found    = self._parse_action(raw_response)
        return raw_response, action, think_text, think_found, action_found

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, state_entry: dict) -> str:
        achieved = [round(v, 4) for v in state_entry["achieved_goal"]]
        desired  = [round(v, 4) for v in state_entry["desired_goal"]]
        distance = float(np.linalg.norm(
            np.array(state_entry["desired_goal"]) - np.array(state_entry["achieved_goal"])
        ))
        return USER_PROMPT_TEMPLATE.format(
            achieved_goal=achieved,
            desired_goal=desired,
            distance=distance,
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

        # Apply the model's chat template to get the input_ids tensor.
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
            # Qwen2-VL's generation config defaults to sampling params
            # (temperature, top_p, top_k).  Mixing do_sample=False with
            # those params triggers warnings; using do_sample=True + low
            # temperature is both warning-free and the intended usage.
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Slice off the prompt tokens; decode only the generated portion.
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

        Qwen2 chat template behaviour: apply_chat_template() with
        add_generation_prompt=True appends <think> to the *prompt* tokens.
        The model generates everything AFTER that prefix, so the opening
        <think> tag is consumed by the prompt slice and never appears in
        the decoded output.  Only </think> is visible in `text`.

        We handle both cases:
          - Full <think>...</think>  (fallback / other models)
          - think content + </think> only  (Qwen2 default)
        """
        # Case 1: full tags present in generated text (other models or explicit tags)
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            return match.group(1).strip(), True

        # Case 2: Qwen2 — opening tag was prepended by chat template.
        # Everything before </think> is the think content.
        if "</think>" in text:
            think_content = text.split("</think>")[0].strip()
            return think_content, True

        return "", False

    @staticmethod
    def _parse_action(text: str) -> tuple[np.ndarray, bool]:
        """
        Extract [dx, dy, dz, grip] from <action>[...]</action>.
        Returns (action_array, tag_was_found).

        Falls back to zeros if the format is wrong — so a bad model response
        never crashes the episode loop.  The caller uses tag_was_found to
        distinguish "model output zero action" from "tag missing entirely".
        """
        match = re.search(r"<action>\s*\[([^\]]+)\]\s*</action>", text)
        if not match:
            return np.zeros(4, dtype=np.float32), False
        try:
            values = [float(v.strip()) for v in match.group(1).split(",")]
            if len(values) != 4:
                return np.zeros(4, dtype=np.float32), False
            action = np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)
            return action, True
        except ValueError:
            return np.zeros(4, dtype=np.float32), False
