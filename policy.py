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
    <action>dx dy dz grip</action>

Where dx/dy/dz/grip are integers in [0, 16] selecting from 17 evenly-spaced
bins across [-1, 1].  Bin 0 = -1.0, bin 8 = 0.0 (no movement), bin 16 = +1.0.
Grip: bin 0 = fully closed, bin 16 = fully open.
"""

from __future__ import annotations

import re
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Action discretisation — 17 bins, step 0.125, range [-1.0, 1.0].
# Bin 0 = -1.0, bin 8 = 0.0 (no movement), bin 16 = +1.0.
N_BINS = 17

def encode_action(value: float) -> int:
    """Continuous [-1, 1] → bin index [0, N_BINS-1]."""
    idx = round((float(value) + 1.0) / 2.0 * (N_BINS - 1))
    return int(np.clip(idx, 0, N_BINS - 1))

def decode_action(bin_idx: int) -> float:
    """Bin index [0, N_BINS-1] → continuous [-1, 1]."""
    return float(bin_idx) / (N_BINS - 1) * 2.0 - 1.0

# System prompt shown to the model before every turn.
# The concrete example at the end dramatically improves format compliance
# because the model can copy the structure rather than invent it.
SYSTEM_PROMPT = """\
You are a controller for a 7-DOF Fetch robot arm performing a pick-and-place task.

TASK: Pick up the block on the table and move it to the target position (shown in the image as a small sphere).

ACTION SPACE — output four integers, each in [0, 16]:
  dx   : left (0) ↔ right (16),    bin 8 = no movement
  dy   : backward (0) ↔ forward (16), bin 8 = no movement
  dz   : down (0) ↔ up (16),       bin 8 = no movement
  grip : closed (0) ↔ open (16),   bin 8 = half open

Each bin step = 0.125 in normalised units.  Use large offsets from 8 (e.g. 0–3 or 13–16) when far, moderate (4–6 or 10–12) when close.

YOU MUST ALWAYS RESPOND IN THIS EXACT FORMAT:
<think>
[your reasoning about the current scene]
</think>
<action>dx dy dz grip</action>

Example: <action>13 8 5 16</action>  (move right, stay, move down slightly, open gripper)

STRATEGY — identify your phase and act accordingly:
  APPROACH : gripper not near block → move gripper toward block, keep open (grip=16)
  GRASP    : gripper above block → descend (dz < 8), then close (grip=0)
  CARRY    : block grasped → move toward target sphere, keep closed (grip=0)

Analyse the image carefully. Your chosen bins must reflect the CURRENT visual state."""

# User prompt template — filled in with live state values each step.
USER_PROMPT_TEMPLATE = """\
Current state:
  Gripper position : {gripper_pos}
  Block position   : {achieved_goal}
  Target position  : {desired_goal}

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
        lora_path: str | None = None,
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

        if lora_path:
            from peft import PeftModel
            print(f"[VLMPolicy] Loading LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print("[VLMPolicy] LoRA adapter loaded.")

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
        obs_arr     = np.array(state_entry["observation"])
        gripper_pos = [round(v, 4) for v in obs_arr[0:3]]
        achieved    = [round(v, 4) for v in state_entry["achieved_goal"]]
        desired     = [round(v, 4) for v in state_entry["desired_goal"]]
        return USER_PROMPT_TEMPLATE.format(
            gripper_pos=gripper_pos,
            achieved_goal=achieved,
            desired_goal=desired,
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
        Extract four bin indices from <action>dx dy dz grip</action> and decode
        to continuous floats in [-1, 1].  Returns (action_array, tag_was_found).

        Falls back to zeros on any parse failure so a bad model response never
        crashes the episode loop.
        """
        match = re.search(r"<action>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</action>", text)
        if not match:
            return np.zeros(4, dtype=np.float32), False
        try:
            bins = [int(match.group(i)) for i in range(1, 5)]
            if any(b < 0 or b >= N_BINS for b in bins):
                return np.zeros(4, dtype=np.float32), False
            action = np.array([decode_action(b) for b in bins], dtype=np.float32)
            return action, True
        except ValueError:
            return np.zeros(4, dtype=np.float32), False
