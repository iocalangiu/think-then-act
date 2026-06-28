"""
visual_test.py

Passes a real environment frame to base Qwen2-VL-2B (no fine-tuning)
and asks it to locate the block, target, and gripper.

Reads the first frame from sft_data.jsonl so we use an actual scene.

Run with:
    modal run visual_test.py
    modal run visual_test.py --frame-index 10   # try a different frame
"""

import modal
from modal_config import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=300,
)
def query_frame(frame_index: int = 0) -> dict:
    import os, json, base64, io
    import torch
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    os.environ["HF_HOME"]            = MODEL_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

    # ------------------------------------------------------------------
    # Load a frame from sft_data.jsonl
    # ------------------------------------------------------------------
    data_path = os.path.join(MODEL_CACHE_DIR, "sft_data.jsonl")
    print(f"Reading frame {frame_index} from {data_path}...")

    with open(data_path) as f:
        for i, line in enumerate(f):
            if i == frame_index:
                ex = json.loads(line)
                break

    pil_image = Image.open(
        io.BytesIO(base64.b64decode(ex["frame_b64"]))
    )
    frame_path = os.path.join(MODEL_CACHE_DIR, f"visual_test_frame{frame_index}.png")
    pil_image.save(frame_path)
    model_volume.commit()
    print(f"  Frame: {pil_image.size}  phase={ex['phase']}  saved → {frame_path}")
    print(f"  True achieved_goal : {[round(v,3) for v in ex['achieved_goal']]}")
    print(f"  True desired_goal  : {[round(v,3) for v in ex['desired_goal']]}")

    # ------------------------------------------------------------------
    # Load base model — no SFT, no LoRA
    # ------------------------------------------------------------------
    print("\nLoading Qwen2-VL-2B-Instruct (base)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=MODEL_CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        cache_dir=MODEL_CACHE_DIR,
    )
    model.eval()

    # ------------------------------------------------------------------
    # Query 1: free-form localization
    # ------------------------------------------------------------------
    def ask(question: str, max_new_tokens: int = 200) -> str:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": question},
            ]},
        ]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(messages)
        enc = processor(
            text=[text_input], images=img_inputs, videos=vid_inputs,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    questions = [
        "Describe what you see in this image.",
        "Where is the red block in this image? Where is the robot gripper? Where is the target location (the red marker on the table)?",
        "Look at the robot arm. Is the gripper close to the block or far away? Is the block touching the target?",
    ]

    print("\n" + "=" * 60)
    results = {}
    for q in questions:
        print(f"\nQ: {q}")
        ans = ask(q)
        print(f"A: {ans}")
        results[q] = ans

    print("\n" + "=" * 60)
    print("True state for reference:")
    print(f"  block  : {[round(v,3) for v in ex['achieved_goal']]}")
    print(f"  target : {[round(v,3) for v in ex['desired_goal']]}")
    print(f"  phase  : {ex['phase']}")
    print("=" * 60)

    return results


@app.local_entrypoint()
def main(frame_index: int = 0):
    print(f"\nQuerying Qwen2-VL-2B on frame {frame_index} from sft_data.jsonl...")
    query_frame.remote(frame_index=frame_index)
    print(f"\nDownload the frame:")
    print(f"  modal volume get rl-harness-model-cache visual_test_frame{frame_index}.png ./visual_test_frame{frame_index}.png")
    print(f"  open visual_test_frame{frame_index}.png")
