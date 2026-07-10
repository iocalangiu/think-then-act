"""
think_then_act.policy.model_loader

Shared Qwen2-VL + LoRA loading, used by VLMPolicy (inference), GRPOTrainer
(training), and the eval/sft_train/visual_test scripts. Previously each of
these five call sites reimplemented this loading logic separately, and it
had already drifted out of sync once (eval.py silently kept an old prompt
format after policy.py's changed). Centralizing it here means a change to
dtype, cache_dir handling, or LoRA target_modules only needs to happen once.
"""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def load_base_model(
    model_id: str = MODEL_ID,
    cache_dir: str | None = None,
    device_map: str = "auto",
    load_in_4bit: bool = False,
):
    """
    Load base Qwen2-VL model + processor. Returns (model, processor).

    load_in_4bit: quantize the frozen base to NF4 (bitsandbytes) with a
    bf16 compute dtype — standard QLoRA. Autoregressive generation is
    memory-bandwidth-bound (loading weights per token dominates over
    compute), so this is a real wall-clock win for rollout collection, not
    just a memory optimization. Compatible with attach_lora()'s backward
    pass — only the base is quantized, LoRA adapters stay full precision.
    """
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    load_kwargs: dict = {"device_map": device_map}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(
        model_id, **({"cache_dir": cache_dir} if cache_dir else {})
    )
    return model, processor


def attach_lora(
    base_model,
    lora_rank: int = 8,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    target_modules=("q_proj", "v_proj"),
):
    """
    Wrap base_model with a trainable LoRA adapter.

    enable_input_require_grads() must come before gradient_checkpointing_enable()
    when using PEFT — otherwise frozen base weights break the backward graph.
    """
    from peft import get_peft_model, LoraConfig

    if getattr(base_model, "is_loaded_in_4bit", False):
        # QLoRA prep: casts norms/embeddings to fp32 for stability under a
        # quantized base and wires up enable_input_require_grads() itself.
        from peft import prepare_model_for_kbit_training
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=True
        )

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha if lora_alpha is not None else lora_rank * 2,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    return model


def load_lora_checkpoint(base_model, checkpoint_path: str, trainable: bool = False):
    """
    Load a saved LoRA adapter onto base_model.

    trainable=False (default): inference-only, used by VLMPolicy/eval.py.
    trainable=True: re-enables gradient checkpointing so training can resume
    from this checkpoint (used by GRPOTrainer.load_checkpoint).
    """
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=trainable)
    if trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    return model
