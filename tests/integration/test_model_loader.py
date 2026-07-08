"""
Integration test: actually loads Qwen2-VL-2B-Instruct and attaches a LoRA
adapter on a real GPU. This is the expensive one — it downloads/loads a
~4GB model — so it's split from test_env_rollout.py and only runs via
`modal run tests/run_integration.py --gpu-tests`.
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/model-cache")


def test_load_base_model_and_attach_lora():
    from think_then_act.policy.model_loader import MODEL_ID, attach_lora, load_base_model

    model, processor = load_base_model(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    assert processor is not None

    lora_model = attach_lora(model, lora_rank=4)
    trainable = [p for p in lora_model.parameters() if p.requires_grad]
    total = list(lora_model.parameters())

    # Only the LoRA adapter should be trainable; the base model stays frozen.
    assert 0 < len(trainable) < len(total)
