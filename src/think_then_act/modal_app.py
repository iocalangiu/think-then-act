"""
think_then_act.modal_app

Central Modal image definition for the rl_harness_robotics project.
Import `app`, `rl_image`, and `model_volume` from this module in every
other Modal script so all functions share one consistent environment.

MuJoCo on a headless server needs:
  - libGL / libEGL / osmesa  (software OpenGL renderer; no display)
  - libglfw3                 (MuJoCo windowing dependency, even headless)
  - Python mujoco binding    (compiled against the system libs above)
  - gymnasium-robotics       (Fetch* envs built on MuJoCo)

VLM policy actor (M3+) additionally needs:
  - torch + CUDA             (GPU inference)
  - transformers             (HuggingFace model loading)
  - qwen-vl-utils            (image pre-processing for Qwen2-VL)
"""

import modal

# ---------------------------------------------------------------------------
# Modal App — one object shared across the entire project.
# ---------------------------------------------------------------------------
app = modal.App("rl-harness-robotics")

# ---------------------------------------------------------------------------
# Persistent volume for HuggingFace model weights.
# Qwen2-VL-2B is ~4 GB; caching avoids re-downloading on every run.
# The volume is created once and reused across all Modal functions.
# ---------------------------------------------------------------------------
model_volume = modal.Volume.from_name(
    "rl-harness-model-cache",
    create_if_missing=True,
)
MODEL_CACHE_DIR = "/model-cache"   # mount point inside the container

# ---------------------------------------------------------------------------
# Cloud container image — layered so Docker cache is maximised.
# Each .pip_install() call is a separate Docker layer; change one group
# without invalidating the cache for the others.
# ---------------------------------------------------------------------------
rl_image = (
    modal.Image.debian_slim(python_version="3.11")

    # --- Layer 1: OS libs for headless OpenGL rendering (rarely changes) ---
    .apt_install(
        "libgl1-mesa-glx",       # software GL implementation
        "libgl1-mesa-dev",
        "libglfw3",              # windowing (mujoco links against it)
        "libglfw3-dev",
        "libgles2-mesa-dev",
        "libegl1-mesa-dev",
        "libosmesa6",            # OSMesa: off-screen software renderer
        "libosmesa6-dev",
        "libglew-dev",
        "patchelf",              # needed by mujoco wheel post-install
        "ffmpeg",                # frame/video I/O
    )

    # --- Layer 2: RL simulation packages ---
    # Compatibility chain (learned from build failures):
    #   gymnasium-robotics 1.2.x → requires mujoco<3.0          (too old)
    #   gymnasium-robotics 1.3.x → requires mujoco>=3.0 AND gymnasium>=1.0.0
    #   gymnasium 0.29.x is the last 0.x release; 1.0.0 is the first 1.x.
    # All three must move together.
    .pip_install(
        "mujoco==3.1.6",              # MuJoCo physics engine Python binding
        "gymnasium==1.0.0",           # first 1.x; required by gym-robotics 1.3.x
        "gymnasium-robotics==1.3.1",  # Fetch* envs; requires mujoco>=3.0, gym>=1.0
        "numpy==1.26.4",
        "imageio==2.34.1",            # frame I/O
    )

    # --- Layer 3: Vision / ML packages for the VLM policy actor + RL trainer ---
    # Separated from Layer 2 so changing ML deps doesn't bust the
    # (slower) simulation layer cache and vice versa.
    .pip_install(
        # PyTorch with CUDA — Modal containers have CUDA 12.x pre-installed.
        "torch==2.3.0",
        "torchvision==0.18.0",
        # HuggingFace stack — transformers 4.45+ required for Qwen2-VL.
        "transformers==4.45.2",
        "accelerate==0.33.0",
        "huggingface_hub==0.24.6",
        # Qwen2-VL image pre-processing utilities.
        "qwen-vl-utils==0.0.8",
        # Image manipulation (PIL) used in the policy prompt builder.
        "Pillow==10.4.0",
        # LoRA fine-tuning — M5 GRPO trainer
        "peft==0.12.0",
        # 4-bit (NF4) base model quantization — QLoRA, faster generation
        "bitsandbytes==0.43.3",
        # Experiment tracking — M6C training run
        "wandb>=0.17.0",
    )

    # --- Layer 4: local project package ---
    # modal.Mount was removed in Modal 1.x.  add_local_python_source()
    # bakes local .py files into the image so containers can import them.
    # Bake the whole think_then_act package — new submodules just work,
    # no per-module list to maintain.
    # copy=True: without it, Modal defers this to a runtime file-mount and
    # forbids any further build steps (e.g. tests/run_integration.py's
    # extra .pip_install("pytest")) after this point in the chain.
    .add_local_python_source("think_then_act", copy=True)
)
