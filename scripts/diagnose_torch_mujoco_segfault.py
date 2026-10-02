"""
diagnose_torch_mujoco_segfault.py

One-off diagnostic (NOT a training run, writes nothing): isolates whether
collect_sb3_teacher_demos.py's segfault is an environment-level issue (torch
+ mujoco/gymnasium-robotics simply can't coexist in this image, regardless
of SB3/TQC) or specific to what TQC.load() does. No SB3, no TQC, no
checkpoint — just torch building a trivial network, then
gymnasium_robotics building a real FetchPickAndPlace-v3 env and stepping it
once.

Image is a byte-for-byte copy of collect_sb3_teacher_demos.py's sb3_image
(same package set/versions) — defined inline here rather than imported from
that script, since scripts/ has no __init__.py and isn't set up for
cross-script imports under `modal run` (no other script in this project
does that).

Run with:
    modal run scripts/diagnose_torch_mujoco_segfault.py
"""

import modal
from think_then_act.modal_app import app

sb3_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx", "libgl1-mesa-dev", "libglfw3", "libglfw3-dev",
        "libgles2-mesa-dev", "libegl1-mesa-dev", "libosmesa6", "libosmesa6-dev",
        "libglew-dev", "patchelf", "ffmpeg",
    )
    .pip_install(
        "mujoco==3.1.6",
        "gymnasium==1.0.0",
        "gymnasium-robotics==1.3.1",
        "numpy==1.26.4",
        "imageio==2.34.1",
    )
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "stable-baselines3==2.4.0",
        "sb3-contrib==2.4.0",
        "huggingface_sb3==3.0",
        "gym",
        "shimmy>=0.2.1",
    )
    .add_local_python_source("think_then_act", copy=True)
)


@app.function(image=sb3_image, gpu=None, cpu=4.0, memory=8192, timeout=300)
def diagnose_torch_mujoco_segfault() -> dict:
    import os

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    print("step 1: import torch", flush=True)
    import torch
    torch.set_num_threads(1)
    print("step 2: build + run a trivial torch network", flush=True)
    net = torch.nn.Sequential(torch.nn.Linear(10, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4))
    x = torch.randn(1, 10)
    y = net(x)
    print(f"step 2 OK, y={y.detach().numpy()}", flush=True)

    print("step 3: import gymnasium + gymnasium_robotics", flush=True)
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    print("step 3 OK", flush=True)

    print("step 4: gym.make FetchPickAndPlace-v3 + reset + step", flush=True)
    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=50)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
    print(f"step 4 OK, obs keys={list(obs.keys())}", flush=True)

    print("step 5: a torch forward pass AFTER mujoco is loaded", flush=True)
    y2 = net(torch.randn(1, 10))
    print(f"step 5 OK, y2={y2.detach().numpy()}", flush=True)

    import stable_baselines3
    import sb3_contrib
    print(f"step 6: versions — torch={torch.__version__}  "
          f"sb3={stable_baselines3.__version__}  sb3_contrib={sb3_contrib.__version__}", flush=True)

    print("step 6a: construct a DEFAULT-size SAC (core stable_baselines3, "
          "not sb3_contrib)", flush=True)
    from stable_baselines3 import SAC
    sac_env = gym.make("FetchPickAndPlace-v3", max_episode_steps=50)
    sac_model = SAC("MultiInputPolicy", sac_env, device="cpu")
    print("step 6a OK — core SB3's SAC construction did not crash", flush=True)
    obs, info = sac_env.reset(seed=0)
    action, _ = sac_model.predict(obs, deterministic=True)
    print(f"step 6a predict OK, action={action}", flush=True)
    sac_env.close()

    print("step 6b: construct a DEFAULT-size TQC (sb3_contrib, no custom "
          "net_arch/n_critics override)", flush=True)
    from sb3_contrib import TQC
    tqc_env = gym.make("FetchPickAndPlace-v3", max_episode_steps=50)
    tqc_model = TQC("MultiInputPolicy", tqc_env, device="cpu")
    print("step 6b OK — default-size TQC construction did not crash", flush=True)
    obs, info = tqc_env.reset(seed=0)
    action, _ = tqc_model.predict(obs, deterministic=True)
    print(f"step 6b predict OK, action={action}", flush=True)
    tqc_env.close()

    print("step 6c: construct TQC with the checkpoint's actual "
          "net_arch=[512,512,512], n_critics=2", flush=True)
    tqc_big_env = gym.make("FetchPickAndPlace-v3", max_episode_steps=50)
    tqc_big_model = TQC(
        "MultiInputPolicy", tqc_big_env, device="cpu",
        policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    )
    print("step 6c OK — large-net_arch TQC construction did not crash", flush=True)
    obs, info = tqc_big_env.reset(seed=0)
    action, _ = tqc_big_model.predict(obs, deterministic=True)
    print(f"step 6c predict OK, action={action}", flush=True)
    tqc_big_env.close()

    return {"status": "all steps completed with no segfault"}


@app.local_entrypoint()
def main():
    result = diagnose_torch_mujoco_segfault.remote()
    print(f"\nResult: {result}")
