"""
tests/run_integration.py

Runs the integration test suite (tests/integration/) inside the Modal
container — it needs mujoco/gymnasium-robotics (and, for the `gpu`-marked
test, torch/peft on an actual GPU), none of which are installed locally.

Usage:
    modal run tests/run_integration.py                # env + reward tests only (CPU, cheap)
    modal run tests/run_integration.py --gpu-tests     # + model_loader test (GPU, loads Qwen2-VL-2B)
"""

import pathlib

import modal

from think_then_act.modal_app import MODEL_CACHE_DIR, app, model_volume, rl_image

TESTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

test_image = (
    rl_image.pip_install("pytest==8.2.2")
    .add_local_dir(str(TESTS_DIR), remote_path="/root/tests")
    .add_local_file(str(REPO_ROOT / "pyproject.toml"), remote_path="/root/pyproject.toml")
)


@app.function(image=test_image, gpu=None, timeout=600)
def run_cpu_tests() -> int:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/integration", "-v", "-m", "integration and not gpu"],
        cwd="/root",
    )
    return result.returncode


@app.function(image=test_image, gpu="T4", volumes={MODEL_CACHE_DIR: model_volume}, timeout=900)
def run_gpu_tests() -> int:
    import os
    import subprocess
    import sys

    env = {**os.environ, "MODEL_CACHE_DIR": MODEL_CACHE_DIR}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/integration", "-v", "-m", "gpu"],
        cwd="/root",
        env=env,
    )
    return result.returncode


@app.local_entrypoint()
def main(gpu_tests: bool = False):
    print("\nRunning CPU integration tests (env + reward, real MuJoCo)...")
    rc = run_cpu_tests.remote()
    if rc != 0:
        raise SystemExit(f"CPU integration tests failed (exit code {rc})")
    print("CPU integration tests passed.")

    if gpu_tests:
        print("\nRunning GPU integration test (loads Qwen2-VL-2B-Instruct)...")
        rc = run_gpu_tests.remote()
        if rc != 0:
            raise SystemExit(f"GPU integration test failed (exit code {rc})")
        print("GPU integration test passed.")
