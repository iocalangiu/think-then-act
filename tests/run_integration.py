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


def _run_and_log(args: list, log_name: str, extra_env: dict = None) -> int:
    """
    Run pytest, capture combined stdout+stderr in full (not just what scrolls
    by in the terminal), write it to /model-cache/logs/<log_name> and commit
    the volume — so `modal volume get` pulls the complete output, including
    full tracebacks, without anyone needing to copy-paste a truncated
    terminal view.
    """
    import os
    import subprocess
    import sys

    env = {**os.environ, **(extra_env or {})}

    # Snapshot actually-installed versions of the packages this project pins,
    # so a version-mismatch failure is provable from the log instead of
    # inferred from a traceback (see bugs_and_fixes memory, 2026-07-11:
    # a pin re-added after this same failure didn't change the outcome last
    # time, so this run needs to show what's ACTUALLY resolved, not assumed).
    pip_list = subprocess.run(
        [sys.executable, "-m", "pip", "list"],
        capture_output=True, text=True,
    ).stdout
    watched = {"gymnasium", "gymnasium-robotics", "mujoco"}
    versions_snapshot = "\n".join(
        line for line in pip_list.splitlines()
        if line.split()[0].lower() in watched
    ) if pip_list else "(pip list produced no output)"

    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + args,
        cwd="/root", env=env,
        capture_output=True, text=True,
    )
    output = (
        "=== Installed package versions (watched) ===\n"
        f"{versions_snapshot}\n\n"
        "=== pytest output ===\n"
        + result.stdout + result.stderr
    )
    print(output)   # still visible live in `modal run` output

    log_dir = os.path.join(MODEL_CACHE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    with open(log_path, "w") as f:
        f.write(output)
    model_volume.commit()
    print(f"\n[log] full output written -> {log_path}")

    return result.returncode


@app.function(image=test_image, gpu=None, volumes={MODEL_CACHE_DIR: model_volume}, timeout=600)
def run_cpu_tests() -> int:
    return _run_and_log(
        ["tests/integration", "-v", "-m", "integration and not gpu"],
        log_name="integration_test_cpu.log",
    )


@app.function(image=test_image, gpu="T4", volumes={MODEL_CACHE_DIR: model_volume}, timeout=900)
def run_gpu_tests() -> int:
    return _run_and_log(
        ["tests/integration", "-v", "-m", "gpu"],
        log_name="integration_test_gpu.log",
        extra_env={"MODEL_CACHE_DIR": MODEL_CACHE_DIR},
    )


@app.local_entrypoint()
def main(gpu_tests: bool = False):
    print("\nRunning CPU integration tests (env + reward, real MuJoCo)...")
    rc = run_cpu_tests.remote()
    print(f"\nDownload full log with:")
    print(f"  python3 -m modal volume get --force rl-harness-model-cache "
          f"logs/integration_test_cpu.log ./artifacts/logs/integration_test_cpu.log")
    if rc != 0:
        raise SystemExit(f"CPU integration tests failed (exit code {rc}) — see log above.")
    print("CPU integration tests passed.")

    if gpu_tests:
        print("\nRunning GPU integration test (loads Qwen2-VL-2B-Instruct)...")
        rc = run_gpu_tests.remote()
        print(f"\nDownload full log with:")
        print(f"  python3 -m modal volume get --force rl-harness-model-cache "
              f"logs/integration_test_gpu.log ./artifacts/logs/integration_test_gpu.log")
        if rc != 0:
            raise SystemExit(f"GPU integration test failed (exit code {rc}) — see log above.")
        print("GPU integration test passed.")
