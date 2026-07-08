"""
Unit tests for scripts/compare_runs.py's pure log-parsing logic.

compare_runs.py is a standalone CLI script (not part of the installed
think_then_act package), so it's loaded directly from its file path.
"""

import importlib.util
import json
import pathlib

import pytest

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "compare_runs.py"


@pytest.fixture(scope="module")
def compare_runs():
    spec = importlib.util.spec_from_file_location("compare_runs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_groups_records_by_type_and_iteration(compare_runs, tmp_path):
    log_path = tmp_path / "metrics.jsonl"
    _write_jsonl(log_path, [
        {"type": "train", "iteration": 0, "mean_reward": -10.0},
        {"type": "train", "iteration": 1, "mean_reward": -8.0},
        {"type": "eval", "iteration": 0, "mean_return": -5.0, "success_rate": 0.2},
    ])

    by_type = compare_runs.load(str(log_path))

    assert set(by_type["train"].keys()) == {0, 1}
    assert by_type["train"][0]["mean_reward"] == -10.0
    assert by_type["eval"][0]["success_rate"] == 0.2


def test_per_step_reward_uses_mean_reward_when_present(compare_runs):
    record = {"mean_reward": -45.0}
    assert compare_runs.per_step_reward(record, max_steps=45) == pytest.approx(-1.0)


def test_per_step_reward_falls_back_to_mean_return(compare_runs):
    record = {"mean_return": -20.0}
    assert compare_runs.per_step_reward(record, max_steps=20) == pytest.approx(-1.0)
