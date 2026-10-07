"""
think_then_act multi-cube-experiment model_registry

Thin loader for models.yaml -- the single source of truth for "which
checkpoint is which" in this experiment (architecture, task, how it was
trained, its verified completion rate and the exact eval protocol that
number came from). Used by rollout.py so every rollout script stops
re-deciding "is this a BC or PPO checkpoint, which env does it need" for
itself the way every one-off script in this folder's history did.

Usage:
    from model_registry import load, all_models
    entry = load("multicube_stack_ppo_selfimitation_v2")
    for e in all_models():
        print(e.name, e.completion_rate)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "models.yaml")


@dataclass
class ModelEntry:
    name: str
    path: str | None
    task: str | None            # "basic_skill" | "stack2" | None (e.g. the external teacher)
    architecture: str | None    # "mse" | "cvae" | "transformer" | "diffusion" | None
    trained_via: str | None     # "bc" | "ppo" | "dagger" | "ppo_selfimitation" | None
    completion_rate: float
    eval_protocol: dict = field(default_factory=dict)
    notes: str = ""
    superseded: bool = False
    superseded_by: str | None = None


def _load_all() -> dict[str, ModelEntry]:
    import yaml
    with open(_REGISTRY_PATH) as f:
        rows = yaml.safe_load(f)
    entries = {}
    for row in rows:
        entry = ModelEntry(
            name=row["name"],
            path=row.get("path"),
            task=row.get("task"),
            architecture=row.get("architecture"),
            trained_via=row.get("trained_via"),
            completion_rate=float(row["completion_rate"]),
            eval_protocol=row.get("eval_protocol", {}),
            notes=(row.get("notes") or "").strip(),
            superseded=bool(row.get("superseded", False)),
            superseded_by=row.get("superseded_by"),
        )
        entries[entry.name] = entry
    return entries


_CACHE: dict[str, ModelEntry] | None = None


def _entries() -> dict[str, ModelEntry]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_all()
    return _CACHE


def load(name: str) -> ModelEntry:
    entries = _entries()
    if name not in entries:
        raise KeyError(f"No model named {name!r} in {_REGISTRY_PATH}. "
                        f"Available: {sorted(entries)}")
    return entries[name]


def all_models() -> list[ModelEntry]:
    return list(_entries().values())


if __name__ == "__main__":
    for e in all_models():
        flag = " (superseded)" if e.superseded else ""
        print(f"{e.name:38s} {e.completion_rate:6.1%}  task={e.task!s:12s} "
              f"arch={e.architecture!s:12s} via={e.trained_via!s:18s}{flag}")
