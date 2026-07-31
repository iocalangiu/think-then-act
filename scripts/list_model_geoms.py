"""
list_model_geoms.py

Standalone dump of every part of the FetchPickAndPlace-v3 MuJoCo model —
pulled out of validate_collision_labels.py, which needed the same listing
internally to find the real table geom (see bugs_and_fixes memory,
2026-07-12: "geom22" turned out to be an unnamed geom whose body is really
called "table0" — this script is the tool that answers "what is X?" without
having to run a full episode to find out).

Prints two views:
    1. Every geom: name (or a fallback label if unnamed in the MJCF), its
       parent body, shape type, size, and world position.
    2. The body kinematic tree (parent -> child indentation) — geoms are
       just collision shapes; bodies are the actual "parts" (robot links,
       the table, the block) and how they're connected to each other.

No GPU needed — just needs one reset to populate world positions.

Run with:
    modal run scripts/list_model_geoms.py
    modal run scripts/list_model_geoms.py --filter robot0:shoulder
"""

import modal
from think_then_act.modal_app import app, rl_image


@app.function(image=rl_image, gpu=None, timeout=300)
def list_model_geoms(filter_substring: str = "") -> dict:
    import os
    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import mujoco
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env
    from think_then_act.env.wrapper import ObservationHarness

    env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array")
    )
    setup_env(env)
    env.reset(seed=0)
    raw = env.unwrapped

    _GEOM_TYPE_NAMES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule",
                        4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}
    needle = filter_substring.lower()

    print("\n" + "=" * 70)
    print("  GEOMS (collision/visual shapes)")
    print("=" * 70)
    geom_lines = []
    for gid in range(raw.model.ngeom):
        name = mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid} (unnamed in MJCF)"
        body_id = raw.model.geom_bodyid[gid]
        body_name = mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"
        gtype = _GEOM_TYPE_NAMES.get(int(raw.model.geom_type[gid]), str(int(raw.model.geom_type[gid])))
        xpos = raw.data.geom_xpos[gid]
        size = raw.model.geom_size[gid]
        line = (f"  {name:28s} body={body_name:25s} type={gtype:8s} "
                f"xpos=[{xpos[0]:6.3f},{xpos[1]:6.3f},{xpos[2]:6.3f}] size={size}")
        if needle and needle not in name.lower() and needle not in body_name.lower():
            continue
        print(line)
        geom_lines.append(line)

    print("\n" + "=" * 70)
    print("  BODY KINEMATIC TREE (parent -> children)")
    print("=" * 70)

    children_by_parent: dict = {}
    for bid in range(raw.model.nbody):
        parent_id = raw.model.body_parentid[bid]
        children_by_parent.setdefault(int(parent_id), []).append(bid)

    body_lines = []

    def print_tree(bid: int, depth: int = 0) -> None:
        name = mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"
        xpos = raw.data.xpos[bid]
        line = f"  {'  ' * depth}{name:30s} xpos=[{xpos[0]:6.3f},{xpos[1]:6.3f},{xpos[2]:6.3f}]"
        show = not needle or needle in name.lower()
        if show:
            print(line)
            body_lines.append(line)
        for child_id in children_by_parent.get(bid, []):
            if child_id != bid:   # world body (id 0) is its own parent in MuJoCo
                print_tree(child_id, depth + 1)

    print_tree(0)   # id 0 is always the world body

    env.close()
    print(f"\n  {raw.model.ngeom} geoms, {raw.model.nbody} bodies total.")

    return {"status": "PASS", "n_geoms": int(raw.model.ngeom), "n_bodies": int(raw.model.nbody)}


@app.local_entrypoint()
def main(filter: str = ""):
    result = list_model_geoms.remote(filter_substring=filter)
    print(f"\nDone. {result['n_geoms']} geoms, {result['n_bodies']} bodies.")
