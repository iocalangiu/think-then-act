"""
debug_block_size_render.py

One-off: render a single frame with the ORIGINAL fixed 5cm cube next to one
with block-size randomization on, same seed/camera, to directly compare
scale — a verify_close_gripper_grasp.py video (2026-08-11) showed what
LOOKS like a wildly oversized block, but that could also just be a
misjudged camera framing (the table itself is a fairly large white box).
Also prints the actual geom_size/reported dims numerically, not just a
picture, so this is conclusive either way.

Run with:
    modal run scripts/debug_block_size_render.py
"""

import modal
from think_then_act.modal_app import app, rl_image


@app.function(image=rl_image, gpu=None, timeout=120)
def debug_block_size_render(seed: int = 0) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import mujoco
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, BLOCK_BODY_NAME
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.env.block_randomization import get_block_dims

    results = {}
    for label, randomize, forced_size in [
        ("fixed_cube", False, None),
        ("randomized", True, None),
        ("huge_forced", False, 0.30),   # deliberately huge, applied AFTER init_random_episode,
                                          # to test whether ANY geom_size change ever renders at all
    ]:
        env = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=60)
        )
        setup_env(env)
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        obs, ok = init_random_episode(env, rng, randomize_block_size=randomize)

        raw = env.unwrapped
        gid = None
        body_id = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, BLOCK_BODY_NAME)
        all_geoms_for_body = [g for g in range(raw.model.ngeom) if raw.model.geom_bodyid[g] == body_id]
        print(f"  {label}: ALL geoms for body {BLOCK_BODY_NAME!r} = {all_geoms_for_body}  "
              f"(rgba each: {[raw.model.geom_rgba[g].tolist() for g in all_geoms_for_body]})  "
              f"(type each: {[int(raw.model.geom_type[g]) for g in all_geoms_for_body]})")
        for g in range(raw.model.ngeom):
            if raw.model.geom_bodyid[g] == body_id:
                gid = g
                break

        if forced_size is not None:
            raw.model.geom_size[gid] = [forced_size / 2, forced_size / 2, forced_size / 2]
            mujoco.mj_forward(raw.model, raw.data)
            env.step(np.zeros(4, dtype=np.float32))  # force a render-side refresh too

        geom_size = raw.model.geom_size[gid].copy().tolist()
        dims = get_block_dims(raw.model) if forced_size is None else {"forced_full_size": forced_size}
        xpos = raw.data.xpos[body_id].copy().tolist()
        achieved_goal = obs["achieved_goal"].tolist() if isinstance(obs, dict) else None

        frame = env.last_frame()
        import io, base64
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        print(f"  {label}: ok={ok}  geom_size(half-extents)={geom_size}  "
              f"dims={dims}  xpos={xpos}  achieved_goal={achieved_goal}")
        results[label] = {
            "ok": ok, "geom_size": geom_size, "dims": dims,
            "xpos": xpos, "achieved_goal": achieved_goal, "png_b64": png_b64,
        }
        env.close()

    return {"status": "PASS", "results": results}


@app.local_entrypoint()
def main(seed: int = 0):
    import base64, os
    result = debug_block_size_render.remote(seed=seed)
    out_dir = "./artifacts/debug_block_size_render"
    os.makedirs(out_dir, exist_ok=True)
    for label, row in result["results"].items():
        path = os.path.join(out_dir, f"{label}.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(row["png_b64"]))
        print(f"  saved -> {path}")
    print(f"\nDone.")
