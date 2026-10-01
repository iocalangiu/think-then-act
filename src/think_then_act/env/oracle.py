"""
think_then_act.env.oracle

Scripted pick-and-place oracle for FetchPickAndPlace-v3. Originally lived
inline in scripts/generate_sft_data.py (VLM SFT data generation); moved
here (2026-07-14) so training/subgoal_env.py's per-subgoal episode setup
can reuse the SAME proven APPROACH->GRASP->CARRY logic to construct valid
starting states for subgoals that only make sense mid-sequence (see
env/setup.py's init_episode_before_subgoal) — writing a second scripted
policy from scratch for that would just be re-deriving this one, with a
real chance of drifting out of sync with it.

Pure function, no mujoco/gymnasium import — safe to call from anywhere
that already has an obs/achieved_goal/desired_goal in hand.
"""

from __future__ import annotations

import numpy as np


def oracle_action(obs_arr, achieved_goal, desired_goal, carrying: bool = False,
                   block_resting_z: float = 0.425):
    """
    Scripted heuristic for FetchPickAndPlace. Returns (action, phase, carrying).

    `carrying` is stateful hysteresis: once the block is grasped and lifted, stay
    in CARRY until the block clearly escapes the gripper (d_3d > 0.12).  Without
    this, CARRY moves toward the target (which is at table height), descending the
    block below the block_lifted threshold and causing GRASP/CARRY oscillation.

    block_resting_z: the block's own resting-CENTER height for THIS episode
    (table surface + half its own height) — defaults to 0.425, correct only
    for the original fixed 5cm cube every caller assumed before block-size
    randomization existed. Callers spawning a randomized-height block (see
    env/block_randomization.py) must pass the actual per-episode value, or
    `block_lifted` below silently misfires: a short block would need a much
    bigger rise than intended to cross a fixed absolute threshold, a tall
    one would cross it almost immediately, in both cases scrambling the
    APPROACH->GRASP->CARRY phase transitions this heuristic (and anything
    that fast-forwards through them, e.g. init_episode_before_subgoal) rely
    on. Fixed 2026-08-10 (see hierarchical_architecture memory, Stage 4) —
    previously an unparameterized `block_z > 0.45`.

    Grip convention: +1.0 = OPEN fingers, -1.0 = CLOSE fingers.
    """
    rel          = obs_arr[6:9]              # object_rel_pos = block - gripper
    finger_width = float(np.sum(obs_arr[9:11]))

    d_3d = float(np.linalg.norm(rel))
    d_xy = float(np.linalg.norm(rel[:2]))

    block_z = float(achieved_goal[2])
    grip_z  = block_z - float(rel[2])

    # 0.025 preserves the ORIGINAL fixed-cube margin exactly (0.45 - 0.425)
    # — not a new number, just the same margin re-expressed relative to
    # this episode's own resting height instead of baked into an absolute
    # constant.
    block_lifted  = block_z > block_resting_z + 0.025
    # Once carrying, only exit if block escapes (wider tolerance than initial grasp).
    is_grasped    = (block_lifted and d_3d < 0.10) or (carrying and d_3d < 0.12)
    at_block_zone = grip_z <= block_z + 0.10

    if is_grasped:
        phase     = "CARRY"
        carrying  = True
        direction = np.array(desired_goal) - np.array(achieved_goal)
        grip      = -1.0                              # CLOSE — keep block grasped
    elif d_3d < 0.10 or at_block_zone:
        phase    = "GRASP"
        carrying = False
        if grip_z > block_z + 0.025:
            direction = np.array(rel)                 # move toward block (XY+Z)
            grip      = 1.0                           # OPEN during descent
        elif finger_width > 0.07:
            direction = np.zeros(3)                   # stay still, close fingers
            grip      = -1.0                          # CLOSE
        else:
            direction = np.array([0.0, 0.0, 1.0])    # lift
            grip      = -1.0                          # CLOSE — maintain grasp
    elif d_xy > 0.1:
        phase     = "APPROACH"
        carrying  = False
        direction = np.array([rel[0], rel[1], 0.0])   # lateral only
        grip      = 1.0                               # OPEN
    else:
        phase     = "APPROACH"
        carrying  = False
        direction = np.array([rel[0], rel[1], rel[2]])  # full 3D descent
        grip      = 1.0                               # OPEN

    norm  = float(np.linalg.norm(direction)) + 1e-8
    scale = min(1.0, float(np.linalg.norm(direction)) / 0.05)
    dx, dy, dz = (direction / norm) * scale

    return np.clip([dx, dy, dz, grip], -1.0, 1.0).astype(np.float32), phase, carrying
