"""
MuJoCo reads MUJOCO_GL/PYOPENGL_PLATFORM at import time, so these must be
set before any test module in this directory imports gymnasium/mujoco.
conftest.py is collected first, which guarantees that ordering.
"""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
