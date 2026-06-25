"""
obs_wrapper.py

Custom gymnasium.Wrapper that augments FetchPickAndPlace-v3 (or any
dict-observation robotics env) with:

  1. RGB frame capture after every reset() and step() call.
     Requires the inner env to be created with render_mode="rgb_array".

  2. Structured per-step recording: frame + full state dict + reward + flags.
     Stored in self.episode_log as a list of dicts (one per timestep).

  3. Helpers to extract frames, metadata, or a stacked tensor from the log.

This wrapper sits between the raw gymnasium env and everything else in the
project (policy actor, reward computer, training loop).  It has no Modal
dependencies so it can be unit-tested locally without a cloud GPU.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class ObservationHarness(gym.Wrapper):
    """
    Wraps a dict-observation robotics env to capture RGB frames + state.

    Usage:
        env = gym.make("FetchPickAndPlace-v3", render_mode="rgb_array")
        env = ObservationHarness(env)
        obs, info = env.reset(seed=0)
        for _ in range(10):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
        frames = env.frames_as_array()   # shape (T, H, W, 3)
        log    = env.metadata_only()     # JSON-safe list of dicts
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.episode_log: list[dict] = []
        self._step_idx: int = 0

    # ------------------------------------------------------------------
    # gymnasium.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict, dict]:
        obs, info = self.env.reset(**kwargs)
        self._step_idx = 0
        self.episode_log = []
        frame = self._capture_frame()
        self._record(
            step=0, obs=obs, frame=frame,
            action=None, reward=None,
            terminated=False, truncated=False, info=info,
        )
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_idx += 1
        frame = self._capture_frame()
        self._record(
            step=self._step_idx, obs=obs, frame=frame,
            action=action, reward=float(reward),
            terminated=bool(terminated), truncated=bool(truncated), info=info,
        )
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _capture_frame(self) -> np.ndarray:
        """
        Call env.render() and return an (H, W, 3) uint8 RGB array.
        Returns a black frame if the renderer returns None (should not
        happen with render_mode="rgb_array", but guards against it).
        """
        frame = self.env.render()
        if frame is None or not isinstance(frame, np.ndarray):
            # Fall back to a zero frame so the log entry is never missing.
            h, w = 480, 480
            return np.zeros((h, w, 3), dtype=np.uint8)
        return frame.astype(np.uint8)

    def _record(
        self,
        step: int,
        obs: dict,
        frame: np.ndarray,
        action,
        reward,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> None:
        self.episode_log.append({
            # Timestep index (0 = after reset, 1+ = after each step)
            "step"         : step,
            # Raw pixel data — (H, W, 3) uint8 numpy array
            "frame"        : frame,
            "frame_shape"  : list(frame.shape),
            # State vectors — all converted to plain Python lists for
            # easy JSON serialisation and VLM prompt formatting
            "observation"  : obs["observation"].tolist(),   # len 25
            "achieved_goal": obs["achieved_goal"].tolist(), # len 3, object XYZ
            "desired_goal" : obs["desired_goal"].tolist(),  # len 3, target XYZ
            # Action that caused this state (None at step 0)
            "action"       : action.tolist() if action is not None else None,
            # Scalar reward and episode flags
            "reward"       : reward,
            "terminated"   : terminated,
            "truncated"    : truncated,
            "is_success"   : bool(info.get("is_success", False)),
        })

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def frames(self) -> list[np.ndarray]:
        """Return every captured frame as a list of (H, W, 3) uint8 arrays."""
        return [entry["frame"] for entry in self.episode_log]

    def frames_as_array(self) -> np.ndarray:
        """
        Stack all frames into a single (T, H, W, 3) uint8 tensor.
        T = number of timesteps including the initial reset frame.
        """
        return np.stack(self.frames(), axis=0)

    def metadata_only(self) -> list[dict]:
        """
        Return the episode log with raw frame arrays removed.
        Safe to JSON-serialise and return from a Modal function.
        """
        return [
            {k: v for k, v in entry.items() if k != "frame"}
            for entry in self.episode_log
        ]

    def last_frame(self) -> np.ndarray:
        """Convenience: the most recently captured frame."""
        return self.episode_log[-1]["frame"]

    def episode_summary(self) -> dict:
        """High-level stats about the completed episode."""
        rewards = [e["reward"] for e in self.episode_log if e["reward"] is not None]
        return {
            "total_steps"   : len(self.episode_log) - 1,  # exclude reset frame
            "total_reward"  : sum(rewards),
            "any_success"   : any(e["is_success"] for e in self.episode_log),
            "frame_shape"   : self.episode_log[0]["frame_shape"],
            "frame_dtype"   : str(self.episode_log[0]["frame"].dtype),
            "pixel_min"     : int(self.episode_log[0]["frame"].min()),
            "pixel_max"     : int(self.episode_log[0]["frame"].max()),
        }
