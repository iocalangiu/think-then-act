"""
Unit tests for training.low_level_grpo — pure PyTorch + synthetic rollout
dicts, no mujoco/gymnasium needed (collect_rollouts is the only method that
needs a live env, and it's not exercised here).
"""

import numpy as np
import pytest
import torch

from think_then_act.training.low_level_grpo import LowLevelGRPOConfig, LowLevelGRPOTrainer

OBS_DIM = 38


def _make_trainer(**overrides) -> LowLevelGRPOTrainer:
    config = LowLevelGRPOConfig(obs_dim=OBS_DIM, hidden_dim=16, **overrides)
    return LowLevelGRPOTrainer(config)


def _make_rollout(obs, raw_sample, reward) -> dict:
    return {
        "steps": [{"obs": obs, "raw_sample": raw_sample, "reward": reward}],
        "total_reward": reward,
        "n_steps": 1,
    }


def test_rollout_log_prob_and_entropy_shapes_and_gradient():
    trainer = _make_trainer()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    raw_sample = np.zeros(4, dtype=np.float32)
    rollout = {
        "steps": [
            {"obs": obs, "raw_sample": raw_sample, "reward": 0.0},
            {"obs": obs, "raw_sample": raw_sample, "reward": 0.0},
        ],
        "total_reward": 0.0, "n_steps": 2,
    }

    log_probs, mean_entropy = trainer._rollout_log_prob_and_entropy(rollout)
    assert log_probs.shape == (2,)
    assert mean_entropy.shape == ()
    assert log_probs.requires_grad
    assert mean_entropy.requires_grad


def test_reward_to_go_undiscounted():
    returns = LowLevelGRPOTrainer._reward_to_go([1.0, 2.0, 3.0], gamma=1.0)
    np.testing.assert_allclose(returns, [6.0, 5.0, 3.0])


def test_reward_to_go_discounted():
    returns = LowLevelGRPOTrainer._reward_to_go([1.0, 2.0, 4.0], gamma=0.5)
    np.testing.assert_allclose(returns, [3.0, 4.0, 4.0])


def test_group_relative_advantages_matches_manual_per_timestep_computation():
    # 3 rollouts, all length 2, return-to-go differs at both t=0 and t=1.
    returns_to_go = [
        np.array([-3.0, -2.0]),
        np.array([0.0, 0.0]),
        np.array([3.0, 2.0]),
    ]
    advantages = LowLevelGRPOTrainer._group_relative_advantages(returns_to_go)

    t0 = np.array([-3.0, 0.0, 3.0])
    t1 = np.array([-2.0, 0.0, 2.0])
    expected_t0 = (t0 - t0.mean()) / (t0.std() + 1e-8)
    expected_t1 = (t1 - t1.mean()) / (t1.std() + 1e-8)

    for i, adv in enumerate(advantages):
        assert adv[0] == pytest.approx(expected_t0[i])
        assert adv[1] == pytest.approx(expected_t1[i])


def test_group_relative_advantages_excludes_terminated_rollouts_from_later_baseline():
    # Rollout 0 terminates after 1 step (e.g. hit its subgoal early);
    # rollouts 1 and 2 run for 2 steps. The t=1 baseline must only be
    # computed from rollouts that actually have a step 1.
    returns_to_go = [
        np.array([5.0]),
        np.array([1.0, -1.0]),
        np.array([1.0, 1.0]),
    ]
    advantages = LowLevelGRPOTrainer._group_relative_advantages(returns_to_go)

    assert len(advantages[0]) == 1
    assert len(advantages[1]) == 2
    assert len(advantages[2]) == 2

    t1 = np.array([-1.0, 1.0])
    expected_t1 = (t1 - t1.mean()) / (t1.std() + 1e-8)
    assert advantages[1][1] == pytest.approx(expected_t1[0])
    assert advantages[2][1] == pytest.approx(expected_t1[1])


def test_grpo_step_handles_ragged_rollout_lengths_within_a_group():
    """Regression guard: one rollout in a group ending early (subgoal hit
    before max_episode_steps) must not break advantage computation or the
    backward pass — this is the normal case in real training, not an edge
    case."""
    trainer = _make_trainer()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    raw_sample = np.zeros(4, dtype=np.float32)

    def _rollout(rewards):
        return {
            "steps": [{"obs": obs, "raw_sample": raw_sample, "reward": r} for r in rewards],
            "total_reward": sum(rewards), "n_steps": len(rewards),
        }

    group = [_rollout([-1.0]), _rollout([-0.5, -0.2]), _rollout([-0.3, -0.1, 0.0])]
    metrics = trainer.grpo_step([group])

    for key in ("loss", "policy_loss", "mean_entropy", "mean_reward",
                "std_reward", "mean_abs_adv", "avg_within_std"):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_grpo_step_returns_finite_metrics_with_expected_keys():
    trainer = _make_trainer()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    group = [
        _make_rollout(obs, np.array([-1.0, 0, 0, 0], dtype=np.float32), -1.0),
        _make_rollout(obs, np.array([0.0, 0, 0, 0], dtype=np.float32), 0.0),
        _make_rollout(obs, np.array([1.0, 0, 0, 0], dtype=np.float32), 1.0),
    ]

    metrics = trainer.grpo_step([group])

    for key in ("loss", "policy_loss", "mean_entropy", "mean_reward",
                "std_reward", "mean_abs_adv", "avg_within_std"):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_grpo_step_advantage_normalization_matches_manual_computation():
    trainer = _make_trainer()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    rewards = [-2.0, 0.0, 2.0]
    group = [
        _make_rollout(obs, np.array([r, 0, 0, 0], dtype=np.float32), r)
        for r in rewards
    ]

    metrics = trainer.grpo_step([group])

    expected_mean_reward = float(np.mean(rewards))
    expected_std_reward  = float(np.std(rewards))
    assert metrics["mean_reward"] == pytest.approx(expected_mean_reward)
    assert metrics["std_reward"]  == pytest.approx(expected_std_reward)
    assert metrics["avg_within_std"] == pytest.approx(expected_std_reward, abs=1e-6)


def test_grpo_step_actually_changes_policy_parameters():
    trainer = _make_trainer()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    group = [
        _make_rollout(obs, np.array([-1.0, 0.2, 0, 0], dtype=np.float32), -1.0),
        _make_rollout(obs, np.array([1.0, -0.3, 0, 0], dtype=np.float32), 1.0),
    ]

    before = [p.clone() for p in trainer.policy.parameters()]
    trainer.grpo_step([group])
    after = list(trainer.policy.parameters())

    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed


def test_repeated_grpo_steps_shift_mean_toward_higher_reward_raw_samples():
    """
    Meaningful correctness check for the whole gradient pipeline (advantage
    computation + backward + optimizer step), not just "does it run": with a
    fixed obs and a reward structure that consistently favors higher
    raw_sample[0] values, repeated grpo_step calls should move the policy's
    mean output for that obs toward higher raw_sample[0] — same spirit as
    the toy REINFORCE test in test_subgoal_policy.py, but exercising
    grpo_step itself (group-relative advantage math included).
    """
    torch.manual_seed(0)
    trainer = _make_trainer(lr=1e-2, entropy_coef=0.0)

    obs = np.zeros(OBS_DIM, dtype=np.float32)
    raw_samples_and_rewards = [
        (np.array([-1.5, 0, 0, 0], dtype=np.float32), -1.0),
        (np.array([-0.5, 0, 0, 0], dtype=np.float32), -0.3),
        (np.array([0.5, 0, 0, 0], dtype=np.float32), 0.3),
        (np.array([1.5, 0, 0, 0], dtype=np.float32), 1.0),
    ]

    def make_group():
        return [_make_rollout(obs, rs, r) for rs, r in raw_samples_and_rewards]

    obs_t = torch.from_numpy(obs).unsqueeze(0)
    with torch.no_grad():
        initial_mean0 = trainer.policy.forward(obs_t)[0][0, 0].item()

    for _ in range(100):
        trainer.grpo_step([make_group()])

    with torch.no_grad():
        final_mean0 = trainer.policy.forward(obs_t)[0][0, 0].item()

    assert final_mean0 > initial_mean0


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    trainer = _make_trainer()
    ckpt_path = str(tmp_path / "low_level_align_xy.pt")
    trainer.save_checkpoint(ckpt_path)

    reloaded = _make_trainer()
    reloaded.load_checkpoint(ckpt_path)

    obs_t = torch.rand(1, OBS_DIM)
    with torch.no_grad():
        mean_a, log_std_a = trainer.policy.forward(obs_t)
        mean_b, log_std_b = reloaded.policy.forward(obs_t)

    torch.testing.assert_close(mean_a, mean_b)
    torch.testing.assert_close(log_std_a, log_std_b)
