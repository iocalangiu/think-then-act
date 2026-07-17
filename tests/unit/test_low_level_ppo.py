"""
Unit tests for training.low_level_ppo — pure PyTorch + synthetic rollout
dicts, no mujoco/gymnasium needed (collection is delegated to
rollout_workers.py, which needs a live env and isn't exercised here — same
split as test_low_level_grpo.py).
"""

import numpy as np
import pytest
import torch

from think_then_act.training.low_level_ppo import LowLevelPPOConfig, LowLevelPPOTrainer

OBS_DIM = 38


def _make_trainer(**overrides) -> LowLevelPPOTrainer:
    config = LowLevelPPOConfig(obs_dim=OBS_DIM, hidden_dim=16, **overrides)
    return LowLevelPPOTrainer(config)


def _make_rollout(trainer, obs, raw_samples_rewards) -> dict:
    """
    raw_samples_rewards: list of (raw_sample, reward) for one episode, in
    order. old_log_prob/value are computed from the TRAINER'S CURRENT
    actor/critic (as real collection would, right before any update this
    iteration) so ratio starts at exactly 1.0.
    """
    steps = []
    with torch.no_grad():
        for raw_sample, reward in raw_samples_rewards:
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            raw_t = torch.from_numpy(raw_sample).unsqueeze(0)
            log_prob, _ = trainer.actor.recompute_log_prob(obs_t, raw_t)
            value = trainer.critic(obs_t)
            steps.append({
                "obs": obs, "raw_sample": raw_sample,
                "old_log_prob": float(log_prob.item()),
                "value": float(value.item()),
                "reward": reward,
            })
    return {
        "steps": steps, "bootstrap_value": 0.0,
        "total_reward": sum(r for _, r in raw_samples_rewards),
        "n_steps": len(steps),
    }


def test_compute_gae_matches_manual_computation_undiscounted_no_bootstrap():
    rewards = np.array([1.0, 2.0, 3.0])
    values  = np.array([0.0, 0.0, 0.0])
    advantages, returns = LowLevelPPOTrainer.compute_gae(
        rewards, values, bootstrap_value=0.0, gamma=1.0, gae_lambda=1.0
    )
    # gamma=lambda=1, values=0 -> GAE reduces to plain undiscounted
    # return-to-go, same as low_level_grpo.py's _reward_to_go.
    np.testing.assert_allclose(advantages, [6.0, 5.0, 3.0])
    np.testing.assert_allclose(returns, [6.0, 5.0, 3.0])


def test_compute_gae_bootstraps_when_truncated_not_terminated():
    rewards = np.array([1.0])
    values  = np.array([0.5])
    adv_with_bootstrap, _ = LowLevelPPOTrainer.compute_gae(
        rewards, values, bootstrap_value=2.0, gamma=0.9, gae_lambda=1.0
    )
    adv_no_bootstrap, _ = LowLevelPPOTrainer.compute_gae(
        rewards, values, bootstrap_value=0.0, gamma=0.9, gae_lambda=1.0
    )
    # delta_0 = r_0 + gamma*bootstrap - v_0 -> larger bootstrap raises the advantage.
    assert adv_with_bootstrap[0] > adv_no_bootstrap[0]
    expected = 1.0 + 0.9 * 2.0 - 0.5
    assert adv_with_bootstrap[0] == pytest.approx(expected)


def test_ppo_step_returns_finite_metrics_with_expected_keys():
    trainer = _make_trainer(n_epochs=2, minibatch_size=8)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    rollouts = [
        _make_rollout(trainer, obs, [
            (np.array([-1.0, 0, 0, 0], dtype=np.float32), -1.0),
            (np.array([-0.5, 0, 0, 0], dtype=np.float32), 0.0),
        ]),
        _make_rollout(trainer, obs, [
            (np.array([1.0, 0, 0, 0], dtype=np.float32), 1.0),
        ]),
    ]

    metrics = trainer.ppo_step(rollouts)

    for key in ("policy_loss", "value_loss", "mean_entropy", "approx_kl",
                "clip_fraction", "mean_reward", "std_reward"):
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert 0.0 <= metrics["clip_fraction"] <= 1.0


def test_ppo_step_d_grip_block_metrics_are_none_when_not_tracked():
    # align_xy/descend/etc. don't surface d_grip_block in info — _make_rollout's
    # synthetic rollouts don't have the keys either, matching that.
    trainer = _make_trainer(n_epochs=1, minibatch_size=8)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    rollouts = [_make_rollout(trainer, obs, [(np.array([0.0, 0, 0, 0], dtype=np.float32), 0.0)])]

    metrics = trainer.ppo_step(rollouts)
    assert metrics["mean_initial_d_grip_block"] is None
    assert metrics["mean_final_d_grip_block"] is None


def test_ppo_step_averages_initial_and_final_d_grip_block_when_present():
    # Only close_gripper's rollouts (via rollout_workers.py) carry these —
    # simulate that by attaching them directly to synthetic rollout dicts.
    trainer = _make_trainer(n_epochs=1, minibatch_size=8)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    rollout_a = _make_rollout(trainer, obs, [(np.array([0.0, 0, 0, 0], dtype=np.float32), 0.0)])
    rollout_a["initial_d_grip_block"] = 0.005
    rollout_a["final_d_grip_block"]   = 0.02
    rollout_b = _make_rollout(trainer, obs, [(np.array([0.0, 0, 0, 0], dtype=np.float32), 0.0)])
    rollout_b["initial_d_grip_block"] = 0.009
    rollout_b["final_d_grip_block"]   = 0.06

    metrics = trainer.ppo_step([rollout_a, rollout_b])
    assert metrics["mean_initial_d_grip_block"] == pytest.approx(0.007)
    assert metrics["mean_final_d_grip_block"] == pytest.approx(0.04)


def test_ppo_step_actually_changes_actor_and_critic_parameters():
    trainer = _make_trainer(n_epochs=2, minibatch_size=8)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    rollouts = [
        _make_rollout(trainer, obs, [
            (np.array([-1.0, 0.2, 0, 0], dtype=np.float32), -1.0),
        ]),
        _make_rollout(trainer, obs, [
            (np.array([1.0, -0.3, 0, 0], dtype=np.float32), 1.0),
        ]),
    ]

    actor_before  = [p.clone() for p in trainer.actor.parameters()]
    critic_before = [p.clone() for p in trainer.critic.parameters()]
    trainer.ppo_step(rollouts)

    actor_changed  = any(not torch.equal(b, a) for b, a in zip(actor_before, trainer.actor.parameters()))
    critic_changed = any(not torch.equal(b, a) for b, a in zip(critic_before, trainer.critic.parameters()))
    assert actor_changed
    assert critic_changed


def test_repeated_ppo_steps_shift_mean_toward_higher_reward_raw_samples():
    """
    Same spirit as low_level_grpo.py's equivalent test: with a reward
    structure that consistently favors higher raw_sample[0], repeated PPO
    updates (recollecting old_log_prob/value fresh each outer iteration,
    as real training would) should move mean_head's output for that obs
    toward higher raw_sample[0].
    """
    torch.manual_seed(0)
    trainer = _make_trainer(lr=1e-2, entropy_coef=0.0, n_epochs=4, minibatch_size=8)

    obs = np.zeros(OBS_DIM, dtype=np.float32)
    raw_samples_and_rewards = [
        (np.array([-1.5, 0, 0, 0], dtype=np.float32), -1.0),
        (np.array([-0.5, 0, 0, 0], dtype=np.float32), -0.3),
        (np.array([0.5, 0, 0, 0], dtype=np.float32), 0.3),
        (np.array([1.5, 0, 0, 0], dtype=np.float32), 1.0),
    ]

    obs_t = torch.from_numpy(obs).unsqueeze(0)
    with torch.no_grad():
        initial_mean0 = trainer.actor.forward(obs_t)[0][0, 0].item()

    for _ in range(100):
        rollouts = [_make_rollout(trainer, obs, [(rs, r)]) for rs, r in raw_samples_and_rewards]
        trainer.ppo_step(rollouts)

    with torch.no_grad():
        final_mean0 = trainer.actor.forward(obs_t)[0][0, 0].item()

    assert final_mean0 > initial_mean0


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    trainer = _make_trainer()
    ckpt_path = str(tmp_path / "low_level_align_xy_ppo.pt")
    trainer.save_checkpoint(ckpt_path)

    reloaded = _make_trainer()
    reloaded.load_checkpoint(ckpt_path)

    obs_t = torch.rand(1, OBS_DIM)
    with torch.no_grad():
        mean_a, log_std_a = trainer.actor.forward(obs_t)
        mean_b, log_std_b = reloaded.actor.forward(obs_t)
        value_a = trainer.critic(obs_t)
        value_b = reloaded.critic(obs_t)

    torch.testing.assert_close(mean_a, mean_b)
    torch.testing.assert_close(log_std_a, log_std_b)
    torch.testing.assert_close(value_a, value_b)
