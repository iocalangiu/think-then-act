"""
Unit tests for training.low_level_ppo_recurrent — pure PyTorch + synthetic
rollout dicts, no mujoco/gymnasium needed (collection is delegated to
rollout_workers.py's recurrent-specific functions, not exercised here —
same split as test_low_level_ppo.py).
"""

import numpy as np
import pytest
import torch

from think_then_act.training.low_level_ppo_recurrent import (
    RecurrentLowLevelPPOConfig, RecurrentLowLevelPPOTrainer,
)

OBS_DIM = 43
ACTION_DIM = 4


def _make_trainer(**overrides) -> RecurrentLowLevelPPOTrainer:
    config = RecurrentLowLevelPPOConfig(obs_dim=OBS_DIM, hidden_dim=16, rnn_hidden_size=8, **overrides)
    return RecurrentLowLevelPPOTrainer(config)


def _make_rollout(trainer, obs, raw_samples_rewards) -> dict:
    """
    raw_samples_rewards: list of (raw_sample, reward) for one episode, in
    order. old_log_prob/value are computed from the TRAINER'S CURRENT
    actor/critic, threading hidden state step-by-step exactly as
    rollout_workers.py's _run_episode_recurrent does (so ratio starts at
    exactly 1.0, and the per-step hidden-state evolution matches what
    ppo_step's whole-sequence-from-zero recompute later reproduces).
    """
    steps = []
    actor_h = None
    critic_h = None
    with torch.no_grad():
        for raw_sample, reward in raw_samples_rewards:
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            raw_t = torch.from_numpy(raw_sample).unsqueeze(0)
            log_prob, _, actor_h = trainer.actor.recompute_log_prob(obs_t, raw_t, actor_h)
            value, critic_h = trainer.critic(obs_t, critic_h)
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


def test_ppo_step_returns_finite_metrics_with_expected_keys():
    trainer = _make_trainer(n_epochs=2, episodes_per_minibatch=8)
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
                "clip_fraction", "mean_reward", "std_reward", "force_onset_rate"):
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    # Synthetic rollouts carry none of the perturbation-diagnostic fields —
    # these must degrade to None, not raise.
    assert metrics["mean_discrepancy_norm"] is None
    assert metrics["mean_recovery_steps"] is None
    assert metrics["force_onset_rate"] == 0.0


def test_ppo_step_handles_unequal_length_episodes_in_one_minibatch():
    trainer = _make_trainer(n_epochs=1, episodes_per_minibatch=8)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    short_ep = _make_rollout(trainer, obs, [(np.array([0.1, 0, 0, 0], dtype=np.float32), 0.5)])
    long_ep = _make_rollout(trainer, obs, [
        (np.array([0.1, 0, 0, 0], dtype=np.float32), 0.1),
        (np.array([0.2, 0, 0, 0], dtype=np.float32), 0.2),
        (np.array([0.3, 0, 0, 0], dtype=np.float32), 0.3),
        (np.array([0.4, 0, 0, 0], dtype=np.float32), 0.4),
        (np.array([0.5, 0, 0, 0], dtype=np.float32), 0.5),
    ])

    metrics = trainer.ppo_step([short_ep, long_ep])
    for key in ("policy_loss", "value_loss", "mean_entropy"):
        assert np.isfinite(metrics[key])


def test_mean_discrepancy_norm_and_force_onset_rate_are_averaged_when_present():
    trainer = _make_trainer(n_epochs=1, episodes_per_minibatch=8)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    rollout_a = _make_rollout(trainer, obs, [(np.array([0.0, 0, 0, 0], dtype=np.float32), 0.0)])
    rollout_a["mean_discrepancy_norm"] = 0.02
    rollout_a["had_force_onset"] = True
    rollout_a["recovery_steps"] = 4
    rollout_b = _make_rollout(trainer, obs, [(np.array([0.0, 0, 0, 0], dtype=np.float32), 0.0)])
    rollout_b["mean_discrepancy_norm"] = 0.04
    rollout_b["had_force_onset"] = False
    rollout_b["recovery_steps"] = None   # never recovered — must be excluded from the mean, not treated as 0

    metrics = trainer.ppo_step([rollout_a, rollout_b])
    assert metrics["mean_discrepancy_norm"] == pytest.approx(0.03)
    assert metrics["force_onset_rate"] == pytest.approx(0.5)
    assert metrics["mean_recovery_steps"] == pytest.approx(4.0)


def test_masked_losses_ignore_padded_region_regardless_of_its_content():
    """
    Direct regression test for the masking behavior _compute_masked_losses
    implements: two episodes of length 2 and 5 batched together (T_max=5)
    must produce IDENTICAL policy_loss/value_loss/entropy_mean/approx_kl/
    clip_fraction whether the padded region (steps 2:5 of the short
    episode) is filled with zeros or with wildly out-of-range "garbage"
    values — the mask must exclude that region from every loss term
    entirely, not just dilute its contribution.
    """
    trainer = _make_trainer()
    B, T_max = 2, 5
    real_T = [2, 5]

    def build(pad_value: float):
        torch.manual_seed(0)
        obs = torch.zeros(B, T_max, OBS_DIM)
        raw = torch.zeros(B, T_max, ACTION_DIM)
        old_lp = torch.zeros(B, T_max)
        adv = torch.zeros(B, T_max)
        ret = torch.zeros(B, T_max)
        mask = torch.zeros(B, T_max)
        for b, T in enumerate(real_T):
            obs[b, :T] = torch.rand(T, OBS_DIM)
            raw[b, :T] = torch.rand(T, ACTION_DIM) * 2 - 1
            old_lp[b, :T] = torch.rand(T)
            adv[b, :T] = torch.rand(T)
            ret[b, :T] = torch.rand(T)
            mask[b, :T] = 1.0
            if T < T_max:
                obs[b, T:] = pad_value
                raw[b, T:] = pad_value
                old_lp[b, T:] = pad_value
                adv[b, T:] = pad_value
                ret[b, T:] = pad_value
        return obs, raw, old_lp, adv, ret, mask

    obs_zero, raw_zero, old_lp_zero, adv_zero, ret_zero, mask_zero = build(pad_value=0.0)
    obs_huge, raw_huge, old_lp_huge, adv_huge, ret_huge, mask_huge = build(pad_value=1e6)

    # Sanity: the two builds really do share identical REAL content and
    # differ only in the padded region — otherwise this test would prove
    # nothing.
    torch.testing.assert_close(obs_zero[mask_zero.bool()], obs_huge[mask_huge.bool()])
    assert not torch.equal(obs_zero, obs_huge)   # padded region genuinely differs

    losses_zero = trainer._compute_masked_losses(obs_zero, raw_zero, old_lp_zero, adv_zero, ret_zero, mask_zero, clip_eps=0.2)
    losses_huge = trainer._compute_masked_losses(obs_huge, raw_huge, old_lp_huge, adv_huge, ret_huge, mask_huge, clip_eps=0.2)

    torch.testing.assert_close(losses_zero["policy_loss"], losses_huge["policy_loss"])
    torch.testing.assert_close(losses_zero["value_loss"], losses_huge["value_loss"])
    torch.testing.assert_close(losses_zero["entropy_mean"], losses_huge["entropy_mean"])
    assert losses_zero["approx_kl"] == pytest.approx(losses_huge["approx_kl"])
    assert losses_zero["clip_fraction"] == pytest.approx(losses_huge["clip_fraction"])
    # And every value must be finite — if masking were broken, the 1e6
    # garbage would blow these up (or NaN them via the GRU/tanh chain).
    for v in losses_huge.values():
        val = v.item() if torch.is_tensor(v) else v
        assert np.isfinite(val)


def test_critic_warmup_step_never_touches_actor_parameters():
    trainer = _make_trainer(n_epochs=3, episodes_per_minibatch=8)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    rollouts = [
        _make_rollout(trainer, obs, [(np.array([0.1, 0, 0, 0], dtype=np.float32), 0.5)]),
        _make_rollout(trainer, obs, [(np.array([-0.2, 0, 0, 0], dtype=np.float32), -0.3)]),
    ]

    actor_before = [p.clone() for p in trainer.actor.parameters()]
    trainer.critic_warmup_step(rollouts)
    actor_unchanged = all(torch.equal(b, a) for b, a in zip(actor_before, trainer.actor.parameters()))
    assert actor_unchanged, "critic_warmup_step must never update actor parameters"


def test_critic_warmup_step_changes_critic_parameters():
    trainer = _make_trainer(n_epochs=3, episodes_per_minibatch=8)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    rollouts = [
        _make_rollout(trainer, obs, [(np.array([0.1, 0, 0, 0], dtype=np.float32), 0.5)]),
        _make_rollout(trainer, obs, [(np.array([-0.2, 0, 0, 0], dtype=np.float32), -0.3)]),
    ]

    critic_before = [p.clone() for p in trainer.critic.parameters()]
    trainer.critic_warmup_step(rollouts)
    critic_changed = any(not torch.equal(b, a) for b, a in zip(critic_before, trainer.critic.parameters()))
    assert critic_changed


def test_critic_warmup_step_reduces_value_loss_over_repeated_calls():
    torch.manual_seed(0)
    trainer = _make_trainer(lr=1e-2, n_epochs=5, episodes_per_minibatch=8)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    rollouts = [
        _make_rollout(trainer, obs, [
            (np.array([0.1, 0, 0, 0], dtype=np.float32), 1.0),
            (np.array([0.2, 0, 0, 0], dtype=np.float32), 1.0),
        ]),
        _make_rollout(trainer, obs, [
            (np.array([-0.1, 0, 0, 0], dtype=np.float32), -1.0),
        ]),
    ]

    losses = [trainer.critic_warmup_step(rollouts)["value_loss"] for _ in range(10)]
    assert losses[-1] < losses[0]
    assert all(np.isfinite(v) for v in losses)


def test_ppo_step_actually_changes_actor_and_critic_parameters():
    trainer = _make_trainer(n_epochs=2, episodes_per_minibatch=8)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    rollouts = [
        _make_rollout(trainer, obs, [(np.array([-1.0, 0.2, 0, 0], dtype=np.float32), -1.0)]),
        _make_rollout(trainer, obs, [(np.array([1.0, -0.3, 0, 0], dtype=np.float32), 1.0)]),
    ]

    actor_before  = [p.clone() for p in trainer.actor.parameters()]
    critic_before = [p.clone() for p in trainer.critic.parameters()]
    trainer.ppo_step(rollouts)

    actor_changed  = any(not torch.equal(b, a) for b, a in zip(actor_before, trainer.actor.parameters()))
    critic_changed = any(not torch.equal(b, a) for b, a in zip(critic_before, trainer.critic.parameters()))
    assert actor_changed
    assert critic_changed


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    trainer = _make_trainer()
    ckpt_path = str(tmp_path / "low_level_align_xy_ppo_rnn.pt")
    trainer.save_checkpoint(ckpt_path)

    reloaded = _make_trainer()
    reloaded.load_checkpoint(ckpt_path)

    obs_t = torch.rand(1, OBS_DIM)
    with torch.no_grad():
        mean_a, log_std_a, _ = trainer.actor.forward(obs_t)
        mean_b, log_std_b, _ = reloaded.actor.forward(obs_t)
        value_a, _ = trainer.critic(obs_t)
        value_b, _ = reloaded.critic(obs_t)

    torch.testing.assert_close(mean_a, mean_b)
    torch.testing.assert_close(log_std_a, log_std_b)
    torch.testing.assert_close(value_a, value_b)
