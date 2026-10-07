"""
Unit tests for training.behavioral_cloning — pure PyTorch + synthetic
demonstrations, no mujoco/gymnasium needed (collection itself is delegated
to training/chained_teacher_rollout.py, not exercised here).
"""
import numpy as np
import pytest
import torch

from think_then_act.training.behavioral_cloning import BCConfig, BehavioralCloningTrainer

OBS_DIM = 29   # RELATIVE_OBS_DIM — align_xy/descend's shared width
ACTION_DIM = 4


def _make_trainer(**overrides) -> BehavioralCloningTrainer:
    config = BCConfig(obs_dim=OBS_DIM, hidden_dim=16, rnn_hidden_size=8, **overrides)
    return BehavioralCloningTrainer(config)


def _make_demo(rng, T: int) -> dict:
    return {
        "obs": rng.normal(size=(T, OBS_DIM)).astype(np.float32),
        "teacher_action": (rng.uniform(-1, 1, size=(T, ACTION_DIM))).astype(np.float32),
    }


def test_fit_raises_on_empty_demonstrations():
    trainer = _make_trainer()
    with pytest.raises(ValueError):
        trainer.fit([])


def test_fit_reduces_loss_over_epochs_on_a_fixed_small_dataset():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    demos = [_make_demo(rng, T) for T in (3, 5, 4)]
    trainer = _make_trainer(lr=1e-2, n_epochs=50, episodes_per_minibatch=8)

    result = trainer.fit(demos)
    losses = result["epoch_losses"]
    assert len(losses) == 50
    assert all(np.isfinite(l) for l in losses)
    # Not strictly monotonic epoch-to-epoch necessarily, but the tail should
    # clearly be lower than the head on a tiny, easily-fit dataset.
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_fit_defaults_missing_weight_to_one():
    """Backward compatibility: demonstrations with no "weight" key (the
    shape every existing test in this file uses) must train identically to
    ones with an explicit weight=1.0 — same seed, same result."""
    rng = np.random.default_rng(2)
    demo = _make_demo(rng, T=4)

    torch.manual_seed(0)
    trainer_no_key = _make_trainer(lr=1e-2, n_epochs=5)
    result_no_key = trainer_no_key.fit([dict(demo)])

    torch.manual_seed(0)
    trainer_explicit = _make_trainer(lr=1e-2, n_epochs=5)
    result_explicit = trainer_explicit.fit([dict(demo, weight=1.0)])

    assert result_no_key["epoch_losses"] == pytest.approx(result_explicit["epoch_losses"])


def test_fit_upweighted_episode_dominates_a_conflicting_target():
    """
    Two episodes with the SAME obs but CONFLICTING teacher_action targets —
    a network with enough capacity can't satisfy both, so whichever one is
    weighted higher should end up closer to what the trained network
    predicts. Directly verifies the weight actually changes training
    outcome, not just that compute_episode_weight computes a number.
    """
    torch.manual_seed(0)
    obs = np.zeros((1, OBS_DIM), dtype=np.float32)
    low_target  = {"obs": obs, "teacher_action": np.array([[-0.8, 0, 0, 0]], dtype=np.float32), "weight": 0.1}
    high_target = {"obs": obs, "teacher_action": np.array([[0.8, 0, 0, 0]], dtype=np.float32), "weight": 5.0}

    trainer = _make_trainer(lr=5e-2, n_epochs=200, episodes_per_minibatch=8)
    trainer.fit([low_target, high_target])

    with torch.no_grad():
        mean, _, _ = trainer.actor.forward(torch.from_numpy(obs))
    predicted = torch.tanh(mean)[0, 0].item()
    # Should land much closer to the heavily-upweighted +0.8 target than the
    # downweighted -0.8 one.
    assert predicted > 0.3


def test_fit_handles_unequal_length_episodes_in_one_minibatch():
    rng = np.random.default_rng(1)
    demos = [_make_demo(rng, T) for T in (1, 2, 10)]
    trainer = _make_trainer(n_epochs=2, episodes_per_minibatch=8)

    result = trainer.fit(demos)
    assert all(np.isfinite(l) for l in result["epoch_losses"])


def test_masked_mse_ignores_padded_region_regardless_of_its_content():
    """
    Same regression-test shape as low_level_ppo_recurrent's masking test:
    two episodes of length 2 and 5 batched together (T_max=5) must give the
    IDENTICAL masked loss whether the padded region (steps 2:5 of the short
    episode) is zeros or huge garbage values.
    """
    trainer = _make_trainer()
    B, T_max = 2, 5
    real_T = [2, 5]

    def build(pad_value: float):
        torch.manual_seed(0)
        obs = torch.zeros(B, T_max, OBS_DIM)
        action = torch.zeros(B, T_max, ACTION_DIM)
        mask = torch.zeros(B, T_max)
        for b, T in enumerate(real_T):
            obs[b, :T] = torch.rand(T, OBS_DIM)
            action[b, :T] = torch.rand(T, ACTION_DIM) * 2 - 1
            mask[b, :T] = 1.0
            if T < T_max:
                obs[b, T:] = pad_value
                action[b, T:] = pad_value
        return obs, action, mask

    obs_zero, action_zero, mask_zero = build(pad_value=0.0)
    obs_huge, action_huge, mask_huge = build(pad_value=1e6)

    torch.testing.assert_close(obs_zero[mask_zero.bool()], obs_huge[mask_huge.bool()])
    assert not torch.equal(obs_zero, obs_huge)

    loss_zero = trainer._compute_masked_mse(obs_zero, action_zero, mask_zero)
    loss_huge = trainer._compute_masked_mse(obs_huge, action_huge, mask_huge)

    torch.testing.assert_close(loss_zero, loss_huge)
    assert np.isfinite(loss_huge.item())


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    trainer = _make_trainer()
    ckpt_path = str(tmp_path / "low_level_align_descend_bc.pt")
    trainer.save_checkpoint(ckpt_path)

    reloaded = _make_trainer()
    reloaded.load_checkpoint(ckpt_path)

    obs_t = torch.rand(1, OBS_DIM)
    with torch.no_grad():
        mean_a, log_std_a, _ = trainer.actor.forward(obs_t)
        mean_b, log_std_b, _ = reloaded.actor.forward(obs_t)

    torch.testing.assert_close(mean_a, mean_b)
    torch.testing.assert_close(log_std_a, log_std_b)


def test_bc_checkpoint_loads_into_recurrent_ppo_trainer_as_warm_start(tmp_path):
    """
    The whole point of saving an (untrained) critic alongside the BC-trained
    actor: a BC checkpoint must load into RecurrentLowLevelPPOTrainer
    UNCHANGED, as its documented --warm-start-ckpt path expects, with no
    special-casing. This is the concrete compatibility contract, not just a
    round-trip of BehavioralCloningTrainer against itself.
    """
    from think_then_act.training.low_level_ppo_recurrent import (
        RecurrentLowLevelPPOConfig, RecurrentLowLevelPPOTrainer,
    )

    bc_trainer = _make_trainer()
    ckpt_path = str(tmp_path / "low_level_align_descend_bc.pt")
    bc_trainer.save_checkpoint(ckpt_path)

    ppo_config = RecurrentLowLevelPPOConfig(obs_dim=OBS_DIM, hidden_dim=16, rnn_hidden_size=8)
    ppo_trainer = RecurrentLowLevelPPOTrainer(ppo_config)
    ppo_trainer.load_checkpoint(ckpt_path)   # must not raise

    obs_t = torch.rand(1, OBS_DIM)
    with torch.no_grad():
        bc_mean, _, _ = bc_trainer.actor.forward(obs_t)
        ppo_mean, _, _ = ppo_trainer.actor.forward(obs_t)
    torch.testing.assert_close(bc_mean, ppo_mean)
