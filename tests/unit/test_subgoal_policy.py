"""
Unit tests for policy.subgoal_policy — pure PyTorch, no mujoco needed.
"""

import numpy as np
import pytest
import torch

from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy

OBS_DIM = 38


def test_forward_shapes():
    policy = SubgoalGaussianPolicy(obs_dim=OBS_DIM)
    obs = torch.rand(5, OBS_DIM)
    mean, log_std = policy(obs)
    assert mean.shape == (5, 4)
    assert log_std.shape == (5, 4)


def test_sample_action_is_bounded_by_tanh():
    policy = SubgoalGaussianPolicy(obs_dim=OBS_DIM)
    obs = torch.rand(32, OBS_DIM) * 10 - 5   # wide range, including large values
    action, raw_sample, log_prob, entropy = policy.sample(obs)
    assert action.shape == (32, 4)
    assert torch.all(action > -1.0) and torch.all(action < 1.0)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()


def test_recompute_log_prob_matches_sample_log_prob_for_same_raw_sample():
    """
    Critical correctness check: low_level_grpo.py recomputes log_prob from a
    STORED raw_sample at gradient time, under the (unchanged) policy params
    used at collection time. If recompute_log_prob doesn't reproduce exactly
    what sample() returned for the same (obs, raw_sample), the GRPO gradient
    would be silently wrong.
    """
    torch.manual_seed(0)
    policy = SubgoalGaussianPolicy(obs_dim=OBS_DIM)
    obs = torch.rand(8, OBS_DIM)

    action, raw_sample, log_prob, entropy = policy.sample(obs)
    recomputed_log_prob, recomputed_entropy = policy.recompute_log_prob(obs, raw_sample)

    torch.testing.assert_close(log_prob, recomputed_log_prob)
    torch.testing.assert_close(entropy, recomputed_entropy)


def test_entropy_matches_closed_form_gaussian_formula():
    policy = SubgoalGaussianPolicy(obs_dim=OBS_DIM)
    with torch.no_grad():
        policy.log_std.fill_(-1.0)   # fixed, known std = exp(-1)
    obs = torch.rand(4, OBS_DIM)
    _, _, _, entropy = policy.sample(obs)

    expected = 4 * (0.5 * (1.0 + np.log(2 * np.pi)) + (-1.0))   # summed over 4 action dims
    torch.testing.assert_close(entropy, torch.full((4,), expected, dtype=entropy.dtype))


def test_higher_log_std_gives_higher_entropy():
    policy = SubgoalGaussianPolicy(obs_dim=OBS_DIM)
    obs = torch.rand(1, OBS_DIM)

    with torch.no_grad():
        policy.log_std.fill_(-2.0)
    _, _, _, low_entropy = policy.sample(obs)

    with torch.no_grad():
        policy.log_std.fill_(1.0)
    _, _, _, high_entropy = policy.sample(obs)

    assert high_entropy.item() > low_entropy.item()


def test_act_deterministic_uses_tanh_of_mean():
    policy = SubgoalGaussianPolicy(obs_dim=OBS_DIM)
    obs = np.random.default_rng(0).normal(size=OBS_DIM)

    action = policy.act(obs, deterministic=True)
    assert action.shape == (4,)
    assert np.all(np.abs(action) < 1.0)

    with torch.no_grad():
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        mean, _ = policy.forward(obs_t)
        expected = torch.tanh(mean).squeeze(0).numpy()
    np.testing.assert_allclose(action, expected, atol=1e-6)


def test_act_restores_training_mode():
    policy = SubgoalGaussianPolicy(obs_dim=OBS_DIM)
    obs = np.zeros(OBS_DIM)

    policy.train()
    policy.act(obs)
    assert policy.training is True

    policy.eval()
    policy.act(obs)
    assert policy.training is False


def test_gradient_step_reduces_a_toy_reinforce_style_loss():
    """
    Sanity check: a REINFORCE-style loss (-advantage * log_prob) actually
    moves the policy's mean toward higher-reward actions over a few steps,
    on a toy synthetic setup — same spirit as the collision predictor's
    "does training actually train" check.
    """
    torch.manual_seed(0)
    policy = SubgoalGaussianPolicy(obs_dim=OBS_DIM, hidden_dim=16)

    obs = torch.zeros(1, OBS_DIM)
    target = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    with torch.no_grad():
        initial_mean, _ = policy.forward(obs)
    initial_dist = torch.norm(torch.tanh(initial_mean).squeeze(0) - target.squeeze(0)).item()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    for _ in range(50):
        optimizer.zero_grad()
        action, _, log_prob, _ = policy.sample(obs)
        # Synthetic "reward": negative distance to a fixed target action —
        # higher (less negative) when the sampled action is closer to target.
        advantage = -torch.norm(action - target, dim=-1).detach()
        loss = -(advantage * log_prob).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_mean, _ = policy.forward(obs)
    final_dist = torch.norm(torch.tanh(final_mean).squeeze(0) - target.squeeze(0)).item()

    assert final_dist < initial_dist
