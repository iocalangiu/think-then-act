"""
Unit tests for policy.subgoal_recurrent_policy — pure PyTorch, no mujoco needed.
"""
import numpy as np
import torch

from think_then_act.policy.subgoal_recurrent_policy import (
    SubgoalRecurrentPolicy, SubgoalRecurrentValueNetwork,
)

OBS_DIM = 43


def test_forward_single_step_shapes_and_default_hidden_state():
    policy = SubgoalRecurrentPolicy(obs_dim=OBS_DIM)
    obs = torch.rand(5, OBS_DIM)
    mean, log_std, next_hidden = policy(obs)   # hidden_state=None
    assert mean.shape == (5, 4)
    assert log_std.shape == (5, 4)
    assert next_hidden.shape == (1, 5, policy.rnn_hidden_size)


def test_forward_chunk_shapes():
    policy = SubgoalRecurrentPolicy(obs_dim=OBS_DIM)
    obs = torch.rand(3, 10, OBS_DIM)   # (B, T, obs_dim)
    mean, log_std, next_hidden = policy(obs)
    assert mean.shape == (3, 10, 4)
    assert log_std.shape == (3, 10, 4)
    assert next_hidden.shape == (1, 3, policy.rnn_hidden_size)


def test_sample_action_is_bounded_by_tanh_and_returns_hidden_state():
    policy = SubgoalRecurrentPolicy(obs_dim=OBS_DIM)
    obs = torch.rand(32, OBS_DIM) * 10 - 5
    action, raw_sample, log_prob, entropy, next_hidden = policy.sample(obs)
    assert action.shape == (32, 4)
    assert torch.all(action > -1.0) and torch.all(action < 1.0)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    assert next_hidden.shape == (1, 32, policy.rnn_hidden_size)


def test_recompute_log_prob_matches_sample_log_prob_for_same_raw_sample_single_step():
    torch.manual_seed(0)
    policy = SubgoalRecurrentPolicy(obs_dim=OBS_DIM)
    obs = torch.rand(8, OBS_DIM)

    action, raw_sample, log_prob, entropy, _ = policy.sample(obs)
    recomputed_log_prob, recomputed_entropy, _ = policy.recompute_log_prob(obs, raw_sample)

    torch.testing.assert_close(log_prob, recomputed_log_prob)
    torch.testing.assert_close(entropy, recomputed_entropy)


def test_recompute_log_prob_chunk_shape():
    policy = SubgoalRecurrentPolicy(obs_dim=OBS_DIM)
    obs = torch.rand(4, 6, OBS_DIM)
    raw_sample = torch.randn(4, 6, 4)
    log_prob, entropy, next_hidden = policy.recompute_log_prob(obs, raw_sample)
    assert log_prob.shape == (4, 6)
    assert entropy.shape == (4, 6)
    assert next_hidden.shape == (1, 4, policy.rnn_hidden_size)


def test_gradients_flow_through_gru_parameters():
    policy = SubgoalRecurrentPolicy(obs_dim=OBS_DIM, hidden_dim=16, rnn_hidden_size=8)
    obs = torch.rand(2, 5, OBS_DIM)
    raw_sample = torch.randn(2, 5, 4)
    log_prob, _, _ = policy.recompute_log_prob(obs, raw_sample)
    loss = -log_prob.mean()
    loss.backward()

    for name, param in policy.gru.named_parameters():
        assert param.grad is not None, f"no gradient reached GRU parameter {name}"
        assert torch.any(param.grad != 0), f"GRU parameter {name} got an all-zero gradient"


def test_different_hidden_states_give_different_mean_for_same_obs():
    """
    Regression test: a genuinely recurrent policy's output depends on
    hidden state, not just the current obs — otherwise this would secretly
    be a relabeled stateless MLP.
    """
    torch.manual_seed(0)
    policy = SubgoalRecurrentPolicy(obs_dim=OBS_DIM)
    policy.eval()
    obs = torch.rand(1, OBS_DIM)

    h1 = torch.zeros(1, 1, policy.rnn_hidden_size)
    h2 = torch.randn(1, 1, policy.rnn_hidden_size) * 5.0

    with torch.no_grad():
        mean1, _, _ = policy(obs, h1)
        mean2, _, _ = policy(obs, h2)

    assert not torch.allclose(mean1, mean2)


def test_act_deterministic_returns_action_and_hidden_state():
    policy = SubgoalRecurrentPolicy(obs_dim=OBS_DIM)
    obs = np.random.default_rng(0).normal(size=OBS_DIM)

    action, next_hidden = policy.act(obs, deterministic=True)
    assert action.shape == (4,)
    assert np.all(np.abs(action) < 1.0)
    assert next_hidden.shape == (1, 1, policy.rnn_hidden_size)

    # Feeding the returned hidden state back in should be accepted and
    # produce a further-advanced hidden state — the recurrent-use-across-
    # steps pattern rollout_workers.py's _run_episode_recurrent relies on.
    action2, next_hidden2 = policy.act(obs, next_hidden, deterministic=True)
    assert action2.shape == (4,)
    assert not torch.equal(next_hidden, next_hidden2)


def test_act_restores_training_mode():
    policy = SubgoalRecurrentPolicy(obs_dim=OBS_DIM)
    obs = np.zeros(OBS_DIM)

    policy.train()
    policy.act(obs)
    assert policy.training is True

    policy.eval()
    policy.act(obs)
    assert policy.training is False


def test_value_network_shapes_single_step_and_chunk():
    critic = SubgoalRecurrentValueNetwork(obs_dim=OBS_DIM)
    obs_single = torch.rand(6, OBS_DIM)
    value, next_hidden = critic(obs_single)
    assert value.shape == (6,)
    assert next_hidden.shape == (1, 6, critic.rnn_hidden_size)

    obs_chunk = torch.rand(6, 12, OBS_DIM)
    value_chunk, next_hidden_chunk = critic(obs_chunk)
    assert value_chunk.shape == (6, 12)
    assert next_hidden_chunk.shape == (1, 6, critic.rnn_hidden_size)
