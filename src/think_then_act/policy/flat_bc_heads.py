"""
think_then_act.policy.flat_bc_heads

Three alternative action-distribution heads for the flat full-task BC
policy, built for the BC-architecture scaling comparison (completion_rate
vs. dataset size, at 100/500/1000 genuine demonstrations) — see memory:
the MSE baseline (SubgoalRecurrentPolicy, policy/subgoal_recurrent_policy.py)
is unchanged and reused as-is for that comparison's "mse" cell.

All three share the EXACT SAME recurrent trunk as SubgoalRecurrentPolicy
(LayerNorm -> Linear+Tanh -> GRU, via the same _run_recurrent_trunk helper)
and differ ONLY in the output head + loss — deliberately, so a difference
in completion_rate across architectures reflects the action-distribution
modeling choice, not a confound from different trunk capacity.

MDNPolicy: Mixture Density Network (Bishop 1994) — K full-covariance-
diagonal Gaussian components over the 4-dim action, loss = NLL of the
teacher action under the mixture. No tanh-squashing on the components
(unlike SubgoalRecurrentPolicy/SAC-style policies) — this is pure BC
maximum-likelihood regression against already-bounded [-1,1] targets, not
an RL policy needing a proper tanh-Jacobian-corrected log_prob, so the
simpler unsquashed MDN formulation applies.

AutoregressiveDiscretePolicy: revives this project's ORIGINAL M1-era action
representation (see project_context memory: 17 bins/dim, one token per
dim, before the hierarchical pivot) — but decoded autoregressively
(dx -> dy -> dz -> grip) via per-dim embeddings fed back into each
subsequent dim's head, conditioned through the SAME GRU hidden state.
Teacher-forced during training (the teacher's own bin feeds the next dim,
not the model's prediction) — standard autoregressive-sequence BC
practice, keeps training stable/parallelizable.

CVAEPolicy: standard conditional VAE (fixed N(0,I) prior, not a learned
conditional prior — simplest standard CVAE form) — encoder q(z|gru_out,
action) at train time, decoder p(action|gru_out,z) with z~prior at
inference (no teacher action available then).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from think_then_act.policy.subgoal_recurrent_policy import _run_recurrent_trunk

ACTION_DIM = 4
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _act_eval_guard(module: nn.Module):
    """Shared was_training/eval()/finally-restore pattern every act() below uses."""
    return module.training


# ---------------------------------------------------------------------------
# Mixture Density Network
# ---------------------------------------------------------------------------

class MDNPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = ACTION_DIM, hidden_dim: int = 64,
                 rnn_hidden_size: int = 64, n_components: int = 5, decode_mode: str = "argmax"):
        """
        decode_mode (deterministic act() only — stochastic sampling is
        unaffected): "argmax" (default) takes the single highest-weight
        component's mean each step. "weighted_mean" instead takes
        sum_k(pi_k * mean_k) — the full mixture's expected value. argmax
        can jump discontinuously between components as pi shifts which one
        dominates (confirmed happens mid-episode, 2026-10-02 inspection);
        weighted_mean varies smoothly with pi instead, at the cost of
        blending genuinely separate modes into an in-between action when
        the task truly is multimodal. Added specifically to test whether
        argmax's discontinuity — not the earlier-fixed tanh bug — is why
        completion_rate stayed at 0% even after that fix.
        """
        super().__init__()
        if decode_mode not in ("argmax", "weighted_mean"):
            raise ValueError(f"decode_mode={decode_mode!r} must be 'argmax' or 'weighted_mean'")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.n_components = n_components
        self.decode_mode = decode_mode

        self.input_norm = nn.LayerNorm(obs_dim)
        self.pre_gru = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.gru = nn.GRU(hidden_dim, rnn_hidden_size, batch_first=True)
        self.pi_head = nn.Linear(rnn_hidden_size, n_components)
        self.mean_head = nn.Linear(rnn_hidden_size, n_components * action_dim)
        self.log_std_head = nn.Linear(rnn_hidden_size, n_components * action_dim)
        # Same small-init rationale as SubgoalRecurrentPolicy's mean_head —
        # keeps initial component means near 0 rather than wherever default
        # init lands.
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor, hidden_state=None) -> tuple:
        """
        obs: (B, obs_dim) or (B, T, obs_dim). Returns (pi_logits, means,
        log_stds, next_hidden) — pi_logits: (..., K), means/log_stds:
        (..., K, action_dim), leading shape matching obs minus obs_dim.
        """
        gru_out, next_hidden, squeeze_back = _run_recurrent_trunk(
            self.input_norm, self.pre_gru, self.gru, obs, hidden_state, self.rnn_hidden_size
        )
        lead_shape = gru_out.shape[:-1]
        pi_logits = self.pi_head(gru_out)
        means = self.mean_head(gru_out).view(*lead_shape, self.n_components, self.action_dim)
        log_stds = self.log_std_head(gru_out).clamp(LOG_STD_MIN, LOG_STD_MAX).view(
            *lead_shape, self.n_components, self.action_dim)
        if squeeze_back:
            pi_logits = pi_logits.squeeze(1)
            means = means.squeeze(1)
            log_stds = log_stds.squeeze(1)
        return pi_logits, means, log_stds, next_hidden

    def nll_loss(self, obs: torch.Tensor, action: torch.Tensor, hidden_state=None) -> tuple:
        """
        action: (..., action_dim) matching obs's leading shape. Returns
        (per-step negative log-likelihood, next_hidden) — per-step shape is
        obs's leading shape (e.g. (B, T)), same convention
        training/behavioral_cloning.py's masked MSE uses.
        """
        pi_logits, means, log_stds, next_hidden = self.forward(obs, hidden_state)
        action_exp = action.unsqueeze(-2)   # (..., 1, action_dim), broadcasts vs (..., K, action_dim)
        std = log_stds.exp()
        log_comp = -0.5 * (((action_exp - means) / std) ** 2 + 2 * log_stds + np.log(2 * np.pi))
        log_comp = log_comp.sum(dim=-1)                       # (..., K)
        log_pi = F.log_softmax(pi_logits, dim=-1)              # (..., K)
        log_mix = torch.logsumexp(log_pi + log_comp, dim=-1)   # (...,)
        return -log_mix, next_hidden

    def act(self, obs: np.ndarray, hidden_state=None, deterministic: bool = False) -> tuple:
        was_training = _act_eval_guard(self)
        self.eval()
        try:
            with torch.no_grad():
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                pi_logits, means, log_stds, next_hidden = self.forward(obs_t, hidden_state)
                # pi_logits: (1, K), means/log_stds: (1, K, action_dim) — single-step, squeezed.
                if deterministic and self.decode_mode == "weighted_mean":
                    probs = F.softmax(pi_logits, dim=-1)                      # (1, K)
                    raw = (probs.unsqueeze(-1) * means).sum(dim=1)            # (1, action_dim)
                elif deterministic:
                    k = torch.argmax(pi_logits, dim=-1)                       # (1,)
                    idx = k.view(-1, 1, 1).expand(-1, 1, self.action_dim)
                    raw = torch.gather(means, 1, idx).squeeze(1)              # (1, action_dim)
                else:
                    probs = F.softmax(pi_logits, dim=-1)
                    k = torch.multinomial(probs, 1).squeeze(-1)               # (1,)
                    idx = k.view(-1, 1, 1).expand(-1, 1, self.action_dim)
                    mean_k = torch.gather(means, 1, idx).squeeze(1)
                    std_k = torch.gather(log_stds.exp(), 1, idx).squeeze(1)
                    raw = mean_k + std_k * torch.randn_like(mean_k)
                # NOT tanh — nll_loss fits these components directly against
                # already-bounded [-1,1] teacher actions (no squashing in the
                # generative model, unlike SAC-style policies), so `raw` is
                # already in the action's own space. Applying tanh here was a
                # confirmed bug (2026-10-02): it added an extra, UNTRAINED
                # nonlinearity that systematically shrank every action toward
                # zero (e.g. a correctly-learned mean of 0.957 was being
                # returned as tanh(0.957)=0.744) — root cause of MDN's flat
                # 0% completion_rate in the architecture-scaling experiment,
                # not an inherent MDN pathology. Clamp instead, only as a
                # safety net for a component mean that drifted slightly
                # outside the valid range during training.
                action = raw.clamp(-1.0, 1.0)
            return action.squeeze(0).numpy(), next_hidden
        finally:
            if was_training:
                self.train()


# ---------------------------------------------------------------------------
# Autoregressive discretized-bin policy
# ---------------------------------------------------------------------------

N_BINS = 17          # matches this project's original (pre-hierarchical-pivot) M1-era
                      # action representation — see module docstring.
EMBED_DIM = 8


class AutoregressiveDiscretePolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = ACTION_DIM, hidden_dim: int = 64,
                 rnn_hidden_size: int = 64, n_bins: int = N_BINS, embed_dim: int = EMBED_DIM,
                 autoregressive: bool = True):
        """
        autoregressive=True (default, original behavior): each dim's head
        sees gru_out PLUS an embedding of dims already decoded this step —
        the chain-rule factorization p(dx,dy,dz,grip|s)=p(dx|s)p(dy|s,dx)...
        can represent correlations BETWEEN action dims at the same
        timestep (e.g. a genuinely diagonal dx/dy move), at the cost of an
        arbitrary fixed decode order (dx->dy->dz->grip here — no
        particular justification for that order over any other).

        autoregressive=False: each dim's head sees ONLY gru_out, no
        cross-dim conditioning at all — p(dx,dy,dz,grip|s) is modeled as
        the PRODUCT of independent per-dim marginals. No ordering choice
        needed (there's no chain to order), but can't represent any
        same-step correlation between dims — e.g. independently-sampled
        "plausible dx" and "plausible dy" might not correspond to any
        single coherent demonstrated direction. Built as a direct test of
        whether this task's action dims are actually coupled enough for
        the autoregressive version's extra complexity to earn its keep
        (see memory: 2026-10 BC architecture scaling comparison).
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.n_bins = n_bins
        self.embed_dim = embed_dim
        self.autoregressive = autoregressive

        self.input_norm = nn.LayerNorm(obs_dim)
        self.pre_gru = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.gru = nn.GRU(hidden_dim, rnn_hidden_size, batch_first=True)
        if autoregressive:
            self.bin_embed = nn.Embedding(n_bins, embed_dim)
        # Fixed-width input for every dim's head: gru_out + one embed_dim
        # SLOT per action dim (zero-filled for dims not yet decoded) when
        # autoregressive — keeps all action_dim heads the same shape, no
        # per-dim special-casing needed in the decode loop below. Just
        # gru_out (no context) when not autoregressive.
        head_input_dim = rnn_hidden_size + (action_dim * embed_dim if autoregressive else 0)
        self.dim_heads = nn.ModuleList([
            nn.Linear(head_input_dim, n_bins) for _ in range(action_dim)
        ])

    @staticmethod
    def value_to_bin(value: torch.Tensor, n_bins: int = N_BINS) -> torch.Tensor:
        step = 2.0 / (n_bins - 1)
        bin_idx = torch.round((value + 1.0) / step)
        return bin_idx.clamp(0, n_bins - 1).long()

    @staticmethod
    def bin_to_value(bin_idx: torch.Tensor, n_bins: int = N_BINS) -> torch.Tensor:
        step = 2.0 / (n_bins - 1)
        return -1.0 + bin_idx.float() * step

    def forward_logits(self, obs: torch.Tensor, hidden_state=None, teacher_bins=None, sample: bool = False) -> tuple:
        """
        teacher_bins: (..., action_dim) long — when given, TEACHER-FORCES
        each dim's "previously decoded" context (training-time use). When
        None, autoregressively decodes from the model's own predictions —
        argmax per dim (sample=False) or a categorical draw (sample=True).

        Returns (logits_list [action_dim tensors, each (..., n_bins)],
        decoded_bins (..., action_dim) long, next_hidden).
        """
        gru_out, next_hidden, squeeze_back = _run_recurrent_trunk(
            self.input_norm, self.pre_gru, self.gru, obs, hidden_state, self.rnn_hidden_size
        )
        # gru_out is always (B, T, H) here — _run_recurrent_trunk never
        # squeezes its own return, only callers do, at the end.
        B, T, _ = gru_out.shape
        if self.autoregressive:
            context = torch.zeros(B, T, self.action_dim * self.embed_dim, dtype=gru_out.dtype, device=gru_out.device)
        logits_list = []
        decoded_bins = []
        for i in range(self.action_dim):
            head_input = torch.cat([gru_out, context], dim=-1) if self.autoregressive else gru_out
            logits_i = self.dim_heads[i](head_input)   # (B, T, n_bins)
            logits_list.append(logits_i)
            if teacher_bins is not None:
                bin_i = teacher_bins[..., i]
            elif sample:
                probs = F.softmax(logits_i, dim=-1)
                bin_i = torch.multinomial(probs.reshape(-1, self.n_bins), 1).view(B, T)
            else:
                bin_i = torch.argmax(logits_i, dim=-1)
            decoded_bins.append(bin_i)
            if self.autoregressive:
                embed_i = self.bin_embed(bin_i)             # (B, T, embed_dim)
                context = context.clone()
                context[..., i * self.embed_dim:(i + 1) * self.embed_dim] = embed_i
        decoded_bins_t = torch.stack(decoded_bins, dim=-1)   # (B, T, action_dim)
        if squeeze_back:
            logits_list = [l.squeeze(1) for l in logits_list]
            decoded_bins_t = decoded_bins_t.squeeze(1)
        return logits_list, decoded_bins_t, next_hidden

    def ce_loss(self, obs: torch.Tensor, action: torch.Tensor, hidden_state=None) -> tuple:
        """
        action: (..., action_dim) float, teacher's continuous [-1,1] action
        (discretized here via value_to_bin, not pre-discretized by the
        caller). Returns (per-step cross-entropy summed over the 4 dims,
        next_hidden) — per-step shape matches obs's leading shape.
        """
        teacher_bins = self.value_to_bin(action)
        logits_list, _, next_hidden = self.forward_logits(obs, hidden_state, teacher_bins=teacher_bins)
        total = None
        for i in range(self.action_dim):
            logits_i = logits_list[i]
            flat_logits = logits_i.reshape(-1, logits_i.shape[-1])
            flat_target = teacher_bins[..., i].reshape(-1)
            ce_i = F.cross_entropy(flat_logits, flat_target, reduction="none").view(teacher_bins.shape[:-1])
            total = ce_i if total is None else total + ce_i
        return total, next_hidden

    def act(self, obs: np.ndarray, hidden_state=None, deterministic: bool = False) -> tuple:
        was_training = _act_eval_guard(self)
        self.eval()
        try:
            with torch.no_grad():
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                _, decoded_bins, next_hidden = self.forward_logits(
                    obs_t, hidden_state, teacher_bins=None, sample=not deterministic
                )
                action = self.bin_to_value(decoded_bins)   # already in [-1,1] by construction
            return action.squeeze(0).numpy(), next_hidden
        finally:
            if was_training:
                self.train()


# ---------------------------------------------------------------------------
# Conditional VAE
# ---------------------------------------------------------------------------

LATENT_DIM = 8


class CVAEPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = ACTION_DIM, hidden_dim: int = 64,
                 rnn_hidden_size: int = 64, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.latent_dim = latent_dim

        self.input_norm = nn.LayerNorm(obs_dim)
        self.pre_gru = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.gru = nn.GRU(hidden_dim, rnn_hidden_size, batch_first=True)

        # q(z | gru_out, action) — training-time only, needs the teacher action.
        self.encoder = nn.Sequential(nn.Linear(rnn_hidden_size + action_dim, hidden_dim), nn.Tanh())
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_log_std = nn.Linear(hidden_dim, latent_dim)

        # p(action | gru_out, z) — z from the encoder at train time,
        # sampled from the fixed N(0,I) prior at inference (no teacher
        # action available then).
        self.decoder = nn.Sequential(nn.Linear(rnn_hidden_size + latent_dim, hidden_dim), nn.Tanh())
        self.dec_mean = nn.Linear(hidden_dim, action_dim)
        nn.init.uniform_(self.dec_mean.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.dec_mean.bias)

    def _gru_forward(self, obs: torch.Tensor, hidden_state=None) -> tuple:
        return _run_recurrent_trunk(self.input_norm, self.pre_gru, self.gru, obs, hidden_state, self.rnn_hidden_size)

    def elbo_loss(self, obs: torch.Tensor, action: torch.Tensor, hidden_state=None, beta: float = 1.0) -> tuple:
        """
        obs: (B, T, obs_dim), action: (B, T, action_dim) — TRAINING-time use
        only (always chunked, never single-step; act() below handles
        rollout-time decoding separately since there's no teacher action to
        encode then). Returns (per-step reconstruction + beta*KL loss
        (B, T), next_hidden).
        """
        gru_out, next_hidden, _ = self._gru_forward(obs, hidden_state)   # (B, T, H)
        enc_in = torch.cat([gru_out, action], dim=-1)
        h_enc = self.encoder(enc_in)
        mu = self.enc_mu(h_enc)
        log_std = self.enc_log_std(h_enc).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        z = mu + std * torch.randn_like(std)

        dec_in = torch.cat([gru_out, z], dim=-1)
        action_mean = torch.tanh(self.dec_mean(self.decoder(dec_in)))
        recon = ((action_mean - action) ** 2).sum(dim=-1)
        kl = -0.5 * (1.0 + 2 * log_std - mu.pow(2) - (2 * log_std).exp()).sum(dim=-1)
        return recon + beta * kl, next_hidden

    def act(self, obs: np.ndarray, hidden_state=None, deterministic: bool = False) -> tuple:
        was_training = _act_eval_guard(self)
        self.eval()
        try:
            with torch.no_grad():
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                gru_out, next_hidden, squeeze_back = self._gru_forward(obs_t, hidden_state)   # (1, 1, H)
                if deterministic:
                    z = torch.zeros(gru_out.shape[0], gru_out.shape[1], self.latent_dim, dtype=gru_out.dtype)
                else:
                    z = torch.randn(gru_out.shape[0], gru_out.shape[1], self.latent_dim, dtype=gru_out.dtype)
                dec_in = torch.cat([gru_out, z], dim=-1)
                action_mean = torch.tanh(self.dec_mean(self.decoder(dec_in)))
                if squeeze_back:
                    action_mean = action_mean.squeeze(1)
            return action_mean.squeeze(0).numpy(), next_hidden
        finally:
            if was_training:
                self.train()


# ---------------------------------------------------------------------------
# Diffusion (DDPM-style) action head
# ---------------------------------------------------------------------------

DIFFUSION_T = 20


class DiffusionPolicy(nn.Module):
    """
    Lightweight DDPM-style diffusion action head, same GRU trunk as every
    other class in this file. Models the action via T=20 steps of learned
    denoising conditioned on the GRU context, instead of a single forward-
    pass regression (mse) or an explicit parametric distribution (mdn/cvae)
    -- the literature's standard tool for genuinely multi-modal action
    distributions, directly relevant given this project's own MDN
    investigation (see memory: flat_policy_bc_scaling) where hedging under
    a likelihood-based loss, not architecture per se, was the leading
    suspect for that head's weaker performance.

    T=20 and a small per-step MLP keep this "lightweight" per this
    project's own standard -- action_dim=4 is tiny, so even T=20
    sequential denoising steps at act()-time is cheap (20 small MLP
    forward passes per env step, negligible next to one MuJoCo physics
    step).
    """

    def __init__(self, obs_dim: int, action_dim: int = ACTION_DIM, hidden_dim: int = 64,
                 rnn_hidden_size: int = 64, diffusion_steps: int = DIFFUSION_T, time_embed_dim: int = 16):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.T = diffusion_steps

        self.input_norm = nn.LayerNorm(obs_dim)
        self.pre_gru = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.gru = nn.GRU(hidden_dim, rnn_hidden_size, batch_first=True)

        self.time_embed = nn.Embedding(self.T, time_embed_dim)
        self.eps_net = nn.Sequential(
            nn.Linear(rnn_hidden_size + action_dim + time_embed_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        # beta_max=0.02 is the standard DDPM value for T~1000 steps -- at
        # this project's deliberately small T=20, that schedule barely
        # corrupts the signal even at the FINAL training step
        # (alphas_cumprod[-1]~=0.82, ~90% signal remaining), while act()'s
        # reverse process always STARTS from true Gaussian noise (0%
        # signal). That mismatch was confirmed directly (2026-10-05): a
        # trained checkpoint's decoded actions clustered near small
        # constant values regardless of context, while teacher actions are
        # almost always near +-1 (saturated) -- the classic signature of a
        # reverse process operating far outside what it was ever exposed
        # to in training. beta_max=0.5 drives alphas_cumprod[-1] to ~0.002
        # (99.8% corrupted), properly matching real inference conditions.
        betas = torch.linspace(1e-4, 0.5, self.T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _gru_forward(self, obs: torch.Tensor, hidden_state=None) -> tuple:
        return _run_recurrent_trunk(self.input_norm, self.pre_gru, self.gru, obs, hidden_state, self.rnn_hidden_size)

    def diffusion_loss(self, obs: torch.Tensor, action: torch.Tensor, hidden_state=None) -> tuple:
        """
        obs: (B, T, obs_dim), action: (B, T, action_dim) -- TRAINING-time
        use, same shape convention as elbo_loss/nll_loss elsewhere in this
        file. Samples an INDEPENDENT random diffusion timestep per (batch,
        episode-step) element -- standard DDPM practice, not per-sequence
        -- and returns the per-step (B, T) noise-prediction MSE.
        """
        gru_out, next_hidden, _ = self._gru_forward(obs, hidden_state)   # (B, Tep, H)
        B, Tep, _ = action.shape
        t_idx = torch.randint(0, self.T, (B, Tep), device=action.device)
        sqrt_ac = self.alphas_cumprod[t_idx].sqrt().unsqueeze(-1)
        sqrt_1m_ac = (1.0 - self.alphas_cumprod[t_idx]).sqrt().unsqueeze(-1)
        eps = torch.randn_like(action)
        x_t = sqrt_ac * action + sqrt_1m_ac * eps

        t_emb = self.time_embed(t_idx)
        eps_in = torch.cat([gru_out, x_t, t_emb], dim=-1)
        eps_hat = self.eps_net(eps_in)
        per_step = ((eps_hat - eps) ** 2).sum(dim=-1)
        return per_step, next_hidden

    def act(self, obs: np.ndarray, hidden_state=None, deterministic: bool = False) -> tuple:
        """
        Full T-step ancestral denoising from x_T (zeros if deterministic,
        else Gaussian noise) down to x_0, conditioned on this step's GRU
        context. deterministic=True also skips the per-step noise
        injection in the reverse process (equivalent to DDIM with eta=0) --
        same input always produces the same output, matching every other
        policy class's deterministic act() convention in this project.
        Final tanh is a numerical safety clamp only (training never
        applies one -- the loss is on predicted NOISE, not the action
        value directly), covering the rare case T=20 steps of
        approximation drift a hair outside [-1, 1].
        """
        was_training = _act_eval_guard(self)
        self.eval()
        try:
            with torch.no_grad():
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                gru_out, next_hidden, _ = self._gru_forward(obs_t, hidden_state)   # (1, 1, H)
                context = gru_out.squeeze(1)   # (1, H)
                x = torch.zeros(1, self.action_dim) if deterministic else torch.randn(1, self.action_dim)
                for t in reversed(range(self.T)):
                    t_idx = torch.full((1,), t, dtype=torch.long)
                    t_emb = self.time_embed(t_idx)
                    eps_hat = self.eps_net(torch.cat([context, x, t_emb], dim=-1))
                    alpha_t = self.alphas[t]
                    alpha_bar_t = self.alphas_cumprod[t]
                    beta_t = self.betas[t]
                    mean = (1.0 / alpha_t.sqrt()) * (x - (beta_t / (1.0 - alpha_bar_t).sqrt()) * eps_hat)
                    if t > 0 and not deterministic:
                        x = mean + beta_t.sqrt() * torch.randn_like(x)
                    else:
                        x = mean
                action = torch.tanh(x)
            return action.squeeze(0).numpy(), next_hidden
        finally:
            if was_training:
                self.train()
