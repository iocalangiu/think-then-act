# Multi-cube stacking: SB3-teacher imitation, architecture comparison, and a 2-cube curriculum

A robotic arm (Fetch, MuJoCo) learns to pick up a cube and place it on a
target, then to stack a second cube on top of the first — starting from
demonstrations distilled off a pretrained SB3 (Stable-Baselines3) expert
policy, not from scratch.

## Pipeline

1. **Teacher demos.** An external SB3 PPO expert (`collect_sb3_teacher_demos.py`)
   is rolled out repeatedly; only genuine completions (two-finger contact +
   lifted, not a slide/nudge that happens to end near the target) are kept
   as the behavioral-cloning (BC) demo pool.
2. **BC architecture comparison** (`train_bc_gripper3d_scaling.py`). Four
   action heads on the same GRU trunk — `mse` (deterministic regression),
   `cvae` (conditional VAE, stochastic latent), `transformer` (causal
   self-attention trunk), `diffusion` (lightweight DDPM-style denoising
   head) — each trained at 100/500/1000/2000 demos, 10 independently-trained
   seeds per cell.
3. **PPO fine-tune** (`train_flat_task_ppo.py`). The best `mse` BC checkpoint
   continues training with PPO (dense drag/idle-churn-penalty reward on top
   of the native task reward) — exceeds the teacher itself, since PPO can
   explore trajectories the teacher never demonstrated instead of only
   copying it.
4. **2-cube curriculum**: a scripted hand-off collects 2-cube demos (run the
   trained policy to place cube 1, scripted release + ascend, run the policy
   again with cube 2's target re-pointed on top of cube 1), then BC
   alternation / BC fine-tune / PPO self-imitation are each tried on top of
   that pool (`train_bc_multicube_alternating.py`, `train_bc_multicube_finetune.py`,
   `train_multicube_stack_ppo_selfimitation.py`).

## Results

All numbers below are genuine-grasp-filtered completion rates (not raw
`is_success`, which also accepts slide/nudge artifacts) on held-out,
pose-randomized seeds. Full detail and exact checkpoint paths: `models.yaml`.

### Basic skill (1 cube to a target)

| | n=100 | n=500 | n=1000 | n=2000 |
|---|---|---|---|---|
| mse (BC) | 1.0% | 31.3% | 33.7% | 42.7% |
| cvae (BC) | 1.0% | 26.3% | 39.0% | 46.0% |
| transformer (BC) | 8.0% | 22.7% | 35.0% | 41.0% |
| diffusion (BC) | 0.0% | 3.7% | 5.7% | 14.0% |

(mean across 10 independently-trained seeds per cell; teacher's own
genuine-completion rate on its demo-collection rollouts: **61.0%**)

| checkpoint | completion rate |
|---|---|
| mse + PPO (`flat_task_ppo_gripper3d_v1_cont_best.pt`) | **70.0%** — best single-cube checkpoint |
| transformer + DAgger (`dagger_transformer_v1_best.pt`) | 53.3% |
| transformer + PPO (unstable, see Finding 4) | 56.7% (not a stable result) |

### 2-cube stacking

All 4 measured on the same fixed 30-seed held-out set, under the FIXED
release-settle criterion (see Finding 5 — earlier numbers for 3 of these 4
checkpoints were measured under a buggy criterion and don't match what's
published elsewhere about this project before 2026-10-06).

| learning strategy | completion rate |
|---|---|
| BC, 1-cube/2-cube alternation (100 demos, from scratch) | 0.0% |
| BC fine-tune of mse+PPO (400 2-cube demos) | 33.3% |
| BC fine-tune, +200 more demos (600 total) | 30.0% (worse) |
| **PPO + self-imitation (current best)** | **56.7%** |

## Findings

### 1. Genuine-grasp contamination bug
`is_success` alone (block within tolerance of target) also fires when a
policy slides or shoves the block near the goal without ever actually
grasping it. Every completion-rate number in this project since has
required `is_success AND (two-finger contact AND lifted > 2cm) at any point
this episode` — the "genuine" filter.

### 2. Diffusion's noise-schedule bug
The diffusion head uses a short `T=20`-step denoising chain (standard DDPM
recipes use ~1000 steps). Its first trained checkpoint scored a flat 0.0% —
traced directly (predicted actions vs. the real teacher's actions on an
actual demo trace) to the default beta schedule (`linspace(1e-4, 0.02, T)`)
barely corrupting the signal at `T=20` (`alphas_cumprod[-1] ≈ 0.82`, ~90%
signal still present at the noisiest training step), while inference always
starts from pure noise — a training/inference distribution mismatch.
Fixed by steepening the schedule (`beta_max` 0.02→0.5, driving
`alphas_cumprod[-1]` to ~0.002); confirmed 0.0%→20.0% on an immediate retrain.
**Lesson: a noise schedule's hyperparameters, calibrated for one `T`, do not
transfer to a much smaller `T` just because the functional form is the same.**

### 3. Why CVAE didn't share diffusion's problem
CVAE also models a distribution (not a point estimate), but its weakest
remaining gap after the fix above — diffusion still trailed every other
architecture throughout the 100-2000 demo sweep — comes down to how each
one decodes at inference:
- **CVAE** (`policy/flat_bc_heads.py::CVAEPolicy.act`): one forward pass.
  Deterministic mode sets `z = 0`; since the encoder is KL-regularized
  toward `N(0, I)` during training, `z = 0` sits near the middle of what the
  decoder actually saw in training.
- **Diffusion** (`policy/flat_bc_heads.py::DiffusionPolicy.act`): 20
  sequential steps, each conditioned on the last. Deterministic mode starts
  from `x_T = zeros` — a point the model never actually saw in training
  (training only ever showed it noisy *actions*, i.e. real signal + scaled
  Gaussian noise, never a flat zero vector).

CVAE pays for its stochastic-distribution modeling with one small KL term;
diffusion pays with a 20-step iterative generation process, each step a
chance to be miscalibrated. For a near-deterministic teacher (one right
action per state most of the time), that flexibility is mostly unneeded
overhead — unsurprising mse/cvae come out ahead.

### 4. Transformer + PPO instability — a real, confirmed bug, not architecture-inherent
Transformer PPO collapsed at both `lr=3e-4` (by iteration 20) and `lr=1e-4`
(peaked 56.7% at iteration 40, then declined for the rest of a 150-iteration
run down to 13.3%). Root cause, found via a direct diagnostic (collect a
rollout batch, then immediately — no gradient step — compare `sample()`'s
recorded `old_log_prob` against a fresh `recompute_log_prob()` on the
identical `(obs, raw_sample)` pairs with identical weights; these should be
exactly equal): mean abs diff 0.24, max 2.88. `nn.TransformerEncoderLayer`
defaults to `dropout=0.1`, never set to 0 — rollout-time `sample()` ran
through a separately-reconstructed, `.eval()`'d actor copy (dropout off),
training-time `recompute_log_prob()` ran on the live trainer actor (never
switched to `.eval()`), so two calls on identical weights/inputs got two
different live dropout masks. This alone explains the persistently inflated
`approx_kl` and saturated `clip_frac` seen in both failed runs. Fixed
(`dropout=0.0` in `TransformerPolicy.__init__`); re-ran the same diagnostic
on the same already-trained checkpoint — mean abs diff dropped to 0.000000,
max 0.000002. **A plain PPO retry after the fix was never re-attempted**
(DAgger was chosen instead) — the 56.7% transformer+PPO number in the table
above is the pre-fix run's best epoch, shown for completeness, not a
recommended checkpoint.

### 5. The big one: premature "done" criterion (2-cube stacking)
`MultiCubeStackEnv`'s advancement/done logic used to fire the instant a held
cube passed within 5cm of its target — a single-step check, with `carrying`
(from `reward/flat_task_reward.py`) a *sticky* flag that stays `True` for
the rest of the episode once the cube was ever genuinely grasped. So a cube
could be counted "placed" **while still mid-air, held**, not released or
settled. Confirmed directly: rendering a "trustworthy" (release + settle +
gripper-clear) check video of a *supposedly*-genuine trial showed cube 1
sitting on the table **next to** cube 0, not on top — the env had already
called it done before release.

Fixed by tracking a per-cube `_settle_count`: a cube only counts as placed
once UNASSISTED (gripper released) and within `move_to_target_threshold` for
`release_settle_steps` (10) CONSECUTIVE steps. This is a training-signal
change, not just an eval filter — retraining PPO self-imitation from scratch
against the fixed signal took the checkpoint from 26.7% (BC warm-start,
under the strict criterion) to **70.0%** on the training-time eval set
(`approx_kl` settling to 0.002-0.004 by the end — clean convergence).

Re-measuring the 3 OTHER 2-cube checkpoints under the same fixed criterion
moved every number, not just the PPO one — and not all in the same
direction: fine-tune BC dropped 46.7%→33.3% (was genuinely inflated by the
bug), but +200-demos BC rose 20.0%→30.0% (the old criterion's premature
freeze-point for cube 0's "placed" position was *less* forgiving for this
specific checkpoint's stack-integrity check, not more). **Lesson: a
criterion fix doesn't move every number in the same direction — don't
assume which way it'll go.**

### 6. `env.render()` perturbs the physics simulation
Confirmed 2026-10-07: rolling out the same checkpoint on the same seed, once
with `render_mode="rgb_array"` + `env.render()` called every step (what
every video-recording script needs), once with no rendering at all, gives
measurably different trajectories. Across 6 held-out seeds, 5/6 kept the
same final success/failure classification but step counts differed by 1-3
in *every* case (a small floating-point-level perturbation, most likely
from the OSMesa software rendering context); the 6th seed sat right on the
policy's decision boundary and flipped from a clean success (74 steps, no
rendering) to a failure (69 steps, rendered).

**What this does and doesn't affect**: every completion-rate number in the
tables above was measured without rendering and is unaffected. What it does
affect: any specific "this seed is a failure/success" claim drawn only from
a recorded video — the act of recording that exact trial could itself have
nudged a borderline case across the line. `rollout.py`'s `--also-clean-pass`
flag re-runs the same seeds with no rendering specifically so a caller can
see both and trust the non-rendered classification, not the video.

### Also along the way
A **checkpoint overwrite incident**: a BC scaling sweep's checkpoint-path
formula collided with an unrelated, already-load-bearing checkpoint (the
exact weights the single-cube PPO run had historically warm-started from),
silently overwriting it mid-session. Caught via volume modification
timestamps, not before. Damage was contained — PPO only reads a BC
checkpoint once at startup, so the already-completed PPO run and everything
built on it were unaffected — but it's why `mse_bc_n2000_sweep` in
`models.yaml` doesn't point at one specific "seed 0" file.

## Known limitations / not yet done

- **Elevated-target generalization** was never part of this lineage's own
  training/eval distribution (`env.setup.init_random_episode` always pins
  the target to table height for every eval in this project). One
  out-of-distribution probe (target z=0.590 vs. 0.425 resting, native-Fetch-style
  elevation) against the 70% mse+PPO checkpoint succeeded on the first
  held-out seed tried — a single data point, not a measured generalization
  rate.
- **Recovery trajectories** were never collected — every demo and every
  training rollout is a clean, uninterrupted attempt. A policy that's never
  seen "the cube slipped, now what" has no data to learn recovery from.
- The `env.render()` perturbation's exact mechanism (Finding 6) is
  confirmed but not root-caused to a specific line of code in MuJoCo/OSMesa.

## Using this folder

- `models.yaml` / `model_registry.py` — the checkpoint registry. `python3
  model_registry.py` prints every registered model with its completion rate.
- `rollout.py` — roll out any registered (or raw `--ckpt-path`) checkpoint,
  optionally recording video. **Read its module docstring before trusting a
  video-only success/failure label** (Finding 6).
- Everything else is the training/collection/diagnostic history that
  produced the checkpoints above, kept for reference — not meant to be
  re-run as-is without adjusting checkpoint paths for whatever's current on
  your own Modal volume.
