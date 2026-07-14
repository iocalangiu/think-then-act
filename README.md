# think-then-act

Training a robot arm to plan a sequence of actions using images of the environment.

The arm lives in the **FetchPickAndPlace-v3** MuJoCo environment. The task is to pick up a block and move it to a target location.

---
## How it works

<img width="670" height="326" alt="Screenshot 2026-07-14 at 16 33 47" src="https://github.com/user-attachments/assets/27d54f45-de89-4b1f-ad36-5344c00b4d6e" />

Hierarchical architecture (working on the low-level controller; high-level VLM not yet trained). A vision-language model reads the image + text prompt and picks a subgoal (e.g. "align_xy"), passed as a one-hot into a small MLP that outputs the actual continuous action (dx, dy, dz, grip). Both parts are trained in two stages: SFT to teach format/behavior, then RL to improve it — GRPO for the VLM, PPO for the MLP.

The low-level MLP was first trained with GRPO, but it kept collapsing partway through training. PPO worked instead, because our reward is dense (feedback every step, not just at the end) — PPO's per-step critic uses that fully, and its update-clipping stops one bad batch from wrecking the policy, which is what we think was breaking GRPO.

<img width="341" height="180" alt="image" src="https://github.com/user-attachments/assets/5102fe5c-ab35-4b86-86fc-1335ec60abe4" />


Rollouts also run faster now: with 8 CPUs, we run 8 MuJoCo episodes at once to collect data, then pause them while 1 core does the quick PPO update step, then repeat.




