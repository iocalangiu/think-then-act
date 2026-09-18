# Journey

Notes from building this project — the dead ends and what they turned out to mean, written up as I go. Code and commit history have the "what"; this has the "why," and the wrong turns that got filtered out of the final implementation.

## Why text tokens are a bad action representation for continuous control

One of the first things I tried on the robot arm was having a vision-language model (Qwen2-VL) look at a camera frame and directly output the next move as Cartesian displacements `<Δx, Δy, Δz>` as text tokens, trained end-to-end with GRPO.

It didn't work, and it took a while to figure out whether this was a training, model, or capacity problem.

Numbers are tokens, and continuous numbers tokenize inconsistently. `0.2` is `'0'`, `'.'`, `'2'`, while `0.19` is `'0'`, `'.'`, `'1'`, `'9'`. A small difference like 0.01 results in two completely different token sequences. You can avoid this by binning the action space into discrete bins — each bin gets an integer from 0 to N, and the VLM predicts a bin id instead of a float.

That doesn't solve the underlying issue, though. Autoregressive models generate tokens sequentially, conditioning each dimension on whatever came before it: `P(Δx) · P(Δy|Δx) · P(Δz|Δx,Δy)`. This creates an artificial order dependency between action dimensions that have no real reason to depend on each other.

For example, if `(Δx=8, Δy)` succeeds and `(Δx=9, Δy)` fails, the model learns two different things about the *same* `Δy` — that it was good when conditioned on `Δx=8`, and bad when conditioned on `Δx=9` — instead of learning that `Δy` was simply good, independent of `Δx`.

It turned out I was using the wrong action representation (floats, then bins), and the wrong way to use a VLM for continuous control in the first place.

## Measuring gripper-to-brick distance from a point cloud, without color or calibration

The goal of the robot arm is to pick up a vertical brick. For perception, I have a depth camera that generates 3D point clouds. The goal is to guide the gripper to correctly close its fingers around the brick, and to quickly block the control policy if it plans a move that would topple the brick.

To process the point cloud I first use RANSAC to find the table, then PCA to find the brick. Finding the brick was not trivial, even though it's static and has a simple shape.

The harder question: how do I precisely measure the distance between the gripper's fingers and the brick? Initially I wanted to further cluster the points, identify two clusters belonging to the two fingers, and track the distance between the axis spanning those two clusters and the brick. That didn't work — I couldn't reliably identify the two fingers as separate clusters. My best guess is that the wrist angle, combined with the material the gripper is made of, changes how many points land on it and whether the two fingers show up merged or separated in a given frame.

So I switched to using the robot's own internal state via forward kinematics instead — which trades that unreliability for sensitivity to calibration errors instead. The clip below shows the projection: points from the brick cluster and the non-brick cluster (noise or gripper, undistinguished at this stage — a proxy for gripper position) projected onto the gripper's width axis, tracking their spread and proximity to the brick over time.

![Brick and gripper points projected onto the gripper's width axis, tracked over a grasp session](../assets/journey/gripper_brick_projection.gif)

The idea going forward: when the brick's points and the outer edge of the non-brick points sit at roughly the same distance along this axis, that's the gripper's fingers positioned around the brick — a safe regime. If the brick gets too close to those outer points, that's the regime that should trigger a quick stop. Getting there required a lot more precision than this projection alone gave — see `scripts/extract_gripper_brick_geometry.py` for where that ended up: RANSAC-based table/wall removal, PCA-based brick and finger clustering, and using the forward-kinematics prediction as a prior to pick the right point cluster as "the gripper" instead of guessing from geometry alone.
