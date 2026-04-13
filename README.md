# CS780 Capstone — OBELIX Warehouse Robot
**Student:** Vemula Anurag Sai (231138) | anuragsv23@iitk.ac.in  
**Codabench Username:** v_231138 | **Final Rank:** 17
---
## Problem
OBELIX is a partially observable robotic control task. A mobile robot must:
1. **Find** a grey box using sonar sensors
2. **Attach** to it on IR contact
3. **Push** it to the arena boundary
4. **Unwedge** (detach) once at the boundary
Three difficulty levels:
- **Level 1** — static box
- **Level 2** — box randomly blinks invisible (on: 30–60 steps, off: 10–30 steps)
- **Level 3** — box moves at constant velocity + blinks; optional wall obstacle with a gap
---
## Environment Specification
### Observation Space
`numpy.ndarray` of shape `(18,)`, all binary (`0` or `1`):
| Index | Sensor | Description |
|-------|--------|-------------|
| 0–1   | Sonar 1 (left-far) | far-bit, near-bit |
| 2–3   | Sonar 2 (left) | far-bit, near-bit |
| 4–5   | Sonar 3 (fwd-left) | far-bit, near-bit |
| 6–7   | Sonar 4 (fwd-left-near) | far-bit, near-bit |
| 8–9   | Sonar 5 (fwd-right-near) | far-bit, near-bit |
| 10–11 | Sonar 6 (fwd-right) | far-bit, near-bit |
| 12–13 | Sonar 7 (right) | far-bit, near-bit |
| 14–15 | Sonar 8 (right-far) | far-bit, near-bit |
| 16    | IR sensor | 1 if box is directly ahead within range |
| 17    | Stuck flag | 1 if robot could not move forward |
> **Critical:** `obs[17]` fires identically for wall collisions **and** successful boundary delivery. The agent cannot distinguish these from observation alone.
Sonar positions (relative to heading): −112°, −68°, −45°, −22°, +22°, +45°, +68°, +112°  
Sonar FOV: 20° per cone | Far range: 150px | Near range: 90px | IR range: 20px
### Action Space
5 discrete actions:
| Action | Effect |
|--------|--------|
| `L45`  | Rotate left 45° |
| `L22`  | Rotate left 22.5° |
| `FW`   | Move forward 5px |
| `R22`  | Rotate right 22.5° |
| `R45`  | Rotate right 45° |
> Only `FW` can cause the stuck flag. Turns never get stuck.
### Reward Structure (per step, from environment)
| Event | Reward |
|-------|--------|
| Step penalty | −1 |
| Stuck (obs[17]=1) | −200 |
| First sonar activation (per bit, one-time) | +1 to +3 |
| IR sensor activation (one-time) | +5 |
| First box attachment | +100 |
| Success (box reaches boundary) | +2000 |
| Max steps exceeded | episode ends, no bonus |
### Environment Parameters
| Parameter | Value |
|-----------|-------|
| `arena_size` | 500 px |
| `scaling_factor` | 5 |
| `max_steps` (train) | 1000 |
| `max_steps` (eval) | 2000 |
| `box_size` | 60 px (12 × scaling) |
| `bot_radius` | 30 px |
| `forward_step_unit` | 5 px |
| `box_speed` (L3) | 2 px/step |
| `goal_margin` | 100 px |
---
## Agents
| File | Description |
|------|-------------|
| `agent_hierarchical_hybrid.py` | **Final submitted agent** — Hybrid Hierarchical DRQN |
---
## Architecture: Hybrid Hierarchical DRQN
### Feature Vector (64-dim)
| Component | Dims | Description |
|-----------|------|-------------|
| Raw obs | 18 | Direct sensor bits |
| Obs delta | 18 | `obs − prev_obs` (detects blink transitions) |
| Sonar region sums | 5 | left, fwd-left, fwd, fwd-right, right |
| Phase one-hot | 3 | [find, push, unwedge] |
| Blink features | 3 | all-dark flag, hidden-step count (norm), stuck-streak (norm) |
| Sonar persistence | 8 | steps each sonar pair has been continuously active, norm to [0,1] |
| Wall-side indicators | 3 | left/fwd/right persistent sonar (>5 steps) |
| Heading | 2 | (sin θ, cos θ) of cumulative estimated heading |
| Push progress | 1 | steps in push phase, norm to [0,1] |
| Contact history | 3 | decaying left/fwd/right collision counts |
> **Sonar persistence** is the most critical feature: walls sustain readings for 20+ steps; the blinking box resets to zero during invisible periods.
### Network Architecture
```
Input(64) → FC(256) → LayerNorm → ReLU → LSTM(256) → LayerNorm
                                                          ↓
                                              Value Head: FC(128) → ReLU → FC(1)
                                              Advantage Head: FC(128) → ReLU → FC(5)
                                                          ↓
                                              Q(s,a) = V + A − mean(A)
```
- **3 independent copies** — one per phase (find, push, unwedge)
- LSTM hidden state **reset on phase transitions**
- Phase detected from obs: `obs[16]=1` → push, `obs[17]=1` → unwedge, else → find
### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Feature dim | 64 |
| Hidden dim (LSTM) | 256 |
| Learning rate (find) | 3e-4 |
| Learning rate (push) | 4.5e-4 (1.5×) |
| Learning rate (unwedge) | 6e-4 (2×) |
| Discount γ | 0.99 |
| Batch size | 32 sequences |
| Replay capacity | 500K frames |
| Sequence length | 10 |
| Target sync | every 40 episodes |
| ε start / end | 1.0 → 0.03 |
| ε decay steps | 800K |
| Grad clip | 1.0 |
| LSTM burn-in | 6 steps |
---
## Training
### Curriculum
| Phase | Episodes | Difficulty | Walls |
|-------|----------|------------|-------|
| 1 | 0 – 700 | 0 (static) | No |
| 2 | 700 – 1500 | 0 (static) | Yes |
| 3 | 1500 – 2500 | 2 (blinking) | Yes |
| 4 | 2500+ | 3 (moving+blinking) | Yes |
On phase transition, ε is partially rewound (`step × 0.6`) to re-open exploration.
### Reward Shaping (on top of env reward)
| Signal | Shaped bonus |
|--------|-------------|
| Stuck normalisation | +191 (makes net stuck cost ≈ −10 instead of −200) |
| Escape stuck | +30 |
| Sonar approach (curr > prev) | +2.0 |
| Sonar contact (any active) | +0.3 |
| IR activation (first contact) | +5.0 |
| Sustained IR (pushing) | +1.0 |
| IR lost mid-episode | −3.0 |
### Commands
```bash
# Final agent (Hybrid DRQN)
python train_hierarchical_hybrid.py --obelix_py obelix.py --episodes 3000 --out weights/hybrid.pth
# Unified diff3 agent
python train_diff3.py --obelix_py obelix.py --episodes 800 --out weights/diff3/weights_diff3.pth
# Recurrent PPO
python train_recurrent_ppo.py --obelix_py obelix.py --episodes 2000 --out weights/ppo.pth
# D3QN-PER baseline
python train_d3qn_per.py --obelix_py obelix.py --episodes 2000 --out weights/d3qn.pth
```
---
## Evaluation
```bash
# All levels, 10 runs each
python evaluate.py --agent agent_hierarchical_hybrid --level 0 --runs 10
python evaluate.py --agent agent_hierarchical_hybrid --level 2 --runs 10
python evaluate.py --agent agent_hierarchical_hybrid --level 3 --runs 10
python evaluate.py --agent agent_hierarchical_hybrid --level 3 --wall_obstacles --runs 10
# Manual play
python manual_play.py --obelix_py obelix.py
```
---
## Weights
| File | Agent | Size |
|------|-------|------|
| `weights_hybrid_diff3.pth` | Hybrid DRQN (final, best) | 7.3 MB |
| `weights_drqn_diff3.pth` | Unified DRQN | 540 KB |
| `weights_hier_l3.pth` | Hierarchical DRQN | 4.4 MB |
| `weights_adv_d3.pth` | Advanced agent (D3) | 595 KB |
---
## Requirements
```
torch
numpy
opencv-python
```
```bash
pip install -r requirements.txt
```
---
## Honest Note
Most policies under hard difficulty (L3 + walls) converge to persistent rotation. The step penalty (−1/step) is small enough that spinning is a locally optimal policy under the reward signal — the network learns that rotating loses less reward than attempting and failing the full task. The agent maximises reward; it does not always complete the task.
