from __future__ import annotations
from typing import List, Optional, Dict
from collections import deque
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
ACTION_HEADING_DELTA = {0: 45.0, 1: 22.5, 2: 0.0, 3: -22.5, 4: -45.0}

N_FRAMES = 4
PHASES = ["find", "push", "unwedge"]

PHASE_FIND_VEC    = np.array([1, 0, 0], dtype=np.float32)
PHASE_PUSH_VEC    = np.array([0, 1, 0], dtype=np.float32)
PHASE_UNWEDGE_VEC = np.array([0, 0, 1], dtype=np.float32)

RAW_OBS_DIM = 18
# 64-dim for Level 3 (POMDP-aware with wall features)
FEAT_DIM_L3 = 18 + 18 + 5 + 3 + 3 + 8 + 3 + 2 + 1 + 3  # = 64
# 47-dim legacy (Level 2)
FEAT_DIM_L2 = 18 + 18 + 5 + 3 + 3  # = 47


def detect_phase(obs):
    if obs[16] == 1:
        return "push"
    elif obs[17] == 1:
        return "unwedge"
    return "find"


def detect_phase_vec(obs):
    if obs[16] == 1:
        return PHASE_PUSH_VEC
    elif obs[17] == 1:
        return PHASE_UNWEDGE_VEC
    return PHASE_FIND_VEC


def update_sonar_persistence(sonar_persistence, obs):
    """Update per-sonar-pair persistence counters (8 pairs)."""
    for i in range(8):
        far_bit  = obs[2 * i]
        near_bit = obs[2 * i + 1]
        if far_bit or near_bit:
            sonar_persistence[i] += 1
        else:
            sonar_persistence[i] = 0
    return sonar_persistence


def update_obstacle_contact_history(obstacle_contact_hist, obs, decay=0.8):
    """Track which directions caused stuck contacts."""
    obstacle_contact_hist *= decay
    if obs[17] == 1:
        if obs[0] or obs[1] or obs[2] or obs[3]:
            obstacle_contact_hist[0] += 1.0
        if any(obs[4:12]):
            obstacle_contact_hist[1] += 1.0
        if obs[12] or obs[13] or obs[14] or obs[15]:
            obstacle_contact_hist[2] += 1.0
    return obstacle_contact_hist


def engineer_features_l3(obs, prev_obs, blink_hidden_steps=0, stuck_streak=0,
                          sonar_persistence=None, estimated_heading=0.0,
                          push_step_count=0, obstacle_contact_hist=None):
    """64-dim POMDP-aware feature vector for Level 3 (blinking box + walls)."""
    base  = obs.astype(np.float32)
    delta = (obs - prev_obs).astype(np.float32)

    left_any  = float(np.any(obs[0:4]))
    fwd_any   = float(np.any(obs[4:12]))
    right_any = float(np.any(obs[12:16]))
    ir_active = float(obs[16])
    stuck     = float(obs[17])

    phase = detect_phase_vec(obs)

    prev_sonar = prev_obs[:16]
    was_all_dark = float(np.sum(prev_sonar) == 0 and np.sum(obs[:16]) == 0)
    blink_hidden_norm = float(min(blink_hidden_steps, 30)) / 30.0
    stuck_streak_norm = float(min(stuck_streak, 10)) / 10.0

    if sonar_persistence is None:
        sonar_persistence = np.zeros(8, dtype=np.float32)
    sp_norm = np.clip(sonar_persistence / 20.0, 0.0, 1.0).astype(np.float32)

    left_persistent  = float(np.any(sonar_persistence[:2] > 5))
    right_persistent = float(np.any(sonar_persistence[6:8] > 5))
    fwd_persistent   = float(np.any(sonar_persistence[2:6] > 5))

    heading_rad = np.deg2rad(estimated_heading % 360)
    heading_sin = float(np.sin(heading_rad))
    heading_cos = float(np.cos(heading_rad))

    push_progress = float(min(push_step_count, 500)) / 500.0

    if obstacle_contact_hist is None:
        obstacle_contact_hist = np.zeros(3, dtype=np.float32)
    oc_norm = np.clip(obstacle_contact_hist / 5.0, 0.0, 1.0).astype(np.float32)

    return np.concatenate([
        base, delta,
        [left_any, fwd_any, right_any, ir_active, stuck],
        phase,
        [was_all_dark, blink_hidden_norm, stuck_streak_norm],
        sp_norm,
        [left_persistent, fwd_persistent, right_persistent],
        [heading_sin, heading_cos],
        [push_progress],
        oc_norm,
    ])


def engineer_features_l2(obs, prev_obs, blink_hidden_steps=0, stuck_streak=0):
    """47-dim feature vector for Level 2 (legacy fallback)."""
    base  = obs.astype(np.float32)
    delta = (obs - prev_obs).astype(np.float32)

    left_any  = float(np.any(obs[0:4]))
    fwd_any   = float(np.any(obs[4:12]))
    right_any = float(np.any(obs[12:16]))
    ir_active = float(obs[16])
    stuck     = float(obs[17])

    phase = detect_phase_vec(obs)

    prev_sonar = prev_obs[:16]
    was_all_dark = float(np.sum(prev_sonar) == 0 and np.sum(obs[:16]) == 0)
    blink_hidden_norm = float(min(blink_hidden_steps, 30)) / 30.0
    stuck_streak_norm = float(min(stuck_streak, 10)) / 10.0

    return np.concatenate([
        base, delta,
        [left_any, fwd_any, right_any, ir_active, stuck],
        phase,
        [was_all_dark, blink_hidden_norm, stuck_streak_norm],
    ])


def createDuelingNetwork(inDim, outDim, hDim=[512, 256, 256, 128], activation=F.relu):

    class DuelingNetwork(nn.Module):
        def __init__(self, in_d, out_d, h_d, act):
            super(DuelingNetwork, self).__init__()
            self.activation = act
            self.layers = nn.ModuleList()
            self.layer_norms = nn.ModuleList()
            curr_dim = in_d
            for h in h_d:
                self.layers.append(nn.Linear(curr_dim, h))
                self.layer_norms.append(nn.LayerNorm(h))
                curr_dim = h
            self.value_head = nn.Linear(curr_dim, 1)
            self.advantage_head = nn.Linear(curr_dim, out_d)

        def forward(self, x):
            for layer, ln in zip(self.layers, self.layer_norms):
                x = self.activation(ln(layer(x)))
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return q_values

    return DuelingNetwork(inDim, outDim, hDim, activation)


# ── Global agent state ────────────────────────────────────────────────────────
_models: Optional[Dict[str, nn.Module]] = None
_frame_stack: Optional[deque] = None
_prev_raw: Optional[np.ndarray] = None
_last_action: Optional[int] = None
_repeat_count: int = 0
_blink_hidden_steps: int = 0
_stuck_streak: int = 0
_estimated_heading: float = 0.0
_push_step_count: int = 0
_sonar_persistence: Optional[np.ndarray] = None
_obstacle_contact_hist: Optional[np.ndarray] = None
_active_feat_dim: int = FEAT_DIM_L3
_use_l3_features: bool = True

# Tuning constants
_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05
_FORCE_TURN_THRESH = 4        # slightly more aggressive than L2
_FORCE_TURN_HARD_THRESH = 8   # escalate to opposite direction
_FLAT_Q_RANGE = 0.10
_PUSH_WALL_PREEMPT_THRESH = 3 # preemptive turn during push if wall detected


def _reset_episode_state():
    """Reset all per-episode stateful variables."""
    global _frame_stack, _prev_raw, _last_action, _repeat_count
    global _blink_hidden_steps, _stuck_streak
    global _estimated_heading, _push_step_count
    global _sonar_persistence, _obstacle_contact_hist
    _frame_stack = None
    _prev_raw = None
    _last_action = None
    _repeat_count = 0
    _blink_hidden_steps = 0
    _stuck_streak = 0
    _estimated_heading = 0.0
    _push_step_count = 0
    _sonar_persistence = np.zeros(8, dtype=np.float32)
    _obstacle_contact_hist = np.zeros(3, dtype=np.float32)


def _detect_new_episode(obs: np.ndarray) -> bool:
    """Heuristic: large observation change ⇒ environment was reset."""
    global _prev_raw
    if _prev_raw is None:
        return True
    diff = float(np.sum(np.abs(obs.astype(np.float32) - _prev_raw.astype(np.float32))))
    if diff > 10.0:
        return True
    return False


def _load_once():
    global _models, _active_feat_dim, _use_l3_features
    if _models is not None:
        return

    here = os.path.dirname(os.path.abspath(__file__))

    # Try weights/d3qn/ folder first, then legacy flat paths
    candidates = [
        os.path.join(here, "weights", "d3qn", "weights_hier_diff3.pth"),
        os.path.join(here, "weights_hier_diff3.pth"),
        os.path.join(here, "weights_hier_l3.pth"),
        os.path.join(here, "weights.pth"),
    ]

    # Environment variable override
    env_weights = os.environ.get("OBELIX_WEIGHTS", None)
    if env_weights:
        candidates.insert(0, os.path.join(here, env_weights))

    wpath = None
    for c in candidates:
        if os.path.exists(c):
            wpath = c
            break

    if wpath is None:
        raise FileNotFoundError(
            f"No weights found. Searched: {[os.path.relpath(c, here) for c in candidates]}"
        )

    sd = torch.load(wpath, map_location="cpu")

    _models = {}

    # Auto-detect feature dimension from weight file
    if isinstance(sd, dict) and all(p in sd for p in PHASES):
        # Hierarchical weights: sd = {"find": state_dict, "push": ..., "unwedge": ...}
        sample_sd = sd[PHASES[0]]
        first_weight_key = [k for k in sample_sd.keys() if "layers.0.weight" in k][0]
        detected_in_dim = sample_sd[first_weight_key].shape[1]
        detected_feat_dim = detected_in_dim // N_FRAMES

        if detected_feat_dim == FEAT_DIM_L3:
            _active_feat_dim = FEAT_DIM_L3
            _use_l3_features = True
        else:
            _active_feat_dim = FEAT_DIM_L2
            _use_l3_features = False

        # Detect hidden dims from weight file
        hidden_dims = []
        layer_idx = 0
        while f"layers.{layer_idx}.weight" in sample_sd:
            hidden_dims.append(sample_sd[f"layers.{layer_idx}.weight"].shape[0])
            layer_idx += 1

        in_dim = _active_feat_dim * N_FRAMES

        for p in PHASES:
            m = createDuelingNetwork(in_dim, 5, hDim=hidden_dims)
            m.load_state_dict(sd[p], strict=True)
            m.eval()
            _models[p] = m
    else:
        # Flat state_dict: use same weights for all phases (legacy)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        first_weight_key = [k for k in sd.keys() if "layers.0.weight" in k][0]
        detected_in_dim = sd[first_weight_key].shape[1]
        detected_feat_dim = detected_in_dim // N_FRAMES

        if detected_feat_dim == FEAT_DIM_L3:
            _active_feat_dim = FEAT_DIM_L3
            _use_l3_features = True
        else:
            _active_feat_dim = FEAT_DIM_L2
            _use_l3_features = False

        hidden_dims = []
        layer_idx = 0
        while f"layers.{layer_idx}.weight" in sd:
            hidden_dims.append(sd[f"layers.{layer_idx}.weight"].shape[0])
            layer_idx += 1

        in_dim = _active_feat_dim * N_FRAMES

        for p in PHASES:
            m = createDuelingNetwork(in_dim, 5, hDim=hidden_dims)
            m.load_state_dict(sd, strict=True)
            m.eval()
            _models[p] = m

    print(f"[agent_hierarchical] weights={wpath}  feat_dim={_active_feat_dim}  "
          f"l3_features={_use_l3_features}  in_dim={_active_feat_dim * N_FRAMES}")


def _get_features(obs):
    """Compute features using the appropriate method based on loaded weights."""
    global _sonar_persistence, _obstacle_contact_hist

    if _use_l3_features:
        return engineer_features_l3(
            obs, _prev_raw, _blink_hidden_steps, _stuck_streak,
            _sonar_persistence, _estimated_heading, _push_step_count,
            _obstacle_contact_hist
        )
    else:
        return engineer_features_l2(obs, _prev_raw, _blink_hidden_steps, _stuck_streak)


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator = None) -> str:
    global _last_action, _repeat_count, _frame_stack, _prev_raw
    global _blink_hidden_steps, _stuck_streak
    global _estimated_heading, _push_step_count
    global _sonar_persistence, _obstacle_contact_hist

    _load_once()

    # ── Episode boundary detection ─────────────────────────────────────────
    if _detect_new_episode(obs):
        _reset_episode_state()

    # ── Init ───────────────────────────────────────────────────────────────
    if _prev_raw is None:
        _prev_raw = obs.copy()
    if _sonar_persistence is None:
        _sonar_persistence = np.zeros(8, dtype=np.float32)
    if _obstacle_contact_hist is None:
        _obstacle_contact_hist = np.zeros(3, dtype=np.float32)

    # ── Update stateful counters ───────────────────────────────────────────
    # Sonar persistence
    _sonar_persistence = update_sonar_persistence(_sonar_persistence, obs)

    # Obstacle contact history
    _obstacle_contact_hist = update_obstacle_contact_history(
        _obstacle_contact_hist, obs
    )

    # Blink counter
    all_sonar_dark = np.sum(obs[:16]) == 0
    in_push_mode   = bool(obs[16])
    if all_sonar_dark and not in_push_mode:
        _blink_hidden_steps += 1
    else:
        _blink_hidden_steps = 0

    # Stuck streak
    if bool(obs[17]):
        _stuck_streak += 1
    else:
        _stuck_streak = 0

    # Push step count
    if in_push_mode:
        _push_step_count += 1

    # ── Features ───────────────────────────────────────────────────────────
    feat      = _get_features(obs)
    _prev_raw = obs.copy()

    if _frame_stack is None:
        _frame_stack = deque([feat] * N_FRAMES, maxlen=N_FRAMES)
    else:
        _frame_stack.append(feat)

    stacked = np.concatenate(list(_frame_stack)).astype(np.float32)
    phase_str = detect_phase(obs)
    net = _models[phase_str]

    x = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    q = net(x).squeeze(0).cpu().numpy()

    # ══════════════════════════════════════════════════════════════════════
    # WALL-AWARE HEURISTIC OVERRIDES (executed before greedy policy)
    # ══════════════════════════════════════════════════════════════════════

    # ── 1. Hard stuck escalation ───────────────────────────────────────────
    # Multi-stage stuck recovery:
    #   4-7 steps: alternate L45/R45
    #   8+ steps:  try the OPPOSITE of what we've been doing
    if _stuck_streak >= _FORCE_TURN_HARD_THRESH:
        # Escalate: try opposite direction from recent pattern
        # Use obstacle contact history to choose: turn away from the wall
        if _obstacle_contact_hist is not None and _obstacle_contact_hist[0] > _obstacle_contact_hist[2]:
            # More contact on left → turn right
            forced = 4  # R45
        elif _obstacle_contact_hist is not None and _obstacle_contact_hist[2] > _obstacle_contact_hist[0]:
            # More contact on right → turn left
            forced = 0  # L45
        else:
            # Equal or unknown → alternate based on stuck count
            forced = 0 if (_stuck_streak % 3 == 0) else 4
        action = forced
        _last_action = action
        _repeat_count = 0
        # Update heading
        _estimated_heading += ACTION_HEADING_DELTA[action]
        return ACTIONS[action]

    if _stuck_streak >= _FORCE_TURN_THRESH:
        forced = 0 if (_stuck_streak % 2 == 1) else 4   # L45 / R45
        _last_action  = forced
        _repeat_count = 0
        _estimated_heading += ACTION_HEADING_DELTA[forced]
        return ACTIONS[forced]

    # ── 2. Push-phase preemptive wall avoidance ────────────────────────────
    # If pushing and side sensors show persistent obstacle (likely wall),
    # preemptively turn before getting stuck.
    if phase_str == "push":
        left_wall  = _sonar_persistence is not None and np.any(_sonar_persistence[:2] > _PUSH_WALL_PREEMPT_THRESH)
        right_wall = _sonar_persistence is not None and np.any(_sonar_persistence[6:8] > _PUSH_WALL_PREEMPT_THRESH)
        fwd_wall   = _sonar_persistence is not None and np.any(_sonar_persistence[2:6] > _PUSH_WALL_PREEMPT_THRESH)

        if fwd_wall and not in_push_mode:
            # Forward blocked by persistent obstacle while finding — sweep to find gap
            sweep = 1 if (_push_step_count % 6 < 3) else 3  # L22 / R22
            _last_action = sweep
            _repeat_count = 0
            _estimated_heading += ACTION_HEADING_DELTA[sweep]
            return ACTIONS[sweep]

        if fwd_wall and in_push_mode:
            # Pushing directly into wall — need to reorient
            # Turn toward the side with LESS wall detection
            if left_wall and not right_wall:
                action = 3  # R22 — turn right (away from left wall)
            elif right_wall and not left_wall:
                action = 1  # L22 — turn left (away from right wall)
            else:
                # Both sides or neither — use Q-values to decide turn direction
                left_q = q[0] + q[1]   # L45 + L22
                right_q = q[3] + q[4]  # R22 + R45
                action = 1 if left_q > right_q else 3
            _last_action = action
            _repeat_count = 0
            _estimated_heading += ACTION_HEADING_DELTA[action]
            return ACTIONS[action]

    # ── 3. Sweep during blink (uncertain network) ──────────────────────────
    q_range = float(q.max() - q.min())
    if _blink_hidden_steps > 5 and q_range < _FLAT_Q_RANGE:
        sweep = 1 if (_blink_hidden_steps % 4 < 2) else 3  # L22 / R22
        _last_action  = sweep
        _repeat_count = 0
        _estimated_heading += ACTION_HEADING_DELTA[sweep]
        return ACTIONS[sweep]

    # ── 4. Greedy with tie-break smoothing ─────────────────────────────────
    best = int(np.argmax(q))

    if _last_action is not None and _stuck_streak == 0:
        order = np.argsort(-q)
        best_q, second_q = float(q[order[0]]), float(q[order[1]])
        if (best_q - second_q) < _CLOSE_Q_DELTA:
            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0

    _last_action = best
    _estimated_heading += ACTION_HEADING_DELTA[best]
    return ACTIONS[best]
