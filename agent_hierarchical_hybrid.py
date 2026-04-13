"""
Hybrid Hierarchical DRQN Agent — Strategy D Inference
=====================================================
Loads weights from 'weights_hybrid_diff3.pth' (or _best variant).
Uses the same 64-dim engineered features as training, fed through the LSTM
with hidden state carried across steps within an episode.
"""
from __future__ import annotations
from typing import Optional, Dict, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_HEADING_DELTA = {0: 45.0, 1: 22.5, 2: 0.0, 3: -22.5, 4: -45.0}
PHASES = ["find", "push", "unwedge"]

PHASE_FIND_VEC    = np.array([1, 0, 0], dtype=np.float32)
PHASE_PUSH_VEC    = np.array([0, 1, 0], dtype=np.float32)
PHASE_UNWEDGE_VEC = np.array([0, 0, 1], dtype=np.float32)

FEAT_DIM = 64


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
    for i in range(8):
        if obs[2 * i] or obs[2 * i + 1]:
            sonar_persistence[i] += 1
        else:
            sonar_persistence[i] = 0
    return sonar_persistence


def update_obstacle_contact_history(hist, obs, decay=0.8):
    hist *= decay
    if obs[17] == 1:
        if obs[0] or obs[1] or obs[2] or obs[3]:
            hist[0] += 1.0
        if any(obs[4:12]):
            hist[1] += 1.0
        if obs[12] or obs[13] or obs[14] or obs[15]:
            hist[2] += 1.0
    return hist


def engineer_features(obs, prev_obs, blink_hidden_steps=0, stuck_streak=0,
                       sonar_persistence=None, estimated_heading=0.0,
                       push_step_count=0, obstacle_contact_hist=None):
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


class HybridDuelingDRQN(nn.Module):
    def __init__(self, in_d, out_d, h_d=256):
        super().__init__()
        self.h_d = h_d
        self.fc_in = nn.Linear(in_d, 256)
        self.ln_in = nn.LayerNorm(256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=h_d, batch_first=True)
        self.ln_lstm = nn.LayerNorm(h_d)
        self.value_head = nn.Sequential(nn.Linear(h_d, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv_head   = nn.Sequential(nn.Linear(h_d, 128), nn.ReLU(), nn.Linear(128, out_d))

    def forward(self, x, hidden_state=None):
        is_unbatched = x.dim() == 2
        if is_unbatched:
            x = x.unsqueeze(0)
        B, S, _ = x.shape
        x_flat = F.relu(self.ln_in(self.fc_in(x.reshape(B * S, -1))))
        lstm_out, new_h = self.lstm(x_flat.reshape(B, S, -1), hidden_state)
        lstm_flat = self.ln_lstm(lstm_out.reshape(B * S, -1))
        val = self.value_head(lstm_flat)
        adv = self.adv_head(lstm_flat)
        q = val + adv - adv.mean(dim=-1, keepdim=True)
        q = q.reshape(B, S, -1)
        if is_unbatched:
            q = q.squeeze(0)
        return q, new_h


# ── Global agent state ────────────────────────────────────────────────────────
_models: Optional[Dict[str, nn.Module]] = None
_prev_raw: Optional[np.ndarray] = None
_hidden_states: Dict[str, Optional[Tuple]] = {}
_current_phase: Optional[str] = None
_sonar_persistence: Optional[np.ndarray] = None
_obstacle_contact_hist: Optional[np.ndarray] = None
_blink_hidden_steps: int = 0
_stuck_streak: int = 0
_estimated_heading: float = 0.0
_push_step_count: int = 0


def _reset():
    global _prev_raw, _hidden_states, _current_phase
    global _sonar_persistence, _obstacle_contact_hist
    global _blink_hidden_steps, _stuck_streak, _estimated_heading, _push_step_count
    _prev_raw = None
    _hidden_states = {p: None for p in PHASES}
    _current_phase = None
    _sonar_persistence = np.zeros(8, dtype=np.float32)
    _obstacle_contact_hist = np.zeros(3, dtype=np.float32)
    _blink_hidden_steps = 0
    _stuck_streak = 0
    _estimated_heading = 0.0
    _push_step_count = 0


def _detect_new_episode(obs):
    global _prev_raw
    if _prev_raw is None:
        return True
    diff = float(np.sum(np.abs(obs.astype(np.float32) - _prev_raw.astype(np.float32))))
    return diff > 10.0


def _load_once():
    global _models
    if _models is not None:
        return

    here = os.path.dirname(os.path.abspath(__file__))

    # Search order: weights/hybrid/ folder → legacy flat paths
    candidates = [
        os.path.join(here, "weights", "hybrid", "weights_hybrid_diff3_best.pth"),
        os.path.join(here, "weights", "hybrid", "weights_hybrid_diff3.pth"),
        os.path.join(here, "weights_hybrid_diff3_best.pth"),
        os.path.join(here, "weights_hybrid_diff3.pth"),
    ]
    # Environment variable override
    env_w = os.environ.get("OBELIX_WEIGHTS", None)
    if env_w:
        candidates.insert(0, os.path.join(here, env_w))

    wpath = None
    for c in candidates:
        if os.path.exists(c):
            wpath = c
            break

    if wpath is None:
        raise FileNotFoundError(
            f"No hybrid weights found. Searched: {[os.path.relpath(c, here) for c in candidates]}"
        )

    sd = torch.load(wpath, map_location="cpu")
    _models = {}

    # Detect if this is a hybrid DRQN weight file or a standard D3QN file
    is_hybrid = "__meta__" in sd and sd["__meta__"].get("architecture") == "hybrid_drqn"

    if is_hybrid:
        meta = sd["__meta__"]
        h_d = meta.get("hidden_dim", 256)
        for p in PHASES:
            m = HybridDuelingDRQN(FEAT_DIM, 5, h_d=h_d)
            m.load_state_dict(sd[p], strict=True)
            m.eval()
            _models[p] = m
        print(f"[agent_hybrid] HYBRID weights={wpath} feat={FEAT_DIM} h_d={h_d}")
    elif isinstance(sd, dict) and all(p in sd for p in PHASES):
        # Standard hierarchical D3QN weights — wrap them, but no LSTM benefit
        sample_sd = sd[PHASES[0]]
        first_key = [k for k in sample_sd if "layers.0.weight" in k][0]
        in_dim = sample_sd[first_key].shape[1]
        # This is a non-LSTM network; we can't use it with our LSTM architecture
        # Fall back to basic dense eval
        print(f"[agent_hybrid] WARNING: Loaded non-LSTM weights from {wpath}")
        print(f"  These weights are from a standard D3QN. LSTM benefits won't apply.")
        # We'll still wrap them but use a compatibility shim
        # For now, raise so the user knows
        raise RuntimeError(
            f"Loaded weights from {wpath} but they are standard D3QN, not Hybrid DRQN. "
            f"Please train with train_hierarchical_hybrid.py first."
        )
    else:
        raise RuntimeError(f"Unrecognized weight format in {wpath}")


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator = None) -> str:
    global _prev_raw, _hidden_states, _current_phase
    global _sonar_persistence, _obstacle_contact_hist
    global _blink_hidden_steps, _stuck_streak, _estimated_heading, _push_step_count

    _load_once()

    if _detect_new_episode(obs):
        _reset()

    if _prev_raw is None:
        _prev_raw = obs.copy()

    # Update trackers
    _sonar_persistence = update_sonar_persistence(_sonar_persistence, obs)
    _obstacle_contact_hist = update_obstacle_contact_history(_obstacle_contact_hist, obs)

    all_dark = np.sum(obs[:16]) == 0
    in_push  = bool(obs[16])
    if all_dark and not in_push:
        _blink_hidden_steps += 1
    else:
        _blink_hidden_steps = 0

    if bool(obs[17]):
        _stuck_streak += 1
    else:
        _stuck_streak = 0

    if in_push:
        _push_step_count += 1

    # Compute features
    feat = engineer_features(obs, _prev_raw, _blink_hidden_steps, _stuck_streak,
                              _sonar_persistence, _estimated_heading,
                              _push_step_count, _obstacle_contact_hist)
    _prev_raw = obs.copy()

    # Phase management
    phase = detect_phase(obs)
    if phase != _current_phase:
        if _current_phase is not None:
            _hidden_states[_current_phase] = None
        _current_phase = phase

    # LSTM forward pass
    net = _models[phase]
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 64)
    q, _hidden_states[phase] = net(x, _hidden_states[phase])

    best = int(q.squeeze().argmax().cpu().numpy())
    _estimated_heading += ACTION_HEADING_DELTA[best]

    return ACTIONS[best]
