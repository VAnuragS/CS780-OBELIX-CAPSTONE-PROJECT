from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
PHASES = ["find", "push", "unwedge"]

PHASE_FIND_VEC    = np.array([1, 0, 0], dtype=np.float32)
PHASE_PUSH_VEC    = np.array([0, 1, 0], dtype=np.float32)
PHASE_UNWEDGE_VEC = np.array([0, 0, 1], dtype=np.float32)

# Raw inputs + phase = 26
FEAT_DIM = 26

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

def engineer_features_pure(obs):
    """26-dim Pure feature vector for DRQN. LSTM does all the memory processing."""
    base = obs.astype(np.float32)
    left_any  = float(np.any(obs[0:4]))
    fwd_any   = float(np.any(obs[4:12]))
    right_any = float(np.any(obs[12:16]))
    ir_active = float(obs[16])
    stuck     = float(obs[17])
    phase = detect_phase_vec(obs)
    
    return np.concatenate([
        base, 
        [left_any, fwd_any, right_any, ir_active, stuck],
        phase
    ])

class DuelingDRQN(nn.Module):
    def __init__(self, in_d, out_d, h_d=128):
        super(DuelingDRQN, self).__init__()
        self.h_d = h_d
        
        self.fc1 = nn.Linear(in_d, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=h_d, batch_first=True)
        
        self.value_head = nn.Linear(h_d, 1)
        self.adv_head = nn.Linear(h_d, out_d)

    def forward(self, x, hidden_state=None):
        is_unbatched = x.dim() == 2
        if is_unbatched:
            x = x.unsqueeze(0)
            
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        x = x.view(batch_size * seq_len, -1)
        x = F.relu(self.fc1(x))
        x = x.view(batch_size, seq_len, -1)
        
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        lstm_out_flat = lstm_out.reshape(batch_size * seq_len, -1)
        
        val = self.value_head(lstm_out_flat)
        adv = self.adv_head(lstm_out_flat)
        
        q = val + adv - adv.mean(dim=-1, keepdim=True)
        q = q.view(batch_size, seq_len, -1)
        
        if is_unbatched:
            q = q.squeeze(0)
            
        return q, new_hidden

# ── Global agent state ────────────────────────────────────────────────────────
_models: Optional[Dict[str, nn.Module]] = None
_prev_raw: Optional[np.ndarray] = None
_hidden_states: Dict[str, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {p: None for p in PHASES}
_current_phase: Optional[str] = None

def _reset_episode_state():
    """Reset all per-episode stateful variables."""
    global _prev_raw, _hidden_states, _current_phase
    _prev_raw = None
    _hidden_states = {p: None for p in PHASES}
    _current_phase = None

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
    global _models

    if _models is not None:
        return

    here = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(here, "weights", "drqn", "weights_drqn_diff3.pth"),
        os.path.join(here, "weights_drqn_diff3.pth"),
    ]
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
            f"DRQN weights not found. Searched: {[os.path.relpath(c, here) for c in candidates]}"
        )

    sd = torch.load(wpath, map_location="cpu")
    _models = {}

    for p in PHASES:
        # Detect hidden dimension size using value_head weight
        hidden_dim = sd[p]["value_head.weight"].shape[1]
        m = DuelingDRQN(FEAT_DIM, 5, h_d=hidden_dim)
        m.load_state_dict(sd[p], strict=True)
        m.eval()
        _models[p] = m

    print(f"[agent_drqn] weights={wpath} feat_dim={FEAT_DIM} hidden_dim={hidden_dim}")

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator = None) -> str:
    global _prev_raw, _hidden_states, _current_phase

    _load_once()

    if _detect_new_episode(obs):
        _reset_episode_state()

    _prev_raw = obs.copy()

    phase = detect_phase(obs)
    
    # If the environment transitions to a new phase, reset its hidden state context
    # so the LSTM doesn't carry over stale history from 50 steps ago.
    if phase != _current_phase:
        if _current_phase is not None:
            _hidden_states[_current_phase] = None
        _current_phase = phase

    feat = engineer_features_pure(obs)
    
    net = _models[phase]
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, feat_dim)
    
    q, _hidden_states[phase] = net(x, _hidden_states[phase])
    
    best = int(q.squeeze().argmax().cpu().numpy())
    return ACTIONS[best]
