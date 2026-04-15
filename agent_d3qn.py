from __future__ import annotations
from typing import List, Optional
from collections import deque
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
N_FRAMES = 4

def createDuelingNetwork(inDim, outDim, hDim=[128, 128], activation=F.relu):

    class DuelingNetwork(nn.Module):
        def __init__(self, in_d, out_d, h_d, act):
            super(DuelingNetwork, self).__init__()
            self.activation = act
            self.layers = nn.ModuleList()
            curr_dim = in_d
            for h in h_d:
                self.layers.append(nn.Linear(curr_dim, h))
                curr_dim = h
            self.value_head = nn.Linear(curr_dim, 1)
            self.advantage_head = nn.Linear(curr_dim, out_d)

        def forward(self, x):
            for layer in self.layers:
                x = self.activation(layer(x))
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return q_values

    return DuelingNetwork(inDim, outDim, hDim, activation)

_model: Optional[nn.Module] = None
_frame_stack: Optional[deque] = None
_last_action: Optional[int] = None
_repeat_count: int = 0
_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05

def _load_once():
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent.py.")
    m = createDuelingNetwork(18 * N_FRAMES, 5, hDim=[128, 128])
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator = None) -> str:
    global _last_action, _repeat_count, _frame_stack
    _load_once()

    if _frame_stack is None:
        _frame_stack = deque([obs] * N_FRAMES, maxlen=N_FRAMES)
    else:
        _frame_stack.append(obs)

    stacked = np.concatenate(list(_frame_stack))
    x = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    q = _model(x).squeeze(0).cpu().numpy()
    best = int(np.argmax(q))

    if _last_action is not None:
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
    return ACTIONS[best]
