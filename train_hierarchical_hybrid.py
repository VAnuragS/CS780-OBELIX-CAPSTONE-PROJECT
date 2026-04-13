"""
Hybrid Hierarchical DRQN Training — Strategy D
================================================
Combines the best of both worlds:
  - 64-dim POMDP-aware engineered features (sonar persistence, heading,
    obstacle contact history, push progress) for sample efficiency
  - LSTM recurrent network for learning temporal patterns the hand-crafted
    features might miss (box trajectory prediction, optimal sweep timing)

Architecture:
  Input(64) → FC(256) → LSTM(hidden_dim) → Dueling(Value + Advantage) → Q(5)

Curriculum (4-phase):
  Phase 1 (ep 0-700):    diff=0, no walls   — basic find + push
  Phase 2 (ep 700-1500): diff=0 + walls     — wall navigation
  Phase 3 (ep 1500-2500):diff=2 + walls     — blinking box + walls
  Phase 4 (ep 2500+):    diff=3 + walls     — moving + blinking + walls
"""
from __future__ import annotations
import argparse, random, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_HEADING_DELTA = {0: 45.0, 1: 22.5, 2: 0.0, 3: -22.5, 4: -45.0}
PHASES = ["find", "push", "unwedge"]

PHASE_FIND_VEC    = np.array([1, 0, 0], dtype=np.float32)
PHASE_PUSH_VEC    = np.array([0, 1, 0], dtype=np.float32)
PHASE_UNWEDGE_VEC = np.array([0, 0, 1], dtype=np.float32)

# 64-dim feature vector (same as train_hierarchical.py)
FEAT_DIM = 18 + 18 + 5 + 3 + 3 + 8 + 3 + 2 + 1 + 3  # = 64


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
        far_bit  = obs[2 * i]
        near_bit = obs[2 * i + 1]
        if far_bit or near_bit:
            sonar_persistence[i] += 1
        else:
            sonar_persistence[i] = 0
    return sonar_persistence


def update_obstacle_contact_history(obstacle_contact_hist, obs, decay=0.8):
    obstacle_contact_hist *= decay
    if obs[17] == 1:
        if obs[0] or obs[1] or obs[2] or obs[3]:
            obstacle_contact_hist[0] += 1.0
        if any(obs[4:12]):
            obstacle_contact_hist[1] += 1.0
        if obs[12] or obs[13] or obs[14] or obs[15]:
            obstacle_contact_hist[2] += 1.0
    return obstacle_contact_hist


def engineer_features(obs, prev_obs, blink_hidden_steps=0, stuck_streak=0,
                       sonar_persistence=None, estimated_heading=0.0,
                       push_step_count=0, obstacle_contact_hist=None):
    """64-dim POMDP-aware feature vector — identical to train_hierarchical.py."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Network Architecture: Hybrid Dueling DRQN
# ═══════════════════════════════════════════════════════════════════════════════

class HybridDuelingDRQN(nn.Module):
    """
    Input(64) → FC(256, ReLU) → LayerNorm → LSTM(hidden_dim) → Dueling Heads
    
    The FC layer compresses the rich engineered features before the LSTM
    processes them as a temporal sequence. The LSTM learns what patterns
    in the feature sequence predict future box positions and optimal actions.
    """
    def __init__(self, in_d, out_d, h_d=256):
        super().__init__()
        self.h_d = h_d

        self.fc_in = nn.Linear(in_d, 256)
        self.ln_in = nn.LayerNorm(256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=h_d, batch_first=True)
        self.ln_lstm = nn.LayerNorm(h_d)

        # Dueling heads
        self.value_head = nn.Sequential(
            nn.Linear(h_d, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(h_d, 128),
            nn.ReLU(),
            nn.Linear(128, out_d)
        )

    def forward(self, x, hidden_state=None):
        """
        x: (batch, seq_len, in_d) or (seq_len, in_d) for unbatched
        Returns: q_values, new_hidden_state
        """
        is_unbatched = x.dim() == 2
        if is_unbatched:
            x = x.unsqueeze(0)

        B, S, _ = x.shape

        # Compress features
        x_flat = x.reshape(B * S, -1)
        x_flat = F.relu(self.ln_in(self.fc_in(x_flat)))
        x_seq = x_flat.reshape(B, S, -1)

        # Recurrent processing
        lstm_out, new_hidden = self.lstm(x_seq, hidden_state)
        lstm_flat = self.ln_lstm(lstm_out.reshape(B * S, -1))

        # Dueling
        val = self.value_head(lstm_flat)
        adv = self.adv_head(lstm_flat)
        q = val + adv - adv.mean(dim=-1, keepdim=True)
        q = q.reshape(B, S, -1)

        if is_unbatched:
            q = q.squeeze(0)

        return q, new_hidden


# ═══════════════════════════════════════════════════════════════════════════════
# Episodic Replay Buffer with Sequence Sampling
# ═══════════════════════════════════════════════════════════════════════════════

class EpisodicReplayBuffer:
    """Stores complete episodes and samples contiguous subsequences for BPTT."""
    def __init__(self, capacity_frames, seq_len=10):
        self.capacity = capacity_frames
        self.seq_len = seq_len
        self.episodes = []
        self.current_episode = []
        self.total_frames = 0

    def store(self, s, a, r, s2, d):
        self.current_episode.append((s.copy(), int(a), float(r), s2.copy(), bool(d)))

    def end_episode(self):
        if len(self.current_episode) >= self.seq_len:
            self.episodes.append(self.current_episode)
            self.total_frames += len(self.current_episode)
            # Evict oldest episodes if over capacity
            while self.total_frames > self.capacity and len(self.episodes) > 1:
                removed = self.episodes.pop(0)
                self.total_frames -= len(removed)
        self.current_episode = []

    def can_sample(self, batch_size):
        return len(self.episodes) >= max(1, batch_size // 4)

    def sample(self, batch_size):
        s_b, a_b, r_b, s2_b, d_b = [], [], [], [], []
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            start = random.randint(0, len(ep) - self.seq_len)
            seq = ep[start : start + self.seq_len]
            s_b.append([t[0] for t in seq])
            a_b.append([t[1] for t in seq])
            r_b.append([t[2] for t in seq])
            s2_b.append([t[3] for t in seq])
            d_b.append([t[4] for t in seq])
        return (np.array(s_b, dtype=np.float32),
                np.array(a_b, dtype=np.int64),
                np.array(r_b, dtype=np.float32),
                np.array(s2_b, dtype=np.float32),
                np.array(d_b, dtype=np.float32))

    def __len__(self):
        return self.total_frames


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_step(online, target, replay, optimizer, batch, gamma, grad_clip, device):
    if not replay.can_sample(batch):
        return 0.0

    sb, ab, rb, s2b, db = replay.sample(batch)
    sb_t  = torch.tensor(sb, dtype=torch.float32).to(device)
    ab_t  = torch.tensor(ab, dtype=torch.int64).to(device).unsqueeze(-1)
    rb_t  = torch.tensor(rb, dtype=torch.float32).to(device)
    s2b_t = torch.tensor(s2b, dtype=torch.float32).to(device)
    db_t  = torch.tensor(db, dtype=torch.float32).to(device)

    with torch.no_grad():
        next_q, _ = online(s2b_t)
        next_a = next_q.argmax(dim=-1, keepdim=True)
        tgt_q, _ = target(s2b_t)
        next_val = tgt_q.gather(-1, next_a).squeeze(-1)
        targets = rb_t + gamma * (1.0 - db_t) * next_val

    cur_q, _ = online(sb_t)
    cur_val = cur_q.gather(-1, ab_t).squeeze(-1)

    loss = F.smooth_l1_loss(cur_val, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


def linear_epsilon(step, start, end, decay_steps):
    if step >= decay_steps:
        return end
    return start + (end - start) * step / decay_steps


def select_action(net, feat, hidden, eps, device):
    """Epsilon-greedy with hidden state propagation."""
    with torch.no_grad():
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        q, new_h = net(x, hidden)

    if np.random.rand() < eps:
        a = np.random.randint(len(ACTIONS))
    else:
        a = int(q.squeeze().argmax().cpu().numpy())
    return a, new_h


def import_obelix(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser(description="Hybrid DRQN training for Difficulty 3")
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights/hybrid/weights_hybrid_diff3.pth")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--replay", type=int, default=500000)
    ap.add_argument("--target_sync", type=int, default=40)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.03)
    ap.add_argument("--eps_decay_steps", type=int, default=800000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--curriculum", action="store_true",
                    help="4-phase curriculum: diff=0 → diff=0+walls → diff=2+walls → diff=3+walls")
    ap.add_argument("--phase2_ep", type=int, default=700)
    ap.add_argument("--phase3_ep", type=int, default=1500)
    ap.add_argument("--phase4_ep", type=int, default=2500)
    ap.add_argument("--save_every", type=int, default=250)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    print(f"[INFO] Hybrid DRQN training on: {device}")
    print(f"[INFO] FEAT_DIM={FEAT_DIM}  hidden_dim={args.hidden_dim}  seq_len={args.seq_len}")

    OBELIX = import_obelix(args.obelix_py)

    # ── Curriculum ──
    def get_curriculum(ep):
        if not args.curriculum:
            return args.difficulty, args.wall_obstacles
        if ep < args.phase2_ep:
            return 0, False
        elif ep < args.phase3_ep:
            return 0, True
        elif ep < args.phase4_ep:
            return 2, True
        else:
            return 3, True

    def make_env(diff, seed, walls=None):
        w = walls if walls is not None else args.wall_obstacles
        return OBELIX(scaling_factor=args.scaling_factor, arena_size=args.arena_size,
                      max_steps=args.max_steps, wall_obstacles=w,
                      difficulty=diff, box_speed=args.box_speed, seed=seed)

    cur_diff, cur_walls = get_curriculum(0)
    env = make_env(cur_diff, args.seed, walls=cur_walls)

    # ── Networks (per-phase) ──
    online = {p: HybridDuelingDRQN(FEAT_DIM, 5, h_d=args.hidden_dim).to(device) for p in PHASES}
    target = {p: HybridDuelingDRQN(FEAT_DIM, 5, h_d=args.hidden_dim).to(device) for p in PHASES}
    for p in PHASES:
        target[p].load_state_dict(online[p].state_dict())
        target[p].eval()

    lr_mult = {"find": 1.0, "push": 1.5, "unwedge": 2.0}
    optimizers = {p: optim.Adam(online[p].parameters(), lr=args.lr * lr_mult[p]) for p in PHASES}

    replays = {p: EpisodicReplayBuffer(args.replay, args.seq_len) for p in PHASES}

    total_steps = 0
    best_avg = -float("inf")
    reward_history = []
    t0 = time.time()

    def save(path):
        sd = {p: online[p].cpu().state_dict() for p in PHASES}
        sd["__meta__"] = {"feat_dim": FEAT_DIM, "hidden_dim": args.hidden_dim,
                          "architecture": "hybrid_drqn"}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(sd, path)
        for p in PHASES:
            online[p].to(device)

    try:
        for ep in range(args.episodes):
            # ── Curriculum phase transitions ──
            ep_diff, ep_walls = get_curriculum(ep)
            if ep_diff != cur_diff or ep_walls != cur_walls:
                cur_diff, cur_walls = ep_diff, ep_walls
                env = make_env(cur_diff, args.seed, walls=cur_walls)
                print(f"\n[CURRICULUM] ep {ep+1}: diff={cur_diff} walls={cur_walls}\n")

            raw = env.reset(seed=args.seed + ep)
            prev_raw = raw.copy()
            ep_ret = 0.0

            # Per-episode state
            sonar_pers = np.zeros(8, dtype=np.float32)
            obs_contact = np.zeros(3, dtype=np.float32)
            blink_hidden = 0
            stuck_streak = 0
            est_heading = 0.0
            push_steps = 0
            hidden_states = {p: None for p in PHASES}
            cur_phase = detect_phase(raw)

            for step_i in range(args.max_steps):
                eps = linear_epsilon(total_steps, args.eps_start, args.eps_end, args.eps_decay_steps)

                # Update trackers
                sonar_pers = update_sonar_persistence(sonar_pers, raw)
                obs_contact = update_obstacle_contact_history(obs_contact, raw)

                all_dark = np.sum(raw[:16]) == 0
                in_push  = bool(raw[16])
                if all_dark and not in_push:
                    blink_hidden += 1
                else:
                    blink_hidden = 0
                if bool(raw[17]):
                    stuck_streak += 1
                else:
                    stuck_streak = 0
                if in_push:
                    push_steps += 1

                feat = engineer_features(raw, prev_raw, blink_hidden, stuck_streak,
                                          sonar_pers, est_heading, push_steps, obs_contact)

                phase = detect_phase(raw)
                if phase != cur_phase:
                    replays[cur_phase].end_episode()
                    hidden_states[cur_phase] = None
                    cur_phase = phase

                a, hidden_states[phase] = select_action(
                    online[phase], feat, hidden_states[phase], eps, device)

                raw2, r, done = env.step(ACTIONS[a], render=False)
                ep_ret += float(r)
                est_heading += ACTION_HEADING_DELTA[a]

                feat2 = engineer_features(raw2, raw, blink_hidden, stuck_streak,
                                           sonar_pers, est_heading, push_steps, obs_contact)

                replays[phase].store(feat, a, float(r), feat2, done)

                prev_raw = raw.copy()
                raw = raw2
                total_steps += 1

                # Train
                train_step(online[phase], target[phase], replays[phase],
                           optimizers[phase], args.batch, args.gamma, args.grad_clip, device)

                if done:
                    break

            # End episode sequences
            replays[cur_phase].end_episode()
            reward_history.append(ep_ret)

            # Target sync
            if (ep + 1) % args.target_sync == 0:
                for p in PHASES:
                    target[p].load_state_dict(online[p].state_dict())

            # Logging
            if (ep + 1) % 5 == 0:
                avg100 = np.mean(reward_history[-100:])
                elapsed = time.time() - t0
                bufs = " ".join(f"{p[0].upper()}:{len(replays[p])}" for p in PHASES)
                print(f"Ep {ep+1}/{args.episodes}  ret={ep_ret:.0f}  avg100={avg100:.1f}  "
                      f"eps={eps:.3f}  phase=[d={cur_diff} w={cur_walls}]  "
                      f"t={elapsed:.0f}s  buf={bufs}")

            # Save checkpoints
            if (ep + 1) % args.save_every == 0:
                ckpt = args.out.replace(".pth", f"_ep{ep+1}.pth")
                save(ckpt)
                print(f"  [CHECKPOINT] {ckpt}")

            # Save best
            if len(reward_history) >= 100:
                avg = np.mean(reward_history[-100:])
                if avg > best_avg:
                    best_avg = avg
                    save(args.out.replace(".pth", "_best.pth"))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        save(args.out)
        print(f"\n[DONE] Saved final weights: {args.out}")
        print(f"  Total episodes: {min(ep+1, args.episodes)}")
        print(f"  Total steps: {total_steps}")
        print(f"  Best avg100: {best_avg:.1f}")
        print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
