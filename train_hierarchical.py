from __future__ import annotations
import argparse, os, random, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
# Action-to-heading-change mapping (degrees)
ACTION_HEADING_DELTA = {0: 45.0, 1: 22.5, 2: 0.0, 3: -22.5, 4: -45.0}

N_FRAMES = 4
N_STEP = 5  # longer credit assignment for wall navigation
PHASES = ["find", "push", "unwedge"]

PHASE_FIND_VEC    = np.array([1, 0, 0], dtype=np.float32)
PHASE_PUSH_VEC    = np.array([0, 1, 0], dtype=np.float32)
PHASE_UNWEDGE_VEC = np.array([0, 0, 1], dtype=np.float32)

RAW_OBS_DIM = 18
# base(18) + delta(18) + summaries(5) + phase(3) + blink_features(3)
# + sonar_persistence(8) + wall_side(3) + heading(2) + push_progress(1) + obstacle_contact(3)
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


def engineer_features(obs, prev_obs, blink_hidden_steps=0, stuck_streak=0,
                       sonar_persistence=None, estimated_heading=0.0,
                       push_step_count=0, obstacle_contact_hist=None):
    """64-dim POMDP-aware feature vector for Level 3 (blinking box + walls).

    Key additions over the 47-dim version:
      - sonar_persistence (8): per-sonar-pair consecutive detection count (normalized).
        Walls produce persistent readings; the box blinks → resets to 0.
      - wall_side_indicator (3): left/right/fwd persistent obstacle summary.
      - heading (2): sin/cos of estimated heading (helps wall-relative navigation).
      - push_progress (1): normalized steps since push started (urgency signal).
      - obstacle_contact_history (3): recent stuck directions [left, fwd, right].
    """
    base  = obs.astype(np.float32)
    delta = (obs - prev_obs).astype(np.float32)

    # Sonar region summaries
    left_any  = float(np.any(obs[0:4]))
    fwd_any   = float(np.any(obs[4:12]))
    right_any = float(np.any(obs[12:16]))
    ir_active = float(obs[16])
    stuck     = float(obs[17])

    phase = detect_phase_vec(obs)

    # Blink awareness
    prev_sonar = prev_obs[:16]
    was_all_dark = float(np.sum(prev_sonar) == 0 and np.sum(obs[:16]) == 0)
    blink_hidden_norm = float(min(blink_hidden_steps, 30)) / 30.0
    stuck_streak_norm = float(min(stuck_streak, 10)) / 10.0

    # Sonar persistence (normalized, walls stay high, box resets on blink)
    if sonar_persistence is None:
        sonar_persistence = np.zeros(8, dtype=np.float32)
    sp_norm = np.clip(sonar_persistence / 20.0, 0.0, 1.0).astype(np.float32)

    # Wall-side indicators: persistent detections (>5 steps) on each side
    left_persistent  = float(np.any(sonar_persistence[:2] > 5))
    right_persistent = float(np.any(sonar_persistence[6:8] > 5))
    fwd_persistent   = float(np.any(sonar_persistence[2:6] > 5))

    # Heading (sin/cos for continuity)
    heading_rad = np.deg2rad(estimated_heading % 360)
    heading_sin = float(np.sin(heading_rad))
    heading_cos = float(np.cos(heading_rad))

    # Push progress (normalized over max_steps)
    push_progress = float(min(push_step_count, 500)) / 500.0

    # Obstacle contact history
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


def createDuelingNetwork(inDim, outDim, hDim=[512, 256, 256, 128], activation=F.relu):

    class DuelingNetwork(nn.Module):
        def __init__(self, in_d, out_d, h_d, act):
            super(DuelingNetwork, self).__init__()
            self.activation = act
            self.layers = nn.ModuleList()
            self.layer_norms = nn.ModuleList()  # LayerNorm for stability
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


class ReplayBuffer:
    def __init__(self, bufferSize, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.bufferSize = bufferSize
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.size = 0
        self.idx = 0
        self.states = np.empty(self.bufferSize, dtype=object)
        self.actions = np.empty(self.bufferSize, dtype=np.int64)
        self.rewards = np.empty(self.bufferSize, dtype=np.float32)
        self.next_states = np.empty(self.bufferSize, dtype=object)
        self.dones = np.empty(self.bufferSize, dtype=bool)
        self.priorities = np.zeros(self.bufferSize, dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities[:self.size]) if self.size > 0 else 1.0
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        self.priorities[self.idx] = max_priority
        self.idx = (self.idx + 1) % self.bufferSize
        self.size = min(self.size + 1, self.bufferSize)

    def sample(self, batchSize):
        current_priorities = self.priorities[:self.size]
        probabilities = current_priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(self.size, batchSize, replace=False, p=probabilities)
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        s = np.stack([self.states[i] for i in indices]).astype(np.float32)
        a = self.actions[indices]
        r = self.rewards[indices]
        s2 = np.stack([self.next_states[i] for i in indices]).astype(np.float32)
        d = self.dones[indices].astype(np.float32)
        return s, a, r, s2, d, indices, weights.astype(np.float32)

    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority + self.epsilon

    def __len__(self):
        return self.size


def select_action(net, state, epsilon, device, blink_hidden_steps=0, phase="find",
                  stuck_streak=0):
    """Epsilon-greedy with wall-aware exploration.

    - During blink in 'find' phase: boost epsilon to keep searching.
    - When stuck: bias random actions toward turns (avoid FW into wall).
    """
    effective_eps = epsilon
    if phase == "find" and blink_hidden_steps > 5:
        effective_eps = max(epsilon, 0.25)

    if np.random.rand() < effective_eps:
        if stuck_streak > 2:
            # Stuck: strongly prefer turns over forward
            # [L45, L22, FW, R22, R45] → bias away from FW
            probs = np.array([0.3, 0.2, 0.0, 0.2, 0.3])
            probs /= probs.sum()
            return int(np.random.choice(len(ACTIONS), p=probs))
        return np.random.randint(len(ACTIONS))
    with torch.no_grad():
        q = net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    return int(np.argmax(q))


def linear_epsilon(step, eps_start, eps_end, eps_decay_steps):
    if step >= eps_decay_steps:
        return eps_end
    return eps_start + (eps_end - eps_start) * step / eps_decay_steps


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def make_stacked_state(frame_stack):
    return np.concatenate(list(frame_stack))


def train_sub_net(online_net, target_net, replay, optimizer,
                  batch, warmup, gamma_n, grad_clip, device):
    if len(replay) < max(warmup, batch):
        return

    sb, ab, rb, s2b, db, indices, is_weights = replay.sample(batch)

    sb_t = torch.tensor(sb, dtype=torch.float32).to(device)
    ab_t = torch.tensor(ab, dtype=torch.int64).to(device)
    rb_t = torch.tensor(rb, dtype=torch.float32).to(device)
    s2b_t = torch.tensor(s2b, dtype=torch.float32).to(device)
    db_t = torch.tensor(db, dtype=torch.float32).to(device)
    weights_t = torch.tensor(is_weights, dtype=torch.float32).to(device)

    with torch.no_grad():
        next_a = online_net(s2b_t).argmax(dim=1).unsqueeze(1)
        next_val = target_net(s2b_t).gather(1, next_a).squeeze(1)
        targets = rb_t + gamma_n * (1.0 - db_t) * next_val

    current_q = online_net(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
    td_errors = (targets - current_q).detach().cpu().numpy()
    replay.update_priorities(indices, np.abs(td_errors))

    elementwise_loss = F.smooth_l1_loss(current_q, targets, reduction='none')
    loss = (weights_t * elementwise_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), grad_clip)
    optimizer.step()


def update_sonar_persistence(sonar_persistence, obs):
    """Update per-sonar-pair persistence counters.

    8 sonar pairs → indices 0..7, each pair has (far, near) bits.
    If either bit in a pair fires, increment persistence; otherwise reset to 0.
    """
    for i in range(8):
        far_bit  = obs[2 * i]
        near_bit = obs[2 * i + 1]
        if far_bit or near_bit:
            sonar_persistence[i] += 1
        else:
            sonar_persistence[i] = 0
    return sonar_persistence


def update_obstacle_contact_history(obstacle_contact_hist, obs, decay=0.8):
    """Track which directions caused stuck contacts.

    When stuck (obs[17] == 1), increment the direction that has active sonar.
    Decay all channels each step for recency.
    """
    obstacle_contact_hist *= decay
    if obs[17] == 1:
        # Left sensors (sonar pairs 0,1)
        if obs[0] or obs[1] or obs[2] or obs[3]:
            obstacle_contact_hist[0] += 1.0
        # Forward sensors (sonar pairs 2,3,4,5)
        if any(obs[4:12]):
            obstacle_contact_hist[1] += 1.0
        # Right sensors (sonar pairs 6,7)
        if obs[12] or obs[13] or obs[14] or obs[15]:
            obstacle_contact_hist[2] += 1.0
    return obstacle_contact_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights/d3qn/weights_hier_diff3.pth")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=200000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target_sync", type=int, default=40)
    ap.add_argument("--hidden", type=int, nargs='+', default=[512, 256, 256, 128])
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.03)
    ap.add_argument("--eps_decay_steps", type=int, default=500000)
    ap.add_argument("--per_alpha", type=float, default=0.6)
    ap.add_argument("--per_beta", type=float, default=0.4)
    ap.add_argument("--per_beta_rate", type=float, default=0.0005)
    ap.add_argument("--per_epsilon", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--load_weights", type=str, default=None)
    ap.add_argument("--curriculum", action="store_true",
                    help="4-phase curriculum: diff=0 → diff=0+walls → diff=2+walls → diff=3+walls")
    ap.add_argument("--phase2_ep", type=int, default=500,
                    help="Episode to switch to diff=0+walls")
    ap.add_argument("--phase3_ep", type=int, default=1200,
                    help="Episode to switch to diff=2+walls")
    ap.add_argument("--phase4_ep", type=int, default=2000,
                    help="Episode to switch to diff=3+walls (moving+blinking)")
    ap.add_argument("--save_every", type=int, default=200,
                    help="Save checkpoint every N episodes")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Training on device: {device}")

    OBELIX = import_obelix(args.obelix_py)

    def get_curriculum_params(ep):
        if not args.curriculum:
            return args.difficulty, args.wall_obstacles
        if ep < args.phase2_ep:
            return 0, False           # Phase 1: static box, no walls
        elif ep < args.phase3_ep:
            return 0, True            # Phase 2: static box + walls (learn navigation)
        elif ep < args.phase4_ep:
            return 2, True            # Phase 3: blinking box + walls
        else:
            return 3, True            # Phase 4: moving/blinking box + walls

    def make_env(difficulty, seed, wall_obstacles=None):
        walls = wall_obstacles if wall_obstacles is not None else args.wall_obstacles
        return OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=walls,
            difficulty=difficulty,
            box_speed=args.box_speed,
            seed=seed,
        )

    init_diff, init_walls = get_curriculum_params(0)
    env = make_env(init_diff, args.seed, wall_obstacles=init_walls)
    current_diff, current_walls = init_diff, init_walls

    in_dim = FEAT_DIM * N_FRAMES
    print(f"[INFO] FEAT_DIM={FEAT_DIM}  N_FRAMES={N_FRAMES}  in_dim={in_dim}")
    print(f"[INFO] N_STEP={N_STEP}  hidden={args.hidden}")

    online_nets = {p: createDuelingNetwork(in_dim, 5, hDim=args.hidden).to(device) for p in PHASES}
    target_nets = {p: createDuelingNetwork(in_dim, 5, hDim=args.hidden).to(device) for p in PHASES}
    for p in PHASES:
        target_nets[p].load_state_dict(online_nets[p].state_dict())
        target_nets[p].eval()

    if args.load_weights:
        sd = torch.load(args.load_weights, map_location=device)
        for p in PHASES:
            if p in sd:
                try:
                    online_nets[p].load_state_dict(sd[p])
                    target_nets[p].load_state_dict(sd[p])
                    print(f"[INFO] Loaded {p} weights from {args.load_weights}")
                except RuntimeError as e:
                    print(f"[WARN] Could not load {p} weights (dimension mismatch?): {e}")
                    print(f"[WARN] Training {p} from scratch.")

    # Separate learning rates: push/unwedge get slightly higher LR (more wall exposure needed)
    optimizers = {
        "find":    optim.Adam(online_nets["find"].parameters(), lr=args.lr),
        "push":    optim.Adam(online_nets["push"].parameters(), lr=args.lr * 1.5),
        "unwedge": optim.Adam(online_nets["unwedge"].parameters(), lr=args.lr * 2.0),
    }

    replays = {
        p: ReplayBuffer(
            bufferSize=args.replay,
            alpha=args.per_alpha,
            beta=args.per_beta,
            beta_increment=args.per_beta_rate,
            epsilon=args.per_epsilon,
        )
        for p in PHASES
    }

    steps = 0
    start_time = time.time()
    gamma_n = args.gamma ** N_STEP
    phase_counts = {p: 0 for p in PHASES}

    # Track running average return for logging
    recent_returns = deque(maxlen=100)

    def save_weights(suffix=""):
        sd = {p: online_nets[p].cpu().state_dict() for p in PHASES}
        fname = args.out if not suffix else args.out.replace(".pth", f"_{suffix}.pth")
        os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
        torch.save(sd, fname)
        print(f"Saved (CPU format): {fname}")
        for p in PHASES:
            online_nets[p].to(device)

    try:
        for ep in range(args.episodes):

            # ── Curriculum: rebuild env when phase changes ──────────────────
            ep_diff, ep_walls = get_curriculum_params(ep)
            if ep_diff != current_diff or ep_walls != current_walls:
                current_diff, current_walls = ep_diff, ep_walls
                env = make_env(current_diff, args.seed, wall_obstacles=current_walls)
                label = f"diff={current_diff} walls={current_walls}"
                print(f"\n[CURRICULUM] Phase change at ep {ep+1}: {label}\n")

            raw = env.reset(seed=args.seed + ep)
            prev_raw = raw.copy()

            # Per-episode stateful counters
            blink_hidden_steps = 0
            stuck_streak       = 0
            estimated_heading  = 0.0   # relative heading tracker
            push_step_count    = 0
            sonar_persistence  = np.zeros(8, dtype=np.float32)
            obstacle_contact_hist = np.zeros(3, dtype=np.float32)

            # Initialize sonar persistence from first observation
            sonar_persistence = update_sonar_persistence(sonar_persistence, raw)

            feat = engineer_features(
                raw, prev_raw, blink_hidden_steps, stuck_streak,
                sonar_persistence, estimated_heading, push_step_count,
                obstacle_contact_hist
            )
            frame_stack = deque([feat] * N_FRAMES, maxlen=N_FRAMES)
            s = make_stacked_state(frame_stack)
            ep_ret = 0.0
            n_step_buffers = {p: deque(maxlen=N_STEP) for p in PHASES}

            for _ in range(args.max_steps):
                eps = linear_epsilon(steps, args.eps_start, args.eps_end, args.eps_decay_steps)
                phase = detect_phase(raw)
                phase_counts[phase] += 1

                a = select_action(
                    online_nets[phase], s, eps, device,
                    blink_hidden_steps, phase, stuck_streak
                )
                raw2, r, done = env.step(ACTIONS[a], render=False)

                # Update heading estimate
                estimated_heading += ACTION_HEADING_DELTA[a]

                # Update sonar persistence
                sonar_persistence = update_sonar_persistence(sonar_persistence, raw2)

                # Update obstacle contact history
                obstacle_contact_hist = update_obstacle_contact_history(
                    obstacle_contact_hist, raw2
                )

                # Update blink counter
                all_sonar_dark = np.sum(raw2[:16]) == 0
                if all_sonar_dark and not bool(raw2[16]):
                    blink_hidden_steps += 1
                else:
                    blink_hidden_steps = 0

                # Update stuck streak
                if bool(raw2[17]):
                    stuck_streak += 1
                else:
                    stuck_streak = 0

                # Track push duration
                if bool(raw2[16]):
                    push_step_count += 1

                ep_ret += float(r)

                feat2 = engineer_features(
                    raw2, raw, blink_hidden_steps, stuck_streak,
                    sonar_persistence, estimated_heading, push_step_count,
                    obstacle_contact_hist
                )
                frame_stack.append(feat2)
                s2 = make_stacked_state(frame_stack)

                # N-step return
                buf = n_step_buffers[phase]
                buf.append((s, a, float(r), s2, bool(done)))

                if len(buf) == N_STEP or done:
                    R = 0.0
                    for i in reversed(range(len(buf))):
                        R = buf[i][2] + args.gamma * R
                    s0 = buf[0][0]
                    a0 = buf[0][1]
                    sn = buf[-1][3]
                    dn = buf[-1][4]
                    replays[phase].store(s0, a0, R, sn, dn)

                    if done:
                        while len(buf) > 1:
                            buf.popleft()
                            R = 0.0
                            for i in reversed(range(len(buf))):
                                R = buf[i][2] + args.gamma * R
                            s0 = buf[0][0]
                            a0 = buf[0][1]
                            replays[phase].store(s0, a0, R, sn, dn)

                prev_raw = raw2
                raw = raw2
                s = s2
                steps += 1

                # Train the active sub-network
                train_sub_net(
                    online_nets[phase], target_nets[phase],
                    replays[phase], optimizers[phase],
                    args.batch, args.warmup, gamma_n, args.grad_clip, device
                )

                if done:
                    break

            recent_returns.append(ep_ret)

            # Target net sync
            if (ep + 1) % args.target_sync == 0:
                for p in PHASES:
                    target_nets[p].load_state_dict(online_nets[p].state_dict())

            # Periodic checkpoint
            if (ep + 1) % args.save_every == 0:
                save_weights(suffix=f"ep{ep+1}")

            # Logging
            if (ep + 1) % 5 == 0:
                elapsed = time.time() - start_time
                eps_cur = linear_epsilon(steps, args.eps_start, args.eps_end, args.eps_decay_steps)
                buf_sizes = {p: len(replays[p]) for p in PHASES}
                phase_label = f"diff={current_diff} walls={current_walls}"
                avg_ret = np.mean(recent_returns) if recent_returns else 0.0
                print(f"Episode {ep+1}/{args.episodes}  return={ep_ret:.1f}  "
                      f"avg100={avg_ret:.1f}  eps={eps_cur:.3f}  "
                      f"phase=[{phase_label}]  time={elapsed:.0f}s  "
                      f"buffers=F:{buf_sizes['find']} P:{buf_sizes['push']} U:{buf_sizes['unwedge']}")

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    finally:
        save_weights()


if __name__ == "__main__":
    main()
