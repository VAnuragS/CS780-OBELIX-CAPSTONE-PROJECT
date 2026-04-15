from __future__ import annotations
import argparse, random, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_FRAMES = 4
N_STEP = 3

PHASE_FIND    = np.array([1, 0, 0], dtype=np.float32)
PHASE_PUSH    = np.array([0, 1, 0], dtype=np.float32)
PHASE_UNWEDGE = np.array([0, 0, 1], dtype=np.float32)

RAW_OBS_DIM = 18
# base(18) + delta(18) + summaries(5) + phase(3) + blink_features(3) = 47
FEAT_DIM = RAW_OBS_DIM + RAW_OBS_DIM + 5 + 3 + 3


def detect_phase(obs):
    if obs[16] == 1:
        return PHASE_PUSH
    elif obs[17] == 1:
        return PHASE_UNWEDGE
    return PHASE_FIND


def engineer_features(obs, prev_obs, blink_hidden_steps=0, stuck_streak=0):
    """47-dim feature vector for difficulty-2 (blinking box + wall obstacles)."""
    base  = obs.astype(np.float32)
    delta = (obs - prev_obs).astype(np.float32)

    left_any  = float(np.any(obs[0:4]))
    fwd_any   = float(np.any(obs[4:12]))
    right_any = float(np.any(obs[12:16]))
    ir_active = float(obs[16])
    stuck     = float(obs[17])

    phase = detect_phase(obs)

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


def createDuelingNetwork(inDim, outDim, hDim=[256, 256, 128], activation=F.relu):

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


def select_action(net, state, epsilon, device, blink_hidden_steps=0):
    """Epsilon-greedy with adaptive exploration boost during blink."""
    effective_eps = epsilon
    if blink_hidden_steps > 5:
        effective_eps = max(epsilon, 0.25)
    if np.random.rand() < effective_eps:
        return np.random.randint(len(ACTIONS))
    with torch.no_grad():
        q = net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    return int(np.argmax(q))


def linear_epsilon(step, eps_start, eps_end, eps_decay_steps):
    if step >= eps_decay_steps:
        return eps_end
    return eps_start + (eps_end - eps_start) * step / eps_decay_steps


def find_obelix_py(given):
    """
    Resolve the path to obelix.py.
    Search order:
      1. Explicit --obelix_py value (if provided)
      2. Same directory as this script
      3. Current working directory
    """
    import pathlib
    candidates = []
    if given:
        candidates.append(pathlib.Path(given))
    candidates.append(pathlib.Path(__file__).parent / "obelix.py")
    candidates.append(pathlib.Path.cwd() / "obelix.py")
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "obelix.py not found. Pass --obelix_py /path/to/obelix.py explicitly."
    )


def import_obelix(obelix_py):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def make_stacked_state(frame_stack):
    return np.concatenate(list(frame_stack))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default=None,
                    help="Path to obelix.py (auto-discovered if omitted)")
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target_sync", type=int, default=50)
    ap.add_argument("--hidden", type=int, nargs='+', default=[128, 128])
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=200000)
    ap.add_argument("--per_alpha", type=float, default=0.6)
    ap.add_argument("--per_beta", type=float, default=0.4)
    ap.add_argument("--per_beta_rate", type=float, default=0.001)
    ap.add_argument("--per_epsilon", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--load_weights", type=str, default=None,
                    help="Path to weights .pth to initialise from (curriculum transfer)")
    ap.add_argument("--curriculum", action="store_true",
                    help="3-phase curriculum: diff=0 → diff=2 → diff=2+walls")
    ap.add_argument("--phase2_ep", type=int, default=600,
                    help="Episode to switch to difficulty=2 in curriculum")
    ap.add_argument("--phase3_ep", type=int, default=1200,
                    help="Episode to enable wall_obstacles in curriculum")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Training on device: {device}")

    OBELIX = import_obelix(find_obelix_py(args.obelix_py))

    # ── Curriculum phase helper ─────────────────────────────────────────────
    def get_curriculum_params(ep):
        if not args.curriculum:
            return args.difficulty, args.wall_obstacles
        if ep < args.phase2_ep:
            return 0, False
        elif ep < args.phase3_ep:
            return 2, False
        else:
            return 2, True

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
    online_net = createDuelingNetwork(in_dim, 5, hDim=args.hidden).to(device)
    target_net = createDuelingNetwork(in_dim, 5, hDim=args.hidden).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    if args.load_weights:
        sd = torch.load(args.load_weights, map_location=device)
        online_net.load_state_dict(sd)
        target_net.load_state_dict(sd)
        print(f"[INFO] Loaded weights from {args.load_weights}")

    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)

    replay = ReplayBuffer(
        bufferSize=args.replay,
        alpha=args.per_alpha,
        beta=args.per_beta,
        beta_increment=args.per_beta_rate,
        epsilon=args.per_epsilon,
    )

    steps = 0
    start_time = time.time()
    gamma_n = args.gamma ** N_STEP

    def save_weights():
        online_net_cpu = online_net.cpu()
        torch.save(online_net_cpu.state_dict(), args.out)
        print(f"Saved (CPU format): {args.out}")

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

            blink_hidden_steps = 0
            stuck_streak       = 0

            feat        = engineer_features(raw, prev_raw, blink_hidden_steps, stuck_streak)
            frame_stack = deque([feat] * N_FRAMES, maxlen=N_FRAMES)
            s           = make_stacked_state(frame_stack)
            ep_ret = 0.0
            n_step_buffer = deque(maxlen=N_STEP)

            for _ in range(args.max_steps):
                eps = linear_epsilon(steps, args.eps_start, args.eps_end, args.eps_decay_steps)
                a = select_action(online_net, s, eps, device, blink_hidden_steps)
                raw2, r, done = env.step(ACTIONS[a], render=False)

                # Update stateful counters
                all_sonar_dark = np.sum(raw2[:16]) == 0
                if all_sonar_dark and not bool(raw2[16]):
                    blink_hidden_steps += 1
                else:
                    blink_hidden_steps = 0
                if bool(raw2[17]):
                    stuck_streak += 1
                else:
                    stuck_streak = 0

                ep_ret += float(r)

                feat2       = engineer_features(raw2, raw, blink_hidden_steps, stuck_streak)
                frame_stack.append(feat2)
                s2          = make_stacked_state(frame_stack)

                n_step_buffer.append((s, a, float(r), s2, bool(done)))

                if len(n_step_buffer) == N_STEP or done:
                    R = 0.0
                    for i in reversed(range(len(n_step_buffer))):
                        R = n_step_buffer[i][2] + args.gamma * R
                    s0 = n_step_buffer[0][0]
                    a0 = n_step_buffer[0][1]
                    sn = n_step_buffer[-1][3]
                    dn = n_step_buffer[-1][4]
                    replay.store(s0, a0, R, sn, dn)

                    if done:
                        while len(n_step_buffer) > 1:
                            n_step_buffer.popleft()
                            R = 0.0
                            for i in reversed(range(len(n_step_buffer))):
                                R = n_step_buffer[i][2] + args.gamma * R
                            s0 = n_step_buffer[0][0]
                            a0 = n_step_buffer[0][1]
                            replay.store(s0, a0, R, sn, dn)

                prev_raw = raw2
                raw      = raw2
                s = s2
                steps += 1

                if len(replay) >= max(args.warmup, args.batch):
                    sb, ab, rb, s2b, db, indices, is_weights = replay.sample(args.batch)

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
                    nn.utils.clip_grad_norm_(online_net.parameters(), args.grad_clip)
                    optimizer.step()

                if done:
                    break

            if (ep + 1) % args.target_sync == 0:
                target_net.load_state_dict(online_net.state_dict())

            if (ep + 1) % 5 == 0:
                elapsed = time.time() - start_time
                phase = f"diff={current_diff} walls={current_walls}"
                print(f"Episode {ep+1}/{args.episodes}  return={ep_ret:.1f}  "
                      f"eps={linear_epsilon(steps, args.eps_start, args.eps_end, args.eps_decay_steps):.3f}  "
                      f"replay={len(replay)}  beta={replay.beta:.3f}  "
                      f"phase=[{phase}]  time={elapsed:.0f}s")

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    finally:
        save_weights()


if __name__ == "__main__":
    main()
