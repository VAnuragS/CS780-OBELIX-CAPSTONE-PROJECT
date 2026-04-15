from __future__ import annotations
import argparse, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
PHASES = ["find", "push", "unwedge"]

PHASE_FIND_VEC    = np.array([1, 0, 0], dtype=np.float32)
PHASE_PUSH_VEC    = np.array([0, 1, 0], dtype=np.float32)
PHASE_UNWEDGE_VEC = np.array([0, 0, 1], dtype=np.float32)


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
    """26-dim Pure feature vector for DRQN. No memory trackers added. LSTM does the memory."""
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
    
FEAT_DIM = 26

class DuelingDRQN(nn.Module):
    def __init__(self, in_d, out_d, h_d=128):
        super(DuelingDRQN, self).__init__()
        self.h_d = h_d
        
        self.fc1 = nn.Linear(in_d, 256)
        # batch_first=True means tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=256, hidden_size=h_d, batch_first=True)
        
        self.value_head = nn.Linear(h_d, 1)
        self.adv_head = nn.Linear(h_d, out_d)

    def forward(self, x, hidden_state=None):
        """
        x: (batch, seq_len, in_d) or (seq_len, in_d)
        """
        # If unbatched (evaluation), add batch dim
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

class SequentialReplayBuffer:
    def __init__(self, capacity, seq_len=8):
        self.capacity = capacity
        self.seq_len = seq_len
        self.episodes = []
        self.current_episode = []
        self.total_frames = 0
        
    def store_transition(self, s, a, r, s2, d):
        self.current_episode.append((s, a, r, s2, d))
        
    def end_episode(self):
        if len(self.current_episode) >= self.seq_len:
            self.episodes.append(self.current_episode)
            self.total_frames += len(self.current_episode)
            
            while self.total_frames > self.capacity and len(self.episodes) > 1:
                removed_ep = self.episodes.pop(0)
                self.total_frames -= len(removed_ep)
                
        self.current_episode = []
        
    def sample(self, batch_size):
        s_batch, a_batch, r_batch, s2_batch, d_batch = [], [], [], [], []
        
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            start_idx = random.randint(0, len(ep) - self.seq_len)
            
            seq = ep[start_idx : start_idx + self.seq_len]
            
            s_batch.append([x[0] for x in seq])
            a_batch.append([x[1] for x in seq])
            r_batch.append([x[2] for x in seq])
            s2_batch.append([x[3] for x in seq])
            d_batch.append([x[4] for x in seq])
            
        return (np.array(s_batch, dtype=np.float32), 
                np.array(a_batch, dtype=np.int64),
                np.array(r_batch, dtype=np.float32),
                np.array(s2_batch, dtype=np.float32),
                np.array(d_batch, dtype=np.float32))

    def __len__(self):
        return self.total_frames

def train_drqn(online_net, target_net, replay, optimizer, batch_size, gamma, grad_clip, device):
    if len(replay.episodes) < batch_size:
        return
        
    sb, ab, rb, s2b, db = replay.sample(batch_size)
    
    sb_t = torch.tensor(sb, dtype=torch.float32).to(device)
    ab_t = torch.tensor(ab, dtype=torch.int64).to(device).unsqueeze(-1)
    rb_t = torch.tensor(rb, dtype=torch.float32).to(device)
    s2b_t = torch.tensor(s2b, dtype=torch.float32).to(device)
    db_t = torch.tensor(db, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        next_q, _ = online_net(s2b_t)
        next_a = next_q.argmax(dim=-1, keepdim=True)
        
        target_q_all, _ = target_net(s2b_t)
        next_val = target_q_all.gather(-1, next_a).squeeze(-1)
        
        targets = rb_t + gamma * (1.0 - db_t) * next_val
        
    current_q_all, _ = online_net(sb_t)
    current_q = current_q_all.gather(-1, ab_t).squeeze(-1)
    
    loss = F.smooth_l1_loss(current_q, targets)
    
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), grad_clip)
    optimizer.step()

def linear_epsilon(step, eps_start, eps_end, eps_decay_steps):
    if step >= eps_decay_steps:
        return eps_end
    return eps_start + (eps_end - eps_start) * step / eps_decay_steps

def select_action(net, s, hidden, eps, device):
    if np.random.rand() < eps:
        a = np.random.randint(len(ACTIONS))
        # Need to advance hidden state anyway so we don't break BPTT inference
        with torch.no_grad():
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            _, next_hidden = net(s_t, hidden)
        return a, next_hidden
        
    with torch.no_grad():
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        q, next_hidden = net(s_t, hidden)
    a = int(q.squeeze().argmax().cpu().numpy())
    return a, next_hidden

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights/drqn/weights_drqn_diff3.pth")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=8, help="Length of recurrent sequence to sample")
    ap.add_argument("--replay", type=int, default=500000, help="Capacity in frames")
    ap.add_argument("--target_sync", type=int, default=40)
    ap.add_argument("--hidden_dim", type=int, default=128)
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
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Training DRQN on device: {device}")

    OBELIX = import_obelix(args.obelix_py)

    def get_curriculum_params(ep):
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

    # Hierarchical set of networks
    online_nets = {p: DuelingDRQN(FEAT_DIM, 5, h_d=args.hidden_dim).to(device) for p in PHASES}
    target_nets = {p: DuelingDRQN(FEAT_DIM, 5, h_d=args.hidden_dim).to(device) for p in PHASES}
    for p in PHASES:
        target_nets[p].load_state_dict(online_nets[p].state_dict())
        target_nets[p].eval()

    optimizers = {
        "find":    optim.Adam(online_nets["find"].parameters(), lr=args.lr),
        "push":    optim.Adam(online_nets["push"].parameters(), lr=args.lr),
        "unwedge": optim.Adam(online_nets["unwedge"].parameters(), lr=args.lr),
    }

    replays = {
        p: SequentialReplayBuffer(capacity=args.replay, seq_len=args.seq_len)
        for p in PHASES
    }

    steps = 0
    start_time = time.time()

    def save_weights():
        sd = {p: online_nets[p].cpu().state_dict() for p in PHASES}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(sd, args.out)
        print(f"Saved (CPU format): {args.out}")
        for p in PHASES:
            online_nets[p].to(device)

    try:
        for ep in range(args.episodes):
            ep_diff, ep_walls = get_curriculum_params(ep)
            if ep_diff != current_diff or ep_walls != current_walls:
                current_diff, current_walls = ep_diff, ep_walls
                env = make_env(current_diff, args.seed, wall_obstacles=current_walls)
                print(f"\n[CURRICULUM] Phase change at ep {ep+1}: diff={current_diff} walls={current_walls}\n")

            raw = env.reset(seed=args.seed + ep)
            s = engineer_features_pure(raw)
            ep_ret = 0.0
            
            # DRQN requires tracking hidden state
            hidden_states = {p: None for p in PHASES}
            current_phase = detect_phase(raw)

            for _ in range(args.max_steps):
                eps = linear_epsilon(steps, args.eps_start, args.eps_end, args.eps_decay_steps)
                phase = detect_phase(raw)
                
                # Phase Transition Boundary: End sequences for sub-replay buffers
                if phase != current_phase:
                    replays[current_phase].end_episode()
                    hidden_states[current_phase] = None
                    current_phase = phase

                a, hidden_states[phase] = select_action(online_nets[phase], s, hidden_states[phase], eps, device)
                raw2, r, done = env.step(ACTIONS[a], render=False)

                ep_ret += float(r)
                s2 = engineer_features_pure(raw2)

                replays[phase].store_transition(s, a, float(r), s2, bool(done))

                s = s2
                raw = raw2
                steps += 1

                train_drqn(
                    online_nets[phase], target_nets[phase],
                    replays[phase], optimizers[phase],
                    args.batch, args.gamma, args.grad_clip, device
                )

                if done:
                    break
            
            # End all open sequences at end of episode
            replays[current_phase].end_episode()

            if (ep + 1) % args.target_sync == 0:
                for p in PHASES:
                    target_nets[p].load_state_dict(online_nets[p].state_dict())

            if (ep + 1) % 5 == 0:
                elapsed = time.time() - start_time
                eps_cur = linear_epsilon(steps, args.eps_start, args.eps_end, args.eps_decay_steps)
                buf_sizes = {p: len(replays[p]) for p in PHASES}
                print(f"Episode {ep+1}/{args.episodes}  return={ep_ret:.1f}  "
                      f"eps={eps_cur:.3f}  time={elapsed:.0f}s  "
                      f"frames=F:{buf_sizes['find']} P:{buf_sizes['push']} U:{buf_sizes['unwedge']}")

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    finally:
        save_weights()

if __name__ == "__main__":
    main()
