import argparse
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical


class BaselinePolicy(nn.Module):
    def __init__(self, obs_dim=1121, action_dim=91, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
        )
        self.gru    = nn.GRUCell(128, hidden_dim)
        self.actor  = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def initial_state(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs, hidden):
        emb    = self.encoder(obs)
        hidden = self.gru(emb, hidden)
        return self.actor(hidden), self.critic(hidden), hidden


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    T   = len(rewards)
    adv = torch.zeros_like(rewards)
    last = torch.zeros(rewards.shape[1:], device=rewards.device) if rewards.dim() > 1 else torch.tensor(0.0)
    for t in reversed(range(T)):
        nv   = next_value if t == T - 1 else values[t + 1]
        d    = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        last = d + gamma * lam * (1 - dones[t]) * last
        adv[t] = last
    return adv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      type=str,   default="resources/drive/binaries/training")
    p.add_argument("--num_agents",    type=int,   default=16)
    p.add_argument("--num_maps",      type=int,   default=100)
    p.add_argument("--total_steps",   type=int,   default=50_000_000)
    p.add_argument("--rollout_steps", type=int,   default=256)
    p.add_argument("--ppo_epochs",    type=int,   default=4)
    p.add_argument("--num_minibatch", type=int,   default=4)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae_lambda",    type=float, default=0.95)
    p.add_argument("--clip_coef",     type=float, default=0.2)
    p.add_argument("--ent_coef",      type=float, default=0.01)
    p.add_argument("--vf_coef",       type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--save_dir",      type=str,   default="roma_pufferdrive/checkpoints/baseline")
    p.add_argument("--save_interval", type=int,   default=10_000_000)
    p.add_argument("--log_interval",  type=int,   default=50_000)
    p.add_argument("--seed",          type=int,   default=0)
    return p.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    print(f"[BASELINE] Device: {device}")

    from pufferlib.ocean.drive.drive import Drive
    obs_dim    = 1121
    action_dim = 91

    env = Drive(
        num_maps       = args.num_maps,
        num_agents     = args.num_agents,
        map_dir        = args.data_dir,
        episode_length = 91,
    )
    print(f"[BASELINE] Environment ready. obs={obs_dim} actions={action_dim} agents={args.num_agents}")

    policy    = BaselinePolicy(obs_dim, action_dim, 128).to(device)
    optimizer = Adam(policy.parameters(), lr=args.lr)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    B           = args.num_agents
    global_step = 0
    next_save   = args.save_interval
    ep_returns  = deque(maxlen=100)
    ep_scores   = deque(maxlen=100)
    start_time  = time.time()

    N     = args.rollout_steps * B
    b_obs = torch.zeros(N, obs_dim, device=device)
    b_act = torch.zeros(N, dtype=torch.long, device=device)
    b_lp  = torch.zeros(N, device=device)
    b_rew = torch.zeros(N, device=device)
    b_don = torch.zeros(N, device=device)
    b_val = torch.zeros(N, device=device)

    hidden      = policy.initial_state(B, device)
    obs_np, _   = env.reset()
    obs         = torch.tensor(obs_np, dtype=torch.float32, device=device)

    print(f"[BASELINE] Training for {args.total_steps:,} steps ...")

    while global_step < args.total_steps:
        policy.eval()
        ptr = 0
        with torch.no_grad():
            for _ in range(args.rollout_steps):
                logits, value, hidden = policy(obs, hidden)
                dist   = Categorical(logits=logits)
                action = dist.sample()
                logprob= dist.log_prob(action)

                actions_np = action.cpu().numpy().reshape(B, 1)
                step_result = env.step(actions_np)
                obs_np, rew_np, term_np, trunc_np, info = step_result

                obs  = torch.tensor(obs_np,  dtype=torch.float32, device=device)
                rew  = torch.tensor(rew_np,  dtype=torch.float32, device=device)
                done = torch.tensor((term_np | trunc_np), dtype=torch.float32, device=device)

                b_obs[ptr:ptr+B] = obs
                b_act[ptr:ptr+B] = action
                b_lp [ptr:ptr+B] = logprob
                b_rew[ptr:ptr+B] = rew
                b_don[ptr:ptr+B] = done
                b_val[ptr:ptr+B] = value.squeeze(-1)
                ptr += B
                global_step += B

                rm = done.bool()
                if rm.any():
                    hidden = torch.where(rm.unsqueeze(-1), torch.zeros_like(hidden), hidden)

                if isinstance(info, dict):
                    if "score" in info:
                        ep_scores.append(float(np.mean(info["score"])))
                    if "episode_return" in info:
                        ep_returns.append(float(np.mean(info["episode_return"])))

        with torch.no_grad():
            _, last_val, _ = policy(obs, hidden)
        adv     = compute_gae(b_rew[:ptr], b_val[:ptr], b_don[:ptr],
                               last_val.squeeze(-1).mean(), args.gamma, args.gae_lambda)
        returns = adv + b_val[:ptr]
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy.train()
        idx     = torch.randperm(ptr, device=device)
        mb_size = max(1, ptr // args.num_minibatch)

        pl = vl = torch.tensor(0.0)
        for _ in range(args.ppo_epochs):
            for start in range(0, ptr, mb_size):
                mb   = idx[start:start+mb_size]
                init = policy.initial_state(mb.size(0), device)
                logits, value, _ = policy(b_obs[mb], init)
                dist     = Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[mb])
                entropy  = dist.entropy()
                ratio = (new_logp - b_lp[mb]).exp()
                s1 = ratio * adv[mb]
                s2 = ratio.clamp(1-args.clip_coef, 1+args.clip_coef) * adv[mb]
                pl = -torch.min(s1, s2).mean()
                vl = F.mse_loss(value.squeeze(-1), returns[mb])
                loss = pl + args.vf_coef * vl - args.ent_coef * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

        if global_step % args.log_interval < B * args.rollout_steps:
            sps = global_step / (time.time() - start_time)
            msg = (f"[BASELINE step {global_step:>10,}] sps={sps:>6,.0f} "
                   f"policy_loss={pl.item():.4f} value_loss={vl.item():.4f}")
            if ep_scores:
                msg += f" score={np.mean(ep_scores):.3f}"
            if ep_returns:
                msg += f" mean_ret={np.mean(ep_returns):.3f}"
            print(msg)

        if global_step >= next_save:
            path = Path(args.save_dir) / f"baseline_step{global_step}.pt"
            torch.save({"global_step": global_step, "policy": policy.state_dict()}, path)
            print(f"[BASELINE] Saved -> {path}")
            next_save += args.save_interval

    path = Path(args.save_dir) / "baseline_final.pt"
    torch.save({"global_step": global_step, "policy": policy.state_dict()}, path)
    print(f"[BASELINE] Training complete. Saved -> {path}")


if __name__ == "__main__":
    train(parse_args())
