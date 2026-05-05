"""
train_roma.py
=============
Train the ROMA policy on PufferDrive.

Usage (from ~/PufferDrive):
    # Original ROMA — scalar role
    python3 roma_pufferdrive/train_roma.py --role_dim 1

    # Proposed extension — multidimensional role
    python3 roma_pufferdrive/train_roma.py --role_dim 8

IMPORTANT: Run check_env.py first and pass the printed obs_dim here
if it differs from the default 1848:
    python3 roma_pufferdrive/train_roma.py --obs_dim <VALUE> --role_dim 8
"""

import argparse
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical

# Import the shared policy — NOT a local inline copy
from roma.policy    import RomaPolicy
from roma.aux_losses import RomaAuxLoss


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    T   = len(rewards)
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        nv   = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        last  = delta + gamma * lam * (1 - dones[t]) * last
        adv[t] = last
    return adv


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Environment
    p.add_argument("--data_dir",      type=str,   default="resources/drive/binaries/training")
    p.add_argument("--num_agents",    type=int,   default=16)
    p.add_argument("--num_maps",      type=int,   default=100)
    p.add_argument("--obs_dim",       type=int,   default=1848,
                   help="Flat observation dimension — run check_env.py to verify")
    # Role
    p.add_argument("--role_dim",      type=int,   default=8)
    p.add_argument("--role_hidden",   type=int,   default=64)
    p.add_argument("--policy_hidden", type=int,   default=128)
    p.add_argument("--var_floor",     type=float, default=1e-4)
    p.add_argument("--mi_weight",     type=float, default=1.0)
    p.add_argument("--div_weight",    type=float, default=5e-2)
    # PPO
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
    # Logging / saving
    p.add_argument("--save_dir",      type=str,   default="roma_pufferdrive/checkpoints/roma")
    p.add_argument("--save_interval", type=int,   default=10_000_000)
    p.add_argument("--log_interval",  type=int,   default=50_000)
    p.add_argument("--seed",          type=int,   default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    print(f"[ROMA] obs_dim={args.obs_dim}  role_dim={args.role_dim}  "
          f"mi_weight={args.mi_weight}  div_weight={args.div_weight}")

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(
        num_maps       = args.num_maps,
        num_agents     = args.num_agents,
        map_dir        = args.data_dir,
        episode_length = 91,
    )

    # --- Verify obs_dim matches environment ---
    obs_probe, _ = env.reset()
    actual_obs_dim = obs_probe.shape[-1]
    if actual_obs_dim != args.obs_dim:
        print(f"[ROMA] WARNING: --obs_dim {args.obs_dim} but env returned {actual_obs_dim}. "
              f"Overriding to {actual_obs_dim}. Run check_env.py to confirm.")
        args.obs_dim = actual_obs_dim

    action_dim = 91
    B = args.num_agents

    # Build policy (structured encoders from roma/policy.py)
    policy = RomaPolicy(
        obs_dim        = args.obs_dim,
        action_dim     = action_dim,
        role_dim       = args.role_dim,
        role_hidden    = args.role_hidden,
        policy_hidden  = args.policy_hidden,
        var_floor      = args.var_floor,
        obs_window_len = 8,
    ).to(device)

    aux_loss_fn = RomaAuxLoss(
        role_dim   = args.role_dim,
        obs_dim    = args.obs_dim,
        mi_weight  = args.mi_weight,
        div_weight = args.div_weight,
    ).to(device)

    optimizer = Adam(
        list(policy.parameters()) + list(aux_loss_fn.parameters()),
        lr=args.lr,
    )
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Rollout buffers
    N        = args.rollout_steps * B
    b_obs    = torch.zeros(N, args.obs_dim,    device=device)
    b_act    = torch.zeros(N, dtype=torch.long, device=device)
    b_lp     = torch.zeros(N,                  device=device)
    b_rew    = torch.zeros(N,                  device=device)
    b_don    = torch.zeros(N,                  device=device)
    b_val    = torch.zeros(N,                  device=device)
    b_rz     = torch.zeros(N, args.role_dim,   device=device)
    b_rmean  = torch.zeros(N, args.role_dim,   device=device)
    b_rlv    = torch.zeros(N, args.role_dim,   device=device)
    b_obswin = torch.zeros(N, 8, args.obs_dim, device=device)

    state   = policy.initial_state(B, device)
    obs     = torch.tensor(obs_probe, dtype=torch.float32, device=device)

    global_step = 0
    next_save   = args.save_interval
    ep_scores   = deque(maxlen=100)
    ep_returns  = deque(maxlen=100)
    start_time  = time.time()

    print(f"[ROMA] Training for {args.total_steps:,} steps ...")

    while global_step < args.total_steps:
        # ---- Rollout ----
        policy.eval()
        ptr = 0
        with torch.no_grad():
            for _ in range(args.rollout_steps):
                logits, value, state, role_info = policy(obs, state)
                dist    = Categorical(logits=logits)
                action  = dist.sample()
                logprob = dist.log_prob(action)

                actions_np = action.cpu().numpy().reshape(B, 1)
                obs_np, rew_np, term_np, trunc_np, info = env.step(actions_np)

                obs  = torch.tensor(obs_np,  dtype=torch.float32, device=device)
                rew  = torch.tensor(rew_np,  dtype=torch.float32, device=device)
                done = torch.tensor((term_np | trunc_np), dtype=torch.float32, device=device)

                b_obs   [ptr:ptr+B] = obs
                b_act   [ptr:ptr+B] = action
                b_lp    [ptr:ptr+B] = logprob
                b_rew   [ptr:ptr+B] = rew
                b_don   [ptr:ptr+B] = done
                b_val   [ptr:ptr+B] = value.squeeze(-1)
                b_rz    [ptr:ptr+B] = role_info["role_z"]
                b_rmean [ptr:ptr+B] = role_info["role_mean"]
                b_rlv   [ptr:ptr+B] = role_info["role_log_var"]
                b_obswin[ptr:ptr+B] = role_info["obs_window"]
                ptr += B
                global_step += B

                # Reset hidden states for finished episodes
                rm = done.bool()
                if rm.any():
                    state = (
                        torch.where(rm.unsqueeze(-1),
                                    torch.zeros_like(state[0]), state[0]),
                        torch.where(rm.unsqueeze(-1),
                                    torch.zeros_like(state[1]), state[1]),
                        torch.where(rm.unsqueeze(-1).unsqueeze(-1),
                                    torch.zeros_like(state[2]), state[2]),
                    )

                if isinstance(info, dict):
                    if "score" in info:
                        ep_scores.append(float(np.mean(info["score"])))
                    if "episode_return" in info:
                        ep_returns.append(float(np.mean(info["episode_return"])))

        # ---- Compute returns and advantages ----
        with torch.no_grad():
            _, last_val, _, _ = policy(obs, state)
        adv     = compute_gae(b_rew[:ptr], b_val[:ptr], b_don[:ptr],
                               last_val.squeeze(-1).mean(), args.gamma, args.gae_lambda)
        returns = adv + b_val[:ptr]
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---- PPO update ----
        policy.train()
        aux_loss_fn.train()
        idx     = torch.randperm(ptr, device=device)
        mb_size = max(1, ptr // args.num_minibatch)

        pl  = vl  = torch.tensor(0.0)
        aux = {"mi_loss": torch.tensor(0.0), "div_loss": torch.tensor(0.0)}

        for _ in range(args.ppo_epochs):
            for start in range(0, ptr, mb_size):
                mb       = idx[start:start + mb_size]
                mb_state = policy.initial_state(mb.size(0), device)
                logits, value, _, role_info = policy(b_obs[mb], mb_state)

                dist        = Categorical(logits=logits)
                new_logprob = dist.log_prob(b_act[mb])
                entropy     = dist.entropy()

                ratio = (new_logprob - b_lp[mb]).exp()
                s1 = ratio * adv[mb]
                s2 = ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef) * adv[mb]
                pl  = -torch.min(s1, s2).mean()
                vl  = F.mse_loss(value.squeeze(-1), returns[mb])
                aux = aux_loss_fn(b_rz[mb], b_rmean[mb], b_rlv[mb], b_obswin[mb])

                loss = (pl
                        + args.vf_coef * vl
                        - args.ent_coef * entropy.mean()
                        + aux["aux_loss"])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

        # ---- Logging ----
        if global_step % args.log_interval < B * args.rollout_steps:
            sps = global_step / (time.time() - start_time)
            msg = (f"[ROMA step {global_step:>10,}] sps={sps:>6,.0f} "
                   f"policy_loss={pl.item():.4f}  value_loss={vl.item():.4f}  "
                   f"mi_loss={aux['mi_loss'].item():.4f}  "
                   f"div_loss={aux['div_loss'].item():.4f}")
            if ep_scores:
                msg += f"  score={np.mean(ep_scores):.3f}"
            if ep_returns:
                msg += f"  return={np.mean(ep_returns):.3f}"
            print(msg)

        # ---- Checkpoint ----
        if global_step >= next_save:
            ckpt = Path(args.save_dir) / f"roma_dim{args.role_dim}_step{global_step}.pt"
            torch.save({
                "global_step"   : global_step,
                "policy_state"  : policy.state_dict(),
                "aux_loss_state": aux_loss_fn.state_dict(),
                "args"          : vars(args),
            }, ckpt)
            print(f"[ROMA] Saved -> {ckpt}")
            next_save += args.save_interval

    # ---- Final save ----
    final = Path(args.save_dir) / f"roma_dim{args.role_dim}_final.pt"
    torch.save({
        "global_step" : global_step,
        "policy_state": policy.state_dict(),
        "args"        : vars(args),
    }, final)
    print(f"[ROMA] Training complete. Saved -> {final}")


if __name__ == "__main__":
    train(parse_args())
