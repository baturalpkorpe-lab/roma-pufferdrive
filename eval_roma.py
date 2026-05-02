"""
eval_roma.py
============
Evaluate a trained ROMA or baseline checkpoint on PufferDrive.
Prints metrics table and optionally saves t-SNE and correlation plots.

Usage
-----
    cd ~/PufferDrive
    PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 roma_pufferdrive/eval_roma.py \
        --checkpoint roma_pufferdrive/checkpoints/roma/roma_dim8_final.pt \
        --role_dim 8 \
        --save_plots
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.distributions import Categorical
from pufferlib.ocean.drive.drive import Drive


class RomaPolicy(nn.Module):
    def __init__(self, obs_dim=1121, action_dim=91, role_dim=8,
                 role_hidden=64, policy_hidden=128, var_floor=1e-4,
                 obs_window_len=8):
        super().__init__()
        self.role_hidden    = role_hidden
        self.policy_hidden  = policy_hidden
        self.obs_window_len = obs_window_len
        self.var_floor      = var_floor
        self.role_dim       = role_dim
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
        )
        self.role_fc     = nn.Sequential(nn.Linear(obs_dim, role_hidden), nn.ReLU())
        self.role_gru    = nn.GRUCell(role_hidden, role_hidden)
        self.mu_head     = nn.Linear(role_hidden, role_dim)
        self.logvar_head = nn.Linear(role_hidden, role_dim)
        self.policy_gru  = nn.GRUCell(128 + role_dim, policy_hidden)
        self.actor       = nn.Linear(policy_hidden, action_dim)
        self.critic      = nn.Linear(policy_hidden, 1)

    def initial_state(self, B, device):
        return (
            torch.zeros(B, self.role_hidden,    device=device),
            torch.zeros(B, self.policy_hidden,  device=device),
            torch.zeros(B, self.obs_window_len, 1121, device=device),
        )

    def forward(self, obs, state):
        role_h, policy_h, obs_win = state
        new_role_h   = self.role_gru(self.role_fc(obs), role_h)
        role_z       = self.mu_head(new_role_h)
        env_emb      = self.obs_encoder(obs)
        new_policy_h = self.policy_gru(torch.cat([env_emb, role_z], dim=-1), policy_h)
        logits       = self.actor(new_policy_h)
        value        = self.critic(new_policy_h)
        new_obs_win  = torch.cat([obs_win[:, 1:, :], obs.unsqueeze(1)], dim=1)
        return logits, value, (new_role_h, new_policy_h, new_obs_win), role_z


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--role_dim",    type=int, default=8)
    p.add_argument("--num_agents",  type=int, default=16)
    p.add_argument("--n_episodes",  type=int, default=30)
    p.add_argument("--map_dir",     type=str, default="resources/drive/binaries/training")
    p.add_argument("--save_plots",  action="store_true")
    p.add_argument("--output_dir",  type=str, default="roma_pufferdrive/eval_results")
    return p.parse_args()


def evaluate(args):
    device = torch.device("cpu")

    policy = RomaPolicy(role_dim=args.role_dim)
    ckpt   = torch.load(args.checkpoint, map_location=device)
    key    = "policy_state" if "policy_state" in ckpt else "policy"
    policy.load_state_dict(ckpt[key])
    policy.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    env = Drive(
        num_maps       = 50,
        num_agents     = args.num_agents,
        map_dir        = args.map_dir,
        episode_length = 91,
    )

    all_scores, all_collisions  = [], []
    all_offroads, all_completions = [], []
    all_returns   = []
    all_roles     = []
    all_speeds    = []

    print(f"Running {args.n_episodes} evaluation episodes...")
    for ep in range(args.n_episodes):
        obs_np, _ = env.reset()
        obs       = torch.tensor(obs_np, dtype=torch.float32)
        state     = policy.initial_state(args.num_agents, device)

        for step in range(91):
            with torch.no_grad():
                logits, _, state, role_z = policy(obs, state)
            action     = Categorical(logits=logits).sample()
            actions_np = action.numpy().reshape(args.num_agents, 1)
            obs_np, rew_np, term_np, trunc_np, info = env.step(actions_np)
            obs = torch.tensor(obs_np, dtype=torch.float32)

            all_roles.append(role_z.numpy())
            all_speeds.append(obs_np[:, 2])

            if isinstance(info, list) and info:
                for item in info:
                    if "score" in item:
                        all_scores.append(item["score"])
                        all_collisions.append(item["collision_rate"])
                        all_offroads.append(item["offroad_rate"])
                        all_completions.append(item["completion_rate"])
                        all_returns.append(item["episode_return"])
                        print(f"  Ep {ep+1:>3} | score={item['score']:.3f} | "
                              f"collision={item['collision_rate']:.3f} | "
                              f"offroad={item['offroad_rate']:.3f} | "
                              f"completion={item['completion_rate']:.3f} | "
                              f"return={item['episode_return']:.2f}")

    print("\n" + "="*55)
    print("  EVALUATION RESULTS")
    print("="*55)
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Role dim       : {args.role_dim}")
    print(f"  Episodes       : {len(all_scores)}")
    print(f"  Score          : {np.mean(all_scores):.4f}")
    print(f"  Collision rate : {np.mean(all_collisions):.4f}")
    print(f"  Off-road rate  : {np.mean(all_offroads):.4f}")
    print(f"  Completion rate: {np.mean(all_completions):.4f}")
    print(f"  Mean return    : {np.mean(all_returns):.4f}")
    print("="*55)

    if args.role_dim > 1 and all_roles:
        role_matrix = np.concatenate(all_roles,  axis=0)
        speed_vec   = np.concatenate(all_speeds, axis=0)
        print("\n  Role dimension correlations with speed:")
        print(f"  {'Dim':<6} {'Pearson r':>10}")
        from scipy.stats import pearsonr
        for d in range(args.role_dim):
            r, _ = pearsonr(role_matrix[:, d], speed_vec)
            print(f"  z[{d}]  {r:>10.4f}")

        if args.save_plots:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from sklearn.manifold import TSNE

                idx = np.random.choice(len(role_matrix), min(3000, len(role_matrix)), replace=False)
                emb = TSNE(n_components=2, random_state=42).fit_transform(role_matrix[idx])

                fig, ax = plt.subplots(figsize=(8, 6))
                sc = ax.scatter(emb[:, 0], emb[:, 1], c=speed_vec[idx],
                                cmap="viridis", alpha=0.5, s=5)
                plt.colorbar(sc, ax=ax, label="Agent speed")
                ax.set_title(f"t-SNE of role vectors (role_dim={args.role_dim})")
                out = Path(args.output_dir) / f"tsne_dim{args.role_dim}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"\n  t-SNE plot saved -> {out}")
            except ImportError:
                print("  Install matplotlib and scikit-learn for plots")


if __name__ == "__main__":
    evaluate(parse_args())
