"""
eval_roma.py
============
Evaluate a trained ROMA or baseline checkpoint on PufferDrive.

Computes two sets of metrics:
  1. Environment metrics  — score, collision rate, off-road rate, completion rate
  2. WOSAC realism metrics — kinematic, interactive, map-based log-likelihoods
                             and the realism meta-score

Automatically detects whether the checkpoint used the old flat-MLP
architecture or the new structured encoder architecture.

Usage (from ~/PufferDrive):
─────────────────────────────────────────────────────────
CPU TESTING (quick):
    PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 roma_pufferdrive/eval_roma.py \
        --checkpoint roma_pufferdrive/checkpoints/roma_structured/roma_dim8_final.pt \
        --role_dim 8 --obs_dim 1121 --n_episodes 10 \
        --wosac --wosac_rollouts 4 --wosac_num_maps 100 --wosac_max_batches 30

DELFT / SUPERCOMPUTER (full evaluation on all 10,000 scenarios):
    PYTHONPATH=/path/to/roma_pufferdrive python3 roma_pufferdrive/eval_roma.py \
        --checkpoint roma_pufferdrive/checkpoints/roma/roma_dim8_final.pt \
        --role_dim 8 --obs_dim 1121 --n_episodes 30 \
        --wosac --wosac_rollouts 32 --wosac_num_maps 10000 --wosac_max_batches 500
─────────────────────────────────────────────────────────
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Legacy flat-MLP architectures (match checkpoints trained before the fix)
# ---------------------------------------------------------------------------

class LegacyRomaPolicy(nn.Module):
    """Flat-MLP ROMA policy — matches checkpoints trained before the fix."""
    def __init__(self, obs_dim=1121, action_dim=91, role_dim=8,
                 role_hidden=64, policy_hidden=128, obs_window_len=8):
        super().__init__()
        self.role_hidden    = role_hidden
        self.policy_hidden  = policy_hidden
        self.obs_window_len = obs_window_len
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
        return logits, value, (new_role_h, new_policy_h, new_obs_win), {"role_z": role_z}


class LegacyBaselinePolicy(nn.Module):
    """Flat-MLP baseline — matches baseline checkpoints."""
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

    def initial_state(self, B, device):
        return (
            torch.zeros(B, self.hidden_dim, device=device),
            torch.zeros(B, 8, 1121,         device=device),
        )

    def forward(self, obs, state):
        hidden, obs_win = state
        hidden      = self.gru(self.encoder(obs), hidden)
        logits      = self.actor(hidden)
        value       = self.critic(hidden)
        new_obs_win = torch.cat([obs_win[:, 1:, :], obs.unsqueeze(1)], dim=1)
        return logits, value, (hidden, new_obs_win), {"role_z": torch.zeros(obs.size(0), 1)}


# ---------------------------------------------------------------------------
# WOSAC adapter
# ---------------------------------------------------------------------------

class WOSACPolicyAdapter:
    """Wraps our recurrent policy to match WOSACEvaluator's per-step interface."""
    def __init__(self, policy, num_agents, device):
        self.policy     = policy
        self.num_agents = num_agents
        self.device     = device
        self._state     = None

    def reset_state(self):
        self._state = self.policy.initial_state(self.num_agents, self.device)

    def forward_eval(self, obs, lstm_state=None):
        with torch.no_grad():
            logits, value, new_state, _ = self.policy(obs, self._state)
        self._state = new_state
        return logits, value


# ---------------------------------------------------------------------------
# Auto-detect architecture from checkpoint keys
# ---------------------------------------------------------------------------

def load_policy(checkpoint_path, role_dim, obs_dim, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    key  = "policy_state" if "policy_state" in ckpt else "policy"
    sd   = ckpt[key]
    keys = set(sd.keys())

    if any("ego_enc" in k or "role_encoder" in k for k in keys):
        print("  Architecture : new structured encoder (RomaPolicy)")
        from roma.policy import RomaPolicy
        policy = RomaPolicy(obs_dim=obs_dim, role_dim=role_dim)
    elif any("role_fc" in k for k in keys):
        print("  Architecture : legacy flat-MLP ROMA (LegacyRomaPolicy)")
        policy = LegacyRomaPolicy(obs_dim=obs_dim, role_dim=role_dim)
    else:
        print("  Architecture : legacy flat-MLP baseline (LegacyBaselinePolicy)")
        policy = LegacyBaselinePolicy(obs_dim=obs_dim)

    policy.load_state_dict(sd)
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# WOSAC trajectory collection
# ---------------------------------------------------------------------------

def collect_wosac_trajectories(env, policy_adapter, num_rollouts, num_steps=91, silent=False):
    """
    Roll out policy for num_rollouts independent rollouts and collect
    (x, y, z, heading, id) trajectories for WOSACEvaluator.compute_metrics().

    Returns dict with each key shaped (num_agents, num_rollouts, num_steps).
    """
    num_agents = env.num_agents
    trajectories = {
        "x":       np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32),
        "y":       np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32),
        "z":       np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32),
        "heading": np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32),
        "id":      np.zeros((num_agents, num_rollouts, num_steps), dtype=np.int32),
    }

    for r in range(num_rollouts):
        if not silent:
            print(f"\r  WOSAC rollout {r+1}/{num_rollouts} ...", end="", flush=True)
        obs_np, _ = env.reset()
        policy_adapter.reset_state()
        obs = torch.tensor(obs_np, dtype=torch.float32, device=policy_adapter.device)

        for t in range(num_steps):
            agent_state = env.get_global_agent_state()
            trajectories["x"]      [:, r, t] = agent_state["x"]
            trajectories["y"]      [:, r, t] = agent_state["y"]
            trajectories["z"]      [:, r, t] = agent_state.get("z", np.zeros(num_agents))
            trajectories["heading"][:, r, t] = agent_state["heading"]
            trajectories["id"]     [:, r, t] = agent_state["id"]

            logits, _ = policy_adapter.forward_eval(obs)
            action     = Categorical(logits=logits).sample()
            obs_np, _, _, _, _ = env.step(action.cpu().numpy().reshape(num_agents, 1))
            obs = torch.tensor(obs_np, dtype=torch.float32, device=policy_adapter.device)

    if not silent:
        print()
    return trajectories


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",        type=str,  required=True)
    p.add_argument("--role_dim",          type=int,  default=8,
                   help="Role dimension used during training. Use 0 for baseline.")
    p.add_argument("--obs_dim",           type=int,  default=1121)
    p.add_argument("--num_agents",        type=int,  default=16)
    p.add_argument("--n_episodes",        type=int,  default=30,
                   help="Episodes for environment metric evaluation.")
    p.add_argument("--map_dir",           type=str,  default="resources/drive/binaries/training")
    p.add_argument("--wosac",             action="store_true",
                   help="Also compute WOSAC realism metrics.")
    p.add_argument("--wosac_rollouts",    type=int,  default=32,
                   help="Rollouts per scenario. Official WOSAC = 32. Use 4 for quick CPU test.")
    p.add_argument("--wosac_num_maps",    type=int,  default=10000,
                   help="Maps loaded into memory. Use 10000 for full evaluation, 100 for CPU test.")
    p.add_argument("--wosac_max_batches", type=int,  default=500,
                   help="Max resample batches. 500 batches with num_maps=10000 covers most scenarios.")
    p.add_argument("--save_plots",        action="store_true")
    p.add_argument("--output_dir",        type=str,  default="roma_pufferdrive/eval_results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cpu")

    print(f"\nCheckpoint : {args.checkpoint}")
    policy = load_policy(args.checkpoint, max(args.role_dim, 1), args.obs_dim, device)
    print(f"obs_dim    : {args.obs_dim}  |  role_dim : {args.role_dim}")

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(
        num_maps       = args.wosac_num_maps,
        num_agents     = args.num_agents,
        map_dir        = args.map_dir,
        episode_length = 91,
    )

    # -----------------------------------------------------------------------
    # Part 1 — Environment metrics
    # -----------------------------------------------------------------------
    all_scores, all_collisions    = [], []
    all_offroads, all_completions = [], []
    all_returns                   = []
    all_roles, all_speeds         = [], []

    print(f"\nPart 1: Environment metrics ({args.n_episodes} episodes) ...\n")
    obs_np, _ = env.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32)

    for ep in range(args.n_episodes):
        if ep > 0:
            obs_np, _ = env.reset()
            obs = torch.tensor(obs_np, dtype=torch.float32)

        state = policy.initial_state(args.num_agents, device)

        for step in range(91):
            with torch.no_grad():
                logits, _, state, role_info = policy(obs, state)
            action     = Categorical(logits=logits).sample()
            actions_np = action.numpy().reshape(args.num_agents, 1)
            obs_np, _, term_np, trunc_np, info = env.step(actions_np)
            obs = torch.tensor(obs_np, dtype=torch.float32)

            all_roles.append(role_info["role_z"].numpy())
            all_speeds.append(obs_np[:, 2])

            if isinstance(info, list):
                for item in info:
                    if isinstance(item, dict) and "score" in item:
                        all_scores.append(item["score"])
                        all_collisions.append(item["collision_rate"])
                        all_offroads.append(item["offroad_rate"])
                        all_completions.append(item["completion_rate"])
                        all_returns.append(item["episode_return"])
                        print(f"  Ep {ep+1:>3} | score={item['score']:.4f} | "
                              f"collision={item['collision_rate']:.4f} | "
                              f"offroad={item['offroad_rate']:.4f} | "
                              f"completion={item['completion_rate']:.4f} | "
                              f"return={item['episode_return']:.2f}")
            elif isinstance(info, dict) and "score" in info:
                all_scores.append(float(np.mean(info["score"])))
                all_collisions.append(float(np.mean(info["collision_rate"])))
                all_offroads.append(float(np.mean(info["offroad_rate"])))
                all_completions.append(float(np.mean(info["completion_rate"])))
                all_returns.append(float(np.mean(info["episode_return"])))

    print("\n" + "=" * 57)
    print("  ENVIRONMENT METRICS")
    print("=" * 57)
    print(f"  Checkpoint      : {Path(args.checkpoint).name}")
    print(f"  Episodes logged : {len(all_scores)}")
    if all_scores:
        print(f"  Score           : {np.mean(all_scores):.4f}")
        print(f"  Collision rate  : {np.mean(all_collisions):.4f}")
        print(f"  Off-road rate   : {np.mean(all_offroads):.4f}")
        print(f"  Completion rate : {np.mean(all_completions):.4f}")
        print(f"  Mean return     : {np.mean(all_returns):.4f}")
    print("=" * 57)

    # Role-speed correlations
    if args.role_dim > 1 and all_roles:
        try:
            from scipy.stats import pearsonr
            role_matrix = np.concatenate(all_roles,  axis=0)
            speed_vec   = np.concatenate(all_speeds, axis=0)
            print("\n  Role dimension correlations with speed:")
            print(f"  {'Dim':<8} {'Pearson r':>10}")
            for d in range(args.role_dim):
                r, _ = pearsonr(role_matrix[:, d], speed_vec)
                print(f"  z[{d}]     {r:>10.4f}")
        except ImportError:
            print("  (install scipy for role-speed correlations)")

    # -----------------------------------------------------------------------
    # Part 2 — WOSAC realism metrics (multi-batch)
    # -----------------------------------------------------------------------
    if args.wosac:
        print(f"\nPart 2: WOSAC realism metrics")
        print(f"  rollouts={args.wosac_rollouts} | "
              f"num_maps={args.wosac_num_maps} | "
              f"max_batches={args.wosac_max_batches}")
        print("  Comparing simulated trajectories against real Waymo human drivers.\n")

        try:
            import pandas as pd
            from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

            wosac_config = {
                "eval": {
                    "wosac_init_steps": 0,
                    "wosac_num_rollouts": args.wosac_rollouts,
                },
                "train": {"device": "cpu"},
            }
            evaluator        = WOSACEvaluator(wosac_config)
            adapter          = WOSACPolicyAdapter(policy, args.num_agents, device)
            all_results      = []
            unique_scenarios = set()

            for batch in range(args.wosac_max_batches):
                env.resample_maps()
                env.reset()
                gt          = env.get_ground_truth_trajectories()
                agent_state = env.get_global_agent_state()
                road_edges  = env.get_road_edge_polylines()

                sim = collect_wosac_trajectories(
                    env, adapter,
                    num_rollouts = args.wosac_rollouts,
                    num_steps    = 91,
                    silent       = True,
                )

                try:
                    df  = evaluator.compute_metrics(
                        gt, sim, agent_state, road_edges,
                        aggregate_results=False,
                    )
                    new = set(df.index.tolist()) - unique_scenarios
                    if new:
                        all_results.append(df[df.index.isin(new)])
                        unique_scenarios.update(new)
                except Exception:
                    pass

                if (batch + 1) % 10 == 0:
                    score = pd.concat(all_results)["realism_meta_score"].mean() \
                            if all_results else 0.0
                    print(f"  Batch {batch+1}/{args.wosac_max_batches} | "
                          f"unique scenarios: {len(unique_scenarios)} | "
                          f"realism: {score:.4f}")

            if not all_results:
                print("  No WOSAC results collected.")
            else:
                combined = pd.concat(all_results)
                agg      = combined.mean()

                print("\n" + "=" * 57)
                print("  WOSAC REALISM METRICS")
                print("=" * 57)
                print(f"  Scenarios evaluated       : {len(combined)}")
                print(f"  Rollouts per scenario     : {args.wosac_rollouts}")
                print(f"  Realism meta-score        : {agg['realism_meta_score']:.4f}")
                print(f"  Kinematic metrics         : {agg['kinematic_metrics']:.4f}")
                print(f"  Interactive metrics       : {agg['interactive_metrics']:.4f}")
                print(f"  Map-based metrics         : {agg['map_based_metrics']:.4f}")
                print()
                print(f"  ADE                       : {agg['ade']:.4f} m")
                print(f"  minADE                    : {agg['min_ade']:.4f} m")
                print()
                print(f"  likelihood_linear_speed   : {agg['likelihood_linear_speed']:.4f}")
                print(f"  likelihood_linear_accel   : {agg['likelihood_linear_acceleration']:.4f}")
                print(f"  likelihood_angular_speed  : {agg['likelihood_angular_speed']:.4f}")
                print(f"  likelihood_angular_accel  : {agg['likelihood_angular_acceleration']:.4f}")
                print(f"  likelihood_collision      : {agg['likelihood_collision_indication']:.4f}")
                print(f"  likelihood_dist_obj       : {agg['likelihood_distance_to_nearest_object']:.4f}")
                print(f"  likelihood_ttc            : {agg['likelihood_time_to_collision']:.4f}")
                print(f"  likelihood_dist_road_edge : {agg['likelihood_distance_to_road_edge']:.4f}")
                print(f"  likelihood_offroad        : {agg['likelihood_offroad_indication']:.4f}")
                print("=" * 57)

                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                ckpt_name = Path(args.checkpoint).stem
                csv_path  = Path(args.output_dir) / f"wosac_{ckpt_name}.csv"
                combined.to_csv(csv_path)
                print(f"\n  Full results saved -> {csv_path}")

        except Exception as e:
            print(f"\n  WOSAC evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Optional t-SNE plots
    # -----------------------------------------------------------------------
    if args.save_plots and args.role_dim > 1 and all_roles:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE

            role_matrix = np.concatenate(all_roles,  axis=0)
            speed_vec   = np.concatenate(all_speeds, axis=0)
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
            print("  Install matplotlib and scikit-learn for t-SNE plots.")


if __name__ == "__main__":
    evaluate(parse_args())