"""
eval_roma.py — WOSAC realism evaluation for ROMA checkpoints.

Runs two things in order:
  1. Role diversity analysis — PCA, correlation heatmap, and temporal dynamics.
  2. WOSAC realism metrics — realism meta-score + all 9 sub-metrics.

Usage (from ~/PufferDrive):
    PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 eval_roma.py \
        --checkpoint roma_main/checkpoints/roma_dim8/roma_dim8_final.pt \
        --wosac_rollouts 32 --wosac_num_maps 10000 --wosac_max_batches 500 \
        --wandb
"""

import argparse
import ast
import configparser
import os
import numpy as np
import torch
from pathlib import Path
from torch.distributions import Categorical


def load_drive_config():
    """Read pufferlib's drive.ini and return a nested dict of parsed values."""
    import pufferlib
    puffer_dir = os.path.dirname(pufferlib.__file__)
    default_ini = os.path.join(puffer_dir, "config", "default.ini")
    drive_ini   = os.path.join(puffer_dir, "config", "ocean", "drive.ini")
    p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    p.read([default_ini, drive_ini])

    def _parse(v):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v

    return {section: {k: _parse(v) for k, v in p[section].items()}
            for section in p.sections()}



class WOSACPolicyAdapter:
    def __init__(self, policy, num_agents, device):
        self.policy     = policy
        self.num_agents = num_agents
        self.device     = device
        self._state     = None

    def reset_state(self):
        self._state = self.policy.initial_state(self.num_agents, self.device)

    def forward_eval(self, obs):
        with torch.no_grad():
            logits, _, new_state, _ = self.policy(obs, self._state)
        self._state = new_state
        return logits


def load_policy(checkpoint_path, role_dim, obs_dim, device):
    from roma_pufferdrive.roma.policy import RomaPolicy
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    key    = "policy_state" if "policy_state" in ckpt else "policy"
    policy = RomaPolicy(obs_dim=obs_dim, role_dim=role_dim)
    policy.load_state_dict(ckpt[key])
    policy.to(device)
    policy.eval()
    return policy


def collect_wosac_trajectories(env, adapter, num_rollouts, num_steps=91):
    num_agents = env.num_agents
    traj = {
        "x":       np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32),
        "y":       np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32),
        "z":       np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32),
        "heading": np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32),
        "id":      np.zeros((num_agents, num_rollouts, num_steps), dtype=np.int32),
    }
    for r in range(num_rollouts):
        print(f"\r  rollout {r+1}/{num_rollouts}", end="", flush=True)
        obs_np, _ = env.reset()
        adapter.reset_state()
        obs = torch.as_tensor(obs_np, dtype=torch.float32).to(adapter.device)
        for t in range(num_steps):
            ag = env.get_global_agent_state()
            traj["x"]      [:, r, t] = ag["x"]
            traj["y"]      [:, r, t] = ag["y"]
            traj["z"]      [:, r, t] = ag.get("z", np.zeros(num_agents))
            traj["heading"][:, r, t] = ag["heading"]
            traj["id"]     [:, r, t] = ag["id"]
            action = Categorical(logits=adapter.forward_eval(obs)).sample()
            obs_np, _, _, _, _ = env.step(action.cpu().numpy().reshape(num_agents, 1))
            obs = torch.as_tensor(obs_np, dtype=torch.float32).to(adapter.device)
    print()
    return traj


def wosac_metric_dict(agg, scenarios, prefix="wosac/"):
    return {
        f"{prefix}realism_meta_score":              agg["realism_meta_score"],
        f"{prefix}kinematic_metrics":               agg["kinematic_metrics"],
        f"{prefix}interactive_metrics":             agg["interactive_metrics"],
        f"{prefix}map_based_metrics":               agg["map_based_metrics"],
        f"{prefix}min_ade":                         agg["min_ade"],
        f"{prefix}likelihood_linear_speed":         agg["likelihood_linear_speed"],
        f"{prefix}likelihood_linear_acceleration":  agg["likelihood_linear_acceleration"],
        f"{prefix}likelihood_angular_speed":        agg["likelihood_angular_speed"],
        f"{prefix}likelihood_angular_acceleration": agg["likelihood_angular_acceleration"],
        f"{prefix}likelihood_collision":            agg["likelihood_collision_indication"],
        f"{prefix}likelihood_dist_obj":             agg["likelihood_distance_to_nearest_object"],
        f"{prefix}likelihood_ttc":                  agg["likelihood_time_to_collision"],
        f"{prefix}likelihood_dist_road_edge":       agg["likelihood_distance_to_road_edge"],
        f"{prefix}likelihood_offroad":              agg["likelihood_offroad_indication"],
        f"{prefix}scenarios_evaluated":             scenarios,
    }


def print_progress(agg, batch, max_batches, scenarios):
    print(f"  Batch {batch}/{max_batches} | scenarios: {scenarios} | "
          f"realism: {agg['realism_meta_score']:.4f}")
    print(f"      kinematic {agg['kinematic_metrics']:.4f} | "
          f"interactive {agg['interactive_metrics']:.4f} | "
          f"map_based {agg['map_based_metrics']:.4f}")
    print(f"      lin_spd {agg['likelihood_linear_speed']:.3f} | "
          f"lin_acc {agg['likelihood_linear_acceleration']:.3f} | "
          f"ang_spd {agg['likelihood_angular_speed']:.3f} | "
          f"ang_acc {agg['likelihood_angular_acceleration']:.3f}")
    print(f"      collision {agg['likelihood_collision_indication']:.3f} | "
          f"dist_obj {agg['likelihood_distance_to_nearest_object']:.3f} | "
          f"ttc {agg['likelihood_time_to_collision']:.3f} | "
          f"dist_edge {agg['likelihood_distance_to_road_edge']:.3f} | "
          f"offroad {agg['likelihood_offroad_indication']:.3f}")


def run_role_analysis(policy, env, num_episodes, role_dim, device, out_dir, ckpt_name,
                      wandb_run=None):
    """
    Collect role vectors and behavioral stats over N episodes, then save:
      role_pca_<ckpt>.png         — PCA scatter colored by speed / steering / accel
      role_correlation_<ckpt>.png — Pearson-r heatmap: role dims × behavioral stats
      role_temporal_<ckpt>.png    — mean role dim values across the 91 timesteps

    Temporal metrics (dynamic role design):
      role/temporal_std        — how much each agent's role shifts within an episode
      role/mean_step_l2_dist    — mean L2 distance of role vector between consecutive steps
    """
    print("\n" + "=" * 57)
    print("  ROLE ANALYSIS")
    print("=" * 57)

    try:
        from sklearn.decomposition import PCA
        from scipy.stats import pearsonr
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  Skipped (missing library: {e})")
        print("  Install with: pip install scikit-learn scipy matplotlib")
        return

    B = env.num_agents
    all_role_means   = []
    all_mean_speed   = []
    all_mean_ang_sp  = []
    all_accel_std    = []
    all_temporal_std = []   # (B,) per episode — role variance over 91 steps per agent
    all_step_dists   = []   # (B,) per episode — mean L2 dist of role vector per step
    all_role_seqs    = []   # (91, role_dim) per episode — for temporal plot

    policy.eval()

    for ep in range(num_episodes):
        obs_np, _ = env.reset()
        obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        state = policy.initial_state(B, device)

        xs       = np.zeros((91, B), dtype=np.float32)
        ys       = np.zeros((91, B), dtype=np.float32)
        headings = np.zeros((91, B), dtype=np.float32)
        role_seq = np.zeros((91, B, role_dim), dtype=np.float32)

        for t in range(91):
            ag = env.get_global_agent_state()
            xs[t]       = ag["x"]
            ys[t]       = ag["y"]
            headings[t] = ag["heading"]

            with torch.no_grad():
                logits, _, state, role_info = policy(obs, state)
            role_seq[t] = role_info["role_mean"].float().cpu().numpy()

            action = Categorical(logits=logits.float()).sample()
            obs_np, _, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        dx    = np.diff(xs,       axis=0)
        dy    = np.diff(ys,       axis=0)
        dh    = np.diff(headings, axis=0)
        dh    = (dh + np.pi) % (2 * np.pi) - np.pi

        step_speed = np.sqrt(dx**2 + dy**2) * 10   # m/s at 10 Hz
        ang_speed  = np.abs(dh)
        accel      = np.diff(step_speed, axis=0)

        # Time-averaged role (same as before, for PCA / correlation)
        all_role_means .append(role_seq.mean(axis=0))           # (B, role_dim)
        all_mean_speed .append(step_speed.mean(axis=0))
        all_mean_ang_sp.append(ang_speed.mean(axis=0))
        all_accel_std  .append(accel.std(axis=0))

        # Temporal role dynamics
        temporal_std = role_seq.std(axis=0).mean(axis=1)        # (B,) — std over time per agent
        role_diff    = np.diff(role_seq, axis=0)                # (90, B, role_dim)
        step_dist    = np.linalg.norm(role_diff, axis=2)        # (90, B) — L2 dist each step
        mean_step_dist = step_dist.mean(axis=0)                 # (B,) — avg L2 per agent
        all_temporal_std.append(temporal_std)
        all_step_dists  .append(mean_step_dist)
        all_role_seqs   .append(role_seq.mean(axis=1))          # (91, role_dim)

        last_std = role_info["role_mean"].float().std(dim=0).mean().item()
        print(f"  episode {ep+1:>3}/{num_episodes}  "
              f"mean_speed={all_mean_speed[-1].mean():.2f} m/s  "
              f"role_std={last_std:.4f}  "
              f"temporal_std={temporal_std.mean():.4f}  "
              f"step_dist={mean_step_dist.mean():.4f}")

    role_means    = np.concatenate(all_role_means,   axis=0).astype(np.float32)
    mean_speed    = np.concatenate(all_mean_speed,   axis=0)
    mean_ang_sp   = np.concatenate(all_mean_ang_sp,  axis=0)
    accel_std     = np.concatenate(all_accel_std,    axis=0)
    temporal_std   = np.concatenate(all_temporal_std, axis=0)
    mean_step_dist = np.concatenate(all_step_dists,   axis=0)
    role_over_time = np.stack(all_role_seqs, axis=0).mean(axis=0)  # (91, role_dim)

    N = role_means.shape[0]
    print(f"\n  Total data points    : {N:,}  (episodes x agents)")
    print(f"  Temporal role std    : {temporal_std.mean():.4f}  (0=static, >0=dynamic)")
    print(f"  Mean step L2 dist    : {mean_step_dist.mean():.4f}  (role vector movement per step)")

    # PCA
    pca    = PCA(n_components=2)
    role2d = pca.fit_transform(role_means)
    ev     = pca.explained_variance_ratio_
    print(f"  PCA variance: PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  total={ev.sum():.1%}")

    # Pearson correlation
    stat_names = ["mean_speed", "angular_speed", "accel_std"]
    stats_mat  = np.stack([mean_speed, mean_ang_sp, accel_std], axis=1)
    corr = np.zeros((role_dim, 3), dtype=np.float32)
    for i in range(role_dim):
        for j in range(3):
            if role_means[:, i].std() == 0 or stats_mat[:, j].std() == 0:
                corr[i, j] = float("nan")
            else:
                r, _ = pearsonr(role_means[:, i], stats_mat[:, j])
                corr[i, j] = r

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    MAX_PTS = 20_000
    idx = np.random.choice(N, min(N, MAX_PTS), replace=False)

    # PCA scatter — 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (stat, label, cmap) in zip(axes, [
        (mean_speed,  "mean speed (m/s)",          "viridis"),
        (mean_ang_sp, "angular speed (rad/step)",   "plasma"),
        (accel_std,   "accel std (m/s2)",           "inferno"),
    ]):
        sc = ax.scatter(role2d[idx, 0], role2d[idx, 1],
                        c=stat[idx], cmap=cmap, alpha=0.25, s=4, rasterized=True)
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8)
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.set_title(label)
    fig.suptitle(f"ROMA role space (PCA) -- {N:,} agent-episodes  [{ckpt_name}]", fontsize=11)
    plt.tight_layout()
    pca_out = out_path / f"role_pca_{ckpt_name}.png"
    plt.savefig(pca_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {pca_out}")

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(5, 1 + role_dim * 0.55))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(stat_names, rotation=25, ha="right")
    ax.set_yticks(range(role_dim))
    ax.set_yticklabels([f"dim_{i}" for i in range(role_dim)])
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.7)
    for i in range(role_dim):
        for j in range(3):
            val = corr[i, j]
            label = "N/A" if np.isnan(val) else f"{val:.2f}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=9, color="white" if abs(val) > 0.5 else "black")
    ax.set_title("Role dim x behavior correlations")
    plt.tight_layout()
    corr_out = out_path / f"role_correlation_{ckpt_name}.png"
    plt.savefig(corr_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {corr_out}")

    # Temporal role plot — mean role dim values across 91 timesteps
    fig, ax = plt.subplots(figsize=(12, 4))
    timesteps = np.arange(91)
    for d in range(role_dim):
        ax.plot(timesteps, role_over_time[:, d], label=f"dim_{d}", linewidth=1.2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean role value")
    ax.set_title(f"Role dynamics over episode  [{ckpt_name}]  "
                 f"(temporal_std={temporal_std.mean():.4f}, "
                 f"step_dist={mean_step_dist.mean():.4f})")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    temporal_out = out_path / f"role_temporal_{ckpt_name}.png"
    plt.savefig(temporal_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {temporal_out}")

    # Log images and scalars to wandb
    if wandb_run is not None:
        try:
            import wandb as _wandb
            wandb_run.log({
                "role/pca_scatter":          _wandb.Image(str(pca_out)),
                "role/correlation_heatmap":  _wandb.Image(str(corr_out)),
                "role/temporal_dynamics":    _wandb.Image(str(temporal_out)),
                "role/temporal_std":         float(temporal_std.mean()),
                "role/mean_step_l2_dist":    float(mean_step_dist.mean()),
            })
            print("  Logged images and scalars to wandb")
        except Exception as e:
            print(f"  wandb image log failed: {e}")

    print("\n  Correlation summary (|r| > 0.30 = interpretable):")
    found = False
    for i in range(role_dim):
        for j, name in enumerate(stat_names):
            if abs(corr[i, j]) > 0.30:
                print(f"    dim_{i:>2} <-> {name:<18}  r = {corr[i, j]:+.3f}")
                found = True
    if not found:
        print("    (no dim exceeded |r| = 0.30 -- roles may encode non-linear"
              " structure; try more episodes)")
    print("=" * 57)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",        type=str,   required=True)
    p.add_argument("--role_dim",          type=int,   default=8)
    p.add_argument("--obs_dim",           type=int,   default=1121)
    p.add_argument("--num_agents",        type=int,   default=3072)
    # goal/env settings come from drive.ini via load_drive_config()
    p.add_argument("--device",            type=str,   default="cuda")
    p.add_argument("--map_dir",           type=str,   default="resources/drive/binaries/training")
    p.add_argument("--wosac_rollouts",    type=int,   default=32)
    p.add_argument("--wosac_num_maps",    type=int,   default=10000)
    p.add_argument("--wosac_max_batches", type=int,   default=500)
    p.add_argument("--output_dir",        type=str,   default="eval_results")
    p.add_argument("--wandb",             action="store_true")
    p.add_argument("--wandb_project",     type=str,   default="roma-pufferdrive")
    p.add_argument("--wandb_run_name",    type=str,   default=None)
    p.add_argument("--role_episodes",     type=int,   default=10,
                   help="Episodes to collect for role PCA + correlation analysis. 0 = skip.")
    return p.parse_args()


def evaluate(args):
    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or Path(args.checkpoint).stem
        wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        wandb_run.define_metric("wosac/batch")
        wandb_run.define_metric("wosac/*", step_metric="wosac/batch")

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"Checkpoint : {args.checkpoint}")
    policy = load_policy(args.checkpoint, args.role_dim, args.obs_dim, device)

    from pufferlib.ocean.drive.drive import Drive
    ini = load_drive_config()
    wosac_cfg = dict(ini["env"])
    wosac_cfg.update({
        "num_maps":      args.wosac_num_maps,
        "num_agents":    args.num_agents,
        "map_dir":       args.map_dir,
        "control_mode":  ini["eval"]["wosac_control_mode"],
        "goal_behavior": ini["eval"]["wosac_goal_behavior"],
        "goal_radius":   ini["eval"]["wosac_goal_radius"],
    })
    env = Drive(**wosac_cfg)

    # --- Part 1: Role diversity analysis ---
    if args.role_episodes > 0:
        try:
            run_role_analysis(
                policy, env, args.role_episodes, args.role_dim,
                device, args.output_dir, Path(args.checkpoint).stem,
                wandb_run=wandb_run,
            )
        except Exception as e:
            print(f"[role analysis skipped: {e}]")
            import traceback
            traceback.print_exc()

    # --- Part 2: WOSAC realism metrics ---
    import pandas as pd
    from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

    wosac_config = {
        "eval": {"wosac_init_steps": 0, "wosac_num_rollouts": args.wosac_rollouts},
        "train": {"device": str(device)},
    }
    evaluator        = WOSACEvaluator(wosac_config)
    adapter          = WOSACPolicyAdapter(policy, env.num_agents, device)
    all_results      = []
    unique_scenarios = set()

    print(f"rollouts={args.wosac_rollouts} | num_maps={args.wosac_num_maps} | "
          f"max_batches={args.wosac_max_batches} | num_agents={env.num_agents}\n")

    for batch in range(args.wosac_max_batches):
        if batch > 0:
            env.resample_maps()
        env.reset()
        gt          = env.get_ground_truth_trajectories()
        agent_state = env.get_global_agent_state()
        road_edges  = env.get_road_edge_polylines()

        sim = collect_wosac_trajectories(env, adapter, args.wosac_rollouts)

        try:
            df  = evaluator.compute_metrics(gt, sim, agent_state, road_edges,
                                            aggregate_results=False)
            new = set(df.index.tolist()) - unique_scenarios
            if new:
                all_results.append(df[df.index.isin(new)])
                unique_scenarios.update(new)
        except Exception:
            pass

        if (batch + 1) % 10 == 0:
            if all_results:
                agg = pd.concat(all_results).mean()
                print_progress(agg, batch + 1, args.wosac_max_batches,
                               len(unique_scenarios))
                if wandb_run:
                    d = wosac_metric_dict(agg, len(unique_scenarios))
                    d["wosac/batch"] = batch + 1
                    wandb_run.log(d)

    if not all_results:
        print("No WOSAC results collected.")
        return

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

    if wandb_run:
        final = wosac_metric_dict(agg, len(combined))
        final["wosac/batch"] = args.wosac_max_batches
        wandb_run.log(final)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.output_dir) / f"wosac_{Path(args.checkpoint).stem}.csv"
    combined.to_csv(csv_path)
    print(f"\n  Results saved -> {csv_path}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    evaluate(parse_args())
