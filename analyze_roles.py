"""
analyze_roles.py
================
Level-2 ROMA role analysis.

Loads a checkpoint, runs N episodes, collects (role_mean, behavioral stats)
per agent per episode, then produces two plots:

  1. role_pca.png       — PCA scatter of role space colored by speed / steering / accel
  2. role_correlation.png — Pearson-r heatmap: role dims × behavioral stats

Behavioral stats (computed from env.get_global_agent_state positions):
  mean_speed     — average displacement per step × 10  (m/s at 10 Hz)
  mean_ang_speed — average |Δheading| per step         (rad/step, turning proxy)
  accel_std      — std of per-step speed changes       (jerk/aggression proxy)

Usage:
    python analyze_roles.py \\
        --checkpoint roma_pufferdrive/checkpoints/roma/roma_dim8_step1000000000.pt \\
        --data_dir resources/drive/binaries/training \\
        --n_episodes 20 \\
        --device cuda \\
        --out_dir role_analysis
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint saved by train_roma.py")
    p.add_argument("--data_dir", default="resources/drive/binaries/training",
                   help="Map binary directory (override saved path if running locally)")
    p.add_argument("--n_episodes", type=int, default=20,
                   help="Episodes to run. More → more points, slower. 20 is a good start.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_maps", type=int, default=1000,
                   help="Map pool for the analysis env (smaller than full training pool is fine)")
    p.add_argument("--out_dir", default="role_analysis",
                   help="Directory for output PNG files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[analyze_roles] device  : {device}")
    print(f"[analyze_roles] ckpt    : {args.checkpoint}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved = argparse.Namespace(**ckpt["args"])
    print(f"[analyze_roles] role_dim: {saved.role_dim}")
    print(f"[analyze_roles] episodes: {args.n_episodes}")

    # Build env (smaller map pool is fine for analysis)
    # Use getattr fallbacks so older checkpoints without these args still work.
    from pufferlib.ocean.drive.drive import Drive
    env = Drive(
        num_maps                 = args.num_maps,
        num_agents               = getattr(saved, "num_agents",               3072),
        map_dir                  = args.data_dir,
        episode_length           = 91,
        reward_vehicle_collision = getattr(saved, "reward_vehicle_collision", -0.5),
        reward_offroad_collision = getattr(saved, "reward_offroad_collision", -0.5),
        goal_speed               = getattr(saved, "goal_speed",               100.0),
        reward_goal_post_respawn = getattr(saved, "reward_goal_post_respawn", 0.25),
        goal_target_distance     = getattr(saved, "goal_target_distance",     30.0),
        resample_frequency       = getattr(saved, "resample_frequency",       910),
        termination_mode         = getattr(saved, "termination_mode",         1),
    )
    obs_probe, _ = env.reset()
    obs_dim = obs_probe.shape[-1]
    B = getattr(saved, "num_agents", 3072)

    # Build and load policy
    from roma_pufferdrive.roma.policy import RomaPolicy
    policy = RomaPolicy(
        obs_dim       = obs_dim,
        action_dim    = 91,
        role_dim      = saved.role_dim,
        role_hidden   = getattr(saved, "role_hidden",   64),
        policy_hidden = getattr(saved, "policy_hidden", 128),
        var_floor     = getattr(saved, "var_floor",     1e-4),
        obs_window_len= 8,
    ).to(device)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()
    print(f"[analyze_roles] policy loaded  ({sum(p.numel() for p in policy.parameters()):,} params)")

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    # Per episode we accumulate (B,) arrays of behavioral stats and
    # (B, role_dim) mean role vector. Memory: O(B × role_dim) per episode.

    all_role_means  = []   # each (B, role_dim)
    all_mean_speed  = []   # each (B,)
    all_mean_ang_sp = []   # each (B,)
    all_accel_std   = []   # each (B,)

    for ep in range(args.n_episodes):
        obs_np, _ = env.reset()
        obs   = torch.tensor(obs_np, dtype=torch.float32, device=device)
        state = policy.initial_state(B, device)

        # Per-step buffers (91 steps)
        xs       = np.zeros((91, B), dtype=np.float32)
        ys       = np.zeros((91, B), dtype=np.float32)
        headings = np.zeros((91, B), dtype=np.float32)
        role_acc = np.zeros((B, saved.role_dim), dtype=np.float64)

        for t in range(91):
            ag = env.get_global_agent_state()
            xs[t]       = ag["x"]
            ys[t]       = ag["y"]
            headings[t] = ag["heading"]

            with torch.no_grad():
                logits, _, state, role_info = policy(obs, state)
            role_acc += role_info["role_mean"].float().cpu().numpy()

            action = Categorical(logits=logits.float()).sample()
            obs_np, _, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
            obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

        # Behavioral stats from trajectory
        dx     = np.diff(xs,       axis=0)   # (90, B)
        dy     = np.diff(ys,       axis=0)   # (90, B)
        dh     = np.diff(headings, axis=0)   # (90, B)

        # Wrap heading diff to [-pi, pi]
        dh = (dh + np.pi) % (2 * np.pi) - np.pi

        step_speed = np.sqrt(dx**2 + dy**2) * 10   # m/s at 10 Hz  (90, B)
        ang_speed  = np.abs(dh)                     # rad/step      (90, B)
        accel      = np.diff(step_speed, axis=0)    # Δspeed        (89, B)

        all_role_means.append(role_acc / 91.0)               # episode-mean role
        all_mean_speed .append(step_speed.mean(axis=0))
        all_mean_ang_sp.append(ang_speed.mean(axis=0))
        all_accel_std  .append(accel.std(axis=0))

        print(f"  episode {ep+1:>3}/{args.n_episodes}  "
              f"mean_speed={all_mean_speed[-1].mean():.2f} m/s  "
              f"role_std={role_info['role_mean'].float().std(dim=0).mean().item():.4f}")

    try:
        env.close()
    except Exception:
        pass

    # Concatenate: (n_ep × B, ...)
    role_means  = np.concatenate(all_role_means,  axis=0).astype(np.float32)
    mean_speed  = np.concatenate(all_mean_speed,  axis=0)
    mean_ang_sp = np.concatenate(all_mean_ang_sp, axis=0)
    accel_std   = np.concatenate(all_accel_std,   axis=0)

    N_total = role_means.shape[0]
    print(f"\n[analyze_roles] Total data points : {N_total:,}  (episodes × agents)")

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("scikit-learn not found — install with: pip install scikit-learn")
        sys.exit(1)

    pca    = PCA(n_components=2)
    role2d = pca.fit_transform(role_means)
    ev     = pca.explained_variance_ratio_
    print(f"[analyze_roles] PCA explained variance: PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  "
          f"total={ev.sum():.1%}")

    # ------------------------------------------------------------------
    # Pearson correlation
    # ------------------------------------------------------------------
    from scipy.stats import pearsonr

    role_dim   = role_means.shape[1]
    stat_names = ["mean_speed", "angular_speed", "accel_std"]
    stats_mat  = np.stack([mean_speed, mean_ang_sp, accel_std], axis=1)

    corr = np.zeros((role_dim, 3), dtype=np.float32)
    for i in range(role_dim):
        for j in range(3):
            r, _ = pearsonr(role_means[:, i], stats_mat[:, j])
            corr[i, j] = r

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found — install with: pip install matplotlib")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Downsample scatter for readability (cap at 20k points)
    MAX_SCATTER = 20_000
    if N_total > MAX_SCATTER:
        idx = np.random.choice(N_total, MAX_SCATTER, replace=False)
    else:
        idx = np.arange(N_total)

    # ---- PCA scatter (3 subplots) ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    scatter_data = [
        (mean_speed,  "mean speed (m/s)",    "viridis"),
        (mean_ang_sp, "angular speed (rad/step)", "plasma"),
        (accel_std,   "accel std (m/s²)",    "inferno"),
    ]
    for ax, (stat, label, cmap) in zip(axes, scatter_data):
        sc = ax.scatter(
            role2d[idx, 0], role2d[idx, 1],
            c=stat[idx], cmap=cmap, alpha=0.25, s=4, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8)
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.set_title(label)

    fig.suptitle(
        f"ROMA role space (PCA)  —  {N_total:,} agent-episodes  "
        f"[{Path(args.checkpoint).name}]",
        fontsize=11,
    )
    plt.tight_layout()
    pca_out = out_dir / "role_pca.png"
    plt.savefig(pca_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analyze_roles] Saved: {pca_out}")

    # ---- Correlation heatmap ----
    fig, ax = plt.subplots(figsize=(5, 1 + role_dim * 0.55))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(stat_names, rotation=25, ha="right")
    ax.set_yticks(range(role_dim))
    ax.set_yticklabels([f"dim_{i}" for i in range(role_dim)])
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.7)
    for i in range(role_dim):
        for j in range(3):
            ax.text(j, i, f"{corr[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(corr[i, j]) > 0.5 else "black")
    ax.set_title("Role dim × behavior correlations")
    plt.tight_layout()
    corr_out = out_dir / "role_correlation.png"
    plt.savefig(corr_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analyze_roles] Saved: {corr_out}")

    # ---- Text summary ----
    print("\nCorrelation summary (|r| > 0.30 = interpretable):")
    found = False
    for i in range(role_dim):
        for j, name in enumerate(stat_names):
            if abs(corr[i, j]) > 0.30:
                print(f"  dim_{i:>2} <-> {name:<18}  r = {corr[i, j]:+.3f}")
                found = True
    if not found:
        print("  (no dimension exceeded |r| = 0.30 — roles may not correlate with"
              " speed/steering in a linear way, or need more episodes)")


if __name__ == "__main__":
    main()
