"""
eval_roma.py — WOSAC realism evaluation + deep role analysis for ROMA checkpoints.

Runs two things in order:
  1. Role analysis  — structural + causal analysis of the role variable.
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
        "dones":   np.zeros((num_agents, num_rollouts, num_steps), dtype=np.bool_),
    }
    for r in range(num_rollouts):
        print(f"\r  rollout {r+1}/{num_rollouts}", end="", flush=True)
        obs_np, _ = env.reset()
        adapter.reset_state()
        obs = torch.as_tensor(obs_np, dtype=torch.float32).to(adapter.device)
        truncations = np.zeros(num_agents, dtype=bool)
        for t in range(num_steps):
            ag = env.get_global_agent_state()
            traj["x"]      [:, r, t] = ag["x"]
            traj["y"]      [:, r, t] = ag["y"]
            traj["z"]      [:, r, t] = ag.get("z", np.zeros(num_agents))
            traj["heading"][:, r, t] = ag["heading"]
            traj["id"]     [:, r, t] = ag["id"]
            traj["dones"]  [:, r, t] = truncations
            action = Categorical(logits=adapter.forward_eval(obs)).sample()
            obs_np, _, _, truncations, _ = env.step(action.cpu().numpy().reshape(num_agents, 1))
            truncations = np.asarray(truncations).reshape(num_agents)
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


# ---------------------------------------------------------------------------
# Role analysis — data collection helpers
# ---------------------------------------------------------------------------

def _behavioral_stats(xs, ys, headings):
    """Compute per-(episode × agent) behavioral stats from position sequences.

    xs, ys, headings: (N_ep, T, B) float32 arrays.
    Returns dict of 1-D arrays of length N_ep * B.
    """
    dx = np.diff(xs, axis=1)                         # (N_ep, T-1, B)
    dy = np.diff(ys, axis=1)
    dh = np.diff(headings, axis=1)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi         # wrap to [-π, π]

    speed = np.sqrt(dx**2 + dy**2) * 10              # m/s at 10 Hz
    accel = np.diff(speed, axis=1)                   # (N_ep, T-2, B)

    return {
        "mean_speed":  speed.mean(axis=1).reshape(-1),
        "max_speed":   speed.max(axis=1).reshape(-1),
        "min_speed":   speed.min(axis=1).reshape(-1),
        "mean_accel":  np.abs(accel).mean(axis=1).reshape(-1),
        "jerk":        accel.std(axis=1).reshape(-1),  # std of acceleration
        "steering":    np.abs(dh).mean(axis=1).reshape(-1),
        # aliases used by legacy PCA / correlation code
        "mean_ang_sp": np.abs(dh).mean(axis=1).reshape(-1),
        "accel_std":   accel.std(axis=1).reshape(-1),
    }


def _collect_role_data(policy, env, num_episodes, device, obs_agent_sample=256):
    """Run num_episodes and collect role vectors, positions, and obs snapshots.

    Returns:
        role_means  (N_ep*B, role_dim)      time-averaged role per agent-episode
        role_seqs   (N_ep, T, B, role_dim)  full per-step sequences
        xs/ys/headings (N_ep, T, B)         agent positions
        obs_snaps   list of (agent_sample, obs_dim) tensors at steps 0, 45, 90
    """
    B        = env.num_agents
    T        = 91
    role_dim = policy.role_dim

    role_seqs = np.zeros((num_episodes, T, B, role_dim), dtype=np.float32)
    xs        = np.zeros((num_episodes, T, B), dtype=np.float32)
    ys        = np.zeros((num_episodes, T, B), dtype=np.float32)
    headings  = np.zeros((num_episodes, T, B), dtype=np.float32)
    obs_snaps = []

    agent_idx = np.random.choice(B, min(obs_agent_sample, B), replace=False)

    policy.eval()
    for ep in range(num_episodes):
        obs_np, _ = env.reset()
        obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        state = policy.initial_state(B, device)

        for t in range(T):
            ag = env.get_global_agent_state()
            xs[ep, t]       = ag["x"]
            ys[ep, t]       = ag["y"]
            headings[ep, t] = ag["heading"]

            with torch.no_grad():
                logits, _, state, role_info = policy(obs, state)
            role_seqs[ep, t] = role_info["role_mean"].float().cpu().numpy()

            if t in (0, 45, 90):
                obs_snaps.append(obs[agent_idx].cpu())

            action = Categorical(logits=logits.float()).sample()
            obs_np, _, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        print(f"  data collection: episode {ep+1}/{num_episodes}", end="\r", flush=True)

    print()
    return {
        "role_means": role_seqs.mean(axis=1).reshape(-1, role_dim),
        "role_seqs":  role_seqs,
        "xs": xs, "ys": ys, "headings": headings,
        "obs_snaps":  obs_snaps,
    }


# ---------------------------------------------------------------------------
# Idea 5 — dead dimension detection
# ---------------------------------------------------------------------------

def _plot_dead_dims(role_means, role_dim, ckpt_name):
    """Bar chart of per-dim std. Dead dims (red) never differentiate agents."""
    import matplotlib.pyplot as plt

    per_dim_std = role_means.std(axis=0)
    threshold   = 0.05

    fig, ax = plt.subplots(figsize=(max(6, role_dim * 0.9), 4))
    colors  = ["#d73027" if s < threshold else "#1a9850" for s in per_dim_std]
    bars    = ax.bar(range(role_dim), per_dim_std, color=colors,
                     edgecolor="black", linewidth=0.5)
    ax.axhline(threshold, color="orange", linestyle="--", linewidth=1.2,
               label=f"dead threshold ({threshold})")
    ax.set_xticks(range(role_dim))
    ax.set_xticklabels([f"dim_{i}" for i in range(role_dim)])
    ax.set_ylabel("Std across agents & episodes")
    ax.set_title(f"Role dim utilisation  (red = dead)  [{ckpt_name}]")
    ax.legend()

    for bar, s in zip(bars, per_dim_std):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{s:.3f}", ha="center", va="bottom", fontsize=8)

    n_dead = int((per_dim_std < threshold).sum())
    print(f"  Dead dims (std<{threshold}): {n_dead}/{role_dim}  "
          f"{'— collapse risk!' if n_dead > role_dim // 2 else '— healthy'}")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Idea 6 — inter-dimension correlation matrix
# ---------------------------------------------------------------------------

def _plot_dim_correlations(role_means, role_dim, ckpt_name):
    """Pearson r heatmap between every pair of role dims.

    Off-diagonal near 0 = independent dims.  High off-diagonal = redundancy.
    """
    import matplotlib.pyplot as plt

    corr = np.corrcoef(role_means.T)   # (role_dim, role_dim)

    mask = np.abs(corr) > 0.6
    np.fill_diagonal(mask, False)
    n_corr = int(mask.sum()) // 2
    print(f"  Strongly correlated dim pairs (|r|>0.6): {n_corr}  "
          f"{'— redundancy' if n_corr > 0 else '— independent'}")

    fig, ax = plt.subplots(figsize=(role_dim + 1, role_dim))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(role_dim))
    ax.set_yticks(range(role_dim))
    ax.set_xticklabels([f"d{i}" for i in range(role_dim)], fontsize=8)
    ax.set_yticklabels([f"d{i}" for i in range(role_dim)], fontsize=8)
    for i in range(role_dim):
        for j in range(role_dim):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.5 else "black")
    ax.set_title(f"Role dim inter-correlation  [{ckpt_name}]\n"
                 f"off-diagonal ≈ 0 = independent dims")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PCA behavioral scatter — min speed, max speed, jerk
# ---------------------------------------------------------------------------

def _plot_pca_behavioral(role_means, stats, ckpt_name):
    """PCA scatter: 3 subplots colored by min_speed, max_speed, jerk.

    If PC1 drives the color gradient in one subplot, that principal component
    encodes that behavioral dimension.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca    = PCA(n_components=2)
    role2d = pca.fit_transform(role_means)
    ev     = pca.explained_variance_ratio_

    N       = len(role2d)
    MAX_PTS = 10_000
    idx     = np.random.choice(N, min(N, MAX_PTS), replace=False)

    metrics = [
        (stats["min_speed"], "Min Speed (m/s)",   "Blues"),
        (stats["max_speed"], "Max Speed (m/s)",   "Reds"),
        (stats["jerk"],      "Jerk (accel std)",  "Purples"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric, label, cmap) in zip(axes, metrics):
        sc = ax.scatter(role2d[idx, 0], role2d[idx, 1],
                        c=metric[idx], cmap=cmap, alpha=0.3, s=5, rasterized=True)
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8)
        ax.set_xlabel(f"PC1  ({ev[0]:.1%})")
        ax.set_ylabel(f"PC2  ({ev[1]:.1%})")
        ax.set_title(label)

    fig.suptitle(
        f"Role PCA — behavioral coloring  [{ckpt_name}]"
        f"   PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  total={ev[:2].sum():.1%}",
        fontsize=11,
    )
    plt.tight_layout()
    return fig, ev


# ---------------------------------------------------------------------------
# Idea 8 — role cluster spider / radar chart
# ---------------------------------------------------------------------------

def _plot_cluster_spider(role_means, stats, labels, k, ckpt_name):
    """Radar chart: one polygon per role cluster, axes = behavioral stats."""
    import matplotlib.pyplot as plt

    metric_keys  = ["mean_speed", "max_speed", "min_speed",
                    "mean_accel", "jerk", "steering"]
    metric_names = ["Mean Speed", "Max Speed", "Min Speed",
                    "Mean Accel", "Jerk", "Steering"]
    n_metrics    = len(metric_keys)

    cluster_stats = np.zeros((k, n_metrics))
    for c in range(k):
        mask = labels == c
        if not mask.any():
            continue
        for j, key in enumerate(metric_keys):
            cluster_stats[c, j] = stats[key][mask].mean()

    print(f"\n  Role cluster behavioral profiles (k={k}):")
    header = f"  {'':14}" + "".join(f"{m:>13}" for m in metric_names)
    print(header)
    for c in range(k):
        n_ag = int((labels == c).sum())
        row  = f"  Cluster {c} ({n_ag:>5})" + \
               "".join(f"{cluster_stats[c, j]:>13.3f}" for j in range(n_metrics))
        print(row)

    col_min = cluster_stats.min(axis=0)
    col_max = cluster_stats.max(axis=0)
    col_rng = col_max - col_min
    col_rng[col_rng == 0] = 1
    normed  = (cluster_stats - col_min) / col_rng

    angles  = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors  = plt.cm.tab10(np.linspace(0, 1, k))

    for c in range(k):
        values = normed[c].tolist() + [normed[c][0]]
        n_ag   = int((labels == c).sum())
        ax.plot(angles, values, "o-", linewidth=2, color=colors[c],
                label=f"Cluster {c}  (n={n_ag})")
        ax.fill(angles, values, alpha=0.08, color=colors[c])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    ax.set_title(
        f"Role clusters — behavioral profiles  [{ckpt_name}]  k={k}",
        pad=20, fontsize=11,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# Idea 3 — action distribution sensitivity across real roles
# ---------------------------------------------------------------------------

def _plot_action_sensitivity(policy, rep_roles, obs_snaps, device, ckpt_name):
    """Pairwise mean KL divergence of action distributions under K representative
    real role vectors, averaged over many real observations.

    High KL → role causally changes what the policy does.
    KL ≈ 0  → policy ignores role (role encoder trains but decoder doesn't listen).
    """
    import matplotlib.pyplot as plt

    K = len(rep_roles)
    if not obs_snaps:
        print("  [Action sensitivity] no obs snapshots — skipped")
        return None

    obs_cat = torch.cat(obs_snaps, dim=0).to(device)   # (N_total, obs_dim)
    N       = obs_cat.shape[0]

    # Zero hidden state — probing snapshot sensitivity, not sequential context.
    role_h_z   = torch.zeros(N, policy.role_encoder.hidden_dim,   device=device)
    policy_h_z = torch.zeros(N, policy.policy_hidden,             device=device)
    emb_win_z  = torch.zeros(N, policy.obs_window_len, policy.env_embed_dim, device=device)
    zero_state = (role_h_z, policy_h_z, emb_win_z)

    probs = []
    with torch.no_grad():
        for role_vec in rep_roles:
            forced = torch.tensor(role_vec, dtype=torch.float32, device=device) \
                         .unsqueeze(0).expand(N, -1)
            logits, _, _, _ = policy(obs_cat, zero_state, forced_role=forced)
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())

    probs = np.array(probs)   # (K, N, action_dim)
    eps   = 1e-8

    kl_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                p = probs[i] + eps
                q = probs[j] + eps
                kl_matrix[i, j] = (p * np.log(p / q)).sum(axis=-1).mean()

    mean_kl = kl_matrix[kl_matrix > 0].mean() if (kl_matrix > 0).any() else 0.0
    print(f"\n  Action sensitivity — mean pairwise KL = {mean_kl:.4f}")
    if mean_kl < 0.001:
        print("  *** WARNING: role has NO detectable effect on action distribution ***")
    elif mean_kl < 0.05:
        print("  role has a weak effect on actions")
    else:
        print("  role has a meaningful effect on actions")

    fig, ax = plt.subplots(figsize=(K + 2, K + 1))
    im = ax.imshow(kl_matrix, cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, label="Mean KL divergence")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    role_labels = [f"Role {i}" for i in range(K)]
    ax.set_xticklabels(role_labels, fontsize=8)
    ax.set_yticklabels(role_labels, fontsize=8)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{kl_matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if kl_matrix.max() > 0 and
                    kl_matrix[i, j] > kl_matrix.max() * 0.6 else "black")
    ax.set_title(
        f"Action KL divergence between role clusters  [{ckpt_name}]\n"
        f"mean pairwise KL = {mean_kl:.4f}   (0=role ignored  >0.05=role matters)",
        fontsize=9,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Idea 2 — role intervention with real role values (Option B)
# ---------------------------------------------------------------------------

def _plot_role_intervention(policy, env, rep_roles, device, ckpt_name,
                            n_rollouts=5, show_agents=4):
    """Re-run n_rollouts episodes per representative real role, injecting that
    role at every step.  Compare speed/steering profiles and 2D trajectories.

    Causal test: if profiles are identical across roles, the policy ignores the
    role vector regardless of what the encoder produces.
    """
    import matplotlib.pyplot as plt

    K           = len(rep_roles)
    B           = env.num_agents
    T           = 91
    colors      = plt.cm.tab10(np.linspace(0, 1, K))
    show_agents = min(show_agents, B)

    speed_per_role    = []
    steering_per_role = []
    trajs_per_role    = []

    print("\n  Role intervention rollouts...")
    for ki, role_vec in enumerate(rep_roles):
        forced = torch.tensor(role_vec, dtype=torch.float32, device=device) \
                     .unsqueeze(0).expand(B, -1)

        ki_speeds, ki_steerings, ki_trajs = [], [], []

        for r in range(n_rollouts):
            obs_np, _ = env.reset()
            obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            state = policy.initial_state(B, device)

            ep_x = np.zeros((T, B), dtype=np.float32)
            ep_y = np.zeros((T, B), dtype=np.float32)
            ep_h = np.zeros((T, B), dtype=np.float32)

            for t in range(T):
                ag = env.get_global_agent_state()
                ep_x[t] = ag["x"]
                ep_y[t] = ag["y"]
                ep_h[t] = ag["heading"]

                with torch.no_grad():
                    logits, _, state, _ = policy(obs, state, forced_role=forced)
                action = Categorical(logits=logits.float()).sample()
                obs_np, _, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
                obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

            dx = np.diff(ep_x, axis=0) * 10
            dy = np.diff(ep_y, axis=0) * 10
            dh = np.diff(ep_h, axis=0)
            dh = (dh + np.pi) % (2 * np.pi) - np.pi

            spd    = np.sqrt(dx**2 + dy**2)                 # (T-1, B)
            ok     = spd <= 45.0                            # mask respawn teleports
            spd_m  = np.where(ok, spd, np.nan)
            moving = np.nanmax(spd_m, axis=0) > 1.0         # agents that actually drive
            if not moving.any():
                moving = np.ones(spd_m.shape[1], dtype=bool)
            ki_speeds.append(np.nanmean(spd_m[:, moving], axis=1))   # (T-1,)
            dh_m = np.where(ok, np.abs(dh), np.nan)
            ki_steerings.append(np.nanmean(dh_m[:, moving], axis=1))
            disp = np.nansum(np.where(ok, spd, 0.0), axis=0)  # distance driven per agent
            top  = np.argsort(disp)[::-1][:show_agents]       # most-active agents
            ki_trajs.append(
                np.stack([ep_x[:, top],
                          ep_y[:, top]], axis=-1)             # (T, show_agents, 2)
            )
            print(f"    role {ki+1}/{K}  rollout {r+1}/{n_rollouts}", end="\r", flush=True)

        speed_per_role.append(ki_speeds)
        steering_per_role.append(ki_steerings)
        trajs_per_role.append(ki_trajs)

    print()
    ts = np.arange(T - 1)

    # ── Plot 1: speed & steering profiles ────────────────────────────────────
    fig_profiles, (ax_spd, ax_str) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ki in range(K):
        speeds = np.array(speed_per_role[ki])
        steers = np.array(steering_per_role[ki])
        m_s, s_s = speeds.mean(axis=0), speeds.std(axis=0)
        m_h, s_h = steers.mean(axis=0), steers.std(axis=0)

        ax_spd.plot(ts, m_s, color=colors[ki], linewidth=2, label=f"Role {ki}")
        ax_spd.fill_between(ts, m_s - s_s, m_s + s_s, color=colors[ki], alpha=0.15)
        ax_str.plot(ts, m_h, color=colors[ki], linewidth=2, label=f"Role {ki}")
        ax_str.fill_between(ts, m_h - s_h, m_h + s_h, color=colors[ki], alpha=0.15)

    ax_spd.set_ylabel("Mean speed  (m/s)")
    ax_spd.set_title(f"Role intervention: speed profiles  [{ckpt_name}]  "
                     f"(±1 std across {n_rollouts} rollouts)")
    ax_spd.legend(fontsize=8)
    ax_spd.grid(True, alpha=0.3)
    ax_str.set_ylabel("Mean |Δheading|  (rad/step)")
    ax_str.set_xlabel("Timestep")
    ax_str.set_title("Role intervention: steering profiles")
    ax_str.legend(fontsize=8)
    ax_str.grid(True, alpha=0.3)
    plt.tight_layout()

    # ── Plot 2: 2D trajectories (first rollout, first show_agents agents) ─────
    fig_traj, axes = plt.subplots(1, K, figsize=(4 * K, 5))
    if K == 1:
        axes = [axes]
    for ki in range(K):
        ax   = axes[ki]
        traj = trajs_per_role[ki][0]   # (T, show_agents, 2)
        for ag in range(show_agents):
            ax.plot(traj[:, ag, 0], traj[:, ag, 1],
                    alpha=0.75, linewidth=1.3, color=colors[ki])
            ax.scatter(traj[0,  ag, 0], traj[0,  ag, 1], s=25, color="green", zorder=3)
            ax.scatter(traj[-1, ag, 0], traj[-1, ag, 1], s=25, color="red",
                       zorder=3, marker="x")
        ax.set_title(f"Role {ki}", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
    fig_traj.suptitle(
        f"Role intervention: 2D trajectories  [{ckpt_name}]  "
        f"(green=start  red=end)",
        fontsize=10,
    )
    plt.tight_layout()
    return fig_profiles, fig_traj


# ---------------------------------------------------------------------------
# Main role analysis orchestrator
# ---------------------------------------------------------------------------

def run_role_analysis(policy, env, num_episodes, role_dim, device,
                      out_dir, ckpt_name, wandb_run=None):
    """
    Full role analysis suite.  Always called BEFORE WOSAC so results are
    guaranteed even when the 24-hour training limit is hit mid-WOSAC.

    Analyses produced
    -----------------
    dead_dims              Idea 5 — which dims are inactive
    dim_correlations       Idea 6 — inter-dim redundancy
    pca_behavioral         PCA scatter coloured by min_speed / max_speed / jerk
    cluster_spider         Idea 8 — k-means clusters + radar behavioural profiles
    action_kl              Idea 3 — does swapping roles change action distributions?
    intervention_profiles  Idea 2 — speed & steering under injected real roles
    intervention_trajs     Idea 2 — 2-D trajectory comparison per role
    pca_scatter            Legacy — mean_speed / ang_speed / accel_std PCA
    temporal               Legacy — mean role dim values over 91 timesteps

    Wandb fix
    ---------
    Every figure is logged via wandb.Image(fig) BEFORE fig.savefig(), passing
    the matplotlib Figure object directly.  This eliminates the "no matching
    media" error that occurs when wandb reads a file path after plt.close().
    """
    print("\n" + "=" * 60)
    print("  ROLE ANALYSIS")
    print("=" * 60)

    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from scipy.stats import pearsonr
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  Skipped (missing library: {e})")
        print("  Install with: pip install scikit-learn scipy matplotlib")
        return

    if num_episodes <= 0:
        print("  Skipped (role_episodes=0)")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    K = 5   # clusters — covers main driving archetypes

    def _save_log(fig, name):
        """Save the figure to disk first, then log the saved PNG *file* to wandb.

        Logging the file path (not the live matplotlib Figure) avoids the
        "No matching media" panels that appear when wandb can't resolve a
        figure object after it's been closed.
        """
        path = out_path / f"role_{name}_{ckpt_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if wandb_run is not None:
            try:
                import wandb as _wandb
                wandb_run.log({f"role/{name}": _wandb.Image(str(path))})
            except Exception as e:
                print(f"  [wandb/{name}] {e}")
        print(f"  Saved: {path}")

    # ── 1. Collect data ───────────────────────────────────────────────────────
    print(f"\n  Collecting {num_episodes} episodes...")
    data       = _collect_role_data(policy, env, num_episodes, device)
    role_means = data["role_means"]   # (N, role_dim)
    N          = len(role_means)
    print(f"  {N:,} agent-episode data points")

    # ── 2. Behavioral stats ───────────────────────────────────────────────────
    stats = _behavioral_stats(data["xs"], data["ys"], data["headings"])

    # -- 2b. Keep only real driving agents --
    # A Waymo scene is mostly parked cars / static context / padding slots,
    # and respawn teleports create impossible speeds. Analyse only agents
    # that actually drive so role clusters are behavioural archetypes.
    SPEED_CAP  = 45.0   # m/s -- above any real vehicle: teleport/respawn artifact
    MIN_MOTION = 1.0    # m/s -- below this the agent never really moved
    valid  = (np.isfinite(stats["max_speed"])
              & (stats["max_speed"] <= SPEED_CAP)
              & (stats["max_speed"] >  MIN_MOTION))
    n_drop = int((~valid).sum())
    if valid.sum() >= K:
        role_means = role_means[valid]
        stats      = {k: v[valid] for k, v in stats.items()}
        N          = len(role_means)
        print(f"  Dropped {n_drop:,} parked/padding/artifact agents "
              f"-> {N:,} moving agents analysed")
    else:
        print(f"  [warn] only {int(valid.sum())} moving agents; skipping filter")

    # ── 3. k-means: K representative real roles ───────────────────────────────
    print(f"\n  k-means (k={K})...")
    km        = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels    = km.fit_predict(role_means)
    rep_roles = km.cluster_centers_   # (K, role_dim)

    # ── 4. Idea 5: dead dims ──────────────────────────────────────────────────
    print("\n  [5] Dead dim detection")
    _save_log(_plot_dead_dims(role_means, role_dim, ckpt_name), "dead_dims")

    # ── 5. Idea 6: inter-dim correlation ─────────────────────────────────────
    print("  [6] Inter-dim correlation")
    _save_log(_plot_dim_correlations(role_means, role_dim, ckpt_name), "dim_correlations")

    # ── 6. PCA behavioral scatter ─────────────────────────────────────────────
    print("  [PCA] Behavioral scatter (min/max speed + jerk)")
    fig_pca_beh, ev = _plot_pca_behavioral(role_means, stats, ckpt_name)
    _save_log(fig_pca_beh, "pca_behavioral")
    print(f"  PCA variance: PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  total={ev[:2].sum():.1%}")
    if wandb_run is not None:
        try:
            import wandb as _wandb
            wandb_run.log({"role/pca_var_pc1": float(ev[0]),
                           "role/pca_var_pc2": float(ev[1])})
        except Exception:
            pass

    # ── 7. Idea 8: cluster spider ─────────────────────────────────────────────
    print("  [8] Cluster spider chart")
    _save_log(
        _plot_cluster_spider(role_means, stats, labels, K, ckpt_name),
        "cluster_spider",
    )

    # ── 8. Idea 3: action sensitivity ────────────────────────────────────────
    print("  [3] Action distribution sensitivity")
    try:
        fig_kl = _plot_action_sensitivity(
            policy, rep_roles, data["obs_snaps"], device, ckpt_name
        )
        if fig_kl is not None:
            _save_log(fig_kl, "action_kl")
    except Exception as e:
        print(f"  [Action sensitivity skipped: {e}]")
        import traceback; traceback.print_exc()

    # ── 9. Idea 2: role intervention ──────────────────────────────────────────
    print("  [2] Role intervention (injecting real role values)")
    try:
        fig_prof, fig_traj = _plot_role_intervention(
            policy, env, rep_roles, device, ckpt_name, n_rollouts=5
        )
        _save_log(fig_prof, "intervention_profiles")
        _save_log(fig_traj, "intervention_trajectories")
    except Exception as e:
        print(f"  [Role intervention skipped: {e}]")
        import traceback; traceback.print_exc()

    # ── 10. Legacy PCA scatter (mean_speed / ang_speed / accel_std) ──────────
    print("  [Legacy PCA] mean_speed / angular_speed / accel_std")
    try:
        pca    = PCA(n_components=2)
        role2d = pca.fit_transform(role_means)
        ev2    = pca.explained_variance_ratio_
        idx    = np.random.choice(N, min(N, 20_000), replace=False)

        fig_leg, axes_leg = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (metric, label, cmap) in zip(axes_leg, [
            (stats["mean_speed"],  "mean speed (m/s)",         "viridis"),
            (stats["mean_ang_sp"], "angular speed (rad/step)", "plasma"),
            (stats["accel_std"],   "accel std (m/s²)",         "inferno"),
        ]):
            sc = ax.scatter(role2d[idx, 0], role2d[idx, 1],
                            c=metric[idx], cmap=cmap, alpha=0.25, s=4, rasterized=True)
            plt.colorbar(sc, ax=ax, label=label, shrink=0.8)
            ax.set_xlabel(f"PC1 ({ev2[0]:.1%})")
            ax.set_ylabel(f"PC2 ({ev2[1]:.1%})")
            ax.set_title(label)
        fig_leg.suptitle(
            f"ROMA role space (PCA) — {N:,} agent-episodes  [{ckpt_name}]",
            fontsize=11,
        )
        plt.tight_layout()
        _save_log(fig_leg, "pca_scatter")
    except Exception as e:
        print(f"  [Legacy PCA skipped: {e}]")

    # ── 12. Legacy temporal dynamics ──────────────────────────────────────────
    print("  [Legacy] Temporal role dynamics")
    try:
        role_seqs    = data["role_seqs"]                      # (N_ep, T, B, role_dim)
        role_ot      = role_seqs.mean(axis=(0, 2))            # (T, role_dim)
        temporal_std = role_seqs.std(axis=1).mean()
        step_dist    = np.linalg.norm(np.diff(role_seqs, axis=1), axis=3).mean()

        print(f"  Temporal role std  : {temporal_std:.4f}")
        print(f"  Mean step L2 dist  : {step_dist:.4f}")

        fig_tmp, ax_t = plt.subplots(figsize=(12, 4))
        for d in range(role_dim):
            ax_t.plot(np.arange(91), role_ot[:, d], label=f"dim_{d}", linewidth=1.2)
        ax_t.set_xlabel("Timestep")
        ax_t.set_ylabel("Mean role value")
        ax_t.set_title(
            f"Role dynamics over episode  [{ckpt_name}]  "
            f"(temporal_std={temporal_std:.4f}  step_dist={step_dist:.4f})"
        )
        ax_t.legend(loc="upper right", fontsize=8, ncol=2)
        ax_t.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_log(fig_tmp, "temporal")

        if wandb_run is not None:
            try:
                import wandb as _wandb
                wandb_run.log({
                    "role/temporal_std":      float(temporal_std),
                    "role/mean_step_l2_dist": float(step_dist),
                })
            except Exception:
                pass
    except Exception as e:
        print(f"  [Temporal skipped: {e}]")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point (eval_roma.py --checkpoint ...)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",        type=str,   required=True)
    p.add_argument("--role_dim",          type=int,   default=8)
    p.add_argument("--obs_dim",           type=int,   default=1121)
    p.add_argument("--num_agents",        type=int,   default=3072)
    p.add_argument("--device",            type=str,   default="cuda")
    p.add_argument("--map_dir",           type=str,   default="resources/drive/binaries/training")
    p.add_argument("--wosac_rollouts",    type=int,   default=32)
    p.add_argument("--wosac_num_maps",    type=int,   default=10000)
    p.add_argument("--wosac_max_batches", type=int,   default=100)
    p.add_argument("--output_dir",        type=str,   default="eval_results")
    p.add_argument("--wandb",             action="store_true")
    p.add_argument("--wandb_project",     type=str,   default="roma-pufferdrive")
    p.add_argument("--wandb_run_name",    type=str,   default=None)
    p.add_argument("--role_episodes",     type=int,   default=10,
                   help="Episodes for role analysis. 0 = skip.")
    # Route guidance — MUST match the values used in training so the eval env
    # reproduces the goal/route regime the policy was trained under.
    p.add_argument("--use_guided_autonomy",       type=int,   default=0,
                   help="1=route-guidance env (waypoint goals), 0=off. Match training.")
    p.add_argument("--waypoint_reach_threshold",  type=float, default=2.0)
    p.add_argument("--guidance_speed_weight",     type=float, default=0.0)
    p.add_argument("--guidance_heading_weight",   type=float, default=0.0)
    p.add_argument("--use_guidance_observations", type=int,   default=0,
                   help="1=add 182 waypoint coords to obs. Match training (you used 0).")
    return p.parse_args()


def evaluate(args):
    wandb_run = None
    if args.wandb:
        import wandb
        run_name  = args.wandb_run_name or Path(args.checkpoint).stem
        wandb_run = wandb.init(project=args.wandb_project, name=run_name,
                               config=vars(args))
        wandb_run.define_metric("wosac/batch")
        wandb_run.define_metric("wosac/*", step_metric="wosac/batch")

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"Checkpoint : {args.checkpoint}")

    from pufferlib.ocean.drive.drive import Drive
    ini       = load_drive_config()
    wosac_cfg = dict(ini["env"])
    wosac_cfg.update({
        "num_maps":      args.wosac_num_maps,
        "num_agents":    args.num_agents,
        "map_dir":       args.map_dir,
        "control_mode":  ini["eval"]["wosac_control_mode"],
        "goal_behavior": ini["eval"]["wosac_goal_behavior"],
        "goal_radius":   ini["eval"]["wosac_goal_radius"],
        # Route-guidance: the eval env MUST match training, otherwise a
        # guidance-trained policy is rolled out under goal/route conditions it
        # never saw, corrupting WOSAC.
        "use_guided_autonomy":       args.use_guided_autonomy,
        "guidance_speed_weight":     args.guidance_speed_weight,
        "guidance_heading_weight":   args.guidance_heading_weight,
        "waypoint_reach_threshold":  args.waypoint_reach_threshold,
        "use_guidance_observations": args.use_guidance_observations,
    })
    env = Drive(**wosac_cfg)

    # Auto-detect obs_dim from the env (so use_guidance_observations on/off can
    # never mismatch a hardcoded width), then load the policy at that width.
    obs_probe, _ = env.reset()
    obs_dim      = obs_probe.shape[-1]
    if obs_dim != args.obs_dim:
        print(f"obs_dim auto-detected as {obs_dim} (overriding --obs_dim {args.obs_dim})")
    policy = load_policy(args.checkpoint, args.role_dim, obs_dim, device)

    # ── Role analysis FIRST — guaranteed even if WOSAC times out ─────────────
    if args.role_episodes > 0:
        try:
            run_role_analysis(
                policy, env, args.role_episodes, args.role_dim,
                device, args.output_dir, Path(args.checkpoint).stem,
                wandb_run=wandb_run,
            )
        except Exception as e:
            print(f"[role analysis skipped: {e}]")
            import traceback; traceback.print_exc()

    # ── WOSAC realism metrics ─────────────────────────────────────────────────
    import pandas as pd
    from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

    wosac_config = {
        "eval":  {"wosac_init_steps": 10, "wosac_num_rollouts": args.wosac_rollouts},
        "train": {"device": str(device)},
    }
    evaluator        = WOSACEvaluator(wosac_config)
    adapter          = WOSACPolicyAdapter(policy, env.num_agents, device)
    all_results      = []
    unique_scenarios = set()

    print(f"\nrollouts={args.wosac_rollouts} | num_maps={args.wosac_num_maps} | "
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
        except Exception as e:
            import traceback
            print(f"[WOSAC ERROR batch {batch+1}: {type(e).__name__}: {e}]", flush=True)
            traceback.print_exc()

        if (batch + 1) % 10 == 0 and all_results:
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
    if "ade" in agg:
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
