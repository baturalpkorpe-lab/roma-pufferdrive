"""
show_diversity.py
=================
Behavioural diversity analysis of the Waymo dataset.
Now includes:
  1. Mean speed distribution
  2. Max speed distribution
  3. Acceleration distribution
  4. Steering activity distribution
  5. Within-scenario speed diversity (box plot per scenario)
  6. Behavioural style space — ground truth vs simulated agents overlay
  7. Steering vs max speed correlation
  8. Acceleration vs mean speed correlation

Usage (from ~/PufferDrive):
    python3 roma_pufferdrive/show_diversity.py --num_maps 100 --num_resets 30
    python3 roma_pufferdrive/show_diversity.py --num_maps 100 --num_resets 30 \
        --checkpoint roma_pufferdrive/checkpoints/roma_structured/roma_dim8_final.pt \
        --role_dim 8
"""

import argparse
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--map_dir",     type=str, default="resources/drive/binaries/training")
    p.add_argument("--num_maps",    type=int, default=100)
    p.add_argument("--num_agents",  type=int, default=32)
    p.add_argument("--output_dir",  type=str, default="roma_pufferdrive/diversity_plots")
    p.add_argument("--num_resets",  type=int, default=30)
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Optional: path to checkpoint to overlay simulated agents")
    p.add_argument("--role_dim",    type=int, default=8)
    p.add_argument("--obs_dim",     type=int, default=1121)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ground truth collection
# ---------------------------------------------------------------------------

def collect_ground_truth_stats(env, num_resets):
    all_mean_speeds     = []
    all_max_speeds      = []
    all_mean_accels     = []
    all_heading_changes = []
    all_scenario_speeds = []

    print(f"Collecting ground truth trajectories from {num_resets} resets...")
    seen_scenario_ids = set()

    for i in range(num_resets):
        print(f"  Reset {i+1}/{num_resets} ...", end="\r")
        # Resample maps to get new unique scenarios each iteration
        env.resample_maps()
        env.reset()

        try:
            gt = env.get_ground_truth_trajectories()
        except Exception as e:
            print(f"\n  Warning: {e}")
            continue

        x       = gt["x"][:, 0, :]
        y       = gt["y"][:, 0, :]
        heading = gt["heading"][:, 0, :]
        valid   = gt["valid"][:, 0, :]

        dx = np.diff(x, axis=1)
        dy = np.diff(y, axis=1)
        speed = np.sqrt(dx**2 + dy**2) * 10.0
        speed_valid  = valid[:, :-1] & valid[:, 1:]
        speed_masked = np.where(speed_valid, speed, np.nan)

        agent_mean_speed = np.nanmean(speed_masked, axis=1)
        agent_max_speed  = np.nanmax(speed_masked,  axis=1)

        accel        = np.abs(np.diff(speed, axis=1)) * 10.0
        accel_valid  = speed_valid[:, :-1] & speed_valid[:, 1:]
        accel_masked = np.where(accel_valid, accel, np.nan)
        agent_mean_accel = np.nanmean(accel_masked, axis=1)

        dh = np.abs(np.diff(heading, axis=1))
        dh = np.where(dh > np.pi, 2*np.pi - dh, dh)
        dh_valid  = valid[:, :-1] & valid[:, 1:]
        dh_masked = np.where(dh_valid, dh, np.nan)
        agent_heading_change = np.nanmean(dh_masked, axis=1)

        enough_valid = np.sum(valid, axis=1) >= 10
        if enough_valid.sum() == 0:
            continue

        all_mean_speeds.extend(agent_mean_speed[enough_valid].tolist())
        all_max_speeds.extend(agent_max_speed[enough_valid].tolist())
        all_mean_accels.extend(agent_mean_accel[enough_valid].tolist())
        all_heading_changes.extend(agent_heading_change[enough_valid].tolist())

        if "scenario_id" in gt:
            scenario_ids = gt["scenario_id"][:, 0]
            for sid in np.unique(scenario_ids):
                sid_str = str(sid)
                if sid_str in seen_scenario_ids:
                    continue
                seen_scenario_ids.add(sid_str)
                mask = (scenario_ids == sid) & enough_valid
                if mask.sum() > 1:
                    all_scenario_speeds.append(agent_mean_speed[mask])
        else:
            valid_speeds = agent_mean_speed[enough_valid]
            if len(valid_speeds) > 1:
                all_scenario_speeds.append(valid_speeds)

    print()
    return {
        "mean_speeds":     np.array(all_mean_speeds),
        "max_speeds":      np.array(all_max_speeds),
        "mean_accels":     np.array(all_mean_accels),
        "heading_changes": np.array(all_heading_changes),
        "scenario_speeds": all_scenario_speeds,
    }


# ---------------------------------------------------------------------------
# Simulated agent collection (optional overlay)
# ---------------------------------------------------------------------------

def collect_simulated_stats(env, checkpoint_path, role_dim, obs_dim, num_episodes=10):
    """Roll out trained policy and collect same behavioural stats."""
    import torch
    from torch.distributions import Categorical

    print(f"\nCollecting simulated agent stats ({num_episodes} episodes)...")

    # Load policy with auto-detection
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    key  = "policy_state" if "policy_state" in ckpt else "policy"
    sd   = ckpt[key]
    keys = set(sd.keys())

    if any("ego_enc" in k or "role_encoder" in k for k in keys):
        from roma.policy import RomaPolicy
        policy = RomaPolicy(obs_dim=obs_dim, role_dim=role_dim)
    else:
        # Legacy flat MLP
        import torch.nn as nn
        class LegacyPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.obs_encoder = nn.Sequential(
                    nn.Linear(obs_dim, 256), nn.ReLU(),
                    nn.Linear(256, 128),    nn.ReLU())
                self.role_fc  = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU())
                self.role_gru = nn.GRUCell(64, 64)
                self.mu_head  = nn.Linear(64, role_dim)
                self.logvar_head = nn.Linear(64, role_dim)
                self.policy_gru = nn.GRUCell(128 + role_dim, 128)
                self.actor  = nn.Linear(128, 91)
                self.critic = nn.Linear(128, 1)
            def initial_state(self, B, device):
                return (torch.zeros(B,64,device=device),
                        torch.zeros(B,128,device=device),
                        torch.zeros(B,8,obs_dim,device=device))
            def forward(self, obs, state):
                rh, ph, ow = state
                nrh  = self.role_gru(self.role_fc(obs), rh)
                rz   = self.mu_head(nrh)
                emb  = self.obs_encoder(obs)
                nph  = self.policy_gru(torch.cat([emb, rz], dim=-1), ph)
                now  = torch.cat([ow[:, 1:, :], obs.unsqueeze(1)], dim=1)
                return self.actor(nph), self.critic(nph), (nrh, nph, now), {"role_z": rz}
        policy = LegacyPolicy()

    policy.load_state_dict(sd)
    policy.eval()

    sim_mean_speeds     = []
    sim_max_speeds      = []
    sim_mean_accels     = []
    sim_heading_changes = []
    num_agents = env.num_agents

    for ep in range(num_episodes):
        print(f"  Episode {ep+1}/{num_episodes} ...", end="\r")
        obs_np, _ = env.reset()
        obs   = torch.tensor(obs_np, dtype=torch.float32)
        state = policy.initial_state(num_agents, device)

        # Use get_global_agent_state for accurate x, y, heading
        agent_state = env.get_global_agent_state()
        traj_x   = [agent_state["x"].copy()]
        traj_y   = [agent_state["y"].copy()]
        traj_h   = [agent_state["heading"].copy()]

        for _ in range(91):
            with torch.no_grad():
                logits, _, state, _ = policy(obs, state)
            action  = Categorical(logits=logits).sample()
            obs_np, _, _, _, _ = env.step(action.numpy().reshape(num_agents, 1))
            obs = torch.tensor(obs_np, dtype=torch.float32)
            agent_state = env.get_global_agent_state()
            traj_x.append(agent_state["x"].copy())
            traj_y.append(agent_state["y"].copy())
            traj_h.append(agent_state["heading"].copy())

        tx = np.array(traj_x).T   # (num_agents, 92)
        ty = np.array(traj_y).T
        th = np.array(traj_h).T

        dx    = np.diff(tx, axis=1)
        dy    = np.diff(ty, axis=1)
        speed = np.sqrt(dx**2 + dy**2) * 10.0

        # Drop last 2 steps — PufferDrive respawns agents at episode end
        # causing position jumps of 100,000+ m/s which are not real speeds
        speed = speed[:, :-2]

        # Also clip any remaining respawn artifacts (mid-episode collisions)
        speed = np.where(speed > 40.0, np.nan, speed)

        sim_mean_speeds.extend(np.nanmean(speed, axis=1).tolist())
        sim_max_speeds.extend(np.nanmax(speed,  axis=1).tolist())

        accel = np.abs(np.diff(speed, axis=1)) * 10.0
        sim_mean_accels.extend(np.nanmean(accel, axis=1).tolist())

        dh = np.abs(np.diff(th, axis=1))
        dh = np.where(dh > np.pi, 2*np.pi - dh, dh)
        dh = dh[:, :-2]  # drop last 2 steps to match speed array
        sim_heading_changes.extend(np.nanmean(dh, axis=1).tolist())

    print()
    return {
        "mean_speeds":     np.array(sim_mean_speeds),
        "max_speeds":      np.array(sim_max_speeds),
        "mean_accels":     np.array(sim_mean_accels),
        "heading_changes": np.array(sim_heading_changes),
    }


# ---------------------------------------------------------------------------
# Summary print
# ---------------------------------------------------------------------------

def print_summary(stats):
    ms = stats["mean_speeds"]
    ma = stats["mean_accels"]
    hc = stats["heading_changes"]
    sc = stats["scenario_speeds"]

    from scipy.stats import pearsonr

    print("\n" + "=" * 65)
    print("  WAYMO DATASET BEHAVIOURAL DIVERSITY SUMMARY")
    print("=" * 65)
    print(f"  Agents analysed : {len(ms)}")
    print()
    print(f"  Mean speed (m/s)     Mean={np.mean(ms):.3f}  Std={np.std(ms):.3f}  CV={np.std(ms)/np.mean(ms):.3f}")
    print(f"  Max speed  (m/s)     Mean={np.mean(stats['max_speeds']):.3f}  Std={np.std(stats['max_speeds']):.3f}")
    print(f"  Acceleration(m/s²)   Mean={np.mean(ma):.3f}  Std={np.std(ma):.3f}  CV={np.std(ma)/(np.mean(ma)+1e-8):.3f}")
    print(f"  Steering(rad/step)   Mean={np.mean(hc):.4f} Std={np.std(hc):.4f}  CV={np.std(hc)/(np.mean(hc)+1e-8):.3f}")
    print()

    # Pairwise correlations
    mxs = stats["max_speeds"]
    pairs = [
        ("Mean speed",  ms,  "Max speed",   mxs),
        ("Mean speed",  ms,  "Steering",    hc),
        ("Mean speed",  ms,  "Acceleration",ma),
        ("Max speed",   mxs, "Steering",    hc),
        ("Max speed",   mxs, "Acceleration",ma),
        ("Steering",    hc,  "Acceleration",ma),
    ]
    print("  Pairwise Pearson correlations:")
    print(f"  {'Pair':<40} {'r':>8}  {'p':>10}")
    for name1, a1, name2, a2 in pairs:
        n = min(len(a1), len(a2))
        valid = ~(np.isnan(a1[:n]) | np.isnan(a2[:n]))
        r, p = pearsonr(a1[:n][valid], a2[:n][valid])
        print(f"  {name1+' vs '+name2:<40} {r:>8.3f}  {p:>10.2e}")

    if sc:
        within_stds = [np.std(s) for s in sc if len(s) > 1]
        print(f"\n  Within-scenario speed std: mean={np.mean(within_stds):.3f}  "
              f"min={np.min(within_stds):.3f}  max={np.max(within_stds):.3f}")

    cv = np.std(ms) / np.mean(ms)
    print(f"\n  Verdict: CV={cv:.3f} → ", end="")
    if cv > 0.5:
        print("HIGH diversity. Diversity loss is well motivated.")
    elif cv > 0.2:
        print("MODERATE diversity.")
    else:
        print("LOW diversity.")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Plots — 8 panels
# ---------------------------------------------------------------------------

def make_plots(stats, sim_stats, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.stats import pearsonr

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ms  = stats["mean_speeds"]
    mxs = stats["max_speeds"]
    ma  = stats["mean_accels"]
    hc  = stats["heading_changes"]
    sc  = stats["scenario_speeds"]

    ACCENT  = "#00d4ff"
    ACCENT2 = "#ff6b6b"
    ACCENT3 = "#a8ff78"
    ACCENT4 = "#ffaa00"
    BG      = "#0f0f1a"
    PANEL   = "#1a1a2e"
    TEXT    = "#e8e8f0"

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(BG)
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    def styled_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[:].set_color("#333355")
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=7)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)

    def cv_text(ax, arr):
        cv = np.nanstd(arr) / (np.nanmean(arr) + 1e-8)
        ax.text(0.97, 0.92, f"CV={cv:.2f}", transform=ax.transAxes,
                ha="right", color=ACCENT3, fontsize=8)

    # 1. Mean speed
    ax1 = fig.add_subplot(gs[0, 0])
    styled_ax(ax1, "Mean Speed Distribution")
    ax1.hist(ms[~np.isnan(ms)], bins=35, color=ACCENT, alpha=0.85, edgecolor="none")
    ax1.axvline(np.nanmean(ms), color=ACCENT2, lw=2, label=f"mean={np.nanmean(ms):.2f}")
    ax1.set_xlabel("Speed (m/s)"); ax1.set_ylabel("Count")
    ax1.legend(fontsize=7, labelcolor=TEXT, framealpha=0.2)
    cv_text(ax1, ms)

    # 2. Max speed
    ax2 = fig.add_subplot(gs[0, 1])
    styled_ax(ax2, "Max Speed Distribution")
    ax2.hist(mxs[~np.isnan(mxs)], bins=35, color=ACCENT, alpha=0.85, edgecolor="none")
    ax2.axvline(np.nanmean(mxs), color=ACCENT2, lw=2, label=f"mean={np.nanmean(mxs):.2f}")
    ax2.set_xlabel("Speed (m/s)"); ax2.set_ylabel("Count")
    ax2.legend(fontsize=7, labelcolor=TEXT, framealpha=0.2)
    cv_text(ax2, mxs)

    # 3. Acceleration
    ax3 = fig.add_subplot(gs[0, 2])
    styled_ax(ax3, "Acceleration Distribution")
    clean_ma = ma[~np.isnan(ma)]
    clip_ma  = clean_ma[clean_ma < np.percentile(clean_ma, 99)]
    ax3.hist(clip_ma, bins=35, color=ACCENT2, alpha=0.85, edgecolor="none")
    ax3.axvline(np.nanmean(clean_ma), color=ACCENT, lw=2, label=f"mean={np.nanmean(clean_ma):.2f}")
    ax3.set_xlabel("Acceleration (m/s²)"); ax3.set_ylabel("Count")
    ax3.legend(fontsize=7, labelcolor=TEXT, framealpha=0.2)
    cv_text(ax3, clean_ma)

    # 4. Steering
    ax4 = fig.add_subplot(gs[0, 3])
    styled_ax(ax4, "Steering Activity Distribution")
    clean_hc = hc[~np.isnan(hc)]
    ax4.hist(clean_hc, bins=35, color=ACCENT3, alpha=0.85, edgecolor="none")
    ax4.axvline(np.nanmean(clean_hc), color=ACCENT2, lw=2, label=f"mean={np.nanmean(clean_hc):.4f}")
    ax4.set_xlabel("Mean |Δheading| (rad/step)"); ax4.set_ylabel("Count")
    ax4.legend(fontsize=7, labelcolor=TEXT, framealpha=0.2)
    cv_text(ax4, clean_hc)

    # 5. Within-scenario box plot
    ax5 = fig.add_subplot(gs[1, 0])
    styled_ax(ax5, "Within-Scenario Speed Diversity")
    if sc:
        plot_sc = sc[:15]
        bp = ax5.boxplot(plot_sc, patch_artist=True,
                         medianprops=dict(color=ACCENT2, linewidth=2),
                         whiskerprops=dict(color=TEXT),
                         capprops=dict(color=TEXT),
                         flierprops=dict(marker="o", color=ACCENT, markersize=3))
        for patch in bp["boxes"]:
            patch.set_facecolor(PANEL)
            patch.set_edgecolor(ACCENT)
        ax5.set_xlabel("Scenario index"); ax5.set_ylabel("Agent speed (m/s)")
        within_stds = [np.std(s) for s in sc if len(s) > 1]
        ax5.text(0.97, 0.92, f"avg std={np.mean(within_stds):.2f}",
                 transform=ax5.transAxes, ha="right", color=ACCENT3, fontsize=8)

    # 6. Behavioural style space — GT vs simulated overlay
    ax6 = fig.add_subplot(gs[1, 1])
    styled_ax(ax6, "Behavioural Style Space\n(GT vs Simulated)")
    n = min(len(ms), len(hc))
    valid_idx = ~(np.isnan(ms[:n]) | np.isnan(hc[:n]))
    ax6.scatter(ms[:n][valid_idx], hc[:n][valid_idx],
                alpha=0.4, s=10, color=ACCENT, label="Ground truth (real drivers)")
    if sim_stats is not None:
        sm  = sim_stats["mean_speeds"]
        shc = sim_stats["heading_changes"]
        ns  = min(len(sm), len(shc))
        vi  = ~(np.isnan(sm[:ns]) | np.isnan(shc[:ns]))
        ax6.scatter(sm[:ns][vi], shc[:ns][vi],
                    alpha=0.4, s=10, color=ACCENT2, label="Simulated (our policy)")
    ax6.set_xlabel("Mean speed (m/s)")
    ax6.set_ylabel("Steering activity (rad/step)")
    ax6.legend(fontsize=7, labelcolor=TEXT, framealpha=0.3,
               facecolor=PANEL, edgecolor="#333355")

    # 7. Steering vs max speed scatter + correlation
    ax7 = fig.add_subplot(gs[1, 2])
    styled_ax(ax7, "Steering vs Max Speed")
    n7 = min(len(mxs), len(hc))
    vi7 = ~(np.isnan(mxs[:n7]) | np.isnan(hc[:n7]))
    ax7.scatter(mxs[:n7][vi7], hc[:n7][vi7],
                alpha=0.3, s=8, c=mxs[:n7][vi7], cmap="plasma", rasterized=True)
    r7, _ = pearsonr(mxs[:n7][vi7], hc[:n7][vi7])
    ax7.set_xlabel("Max speed (m/s)")
    ax7.set_ylabel("Steering activity (rad/step)")
    ax7.text(0.97, 0.92, f"r={r7:.3f}", transform=ax7.transAxes,
             ha="right", color=ACCENT3, fontsize=9)

    # 8. Acceleration vs mean speed scatter + correlation
    ax8 = fig.add_subplot(gs[1, 3])
    styled_ax(ax8, "Acceleration vs Mean Speed")
    n8 = min(len(ms), len(ma))
    vi8 = ~(np.isnan(ms[:n8]) | np.isnan(ma[:n8]))
    clip_thresh = np.percentile(ma[:n8][vi8], 99)
    vi8 = vi8 & (ma[:n8] < clip_thresh)
    ax8.scatter(ms[:n8][vi8], ma[:n8][vi8],
                alpha=0.3, s=8, c=ms[:n8][vi8], cmap="cool", rasterized=True)
    r8, _ = pearsonr(ms[:n8][vi8], ma[:n8][vi8])
    ax8.set_xlabel("Mean speed (m/s)")
    ax8.set_ylabel("Mean acceleration (m/s²)")
    ax8.text(0.97, 0.92, f"r={r8:.3f}", transform=ax8.transAxes,
             ha="right", color=ACCENT3, fontsize=9)

    fig.suptitle(
        "Waymo Open Motion Dataset — Behavioural Diversity Analysis\n"
        "Justification for ROMA Diversity Loss",
        color=TEXT, fontsize=13, fontweight="bold", y=0.99
    )

    out = Path(output_dir) / "diversity_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Plot saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(
        num_maps       = args.num_maps,
        num_agents     = args.num_agents,
        map_dir        = args.map_dir,
        episode_length = 91,
    )
    print(f"Environment loaded: {args.num_maps} maps, {args.num_agents} agents/scene")

    stats = collect_ground_truth_stats(env, args.num_resets)

    if len(stats["mean_speeds"]) == 0:
        print("\nERROR: No valid ground truth data collected.")
        return

    print_summary(stats)

    sim_stats = None
    if args.checkpoint:
        sim_stats = collect_simulated_stats(
            env, args.checkpoint, args.role_dim, args.obs_dim, num_episodes=10)

    make_plots(stats, sim_stats, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
