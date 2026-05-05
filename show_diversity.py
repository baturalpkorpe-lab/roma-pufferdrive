"""
show_diversity.py
=================
Shows that the Waymo dataset contains genuine behavioural diversity
across real human drivers. This justifies the use of a diversity loss
in ROMA — if all drivers behaved identically, a diversity loss would
have nothing meaningful to capture.

What this script measures (from ground truth trajectories only,
no policy involved):
  1. Speed distribution          — do drivers have different typical speeds?
  2. Acceleration distribution   — do drivers accelerate differently?
  3. Heading change distribution — do drivers steer differently?
  4. Per-scenario speed variance — does each scenario contain diverse agents?
  5. Cross-scenario variance     — do different scenarios show different styles?

Output:
  - Console summary statistics
  - diversity_plots/diversity_overview.png  — 6-panel figure for the paper

Usage (from ~/PufferDrive):
    python3 roma_pufferdrive/show_diversity.py
    python3 roma_pufferdrive/show_diversity.py --num_maps 200 --output_dir roma_pufferdrive/diversity_plots
"""

import argparse
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--map_dir",    type=str, default="resources/drive/binaries/training")
    p.add_argument("--num_maps",   type=int, default=100,
                   help="Number of maps to load. More = more reliable statistics.")
    p.add_argument("--num_agents", type=int, default=32,
                   help="Agents per scene.")
    p.add_argument("--output_dir", type=str, default="roma_pufferdrive/diversity_plots")
    p.add_argument("--num_resets", type=int, default=20,
                   help="Number of env resets to collect trajectories from.")
    return p.parse_args()


def collect_ground_truth_stats(env, num_resets):
    """
    Collect behavioural statistics from ground truth trajectories.
    Returns arrays of per-agent statistics across all collected scenarios.
    """
    all_mean_speeds      = []
    all_max_speeds       = []
    all_mean_accels      = []
    all_heading_changes  = []
    all_scenario_speeds  = []   # list of per-scenario speed arrays (for within-scenario variance)

    print(f"Collecting ground truth trajectories from {num_resets} resets...")

    for i in range(num_resets):
        print(f"  Reset {i+1}/{num_resets} ...", end="\r")
        env.reset()

        try:
            gt = env.get_ground_truth_trajectories()
        except Exception as e:
            print(f"\n  Warning: get_ground_truth_trajectories() failed: {e}")
            continue

        # gt["x"], gt["y"], gt["heading"] — shape (num_agents, 1, num_steps)
        x       = gt["x"][:, 0, :]        # (num_agents, num_steps)
        y       = gt["y"][:, 0, :]
        heading = gt["heading"][:, 0, :]
        valid   = gt["valid"][:, 0, :]    # (num_agents, num_steps)  boolean

        num_agents, num_steps = x.shape

        # --- Speed (m per step, assuming 0.1s per step → ×10 for m/s) ---
        dx = np.diff(x, axis=1)    # (num_agents, num_steps-1)
        dy = np.diff(y, axis=1)
        speed = np.sqrt(dx**2 + dy**2) * 10.0   # m/s

        # Mask: both endpoints must be valid
        speed_valid = valid[:, :-1] & valid[:, 1:]
        speed_masked = np.where(speed_valid, speed, np.nan)

        agent_mean_speed = np.nanmean(speed_masked, axis=1)   # (num_agents,)
        agent_max_speed  = np.nanmax(speed_masked,  axis=1)

        # --- Acceleration (m/s² change between consecutive speed estimates) ---
        accel = np.abs(np.diff(speed, axis=1)) * 10.0         # (num_agents, num_steps-2)
        accel_valid = speed_valid[:, :-1] & speed_valid[:, 1:]
        accel_masked = np.where(accel_valid, accel, np.nan)
        agent_mean_accel = np.nanmean(accel_masked, axis=1)

        # --- Heading change (steering activity, radians/step) ---
        dh = np.abs(np.diff(heading, axis=1))
        # Wrap to [-pi, pi]
        dh = np.where(dh > np.pi, 2*np.pi - dh, dh)
        dh_valid = valid[:, :-1] & valid[:, 1:]
        dh_masked = np.where(dh_valid, dh, np.nan)
        agent_heading_change = np.nanmean(dh_masked, axis=1)

        # Filter agents with enough valid steps
        enough_valid = np.sum(valid, axis=1) >= 10
        if enough_valid.sum() == 0:
            continue

        all_mean_speeds.extend(agent_mean_speed[enough_valid].tolist())
        all_max_speeds.extend(agent_max_speed[enough_valid].tolist())
        all_mean_accels.extend(agent_mean_accel[enough_valid].tolist())
        all_heading_changes.extend(agent_heading_change[enough_valid].tolist())

        # Per-scenario speed array — split by scenario_id so each box
        # in the plot represents genuinely different agents in one scene
        if "scenario_id" in gt:
            scenario_ids = gt["scenario_id"][:, 0]  # (num_agents,)
            for sid in np.unique(scenario_ids):
                mask = (scenario_ids == sid) & enough_valid
                if mask.sum() > 1:
                    all_scenario_speeds.append(agent_mean_speed[mask])
        else:
            # Fallback: treat each reset as one scenario
            valid_speeds = agent_mean_speed[enough_valid]
            if len(valid_speeds) > 1:
                all_scenario_speeds.append(valid_speeds)

    print()  # newline after \r

    return {
        "mean_speeds":     np.array(all_mean_speeds),
        "max_speeds":      np.array(all_max_speeds),
        "mean_accels":     np.array(all_mean_accels),
        "heading_changes": np.array(all_heading_changes),
        "scenario_speeds": all_scenario_speeds,
    }


def print_summary(stats):
    ms  = stats["mean_speeds"]
    ma  = stats["mean_accels"]
    hc  = stats["heading_changes"]
    sc  = stats["scenario_speeds"]

    print("\n" + "=" * 60)
    print("  WAYMO DATASET BEHAVIOURAL DIVERSITY SUMMARY")
    print("=" * 60)
    print(f"  Agents analysed          : {len(ms)}")
    print()
    print(f"  Mean speed (m/s)")
    print(f"    Mean  : {np.mean(ms):.3f}")
    print(f"    Std   : {np.std(ms):.3f}   ← high std = diverse speeds")
    print(f"    Min   : {np.min(ms):.3f}")
    print(f"    Max   : {np.max(ms):.3f}")
    print(f"    CV    : {np.std(ms)/np.mean(ms):.3f}   ← coefficient of variation")
    print()
    print(f"  Mean acceleration (m/s²)")
    print(f"    Mean  : {np.mean(ma):.3f}")
    print(f"    Std   : {np.std(ma):.3f}")
    print(f"    CV    : {np.std(ma)/(np.mean(ma)+1e-8):.3f}")
    print()
    print(f"  Mean heading change (rad/step)")
    print(f"    Mean  : {np.mean(hc):.4f}")
    print(f"    Std   : {np.std(hc):.4f}")
    print(f"    CV    : {np.std(hc)/(np.mean(hc)+1e-8):.3f}")
    print()

    # Within-scenario diversity
    if sc:
        within_vars = [np.std(s) for s in sc if len(s) > 1]
        print(f"  Within-scenario speed std (across {len(within_vars)} unique scenarios)")
        print(f"    Mean  : {np.mean(within_vars):.3f}  ← agents in same scene differ")
        print(f"    Std   : {np.std(within_vars):.3f}  ← variation between scenarios")
        print(f"    Min   : {np.min(within_vars):.3f}")
        print(f"    Max   : {np.max(within_vars):.3f}")
        print()
        print("  Interpretation:")
        cv = np.std(ms) / np.mean(ms)
        if cv > 0.5:
            print("  ✓ HIGH diversity — CV > 0.5. The dataset contains genuinely")
            print("    different driving styles. Diversity loss is well motivated.")
        elif cv > 0.2:
            print("  ✓ MODERATE diversity — CV > 0.2. Meaningful variation exists.")
        else:
            print("  △ LOW diversity — CV < 0.2. Consider whether diversity loss")
            print("    is necessary for this dataset.")

    print("=" * 60)


def make_plots(stats, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  matplotlib not installed — skipping plots.")
        print("  Run: pip install matplotlib")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ms  = stats["mean_speeds"]
    mxs = stats["max_speeds"]
    ma  = stats["mean_accels"]
    hc  = stats["heading_changes"]
    sc  = stats["scenario_speeds"]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT = "#00d4ff"
    ACCENT2 = "#ff6b6b"
    ACCENT3 = "#a8ff78"
    BG      = "#0f0f1a"
    PANEL   = "#1a1a2e"
    TEXT    = "#e8e8f0"

    def styled_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.spines[:].set_color("#333355")
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)

    # 1. Mean speed histogram
    ax1 = fig.add_subplot(gs[0, 0])
    styled_ax(ax1, "Mean Speed Distribution")
    ax1.hist(ms[~np.isnan(ms)], bins=40, color=ACCENT, alpha=0.85, edgecolor="none")
    ax1.axvline(np.nanmean(ms), color=ACCENT2, linewidth=2,
                label=f"mean={np.nanmean(ms):.2f}")
    ax1.set_xlabel("Speed (m/s)")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8, labelcolor=TEXT, framealpha=0.2)
    cv1 = np.nanstd(ms) / np.nanmean(ms)
    ax1.text(0.97, 0.92, f"CV={cv1:.2f}", transform=ax1.transAxes,
             ha="right", color=ACCENT3, fontsize=9)

    # 2. Max speed histogram
    ax2 = fig.add_subplot(gs[0, 1])
    styled_ax(ax2, "Max Speed Distribution")
    ax2.hist(mxs[~np.isnan(mxs)], bins=40, color=ACCENT, alpha=0.85, edgecolor="none")
    ax2.axvline(np.nanmean(mxs), color=ACCENT2, linewidth=2,
                label=f"mean={np.nanmean(mxs):.2f}")
    ax2.set_xlabel("Speed (m/s)")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8, labelcolor=TEXT, framealpha=0.2)

    # 3. Acceleration histogram
    ax3 = fig.add_subplot(gs[0, 2])
    styled_ax(ax3, "Acceleration Distribution")
    clean_ma = ma[~np.isnan(ma)]
    clip_ma  = clean_ma[clean_ma < np.percentile(clean_ma, 99)]
    ax3.hist(clip_ma, bins=40, color=ACCENT2, alpha=0.85, edgecolor="none")
    ax3.axvline(np.nanmean(clean_ma), color=ACCENT, linewidth=2,
                label=f"mean={np.nanmean(clean_ma):.2f}")
    ax3.set_xlabel("Acceleration (m/s²)")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=8, labelcolor=TEXT, framealpha=0.2)
    cv3 = np.nanstd(clean_ma) / (np.nanmean(clean_ma) + 1e-8)
    ax3.text(0.97, 0.92, f"CV={cv3:.2f}", transform=ax3.transAxes,
             ha="right", color=ACCENT3, fontsize=9)

    # 4. Heading change histogram
    ax4 = fig.add_subplot(gs[1, 0])
    styled_ax(ax4, "Steering Activity Distribution")
    clean_hc = hc[~np.isnan(hc)]
    ax4.hist(clean_hc, bins=40, color=ACCENT3, alpha=0.85, edgecolor="none")
    ax4.axvline(np.nanmean(clean_hc), color=ACCENT2, linewidth=2,
                label=f"mean={np.nanmean(clean_hc):.4f}")
    ax4.set_xlabel("Mean |Δheading| (rad/step)")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=8, labelcolor=TEXT, framealpha=0.2)
    cv4 = np.nanstd(clean_hc) / (np.nanmean(clean_hc) + 1e-8)
    ax4.text(0.97, 0.92, f"CV={cv4:.2f}", transform=ax4.transAxes,
             ha="right", color=ACCENT3, fontsize=9)

    # 5. Within-scenario speed variance (box plot per scenario)
    ax5 = fig.add_subplot(gs[1, 1])
    styled_ax(ax5, "Within-Scenario Speed Diversity")
    if sc:
        # Show first 15 scenarios as box plots
        plot_sc = sc[:15]
        bp = ax5.boxplot(plot_sc, patch_artist=True,
                         medianprops=dict(color=ACCENT2, linewidth=2),
                         whiskerprops=dict(color=TEXT),
                         capprops=dict(color=TEXT),
                         flierprops=dict(marker="o", color=ACCENT, markersize=3))
        for patch in bp["boxes"]:
            patch.set_facecolor(PANEL)
            patch.set_edgecolor(ACCENT)
        ax5.set_xlabel("Scenario index")
        ax5.set_ylabel("Agent speed (m/s)")
        within_stds = [np.std(s) for s in sc if len(s) > 1]
        ax5.text(0.97, 0.92, f"avg std={np.mean(within_stds):.2f}",
                 transform=ax5.transAxes, ha="right", color=ACCENT3, fontsize=9)
    else:
        ax5.text(0.5, 0.5, "No scenario data", transform=ax5.transAxes,
                 ha="center", color=TEXT)

    # 6. Speed vs heading change scatter (behavioural style space)
    ax6 = fig.add_subplot(gs[1, 2])
    styled_ax(ax6, "Behavioural Style Space")
    n = min(len(ms), len(hc))
    valid_idx = ~(np.isnan(ms[:n]) | np.isnan(hc[:n]))
    sc6 = ax6.scatter(ms[:n][valid_idx], hc[:n][valid_idx],
                      alpha=0.3, s=8, c=ms[:n][valid_idx],
                      cmap="cool", rasterized=True)
    cbar = plt.colorbar(sc6, ax=ax6)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)
    cbar.set_label("Speed (m/s)", color=TEXT, fontsize=9)
    ax6.set_xlabel("Mean speed (m/s)")
    ax6.set_ylabel("Steering activity (rad/step)")
    ax6.text(0.03, 0.92, "Each point = one agent",
             transform=ax6.transAxes, color=TEXT, fontsize=8, alpha=0.7)

    # Title
    fig.suptitle(
        "Waymo Open Motion Dataset — Behavioural Diversity Analysis\n"
        "Justification for ROMA Diversity Loss",
        color=TEXT, fontsize=13, fontweight="bold", y=0.98
    )

    out = Path(output_dir) / "diversity_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Plot saved -> {out}")


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
        print("Check that get_ground_truth_trajectories() works in your Drive version.")
        return

    print_summary(stats)
    make_plots(stats, args.output_dir)

    print("\nDone. Use diversity_overview.png in your supervisor presentation.")


if __name__ == "__main__":
    main()
