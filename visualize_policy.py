"""
visualize_policy.py
===================
Produces two clean visualizations:

1. behaviour_over_time.png — speed and steering for all agents over 91 steps
2. role_vs_behaviour.png   — one row per agent showing:
     - role vector values as colored cells (blue=negative, red=positive)
     - mean speed, max speed, mean acceleration, mean steering as bars
   Agents sorted by mean speed so you can see if role differs with speed.

Usage (from ~/PufferDrive):
    PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 \
        roma_pufferdrive/visualize_policy.py \
        --checkpoint roma_pufferdrive/checkpoints/roma_structured/roma_dim8_final.pt \
        --role_dim 8 --obs_dim 1121
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from torch.distributions import Categorical


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--role_dim",    type=int, default=8)
    p.add_argument("--obs_dim",     type=int, default=1121)
    p.add_argument("--num_agents",  type=int, default=16)
    p.add_argument("--num_maps",    type=int, default=100)
    p.add_argument("--map_dir",     type=str, default="resources/drive/binaries/training")
    p.add_argument("--output_dir",  type=str, default="roma_pufferdrive/eval_results")
    return p.parse_args()


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    device = torch.device("cpu")

    from roma.policy import RomaPolicy
    policy = RomaPolicy(obs_dim=args.obs_dim, role_dim=args.role_dim)
    ckpt = torch.load(args.checkpoint, map_location=device)
    key  = "policy_state" if "policy_state" in ckpt else "policy"
    policy.load_state_dict(ckpt[key])
    policy.eval()
    print(f"Loaded: {args.checkpoint}")

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(
        num_maps       = args.num_maps,
        num_agents     = args.num_agents,
        map_dir        = args.map_dir,
        episode_length = 91,
    )

    obs_np, _ = env.reset()
    obs   = torch.tensor(obs_np, dtype=torch.float32)
    state = policy.initial_state(args.num_agents, device)

    T      = 91
    T_plot = T - 2
    B      = args.num_agents

    speeds    = np.zeros((B, T_plot))
    steerings = np.zeros((B, T_plot))
    role_vecs = np.zeros((B, T_plot, args.role_dim))
    prev_state = env.get_global_agent_state()

    print("Running episode...")
    for t in range(T):
        with torch.no_grad():
            logits, _, state, role_info = policy(obs, state)
        action = Categorical(logits=logits).sample()
        obs_np, _, _, _, _ = env.step(action.numpy().reshape(B, 1))
        obs = torch.tensor(obs_np, dtype=torch.float32)
        curr_state = env.get_global_agent_state()

        if t < T_plot:
            dx = np.array(curr_state["x"]) - np.array(prev_state["x"])
            dy = np.array(curr_state["y"]) - np.array(prev_state["y"])
            speed = np.clip(np.sqrt(dx**2 + dy**2) * 10.0, 0, 40.0)
            dh = np.abs(np.array(curr_state["heading"]) - np.array(prev_state["heading"]))
            dh = np.clip(np.where(dh > np.pi, 2*np.pi - dh, dh), 0, np.pi)
            speeds    [:, t] = speed
            steerings [:, t] = dh
            role_vecs [:, t] = role_info["role_z"].numpy()

        prev_state = curr_state

    # Per-agent statistics
    mean_speed = speeds.mean(axis=1)
    max_speed  = speeds.max(axis=1)
    mean_accel = np.abs(np.diff(speeds, axis=1)).mean(axis=1) * 10.0
    mean_steer = steerings.mean(axis=1)
    mean_role  = role_vecs.mean(axis=1)  # (B, role_dim)

    # Sort by mean speed
    order      = np.argsort(mean_speed)[::-1]  # fastest first
    mean_speed = mean_speed[order]
    max_speed  = max_speed[order]
    mean_accel = mean_accel[order]
    mean_steer = mean_steer[order]
    mean_role  = mean_role[order]
    timesteps  = np.arange(T_plot)
    COLORS     = plt.cm.tab20(np.linspace(0, 1, B))

    BG    = "#0f0f1a"
    PANEL = "#1a1a2e"
    TEXT  = "#e8e8f0"

    def styled_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[:].set_color("#333355")
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=7)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)

    # -----------------------------------------------------------------------
    # Figure 1 — Behaviour over time
    # -----------------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    fig1.patch.set_facecolor(BG)
    fig1.suptitle("Agent Behaviour Over Time", color=TEXT,
                  fontsize=13, fontweight="bold")

    styled_ax(ax1, "Speed Over Time (all agents)")
    for i in range(B):
        ax1.plot(timesteps, speeds[i], alpha=0.5, linewidth=1, color=COLORS[i])
    ax1.plot(timesteps, speeds.mean(axis=0), color="white",
             linewidth=2.5, linestyle="--", label="mean")
    ax1.set_xlabel("Timestep"); ax1.set_ylabel("Speed (m/s)")
    ax1.legend(fontsize=8, labelcolor=TEXT, framealpha=0.2)

    styled_ax(ax2, "Steering Activity Over Time (all agents)")
    for i in range(B):
        ax2.plot(timesteps, steerings[i], alpha=0.5, linewidth=1, color=COLORS[i])
    ax2.plot(timesteps, steerings.mean(axis=0), color="white",
             linewidth=2.5, linestyle="--", label="mean")
    ax2.set_xlabel("Timestep"); ax2.set_ylabel("|Δheading| (rad/step)")
    ax2.legend(fontsize=8, labelcolor=TEXT, framealpha=0.2)

    fig1.tight_layout()
    out1 = Path(args.output_dir) / "behaviour_over_time.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig1)
    print(f"Saved -> {out1}")

    # -----------------------------------------------------------------------
    # Figure 2 — Role vector vs behaviour: proper aligned table
    # Layout: role_dim colored cells | mean_speed bar | max_speed bar |
    #         mean_accel bar | mean_steer bar
    # Each row = one agent. Aligned so you can read across.
    # -----------------------------------------------------------------------
    row_h    = 0.55
    fig_h    = B * row_h + 1.5
    n_cols   = args.role_dim + 4
    col_w    = 1.1
    fig_w    = n_cols * col_w + 2

    fig2, axes = plt.subplots(
        B, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.04, "hspace": 0.04}
    )
    fig2.patch.set_facecolor(BG)

    # Normalise role values for coloring
    role_abs_max = max(abs(mean_role.min()), abs(mean_role.max()), 1.0)
    cmap_role = plt.cm.coolwarm

    # Normalise stat values for bar width
    stat_data = [mean_speed, max_speed, mean_accel, mean_steer]
    stat_names = ["Mean\nSpeed\n(m/s)", "Max\nSpeed\n(m/s)",
                  "Mean\nAccel\n(m/s²)", "Mean\nSteering\n(rad/step)"]
    stat_colors = ["#00d4ff", "#a8ff78", "#ffaa00", "#ff6b6b"]
    stat_maxes  = [s.max() for s in stat_data]

    for row in range(B):
        agent_idx = order[row]

        # Role dimension cells
        for col in range(args.role_dim):
            ax = axes[row, col]
            val = mean_role[row, col]
            norm_val = (val + role_abs_max) / (2 * role_abs_max)  # 0..1
            color = cmap_role(norm_val)
            ax.set_facecolor(color)
            ax.text(0.5, 0.5, f"{val:+.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > role_abs_max * 0.4
                    else "#111111",
                    transform=ax.transAxes, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#0f0f1a")
                spine.set_linewidth(1.5)

            # Column header (top row only)
            if row == 0:
                ax.set_title(f"z[{col}]", color=TEXT, fontsize=8,
                             fontweight="bold", pad=4)

        # Stat bar cells
        for s_idx, (stat_vals, stat_name, stat_col, stat_max) in enumerate(
                zip(stat_data, stat_names, stat_colors, stat_maxes)):
            col = args.role_dim + s_idx
            ax  = axes[row, col]
            ax.set_facecolor(PANEL)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#0f0f1a")
                spine.set_linewidth(1.5)

            # Draw horizontal bar
            frac = stat_vals[row] / (stat_max + 1e-8)
            ax.barh([0], [frac], color=stat_col, alpha=0.85, height=0.6)
            ax.set_xlim(0, 1.15)
            ax.set_ylim(-0.5, 0.5)
            ax.text(frac + 0.03, 0, f"{stat_vals[row]:.2f}",
                    va="center", color=TEXT, fontsize=7)

            if row == 0:
                ax.set_title(stat_name, color=TEXT, fontsize=8,
                             fontweight="bold", pad=4)

        # Agent label on left
        axes[row, 0].set_ylabel(f"Agt {agent_idx}",
                                color=TEXT, fontsize=7, rotation=0,
                                labelpad=28, va="center")

    fig2.suptitle(
        "Role Vector vs Agent Behaviour\n"
        f"(16 agents sorted fastest→slowest  |  "
        f"Role cells: blue=negative, red=positive  |  "
        f"Bars: fraction of max value)",
        color=TEXT, fontsize=11, fontweight="bold", y=1.01
    )

    out2 = Path(args.output_dir) / "role_vs_behaviour.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig2)
    print(f"Saved -> {out2}")

    # Console table
    print("\nAgent summary (fastest → slowest):")
    print(f"{'Agt':<5} {'MSpd':>6} {'MxSpd':>6} {'MAcc':>6} {'MSteer':>8}  "
          f"Role vector")
    print("-" * 80)
    for i in range(B):
        role_str = " ".join(f"{v:+.2f}" for v in mean_role[i])
        print(f"{order[i]:<5} {mean_speed[i]:>6.2f} {max_speed[i]:>6.2f} "
              f"{mean_accel[i]:>6.3f} {mean_steer[i]:>8.4f}  [{role_str}]")


if __name__ == "__main__":
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run(args)
