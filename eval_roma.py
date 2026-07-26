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
# Main role analysis orchestrator
# ---------------------------------------------------------------------------

def run_role_analysis(policy, env, num_episodes, role_dim, device,
                      out_dir, ckpt_name, wandb_run=None):
    """
    SLIM role-health check (trimmed Jul 2026): only the two cheap metrics
    worth auto-logging for every trained checkpoint.

    dim_correlations   inter-dim Pearson r heatmap (redundancy; for dim-2 the
                       single off-diagonal number IS the readout)
    pca_behavioral     PCA scatter + explained variance (intrinsic
                       dimensionality) + wandb scalars pca_var_pc1/pc2 and
                       max_offdiag_r

    The old suite (dead_dims, K=5 cluster spider, action-KL, clamp-style role
    intervention, legacy PCA scatter, temporal dynamics) was removed: it
    clustered the whole population with no map/trajectory stratification,
    used constant-clamp forcing we have since rejected, and lacked the
    plausibility mask. The rigorous role analysis lives in
    role_paired_sweep.py / role_direct_cluster.py / render_role_alpha.py.
    Still called BEFORE WOSAC so a wall-time kill cannot lose it.
    """
    print("\n" + "=" * 60)
    print("  ROLE HEALTH CHECK (dim correlation + PCA)")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  Skipped (missing library: {e})")
        return

    if num_episodes <= 0:
        print("  Skipped (role_episodes=0)")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _save_log(fig, name):
        """Save the figure to disk first, then log the saved PNG *file* to wandb.

        Logging the file path (not the live matplotlib Figure) avoids the
        "No matching media" panels that appear when wandb can't resolve a
        figure object after it's been closed.
        """
        path = out_path / f"role_{name}_{ckpt_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        if wandb_run is not None:
            try:
                import wandb as _wandb
                wandb_run.log({f"role/{name}": _wandb.Image(str(path))})
            except Exception as e:
                print(f"  [wandb/{name}] {e}")

    # ── 1. Collect data ───────────────────────────────────────────────────────
    print(f"\n  Collecting {num_episodes} episodes...")
    data       = _collect_role_data(policy, env, num_episodes, device)
    role_means = data["role_means"]   # (N, role_dim)
    N          = len(role_means)
    print(f"  {N:,} agent-episode data points")

    # ── 2. Behavioral stats ───────────────────────────────────────────────────
    stats = _behavioral_stats(data["xs"], data["ys"], data["headings"])

    # ── 2b. Drop non-active / artifact agents ─────────────────────────────────
    # Inactive padding slots carry placeholder positions that np.diff turns
    # into impossible speeds; keep only physically-plausible moving agents.
    # A respawn is NOT a reason to delete an agent. This used to test
    # whole-episode max_speed against SPEED_CAP with no teleport handling, so a
    # single mid-episode respawn produced an impossible max_speed and the agent
    # was dropped -- 28,921 of 30,720 on nodiv dim4, i.e. 94%. That discard is
    # BIASED: agents respawn after collisions and off-road events, so the role
    # space was being built from the survivors, and PC1/max|r| disagreed with
    # role_paired_sweep on the same checkpoint.
    #
    # Now match role_paired_sweep: cut each agent at its first >TELEPORT_M jump
    # and judge motion on the segment BEFORE it, exactly as focal_metrics does.
    SPEED_CAP  = 45.0   # m/s -- above any real vehicle
    MIN_MOTION = 1.0    # m/s -- below this the agent never really moved
    from traj_kinematics import TELEPORT_M
    xs_, ys_ = data["xs"], data["ys"]                  # (N_ep, T, B)
    step = np.hypot(np.diff(xs_, axis=1), np.diff(ys_, axis=1))   # (N_ep,T-1,B)
    spd  = step * 10.0
    tele = step > TELEPORT_M
    # steps strictly before the first teleport are the usable segment
    pre  = np.cumsum(tele, axis=1) == 0
    n_pre = pre.sum(axis=1)                            # (N_ep, B)
    seg_max = np.where(pre, spd, -np.inf).max(axis=1)   # (N_ep, B)
    seg_max = np.where(n_pre >= 10, seg_max, np.nan)
    seg_max = seg_max.reshape(-1)
    valid  = (np.isfinite(seg_max)
              & (seg_max <= SPEED_CAP)
              & (seg_max >  MIN_MOTION))
    print(f"  teleport-truncated motion filter: median usable steps "
          f"{np.median(n_pre):.0f}/{xs_.shape[1]-1}")
    n_drop = int((~valid).sum())
    if valid.sum() >= 10:
        role_means = role_means[valid]
        stats      = {k: v[valid] for k, v in stats.items()}
        N          = len(role_means)
        print(f"  Dropped {n_drop:,} parked/padding agents "
              f"-> {N:,} moving agents analysed (respawned agents KEPT, "
              f"judged on their pre-respawn segment)")
    else:
        print(f"  [warn] only {int(valid.sum())} valid agents; skipping artifact filter")

    if role_dim >= 2:
        # ── 3. Inter-dim correlation ──────────────────────────────────────────
        print("\n  [corr] Inter-dim correlation")
        _save_log(_plot_dim_correlations(role_means, role_dim, ckpt_name),
                  "dim_correlations")
        corr = np.corrcoef(role_means.T)
        off  = np.abs(corr - np.eye(role_dim))
        max_off = float(off.max())
        print(f"  max |off-diagonal r| = {max_off:.3f}  (N={N})")

        # ── 4. PCA behavioral scatter + variance ──────────────────────────────
        print("  [PCA] Behavioral scatter (min/max speed + jerk)")
        fig_pca_beh, ev = _plot_pca_behavioral(role_means, stats, ckpt_name)
        _save_log(fig_pca_beh, "pca_behavioral")
        print(f"  PCA variance: PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  "
              f"total={ev[:2].sum():.1%}   (N={N} agents, teleport-truncated "
              f"motion filter -- role_paired_sweep uses vehicles-only and will "
              f"differ if N differs)")
        if wandb_run is not None:
            try:
                import wandb as _wandb
                wandb_run.log({"role/pca_var_pc1": float(ev[0]),
                               "role/pca_var_pc2": float(ev[1]),
                               "role/max_offdiag_r": max_off})
            except Exception:
                pass
    else:
        # role_dim == 1: inter-dim correlation and 2-D PCA are degenerate for a
        # scalar role (PCA(n_components=2) even raises on 1 feature). The dim-1
        # question is whether the scalar role is a smooth CONTINUUM or a
        # collapsed BINARY switch -> Sarle bimodality coefficient (BC > 0.555 =
        # bimodal/collapsed) + a histogram, plus how the scalar tracks speed.
        import matplotlib.pyplot as plt
        z = np.asarray(role_means[:, 0], dtype=np.float64)
        z = z[np.isfinite(z)]
        n = len(z)
        m, sd = float(z.mean()), float(z.std())
        if sd > 0 and n > 3:
            zn  = (z - m) / sd
            g1  = float((zn ** 3).mean())                       # skewness
            kex = float((zn ** 4).mean() - 3.0)                 # excess kurtosis
            bc  = (g1 ** 2 + 1.0) / (kex + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        else:
            g1 = kex = bc = float("nan")
        print("\n  [dim1] scalar role distribution (PCA/inter-dim corr skipped for 1-D)")
        print(f"  role: mean={m:.3f} std={sd:.3f}  Sarle BC={bc:.3f}"
              f"  ({'BIMODAL/collapsed' if bc > 0.555 else 'continuous-ish'})")
        r_speed = float("nan")
        if "mean_speed" in stats and n > 3:
            sp = np.asarray(stats["mean_speed"], dtype=np.float64)
            r  = np.asarray(role_means[:, 0], dtype=np.float64)
            ok = np.isfinite(sp) & np.isfinite(r)
            if ok.sum() > 3 and np.std(sp[ok]) > 0 and np.std(r[ok]) > 0:
                r_speed = float(np.corrcoef(r[ok], sp[ok])[0, 1])
        print(f"  corr(role, mean_speed) = {r_speed:.3f}")
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(z, bins=60, color="#4c72b0")
            ax.set_title(f"dim-1 role distribution  BC={bc:.3f}  [{ckpt_name}]")
            ax.set_xlabel("role value"); ax.set_ylabel("agent-episodes")
            _save_log(fig, "dim1_role_hist")
        except Exception:
            pass
        if wandb_run is not None:
            try:
                wandb_run.log({"role/bimodality_bc": bc,
                               "role/scalar_std": sd,
                               "role/corr_speed": r_speed})
            except Exception:
                pass


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
    policy = load_policy(args.checkpoint, args.role_dim, args.obs_dim, device)

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
    })
    env = Drive(**wosac_cfg)

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
        except Exception:
            pass

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
