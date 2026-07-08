"""
role_regime_analysis.py -- Phase C of the map-regime role analysis.

Answers: do the role variables predict HOW an agent deviates from the human
driver who occupied the same vehicle slot -- and does that association
replicate across map regimes (so it cannot be scenario information)?

Pipeline per episode (reset batch, ~500 scenarios at once):
  1. reset -> ground-truth trajectories (the human counterpart per agent slot)
  2. roll the policy 91 steps, recording positions, per-step role_mean,
     per-step rewards (safety-penalty events)
  3. per agent: cut at the first respawn teleport, keep only timesteps where
     the GT is valid, and compute GT-REFERENCED deltas:
        d_speed      policy mean speed - human mean speed  (same seat!)
        speed_ratio  policy / human mean speed
        ade          mean distance to the human's path at the same timesteps
        d_turn_rate  policy - human mean |dheading|/s
        d_jerk       policy - human accel std
        same_turn    did it take the turn the human took (only where the
                     human turned >= ~30 degrees)
        event_rate   safety-penalty steps per 91 (collision+offroad combined:
                     both penalties are -0.5 so they are indistinguishable)
  4. stratify by map regime (map_clusters.csv from Phase B): core maps only
     (margin > --min_margin), junk clusters excluded.

Analyses / outputs in --out_dir:
  phaseC_agent_data.csv      one row per usable agent-episode (for reuse)
  phaseC_corr_by_regime.png  role dim x delta-metric Pearson r, per regime
  phaseC_replication.png     dim x regime matrix per metric -- THE test:
                             same sign everywhere = real role semantics
  phaseC_within_scene.png    same correlations, within-scene z-scored
                             (strongest control: compares scene-mates only)
  phaseC_variance.png        variance decomposition: regime vs role vs noise,
                             for raw speed AND for the deltas
  phaseC_role_clusters.png   k-means role clusters per regime, mean delta
                             profiles (which cluster = faster-than-human etc.)

Usage (from /scratch/e452103/PufferDrive):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/role_regime_analysis.py \
        --checkpoint /scratch/e452103/checkpoints/roma_baseline/roma_dim8_step3000238080.pt \
        --clusters   /scratch/e452103/map_atlas/map_clusters.csv \
        --data_dir   pufferlib/resources/drive/binaries/training \
        --out_dir    /scratch/e452103/role_regime/dim8
"""

import argparse
import ast
import configparser
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.distributions import Categorical

T = 91
TELEPORT_M   = 4.0    # per-step jump above this = respawn -> cut segment.
                      # Lowered from 8.0: diag_speed_spikes.py confirmed respawns
                      # (goal-reach, +1.0 reward) land ~4-8 m away, under the old
                      # 8 m cut, spiking speed_max/jerk in ~19% of segments. 4 m
                      # (=40 m/s) drops only the artifacts, keeps real behavior.
MIN_STEPS    = 10     # min overlapping valid steps for usable metrics
GT_MOVE_MS   = 1.0    # human counterpart must actually drive
EVENT_REW    = -0.4   # reward <= this counts as a safety penalty event
TURN_NET_RAD = 0.5    # human net heading change >= this = "a turn happened"

METRICS       = ["d_speed", "speed_ratio", "ade", "d_turn_rate", "d_jerk",
                 "event_rate"]
CORR_METRICS  = ["d_speed", "ade", "d_turn_rate", "d_jerk", "event_rate"]
REGIME_NAMES_DEFAULT = ("0:quiet local,1:fast corridors,"
                        "2:parking/low-speed,4:congested urban")


def load_drive_config():
    import pufferlib
    puffer_dir = os.path.dirname(pufferlib.__file__)
    p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    p.read([os.path.join(puffer_dir, "config", "default.ini"),
            os.path.join(puffer_dir, "config", "ocean", "drive.ini")])

    def _parse(v):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v

    return {s: {k: _parse(v) for k, v in p[s].items()} for s in p.sections()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--clusters",   type=str, default=None,
                   help="map_clusters.csv from map_atlas.py (Phase B)")
    p.add_argument("--data_dir",   type=str, default=None)
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--replot",     action="store_true",
                   help="Skip collection: rebuild all figures from "
                        "<out_dir>/phaseC_agent_data.csv (seconds, no GPU)")
    p.add_argument("--episodes",   type=int, default=20,
                   help="Reset batches; each covers ~500 scenarios")
    p.add_argument("--num_agents", type=int, default=3072)
    p.add_argument("--num_maps",   type=int, default=10000)
    p.add_argument("--min_margin", type=float, default=1.5,
                   help="Core-map filter: keep maps clearly inside one regime")
    p.add_argument("--exclude_clusters", type=str, default="3",
                   help="Comma-separated regime ids to drop (GT-artifact bins)")
    p.add_argument("--regime_names", type=str, default=REGIME_NAMES_DEFAULT)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--kmeans_k",   type=int, default=3,
                   help="Role clusters per regime for the profile figure")
    return p.parse_args()


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _squeeze(a):
    a = np.asarray(a)
    return a[:, 0] if a.ndim >= 2 and a.shape[1] == 1 else a


def load_policy(ckpt_path, obs_dim, device):
    from roma_pufferdrive.roma.policy import RomaPolicy
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {}) or {}
    role_dim = saved.get("role_dim", 8)
    policy = RomaPolicy(
        obs_dim=obs_dim, action_dim=91, role_dim=role_dim,
        role_hidden=saved.get("role_hidden", 64),
        policy_hidden=saved.get("policy_hidden", 128),
        var_floor=saved.get("var_floor", 1e-4), obs_window_len=8,
    ).to(device)
    key = "policy_state" if "policy_state" in ckpt else "policy"
    policy.load_state_dict(ckpt[key])
    policy.eval()
    return policy, role_dim


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect(args, device):
    import pandas as pd

    cl = pd.read_csv(args.clusters)
    excl = {int(v) for v in args.exclude_clusters.split(",") if v != ""}
    cl = cl[(cl["margin"] > args.min_margin) & (~cl["cluster"].isin(excl))]
    regime_of = dict(zip(cl["scenario_id"].astype(str), cl["cluster"]))
    print(f"[phaseC] {len(regime_of)} core scenarios across regimes "
          f"{sorted(cl['cluster'].unique())}")

    from pufferlib.ocean.drive.drive import Drive
    env_cfg = dict(load_drive_config()["env"])
    env_cfg.update({"num_maps": args.num_maps, "num_agents": args.num_agents,
                    "map_dir": args.data_dir})
    env = Drive(**env_cfg)
    B   = args.num_agents

    obs_np, _ = env.reset()
    obs_dim   = obs_np.shape[-1]
    policy, role_dim = load_policy(args.checkpoint, obs_dim, device)
    print(f"[phaseC] policy loaded: role_dim={role_dim} obs_dim={obs_dim}")

    rows = []
    for ep in range(args.episodes):
        if ep > 0:
            env.resample_maps()
        obs_np, _ = env.reset()

        gt   = env.get_ground_truth_trajectories()
        gx, gy   = _squeeze(gt["x"]), _squeeze(gt["y"])
        gh       = _squeeze(gt["heading"])
        gvalid   = _squeeze(gt["valid"]).astype(bool)
        is_veh   = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
        sid      = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
        T_gt     = gx.shape[1]

        xs   = np.zeros((T, B), dtype=np.float32)
        ys   = np.zeros((T, B), dtype=np.float32)
        hs   = np.zeros((T, B), dtype=np.float32)
        rews = np.zeros((T, B), dtype=np.float32)
        rl   = np.zeros((T, B, role_dim), dtype=np.float32)

        obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        state = policy.initial_state(B, device)
        for t in range(T):
            ag = env.get_global_agent_state()
            xs[t], ys[t], hs[t] = ag["x"], ag["y"], ag["heading"]
            with torch.no_grad():
                logits, _, state, role_info = policy(obs, state)
            rl[t] = role_info["role_mean"].float().cpu().numpy()
            action = Categorical(logits=logits.float()).sample()
            obs_np, rew_np, _, _, _ = env.step(
                action.cpu().numpy().reshape(B, 1))
            rews[t] = np.asarray(rew_np).reshape(B)
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        # Per-agent metrics
        step_d = np.sqrt(np.diff(xs, axis=0)**2 + np.diff(ys, axis=0)**2)  # (T-1,B)
        for a in range(B):
            regime = regime_of.get(sid[a])
            if regime is None or not is_veh[a]:
                continue

            # pre-respawn segment: cut at the first teleport
            jumps = np.where(step_d[:, a] > TELEPORT_M)[0]
            t_end = int(jumps[0] + 1) if len(jumps) else T
            t_end = min(t_end, T_gt)
            if t_end < MIN_STEPS + 1:
                continue

            v = gvalid[a, :t_end]
            pair = v[:-1] & v[1:]                     # valid consecutive pairs
            if pair.sum() < MIN_STEPS:
                continue

            # human counterpart
            g_dx  = (gx[a, 1:t_end] - gx[a, :t_end-1])[pair]
            g_dy  = (gy[a, 1:t_end] - gy[a, :t_end-1])[pair]
            g_spd = np.hypot(g_dx, g_dy) * 10
            if g_spd.max() <= GT_MOVE_MS:
                continue                               # human was parked
            g_dh  = wrap_angle((gh[a, 1:t_end] - gh[a, :t_end-1])[pair]) * 10
            g_acc = np.diff(g_spd)

            # policy agent over the same timesteps
            p_dx  = np.diff(xs[:t_end, a])[pair]
            p_dy  = np.diff(ys[:t_end, a])[pair]
            p_spd = np.hypot(p_dx, p_dy) * 10
            p_dh  = wrap_angle(np.diff(hs[:t_end, a])[pair]) * 10
            p_acc = np.diff(p_spd)

            # position error vs the human path, same timesteps
            vi   = np.where(v)[0]
            ade  = float(np.mean(np.hypot(xs[vi, a] - gx[a, vi],
                                          ys[vi, a] - gy[a, vi])))

            # same-turn: compare net heading change where the human turned
            g_net = float(np.sum(g_dh)) / 10
            p_net = float(np.sum(p_dh)) / 10
            if abs(g_net) >= TURN_NET_RAD:
                same_turn = float(np.sign(g_net) == np.sign(p_net)
                                  and abs(p_net) >= TURN_NET_RAD / 2)
            else:
                same_turn = np.nan

            row = {
                "scenario_id": sid[a],
                "regime":      int(regime),
                "n_steps":     int(pair.sum()),
                "gt_speed":    float(g_spd.mean()),
                "policy_speed": float(p_spd.mean()),
                "d_speed":     float(p_spd.mean() - g_spd.mean()),
                "speed_ratio": float(p_spd.mean() / max(g_spd.mean(), 0.1)),
                "ade":         ade,
                "d_turn_rate": float(np.abs(p_dh).mean() - np.abs(g_dh).mean()),
                "d_jerk":      float(p_acc.std() - g_acc.std())
                               if len(p_acc) > 2 else np.nan,
                "same_turn":   same_turn,
                "event_rate":  float((rews[:t_end, a] <= EVENT_REW).sum()
                                     / t_end * T),
            }
            role_seg = rl[:t_end, a].mean(axis=0)
            for d in range(role_dim):
                row[f"role_{d}"] = float(role_seg[d])
            rows.append(row)

        print(f"[phaseC] episode {ep+1}/{args.episodes}: "
              f"{len(rows)} usable agent-episodes so far", flush=True)

    env.close()
    import pandas as pd
    return pd.DataFrame(rows), role_dim


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def corr_matrix(df, role_dim, metrics):
    """(role_dim, n_metrics) Pearson r on the given frame."""
    out = np.full((role_dim, len(metrics)), np.nan)
    for i in range(role_dim):
        r_col = df[f"role_{i}"].values
        for j, m in enumerate(metrics):
            v = df[m].values
            ok = np.isfinite(r_col) & np.isfinite(v)
            if ok.sum() > 30 and r_col[ok].std() > 0 and v[ok].std() > 0:
                out[i, j] = np.corrcoef(r_col[ok], v[ok])[0, 1]
    return out


def annotated_heatmap(ax, mat, row_labels, col_labels, title, vmax=0.5):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, "--" if not np.isfinite(v) else f"{v:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if np.isfinite(v) and abs(v) > vmax * 0.7
                    else "black")
    ax.set_title(title, fontsize=9)
    return im


def analyse(df, role_dim, args):
    import pandas as pd
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = dict(item.split(":") for item in args.regime_names.split(","))
    names = {int(k): v for k, v in names.items()}
    regimes = sorted(df["regime"].unique())
    dim_labels = [f"dim {d}" for d in range(role_dim)]

    df.to_csv(out / "phaseC_agent_data.csv", index=False)
    print(f"\n[phaseC] {len(df)} agent-episodes  "
          f"median ADE={df['ade'].median():.1f} m (sanity: should be ~1-20 m; "
          f"hundreds would mean GT misalignment)")
    for rg in regimes:
        sub = df[df["regime"] == rg]
        print(f"[phaseC]   regime {rg} ({names.get(rg, '?')}): {len(sub)} agents, "
              f"human {sub['gt_speed'].mean():.1f} m/s, "
              f"policy {sub['policy_speed'].mean():.1f} m/s, "
              f"d_speed {sub['d_speed'].mean():+.2f} m/s")

    # -- 1. per-regime correlation heatmaps ------------------------------------
    ncol = len(regimes)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 0.6 * role_dim + 2.2))
    axes = np.atleast_1d(axes)
    for ax, rg in zip(axes, regimes):
        sub = df[df["regime"] == rg]
        m   = corr_matrix(sub, role_dim, CORR_METRICS)
        annotated_heatmap(ax, m, dim_labels, CORR_METRICS,
                          f"{names.get(rg, rg)} (n={len(sub)})")
    fig.suptitle("Role dims vs GT-referenced deltas, per map regime "
                 "(Pearson r)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "phaseC_corr_by_regime.png", dpi=140)
    plt.close(fig)

    # -- 2. replication matrix: dim x regime, one panel per metric -------------
    fig, axes = plt.subplots(1, len(CORR_METRICS),
                             figsize=(3.3 * len(CORR_METRICS),
                                      0.6 * role_dim + 2.2))
    reg_labels = [names.get(rg, str(rg)) for rg in regimes]
    for ax, m in zip(np.atleast_1d(axes), CORR_METRICS):
        mat = np.full((role_dim, len(regimes)), np.nan)
        for j, rg in enumerate(regimes):
            mat[:, j] = corr_matrix(df[df["regime"] == rg],
                                    role_dim, [m])[:, 0]
        annotated_heatmap(ax, mat, dim_labels, reg_labels, m)
    fig.suptitle("REPLICATION TEST: same sign in every regime = genuine role "
                 "semantics; sign flips = scenario artifact", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "phaseC_replication.png", dpi=140)
    plt.close(fig)

    # -- 3. within-scene z-scored correlations (strongest control) -------------
    z = df.copy()
    grp = z.groupby("scenario_id")
    keep = grp["d_speed"].transform("count") >= 3
    z = df[keep].copy()
    cols = [f"role_{d}" for d in range(role_dim)] + CORR_METRICS
    g2 = z.groupby("scenario_id")[cols]
    z[cols] = (z[cols] - g2.transform("mean")) / g2.transform("std").replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(5.5, 0.6 * role_dim + 2.2))
    annotated_heatmap(ax, corr_matrix(z, role_dim, CORR_METRICS),
                      dim_labels, CORR_METRICS,
                      f"within-scene z-scored (scene-mates only, "
                      f"n={len(z)})", vmax=0.3)
    fig.tight_layout()
    fig.savefig(out / "phaseC_within_scene.png", dpi=140)
    plt.close(fig)

    # -- 4. variance decomposition: regime vs role vs residual -----------------
    role_cols = [f"role_{d}" for d in range(role_dim)]
    var_metrics = ["policy_speed", "d_speed", "ade", "d_turn_rate"]
    shares = np.zeros((len(var_metrics), 3))
    for i, m in enumerate(var_metrics):
        sub = df[np.isfinite(df[m])]
        y   = sub[m].values.astype(np.float64)
        ss_tot = ((y - y.mean()) ** 2).sum()
        # between-regime
        ss_b = sum(len(g) * (g[m].mean() - y.mean()) ** 2
                   for _, g in sub.groupby("regime"))
        # role R^2 within regimes (centered per regime, pooled OLS)
        ss_w, ssr = 0.0, 0.0
        for _, g in sub.groupby("regime"):
            yy = g[m].values - g[m].mean()
            X  = g[role_cols].values - g[role_cols].values.mean(axis=0)
            X  = np.column_stack([X, np.ones(len(X))])
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            resid = yy - X @ beta
            ss_w += (yy ** 2).sum()
            ssr  += (resid ** 2).sum()
        role_share = (ss_w - ssr) / ss_tot
        shares[i] = [ss_b / ss_tot, role_share,
                     1 - ss_b / ss_tot - role_share]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(var_metrics))
    bot = np.zeros(len(var_metrics))
    for k, (lab, col) in enumerate([("map regime", "#4477aa"),
                                    ("role (within regime)", "#ee6677"),
                                    ("residual", "#bbbbbb")]):
        ax.bar(x, shares[:, k], bottom=bot, label=lab, color=col)
        for i in range(len(var_metrics)):
            if shares[i, k] > 0.04:
                ax.text(i, bot[i] + shares[i, k] / 2,
                        f"{shares[i, k]:.0%}", ha="center", va="center",
                        fontsize=9)
        bot += shares[:, k]
    ax.set_xticks(x)
    ax.set_xticklabels(var_metrics)
    ax.set_ylabel("share of variance")
    ax.set_title("What explains behavioral variance: maps vs roles\n"
                 "(regime share should collapse on the GT-referenced deltas)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "phaseC_variance.png", dpi=140)
    plt.close(fig)
    for i, m in enumerate(var_metrics):
        print(f"[phaseC] variance {m:<12}: regime {shares[i,0]:.1%}  "
              f"role {shares[i,1]:.1%}  residual {shares[i,2]:.1%}")

    # -- 5. role clusters per regime: spider chart of mean delta profiles ------
    # Clusters are ORDERED by their d_speed so the same color/name means the
    # same personality in every regime: conformist (closest to the human),
    # middle, runaway (fastest over-speeder). Axes are min-max scaled across
    # the clusters of that regime; legend shows the real-unit d_speed.
    from sklearn.cluster import KMeans
    K = args.kmeans_k
    CLUSTER_NAMES  = (["conformist", "middle", "runaway"] if K == 3 else
                      [f"cluster {c}" for c in range(K)])
    CLUSTER_COLORS = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    prof_metrics = ["d_speed", "ade", "d_turn_rate", "d_jerk", "event_rate"]
    prof_labels  = ["Δspeed", "path dev\n(ADE)", "Δturn", "Δjerk", "events"]
    n_m    = len(prof_metrics)
    angles = np.linspace(0, 2 * np.pi, n_m, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(regimes),
                             figsize=(4.4 * len(regimes), 4.8),
                             subplot_kw=dict(polar=True))
    for ax, rg in zip(np.atleast_1d(axes), regimes):
        sub = df[df["regime"] == rg].dropna(subset=prof_metrics)
        if len(sub) < 50:
            ax.set_visible(False)
            continue
        km  = KMeans(n_clusters=K, n_init=10, random_state=42)
        lab = km.fit_predict(sub[role_cols].values)
        means = np.array([[sub[m][lab == c].mean() for m in prof_metrics]
                          for c in range(K)])              # (K, n_m)
        order = np.argsort(means[:, 0])                    # by d_speed
        lo, hi = means.min(axis=0), means.max(axis=0)
        rng    = np.where(hi - lo == 0, 1, hi - lo)
        normed = (means - lo) / rng

        for rank, c in enumerate(order):
            vals = normed[c].tolist() + [normed[c][0]]
            ax.plot(angles, vals, "o-", lw=2, color=CLUSTER_COLORS[rank],
                    label=f"{CLUSTER_NAMES[rank]} (n={(lab == c).sum()}, "
                          f"Δv={means[c, 0]:+.1f} m/s)")
            ax.fill(angles, vals, alpha=0.10, color=CLUSTER_COLORS[rank])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(prof_labels, fontsize=8)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([], fontsize=6)
        ax.set_title(f"{names.get(rg, rg)}  (n={len(sub)})",
                     fontsize=10, pad=16)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=7)
    fig.suptitle("Role clusters per regime -- mean deviation from the human "
                 "counterpart (axes min-max scaled per regime)",
                 fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(out / "phaseC_role_clusters.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # -- 6. role-axis dose-response: PC1 quintiles vs deltas in real units -----
    # The most readable summary: one global role axis (conformist -> runaway
    # pole), agents binned into quintiles, mean deviation from the human
    # counterpart per bin and regime. Monotone lines in every regime = the
    # role axis means the same thing everywhere.
    MIN_CELL = 40
    roles = df[role_cols].values
    cen   = roles - roles.mean(axis=0)
    pc1   = np.linalg.svd(cen, full_matrices=False)[2][0]
    axis  = cen @ pc1
    ok    = np.isfinite(df["d_speed"].values)
    if np.corrcoef(axis[ok], df["d_speed"].values[ok])[0, 1] < 0:
        axis = -axis                       # orient: + = over-speeding pole
    NQ    = 5
    edges = np.quantile(axis, np.linspace(0, 1, NQ + 1))
    qbin  = np.clip(np.searchsorted(edges, axis, side="right") - 1, 0, NQ - 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (met, lab) in zip(axs, [("d_speed", "faster than the human (m/s)"),
                                    ("ade", "distance from human's path (m)")]):
        for rg in regimes:
            m = (df["regime"].values == rg) & np.isfinite(df[met].values)
            ys, es, xs_ = [], [], []
            for q in range(NQ):
                v = df[met].values[m & (qbin == q)]
                if len(v) < MIN_CELL:
                    continue
                xs_.append(q)
                ys.append(v.mean())
                es.append(1.96 * v.std() / len(v) ** 0.5)
            ax.errorbar(xs_, ys, yerr=es, marker="o", capsize=3,
                        label=f"{names.get(rg, rg)}")
        ax.set_xticks(range(NQ))
        ax.set_xticklabels(["conformist\npole", "", "middle", "", "runaway\npole"],
                           fontsize=8)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Role axis (PC1) quintiles vs mean deviation from the human "
                 "counterpart", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "phaseC_role_axis.png", dpi=140)
    plt.close(fig)

    print(f"\n[phaseC] outputs -> {out}/phaseC_*.png + phaseC_agent_data.csv")


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    if args.replot:
        import pandas as pd
        df = pd.read_csv(Path(args.out_dir) / "phaseC_agent_data.csv")
        role_dim = sum(c.startswith("role_") for c in df.columns)
        print(f"[phaseC] replot from CSV: {len(df)} rows, role_dim={role_dim}")
    else:
        if not (args.checkpoint and args.clusters and args.data_dir):
            raise SystemExit("--checkpoint, --clusters and --data_dir are "
                             "required unless --replot is given")
        df, role_dim = collect(args, device)
        if len(df) < 500:
            print(f"[phaseC] only {len(df)} usable agents -- increase "
                  f"--episodes or relax --min_margin")
    analyse(df, role_dim, args)


if __name__ == "__main__":
    main()
