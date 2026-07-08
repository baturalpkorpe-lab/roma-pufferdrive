"""
role_direct_cluster.py -- role-variable clustering, characterized by BOTH raw
kinematics and GT-referenced deltas.

Complements role_regime_analysis.py (Phase C). Phase C relates each role DIM to
deltas; this script instead CLUSTERS agents in role-variable (z) space -- K-means
on the per-agent mean role_mean -- separately WITHIN each map regime, then reports
each cluster's mean over a rich feature set so you can judge which family of
features (raw ego kinematics vs policy-minus-human deltas) best characterizes the
learned role clusters.

Why this design (see session discussion):
  - The role z is the object of study (what we control / the novelty), so we
    ANCHOR the clustering on z, not on behavior.
  - We then report raw AND delta features per cluster. The delta = policy - own
    GT is a per-agent SITUATION control (the human faced the same road/turn/goal),
    finer than the scene-level map-regime stratification.
  - eta^2 per feature = share of that feature's within-regime variance explained
    by the z-cluster label. High eta^2 = the role clusters separate strongly on
    that feature = it is a good descriptor of the role. Comparing raw-vs-delta
    eta^2 answers "which approach would be better to classify".

Feature families (per usable agent-episode, pre-respawn, GT-valid overlap only):
  RAW (ego only, no GT):   speed_mean speed_max speed_std accel_abs accel_std
                           jerk_abs turn_abs event_rate
  DELTA (vs own human GT):  d_speed speed_ratio ade d_turn_rate d_jerk
  CONTEXT:                  gt_speed

Usage (from /scratch/e452103/PufferDrive so the C binding finds drive.ini):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/role_direct_cluster.py \
        --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_final.pt \
        --clusters   /scratch/e452103/map_atlas/k4/map_clusters.csv \
        --data_dir   pufferlib/resources/drive/binaries/training \
        --out_dir    /scratch/e452103/role_direct/dim4
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
TELEPORT_M   = 8.0    # per-step jump above this = respawn -> cut segment
MIN_STEPS    = 10     # min overlapping valid steps for usable metrics
GT_MOVE_MS   = 1.0    # human counterpart must actually drive
EVENT_REW    = -0.4   # reward <= this counts as a safety penalty event

# Feature families. Raw = the policy's own kinematics (no GT). Delta = deviation
# from the human who occupied the same seat. Context = the human baseline itself.
RAW_FEATS   = ["speed_mean", "speed_max", "speed_std", "accel_abs", "accel_std",
               "jerk_abs", "turn_abs", "event_rate"]
DELTA_FEATS = ["d_speed", "speed_ratio", "ade", "d_turn_rate", "d_jerk"]
CONTEXT     = ["gt_speed"]
REPORT_FEATS = RAW_FEATS + DELTA_FEATS + CONTEXT


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
                   help="map_clusters.csv from map_atlas.py (the K=4 atlas)")
    p.add_argument("--data_dir",   type=str, default=None)
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--replot",     action="store_true",
                   help="Skip rollout: rebuild figures/tables from "
                        "<out_dir>/role_direct_agent_data.csv (no GPU)")
    p.add_argument("--episodes",   type=int, default=20,
                   help="Reset batches; each covers ~500 scenarios")
    p.add_argument("--num_agents", type=int, default=3072)
    p.add_argument("--num_maps",   type=int, default=10000)
    p.add_argument("--min_margin", type=float, default=1.5,
                   help="Core-map filter: keep maps clearly inside one regime")
    p.add_argument("--exclude_clusters", type=str, default="",
                   help="Comma-separated regime ids to drop. Empty by default: "
                        "the new atlas already removes junk maps before "
                        "clustering, so there is no junk regime to exclude.")
    p.add_argument("--regime_names", type=str, default="",
                   help="Optional 'id:name,id:name' labels (from atlas_names.txt)")
    p.add_argument("--kmeans_k",   type=int, default=3,
                   help="Role clusters per regime")
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--seed",       type=int, default=42)
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
    print(f"[direct] {len(regime_of)} core scenarios across regimes "
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
    print(f"[direct] policy loaded: role_dim={role_dim} obs_dim={obs_dim}")

    rows = []
    for ep in range(args.episodes):
        if ep > 0:
            env.resample_maps()
        obs_np, _ = env.reset()

        gt       = env.get_ground_truth_trajectories()
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

        step_d = np.sqrt(np.diff(xs, axis=0)**2 + np.diff(ys, axis=0)**2)
        for a in range(B):
            regime = regime_of.get(sid[a])
            if regime is None or not is_veh[a]:
                continue

            jumps = np.where(step_d[:, a] > TELEPORT_M)[0]
            t_end = int(jumps[0] + 1) if len(jumps) else T
            t_end = min(t_end, T_gt)
            if t_end < MIN_STEPS + 1:
                continue

            v = gvalid[a, :t_end]
            pair = v[:-1] & v[1:]
            if pair.sum() < MIN_STEPS:
                continue

            # human counterpart, same seat, same valid timesteps (m/s, m/s^2)
            g_dx  = (gx[a, 1:t_end] - gx[a, :t_end-1])[pair]
            g_dy  = (gy[a, 1:t_end] - gy[a, :t_end-1])[pair]
            g_spd = np.hypot(g_dx, g_dy) * 10.0
            if g_spd.max() <= GT_MOVE_MS:
                continue
            g_dh  = wrap_angle((gh[a, 1:t_end] - gh[a, :t_end-1])[pair]) * 10.0
            g_acc = np.diff(g_spd) * 10.0

            # policy agent over the same timesteps
            p_dx  = np.diff(xs[:t_end, a])[pair]
            p_dy  = np.diff(ys[:t_end, a])[pair]
            p_spd = np.hypot(p_dx, p_dy) * 10.0
            p_dh  = wrap_angle(np.diff(hs[:t_end, a])[pair]) * 10.0
            p_acc = np.diff(p_spd) * 10.0                # m/s^2
            p_jrk = np.diff(p_acc) * 10.0                # m/s^3

            vi  = np.where(v)[0]
            ade = float(np.mean(np.hypot(xs[vi, a] - gx[a, vi],
                                         ys[vi, a] - gy[a, vi])))

            row = {
                "scenario_id": sid[a],
                "regime":      int(regime),
                "n_steps":     int(pair.sum()),
                # --- RAW ego kinematics (no GT) ---
                "speed_mean":  float(p_spd.mean()),
                "speed_max":   float(p_spd.max()),
                "speed_std":   float(p_spd.std()),
                "accel_abs":   float(np.abs(p_acc).mean()) if len(p_acc) else np.nan,
                "accel_std":   float(p_acc.std())          if len(p_acc) else np.nan,
                "jerk_abs":    float(np.abs(p_jrk).mean()) if len(p_jrk) else np.nan,
                "turn_abs":    float(np.abs(p_dh).mean()),
                "event_rate":  float((rews[:t_end, a] <= EVENT_REW).sum()
                                     / t_end * T),
                # --- DELTA vs own human GT ---
                "gt_speed":    float(g_spd.mean()),
                "d_speed":     float(p_spd.mean() - g_spd.mean()),
                "speed_ratio": float(p_spd.mean() / max(g_spd.mean(), 0.1)),
                "ade":         ade,
                "d_turn_rate": float(np.abs(p_dh).mean() - np.abs(g_dh).mean()),
                "d_jerk":      float(p_acc.std() - g_acc.std())
                               if len(p_acc) > 2 and len(g_acc) > 2 else np.nan,
            }
            role_seg = rl[:t_end, a].mean(axis=0)
            for d in range(role_dim):
                row[f"role_{d}"] = float(role_seg[d])
            rows.append(row)

        print(f"[direct] episode {ep+1}/{args.episodes}: "
              f"{len(rows)} usable agent-episodes so far", flush=True)

    env.close()
    import pandas as pd
    return pd.DataFrame(rows), role_dim


# ---------------------------------------------------------------------------
# Cluster + profile
# ---------------------------------------------------------------------------

def eta_sq(values, labels):
    """Share of variance in `values` explained by the cluster `labels`."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    v, lab = v[ok], np.asarray(labels)[ok]
    if len(v) < 4 or v.std() == 0:
        return np.nan
    grand = v.mean()
    ss_tot = ((v - grand) ** 2).sum()
    ss_b = 0.0
    for c in np.unique(lab):
        vc = v[lab == c]
        ss_b += len(vc) * (vc.mean() - grand) ** 2
    return float(ss_b / ss_tot) if ss_tot > 0 else np.nan


def analyse(df, role_dim, args):
    import pandas as pd
    from sklearn.cluster import KMeans

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = {}
    if args.regime_names:
        names = {int(k): v for k, v in
                 (item.split(":") for item in args.regime_names.split(","))}

    role_cols = [f"role_{d}" for d in range(role_dim)]
    regimes   = sorted(df["regime"].unique())
    K         = args.kmeans_k
    cnames    = (["conformist", "middle", "runaway"] if K == 3 else
                 [f"cluster {c}" for c in range(K)])

    df.to_csv(out / "role_direct_agent_data.csv", index=False)
    print(f"\n[direct] {len(df)} agent-episodes  "
          f"median ADE={df['ade'].median():.1f} m (sanity ~1-20 m)")

    # -- Cluster in z-space WITHIN each regime; order clusters by d_speed -------
    df["cluster_rank"] = -1
    profile_rows, eta_rows = [], []
    for rg in regimes:
        idx = df.index[df["regime"] == rg]
        sub = df.loc[idx]
        if len(sub) < max(50, K * 10):
            print(f"[direct] regime {rg}: only {len(sub)} agents -- skipped")
            continue
        km  = KMeans(n_clusters=K, n_init=10, random_state=args.seed)
        lab = km.fit_predict(sub[role_cols].values)

        # order raw cluster ids by mean d_speed so rank 0 = conformist pole
        dsp   = [np.nanmean(sub["d_speed"].values[lab == c]) for c in range(K)]
        order = np.argsort(dsp)                      # raw id in ascending d_speed
        raw_to_rank = {c: r for r, c in enumerate(order)}
        ranks = np.array([raw_to_rank[c] for c in lab])
        df.loc[idx, "cluster_rank"] = ranks

        for r in range(K):
            m = ranks == r
            prof = {"regime": rg, "regime_name": names.get(rg, f"regime {rg}"),
                    "cluster_rank": r, "cluster_name": cnames[r],
                    "n_agents": int(m.sum())}
            for f in REPORT_FEATS:
                prof[f] = float(np.nanmean(sub[f].values[m]))
            profile_rows.append(prof)

        for f in REPORT_FEATS:
            eta_rows.append({"regime": rg,
                             "regime_name": names.get(rg, f"regime {rg}"),
                             "feature": f, "eta_sq": eta_sq(sub[f].values, ranks)})

    prof_df = pd.DataFrame(profile_rows)
    eta_df  = pd.DataFrame(eta_rows)
    prof_df.to_csv(out / "role_direct_profiles.csv", index=False)
    eta_df.to_csv(out / "role_direct_separability.csv", index=False)
    df.to_csv(out / "role_direct_agent_data.csv", index=False)   # w/ cluster_rank

    # -- Stdout summary (readable in the .out log) -----------------------------
    for rg in regimes:
        p = prof_df[prof_df["regime"] == rg]
        if p.empty:
            continue
        print(f"\n[direct] === regime {rg} ({names.get(rg, rg)}) ===")
        hdr = "  ".join(f"{f:>11}" for f in ["cluster", "n", *REPORT_FEATS])
        print("  " + hdr)
        for _, r in p.iterrows():
            vals = "  ".join(f"{r[f]:>11.2f}" for f in REPORT_FEATS)
            print(f"  {r['cluster_name']:>11} {int(r['n_agents']):>11}  {vals}")

    _plot_profiles(prof_df, regimes, names, out, K)
    _plot_separability(eta_df, regimes, names, out)
    print(f"\n[direct] outputs -> {out}/role_direct_*.png + *.csv")


def _plot_profiles(prof_df, regimes, names, out, K):
    """Per-regime heatmap: cluster x feature, color = within-feature z-score
    across the K clusters (so separation is visible), text = raw mean."""
    regimes = [rg for rg in regimes if not prof_df[prof_df["regime"] == rg].empty]
    if not regimes:
        return
    n_raw = len(RAW_FEATS)
    fig, axes = plt.subplots(len(regimes), 1,
                             figsize=(1.05 * len(REPORT_FEATS) + 1,
                                      1.0 * K * len(regimes) + 1.5))
    axes = np.atleast_1d(axes)
    for ax, rg in zip(axes, regimes):
        p   = prof_df[prof_df["regime"] == rg].sort_values("cluster_rank")
        raw = p[REPORT_FEATS].values.astype(float)              # (K, F)
        mu  = np.nanmean(raw, axis=0)
        sd  = np.nanstd(raw, axis=0)
        sd[sd == 0] = 1
        z   = (raw - mu) / sd
        im  = ax.imshow(z, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(REPORT_FEATS)))
        ax.set_xticklabels(REPORT_FEATS, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(p)))
        ax.set_yticklabels([f"{r.cluster_name}\n(n={int(r.n_agents)})"
                            for r in p.itertuples()], fontsize=7)
        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                ax.text(j, i, f"{raw[i, j]:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(z[i, j]) > 1 else "black")
        ax.axvline(n_raw - 0.5, color="k", lw=1.5)              # raw | delta
        ax.axvline(n_raw + len(DELTA_FEATS) - 0.5, color="grey", lw=1, ls=":")
        ax.set_title(f"regime {rg}: {names.get(rg, rg)}  "
                     f"(z-clusters ordered by d_speed; RAW | DELTA | context)",
                     fontsize=8)
    fig.suptitle("Role-variable clusters (K-means on z, per regime) — mean of "
                 "raw kinematics vs GT-deltas", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "role_direct_profiles.png", dpi=140)
    plt.close(fig)


def _plot_separability(eta_df, regimes, names, out):
    """feature x regime heatmap of eta^2 = how strongly the z-clusters separate
    each feature. Raw vs delta blocks compared side by side."""
    regimes = [rg for rg in regimes if rg in set(eta_df["regime"])]
    feats = RAW_FEATS + DELTA_FEATS + CONTEXT
    mat = np.full((len(feats), len(regimes)), np.nan)
    for j, rg in enumerate(regimes):
        sub = eta_df[eta_df["regime"] == rg].set_index("feature")["eta_sq"]
        for i, f in enumerate(feats):
            if f in sub.index:
                mat[i, j] = sub[f]
    fig, ax = plt.subplots(figsize=(1.4 * len(regimes) + 2, 0.45 * len(feats) + 2))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=np.nanmax(mat) or 1,
                   aspect="auto")
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([names.get(rg, str(rg)) for rg in regimes],
                       rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    for i in range(len(feats)):
        for j in range(len(regimes)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if mat[i, j] < 0.5 * (np.nanmax(mat) or 1)
                        else "black")
    ax.axhline(len(RAW_FEATS) - 0.5, color="w", lw=2)          # raw | delta
    ax.axhline(len(RAW_FEATS) + len(DELTA_FEATS) - 0.5, color="w", lw=1, ls=":")
    plt.colorbar(im, ax=ax, label="eta^2 (variance explained by z-cluster)")
    ax.set_title("Which features do the role clusters separate?\n"
                 "top block = RAW kinematics, middle = GT-deltas", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "role_direct_separability.png", dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    if args.replot:
        import pandas as pd
        df = pd.read_csv(Path(args.out_dir) / "role_direct_agent_data.csv")
        role_dim = sum(c.startswith("role_") for c in df.columns)
        print(f"[direct] replot from CSV: {len(df)} rows, role_dim={role_dim}")
    else:
        if not (args.checkpoint and args.clusters and args.data_dir):
            raise SystemExit("--checkpoint, --clusters and --data_dir are "
                             "required unless --replot is given")
        df, role_dim = collect(args, device)
        if len(df) < 500:
            print(f"[direct] only {len(df)} usable agents -- increase "
                  f"--episodes or relax --min_margin")
    analyse(df, role_dim, args)


if __name__ == "__main__":
    main()
