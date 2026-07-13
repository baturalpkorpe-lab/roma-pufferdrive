"""
role_paired_sweep.py -- PAIRED forced-role dose-response along PC1/PC2.

THE design requirement: compare alpha=-2 vs alpha=+2 ON THE SAME MAP with the
SAME focal vehicle, so map noise cannot fake a role effect. Every reported
number is a WITHIN-(map,vehicle) paired difference.

How the pairing works:
  - Per episode, resample_maps() is called ONCE. The same map pool then serves
    every condition (each direction x alpha, plus a natural baseline).
  - env.reset() reshuffles the slot<->scenario allocation, so scenes/slots move
    between conditions. Observations are therefore keyed by
    (episode, scenario_id, focal_vehicle_id) and joined ACROSS conditions at
    analysis time. The focal is picked by a deterministic rule (the vehicle
    whose human GT drives farthest in that scene), so the same scene always
    yields the same focal vehicle in every condition.
  - Only (map, vehicle) pairs present in BOTH conditions of a comparison enter
    that comparison. Everything is a paired delta:
        delta_metric(alpha) = metric(alpha) - metric(alpha=0)   [same map+vehicle]

Directions: PC1 and PC2 of the natural role distribution ONLY (no
behavior-fitted directions -- those were confound-chasing and causally dead).

FORCING SCHEME (redesigned): the focal is NOT clamped to a constant vector.
Every step it keeps its OWN live encoder role (which varies over time and
between agents) and a constant SHIFT of alpha*sigma_d*PC_d is ADDED to it:
    z_focal(t) = z_natural(t) + alpha * sigma_d * PC_d
So alpha=0 is exactly the natural policy (the shared baseline condition), and
the dose-response measures "push this agent's own role along the axis",
preserving individuality instead of replacing it with a shifted average.

Outputs in --out_dir:
  role_paired_agent.csv    one row per (episode, condition, map, vehicle)
  role_paired_deltas.csv   paired deltas vs alpha=0 per (direction, alpha, regime)
  role_paired_tests.csv    the headline: alpha=+2 vs -2 SAME-MAP paired diff,
                           mean, sem, t-stat, n per direction x metric
  role_paired_PC{1,2}.png  dose-response of PAIRED deltas (0 at alpha=0 by
                           construction; error bars = paired SEM)

Usage (from /scratch/e452103/PufferDrive):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/role_paired_sweep.py \
        --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_final.pt \
        --clusters   /scratch/e452103/map_atlas/k4/map_clusters.csv \
        --data_dir   pufferlib/resources/drive/binaries/training \
        --out_dir    /scratch/e452103/role_paired/dim4
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import load_policy
from traj_kinematics import ego_kinematics

T = 91
TELEPORT_M = 4.0     # respawn discontinuity cut (validated by diag_speed_spikes)
MIN_STEPS  = 10
GT_MOVE_MS = 1.0
EVENT_REW  = -0.4
ROLLOUT_SEED = 1234

METRICS = ["speed_mean", "accel_abs", "jerk_abs", "turn_abs", "event_rate"]
METRIC_LABEL = {"speed_mean": "speed (m/s)", "accel_abs": "|accel| (m/s2)",
                "jerk_abs": "|jerk| (m/s3)", "turn_abs": "|turn| (rad/s)",
                "event_rate": "safety events / 91"}


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


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _squeeze(a):
    a = np.asarray(a)
    return a[:, 0] if a.ndim >= 2 and a.shape[1] == 1 else a


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--clusters",   type=str, required=True)
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--warmup_episodes", type=int, default=3)
    p.add_argument("--episodes",   type=int, default=6,
                   help="Map-pool draws; ALL conditions run on each draw")
    p.add_argument("--num_agents", type=int, default=3072)
    p.add_argument("--num_maps",   type=int, default=10000)
    p.add_argument("--min_margin", type=float, default=1.5)
    p.add_argument("--exclude_clusters", type=str, default="")
    p.add_argument("--regime_names", type=str, default="")
    p.add_argument("--n_axes",     type=int, default=2, help="PC1..PCn to sweep")
    p.add_argument("--alphas",     type=str, default="-2,-1,0,1,2")
    p.add_argument("--observe_only", action="store_true",
                   help="Stop after the warmup: writes the observational "
                        "PC-vs-behavior scatter (plausibility-masked accel/"
                        "jerk), warmup CSV and axes CSV. Cheap; no sweep.")
    p.add_argument("--device",     type=str, default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-focal metrics (direct ego kinematics; GT only as a validity filter)
# ---------------------------------------------------------------------------

def focal_metrics(a, xs, ys, hs, rews, gx, gy, gvalid, T_gt):
    step_d = np.hypot(np.diff(xs[:, a]), np.diff(ys[:, a]))
    jumps  = np.where(step_d > TELEPORT_M)[0]
    t_end  = min(int(jumps[0] + 1) if len(jumps) else T, T_gt)
    if t_end < MIN_STEPS + 1:
        return None
    v = gvalid[a, :t_end]
    pair = v[:-1] & v[1:]
    if pair.sum() < MIN_STEPS:
        return None
    g_dx = (gx[a, 1:t_end] - gx[a, :t_end-1])[pair]
    g_dy = (gy[a, 1:t_end] - gy[a, :t_end-1])[pair]
    if (np.hypot(g_dx, g_dy) * 10.0).max() <= GT_MOVE_MS:
        return None
    p_dx = np.diff(xs[:t_end, a])[pair]
    p_dy = np.diff(ys[:t_end, a])[pair]
    p_spd = np.hypot(p_dx, p_dy) * 10.0
    p_dh  = wrap_angle(np.diff(hs[:t_end, a])[pair]) * 10.0
    k = ego_kinematics(p_spd, p_dh)               # plausibility-masked kinematics
    if k["n_steps"] < MIN_STEPS:
        return None
    return {
        "speed_mean": k["speed_mean"],
        "accel_abs":  k["accel_abs"],
        "jerk_abs":   k["jerk_abs"],
        "turn_abs":   k["turn_abs"],
        "event_rate": float((rews[:t_end, a] <= EVENT_REW).sum() / t_end * T),
    }


def select_focals(gt, regime_of):
    """Deterministic focal per core scene = vehicle whose human GT drives the
    farthest. Same scene -> same focal vehicle id, in every condition.
    Returns list of (slot, scenario_id, vehicle_id, regime)."""
    sids   = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
    ids    = _squeeze(np.asarray(gt["id"])).reshape(-1)
    gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
    valid  = _squeeze(gt["valid"]).astype(bool)
    is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
    out = []
    for sid in np.unique(sids):
        rg = regime_of.get(sid)
        if rg is None or not sid or str(sid).lower().startswith("map"):
            continue
        cand = np.where(sids == sid)[0]
        best, best_len = -1, 0.0
        for a in cand:
            if not is_veh[a] or valid[a].sum() < MIN_STEPS:
                continue
            px, py = gx[a][valid[a]], gy[a][valid[a]]
            L = np.hypot(np.diff(px), np.diff(py)).sum()
            if L > best_len:
                best, best_len = a, L
        if best >= 0 and best_len > 2.0:
            out.append((int(best), str(sid), int(ids[best]), int(rg)))
    return out


def rollout_condition(env, policy, device, regime_of, shift_vec):
    """reset -> pick focals (deterministic per scene) -> roll 91 steps ADDING
    shift_vec to every focal's OWN live role each step (two-pass; the focal's
    natural role still varies per step/agent -- only the offset is constant).
    shift_vec None = natural, nobody touched.
    Returns rows keyed by (scenario_id, vehicle_id)."""
    B = env.num_agents
    obs_np, _ = env.reset()
    gt = env.get_ground_truth_trajectories()
    focals = select_focals(gt, regime_of)
    slots  = np.array([f[0] for f in focals], dtype=int)

    fv = (None if shift_vec is None else
          torch.as_tensor(shift_vec, dtype=torch.float32, device=device))
    torch.manual_seed(ROLLOUT_SEED)
    xs = np.zeros((T, B), np.float32); ys = np.zeros((T, B), np.float32)
    hs = np.zeros((T, B), np.float32); rews = np.zeros((T, B), np.float32)
    rl = np.zeros((T, B, policy.role_dim), np.float32)

    obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    state = policy.initial_state(B, device)
    for t in range(T):
        ag = env.get_global_agent_state()
        xs[t], ys[t], hs[t] = ag["x"], ag["y"], ag["heading"]
        with torch.no_grad():
            if fv is None or len(slots) == 0:
                logits, _, state, ri = policy(obs, state)
            else:
                _, _, _, ri0 = policy(obs, state)      # pass 1: natural roles
                forced = ri0["role_z"].clone()
                forced[slots] = forced[slots] + fv      # SHIFT own role only
                logits, _, state, ri = policy(obs, state, forced_role=forced)
        if ri.get("role_mean") is not None and ri["role_mean"].shape[-1]:
            rl[t] = ri["role_mean"].float().cpu().numpy()
        action = Categorical(logits=logits.float()).sample()
        obs_np, rew_np, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
        rews[t] = np.asarray(rew_np).reshape(B)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

    gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
    gvalid = _squeeze(gt["valid"]).astype(bool)
    T_gt   = gx.shape[1]
    rows = []
    for slot, sid, vid, rg in focals:
        m = focal_metrics(slot, xs, ys, hs, rews, gx, gy, gvalid, T_gt)
        if m is None:
            continue
        m.update({"sid": sid, "vid": vid, "regime": rg})
        rows.append(m)
    return rows, rl, xs, ys, hs, rews, gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    names = {}
    if args.regime_names:
        names = {int(k): v for k, v in
                 (it.split(":") for it in args.regime_names.split(","))}

    cl = pd.read_csv(args.clusters)
    excl = {int(v) for v in args.exclude_clusters.split(",") if v != ""}
    cl = cl[(cl["margin"] > args.min_margin) & (~cl["cluster"].isin(excl))]
    regime_of = dict(zip(cl["scenario_id"].astype(str), cl["cluster"]))
    print(f"[paired] {len(regime_of)} core scenarios, regimes "
          f"{sorted(cl['cluster'].unique())}")

    from pufferlib.ocean.drive.drive import Drive
    env_cfg = dict(load_drive_config()["env"])
    env_cfg.update({"num_maps": args.num_maps, "num_agents": args.num_agents,
                    "map_dir": args.data_dir})
    env = Drive(**env_cfg)
    obs_np, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs_np.shape[-1], device)
    if role_dim == 0:
        raise SystemExit("role_dim=0 checkpoint -- nothing to sweep")
    n_axes = min(args.n_axes, role_dim)

    # -- Warmup: natural roles + behavior (ALL vehicle agents, plausibility-
    #    masked) -> mu, PC axes, per-axis sigma + the observational pre-check --
    role_vecs, beh_rows = [], []
    for ep in range(args.warmup_episodes):
        if ep > 0:
            env.resample_maps()
        rows, rl, xs, ys, hs, rews, gt = rollout_condition(
            env, policy, device, regime_of, None)
        gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
        gvalid = _squeeze(gt["valid"]).astype(bool)
        is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
        T_gt   = gx.shape[1]
        step_d = np.hypot(np.diff(xs, axis=0), np.diff(ys, axis=0))
        for a in range(env.num_agents):
            if not is_veh[a]:
                continue
            m = focal_metrics(a, xs, ys, hs, rews, gx, gy, gvalid, T_gt)
            if m is None:
                continue
            jumps = np.where(step_d[:, a] > TELEPORT_M)[0]
            t_end = min(int(jumps[0] + 1) if len(jumps) else T, T_gt)
            role_vecs.append(rl[:t_end, a].mean(axis=0))
            beh_rows.append(m)
        print(f"[paired] warmup {ep+1}/{args.warmup_episodes}: "
              f"{len(role_vecs)} (role, behavior) pairs", flush=True)

    R = np.asarray(role_vecs)
    mu = R.mean(axis=0)
    cen = R - mu
    _, svals, vt = np.linalg.svd(cen, full_matrices=False)
    evr = (svals**2) / (svals**2).sum()
    axes_u  = [vt[d] for d in range(n_axes)]
    axes_sg = [float((cen @ vt[d]).std()) for d in range(n_axes)]
    print(f"[paired] PCA: explained var "
          f"{['%.0f%%' % (100*e) for e in evr[:n_axes]]}  "
          f"sigma={np.round(axes_sg, 3)}")

    # Persist mu + axes so render_role_alpha.py forces the SAME directions.
    ax_rows = [{"name": "mu", "sigma": 0.0,
                **{f"c{i}": float(mu[i]) for i in range(role_dim)}}]
    for d in range(n_axes):
        ax_rows.append({"name": f"PC{d+1}", "sigma": axes_sg[d],
                        **{f"c{i}": float(axes_u[d][i])
                           for i in range(role_dim)}})
    pd.DataFrame(ax_rows).to_csv(out / "role_paired_axes.csv", index=False)

    # -- Observational pre-check (plausibility-masked accel/jerk!) -------------
    # Scatter of PC-projection vs each behavior over the natural warmup agents.
    # Scene-confounded (NOT causal) -- it names candidate axes; the sweep tests
    # them. Replaces the old role_axis_sweep observational figure, whose
    # accel/jerk carried the impossible respawn artifacts.
    Bdf = pd.DataFrame(beh_rows)
    wdf = Bdf.copy()
    for i in range(role_dim):               # raw role vectors (role_space_map)
        wdf[f"role_{i}"] = R[:, i]
    for d in range(n_axes):                  # + their PC projections
        wdf[f"pc{d+1}"] = cen @ axes_u[d]
    wdf.to_csv(out / "role_paired_warmup.csv", index=False)

    fig, axs2 = plt.subplots(n_axes, len(METRICS),
                             figsize=(3.0 * len(METRICS), 2.8 * n_axes),
                             squeeze=False)
    print("\n[paired] === observational r (natural, plausibility-masked) ===")
    for d in range(n_axes):
        proj = cen @ axes_u[d]
        for j, met in enumerate(METRICS):
            ax = axs2[d][j]
            y  = Bdf[met].values.astype(float)
            ok = np.isfinite(proj) & np.isfinite(y)
            r  = (np.corrcoef(proj[ok], y[ok])[0, 1]
                  if ok.sum() > 10 and proj[ok].std() > 0 and y[ok].std() > 0
                  else np.nan)
            ax.scatter(proj[ok], y[ok], s=3, alpha=0.12, color="#4477aa",
                       rasterized=True)
            if np.isfinite(r):
                b  = np.polyfit(proj[ok], y[ok], 1)
                xx = np.array([proj[ok].min(), proj[ok].max()])
                ax.plot(xx, b[0] * xx + b[1], "r-", lw=1.5)
            ax.set_title(f"PC{d+1} vs {met}\nr={r:+.2f}", fontsize=8)
            ax.tick_params(labelsize=6)
            print(f"[paired]   PC{d+1} vs {met:<11}: r={r:+.3f}")
    fig.suptitle("Observational: role PCs vs behavior (natural rollouts, "
                 "plausibility-masked; scene-confounded -- the causal test is "
                 "the paired sweep)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "role_paired_observational.png", dpi=140)
    plt.close(fig)

    if args.observe_only:
        env.close()
        print(f"\n[paired] observe_only: wrote role_paired_observational.png, "
              f"role_paired_warmup.csv, role_paired_axes.csv -> {out}")
        return

    alphas = sorted(float(x) for x in args.alphas.split(","))

    # Conditions: one shared natural baseline (= alpha 0 for every axis; the
    # focal keeps its own role untouched) + each (PCd, alpha!=0) as a SHIFT
    # of the focal's own live role along that axis.
    conds = [("natural", None)]
    for d in range(n_axes):
        for al in alphas:
            if al == 0.0:
                continue
            conds.append((f"PC{d+1}|{al:+g}", al * axes_sg[d] * axes_u[d]))
    print(f"[paired] {len(conds)} conditions per episode x "
          f"{args.episodes} episodes (natural-offset forcing)")

    # -- Paired sweep: SAME map pool serves every condition --------------------
    all_rows = []
    for ep in range(args.episodes):
        env.resample_maps()                    # once per episode -- pool fixed
        for cname, cvec in conds:
            rows, *_ = rollout_condition(env, policy, device, regime_of, cvec)
            for r in rows:
                r.update({"episode": ep, "cond": cname})
            all_rows.extend(rows)
            print(f"[paired] ep{ep} {cname:>9}: {len(rows)} focals", flush=True)
    env.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "role_paired_agent.csv", index=False)
    if df.empty:
        raise SystemExit("[paired] no observations")

    # -- Analysis: everything is a within-(episode,map,vehicle) paired delta --
    key = ["episode", "sid", "vid"]
    base = df[df["cond"] == "natural"].set_index(key)
    regimes = sorted(df["regime"].unique())

    delta_rows, test_rows = [], []
    for d in range(n_axes):
        pcname = f"PC{d+1}"
        fig, axs = plt.subplots(1, len(METRICS),
                                figsize=(3.1 * len(METRICS), 3.8))
        cond_of = {al: (f"{pcname}|{al:+g}" if al != 0.0 else "natural")
                   for al in alphas}
        for ax, met in zip(np.atleast_1d(axs), METRICS):
            xs_o, ys_o = [], []
            for al in alphas:
                sub = df[df["cond"] == cond_of[al]].set_index(key)
                j = sub.join(base, how="inner", lsuffix="", rsuffix="_b")
                dd = (j[met] - j[f"{met}_b"]).dropna()
                if not len(dd):
                    continue
                xs_o.append(al); ys_o.append(float(dd.mean()))
                delta_rows.append({"axis": pcname, "alpha": al, "regime": "all",
                                   "metric": met, "mean_delta": float(dd.mean()),
                                   "sem": float(dd.std()/max(len(dd),1)**0.5),
                                   "n_pairs": int(len(dd))})
                for rg in regimes:
                    ddr = (j[met] - j[f"{met}_b"])[j["regime"] == rg].dropna()
                    if len(ddr) >= 5:
                        delta_rows.append({"axis": pcname, "alpha": al,
                                           "regime": rg, "metric": met,
                                           "mean_delta": float(ddr.mean()),
                                           "sem": float(ddr.std()/len(ddr)**0.5),
                                           "n_pairs": int(len(ddr))})
            # per-regime lines
            for rg in regimes:
                xr, yr, er = [], [], []
                for al in alphas:
                    hit = [r for r in delta_rows
                           if r["axis"] == pcname and r["alpha"] == al
                           and r["regime"] == rg and r["metric"] == met]
                    if hit:
                        xr.append(al); yr.append(hit[0]["mean_delta"])
                        er.append(1.96 * hit[0]["sem"])
                if xr:
                    ax.errorbar(xr, yr, yerr=er, marker="o", ms=3, capsize=2,
                                lw=1, alpha=0.7, label=names.get(rg, f"reg {rg}"))
            if len(xs_o) >= 2:
                ax.plot(xs_o, ys_o, "k-o", lw=2.2, ms=4, label="overall")
                slope = float(np.polyfit(xs_o, ys_o, 1)[0])
                yy = np.array(ys_o)[np.argsort(xs_o)]
                mono = bool(np.all(np.diff(yy) >= -1e-9)
                            or np.all(np.diff(yy) <= 1e-9))
                ax.set_title(f"Δ {METRIC_LABEL[met]}\nslope={slope:+.2f}/σ"
                             f"{'  (mono)' if mono else ''}", fontsize=8)
            ax.axhline(0, color="grey", lw=0.7)
            ax.axvline(0, color="grey", lw=0.7, ls=":")
            ax.set_xlabel(f"{pcname} (σ)", fontsize=8)
            ax.grid(alpha=0.3)

            # headline paired test: +2 vs -2 on the SAME maps
            lo, hi = min(alphas), max(alphas)
            s_lo = df[df["cond"] == cond_of[lo]].set_index(key)
            s_hi = df[df["cond"] == cond_of[hi]].set_index(key)
            j2 = s_hi.join(s_lo, how="inner", lsuffix="", rsuffix="_lo")
            dd2 = (j2[met] - j2[f"{met}_lo"]).dropna()
            if len(dd2) >= 5:
                t = float(dd2.mean() / (dd2.std() / len(dd2) ** 0.5))
                test_rows.append({"axis": pcname, "metric": met,
                                  "hi": hi, "lo": lo,
                                  "mean_paired_diff": float(dd2.mean()),
                                  "sem": float(dd2.std()/len(dd2)**0.5),
                                  "t_stat": t, "n_pairs": int(len(dd2))})
        h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=7, ncol=len(l), loc="lower center")
        fig.suptitle(f"PAIRED forced-role dose-response along {pcname} "
                     f"({100*evr[d]:.0f}% var) — Δ vs α=0 on the SAME "
                     f"(map, vehicle); map noise cancels", fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out / f"role_paired_{pcname}.png", dpi=140)
        plt.close(fig)

    pd.DataFrame(delta_rows).to_csv(out / "role_paired_deltas.csv", index=False)
    tdf = pd.DataFrame(test_rows)
    tdf.to_csv(out / "role_paired_tests.csv", index=False)
    print("\n[paired] === HEADLINE: alpha=+2 vs -2, SAME (map, vehicle) ===")
    for _, r in tdf.iterrows():
        print(f"[paired]   {r['axis']}  {r['metric']:<11}: "
              f"diff={r['mean_paired_diff']:+.3f} ± {1.96*r['sem']:.3f}  "
              f"t={r['t_stat']:+.1f}  n={int(r['n_pairs'])} pairs")
    print(f"\n[paired] outputs -> {out}/role_paired_*.png + *.csv")


if __name__ == "__main__":
    main()
