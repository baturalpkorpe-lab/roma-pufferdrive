"""
role_axis_sweep.py -- CAUSAL forced-role dose-response along the role axes.

The question this answers (and nothing observational can): "if we move the role
along axis d, does the agent get faster / more aggressive?" -- as a CAUSAL claim,
because the scene is held fixed and ONLY the focal agent's role is changed.

Design (Phase 2 of the role workplan):
  A. Warmup: roll the policy with natural roles, collect each agent's segment-mean
     role vector. PCA -> population mean mu, principal axes PC1..PCn, and the
     per-axis std sigma (so alpha is measured in sigma units of the real role
     distribution, not arbitrary).
  B. Sweep: for each axis d and each alpha in the grid, force ONE focal agent per
     scene to role = mu + alpha*sigma_d*PC_d while every other agent keeps its
     natural encoder role (two-pass forward, common action noise). Roll the same
     scene, measure the focal's OWN behavior over its pre-respawn segment.
  C. Aggregate behavior vs alpha per axis and per map regime -> a dose-response
     curve + a slope ("+1 sigma along PC1 => +X m/s") + a monotonicity check
     (clean ordered axis vs the folded "aggressive at both ends" pathology) +
     a replication check (same slope sign in every regime = real semantics).

Forcing ONE focal per scene (others natural) isolates the OWN-role effect while
holding the social context fixed -- cleaner than forcing the whole population.

Outputs in --out_dir:
  role_axis_sweep_agent.csv   one row per (axis, alpha, focal) observation
  role_axis_dose.csv          aggregated mean+sem per (axis, regime, alpha)
  role_axis_slopes.csv        per-axis, per-metric slope + Spearman monotonicity
  role_axis_PC{d}.png         dose-response: behavior vs alpha, line per regime

Usage (from /scratch/e452103/PufferDrive):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/role_axis_sweep.py \
        --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_final.pt \
        --clusters   /scratch/e452103/map_atlas/k4/map_clusters.csv \
        --data_dir   pufferlib/resources/drive/binaries/training \
        --out_dir    /scratch/e452103/role_axis/dim4
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

T = 91
TELEPORT_M = 4.0     # matches the fixed segment cut (respawn discontinuity)
MIN_STEPS  = 10
GT_MOVE_MS = 1.0
EVENT_REW  = -0.4
ROLLOUT_SEED = 1234  # identical action noise across every condition

# behavior metrics measured on the focal's own pre-respawn segment
METRICS = ["speed_mean", "accel_abs", "jerk_abs", "turn_abs",
           "d_speed", "ade", "event_rate"]
METRIC_LABEL = {"speed_mean": "speed (m/s)", "accel_abs": "|accel| (m/s2)",
                "jerk_abs": "|jerk| (m/s3)", "turn_abs": "|turn| (rad/s)",
                "d_speed": "speed - human (m/s)", "ade": "path dev vs human (m)",
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
    p.add_argument("--clusters",   type=str, required=True,
                   help="map_clusters.csv (K=4 atlas)")
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--warmup_episodes", type=int, default=3,
                   help="Natural rollouts to estimate the role axes (PCA)")
    p.add_argument("--episodes",   type=int, default=4,
                   help="Sweep rollouts per (axis, alpha) condition")
    p.add_argument("--num_agents", type=int, default=3072)
    p.add_argument("--num_maps",   type=int, default=10000)
    p.add_argument("--min_margin", type=float, default=1.5)
    p.add_argument("--exclude_clusters", type=str, default="",
                   help="Regime ids to drop (junk already removed by the atlas)")
    p.add_argument("--regime_names", type=str, default="")
    p.add_argument("--n_axes",     type=int, default=2,
                   help="How many principal role axes to sweep (role is ~2-D)")
    p.add_argument("--alphas",     type=str, default="-2,-1,0,1,2",
                   help="Sweep grid in sigma units of the role distribution")
    p.add_argument("--device",     type=str, default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Behavior of one focal agent over its pre-respawn, GT-valid segment
# ---------------------------------------------------------------------------

def focal_metrics(a, xs, ys, hs, rews, gx, gy, gh, gvalid, T_gt):
    step_d = np.hypot(np.diff(xs[:, a]), np.diff(ys[:, a]))
    jumps  = np.where(step_d > TELEPORT_M)[0]
    t_end  = int(jumps[0] + 1) if len(jumps) else T
    t_end  = min(t_end, T_gt)
    if t_end < MIN_STEPS + 1:
        return None
    v = gvalid[a, :t_end]
    pair = v[:-1] & v[1:]
    if pair.sum() < MIN_STEPS:
        return None

    g_dx = (gx[a, 1:t_end] - gx[a, :t_end-1])[pair]
    g_dy = (gy[a, 1:t_end] - gy[a, :t_end-1])[pair]
    g_spd = np.hypot(g_dx, g_dy) * 10.0
    if g_spd.max() <= GT_MOVE_MS:
        return None
    g_dh = wrap_angle((gh[a, 1:t_end] - gh[a, :t_end-1])[pair]) * 10.0

    p_dx = np.diff(xs[:t_end, a])[pair]
    p_dy = np.diff(ys[:t_end, a])[pair]
    p_spd = np.hypot(p_dx, p_dy) * 10.0
    p_dh  = wrap_angle(np.diff(hs[:t_end, a])[pair]) * 10.0
    p_acc = np.diff(p_spd) * 10.0
    p_jrk = np.diff(p_acc) * 10.0

    vi  = np.where(v)[0]
    ade = float(np.mean(np.hypot(xs[vi, a] - gx[a, vi], ys[vi, a] - gy[a, vi])))
    return {
        "speed_mean": float(p_spd.mean()),
        "accel_abs":  float(np.abs(p_acc).mean()) if len(p_acc) else np.nan,
        "jerk_abs":   float(np.abs(p_jrk).mean()) if len(p_jrk) else np.nan,
        "turn_abs":   float(np.abs(p_dh).mean()),
        "d_speed":    float(p_spd.mean() - g_spd.mean()),
        "ade":        ade,
        "event_rate": float((rews[:t_end, a] <= EVENT_REW).sum() / t_end * T),
    }


def select_focals(gt, regime_of):
    """One focal vehicle per core scene = the one whose human drives the most.
    Returns (focal_slots, focal_regimes)."""
    sids   = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
    gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
    valid  = _squeeze(gt["valid"]).astype(bool)
    is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
    slots, regimes = [], []
    for sid in np.unique(sids):
        rg = regime_of.get(sid)
        if rg is None:
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
            slots.append(best); regimes.append(int(rg))
    return np.array(slots, dtype=int), np.array(regimes, dtype=int)


def rollout(env, policy, device, force_vec=None, regime_of=None):
    """One episode. force_vec=None -> natural roles (warmup). Otherwise select
    one focal per core scene FROM THIS reset's allocation and force each to
    force_vec every step (two-pass: others keep their natural encoder role)."""
    B = env.num_agents
    obs_np, _ = env.reset()
    gt = env.get_ground_truth_trajectories()

    focal_slots = np.array([], dtype=int)
    focal_rg    = np.array([], dtype=int)
    fv = None
    if force_vec is not None and regime_of is not None:
        focal_slots, focal_rg = select_focals(gt, regime_of)
        fv = torch.as_tensor(force_vec, dtype=torch.float32, device=device)

    torch.manual_seed(ROLLOUT_SEED)                       # identical action noise
    xs = np.zeros((T, B), np.float32); ys = np.zeros((T, B), np.float32)
    hs = np.zeros((T, B), np.float32); rews = np.zeros((T, B), np.float32)
    rl = np.zeros((T, B, policy.role_dim), np.float32)

    obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    state = policy.initial_state(B, device)
    for t in range(T):
        ag = env.get_global_agent_state()
        xs[t], ys[t], hs[t] = ag["x"], ag["y"], ag["heading"]
        with torch.no_grad():
            if fv is None or len(focal_slots) == 0:
                logits, _, state, ri = policy(obs, state)
            else:
                _, _, _, ri0 = policy(obs, state)         # pass 1: natural roles
                forced = ri0["role_z"].clone()
                forced[focal_slots] = fv                   # override focal only
                logits, _, state, ri = policy(obs, state, forced_role=forced)
        if ri.get("role_mean") is not None and ri["role_mean"].shape[-1]:
            rl[t] = ri["role_mean"].float().cpu().numpy()
        action = Categorical(logits=logits.float()).sample()
        obs_np, rew_np, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
        rews[t] = np.asarray(rew_np).reshape(B)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    return {"xs": xs, "ys": ys, "hs": hs, "rews": rews, "roles": rl,
            "gt": gt, "focal_slots": focal_slots, "focal_rg": focal_rg}


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
    print(f"[axis] {len(regime_of)} core scenarios, regimes "
          f"{sorted(cl['cluster'].unique())}")

    from pufferlib.ocean.drive.drive import Drive
    env_cfg = dict(load_drive_config()["env"])
    env_cfg.update({"num_maps": args.num_maps, "num_agents": args.num_agents,
                    "map_dir": args.data_dir})
    env = Drive(**env_cfg)

    obs_np, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs_np.shape[-1], device)
    if role_dim == 0:
        raise SystemExit("checkpoint has role_dim=0 (no-role) -- nothing to sweep")
    n_axes = min(args.n_axes, role_dim)
    print(f"[axis] role_dim={role_dim}, sweeping {n_axes} axes")

    # -- A. Warmup: natural roles -> PCA axes (mu, PCs, per-axis sigma) --------
    role_vecs = []
    for ep in range(args.warmup_episodes):
        if ep > 0:
            env.resample_maps()
        data = rollout(env, policy, device)
        gt = data["gt"]
        gvalid = _squeeze(gt["valid"]).astype(bool)
        is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
        step_d = np.hypot(np.diff(data["xs"], axis=0), np.diff(data["ys"], axis=0))
        for a in range(env.num_agents):
            if not is_veh[a]:
                continue
            jumps = np.where(step_d[:, a] > TELEPORT_M)[0]
            t_end = int(jumps[0] + 1) if len(jumps) else T
            if t_end < MIN_STEPS + 1 or gvalid[a, :t_end].sum() < MIN_STEPS:
                continue
            role_vecs.append(data["roles"][:t_end, a].mean(axis=0))
        print(f"[axis] warmup {ep+1}/{args.warmup_episodes}: "
              f"{len(role_vecs)} role vectors", flush=True)

    R = np.asarray(role_vecs)
    mu = R.mean(axis=0)
    cen = R - mu
    _, svals, vt = np.linalg.svd(cen, full_matrices=False)
    U = vt[:n_axes]                                   # (n_axes, role_dim)
    sigma = np.array([cen @ U[d] for d in range(n_axes)]).std(axis=1)  # (n_axes,)
    evr = (svals**2) / (svals**2).sum()
    print(f"[axis] PCA: explained var {['%.0f%%' % (100*e) for e in evr[:n_axes]]}"
          f"  sigma={np.round(sigma,3)}")

    alphas = [float(x) for x in args.alphas.split(",")]

    # -- B. Sweep: force one focal per scene along each axis at each alpha -----
    rows = []
    for d in range(n_axes):
        for al in alphas:
            fvec = mu + al * sigma[d] * U[d]           # absolute role vector
            got = 0
            for ep in range(args.episodes):
                env.resample_maps()
                data = rollout(env, policy, device, force_vec=fvec,
                               regime_of=regime_of)
                gt = data["gt"]
                gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
                gh     = _squeeze(gt["heading"])
                gvalid = _squeeze(gt["valid"]).astype(bool)
                T_gt   = gx.shape[1]
                for a, rgm in zip(data["focal_slots"], data["focal_rg"]):
                    m = focal_metrics(a, data["xs"], data["ys"], data["hs"],
                                      data["rews"], gx, gy, gh, gvalid, T_gt)
                    if m is None:
                        continue
                    m.update({"axis": d, "alpha": al, "regime": int(rgm)})
                    rows.append(m); got += 1
            print(f"[axis] PC{d+1} alpha={al:+.0f}: {got} focal observations",
                  flush=True)
    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(out / "role_axis_sweep_agent.csv", index=False)
    if df.empty:
        raise SystemExit("[axis] no observations collected")

    # -- C. Aggregate + dose-response figures + slopes ------------------------
    regimes = sorted(df["regime"].unique())
    dose_rows, slope_rows = [], []
    for d in range(n_axes):
        sub_d = df[df["axis"] == d]
        fig, axes = plt.subplots(1, len(METRICS),
                                 figsize=(3.1 * len(METRICS), 3.6))
        for ax, met in zip(np.atleast_1d(axes), METRICS):
            # overall dose-response
            xs_a, ys_a = [], []
            for al in alphas:
                v = sub_d[sub_d["alpha"] == al][met].dropna().values
                if len(v):
                    xs_a.append(al); ys_a.append(v.mean())
            # per-regime lines
            for rg in regimes:
                sr = sub_d[sub_d["regime"] == rg]
                xr, yr, er = [], [], []
                for al in alphas:
                    v = sr[sr["alpha"] == al][met].dropna().values
                    if len(v) >= 5:
                        xr.append(al); yr.append(v.mean())
                        er.append(1.96 * v.std() / len(v) ** 0.5)
                        dose_rows.append({"axis": d, "regime": rg, "alpha": al,
                                          "metric": met, "mean": float(v.mean()),
                                          "sem": float(v.std()/len(v)**0.5),
                                          "n": int(len(v))})
                if xr:
                    ax.errorbar(xr, yr, yerr=er, marker="o", ms=3, capsize=2,
                                lw=1, alpha=0.7,
                                label=names.get(rg, f"reg {rg}"))
            if len(xs_a) >= 2:
                ax.plot(xs_a, ys_a, "k-o", lw=2.2, ms=4, label="overall")
                slope = float(np.polyfit(xs_a, ys_a, 1)[0])
                # monotonicity across alpha (overall means)
                order = np.argsort(xs_a)
                yy = np.array(ys_a)[order]
                mono = bool(np.all(np.diff(yy) >= -1e-9)
                            or np.all(np.diff(yy) <= 1e-9))
                slope_rows.append({"axis": d, "metric": met,
                                   "slope_per_sigma": slope, "monotone": mono})
                ax.set_title(f"{METRIC_LABEL[met]}\nslope={slope:+.2f}/σ"
                             f"{'  (mono)' if mono else '  (NOT mono)'}",
                             fontsize=8)
            else:
                ax.set_title(METRIC_LABEL[met], fontsize=8)
            ax.axvline(0, color="grey", lw=0.7, ls=":")
            ax.set_xlabel(f"PC{d+1} role (σ units)", fontsize=8)
            ax.grid(alpha=0.3)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=7, ncol=len(l), loc="lower center")
        fig.suptitle(f"Forced-role dose-response along PC{d+1} "
                     f"({100*evr[d]:.0f}% of role variance) — scene fixed, only "
                     f"the focal role changes (CAUSAL)", fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out / f"role_axis_PC{d+1}.png", dpi=140)
        plt.close(fig)

    pd.DataFrame(dose_rows).to_csv(out / "role_axis_dose.csv", index=False)
    sl = pd.DataFrame(slope_rows)
    sl.to_csv(out / "role_axis_slopes.csv", index=False)
    print("\n[axis] === causal slopes (overall, per sigma) ===")
    for _, r in sl.iterrows():
        print(f"[axis]   PC{int(r['axis'])+1:d}  {r['metric']:<11}: "
              f"{r['slope_per_sigma']:+.3f}/σ  "
              f"{'monotone' if r['monotone'] else 'NON-monotone'}")
    print(f"\n[axis] outputs -> {out}/role_axis_*.png + *.csv")


if __name__ == "__main__":
    main()
