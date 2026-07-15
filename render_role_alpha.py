"""
render_role_alpha.py -- videos of the SAME scene at every forced alpha.

For each map regime: pick N core scenes; for each scene roll the SAME map with
the SAME focal vehicle at every alpha in the grid (role forced to
mu + alpha*sigma*PC_d; all other agents natural), and write:

  r{rg}_{sid}_a{alpha}.mp4      one video per alpha -- focal is the big colored
                                box (blue=alpha-2 .. red=alpha+2), context cars
                                replay their human GT, goal star shown; the
                                title carries the MEASURED numbers (speed,
                                |accel|, |jerk|, |turn|) for that alpha.
  overlay_r{rg}_{sid}.png       all alpha trajectories on one image.
  metrics_r{rg}_{sid}.csv       per-alpha measured metrics for that scene.
  alpha_render_summary.csv      all scenes x alphas in one table.

The forced directions are read from role_paired_axes.csv (written by
role_paired_sweep.py) so videos and statistics use the IDENTICAL axes; if the
file is missing a quick natural warmup estimates them instead.

Reuses render_role_conditions.py machinery: scene relocation by
scenario_id+vehicle id across reset reshuffles, two-pass focal forcing,
GT-replay background, goal star, all-or-nothing scene writes.

Usage (from /scratch/e452103/PufferDrive):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/render_role_alpha.py \
        --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_final.pt \
        --clusters   /scratch/e452103/map_atlas/k4/map_clusters.csv \
        --axes_csv   /scratch/e452103/role_paired/dim4/role_paired_axes.csv \
        --data_dir   pufferlib/resources/drive/binaries/training \
        --out_dir    /scratch/e452103/renders/role_alpha_dim4
"""

import argparse
import csv
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
from render_role_conditions import (rollout_forced, scan_allocation, scene_view,
                                    hide_after_frame, render_condition_video,
                                    scene_setup, bbox_of, include_point,
                                    first_segment, focal_goal,
                                    focal_collision_frames)
from role_regime_analysis import _squeeze
from traj_kinematics import ego_kinematics

T = 91
TELEPORT_M = 4.0      # metric segment cut (respawn discontinuity)
EVENT_REW  = -0.4
MET_COLS   = ["speed_mean", "accel_abs", "jerk_abs", "turn_abs", "event_rate",
              "goal_min_m", "reached"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--clusters",   type=str, required=True)
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--axes_csv",   type=str, default="",
                   help="role_paired_axes.csv from role_paired_sweep.py "
                        "(same axes as the statistics). Empty = quick warmup.")
    p.add_argument("--axis",       type=str, default="PC1",
                   help="Which axis to sweep in the videos (PC1 or PC2)")
    p.add_argument("--alphas",     type=str, default="-2,-1,0,1,2")
    p.add_argument("--maps_per_regime", type=int, default=1)
    p.add_argument("--map_pool",   type=int, default=128)
    p.add_argument("--total_agents", type=int, default=768)
    p.add_argument("--min_margin", type=float, default=1.5)
    p.add_argument("--min_focal_drive", type=float, default=5.0)
    p.add_argument("--max_resamples", type=int, default=10)
    p.add_argument("--max_relocate", type=int, default=80)
    p.add_argument("--seed_start", type=int, default=100)
    p.add_argument("--goal_radius", type=float, default=2.0)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--fps",        type=int, default=10)
    p.add_argument("--dpi",        type=int, default=110)
    return p.parse_args()


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Axes: load from the paired sweep (preferred) or estimate via quick warmup
# ---------------------------------------------------------------------------

def load_axes(axes_csv, role_dim):
    import pandas as pd
    df = pd.read_csv(axes_csv)
    cols = [f"c{i}" for i in range(role_dim)]
    mu = df[df["name"] == "mu"][cols].values[0].astype(np.float64)
    axes = {}
    for _, r in df[df["name"] != "mu"].iterrows():
        axes[str(r["name"])] = (r[cols].values.astype(np.float64),
                                float(r["sigma"]))
    return mu, axes


def warmup_axes(env, policy, device, episodes=2, n_axes=2):
    """Fallback: natural rollouts -> mu + PC axes (same math as the sweep)."""
    vecs = []
    for ep in range(episodes):
        if ep > 0:
            env.resample_maps()
        obs_np, _ = env.reset()
        B = env.num_agents
        rl = np.zeros((T, B, policy.role_dim), np.float32)
        xs = np.zeros((T, B), np.float32); ys = np.zeros((T, B), np.float32)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        state = policy.initial_state(B, device)
        for t in range(T):
            ag = env.get_global_agent_state()
            xs[t], ys[t] = ag["x"], ag["y"]
            with torch.no_grad():
                logits, _, state, ri = policy(obs, state)
            rl[t] = ri["role_mean"].float().cpu().numpy()
            action = Categorical(logits=logits.float()).sample()
            obs_np, *_ = env.step(action.cpu().numpy().reshape(B, 1))
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        step_d = np.hypot(np.diff(xs, axis=0), np.diff(ys, axis=0))
        for a in range(B):
            jumps = np.where(step_d[:, a] > TELEPORT_M)[0]
            t_end = int(jumps[0] + 1) if len(jumps) else T
            if t_end > 10:
                vecs.append(rl[:t_end, a].mean(axis=0))
    R = np.asarray(vecs)
    mu = R.mean(axis=0)
    cen = R - mu
    _, _, vt = np.linalg.svd(cen, full_matrices=False)
    axes = {f"PC{d+1}": (vt[d], float((cen @ vt[d]).std()))
            for d in range(min(n_axes, R.shape[1]))}
    print(f"[alpha] warmup axes from {len(R)} role vectors")
    return mu, axes


# ---------------------------------------------------------------------------
# Direct ego metrics of the focal over its pre-respawn segment
# ---------------------------------------------------------------------------

def focal_numbers(data):
    f = data["focal"]
    fx = data["xs"][:, f].astype(np.float64)
    fy = data["ys"][:, f].astype(np.float64)
    fh = data["hs"][:, f].astype(np.float64)
    step = np.hypot(np.diff(fx), np.diff(fy))
    tp = np.where(step > TELEPORT_M)[0]
    end = int(tp[0] + 1) if len(tp) else T
    spd = step[:max(end - 1, 1)] * 10.0
    dh  = wrap_angle(np.diff(fh[:end])) * 10.0
    n   = min(len(spd), len(dh))
    k = ego_kinematics(spd[:n], dh[:n])           # plausibility-masked kinematics
    collided = focal_collision_frames(data, end)
    return {
        "speed_mean": k["speed_mean"],
        "accel_abs":  k["accel_abs"],
        "jerk_abs":   k["jerk_abs"],
        "turn_abs":   k["turn_abs"],
        "event_rate": int(collided.sum()),   # vehicle-collision frames (of `end`); offroad not included
        "seg_end":    end,
    }


def render_alpha_overlay(views, alphas, cmap, title, out_path, dpi, metrics):
    """All alpha focal paths on one image, colored blue->red, legend carries
    the measured speed per alpha."""
    ref = views[alphas[0]]
    cuts = {al: first_segment(v["xs"][:, v["focal"]], v["ys"][:, v["focal"]])
            for al, v in views.items()}
    polys = ref["road_polys"]
    all_x = np.concatenate([c[0] for c in cuts.values()])
    all_y = np.concatenate([c[1] for c in cuts.values()])
    x0, x1, y0, y1 = bbox_of(polys, all_x[:, None], all_y[:, None])
    x0, x1, y0, y1 = include_point(x0, x1, y0, y1,
                                   focal_goal(ref["gt"], ref["focal"]))
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#0f0f1a")
    scene_setup(ax, polys, ref["gt"], x0, x1, y0, y1, focal=ref["focal"])
    for al in alphas:
        px, py = cuts[al]
        col = cmap((al - min(alphas)) / max(max(alphas) - min(alphas), 1e-9))
        ax.plot(px, py, color=col, lw=2.4, alpha=0.95,
                label=f"α={al:+g}  v={metrics[al]['speed_mean']:.1f} m/s")
        if len(px):
            ax.scatter(px[-1], py[-1], color=col, s=90, marker="X",
                       zorder=6, edgecolors="white", linewidths=0.7)
    ax.scatter(ref["xs"][0, ref["focal"]], ref["ys"][0, ref["focal"]],
               color="white", s=70, zorder=6, label="start")
    ax.legend(fontsize=9, loc="best")
    ax.set_title(title, color="#e8e8f0", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="#0f0f1a")
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = sorted(float(x) for x in args.alphas.split(","))
    cmap   = plt.get_cmap("coolwarm")

    import pandas as pd
    cl = pd.read_csv(args.clusters)
    cl = cl[cl["margin"] > args.min_margin]
    regime_of = dict(zip(cl["scenario_id"].astype(str), cl["cluster"]))
    regimes   = sorted(cl["cluster"].unique())
    print(f"[alpha] {len(regime_of)} core scenes, regimes {regimes}")

    from pufferlib.ocean.drive.drive import Drive
    # goal_speed=100 disables the respawn speed gate so fast forced agents
    # can register goal-reach (same as render_role_conditions).
    env = Drive(num_maps=args.map_pool, num_agents=args.total_agents,
                map_dir=args.data_dir, episode_length=T, goal_speed=100,
                seed=args.seed_start)
    obs_probe, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs_probe.shape[-1], device)
    if role_dim == 0:
        raise SystemExit("role_dim=0 checkpoint -- nothing to force")

    if args.axes_csv:
        mu, axes = load_axes(args.axes_csv, role_dim)
        print(f"[alpha] axes loaded from {args.axes_csv}: {list(axes)}")
    else:
        mu, axes = warmup_axes(env, policy, device)
    if args.axis not in axes:
        raise SystemExit(f"axis {args.axis} not in {list(axes)}")
    u, sg = axes[args.axis]

    need      = {rg: args.maps_per_regime for rg in regimes}
    found     = {rg: [] for rg in regimes}
    used_sids = set()
    n_target  = args.maps_per_regime * len(regimes)
    n_done    = 0
    summary   = []

    for rnd in range(args.max_resamples + 1):
        if rnd > 0:
            print(f"\n[scan] resampling map pool (round {rnd})", flush=True)
            env.resample_maps()
        scenes = scan_allocation(env, regime_of, need, found, used_sids,
                                 min_drive=args.min_focal_drive)
        print(f"[scan] round {rnd}: {len(scenes)} target scenes", flush=True)

        for sid, rg, focal_vid, human_m in scenes:
            print(f"\n[scene {n_done+1}/{n_target}] regime {rg} map {sid[:10]} "
                  f"focal vid {focal_vid} (human {human_m:.0f} m)", flush=True)

            # Phase 1: all alpha rollouts first (no partial files on failure)
            views, mets, ok = {}, {}, True
            for al in alphas:
                # natural-offset scheme (matches role_paired_sweep): the focal
                # keeps its own live role, shifted by al*sigma along the axis;
                # al=0 = fully natural (no forcing at all).
                vec = None if al == 0.0 else al * sg * u
                data = rollout_forced(env, policy, sid, focal_vid, vec,
                                      device, max_tries=args.max_relocate,
                                      shift=True)
                if data is None:
                    print(f"  [skip] {sid[:10]} not re-dealt on α={al:+g}")
                    found[rg].remove(sid)
                    ok = False
                    break
                mets[al] = focal_numbers(data)
                view = scene_view(data, data["slots"], sid)
                view["focal"] = int(np.where(
                    data["slots"] == data["focal"])[0][0])
                view["hide_after"] = hide_after_frame(view)
                views[al] = view
                print(f"  α={al:+g}: v={mets[al]['speed_mean']:.1f} m/s  "
                      f"|a|={mets[al]['accel_abs']:.1f}  "
                      f"|j|={mets[al]['jerk_abs']:.0f}  "
                      f"turn={mets[al]['turn_abs']:.2f}", flush=True)
            if not ok:
                continue

            # Phase 2: write videos + overlay + per-scene CSV
            try:
                for al in alphas:
                    m   = mets[al]
                    col = cmap((al - min(alphas))
                               / max(max(alphas) - min(alphas), 1e-9))
                    tag = (f"a{al:+g}".replace("+", "p").replace("-", "m")
                           .replace(".", ""))
                    name  = f"r{rg}_{sid[:10]}_{tag}.mp4"
                    title = (f"regime {rg} | {sid[:10]} | {args.axis} "
                             f"α={al:+g} | v={m['speed_mean']:.1f} m/s  "
                             f"|a|={m['accel_abs']:.1f}  "
                             f"turn={m['turn_abs']:.2f}")
                    dist, dmin, fmin, reached = render_condition_video(
                        views[al], views[al]["focal"], col, title,
                        out_dir / name, args.fps, args.dpi,
                        gt_background=True, goal_radius=args.goal_radius)
                    mets[al]["goal_min_m"] = dmin
                    mets[al]["reached"]    = int(reached)
                render_alpha_overlay(
                    views, alphas, cmap,
                    f"regime {rg} | {sid[:10]} | same scene, {args.axis} "
                    f"α swept {min(alphas):+g}..{max(alphas):+g}",
                    out_dir / f"overlay_r{rg}_{sid[:10]}.png", args.dpi, mets)
                with open(out_dir / f"metrics_r{rg}_{sid[:10]}.csv", "w",
                          newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["alpha"] + MET_COLS)
                    for al in alphas:
                        w.writerow([al] + [mets[al].get(k, np.nan)
                                           for k in MET_COLS])
                for al in alphas:
                    summary.append({"regime": rg, "sid": sid, "alpha": al,
                                    **{k: mets[al].get(k, np.nan)
                                       for k in MET_COLS}})
                n_done += 1
            except Exception as e:
                print(f"  [render error {sid[:10]}: {e}] removing partials")
                for fpath in list(out_dir.glob(f"*{sid[:10]}*")):
                    try:
                        fpath.unlink()
                    except OSError:
                        pass
                found[rg].remove(sid)
                import traceback; traceback.print_exc()

        if n_done >= n_target:
            break

    env.close()
    missing = {rg: need[rg] - len(found[rg])
               for rg in need if len(found[rg]) < need[rg]}
    if missing:
        print(f"\n[alpha] WARNING: regimes NOT filled after "
              f"{args.max_resamples} resamples: {missing}. Causes: no scene of "
              f"that regime passed --min_focal_drive ({args.min_focal_drive} m) "
              f"in the {args.map_pool}-map pool, or the scene failed to re-deal "
              f"within --max_relocate resets on some alpha (see '[skip]' lines "
              f"above). Retry with --max_resamples 20 and/or "
              f"--min_focal_drive 3.")
    if summary:
        pd.DataFrame(summary).to_csv(out_dir / "alpha_render_summary.csv",
                                     index=False)
        print("\n[alpha] === summary: mean speed by regime x alpha ===")
        sdf = pd.DataFrame(summary)
        piv = sdf.pivot_table(index="regime", columns="alpha",
                              values="speed_mean", aggfunc="mean")
        print(piv.round(1).to_string())
    print(f"\n[alpha] done -> {out_dir}  ({n_done} scenes)")


if __name__ == "__main__":
    main()
