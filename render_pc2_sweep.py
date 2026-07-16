"""
render_pc2_sweep.py -- hold PC1 fixed, sweep PC2 on ONE specific scene.

For a scene chosen by scenario-id prefix (from an earlier render_role_compare
run), re-locate the SAME focal vehicle and roll the identical scene 5 times
with the focal's own live role shifted by

    delta = pc1_alpha * sigma1 * PC1  +  beta * sigma2 * PC2,   beta in ALPHAS

i.e. the PC1 setting you saw in the compare videos, now with the PC2 knob
turned. Answers: "can PC2 tune the beginner-driver steering that appears at
PC1 = -2?"

--background gt  (default) draws non-focal vehicles at their GT routes (grey,
                 like the compare videos; NOTE ~5 of them are policy-driven in
                 the sim and are NOT drawn where they actually drove)
--background sim draws non-focal vehicles at their ACTUAL simulated positions
                 (what the ego really saw/avoided) -- use to diagnose "ghost/
                 clone" impressions.

Outputs per run: 5 MP4s (p2m2..p2p2) + overlay PNG + metrics CSV.

Usage (from /scratch/e452103/PufferDrive):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/render_pc2_sweep.py \
        --checkpoint    /scratch/e452103/checkpoints/roma_baseline_dim2/roma_dim2_final.pt \
        --axes_csv      /scratch/e452103/role_paired/dim2/role_paired_axes.csv \
        --traj_clusters /scratch/e452103/traj_atlas/k4/trajectory_clusters.csv \
        --data_dir      pufferlib/resources/drive/binaries/training \
        --sid_prefix    1b22f29f4c --traj_type 2 --pc1_alpha -2 \
        --out_dir       /scratch/e452103/renders/pc2_sweep/t2_1b22f29f4c_pc1m2
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import load_policy
from render_role_conditions import (rollout_forced, scene_view,
                                    hide_after_frame, render_condition_video,
                                    scene_setup, bbox_of, include_point,
                                    first_segment, focal_goal)
from render_role_alpha import load_axes, focal_numbers
from role_regime_analysis import _squeeze

T = 91
MET_COLS = ["speed_mean", "accel_abs", "jerk_abs", "turn_abs",
            "event_rate", "offroad_rate", "goal_min_m", "reached"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--axes_csv",      type=str, required=True)
    p.add_argument("--traj_clusters", type=str, required=True,
                   help="SAME clusters CSV the compare run used (k4)")
    p.add_argument("--data_dir",      type=str, required=True)
    p.add_argument("--out_dir",       type=str, required=True)
    p.add_argument("--sid_prefix",    type=str, required=True,
                   help="10-char scenario prefix from the compare filenames")
    p.add_argument("--traj_type",     type=int, required=True,
                   help="trajectory cluster of the focal (t{N} in filenames)")
    p.add_argument("--pc1_alpha",     type=float, required=True,
                   help="PC1 setting held fixed (e.g. -2)")
    p.add_argument("--alphas",        type=str, default="-2,-1,0,1,2",
                   help="PC2 sweep values")
    p.add_argument("--background",    type=str, default="gt",
                   choices=["gt", "sim"],
                   help="gt = grey cars on GT routes (compare-style); sim = "
                        "cars at their actual simulated positions")
    p.add_argument("--min_margin",    type=float, default=1.5)
    p.add_argument("--map_pool",      type=int, default=10000,
                   help="Maps LOADED. Must contain the target scenario, so we "
                        "load the full set (a 128 pool would rarely include a "
                        "named scene). reset/resample then deal it in.")
    p.add_argument("--total_agents",  type=int, default=3072,
                   help="More agents = more scenes live per reset = the target "
                        "is re-dealt faster.")
    p.add_argument("--max_relocate",  type=int, default=600)
    p.add_argument("--seed_start",    type=int, default=100)
    p.add_argument("--goal_radius",   type=float, default=2.0)
    p.add_argument("--device",        type=str, default="cpu")
    p.add_argument("--fps",           type=int, default=10)
    p.add_argument("--dpi",           type=int, default=110)
    return p.parse_args()


def resolve_scene(tc, sid_prefix, traj_type, min_margin):
    """Full sid + candidate focal vids from the clusters CSV."""
    m = tc[tc["scenario_id"].astype(str).str.startswith(sid_prefix)]
    if m.empty:
        raise SystemExit(f"no scenario starting with '{sid_prefix}' in the "
                         f"clusters CSV")
    sids = m["scenario_id"].astype(str).unique()
    if len(sids) > 1:
        raise SystemExit(f"prefix '{sid_prefix}' ambiguous: {list(sids)}")
    sid = str(sids[0])
    core = m[(m["margin"] > min_margin) & (m["cluster"] == traj_type)]
    if "is_edge" in m.columns:
        core = core[core["is_edge"] == 0]
    vids = [int(v) for v in core["vehicle_id"]]
    if not vids:
        raise SystemExit(f"no core type-{traj_type} trajectory in {sid}")
    return sid, vids


def find_focal(env, sid, cand_vids, max_tries):
    """Reset (and periodically resample the live map subset) until the scene is
    dealt in; pick the candidate whose human GT drives farthest (same rule the
    compare run used)."""
    for attempt in range(max_tries):
        if attempt > 0 and attempt % 25 == 0:
            env.resample_maps()          # draw a fresh live subset from the pool
            print(f"    (still locating {sid[:10]} -- {attempt} resets)",
                  flush=True)
        env.reset()
        gt   = env.get_ground_truth_trajectories()
        sids = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
        if sid not in sids:
            continue
        ids    = _squeeze(np.asarray(gt["id"])).reshape(-1)
        gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
        valid  = _squeeze(gt["valid"]).astype(bool)
        best, best_len = None, 0.0
        for a in np.where(sids == sid)[0]:
            if int(ids[a]) not in cand_vids or valid[a].sum() < 10:
                continue
            px, py = gx[a][valid[a]], gy[a][valid[a]]
            L = float(np.hypot(np.diff(px), np.diff(py)).sum())
            if L > best_len:
                best, best_len = int(ids[a]), L
        if best is not None:
            return best, best_len
    raise SystemExit(f"scene {sid[:10]} never dealt in with a candidate focal "
                     f"within {max_tries} resets")


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    betas = sorted(float(x) for x in args.alphas.split(","))
    cmap  = plt.get_cmap("PiYG")            # PC2 sweep: magenta<->green
    gt_bg = args.background == "gt"

    import pandas as pd
    tc = pd.read_csv(args.traj_clusters)
    sid, cand_vids = resolve_scene(tc, args.sid_prefix, args.traj_type,
                                   args.min_margin)
    print(f"[pc2] scene {sid} type {args.traj_type}: "
          f"{len(cand_vids)} candidate focals")

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(num_maps=args.map_pool, num_agents=args.total_agents,
                map_dir=args.data_dir, episode_length=T, goal_speed=100,
                seed=args.seed_start)
    obs_probe, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs_probe.shape[-1], device)
    if role_dim == 0:
        raise SystemExit("role_dim=0 checkpoint -- nothing to sweep")
    mu, axes = load_axes(args.axes_csv, role_dim)
    if "PC1" not in axes or "PC2" not in axes:
        raise SystemExit(f"axes csv must carry PC1 and PC2; has {list(axes)}")
    u1, s1 = axes["PC1"]
    u2, s2 = axes["PC2"]

    focal_vid, human_m = find_focal(env, sid, cand_vids, args.max_relocate)
    print(f"[pc2] focal vid {focal_vid} (human drives {human_m:.0f} m); "
          f"PC1 held at {args.pc1_alpha:+g}, sweeping PC2 {betas}")

    conds = [f"p2{b:+g}" for b in betas]
    colors = {f"p2{b:+g}": cmap((b - min(betas)) /
                                max(max(betas) - min(betas), 1e-9))
              for b in betas}

    views, mets = {}, {}
    for b in betas:
        c   = f"p2{b:+g}"
        vec = args.pc1_alpha * s1 * u1 + b * s2 * u2
        if not np.any(vec):
            vec = None                       # fully natural
        data = rollout_forced(env, policy, sid, focal_vid, vec, device,
                              max_tries=args.max_relocate, shift=True)
        if data is None:
            raise SystemExit(f"scene not re-dealt on {c}")
        mets[c] = focal_numbers(data)
        view = scene_view(data, data["slots"], sid)
        view["focal"] = int(np.where(data["slots"] == data["focal"])[0][0])
        view["hide_after"] = hide_after_frame(view)
        views[c] = view
        print(f"  PC2={b:+g}: v={mets[c]['speed_mean']:.1f} m/s  "
              f"|a|={mets[c]['accel_abs']:.1f}  "
              f"turn={mets[c]['turn_abs']:.2f}", flush=True)

    for b in betas:
        c = f"p2{b:+g}"
        m = mets[c]
        tag  = c.replace("+", "p").replace("-", "m")
        name = f"t{args.traj_type}_{sid[:10]}_pc1{args.pc1_alpha:+g}_{tag}.mp4"
        name = name.replace("+", "p").replace("-", "m")
        ttl  = (f"{sid[:10]} | PC1={args.pc1_alpha:+g} PC2={b:+g} | "
                f"v={m['speed_mean']:.1f} m/s  |a|={m['accel_abs']:.1f}  "
                f"turn={m['turn_abs']:.2f} | bg={args.background}")
        dist, dmin, fmin, reached = render_condition_video(
            views[c], views[c]["focal"], colors[c], ttl, out_dir / name,
            args.fps, args.dpi, gt_background=gt_bg,
            goal_radius=args.goal_radius)
        mets[c]["goal_min_m"] = dmin
        mets[c]["reached"]    = int(reached)

    # overlay of the five PC2 paths
    ref  = views[conds[0]]
    cuts = {c: first_segment(v["xs"][:, v["focal"]], v["ys"][:, v["focal"]])
            for c, v in views.items()}
    all_x = np.concatenate([c[0] for c in cuts.values()])
    all_y = np.concatenate([c[1] for c in cuts.values()])
    x0, x1, y0, y1 = bbox_of(ref["road_polys"], all_x[:, None], all_y[:, None])
    x0, x1, y0, y1 = include_point(x0, x1, y0, y1,
                                   focal_goal(ref["gt"], ref["focal"]))
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#0f0f1a")
    scene_setup(ax, ref["road_polys"], ref["gt"], x0, x1, y0, y1,
                focal=ref["focal"])
    for b in betas:
        c = f"p2{b:+g}"
        px, py = cuts[c]
        ax.plot(px, py, color=colors[c], lw=2.4, alpha=0.95,
                label=f"PC2={b:+g}  v={mets[c]['speed_mean']:.1f}  "
                      f"turn={mets[c]['turn_abs']:.2f}")
        if len(px):
            ax.scatter(px[-1], py[-1], color=colors[c], s=90, marker="X",
                       zorder=6, edgecolors="white", linewidths=0.7)
    ax.scatter(ref["xs"][0, ref["focal"]], ref["ys"][0, ref["focal"]],
               color="white", s=70, zorder=6, label="start")
    ax.legend(fontsize=9, loc="best")
    ax.set_title(f"{sid[:10]} | PC1 fixed {args.pc1_alpha:+g}, PC2 swept "
                 f"-- same scene, same seat", color="#e8e8f0", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"overlay_pc2_{sid[:10]}.png", dpi=args.dpi,
                facecolor="#0f0f1a")
    plt.close(fig)

    with open(out_dir / f"metrics_pc2_{sid[:10]}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pc2"] + MET_COLS)
        for b in betas:
            w.writerow([b] + [mets[f"p2{b:+g}"].get(k, np.nan)
                              for k in MET_COLS])
    env.close()
    print(f"\n[pc2] done -> {out_dir}")


if __name__ == "__main__":
    main()
