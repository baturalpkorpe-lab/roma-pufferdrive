"""
render_role_grid.py -- the overnight render batch: for EVERY trajectory type,
N scenes each, sweep BOTH role axes on the same scene.

Per (type, scene), rollouts (natural-offset forcing, focal only, others as in
the sim; background drawn per --background):

    natural                    (0, 0)  -- shared baseline
    PC1 sweep                  (a, 0)  for a in --alphas, a != 0
    PC2 sweep                  (0, b)  for b in --alphas, b != 0
    corners (--grid 1)         (a, b)  for a, b in {min,max}x{min,max}

Default 5-value alphas -> 1 + 4 + 4 + 4 = 13 videos per scene;
4 types x --scenes_per_type 3 = 12 scenes -> ~156 videos. Plus, per scene:
    overlay_pc1_*.png   the PC1 fan (coolwarm)
    overlay_pc2_*.png   the PC2 fan (PiYG)
    overlay_grid_*.png  natural + 4 corners
    metrics_*.csv       measured numbers per condition
and a global grid_summary.csv + type x condition speed table.

Strata: --traj_clusters = the 6-feature (stop_frac) K-means clustering.
Core trajectories only (margin filter + is_edge==0 when present).

Usage (from /scratch/e452103/PufferDrive): see slurm/render_role_grid.sbatch.
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
                                    hide_after_frame, render_condition_video)
from render_role_compare import scan_for_types, render_compare_overlay
from render_role_alpha import load_axes, focal_numbers

T = 91
MET_COLS = ["speed_mean", "accel_abs", "accel_pos", "decel_abs", "jerk_abs",
            "turn_abs", "event_rate", "offroad_rate", "goal_min_m", "reached"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--axes_csv",        type=str, required=True)
    p.add_argument("--traj_clusters",   type=str, required=True)
    p.add_argument("--data_dir",        type=str, required=True)
    p.add_argument("--out_dir",         type=str, required=True)
    p.add_argument("--alphas",          type=str, default="-2,-1,0,1,2")
    p.add_argument("--grid",            type=int, default=1,
                   help="1 = also render the 4 (PC1,PC2) corner combinations")
    p.add_argument("--scenes_per_type", type=int, default=3)
    p.add_argument("--type_names",      type=str, default="",
                   help="'id:name,...' from the atlas names")
    p.add_argument("--background",      type=str, default="gt",
                   choices=["gt", "sim"])
    p.add_argument("--min_margin",      type=float, default=1.5)
    p.add_argument("--map_pool",        type=int, default=128)
    p.add_argument("--total_agents",    type=int, default=768)
    p.add_argument("--min_focal_drive", type=float, default=5.0)
    p.add_argument("--max_resamples",   type=int, default=15)
    p.add_argument("--max_relocate",    type=int, default=120)
    p.add_argument("--seed_start",      type=int, default=100)
    p.add_argument("--goal_radius",     type=float, default=2.0)
    p.add_argument("--device",          type=str, default="cpu")
    p.add_argument("--fps",             type=int, default=10)
    p.add_argument("--dpi",             type=int, default=110)
    return p.parse_args()


def cond_list(alphas, grid):
    """[(name, a1, a2)] -- natural + PC1 fan + PC2 fan + corners."""
    conds = [("nat", 0.0, 0.0)]
    for a in alphas:
        if a != 0.0:
            conds.append((f"pc1{a:+g}", a, 0.0))
    for b in alphas:
        if b != 0.0:
            conds.append((f"pc2{b:+g}", 0.0, b))
    if grid:
        lo, hi = min(alphas), max(alphas)
        for a in (lo, hi):
            for b in (lo, hi):
                conds.append((f"g{a:+g}_{b:+g}", a, b))
    return conds


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    alphas  = sorted(float(x) for x in args.alphas.split(","))
    conds   = cond_list(alphas, args.grid)
    print(f"[grid] {len(conds)} conditions per scene: "
          f"{[c[0] for c in conds]}")

    import pandas as pd
    tc = pd.read_csv(args.traj_clusters)
    core = tc["margin"] > args.min_margin
    if "is_edge" in tc.columns:
        core &= tc["is_edge"] == 0
    tc = tc[core]
    traj_of = {(str(r.scenario_id), int(r.vehicle_id)): int(r.cluster)
               for r in tc.itertuples()}
    types = sorted(tc["cluster"].unique())
    names = {}
    if args.type_names:
        names = {int(k): v for k, v in
                 (it.split(":") for it in args.type_names.split(","))}
    print(f"[grid] {len(traj_of)} core trajectories, types {types}")

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
        raise SystemExit(f"axes csv needs PC1+PC2, has {list(axes)}")
    u1, s1 = axes["PC1"]
    u2, s2 = axes["PC2"]

    # colors: PC1 fan coolwarm, PC2 fan PiYG, natural white, corners olive tones
    c_pc1, c_pc2 = plt.get_cmap("coolwarm"), plt.get_cmap("PiYG")
    span = max(alphas) - min(alphas) or 1.0
    colors = {}
    for name, a, b in conds:
        if name == "nat":
            colors[name] = "#f0f0f0"
        elif name.startswith("pc1"):
            colors[name] = c_pc1((a - min(alphas)) / span)
        elif name.startswith("pc2"):
            colors[name] = c_pc2((b - min(alphas)) / span)
        else:
            colors[name] = plt.get_cmap("tab10")(
                [( -2, -2), (-2, 2), (2, -2), (2, 2)].index(
                    (int(np.sign(a) * 2), int(np.sign(b) * 2))) + 4)

    need      = {t: args.scenes_per_type for t in types}
    found     = {t: [] for t in types}
    used_sids = set()
    n_target  = args.scenes_per_type * len(types)
    n_done    = 0
    summary   = []

    for rnd in range(args.max_resamples + 1):
        if rnd > 0:
            print(f"\n[scan] resampling map pool (round {rnd})", flush=True)
            env.resample_maps()
        scenes = scan_for_types(env, traj_of, need, found, used_sids,
                                args.min_focal_drive)
        print(f"[scan] round {rnd}: {len(scenes)} target scenes", flush=True)

        for sid, t, focal_vid, human_m in scenes:
            tname = names.get(t, f"type{t}")
            print(f"\n[scene {n_done+1}/{n_target}] type {t} ({tname}) "
                  f"{sid[:10]} focal {focal_vid} (human {human_m:.0f} m)",
                  flush=True)

            views, mets, ok = {}, {}, True
            for name, a, b in conds:
                vec = a * s1 * u1 + b * s2 * u2
                if not np.any(vec):
                    vec = None
                data = rollout_forced(env, policy, sid, focal_vid, vec, device,
                                      max_tries=args.max_relocate, shift=True)
                if data is None:
                    print(f"  [skip] {sid[:10]} not re-dealt on '{name}'")
                    found[t].remove(sid)
                    ok = False
                    break
                mets[name] = focal_numbers(data)
                view = scene_view(data, data["slots"], sid)
                view["focal"] = int(np.where(
                    data["slots"] == data["focal"])[0][0])
                view["hide_after"] = hide_after_frame(view)
                views[name] = view
                print(f"  {name:>8}: v={mets[name]['speed_mean']:.1f}  "
                      f"turn={mets[name]['turn_abs']:.2f}", flush=True)
            if not ok:
                continue

            try:
                for name, a, b in conds:
                    m   = mets[name]
                    tag = name.replace("+", "p").replace("-", "m")
                    fn  = f"t{t}_{sid[:10]}_{tag}.mp4"
                    ttl = (f"{tname} | {sid[:10]} | PC1={a:+g} PC2={b:+g} | "
                           f"v={m['speed_mean']:.1f}  |a|={m['accel_abs']:.1f}"
                           f"  turn={m['turn_abs']:.2f}")
                    dist, dmin, fmin, reached = render_condition_video(
                        views[name], views[name]["focal"], colors[name], ttl,
                        out_dir / fn, args.fps, args.dpi,
                        gt_background=(args.background == "gt"),
                        goal_radius=args.goal_radius)
                    mets[name]["goal_min_m"] = dmin
                    mets[name]["reached"]    = int(reached)

                base = f"t{t}_{sid[:10]}"
                pc1_set = ["nat"] + [c[0] for c in conds
                                     if c[0].startswith("pc1")]
                pc2_set = ["nat"] + [c[0] for c in conds
                                     if c[0].startswith("pc2")]
                render_compare_overlay(
                    views, pc1_set, colors,
                    f"{tname} | {sid[:10]} | PC1 sweep (PC2=0)",
                    out_dir / f"overlay_pc1_{base}.png", args.dpi, mets)
                render_compare_overlay(
                    views, pc2_set, colors,
                    f"{tname} | {sid[:10]} | PC2 sweep (PC1=0)",
                    out_dir / f"overlay_pc2_{base}.png", args.dpi, mets)
                if args.grid:
                    g_set = ["nat"] + [c[0] for c in conds
                                       if c[0].startswith("g")]
                    render_compare_overlay(
                        views, g_set, colors,
                        f"{tname} | {sid[:10]} | PC1 x PC2 corners",
                        out_dir / f"overlay_grid_{base}.png", args.dpi, mets)

                with open(out_dir / f"metrics_{base}.csv", "w",
                          newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["cond", "pc1", "pc2"] + MET_COLS)
                    for name, a, b in conds:
                        w.writerow([name, a, b] +
                                   [mets[name].get(k, np.nan)
                                    for k in MET_COLS])
                for name, a, b in conds:
                    summary.append({"traj_type": t, "sid": sid, "cond": name,
                                    "pc1": a, "pc2": b,
                                    **{k: mets[name].get(k, np.nan)
                                       for k in MET_COLS}})
                n_done += 1
            except Exception as e:
                print(f"  [render error {sid[:10]}: {e}] removing partials")
                for fp in list(out_dir.glob(f"*{sid[:10]}*")):
                    try:
                        fp.unlink()
                    except OSError:
                        pass
                found[t].remove(sid)
                import traceback; traceback.print_exc()

        if n_done >= n_target:
            break

    env.close()
    missing = {t: need[t] - len(found[t])
               for t in need if len(found[t]) < need[t]}
    if missing:
        print(f"\n[grid] WARNING: types NOT filled: {missing} -- raise "
              f"--max_resamples / lower --min_focal_drive")
    if summary:
        sdf = pd.DataFrame(summary)
        sdf.to_csv(out_dir / "grid_summary.csv", index=False)
        print("\n[grid] === mean speed by type x condition ===")
        print(sdf.pivot_table(index="traj_type", columns="cond",
                              values="speed_mean",
                              aggfunc="mean").round(1).to_string())
    print(f"\n[grid] done -> {out_dir}  ({n_done} scenes)")


if __name__ == "__main__":
    main()
