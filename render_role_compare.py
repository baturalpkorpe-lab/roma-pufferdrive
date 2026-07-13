"""
render_role_compare.py -- SAME scene, six policies-worth of videos:
dim-0 (no-role checkpoint) vs a role checkpoint at alpha = -2..+2 along PC1.

For each trajectory TYPE (K=4 atlas, core trajectories only: margin > 1.5 and
not an edge case), pick --scenes_per_type scenes whose focal vehicle carries
that trajectory type. Roll the IDENTICAL scene 6 times:

    dim0        the no-role ablation checkpoint (nothing to force)
    a-2..a+2    the role checkpoint, focal's OWN live role shifted by
                alpha*sigma along PC1 (natural-offset; a0 = fully natural)

Background vehicles replay their human GT (no policy weirdness); focal is the
big colored box; goal star shown; measured speed/|accel|/turn burned into
each title. Per scene: 6 MP4s + 1 overlay PNG (all six paths, dim0 in black)
+ metrics CSV. Total (4 types x 2 scenes): 48 videos.

The focal is identified by scenario_id + GT vehicle id and re-located after
every reset (allocation reshuffles), so all 6 rollouts share map AND seat.

Usage (from /scratch/e452103/PufferDrive):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/render_role_compare.py \
        --checkpoint      /scratch/e452103/checkpoints/roma_baseline_dim2/roma_dim2_final.pt \
        --checkpoint_dim0 /scratch/e452103/checkpoints/roma_baseline_dim0/roma_dim0_final.pt \
        --axes_csv        /scratch/e452103/role_paired/dim2/role_paired_axes.csv \
        --traj_clusters   /scratch/e452103/traj_atlas/k4/trajectory_clusters.csv \
        --data_dir        pufferlib/resources/drive/binaries/training \
        --out_dir         /scratch/e452103/renders/role_compare_dim2
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
            "goal_min_m", "reached"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      type=str, required=True,
                   help="Role checkpoint (dim-2/4/8 ...)")
    p.add_argument("--checkpoint_dim0", type=str, required=True,
                   help="No-role ablation checkpoint")
    p.add_argument("--traj_clusters",   type=str, required=True,
                   help="trajectory_clusters.csv (K=4 atlas)")
    p.add_argument("--axes_csv",        type=str, default="",
                   help="role_paired_axes.csv matching --checkpoint "
                        "(empty = quick natural warmup)")
    p.add_argument("--data_dir",        type=str, required=True)
    p.add_argument("--out_dir",         type=str, required=True)
    p.add_argument("--axis",            type=str, default="PC1")
    p.add_argument("--alphas",          type=str, default="-2,-1,0,1,2")
    p.add_argument("--scenes_per_type", type=int, default=2)
    p.add_argument("--min_margin",      type=float, default=1.5,
                   help="Core-trajectory filter (no transition/edge cases)")
    p.add_argument("--type_names",      type=str, default="",
                   help="Optional 'id:name,id:name' from tatlas_names.txt")
    p.add_argument("--map_pool",        type=int, default=128)
    p.add_argument("--total_agents",    type=int, default=768)
    p.add_argument("--min_focal_drive", type=float, default=5.0)
    p.add_argument("--max_resamples",   type=int, default=12)
    p.add_argument("--max_relocate",    type=int, default=80)
    p.add_argument("--seed_start",      type=int, default=100)
    p.add_argument("--goal_radius",     type=float, default=2.0)
    p.add_argument("--device",          type=str, default="cpu")
    p.add_argument("--fps",             type=int, default=10)
    p.add_argument("--dpi",             type=int, default=110)
    return p.parse_args()


def scan_for_types(env, traj_of, need, found, used_sids, min_drive):
    """Scan the live allocation for scenes whose CORE-trajectory focal belongs
    to a still-needed type. Focal = that (sid, vid) itself, not the farthest
    driver. Returns [(sid, ttype, focal_vid, human_m)]."""
    env.reset()
    gt     = env.get_ground_truth_trajectories()
    sids   = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
    ids    = _squeeze(np.asarray(gt["id"])).reshape(-1)
    gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
    valid  = _squeeze(gt["valid"]).astype(bool)
    is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)

    accepted = []
    for sid in np.unique(sids):
        if not sid or sid in used_sids or str(sid).lower().startswith("map"):
            continue
        # candidate focals in this scene: core trajectories of needed types
        cands = []          # (drive_m, ttype, vid)
        for a in np.where(sids == sid)[0]:
            if not is_veh[a] or valid[a].sum() < 10:
                continue
            key = (sid, int(ids[a]))
            t = traj_of.get(key)
            if t is None or len(found[t]) >= need[t]:
                continue
            px, py = gx[a][valid[a]], gy[a][valid[a]]
            L = float(np.hypot(np.diff(px), np.diff(py)).sum())
            if L >= min_drive:
                cands.append((L, t, int(ids[a])))
        if not cands:
            continue
        L, t, vid = max(cands)                 # farthest-driving qualifying focal
        accepted.append((sid, t, vid, L))
        found[t].append(sid)
        used_sids.add(sid)
    return accepted


def render_compare_overlay(views, conds, colors, title, out_path, dpi, mets):
    ref = views[conds[0]]
    cuts = {c: first_segment(v["xs"][:, v["focal"]], v["ys"][:, v["focal"]])
            for c, v in views.items()}
    polys = ref["road_polys"]
    all_x = np.concatenate([c[0] for c in cuts.values()])
    all_y = np.concatenate([c[1] for c in cuts.values()])
    x0, x1, y0, y1 = bbox_of(polys, all_x[:, None], all_y[:, None])
    x0, x1, y0, y1 = include_point(x0, x1, y0, y1,
                                   focal_goal(ref["gt"], ref["focal"]))
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#0f0f1a")
    scene_setup(ax, polys, ref["gt"], x0, x1, y0, y1, focal=ref["focal"])
    for c in conds:
        px, py = cuts[c]
        ax.plot(px, py, color=colors[c], lw=2.4, alpha=0.95,
                ls="--" if c == "dim0" else "-",
                label=f"{c}  v={mets[c]['speed_mean']:.1f} m/s")
        if len(px):
            ax.scatter(px[-1], py[-1], color=colors[c], s=90, marker="X",
                       zorder=6, edgecolors="white", linewidths=0.7)
    ax.scatter(ref["xs"][0, ref["focal"]], ref["ys"][0, ref["focal"]],
               color="white", s=70, zorder=6, label="start")
    ax.legend(fontsize=9, loc="best")
    ax.set_title(title, color="#e8e8f0", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="#0f0f1a")
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


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
    tc = pd.read_csv(args.traj_clusters)
    core = tc["margin"] > args.min_margin
    if "is_edge" in tc.columns:                 # older CSVs lack the column
        core &= tc["is_edge"] == 0
    tc = tc[core]
    traj_of = {(str(r.scenario_id), int(r.vehicle_id)): int(r.cluster)
               for r in tc.itertuples()}
    types = sorted(tc["cluster"].unique())
    names = {}
    if args.type_names:
        names = {int(k): v for k, v in
                 (it.split(":") for it in args.type_names.split(","))}
    print(f"[compare] {len(traj_of)} core trajectories across types {types}")

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(num_maps=args.map_pool, num_agents=args.total_agents,
                map_dir=args.data_dir, episode_length=T, goal_speed=100,
                seed=args.seed_start)
    obs_probe, _ = env.reset()
    obs_dim = obs_probe.shape[-1]
    policyX, role_dim = load_policy(args.checkpoint, obs_dim, device)
    policy0, rd0      = load_policy(args.checkpoint_dim0, obs_dim, device)
    if role_dim == 0 or rd0 != 0:
        raise SystemExit(f"expected role ckpt (got dim {role_dim}) + dim-0 "
                         f"ckpt (got dim {rd0}) -- check the paths")
    print(f"[compare] role ckpt dim={role_dim}, dim0 ckpt loaded")

    if args.axes_csv:
        mu, axes = load_axes(args.axes_csv, role_dim)
        print(f"[compare] axes from {args.axes_csv}: {list(axes)}")
    else:
        from render_role_alpha import warmup_axes
        mu, axes = warmup_axes(env, policyX, device)
    if args.axis not in axes:
        raise SystemExit(f"axis {args.axis} not in {list(axes)}")
    u, sg = axes[args.axis]

    # conditions: dim0 + the alpha sweep on the role checkpoint
    conds = ["dim0"] + [f"a{al:+g}" for al in alphas]
    colors = {"dim0": "#f0f0f0"}
    for al in alphas:
        colors[f"a{al:+g}"] = cmap(
            (al - min(alphas)) / max(max(alphas) - min(alphas), 1e-9))

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
            print(f"\n[scene {n_done+1}/{n_target}] traj type {t} ({tname}) "
                  f"map {sid[:10]} focal vid {focal_vid} "
                  f"(human {human_m:.0f} m)", flush=True)

            # Phase 1: all 6 rollouts (no partial files on failure)
            views, mets, ok = {}, {}, True
            for c in conds:
                if c == "dim0":
                    pol, vec = policy0, None
                else:
                    al  = float(c[1:])
                    pol = policyX
                    vec = None if al == 0.0 else al * sg * u
                data = rollout_forced(env, pol, sid, focal_vid, vec, device,
                                      max_tries=args.max_relocate, shift=True)
                if data is None:
                    print(f"  [skip] {sid[:10]} not re-dealt on '{c}'")
                    found[t].remove(sid)
                    ok = False
                    break
                mets[c] = focal_numbers(data)
                view = scene_view(data, data["slots"], sid)
                view["focal"] = int(np.where(
                    data["slots"] == data["focal"])[0][0])
                view["hide_after"] = hide_after_frame(view)
                views[c] = view
                print(f"  {c:>5}: v={mets[c]['speed_mean']:.1f} m/s  "
                      f"|a|={mets[c]['accel_abs']:.1f}  "
                      f"turn={mets[c]['turn_abs']:.2f}", flush=True)
            if not ok:
                continue

            # Phase 2: videos + overlay + per-scene CSV
            try:
                for c in conds:
                    m    = mets[c]
                    tag  = c.replace("+", "p").replace("-", "m")
                    name = f"t{t}_{sid[:10]}_{tag}.mp4"
                    ttl  = (f"traj {tname} | {sid[:10]} | "
                            f"{'no-role (dim0)' if c == 'dim0' else f'{args.axis} α={c[1:]}'}"
                            f" | v={m['speed_mean']:.1f} m/s  "
                            f"|a|={m['accel_abs']:.1f}  turn={m['turn_abs']:.2f}")
                    dist, dmin, fmin, reached = render_condition_video(
                        views[c], views[c]["focal"], colors[c], ttl,
                        out_dir / name, args.fps, args.dpi,
                        gt_background=True, goal_radius=args.goal_radius)
                    mets[c]["goal_min_m"] = dmin
                    mets[c]["reached"]    = int(reached)
                render_compare_overlay(
                    views, conds, colors,
                    f"traj {tname} | {sid[:10]} | dim0 (dashed white) vs "
                    f"{args.axis} α sweep -- same scene, same seat",
                    out_dir / f"overlay_t{t}_{sid[:10]}.png", args.dpi, mets)
                with open(out_dir / f"metrics_t{t}_{sid[:10]}.csv", "w",
                          newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["cond"] + MET_COLS)
                    for c in conds:
                        w.writerow([c] + [mets[c].get(k, np.nan)
                                          for k in MET_COLS])
                for c in conds:
                    summary.append({"traj_type": t, "sid": sid, "cond": c,
                                    **{k: mets[c].get(k, np.nan)
                                       for k in MET_COLS}})
                n_done += 1
            except Exception as e:
                print(f"  [render error {sid[:10]}: {e}] removing partials")
                for fpath in list(out_dir.glob(f"*{sid[:10]}*")):
                    try:
                        fpath.unlink()
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
        print(f"\n[compare] WARNING: types NOT filled after "
              f"{args.max_resamples} resamples: {missing} -- raise "
              f"--max_resamples or lower --min_focal_drive")
    if summary:
        import pandas as pd
        sdf = pd.DataFrame(summary)
        sdf.to_csv(out_dir / "compare_summary.csv", index=False)
        print("\n[compare] === mean speed by traj type x condition ===")
        print(sdf.pivot_table(index="traj_type", columns="cond",
                              values="speed_mean",
                              aggfunc="mean").round(1).to_string())
    print(f"\n[compare] done -> {out_dir}  ({n_done} scenes)")


if __name__ == "__main__":
    main()
