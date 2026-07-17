"""
role_natural_extremes.py -- the OBSERVATIONAL natural-contrast: within one
trajectory cluster, find agents whose NATURAL (unforced) roles sit at the two
ends of PC1, and show how differently they drive.

Complement to the causal paired sweep: forcing asks "can we control behavior
by moving z?"; this asks "does the z the encoder assigns BY ITSELF already
separate drivers?". Observational -- z is obs-conditioned, so scene context
is a confound (deliberately accepted here; scene also shapes human driving).

Phases:
  A) natural rollouts over the map pool -> per (scenario, vehicle): mean
     natural role, PC1/PC2 projection (axes loaded from the paired sweep's
     role_paired_axes.csv so directions match every other dim-2 figure),
     plausibility-masked behavior metrics.
  B) join the trajectory clustering; per requested cluster pick the
     --per_side most-negative and most-positive PC1 agents; render each
     one's NATURAL rollout (no forcing; red/orange collision/offroad flags
     included via render_condition_video) + a side-by-side path contrast
     figure + a contrast table CSV.

Usage (from /scratch/e452103/PufferDrive): see slurm/role_natural_extremes.sbatch.
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
from render_topdown import load_policy, BG
from render_role_conditions import (rollout_forced, scene_view,
                                    hide_after_frame, render_condition_video,
                                    scene_setup, bbox_of, include_point,
                                    first_segment, focal_goal)
from render_role_alpha import load_axes, focal_numbers
from role_regime_analysis import _squeeze
from traj_kinematics import ego_kinematics

T = 91
TELEPORT_M = 4.0
ROLLOUT_SEED = 1234


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--axes_csv",      type=str, required=True,
                   help="role_paired_axes.csv of the SAME checkpoint")
    p.add_argument("--traj_clusters", type=str, required=True)
    p.add_argument("--data_dir",      type=str, required=True)
    p.add_argument("--out_dir",       type=str, required=True)
    p.add_argument("--clusters",      type=str, default="0,1,2,3")
    p.add_argument("--per_side",      type=int, default=1,
                   help="Agents to render per PC1 pole per cluster")
    p.add_argument("--warmup_episodes", type=int, default=4)
    p.add_argument("--map_pool",      type=int, default=256,
                   help="Keep this <= the env instance count (~total_agents/6, "
                        "so all maps are dealt with replacement EVERY reset). "
                        "With a big pool (e.g. 10000) only ~3.4%% of maps are "
                        "live at once and env.reset() re-deals, so Phase B can "
                        "never re-locate a specific extreme agent's scene -- "
                        "every cluster silently skipped. Small pool = the pre-"
                        "picked extremes are always present -> re-deal on the "
                        "first reset.")
    p.add_argument("--total_agents",  type=int, default=2048)
    p.add_argument("--min_margin",    type=float, default=1.5)
    p.add_argument("--min_speed",     type=float, default=0.5,
                   help="m/s; skip agents that never really move")
    p.add_argument("--max_relocate",  type=int, default=120)
    p.add_argument("--seed_start",    type=int, default=100)
    p.add_argument("--goal_radius",   type=float, default=2.0)
    p.add_argument("--device",        type=str, default="cpu")
    p.add_argument("--fps",           type=int, default=10)
    p.add_argument("--dpi",           type=int, default=110)
    return p.parse_args()


def natural_rollout(env, policy, device):
    """One unforced episode: everyone drives on their own live role.
    Returns xs/ys/hs (T,B), roles (T,B,D), gt dict."""
    B = env.num_agents
    obs_np, _ = env.reset()
    gt = env.get_ground_truth_trajectories()
    torch.manual_seed(ROLLOUT_SEED)
    xs = np.zeros((T, B), dtype=np.float32)
    ys = np.zeros((T, B), dtype=np.float32)
    hs = np.zeros((T, B), dtype=np.float32)
    roles = None
    obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    state = policy.initial_state(B, device)
    for t in range(T):
        ag = env.get_global_agent_state()
        xs[t], ys[t], hs[t] = ag["x"], ag["y"], ag["heading"]
        with torch.no_grad():
            logits, _, state, ri = policy(obs, state)
        z = ri["role_z"].cpu().numpy()
        if roles is None:
            roles = np.zeros((T, B, z.shape[-1]), dtype=np.float32)
        roles[t] = z
        action = Categorical(logits=logits.float()).sample()
        obs_np, _, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    return xs, ys, hs, roles, gt


def collect_naturals(env, policy, device, args, mu, u1, u2):
    """Phase A: (sid, vid) -> natural role stats + behavior, across
    warmup_episodes resampled pools. Dedup keeps the first sighting.

    Also returns the FINAL episode's scenario ids: env.reset() replays the
    same allocation (only resample_maps() re-deals), so Phase B can only
    reliably re-locate scenes still in the LAST deal -- picking extremes from
    earlier (resampled-away) episodes made every re-deal fail (~3.4%/fresh
    deal at 10k maps, and rollout_forced only resamples every 25 tries)."""
    rows, seen = [], set()
    live_sids = set()
    for ep in range(args.warmup_episodes):
        if ep > 0:
            env.resample_maps()
        xs, ys, hs, roles, gt = natural_rollout(env, policy, device)
        sids   = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
        vids   = _squeeze(np.asarray(gt["id"])).reshape(-1)
        is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
        step_d = np.hypot(np.diff(xs, axis=0), np.diff(ys, axis=0))
        for a in range(env.num_agents):
            sid = str(sids[a])
            if (not is_veh[a] or not sid or sid.lower().startswith("map")
                    or (sid, int(vids[a])) in seen):
                continue
            jumps = np.where(step_d[:, a] > TELEPORT_M)[0]
            end   = int(jumps[0] + 1) if len(jumps) else T
            if end < 10:
                continue
            spd = step_d[:end - 1, a] * 10.0
            dh  = np.diff(hs[:end, a]) * 10.0
            dh  = (dh + np.pi * 10.0) % (2 * np.pi * 10.0) - np.pi * 10.0
            k   = ego_kinematics(spd, dh[:len(spd)])
            if not np.isfinite(k["speed_mean"]) or k["speed_mean"] < args.min_speed:
                continue
            zbar = roles[:end, a].mean(axis=0)
            cen  = zbar - mu
            rows.append({"sid": sid, "vid": int(vids[a]),
                         "pc1": float(cen @ u1), "pc2": float(cen @ u2),
                         "speed_mean": k["speed_mean"],
                         "accel_abs":  k["accel_abs"],
                         "jerk_abs":   k["jerk_abs"],
                         "turn_abs":   k["turn_abs"],
                         **{f"role_{i}": float(zbar[i])
                            for i in range(len(zbar))}})
            seen.add((sid, int(vids[a])))
        live_sids = {str(s) for s in np.unique(sids)
                     if s and not str(s).lower().startswith("map")}
        print(f"[nat] warmup {ep+1}/{args.warmup_episodes}: "
              f"{len(rows)} (scenario, vehicle) naturals", flush=True)
    return rows, live_sids


def contrast_figure(views, mets, picks, cname, out_path, dpi):
    """Side-by-side: each selected agent's natural path on its own scene.
    Left column(s) = PC1-low, right = PC1-high."""
    n = len(picks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(BG)
    for ax, (tag, sid, vid, pc1) in zip(axes, picks):
        v = views[(sid, vid)]
        f = v["focal"]
        px, py = first_segment(v["xs"][:, f], v["ys"][:, f])
        x0, x1, y0, y1 = bbox_of(v["road_polys"], px[:, None], py[:, None])
        x0, x1, y0, y1 = include_point(x0, x1, y0, y1, focal_goal(v["gt"], f))
        color = "#4477dd" if tag == "lo" else "#dd4444"
        scene_setup(ax, v["road_polys"], v["gt"], x0, x1, y0, y1, focal=f,
                    focal_color=color)
        ax.plot(px, py, color=color, lw=2.6, zorder=6)
        if len(px):
            ax.scatter(px[0], py[0], color="white", s=60, zorder=7)
            ax.scatter(px[-1], py[-1], color=color, s=110, marker="X",
                       zorder=7, edgecolors="white", linewidths=0.8)
        m = mets[(sid, vid)]
        ax.set_title(f"PC1 {'LOW' if tag == 'lo' else 'HIGH'} ({pc1:+.2f}) | "
                     f"{sid[:10]} v{vid}\n"
                     f"v={m['speed_mean']:.1f} m/s  |a|={m['accel_abs']:.2f}  "
                     f"jerk={m['jerk_abs']:.1f}  turn={m['turn_abs']:.2f}  "
                     f"coll={m['event_rate']}  offr={m['offroad_rate']}",
                     color="#e8e8f0", fontsize=9)
    fig.suptitle(f"NATURAL roles, same trajectory type ({cname}): "
                 f"the encoder's own PC1 extremes -- no forcing",
                 color="#e8e8f0", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def main():
    args = parse_args()
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters = [int(c) for c in args.clusters.split(",")]

    import pandas as pd
    tc = pd.read_csv(args.traj_clusters)
    core = tc["margin"] > args.min_margin
    if "is_edge" in tc.columns:
        core &= tc["is_edge"] == 0
    tc = tc[core]
    cluster_of = {(str(r.scenario_id), int(r.vehicle_id)): int(r.cluster)
                  for r in tc.itertuples()}
    print(f"[nat] {len(cluster_of)} core (scenario, vehicle) cluster labels")

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(num_maps=args.map_pool, num_agents=args.total_agents,
                map_dir=args.data_dir, episode_length=T, goal_speed=100,
                seed=args.seed_start)
    obs_probe, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs_probe.shape[-1], device)
    if role_dim == 0:
        raise SystemExit("role_dim=0 checkpoint -- no role to contrast")
    mu, axes = load_axes(args.axes_csv, role_dim)
    if "PC1" not in axes:
        raise SystemExit(f"axes csv needs PC1, has {list(axes)}")
    u1, _ = axes["PC1"]
    u2, _ = axes.get("PC2", (np.zeros_like(u1), 0.0))

    # -- Phase A ---------------------------------------------------------------
    rows, live_sids = collect_naturals(env, policy, device, args, mu, u1, u2)
    df = pd.DataFrame(rows)
    df["cluster"] = [cluster_of.get((r.sid, r.vid), -1) for r in df.itertuples()]
    df.to_csv(out_dir / "natural_agents.csv", index=False)
    lab = df[df["cluster"] >= 0]
    print(f"[nat] {len(lab)}/{len(df)} naturals carry a core cluster label")

    # -- Selection + Phase B renders --------------------------------------------
    selected = []
    for k in clusters:
        sub_all = lab[lab["cluster"] == k].sort_values("pc1")
        # Prefer scenes still in the LIVE (final-episode) allocation so the
        # Phase B re-deal succeeds on the first reset; fall back to the full
        # candidate set only if the live subset is too thin.
        sub = sub_all[sub_all["sid"].isin(live_sids)]
        if len(sub) < 2 * args.per_side:
            print(f"[nat] cluster {k}: only {len(sub)} LIVE labeled naturals; "
                  f"falling back to all {len(sub_all)} (re-deal may be slow)")
            sub = sub_all
        if len(sub) < 2 * args.per_side:
            print(f"[nat] cluster {k}: only {len(sub)} labeled naturals -- skipped")
            continue
        picks = ([("lo", r.sid, r.vid, r.pc1)
                  for r in sub.head(args.per_side).itertuples()] +
                 [("hi", r.sid, r.vid, r.pc1)
                  for r in sub.tail(args.per_side).itertuples()])
        print(f"\n[nat] cluster {k}: PC1 spans {sub['pc1'].min():+.2f} .. "
              f"{sub['pc1'].max():+.2f} over {len(sub)} agents")

        views, mets, ok = {}, {}, True
        for tag, sid, vid, pc1 in picks:
            print(f"[nat] c{k} {tag} {sid[:10]} v{vid} (pc1={pc1:+.2f}) "
                  f"-- natural rollout", flush=True)
            data = rollout_forced(env, policy, sid, vid, None, device,
                                  max_tries=args.max_relocate)
            if data is None:
                print(f"  [skip] {sid[:10]} never re-dealt -- cluster {k} "
                      f"contrast incomplete")
                ok = False
                break
            m = focal_numbers(data)
            view = scene_view(data, data["slots"], sid)
            view["focal"] = int(np.where(data["slots"] == data["focal"])[0][0])
            view["hide_after"] = hide_after_frame(view)
            views[(sid, vid)] = view
            mets[(sid, vid)]  = m
            fn = out_dir / f"nat_c{k}_{tag}_{sid[:10]}_v{vid}.mp4"
            render_condition_video(
                view, view["focal"],
                "#4477dd" if tag == "lo" else "#dd4444",
                f"cluster {k} | NATURAL role, PC1={pc1:+.2f} ({tag}) | "
                f"{sid[:10]} v{vid} | v={m['speed_mean']:.1f} m/s",
                fn, args.fps, args.dpi, goal_radius=args.goal_radius)
            selected.append({"cluster": k, "pole": tag, "sid": sid,
                             "vid": vid, "pc1": pc1, **m})
        if ok:
            contrast_figure(views, mets, picks, f"C{k}",
                            out_dir / f"contrast_c{k}.png", args.dpi)

    env.close()
    if selected:
        pd.DataFrame(selected).to_csv(out_dir / "natural_extremes_selected.csv",
                                      index=False)
        print("\n[nat] === selected extremes ===")
        print(pd.DataFrame(selected)[
            ["cluster", "pole", "sid", "vid", "pc1", "speed_mean",
             "accel_abs", "jerk_abs", "turn_abs", "event_rate",
             "offroad_rate"]].to_string(index=False))
    print(f"\n[nat] done -> {out_dir}")


if __name__ == "__main__":
    main()
