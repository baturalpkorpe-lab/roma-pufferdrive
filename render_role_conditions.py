"""
render_role_conditions.py -- Controlled role-condition videos (Phase C follow-up).

For each map regime: pick 2 core maps, pick one FOCAL agent (the one whose
human counterpart drives the most), and roll the SAME scene three times with
the focal agent's role forced to that regime's cluster-mean role vectors
(conformist / middle / runaway, identical k-means as the Phase C spider
charts). All other agents keep their natural encoder roles; torch RNG is
re-seeded identically per condition, so the ONLY difference between the three
rollouts is the focal agent's role vector.

Outputs per (regime, map):
  3 videos  r{regime}_map{seed}_{conformist|middle|runaway}.mp4
            focal agent = large colored box + trail, others gray,
            human GT paths dashed
  1 overlay overlay_r{regime}_map{seed}.png
            the focal agent's three trajectories on one image -- path
            divergence at a glance

Default: 4 regimes x 2 maps x 3 conditions = 24 videos + 8 overlays.

Usage (from /scratch/e452103/PufferDrive):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/render_role_conditions.py \
        --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_step3000238080.pt \
        --clusters   /scratch/e452103/map_atlas/map_clusters.csv \
        --agent_data /scratch/e452103/role_regime/dim4/phaseC_agent_data.csv \
        --data_dir   pufferlib/resources/drive/binaries/training \
        --out_dir    /scratch/e452103/renders/role_conditions_dim4
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection, PolyCollection
from torch.distributions import Categorical

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import (normalize_polylines, vehicle_corners,
                            load_policy, BG, ROAD, GTC)
from role_regime_analysis import _squeeze

T = 91
CONDITIONS = ["conformist", "middle", "runaway"]
COND_COLORS = {"conformist": "#4477aa", "middle": "#ee8833",
               "runaway": "#33aa44"}
REGIME_NAMES = {0: "quiet_local", 1: "fast_corridors",
                2: "parking_low_speed", 4: "congested_urban"}
ROLLOUT_SEED = 1234   # identical action noise across the 3 conditions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--clusters",   type=str, required=True)
    p.add_argument("--agent_data", type=str, required=True,
                   help="phaseC_agent_data.csv of the SAME checkpoint")
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--maps_per_regime", type=int, default=2)
    p.add_argument("--num_agents", type=int, default=32)
    p.add_argument("--min_margin", type=float, default=1.5)
    p.add_argument("--seed_start", type=int, default=100)
    p.add_argument("--max_probes", type=int, default=400)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--fps",        type=int, default=10)
    p.add_argument("--dpi",        type=int, default=110)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Role conditions: identical k-means to the Phase C spider charts
# ---------------------------------------------------------------------------

def cluster_role_conditions(agent_csv):
    import pandas as pd
    from sklearn.cluster import KMeans
    df = pd.read_csv(agent_csv)
    role_cols = [c for c in df.columns if c.startswith("role_")]
    out = {}
    for rg, sub in df.groupby("regime"):
        sub = sub.dropna(subset=["d_speed"])
        km  = KMeans(n_clusters=3, n_init=10, random_state=42)
        lab = km.fit_predict(sub[role_cols].values)
        d_speed = np.array([sub["d_speed"][lab == c].mean() for c in range(3)])
        order   = np.argsort(d_speed)             # conformist -> runaway
        out[int(rg)] = {CONDITIONS[rank]:
                        (km.cluster_centers_[c], float(d_speed[c]))
                        for rank, c in enumerate(order)}
    return out, len(role_cols)


# ---------------------------------------------------------------------------
# Map probing: hop maps with env.resample_maps() (the seed does NOT select
# the map when num_maps=1 -- recreating the env with a new seed loads the
# same scenario again)
# ---------------------------------------------------------------------------

def check_map(env, regime_of, need, found, used_sids):
    """Inspect the currently loaded map. Returns (sid, rg, focal) or None."""
    env.reset()
    gt  = env.get_ground_truth_trajectories()
    sid = _squeeze(np.asarray(gt["scenario_id"]).astype(str))[0]
    rg  = regime_of.get(sid)
    if rg is None or rg not in need or sid in used_sids \
            or len(found[rg]) >= need[rg]:
        return None
    gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
    valid  = _squeeze(gt["valid"]).astype(bool)
    is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
    # focal = vehicle whose human drives the farthest
    lens = np.zeros(gx.shape[0])
    for a in range(gx.shape[0]):
        if not is_veh[a] or valid[a].sum() < 10:
            continue
        px, py = gx[a][valid[a]], gy[a][valid[a]]
        lens[a] = np.hypot(np.diff(px), np.diff(py)).sum()
    focal = int(np.argmax(lens))
    if lens[focal] < 10.0:                        # humans barely move: skip
        return None
    return sid, int(rg), focal, float(lens[focal])


# ---------------------------------------------------------------------------
# One controlled rollout: forced role on the focal agent only
# ---------------------------------------------------------------------------

def rollout_forced(env, policy, focal, cond_vec, device):
    B = env.num_agents
    obs_np, _ = env.reset()
    road = None
    gt   = None
    try:
        road = env.get_road_edge_polylines()
        gt   = env.get_ground_truth_trajectories()
    except Exception as e:
        print(f"  [scene data: {e}]")

    torch.manual_seed(ROLLOUT_SEED)              # identical noise per condition
    xs = np.zeros((T, B), dtype=np.float32)
    ys = np.zeros((T, B), dtype=np.float32)
    hs = np.zeros((T, B), dtype=np.float32)
    length = np.full(B, 4.5, dtype=np.float32)
    width  = np.full(B, 2.0, dtype=np.float32)

    cond = torch.as_tensor(cond_vec, dtype=torch.float32, device=device)
    obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    state = policy.initial_state(B, device)

    for t in range(T):
        ag = env.get_global_agent_state()
        xs[t], ys[t], hs[t] = ag["x"], ag["y"], ag["heading"]
        if t == 0:
            for k, arr in (("length", length), ("width", width)):
                try:
                    v = np.asarray(ag[k], dtype=np.float32).reshape(-1)
                    if v.shape[0] == B and (v > 0).all():
                        arr[:] = v
                except Exception:
                    pass
        with torch.no_grad():
            # pass 1: natural roles for everyone (state NOT advanced)
            _, _, _, ri = policy(obs, state)
            forced = ri["role_z"].clone()
            forced[focal] = cond
            # pass 2: same input state, focal role replaced
            logits, _, state, _ = policy(obs, state, forced_role=forced)
        action = Categorical(logits=logits.float()).sample()
        obs_np, _, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

    return {"xs": xs, "ys": ys, "hs": hs, "length": length, "width": width,
            "road_edges": road, "gt": gt}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def scene_setup(ax, polys, gt, x0, x1, y0, y1):
    ax.set_facecolor(BG)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.tick_params(colors="#888899", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#333344")
    if polys:
        ax.add_collection(LineCollection(polys, colors=ROAD, linewidths=0.8))
    if isinstance(gt, dict):
        try:
            gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
            gv     = _squeeze(gt["valid"]).astype(bool)
            segs = []
            for a in range(gx.shape[0]):
                pts = np.stack([gx[a][gv[a]], gy[a][gv[a]]], axis=-1)
                if len(pts) >= 2 and np.hypot(*np.diff(pts, axis=0).T).sum() > 2:
                    segs.append(pts)
            if segs:
                ax.add_collection(LineCollection(
                    segs, colors=GTC, linewidths=0.7,
                    linestyles=(0, (3, 3)), alpha=0.35))
        except Exception:
            pass


def bbox_of(polys, xs, ys):
    if polys:
        allp = np.concatenate(polys, axis=0)
        x0, x1 = allp[:, 0].min(), allp[:, 0].max()
        y0, y1 = allp[:, 1].min(), allp[:, 1].max()
    else:
        x0, x1 = np.percentile(xs, [2, 98])
        y0, y1 = np.percentile(ys, [2, 98])
    mx, my = 0.03 * (x1 - x0 + 1), 0.03 * (y1 - y0 + 1)
    return x0 - mx, x1 + mx, y0 - my, y1 + my


def render_condition_video(data, focal, color, title, out_path, fps, dpi):
    xs, ys, hs = data["xs"], data["ys"], data["hs"]
    B = xs.shape[1]
    polys = normalize_polylines(data["road_edges"])
    x0, x1, y0, y1 = bbox_of(polys, xs, ys)

    w, h = x1 - x0, y1 - y0
    fw = 10.0 if w >= h else max(10.0 * w / h, 4)
    fh = 10.0 if h >= w else max(10.0 * h / w, 4)
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(BG)
    scene_setup(ax, polys, data["gt"], x0, x1, y0, y1)
    ax.set_title(title, color="#e8e8f0", fontsize=10)

    others = PolyCollection([], facecolors=(0.55, 0.55, 0.60, 1.0),
                            edgecolors="#000000", linewidths=0.3, zorder=4)
    focal_c = PolyCollection([], facecolors=color, edgecolors="#ffffff",
                             linewidths=0.8, zorder=6)
    trail = LineCollection([], colors=color, linewidths=2.0, alpha=0.8,
                           zorder=5)
    ax.add_collection(others)
    ax.add_collection(focal_c)
    ax.add_collection(trail)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="#e8e8f0",
                  fontsize=10, va="top", family="monospace")

    def update(t):
        vis = ((xs[t] > x0) & (xs[t] < x1) & (ys[t] > y0) & (ys[t] < y1))
        idx = np.array([a for a in np.where(vis)[0] if a != focal], dtype=int)
        others.set_verts(list(vehicle_corners(
            xs[t, idx], ys[t, idx], hs[t, idx],
            data["length"][idx], data["width"][idx])) if len(idx) else [])
        focal_c.set_verts(list(vehicle_corners(
            xs[t, focal:focal+1], ys[t, focal:focal+1], hs[t, focal:focal+1],
            data["length"][focal:focal+1] * 1.15,
            data["width"][focal:focal+1] * 1.15)))
        s = max(0, t - 40)
        px, py = xs[s:t+1, focal], ys[s:t+1, focal]
        jump = np.hypot(np.diff(px), np.diff(py)) > 8.0
        cut  = np.where(jump)[0]
        start = cut[-1] + 1 if len(cut) else 0
        trail.set_segments([np.stack([px[start:], py[start:]], axis=-1)]
                           if t - s - start >= 1 else [])
        txt.set_text(f"t = {t:2d}/{T-1}  ({t/10:.1f}s)")
        return others, focal_c, trail, txt

    ani = animation.FuncAnimation(fig, update, frames=T, blit=False)
    writer = (animation.FFMpegWriter(fps=fps, bitrate=2500)
              if animation.FFMpegWriter.isAvailable()
              else animation.PillowWriter(fps=fps))
    if not animation.FFMpegWriter.isAvailable():
        out_path = out_path.with_suffix(".gif")
    ani.save(str(out_path), writer=writer, dpi=dpi,
             savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def render_overlay(datas, focal, title, out_path, dpi):
    """datas: {condition: data}. One image, three focal trajectories."""
    ref   = datas[CONDITIONS[0]]
    polys = normalize_polylines(ref["road_edges"])
    all_x = np.concatenate([d["xs"][:, focal] for d in datas.values()])
    all_y = np.concatenate([d["ys"][:, focal] for d in datas.values()])
    x0, x1, y0, y1 = bbox_of(polys, all_x[:, None], all_y[:, None])

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor(BG)
    scene_setup(ax, polys, ref["gt"], x0, x1, y0, y1)
    for cond in CONDITIONS:
        d = datas[cond]
        px, py = d["xs"][:, focal], d["ys"][:, focal]
        jump = np.hypot(np.diff(px), np.diff(py)) > 8.0
        end  = int(np.where(jump)[0][0] + 1) if jump.any() else T
        ax.plot(px[:end], py[:end], color=COND_COLORS[cond], lw=2.2,
                alpha=0.95, label=cond, zorder=5)
        ax.scatter(px[end-1], py[end-1], color=COND_COLORS[cond], s=90,
                   marker="X", zorder=6, edgecolors="white", linewidths=0.7)
    ax.scatter(ref["xs"][0, focal], ref["ys"][0, focal], color="white",
               s=70, zorder=6, label="start")
    ax.legend(fontsize=9, loc="best")
    ax.set_title(title, color="#e8e8f0", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor=BG)
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

    import pandas as pd
    cl = pd.read_csv(args.clusters)
    cl = cl[(cl["margin"] > args.min_margin) & (cl["cluster"].isin(REGIME_NAMES))]
    regime_of = dict(zip(cl["scenario_id"].astype(str), cl["cluster"]))

    conds, role_dim = cluster_role_conditions(args.agent_data)
    for rg, cc in sorted(conds.items()):
        print(f"[conditions] regime {rg} ({REGIME_NAMES.get(rg)}): " +
              "  ".join(f"{n}: dv={dv:+.1f}" for n, (_, dv) in cc.items()))

    # One env for the whole job; hop maps with resample_maps() (a new seed
    # does NOT load a new map when num_maps=1). The env stays on the accepted
    # map while its 3 conditions run: env.reset() replays the same scenario,
    # only resample_maps() changes it.
    from pufferlib.ocean.drive.drive import Drive
    env = Drive(num_maps=1, num_agents=args.num_agents,
                map_dir=args.data_dir, episode_length=T,
                seed=args.seed_start)
    obs_probe, _ = env.reset()
    policy, ckpt_dim = load_policy(args.checkpoint, obs_probe.shape[-1],
                                   device)
    if ckpt_dim != role_dim:
        raise SystemExit(f"checkpoint role_dim={ckpt_dim} but agent_data has "
                         f"{role_dim} role columns -- wrong pairing")

    need      = {rg: args.maps_per_regime for rg in REGIME_NAMES}
    found     = {rg: [] for rg in REGIME_NAMES}
    used_sids = set()
    seen_sids = set()
    n_done    = 0
    n_target  = args.maps_per_regime * len(REGIME_NAMES)

    print(f"\n[probe] hopping maps via resample_maps() "
          f"(target {n_target} maps) ...")
    for probe in range(args.max_probes):
        if n_done >= n_target:
            break
        if probe > 0:
            env.resample_maps()
        hit = check_map(env, regime_of, need, found, used_sids)
        gt  = env.get_ground_truth_trajectories()
        seen_sids.add(_squeeze(np.asarray(gt["scenario_id"]).astype(str))[0])
        if probe == 30 and len(seen_sids) <= 2:
            print("[probe] WARNING: resample_maps() is not changing the map "
                  "-- aborting early; only the maps found so far will render")
            break
        if hit is None:
            continue
        sid, rg, focal, human_m = hit
        found[rg].append(sid)
        used_sids.add(sid)
        n_done += 1
        print(f"[probe {probe}] regime {rg} ({REGIME_NAMES[rg]}) map "
              f"{sid[:10]} focal {focal} (human drives {human_m:.0f} m)  "
              f"[{n_done}/{n_target}]", flush=True)

        datas = {}
        for cond in CONDITIONS:
            vec, dv = conds[rg][cond]
            print(f"[render] regime {rg} map {sid[:10]} focal {focal} "
                  f"cond {cond} (dv={dv:+.1f} m/s)")
            datas[cond] = rollout_forced(env, policy, focal, vec, device)
            name  = f"r{rg}_{REGIME_NAMES[rg]}_{sid[:10]}_{cond}.mp4"
            title = (f"{REGIME_NAMES[rg]} | map {sid[:10]} | focal role = "
                     f"{cond} ({dv:+.1f} m/s vs human in Phase C)")
            render_condition_video(datas[cond], focal, COND_COLORS[cond],
                                   title, out_dir / name, args.fps, args.dpi)
        render_overlay(datas, focal,
                       f"{REGIME_NAMES[rg]} | map {sid[:10]} | same scene, "
                       f"same noise -- only the focal role differs",
                       out_dir / f"overlay_r{rg}_{sid[:10]}.png", args.dpi)

    env.close()
    missing = {REGIME_NAMES[rg]: need[rg] - len(found[rg])
               for rg in need if len(found[rg]) < need[rg]}
    if missing:
        print(f"[probe] note: not all regimes filled after {args.max_probes} "
              f"probes, missing: {missing}")
    print(f"\n[render] done -> {out_dir}")


if __name__ == "__main__":
    main()
