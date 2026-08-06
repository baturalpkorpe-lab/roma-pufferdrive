"""
render_regime_video.py -- videos of a POLICY rollout where every vehicle is
coloured by the situation it is in RIGHT NOW, so cluster changes are visible
frame by frame.

Every agent drives on its own NATURAL role -- nothing is forced. What changes
during the clip is the SITUATION each car is in, and the colour changes with it:
a car crossing from green to blue has just acquired a leader; green to orange
has entered a junction with someone to negotiate against.

This is the moving version of render_regime_strip. That one argues the idea on a
single human trajectory; this one shows the labelling holding up across a whole
scene of interacting agents under a real policy.

Colours match render_regime_strip exactly, including one orange tone per
right-of-way class, so a slide can mix the two without re-explaining the key.

Zones are read from junction_control_audit.csv rather than re-derived, so the
video is labelled by the SAME zones, radii and control classes as every number
in the report.

Usage:
    python render_regime_video.py \
        --checkpoint /scratch/$USER/checkpoints/roma_mifutagent_dim1_H8_r16p64_ep_nodiv/roma_dim1_final.pt \
        --zones_csv  /scratch/$USER/regimes/junction_control_audit.csv \
        --out_dir    /scratch/$USER/regime_videos --n_videos 10
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conflict_metrics as CM
from render_regime_strip import (ZONE_TONE, ZONE_NAME, COL, LABEL,
                                 lab_color, lab_text, regime_per_step)
from render_topdown import (load_policy, rollout, vehicle_corners,
                            normalize_polylines, BG)

T = 91
TELEPORT_M = 4.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--zones_csv", required=True)
    p.add_argument("--data_dir",
                   default="pufferlib/resources/drive/binaries/training")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_videos", type=int, default=10)
    p.add_argument("--map_pool", type=int, default=64)
    p.add_argument("--total_agents", type=int, default=512)
    p.add_argument("--min_vehicles", type=int, default=4,
                   help="skip scenes with fewer tracked vehicles -- a video of "
                        "two cars makes a poor argument")
    p.add_argument("--min_switches", type=int, default=3,
                   help="total regime changes across all vehicles in the scene")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_zones(path):
    Z = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["control"] == "not_a_junction":
                continue
            try:
                Z[str(r["scenario_id"])].append(dict(
                    centre=np.array([float(r["cx"]), float(r["cy"])]),
                    R=float(r.get("zone_r") or 25.0), control=r["control"]))
            except (ValueError, KeyError):
                continue
    return Z


def scene_labels(tracks, zones):
    """(n_veh, T) labels + the conflict points, for one scene."""
    D = CM.dense_scene(tracks, T)
    lead = CM.leaders_dense(D)
    labs, cpts = [], []
    for e, tr in enumerate(tracks):
        czones = set()
        for f, ot in enumerate(tracks):
            if f == e:
                continue
            shared = [zi for zi, z in enumerate(zones)
                      if np.hypot(tr["x"] - z["centre"][0],
                                  tr["y"] - z["centre"][1]).min() <= z["R"]
                      and np.hypot(ot["x"] - z["centre"][0],
                                   ot["y"] - z["centre"][1]).min() <= z["R"]]
            if not shared:
                continue
            cp = CM.conflict_point(tr, ot, 2.0)
            if cp is None:
                continue
            kind, _ = CM.classify_conflict(tr, ot, cp)
            if kind is None:
                continue
            for zi in shared:
                if np.hypot(*(cp["pt"] - zones[zi]["centre"])) <= zones[zi]["R"]:
                    czones.add(zi)
            if e < f:
                cpts.append(cp["pt"])
        labs.append(regime_per_step(D, e, lead[e], zones, czones))
    return D, np.array(labs, dtype=object), cpts


def render(D, labs, tracks, zones, cpts, roads_xy, out, fps, dpi, sid):
    n = len(tracks)
    xs, ys = D["X"], D["Y"]
    ok = np.isfinite(xs)
    pad = 20
    xlo, xhi = np.nanmin(xs[ok]) - pad, np.nanmax(xs[ok]) + pad
    ylo, yhi = np.nanmin(ys[ok]) - pad, np.nanmax(ys[ok]) + pad

    fig, ax = plt.subplots(figsize=(10, 10 * (yhi - ylo) / max(xhi - xlo, 1e-6)))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for px, py in roads_xy:
        ax.plot(px, py, color="#3a3a3a", lw=0.9, zorder=0)
    for z in zones:
        tone = ZONE_TONE.get(z["control"], ZONE_TONE["uncontrolled"])[0]
        ax.add_patch(Circle(z["centre"], z["R"], fill=False, ls="--", lw=1.2,
                            ec=tone, alpha=.65, zorder=1))
        ax.annotate(ZONE_NAME.get(z["control"], z["control"]),
                    (z["centre"][0], z["centre"][1] + z["R"]), color=tone,
                    fontsize=7.5, ha="center", va="bottom", alpha=.9)
    for c in cpts:
        ax.plot(c[0], c[1], "x", ms=8, mew=1.8, color="#e74c3c", zorder=2)

    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ttl = ax.set_title("", color="w", fontsize=11)

    keys = ["freeflow", "following"] + [f"conf:{c}" for c in ZONE_TONE] + \
           [f"zone:{c}" for c in ZONE_TONE]
    present = [k for k in keys if any((l == k).any() for l in labs)]
    ax.legend([plt.Line2D([], [], color=lab_color(k), lw=6) for k in present],
              [lab_text(k) for k in present], loc="upper right", fontsize=7,
              facecolor="#1b1b1b", edgecolor="#555", labelcolor="w",
              framealpha=.9)

    bodies = [Polygon(np.zeros((4, 2)), closed=True, zorder=4) for _ in range(n)]
    for b in bodies:
        ax.add_patch(b)

    # vehicle_corners is vectorised over agents -- (N,) arrays in, (N,4,2) out --
    # so it is called ONCE per frame for the whole scene, not per vehicle.
    L = np.array([tr["length"] for tr in tracks], float)
    W = np.array([tr["width"] for tr in tracks], float)

    def update(t):
        live = 0
        cor = vehicle_corners(np.nan_to_num(xs[:, t]), np.nan_to_num(ys[:, t]),
                              np.nan_to_num(D["H"][:, t]), L, W)
        for e, tr in enumerate(tracks):
            b = bodies[e]
            if not D["M"][e][t]:
                b.set_visible(False)
                continue
            b.set_visible(True); live += 1
            b.set_xy(cor[e])
            k = labs[e][t]
            b.set_facecolor(lab_color(k))
            b.set_edgecolor("w" if not str(k).startswith("zone:") else "#888")
            b.set_linewidth(0.7)
        ttl.set_text(f"{sid[:10]}   t = {t/10:.1f}s   {live} vehicles   "
                     f"(colour = situation right now, natural roles)")
        return bodies + [ttl]

    ani = animation.FuncAnimation(fig, update, frames=T, blit=False)
    if str(out).endswith(".mp4") and animation.FFMpegWriter.isAvailable():
        w = animation.FFMpegWriter(fps=fps, bitrate=2600)
    else:
        out = Path(str(out).replace(".mp4", ".gif"))
        w = animation.PillowWriter(fps=fps)
    ani.save(str(out), writer=w, dpi=dpi, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if (args.device != "cuda"
                                       or torch.cuda.is_available()) else "cpu")
    zones_by_sid = load_zones(args.zones_csv)
    print(f"[vid] zones for {len(zones_by_sid)} scenes  device={dev}")

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(num_maps=args.map_pool, num_agents=args.total_agents,
                map_dir=args.data_dir, episode_length=T, goal_speed=100)
    probe, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, probe.shape[-1], dev)
    print(f"[vid] role_dim={role_dim}  (natural roles, nothing forced)")

    from role_regime_analysis import _squeeze
    made, tries = 0, 0
    while made < args.n_videos and tries < args.n_videos * 8:
        tries += 1
        if tries > 1:
            env.resample_maps()
        data = rollout(env, policy, role_dim, dev, forced_fn=None)
        gt = data["gt"]
        sids = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
        vids = _squeeze(np.asarray(gt["id"])).reshape(-1)
        isv = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
        by_sid = defaultdict(list)
        for a in range(len(vids)):
            s = str(sids[a])
            if isv[a] and s and not s.lower().startswith("map"):
                by_sid[s].append(a)

        # the scene with the most vehicles that we actually have zones for
        cand = sorted((s for s in by_sid if zones_by_sid.get(s)),
                      key=lambda s: -len(by_sid[s]))
        for sid in cand:
            if made >= args.n_videos:
                break
            keep = np.zeros(len(vids), bool); keep[by_sid[sid]] = True
            tracks = CM.tracks_from_arrays(data["xs"], data["ys"], data["hs"],
                                           vids, keep, TELEPORT_M,
                                           widths=data["width"],
                                           lengths=data["length"])
            if len(tracks) < args.min_vehicles:
                continue
            zones = zones_by_sid[sid]
            D, labs, cpts = scene_labels(tracks, zones)
            sw = int(sum((l[1:] != l[:-1]).sum() for l in labs))
            if sw < args.min_switches:
                continue
            # get_road_edge_polylines() returns a DICT of flattened x/y plus
            # per-polyline lengths, not a list of (K,2) arrays -- np.asarray on
            # it yields the key strings and dies on float('x'). render_topdown
            # already has the parser for every shape this can take.
            try:
                roads_xy = [(pl[:, 0], pl[:, 1])
                            for pl in normalize_polylines(data["road_edges"])]
            except Exception as ex:
                print(f"  [road edges unusable: {type(ex).__name__}: {ex}]")
                roads_xy = []          # roads are decoration; never fatal
            out = out_dir / f"regime_{sid[:10]}.mp4"
            try:
                out = render(D, labs, tracks, zones, cpts, roads_xy, out,
                             args.fps, args.dpi, sid)
            except Exception as ex:
                print(f"  skip {sid[:10]}: {type(ex).__name__}: {ex}")
                continue
            made += 1
            print(f"  [{made}/{args.n_videos}] {out.name}  "
                  f"{len(tracks)} vehicles  {len(zones)} zones  "
                  f"{sw} regime changes", flush=True)
            break

    print(f"\n  wrote {made} video(s) -> {out_dir}")
    if made < args.n_videos:
        print("  fewer than asked -- lower --min_vehicles/--min_switches, or "
              "raise --map_pool so more scenes have audited zones")


if __name__ == "__main__":
    main()
