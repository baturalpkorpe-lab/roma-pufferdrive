"""
render_intersection_maps.py -- look at what the intersection labeller decided.

Draws real scenes with the road geometry, the junction markers, the detected
zones, and EVERY trajectory coloured by its own label. Two synthetic tests in
a row passed while being wrong on real data (road_rel_turn came out backwards;
the gap-based axis test rejected 99% of zones), so this exists to check the
labels against something that cannot be faked: the map itself.

It also answers directly whether the label is per-map or per-trajectory. It is
per-TRAJECTORY: one scene holds ~11.6 vehicles but only ~1.5 active junctions,
so a car on a side street is in the same map and not involved. Each panel
prints its own in/out counts and the summary reports how many rendered maps
were MIXED.

Colours:
    grey thin      lane centerlines
    grey thick     road edges
    red triangle   stop sign        blue square  crosswalk
    green circle   ACTIVE zone (radius = --zone_radius, the labelling reach)
    grey dashed    zone rejected as inactive or single-axis
    RED path       intersection = 1     BLUE path  intersection = 0
    faint grey     parked (never labelled)
    dot            trajectory start

Usage:
    python render_intersection_maps.py \
        --data_dir /scratch/$USER/PufferDrive/pufferlib/resources/drive/binaries/training \
        --out_dir  /scratch/$USER/intersection_maps --n 20 --zone_radius 15
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from map_binary import read_map_binary
from intersection_features import scene_rows, junction_zones, MOVE_MS, HZ


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir",  default="intersection_maps")
    p.add_argument("--n",        type=int, default=20, help="maps to render")
    p.add_argument("--scan",     type=int, default=400,
                   help="maps to scan when picking interesting ones")
    p.add_argument("--pick",     choices=["mixed", "first", "involved"],
                   default="mixed",
                   help="mixed = prefer scenes holding BOTH labels (most "
                        "informative); involved = any scene with a label; "
                        "first = just the first N")
    # must match the run that produced the CSV
    p.add_argument("--zone_eps",       type=float, default=20.0)
    p.add_argument("--zone_radius",    type=float, default=15.0)
    p.add_argument("--min_traversals", type=int,   default=1)
    p.add_argument("--min_axes",       type=int,   default=2)
    p.add_argument("--use_lane_crossings", type=int, default=0)
    return p.parse_args()


def draw(m, rows, active, centres, a, path):
    fig, ax = plt.subplots(figsize=(11, 11))

    for rd in m["roads"]:
        if rd["type"] == 4:
            ax.plot(rd["x"], rd["y"], color="0.80", lw=0.8, zorder=1)
        elif rd["type"] == 6:
            ax.plot(rd["x"], rd["y"], color="0.45", lw=1.8, zorder=2)
        elif rd["type"] == 5:
            ax.plot(rd["x"], rd["y"], color="0.88", lw=0.6, ls=":", zorder=1)
        elif rd["type"] == 7:
            ax.plot(rd["x"][0], rd["y"][0], "v", color="crimson", ms=9,
                    zorder=6, mec="k", mew=.4)
        elif rd["type"] == 8:
            ax.fill(rd["x"], rd["y"], color="royalblue", alpha=.35, zorder=3)

    act = {tuple(np.round(c, 3)) for c in active}
    for c in centres:
        on = tuple(np.round(c, 3)) in act
        ax.add_patch(plt.Circle(c, a.zone_radius,
                                fill=on, alpha=.16 if on else 1.0,
                                color="seagreen" if on else "0.6",
                                ls="-" if on else "--",
                                lw=2.0 if on else 1.0, zorder=4))
        ax.plot(*c, "+", color="seagreen" if on else "0.6", ms=9, zorder=5)

    lab = {int(r["vehicle_id"]): int(r["intersection"]) for r in rows}
    n_in = n_out = n_park = 0
    for ob in m["objects"]:
        if ob["type"] != 1:
            continue
        v = ob["valid"]
        if v.sum() < 2:
            continue
        x, y = ob["x"][v], ob["y"][v]
        vid = int(ob["id"])
        if vid not in lab:                       # parked / too short
            ax.plot(x, y, color="0.75", lw=1.0, alpha=.5, zorder=7)
            n_park += 1
            continue
        involved = lab[vid]
        col = "red" if involved else "royalblue"
        n_in += involved; n_out += (1 - involved)
        ax.plot(x, y, color=col, lw=2.2, alpha=.85, zorder=8)
        ax.plot(x[0], y[0], "o", color=col, ms=5, zorder=9)
        ax.annotate(str(vid), (x[0], y[0]), fontsize=6, color=col,
                    xytext=(3, 3), textcoords="offset points", zorder=9)

    ax.set_aspect("equal")
    ax.set_title(
        f"{m['scenario_id']}   zones {len(active)} active / {len(centres)} found"
        f"   |   RED intersection={n_in}   BLUE not={n_out}   grey parked={n_park}",
        fontsize=10)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return n_in, n_out


def main():
    a = parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.data_dir).glob("map_*.bin"))
    if not files:
        raise SystemExit(f"no map_*.bin in {a.data_dir}")

    cfg = SimpleNamespace(zone_eps=a.zone_eps, zone_radius=a.zone_radius,
                          min_traversals=a.min_traversals, min_axes=a.min_axes,
                          use_lane_crossings=a.use_lane_crossings)

    print(f"[render] scanning {min(a.scan, len(files))} maps for --pick {a.pick}")
    cand = []
    for fp in files[:a.scan]:
        try:
            m = read_map_binary(fp)
            rows, _, active, centres = scene_rows(m, cfg)
        except Exception:
            continue
        if not rows:
            continue
        n_in = sum(int(r["intersection"]) for r in rows)
        n_out = len(rows) - n_in
        score = (min(n_in, n_out), n_in) if a.pick == "mixed" else (n_in,)
        if a.pick == "first" or (a.pick == "involved" and n_in) \
           or (a.pick == "mixed" and n_in):
            cand.append((score, fp, m, rows, active, centres, n_in, n_out))
    if not cand:
        raise SystemExit("no scenes with intersection-labelled trajectories")

    cand.sort(key=lambda t: t[0], reverse=(a.pick != "first"))
    sel = cand[:a.n]

    mixed = 0
    print(f"\n  {'file':<14} {'scenario':<18} {'zones':>10}  {'in':>4} {'out':>4}  mix")
    for _, fp, m, rows, active, centres, n_in, n_out in sel:
        png = out / f"{fp.stem}_{m['scenario_id'][:12]}.png"
        draw(m, rows, active, centres, a, png)
        is_mixed = n_in > 0 and n_out > 0
        mixed += is_mixed
        print(f"  {fp.name:<14} {m['scenario_id'][:18]:<18} "
              f"{len(active):>4}/{len(centres):<5} {n_in:>4} {n_out:>4}  "
              f"{'MIXED' if is_mixed else ''}")

    print(f"\n  rendered {len(sel)} maps -> {out}")
    print(f"  {mixed}/{len(sel)} contain BOTH labels")
    print("  -> the label is per-TRAJECTORY: a scene holds ~11.6 vehicles but")
    print("     only ~1.5 active junctions, so cars away from the junction are")
    print("     in the same map and NOT involved. A map is not a cluster.")


if __name__ == "__main__":
    main()
