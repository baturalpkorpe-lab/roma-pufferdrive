"""
render_regime_strip.py -- ONE figure that explains the whole regime idea:
a single real trajectory, coloured by which situation it is in, moment by moment.

WHY THIS FIGURE EXISTS
The regime split is easy to state and hard to picture. "We group by situation,
not by behaviour" sounds abstract until you see one car spend four seconds on
open road, then catch a leader, then reach a junction with cross traffic -- with
a different style parameter identified in each stretch. That is what the old
per-trajectory clustering threw away: it would have called this car ONE type.

Three panels:
  1. the path in x-y, coloured by regime, with the junction zones drawn and the
     conflict partner shown faintly
  2. speed against time, with the regime shaded behind it -- this is where the
     "why" shows: speed drops when a leader appears, not because the driver
     changed
  3. a regime strip: which situation at each of the 91 steps

Ground truth only. No policy, no checkpoint, no GPU -- it runs on a login node
in seconds, and a HUMAN trajectory is the right thing to explain the method
with, because nobody can object that the labelling was tuned to flatter a model.

Usage:
    python render_regime_strip.py \
        --data_dir /scratch/$USER/PufferDrive/pufferlib/resources/drive/binaries/training \
        --out_dir  /scratch/$USER/regime_figs --n 8 --limit 400
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import conflict_metrics as CM
from junction_control_audit import labelled_zones
from map_binary import read_map_binary

T = 91
HZ = CM.HZ

# One colour per situation, used identically in all three panels.
COL = {"freeflow": "#2e9e5b", "following": "#2b6cb0",
       "junction": "#d9822b", "other": "#b0b0b0"}
LABEL = {"freeflow": "free flow (desired speed)",
         "following": "following (time headway)",
         "junction": "junction (gap acceptance)",
         "other": "stopped / unclassified"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=8, help="figures to produce")
    p.add_argument("--limit", type=int, default=400, help="maps to scan")
    p.add_argument("--min_switches", type=int, default=2,
                   help="only draw trajectories that change situation at least "
                        "this many times -- a car that free-flows for 9 s makes "
                        "a dull argument")
    p.add_argument("--zone_radius", type=float, default=25.0)
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args()


def regime_per_step(D, e, lead_row, centres, zone_radius):
    """(T,) array of situation labels for one vehicle. Priority: junction >
    following > free flow, so an agent following a leader INTO a junction is
    drawn as the junction case -- that is the situation whose style parameter
    we would read."""
    n = D["X"].shape[1]
    lab = np.array(["other"] * n, dtype=object)
    valid = D["M"][e]
    v = np.nan_to_num(D["V"][e])
    lab[valid & (lead_row < 0) & (v > 2.0)] = "freeflow"
    lab[valid & (lead_row >= 0)] = "following"
    if len(centres):
        d = np.hypot(D["X"][e][:, None] - centres[None, :, 0],
                     D["Y"][e][:, None] - centres[None, :, 1])
        inz = np.nan_to_num(d.min(1), nan=np.inf) <= zone_radius
        lab[valid & inz] = "junction"
    return lab


def draw(tr, D, e, lab, centres, zones, roads, sid, out, dpi):
    t = np.arange(len(lab)) / HZ
    v = np.nan_to_num(D["V"][e])
    m = D["M"][e]

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.1, 1.5, 0.42], hspace=0.55)

    # ---- 1. the path ----------------------------------------------------
    ax = fig.add_subplot(gs[0])
    for rd in roads:
        if rd["type"] in (4, 5, 6) and rd["n"] >= 2:
            ax.plot(rd["x"], rd["y"], color="#e8e8e8", lw=1.0, zorder=0)
    for z, c in zip(zones, centres):
        ax.add_patch(Circle(c, z["R"], fill=False, ls="--", lw=1.2,
                            ec=COL["junction"], alpha=.7, zorder=1))
        ax.annotate(z["control"], (c[0], c[1] + z["R"]),
                    color=COL["junction"], fontsize=8, ha="center",
                    va="bottom", alpha=.9)
    for k in range(len(lab) - 1):
        if not (m[k] and m[k + 1]):
            continue
        ax.plot(D["X"][e][k:k + 2], D["Y"][e][k:k + 2],
                color=COL[lab[k]], lw=4, solid_capstyle="round", zorder=3)
    i0 = int(np.flatnonzero(m)[0])
    ax.plot(D["X"][e][i0], D["Y"][e][i0], "o", ms=9, mfc="w", mec="k",
            mew=1.6, zorder=4)
    ax.annotate("start", (D["X"][e][i0], D["Y"][e][i0]),
                textcoords="offset points", xytext=(8, 8), fontsize=9)
    pad = 25
    ok = m & np.isfinite(D["X"][e])
    ax.set_xlim(np.nanmin(D["X"][e][ok]) - pad, np.nanmax(D["X"][e][ok]) + pad)
    ax.set_ylim(np.nanmin(D["Y"][e][ok]) - pad, np.nanmax(D["Y"][e][ok]) + pad)
    ax.set_aspect("equal")
    ax.set_title(f"One real trajectory, coloured by SITUATION   "
                 f"(scenario {sid[:10]}, vehicle {tr['id']})",
                 fontsize=12, weight="bold")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    seen = [k for k in ("freeflow", "following", "junction", "other")
            if (lab == k).any()]
    ax.legend([plt.Line2D([], [], color=COL[k], lw=4) for k in seen],
              [LABEL[k] for k in seen], loc="best", fontsize=9, framealpha=.9)

    # ---- 2. speed, shaded by situation ----------------------------------
    ax2 = fig.add_subplot(gs[1])
    for k in range(len(lab) - 1):
        ax2.axvspan(t[k], t[k + 1], color=COL[lab[k]], alpha=.20, lw=0)
    ax2.plot(t[m], v[m], color="k", lw=1.9)
    ax2.set_ylabel("speed (m/s)")
    ax2.set_xlim(0, t[-1])
    ax2.grid(alpha=.25)
    ax2.set_title("the same trajectory over time -- speed changes when the "
                  "SITUATION changes, not because the driver did",
                  fontsize=10, pad=10)

    # ---- 3. the strip ---------------------------------------------------
    ax3 = fig.add_subplot(gs[2])
    for k in range(len(lab) - 1):
        ax3.axvspan(t[k], t[k + 1], color=COL[lab[k]], lw=0)
    ax3.set_xlim(0, t[-1]); ax3.set_yticks([])
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("regime", rotation=0, ha="right", va="center", fontsize=9)

    frac = {k: float((lab == k).mean()) for k in COL}
    fig.text(0.5, 0.005,
             "  |  ".join(f"{LABEL[k].split(' (')[0]}: {100*frac[k]:.0f}%"
                          for k in ("freeflow", "following", "junction")
                          if frac[k] > 0),
             ha="center", fontsize=9.5)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.data_dir).glob("map_*.bin"))[:args.limit or None]
    if not files:
        raise SystemExit(f"no map_*.bin in {args.data_dir}")

    made = 0
    for fp in files:
        if made >= args.n:
            break
        try:
            m = read_map_binary(fp)
        except Exception:
            continue
        tracks = CM.vehicle_tracks(m["objects"])
        if len(tracks) < 2:
            continue
        D = CM.dense_scene(tracks, T)
        lead = CM.leaders_dense(D)
        Z = [z for z in labelled_zones(m["roads"])
             if z["control"] != "not_a_junction"]
        centres = (np.array([z["centre"] for z in Z], float).reshape(-1, 2)
                   if Z else np.empty((0, 2)))

        for e, tr in enumerate(tracks):
            if made >= args.n:
                break
            lab = regime_per_step(D, e, lead[e], centres, args.zone_radius)
            sw = int((lab[1:] != lab[:-1]).sum())
            if sw < args.min_switches:
                continue
            if len({k for k in lab if k != "other"}) < 2:
                continue                       # need at least two situations
            out = out_dir / f"regime_{m['scenario_id'][:10]}_v{tr['id']}.png"
            try:
                draw(tr, D, e, lab, centres, Z, m["roads"],
                     m["scenario_id"], out, args.dpi)
            except Exception as ex:
                print(f"  skip {out.name}: {type(ex).__name__}: {ex}")
                continue
            made += 1
            print(f"  [{made}/{args.n}] {out.name}  switches={sw}  "
                  + " ".join(f"{k}={100*float((lab==k).mean()):.0f}%"
                             for k in ("freeflow", "following", "junction")
                             if (lab == k).any()))

    print(f"\n  wrote {made} figure(s) -> {out_dir}")
    if not made:
        print("  nothing matched -- lower --min_switches or raise --limit")


if __name__ == "__main__":
    main()
