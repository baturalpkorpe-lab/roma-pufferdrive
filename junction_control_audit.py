"""
junction_control_audit.py -- how many junctions of each RIGHT-OF-WAY class are
in the map pool, and how much traffic actually uses them.

WHY: intersection_features.py pools stop signs (type 7) and crosswalks
(type 8) into one marker set and asks only "did this trajectory pass near an
active junction". That labels three completely different situations the same
way:

  all-way stop   equal priority on every approach -> the outcome is decided by
                 negotiation between the drivers, and nothing else. This is
                 the only class where a driving-style latent has room to act.
  signalised     priority is imposed externally. PufferDrive has NO traffic
                 light state at all (verified: no such field anywhere in
                 drive.h), so the GT vehicle stops for a light the policy
                 cannot observe. Behaviour here is unreproducible from the
                 observation, by construction.
  uncontrolled   a junction with no markers -- geometry only.

Rahmani et al. (TRR 2025) isolate the first class with: stop signs only, >=3
per junction, one per approach, clustered by MAXIMUM CLIQUE rather than
single-link. This script measures how many of each class survive in this pool
BEFORE any of it is committed to, because their yield was low: ~2,600
conflicts out of 1,570 hours of data.

Two clustering methods are reported side by side. Single-link (what
intersection_features.py uses via DBSCAN) chains: markers at A-B 20 m and
B-C 20 m collapse into one 40 m-wide "junction". Maximum clique requires every
marker in a group to be within the threshold of EVERY other, which cannot
chain. The gap between the two counts is the size of that problem in this pool.

MEASURED on synthetic layouts (test cases in the reply that shipped this file):
  * four markers in a line 18 m apart -> single-link@20 merges all four into
    ONE zone; clique@20 splits them. This is what clique is for.
  * two 4-way stops 80 m apart (a normal block) -> clique@45 correctly gives
    4 + 4.
  * two 4-way stops only 40 m apart (a short block) -> clique@45 OVER-MERGES
    to 6 + 2, while clique@15..30 still gives 4 + 4.
So the 45 m from Rahmani et al. is not automatically right for this pool -- it
was tuned on theirs. Run the audit at 2-3 values of --clique_dist (it is
minutes on CPU) and pick on the printed counts. A single junction's signs sit
one per approach, so 20-25 m is the a priori more plausible scale.

Output: junction_control_audit.csv, one row per junction zone, plus a summary.

Usage (login node is enough for --limit a few hundred; use the sbatch for all):
    python junction_control_audit.py \
        --data_dir /scratch/$USER/PufferDrive/pufferlib/resources/drive/binaries/training \
        --out junction_control_audit.csv --limit 300
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from conflict_metrics import HZ, MOVE_MS, vehicle_tracks, wrap
from intersection_features import _n_axes
from map_binary import read_map_binary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", default="junction_control_audit.csv")
    p.add_argument("--limit", type=int, default=0, help="0 = all maps")
    p.add_argument("--clique_dist", type=float, default=45.0,
                   help="Rahmani et al. use 45 m for stop signs of one "
                        "junction. Mutual, not chained, so it can be this "
                        "wide without merging neighbours.")
    p.add_argument("--single_link_eps", type=float, default=20.0,
                   help="what intersection_features.py uses, for comparison")
    p.add_argument("--zone_radius", type=float, default=25.0)
    p.add_argument("--min_stop_allway", type=int, default=3,
                   help="Rahmani criterion (a): >=3 stop signs, which admits "
                        "T-shaped layouts")
    p.add_argument("--sector_deg", type=float, default=45.0,
                   help="angular bin for counting covered approaches")
    p.add_argument("--progress", type=int, default=200)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cliques(adj):
    """All maximal cliques (Bron-Kerbosch with pivoting).

    Graphs here are tiny -- stop signs per scene are single digits -- so the
    recursive form is fine and needs no networkx (not installed on the
    analysis env).
    """
    out = []

    def expand(R, P, X):
        if not P and not X:
            out.append(R)
            return
        pivot = max(P | X, key=lambda u: len(adj[u]))
        for v in list(P - adj[pivot]):
            expand(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    expand(set(), set(range(len(adj))), set())
    return out


def clique_clusters(M, dist):
    """Group points so every member is within `dist` of every other member.

    Maximal cliques overlap, so they are taken largest-first and their members
    removed -- one marker belongs to one junction. Ties broken by tighter
    total spread, which prefers the geometrically compact reading.
    """
    n = len(M)
    if n == 0:
        return []
    D = np.hypot(M[:, 0, None] - M[None, :, 0], M[:, 1, None] - M[None, :, 1])
    adj = [set(np.flatnonzero((D[i] <= dist) & (np.arange(n) != i)))
           for i in range(n)]
    cl = _cliques(adj)
    cl.sort(key=lambda c: (-len(c), float(D[np.ix_(list(c), list(c))].sum())))
    taken, groups = set(), []
    for c in cl:
        c = c - taken
        if c:
            groups.append(sorted(c))
            taken |= c
    return groups


def single_link_count(M, eps):
    """Zone count under the chaining rule, for comparison only."""
    if not len(M):
        return 0
    n = len(M)
    lab, cur = -np.ones(n, int), 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], cur
        while stack:
            j = stack.pop()
            d = np.hypot(M[:, 0] - M[j, 0], M[:, 1] - M[j, 1])
            new = np.flatnonzero((d <= eps) & (lab < 0))
            lab[new] = cur
            stack.extend(new.tolist())
        cur += 1
    return cur


# ---------------------------------------------------------------------------
# Per-zone description
# ---------------------------------------------------------------------------

def _sectors(angles, sector_deg):
    """How many distinct DIRECTIONS (mod 2pi) are occupied."""
    if not len(angles):
        return 0
    return len(set(np.floor(np.mod(angles, 2 * np.pi) /
                            np.deg2rad(sector_deg)).astype(int).tolist()))


def _approach_dirs(centre, lanes, radius, sector_deg):
    """Directions from which lane centrelines reach a zone centre.

    Direction of the lane's own bearing, not of its position -- an approach is
    defined by which way traffic on it travels.
    """
    b = []
    for x, y in lanes:
        if len(x) < 2:
            continue
        mx, my = 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])
        k = np.hypot(mx - centre[0], my - centre[1]) <= radius
        if k.any():
            b.append(np.arctan2(np.diff(y), np.diff(x))[k])
    return _sectors(np.concatenate(b), sector_deg) if b else 0


def classify(n_stop, n_cross, n_axes, n_stop_dirs, n_appr_dirs, args):
    """Right-of-way class of one zone.

    Deliberately reports the INPUTS too, so the strictness of the all-way rule
    stays a decision made on the measured numbers rather than baked in here.
    Rahmani criterion (b) -- one sign per approach -- is the
    n_stop_dirs >= n_appr_dirs test, approximated by angular coverage.
    """
    if n_axes < 2:
        return "not_a_junction"
    if n_stop >= args.min_stop_allway:
        return "all_way_stop" if n_stop_dirs >= n_appr_dirs else "stop_partial_cover"
    if n_stop >= 1:
        return "partial_stop"
    if n_cross >= 1:
        return "signalised_likely"
    return "uncontrolled"


def scene_rows(m, args):
    roads, objs = m["roads"], m["objects"]
    sid = m["scenario_id"]
    lanes = [(r["x"], r["y"]) for r in roads if r["type"] == 4]

    stops = np.array([[r["x"][0], r["y"][0]] for r in roads if r["type"] == 7],
                     dtype=float).reshape(-1, 2)
    crosses = np.array([[r["x"].mean(), r["y"].mean()]
                        for r in roads if r["type"] == 8],
                       dtype=float).reshape(-1, 2)

    tracks = vehicle_tracks(objs)

    rows = []
    # Zones are seeded from stop signs (clique) and, separately, from
    # crosswalks -- so a signalised junction still gets a row instead of
    # silently vanishing, which is what a stop-sign-only pipeline would do.
    seeds = [("stop", clique_clusters(stops, args.clique_dist), stops)]
    if len(crosses):
        seeds.append(("cross", clique_clusters(crosses, args.clique_dist),
                      crosses))

    used = []
    for seed_kind, groups, M in seeds:
        for g in groups:
            centre = M[g].mean(0)
            # a crosswalk zone that a stop-sign zone already covers is the
            # same junction seen twice
            if seed_kind == "cross" and any(
                    np.hypot(*(centre - c)) <= args.zone_radius for c in used):
                continue
            used.append(centre)

            n_axes = _n_axes(centre, lanes)
            n_stop = int((np.hypot(stops[:, 0] - centre[0],
                                   stops[:, 1] - centre[1])
                          <= args.zone_radius).sum()) if len(stops) else 0
            n_cross = int((np.hypot(crosses[:, 0] - centre[0],
                                    crosses[:, 1] - centre[1])
                           <= args.zone_radius).sum()) if len(crosses) else 0
            if n_stop:
                sel = stops[np.hypot(stops[:, 0] - centre[0],
                                     stops[:, 1] - centre[1]) <= args.zone_radius]
                n_stop_dirs = _sectors(np.arctan2(sel[:, 1] - centre[1],
                                                  sel[:, 0] - centre[0]),
                                       args.sector_deg)
            else:
                n_stop_dirs = 0
            n_appr = _approach_dirs(centre, lanes, args.zone_radius,
                                    args.sector_deg)

            # traffic supply: movers through the zone, and co-present PAIRS,
            # which is the ceiling on how many conflicts this zone can yield
            inz = []
            for tr in tracks:
                d = np.hypot(tr["x"] - centre[0], tr["y"] - centre[1])
                k = np.flatnonzero(d <= args.zone_radius)
                if len(k):
                    inz.append((tr["i0"] + k[0], tr["i0"] + k[-1]))
            n_trav = len(inz)
            n_pairs = sum(1 for i in range(n_trav) for j in range(i + 1, n_trav)
                          if min(inz[i][1], inz[j][1]) >= max(inz[i][0], inz[j][0]))

            rows.append(dict(
                scenario_id=sid, seed=seed_kind,
                cx=round(float(centre[0]), 2), cy=round(float(centre[1]), 2),
                control=classify(n_stop, n_cross, n_axes, n_stop_dirs,
                                 n_appr, args),
                n_stop=n_stop, n_crosswalk=n_cross, n_axes=n_axes,
                n_stop_dirs=n_stop_dirs, n_approach_dirs=n_appr,
                n_traversals=n_trav, n_copresent_pairs=n_pairs,
            ))

    sl = single_link_count(np.vstack([stops, crosses]) if len(stops) or len(crosses)
                           else np.empty((0, 2)), args.single_link_eps)
    return rows, len(used), sl, len(tracks)


def main():
    args = parse_args()
    files = sorted(Path(args.data_dir).glob("map_*.bin"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no map_*.bin in {args.data_dir}")
    print(f"[audit] {len(files)} maps  clique={args.clique_dist}m "
          f"R={args.zone_radius}m", flush=True)

    rows, bad = [], 0
    n_clique, n_single, n_tracks = 0, 0, 0
    for i, fp in enumerate(files):
        try:
            r, nz, sl, nt = scene_rows(read_map_binary(fp), args)
        except Exception as e:
            bad += 1
            if bad <= 3:
                print(f"  FAIL {fp.name}: {type(e).__name__}: {e}")
            continue
        rows += r
        n_clique += nz
        n_single += sl
        n_tracks += nt
        if args.progress and (i + 1) % args.progress == 0:
            print(f"  {i+1}/{len(files)}  zones={len(rows)}", flush=True)

    if not rows:
        raise SystemExit("no zones found")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n  maps ok={len(files)-bad} failed={bad}  moving vehicles={n_tracks}")
    print(f"\n  CLUSTERING: clique={n_clique} zones vs single-link="
          f"{n_single} zones")
    print(f"    single-link/clique = {n_single/max(n_clique,1):.2f}  "
          f"(<1 means chaining merged distinct junctions)")

    cnt = Counter(r["control"] for r in rows)
    print(f"\n  {'control class':<20} {'zones':>7} {'%':>6} "
          f"{'w/ >=2 movers':>13} {'copresent pairs':>16}")
    for k, n in cnt.most_common():
        sub = [r for r in rows if r["control"] == k]
        n2 = sum(1 for r in sub if r["n_traversals"] >= 2)
        pairs = sum(r["n_copresent_pairs"] for r in sub)
        print(f"  {k:<20} {n:>7} {100*n/len(rows):>5.1f}% {n2:>13} {pairs:>16}")

    scenes_all = len({r["scenario_id"] for r in rows
                      if r["control"] == "all_way_stop"
                      and r["n_copresent_pairs"] >= 1})
    print(f"\n  GO/NO-GO: {scenes_all} scenes have an all-way-stop junction "
          f"with >=1 co-present pair")
    print(f"    of {len(files)-bad} scanned ("
          f"{100*scenes_all/max(len(files)-bad,1):.1f}%)")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
