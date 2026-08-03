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
from types import SimpleNamespace

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
    p.add_argument("--zone_radius", type=float, default=25.0,
                   help="fixed zone radius, used only with --fixed_radius 1")
    p.add_argument("--fixed_radius", type=int, default=0,
                   help="1 = legacy fixed --zone_radius. 0 (default) = the "
                        "paper's rule: radius is the max centre-to-marker "
                        "distance plus --zone_buffer, per zone.")
    p.add_argument("--zone_buffer", type=float, default=4.0,
                   help="Rahmani et al. add 4 m to the max centre-to-sign "
                        "distance so the zone overlaps the approach lanes")
    p.add_argument("--zone_radius_min", type=float, default=15.0,
                   help="floor, so a 1-sign zone still has an area")
    p.add_argument("--min_stop_allway", type=int, default=3,
                   help="Rahmani criterion (a): >=3 stop signs, which admits "
                        "T-shaped layouts")
    p.add_argument("--sector_deg", type=float, default=45.0,
                   help="angular separation for counting distinct directions")
    p.add_argument("--require_sign_per_approach", type=int, default=0,
                   help="opt-in: also demand n_stop_dirs >= n_approach_dirs "
                        "(Rahmani criterion b). OFF by default -- without an "
                        "entry/exit lane graph, turn connectors over-count "
                        "approaches and this empties the class. Compare the "
                        "reported n_stop_dirs / n_approach_dirs columns before "
                        "turning it on.")
    p.add_argument("--clique_sweep", type=str, default="",
                   help="comma list of clique distances, e.g. "
                        "'25,45,60,75'. Classifies at each in ONE pass and "
                        "prints the table instead of writing a CSV. This is "
                        "the decisive test for whether the threshold is "
                        "splitting real 4-way stops: widening should convert "
                        "n_stop=1/2 zones into n_stop=4 zones, and the "
                        "n_stop>=6 count is the canary for over-merging two "
                        "adjacent junctions into one.")
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
    """How many distinct DIRECTIONS (mod 2pi) are occupied.

    Greedy separation from a representative, NOT hard bins on the circle.
    _n_axes in intersection_features.py carries an explicit warning about this:
    bin/gap-based grouping was tried on the real pool and rejected ~99% of
    zones, because junction TURN CONNECTORS sweep continuously through every
    angle between the two roads, so a real junction's bearing distribution is
    continuous rather than a few clumps. Hard 45 deg bins reproduced that
    failure exactly -- the first version of this file gated all_way_stop on
    n_stop_dirs >= n_appr_dirs and returned ZERO all-way stops out of 218 zones
    that had 3+ stop signs, because connectors inflated n_appr_dirs to 7-8.
    """
    if not len(angles):
        return 0
    tol = np.deg2rad(sector_deg)
    reps = []
    for a in np.sort(np.mod(np.asarray(angles, float), 2 * np.pi)):
        for g in reps:
            d = abs(a - g)
            if min(d, 2 * np.pi - d) < tol:
                break
        else:
            reps.append(a)
    return len(reps)


def _approach_dirs(centre, lanes, radius, sector_deg):
    """Directions from which lane centrelines reach a zone centre.

    Direction of the lane's own bearing, not of its position -- an approach is
    defined by which way traffic on it travels. Still only a DIAGNOSTIC: with
    connectors present this over-counts, and it is not used as a gate unless
    --require_sign_per_approach is passed explicitly.
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

    The gate is Rahmani criterion (a) only -- >=3 stop signs, which is what
    they state admits three-leg (T-shaped) layouts. Their criterion (b), one
    sign per approach, needs the entry/exit lane graph of an HD map; here the
    road id is a SEGMENT id and turn connectors make any bearing-based
    approximation over-count approaches (see _sectors). So the coverage numbers
    are REPORTED per zone and available as an opt-in gate, but not imposed:
    a broken approximation of (b) silently emptied this class once already.
    """
    if n_axes < 2:
        return "not_a_junction"
    if n_stop >= args.min_stop_allway:
        if args.require_sign_per_approach and n_stop_dirs < n_appr_dirs:
            return "stop_partial_cover"
        return "all_way_stop"
    if n_stop >= 1:
        return "partial_stop"
    if n_cross >= 1:
        return "signalised_likely"
    return "uncontrolled"


def labelled_zones(roads, clique_dist=45.0, zone_buffer=4.0,
                   zone_radius_min=15.0, min_stop_allway=3, sector_deg=45.0):
    """Junction zones with a right-of-way label, for callers outside this file.

    Returns [dict(centre=(2,), R, control, n_stop, n_crosswalk, n_axes)].

    Deliberately ADDITIVE: scene_rows() keeps its own copy of this flow and is
    not refactored to call this. The audit's full-pool numbers are already in
    hand, there is no map data outside the cluster to verify a refactor against,
    and a regression in the audit costs more than the duplication. The
    CLASSIFICATION itself is not duplicated -- both paths call classify().
    """
    lanes = [(r["x"], r["y"]) for r in roads if r["type"] == 4]
    stops = np.array([[r["x"][0], r["y"][0]] for r in roads if r["type"] == 7],
                     dtype=float).reshape(-1, 2)
    crosses = np.array([[r["x"].mean(), r["y"].mean()]
                        for r in roads if r["type"] == 8],
                       dtype=float).reshape(-1, 2)

    cargs = SimpleNamespace(min_stop_allway=min_stop_allway,
                            require_sign_per_approach=0)
    out, used = [], []
    for seed_kind, M in (("stop", stops), ("cross", crosses)):
        if not len(M):
            continue
        for g in clique_clusters(M, clique_dist):
            mem = M[g]
            centre = mem.mean(0)
            spread = (float(np.hypot(*(mem - centre).T).max())
                      if len(mem) > 1 else 0.0)
            R = max(spread + zone_buffer, zone_radius_min)
            if seed_kind == "cross" and any(
                    np.hypot(*(centre - c)) <= R for c in used):
                continue                      # same junction already seeded
            used.append(centre)

            if seed_kind == "stop":
                sel = mem
            else:
                sel = (stops[np.hypot(stops[:, 0] - centre[0],
                                      stops[:, 1] - centre[1]) <= R]
                       if len(stops) else np.empty((0, 2)))
            n_stop = len(sel)
            n_cross = int((np.hypot(crosses[:, 0] - centre[0],
                                    crosses[:, 1] - centre[1]) <= R).sum()) \
                if len(crosses) else 0
            n_axes = _n_axes(centre, lanes)
            n_dirs = _sectors(np.arctan2(sel[:, 1] - centre[1],
                                         sel[:, 0] - centre[0]),
                              sector_deg) if n_stop else 0
            n_appr = _approach_dirs(centre, lanes, R, sector_deg)
            out.append(dict(
                centre=centre, R=R,
                control=classify(n_stop, n_cross, n_axes, n_dirs, n_appr, cargs),
                n_stop=n_stop, n_crosswalk=n_cross, n_axes=n_axes))
    return out


def scene_rows(m, args, dist=None, tracks=None):
    roads, objs = m["roads"], m["objects"]
    sid = m["scenario_id"]
    dist = args.clique_dist if dist is None else dist
    lanes = [(r["x"], r["y"]) for r in roads if r["type"] == 4]

    stops = np.array([[r["x"][0], r["y"][0]] for r in roads if r["type"] == 7],
                     dtype=float).reshape(-1, 2)
    crosses = np.array([[r["x"].mean(), r["y"].mean()]
                        for r in roads if r["type"] == 8],
                       dtype=float).reshape(-1, 2)

    tracks = vehicle_tracks(objs) if tracks is None else tracks

    rows = []
    # Zones are seeded from stop signs (clique) and, separately, from
    # crosswalks -- so a signalised junction still gets a row instead of
    # silently vanishing, which is what a stop-sign-only pipeline would do.
    seeds = [("stop", clique_clusters(stops, dist), stops)]
    if len(crosses):
        seeds.append(("cross", clique_clusters(crosses, dist), crosses))

    used = []
    for seed_kind, groups, M in seeds:
        for g in groups:
            mem = M[g]
            centre = mem.mean(0)
            # Zone radius per the paper: the max centre-to-marker distance plus
            # a buffer, NOT a fixed constant. A fixed 25 m was the binding
            # constraint at wide clique distances -- a junction whose signs span
            # ~50 m has its outer signs sitting AT 25 m from the centroid, so
            # they fell outside the counting radius and n_stop fell as
            # clique_dist grew (n=4 went 95 -> 83 -> 79 over clique 45/60/75,
            # the opposite of the expected direction).
            if args.fixed_radius:
                R = args.zone_radius
            else:
                spread_r = (float(np.hypot(*(mem - centre).T).max())
                            if len(mem) > 1 else 0.0)
                R = max(spread_r + args.zone_buffer, args.zone_radius_min)

            # a crosswalk zone that a stop-sign zone already covers is the
            # same junction seen twice
            if seed_kind == "cross" and any(
                    np.hypot(*(centre - c)) <= R for c in used):
                continue
            used.append(centre)

            n_axes = _n_axes(centre, lanes)
            # For a stop-seeded zone the clique IS the set of signs assigned to
            # that junction, so its size is exact. Counting by radius instead
            # was wrong in both directions: at tight thresholds several
            # fragments of one junction each saw the same 3-4 signs and every
            # fragment reported n_stop>=3 (the 218-vs-153 double count), and at
            # wide thresholds it truncated as described above.
            if seed_kind == "stop":
                sel = mem
            else:
                sel = (stops[np.hypot(stops[:, 0] - centre[0],
                                      stops[:, 1] - centre[1]) <= R]
                       if len(stops) else np.empty((0, 2)))
            n_stop = len(sel)
            n_cross = int((np.hypot(crosses[:, 0] - centre[0],
                                    crosses[:, 1] - centre[1])
                           <= R).sum()) if len(crosses) else 0

            stop_spread = 0.0
            if n_stop:
                n_stop_dirs = _sectors(np.arctan2(sel[:, 1] - centre[1],
                                                  sel[:, 0] - centre[0]),
                                       args.sector_deg)
                if len(sel) > 1:
                    dd = np.hypot(sel[:, 0, None] - sel[None, :, 0],
                                  sel[:, 1, None] - sel[None, :, 1])
                    stop_spread = float(dd.max())
            else:
                n_stop_dirs = 0
            n_appr = _approach_dirs(centre, lanes, R, args.sector_deg)

            # traffic supply: movers through the zone, and co-present PAIRS,
            # which is the ceiling on how many conflicts this zone can yield
            inz = []
            for tr in tracks:
                d = np.hypot(tr["x"] - centre[0], tr["y"] - centre[1])
                k = np.flatnonzero(d <= R)
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
                stop_spread=round(stop_spread, 1), zone_r=round(R, 1),
                n_traversals=n_trav, n_copresent_pairs=n_pairs,
            ))

    # Single-link at the SAME distance as the clique. Comparing clique@45 with
    # single-link@20 conflates method and threshold: it reported 1.18 in one
    # run and 0.94 in another purely because the thresholds differed, which
    # says nothing about chaining. Same distance, so the only difference left
    # is mutual-proximity vs transitive-reachability.
    M = (np.vstack([stops, crosses]) if len(stops) or len(crosses)
         else np.empty((0, 2)))
    sl = single_link_count(M, dist)
    return rows, len(used), sl, len(tracks)


def sweep(files, args):
    """Classify at several clique distances in one pass and print the table.

    The point is the SHAPE of the sign histogram as the threshold widens. A
    genuine all-way stop at a crossroads has exactly 4 signs, so if the
    threshold is truncating junctions, widening it moves mass from n_stop=1/2
    into n_stop=4. If instead n_stop>=6 grows, two adjacent junctions are being
    merged and the threshold has gone too far.
    """
    dists = [float(d) for d in args.clique_sweep.split(",") if d.strip()]
    print(f"[audit] SWEEP over clique={dists} on {len(files)} maps", flush=True)

    acc = {d: dict(zones=0, single=0, cls=Counter(), hist=Counter(),
                   allway_zones=0, allway_pairs=0, allway_scenes=set(),
                   spread=[]) for d in dists}
    bad = 0
    for i, fp in enumerate(files):
        try:
            m = read_map_binary(fp)
            tracks = vehicle_tracks(m["objects"])
        except Exception as e:
            bad += 1
            if bad <= 3:
                print(f"  FAIL {fp.name}: {type(e).__name__}: {e}")
            continue
        for d in dists:
            try:
                rows, nz, sl, _ = scene_rows(m, args, dist=d, tracks=tracks)
            except Exception as e:
                bad += 1
                continue
            a = acc[d]
            a["zones"] += nz
            a["single"] += sl
            for r in rows:
                a["cls"][r["control"]] += 1
                if r["n_stop"] >= 1:
                    a["hist"][min(r["n_stop"], 6)] += 1
                if r["n_stop"] >= 2:
                    a["spread"].append(r["stop_spread"])
                if r["control"] == "all_way_stop":
                    a["allway_zones"] += 1
                    a["allway_pairs"] += r["n_copresent_pairs"]
                    if r["n_copresent_pairs"] >= 1:
                        a["allway_scenes"].add(r["scenario_id"])
        if args.progress and (i + 1) % args.progress == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    n_ok = len(files) - 0
    print(f"\n  {'clique':>7} {'zones':>7} {'single':>7} {'allway':>7} "
          f"{'pairs':>7} {'scenes':>7} | " +
          " ".join(f"n={k if k<6 else '6+':<3}" for k in range(1, 7)))
    for d in dists:
        a = acc[d]
        h = a["hist"]
        print(f"  {d:>7.0f} {a['zones']:>7} {a['single']:>7} "
              f"{a['allway_zones']:>7} {a['allway_pairs']:>7} "
              f"{len(a['allway_scenes']):>7} | " +
              " ".join(f"{h.get(k,0):<5}" for k in range(1, 7)))

    print(f"\n  spread of stop signs within a zone (m), n_stop>=2:")
    for d in dists:
        sp = np.asarray(acc[d]["spread"], float)
        if len(sp):
            print(f"    clique={d:>5.0f}  med={np.median(sp):>5.1f} "
                  f"p90={np.percentile(sp,90):>5.1f} max={sp.max():>5.1f} "
                  f"(max is capped at the threshold by construction)")

    print(f"\n  READ IT AS: widening should move mass from n=1/n=2 into n=4 "
          f"(a crossroads all-way\n  stop has exactly 4 signs). Stop widening "
          f"when n=6+ starts growing -- that is two\n  adjacent junctions "
          f"merging into one zone.")
    print(f"\n  maps scanned={len(files)} failures={bad}")


def main():
    args = parse_args()
    files = sorted(Path(args.data_dir).glob("map_*.bin"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no map_*.bin in {args.data_dir}")
    if args.clique_sweep:
        return sweep(files, args)

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

    # --- is --clique_dist the reason a class is empty? --------------------
    sg = [r for r in rows if r["n_stop"] >= 1]
    if sg:
        hist = Counter(min(r["n_stop"], 6) for r in sg)
        print(f"\n  STOP SIGNS PER ZONE (of {len(sg)} zones with any)")
        for k in sorted(hist):
            lbl = f"{k}" if k < 6 else "6+"
            print(f"    n_stop={lbl:<3} {hist[k]:>7}  {100*hist[k]/len(sg):>5.1f}%")
        sp = np.array([r["stop_spread"] for r in sg if r["n_stop"] >= 2])
        if len(sp):
            print(f"    stop-sign spread within a zone (m): med={np.median(sp):.1f} "
                  f"p90={np.percentile(sp,90):.1f} max={sp.max():.1f}")
            print(f"    -> if p90 is near --clique_dist ({args.clique_dist:.0f} m), "
                  f"the threshold is truncating real junctions; re-run wider")

    # coverage diagnostic: what the opt-in criterion (b) would cost
    cand = [r for r in rows if r["n_stop"] >= args.min_stop_allway
            and r["n_axes"] >= 2]
    if cand:
        pass_b = sum(1 for r in cand
                     if r["n_stop_dirs"] >= r["n_approach_dirs"])
        print(f"\n  CRITERION (b) one-sign-per-approach, as a DIAGNOSTIC:")
        print(f"    {pass_b}/{len(cand)} zones with >={args.min_stop_allway} "
              f"signs would survive it "
              f"({'ON' if args.require_sign_per_approach else 'OFF'} in this run)")
        print(f"    median n_stop_dirs={np.median([r['n_stop_dirs'] for r in cand]):.1f} "
              f"vs n_approach_dirs={np.median([r['n_approach_dirs'] for r in cand]):.1f}"
              f"  (approach dirs are inflated by turn connectors)")

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
