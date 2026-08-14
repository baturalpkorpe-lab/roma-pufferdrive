"""
intersection_features.py -- per-trajectory intersection involvement, from the
map binaries.

WHY: the stop_frac clustering puts a trajectory in "turning" whenever net_turn
is high, which conflates two different things -- a car following a curved road
(no decision, no interaction) and a car turning at a junction (yield, check
cross traffic, choose a gap). And a car going STRAIGHT through a 4-way stop is
an intersection trajectory with almost no turning at all, so net_turn misses it
entirely. This file labels the ENVIRONMENT the trajectory happens in, not the
shape of the trajectory.

DEFINITION (structure, gated by real traffic):
  1. Junction zones are built from map structure:
       - stop signs      (type 7)   one per approach lane, so they CLUSTER
       - crosswalks      (type 8)   centroid; covers signalised junctions,
                                    which have NO stop sign -- save_map_binary
                                    writes no traffic-light state at all
       - lane crossings            two lane centerlines (type 4) that actually
                                    cross. On a normal road parallel lanes
                                    never cross; at a junction through- and
                                    turning-lanes do. Robust to the ~16 m
                                    segmentation, unlike anything id-based
                                    (the road id is a SEGMENT id: 73082/73082
                                    unique, so "id changed" says nothing).
     Those markers are spatially clustered (--zone_eps) into one zone per
     junction.
  2. A zone is ACTIVE only if >= --min_traversals moving vehicles actually
     drive through it. A junction nobody uses is not an intersection anyone
     had to deal with.
  2b. A zone must have >= --min_axes distinct road AXES near it, where two
     axes count as distinct only if they differ by >= 60 deg. A mid-block
     crosswalk (or a stop sign on a straight road) sits on ONE axis -- the two
     travel directions are 180 deg apart, i.e. the same line -- so without this
     both directions of an ordinary street get labelled.
     The 60 deg threshold is deliberate: a curved road sweeps its bearing
     continuously, so a smaller threshold splits one curve into several
     "axes" and a crosswalk on a bend passes. Known limit: a curve of radius
     ~25 m still reads as 2 axes (it sweeps ~100 deg within the sampling
     radius) -- rare, and harmless unless a marker sits on it.
  3. A trajectory is intersection-involved if, while MOVING, it passes within
     --zone_radius of an ACTIVE zone.

DISCRIMINATOR: turn_in_zone_frac -- the share of the trajectory's total
|heading change| that happens INSIDE a junction zone. A junction turn
concentrates its turning there (1.0); a curved road spreads it along the path
(0.0). Measured on synthetic data: 1.00 vs 0.00 at |net_turn| 1.57 vs 1.48.

road_rel_turn is still emitted but DOES NOT WORK as a discriminator and should
not be used. It takes the nearest lane bearing at the trajectory start and end,
but at a junction the end lane is the NEW road, so the road bearing change
already contains the turn and the residual is ~0 by construction. On real data
it came out backwards: 0.172 for junction turns vs 0.531 for the rest.

Output: one row per (scenario_id, vehicle_id) ->
    intersection_features.csv

Usage:
    python intersection_features.py \
        --data_dir /scratch/$USER/PufferDrive/pufferlib/resources/drive/binaries/training \
        --out intersection_features.csv
"""

import argparse
from pathlib import Path

import numpy as np

from map_binary import read_map_binary

MOVE_MS = 1.0          # max speed below this = parked (matches traj_kinematics)
HZ = 10.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", default="intersection_features.csv")
    p.add_argument("--limit", type=int, default=0, help="0 = all maps")
    p.add_argument("--zone_eps", type=float, default=20.0,
                   help="markers within this distance form ONE junction (m)")
    p.add_argument("--zone_radius", type=float, default=25.0,
                   help="a trajectory within this of a zone centre is in it (m)")
    p.add_argument("--min_traversals", type=int, default=1,
                   help="moving vehicles that must traverse a zone for it to "
                        "count as ACTIVE")
    p.add_argument("--min_axes", type=int, default=2,
                   help="distinct road axes required near a zone. 2 rejects a "
                        "mid-block crosswalk / stop sign on a straight road, "
                        "whose two travel directions are ONE axis. 1 = off.")
    p.add_argument("--use_lane_crossings", type=int, default=0,
                   help="Infer junctions from lane geometry as well as signs. "
                        "DEFAULT OFF, measured on the full 10k pool: it adds "
                        "2.08M markers (21:1 vs signs+crosswalks) for +7pp of "
                        "involvement (31.1%% -> 38.0%%) and produces NO extra "
                        "active zones (15349 markers-only vs 15031 with). It "
                        "is also immune to --min_axes, since a lane T-junction "
                        "marker requires >=30 deg between lanes and the axis "
                        "test requires >=30 deg between axes -- circular. Signs "
                        "and crosswalks are Waymo ground truth; this is "
                        "inference. Turn on only with rendered validation.")
    p.add_argument("--progress", type=int, default=200)
    return p.parse_args()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Junction zones
# ---------------------------------------------------------------------------

def _seg_crossings(polys, max_pts=6000):
    """Points where two DIFFERENT lane centerlines cross.

    Segment midpoints go into a KD-tree and only pairs closer than the segment
    scale are tested exactly -- all-pairs would be ~32M comparisons on the
    biggest scenes (810 lanes). Returns (M, 2) crossing points.
    """
    P1, P2, owner = [], [], []
    for k, (x, y) in enumerate(polys):
        if len(x) < 2:
            continue
        P1.append(np.stack([x[:-1], y[:-1]], 1))
        P2.append(np.stack([x[1:],  y[1:]],  1))
        owner.append(np.full(len(x) - 1, k))
    if not P1:
        return np.empty((0, 2))
    P1 = np.concatenate(P1); P2 = np.concatenate(P2)
    owner = np.concatenate(owner)
    if len(P1) > max_pts:                       # pathological scene: skip
        return np.empty((0, 2))

    mid = 0.5 * (P1 + P2)
    seglen = np.hypot(*(P2 - P1).T)
    r = float(max(seglen.max(), 1.0)) * 1.5
    try:
        from scipy.spatial import cKDTree
        pairs = np.array(list(cKDTree(mid).query_pairs(r)), dtype=int)
    except Exception:                            # no scipy -> all pairs
        n = len(mid)
        if n > 1500:
            return np.empty((0, 2))
        i, j = np.triu_indices(n, 1)
        pairs = np.stack([i, j], 1)
    if len(pairs) == 0:
        return np.empty((0, 2))
    a, b = pairs[:, 0], pairs[:, 1]
    keep = owner[a] != owner[b]                  # different lanes only
    a, b = a[keep], b[keep]
    if len(a) == 0:
        return np.empty((0, 2))

    p, r_ = P1[a], P2[a] - P1[a]
    q, s_ = P1[b], P2[b] - P1[b]
    den = r_[:, 0] * s_[:, 1] - r_[:, 1] * s_[:, 0]
    ok = np.abs(den) > 1e-12
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[:, 0] * s_[:, 1] - qp[:, 1] * s_[:, 0]) / den
        u = (qp[:, 0] * r_[:, 1] - qp[:, 1] * r_[:, 0]) / den
    hit = ok & (t > 0.01) & (t < 0.99) & (u > 0.01) & (u < 0.99)
    if not hit.any():
        return np.empty((0, 2))
    return p[hit] + t[hit, None] * r_[hit]


def _lane_tjunctions(polys, tol=4.0, min_angle=np.deg2rad(30)):
    """T-junctions: a lane ENDPOINT sitting on another lane, at an angle.

    _seg_crossings only finds lanes that pass THROUGH each other (a crossroads).
    At a T the side lane terminates against the main lane, so the meeting point
    is an endpoint and the interior-hit test misses it -- verified on a
    synthetic T with no sign and no crosswalk, which scored zero markers.

    The angle test is what separates a junction from mere segmentation: the
    road id is a SEGMENT id and one road is chopped into ~16 m pieces, so
    lane endpoints touch their own continuation everywhere. A continuation is
    collinear (angle ~0 or ~pi); a real T meets at an angle.
    """
    segs, bears, owner = [], [], []
    for k, (x, y) in enumerate(polys):
        if len(x) < 2:
            continue
        segs.append(np.stack([0.5 * (x[:-1] + x[1:]),
                              0.5 * (y[:-1] + y[1:])], 1))
        bears.append(np.arctan2(np.diff(y), np.diff(x)))
        owner.append(np.full(len(x) - 1, k))
    if not segs:
        return np.empty((0, 2))
    S = np.concatenate(segs); B = np.concatenate(bears)
    O = np.concatenate(owner)

    ends, ebear, eowner = [], [], []
    for k, (x, y) in enumerate(polys):
        if len(x) < 2:
            continue
        ends += [[x[0], y[0]], [x[-1], y[-1]]]
        ebear += [np.arctan2(y[1] - y[0], x[1] - x[0]),
                  np.arctan2(y[-1] - y[-2], x[-1] - x[-2])]
        eowner += [k, k]
    if not ends:
        return np.empty((0, 2))
    E = np.asarray(ends, float); EB = np.asarray(ebear); EO = np.asarray(eowner)

    # Endpoints of the segments, needed for a true point-to-SEGMENT distance.
    # Querying midpoints alone is wrong: with ~16 m segments an endpoint lying
    # exactly ON a segment can still be 8 m from its midpoint, so a 4 m
    # tolerance silently misses every T. Search a midpoint radius wide enough
    # to cover any segment, then measure the real distance.
    A, Bp = [], []
    for k, (x, y) in enumerate(polys):
        if len(x) < 2:
            continue
        A.append(np.stack([x[:-1], y[:-1]], 1))
        Bp.append(np.stack([x[1:], y[1:]], 1))
    A = np.concatenate(A); Bp = np.concatenate(Bp)
    half = float(np.hypot(*(Bp - A).T).max()) * 0.5 + tol

    try:
        from scipy.spatial import cKDTree
        near = cKDTree(S).query_ball_point(E, r=half)
    except Exception:
        near = [np.where(np.hypot(S[:, 0] - e[0], S[:, 1] - e[1]) <= half)[0]
                for e in E]

    out = []
    for i, idxs in enumerate(near):
        idxs = np.atleast_1d(np.asarray(idxs, dtype=int))
        idxs = idxs[O[idxs] != EO[i]]            # different lane only
        if not len(idxs):
            continue
        ab = Bp[idxs] - A[idxs]
        ap = E[i] - A[idxs]
        L2 = (ab * ab).sum(1)
        t = np.where(L2 > 1e-12, (ap * ab).sum(1) / np.maximum(L2, 1e-12), 0.0)
        t = np.clip(t, 0.0, 1.0)
        proj = A[idxs] + t[:, None] * ab
        d = np.hypot(*(E[i] - proj).T)
        ang = np.abs(wrap(B[idxs] - EB[i]))
        ang = np.minimum(ang, np.pi - ang)       # direction-agnostic
        if np.any((d <= tol) & (ang >= min_angle)):
            out.append(E[i])
    return np.asarray(out, float) if out else np.empty((0, 2))


def _n_axes(centre, lanes, radius=22.0, tol=np.deg2rad(60)):
    """How many distinct road AXES pass near a point.

    A mid-block crosswalk or a stop sign on a straight road sits on ONE axis:
    the two travel directions are 180 deg apart, i.e. the same line. A real
    junction has at least two. Bearings are taken mod pi so direction does not
    matter, then greedily grouped at `tol`.
    """
    if not lanes:
        return 0
    b = []
    for x, y in lanes:
        if len(x) < 2:
            continue
        mx = 0.5 * (x[:-1] + x[1:]); my = 0.5 * (y[:-1] + y[1:])
        d = np.hypot(mx - centre[0], my - centre[1])
        k = d <= radius
        if k.any():
            b.append(np.arctan2(np.diff(y), np.diff(x))[k])
    if not b:
        return 0
    b = np.sort(np.mod(np.concatenate(b), np.pi))   # axis, not direction
    if len(b) == 1:
        return 1
    # Two axes count as distinct only when they differ by >= tol. tol is large
    # (60 deg) ON PURPOSE: a curved road sweeps its bearing continuously, so a
    # small tol splits one curve into several "axes" and lets a mid-block
    # crosswalk on a bend through. A real junction differs by ~90 deg.
    #
    # Do NOT replace this with gap-based clustering on the sorted circle. That
    # was tried and rejected 99% of all zones on the real pool: junction TURN
    # CONNECTORS sweep through every angle between the two roads, so a real
    # junction's bearing distribution is CONTINUOUS, not two discrete clumps.
    # Separation from a representative is what survives connectors.
    reps = []
    for ang in b:
        for g in reps:
            d = abs(ang - g)
            if min(d, np.pi - d) < tol:
                break
        else:
            reps.append(ang)
    return len(reps)


def junction_zones(roads, eps, use_lane_crossings, min_axes=2):
    """Cluster intersection markers into one zone per junction.
    Returns (centres (K,2), source counts dict)."""
    marks, src = [], {"stop_sign": 0, "crosswalk": 0,
                      "lane_cross": 0, "lane_tjunc": 0}

    for rd in roads:
        if rd["type"] == 7:                      # stop sign: single point
            marks.append([rd["x"][0], rd["y"][0]]); src["stop_sign"] += 1
        elif rd["type"] == 8:                    # crosswalk: polygon centroid
            marks.append([rd["x"].mean(), rd["y"].mean()]); src["crosswalk"] += 1

    if use_lane_crossings:
        lanes = [(rd["x"], rd["y"]) for rd in roads if rd["type"] == 4]
        xs = _seg_crossings(lanes)
        src["lane_cross"] = len(xs)
        if len(xs):
            marks.extend(xs.tolist())
        # T-junctions: side lane terminates against a through lane
        ts = _lane_tjunctions(lanes)
        src["lane_tjunc"] = len(ts)
        if len(ts):
            marks.extend(ts.tolist())

    if not marks:
        return np.empty((0, 2)), src
    M = np.asarray(marks, float)
    src["_lanes"] = [(rd["x"], rd["y"]) for rd in roads if rd["type"] == 4]

    # single-link clustering at eps: a 4-way stop has one marker per approach,
    # so the markers of ONE junction must collapse to ONE zone.
    try:
        from sklearn.cluster import DBSCAN
        lab = DBSCAN(eps=eps, min_samples=1).fit_predict(M)
    except Exception:
        lab = _greedy_cluster(M, eps)
    centres = np.array([M[lab == c].mean(0) for c in np.unique(lab)])
    lanes = src.pop("_lanes", [])
    if min_axes > 1 and len(centres):
        keep = np.array([_n_axes(c, lanes) >= min_axes for c in centres])
        src["zones_dropped_1axis"] = int((~keep).sum())
        centres = centres[keep]
    return centres, src


def _greedy_cluster(M, eps):
    lab = -np.ones(len(M), int)
    c = 0
    for i in range(len(M)):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], c
        while stack:
            j = stack.pop()
            d = np.hypot(*(M - M[j]).T)
            new = np.where((d <= eps) & (lab < 0))[0]
            lab[new] = c
            stack.extend(new.tolist())
        c += 1
    return lab


# ---------------------------------------------------------------------------
# Per-trajectory
# ---------------------------------------------------------------------------

def lane_bearing_at(pt, lanes_xy, lanes_bear):
    """Bearing of the nearest lane-centerline segment to pt (NaN if none)."""
    if not len(lanes_xy):
        return np.nan
    d = np.hypot(lanes_xy[:, 0] - pt[0], lanes_xy[:, 1] - pt[1])
    return float(lanes_bear[int(np.argmin(d))])


def scene_rows(m, args):
    roads, objs = m["roads"], m["objects"]
    sid = m["scenario_id"]
    centres, src = junction_zones(roads, args.zone_eps,
                              args.use_lane_crossings, args.min_axes)

    # lane segment midpoints + bearings, for the road-relative turn
    mids, bears = [], []
    for rd in roads:
        if rd["type"] != 4 or rd["n"] < 2:
            continue
        x, y = rd["x"], rd["y"]
        mids.append(np.stack([0.5*(x[:-1]+x[1:]), 0.5*(y[:-1]+y[1:])], 1))
        bears.append(np.arctan2(np.diff(y), np.diff(x)))
    lanes_xy = np.concatenate(mids) if mids else np.empty((0, 2))
    lanes_bear = np.concatenate(bears) if bears else np.empty(0)

    # --- pass 1: which zones does traffic actually use? -------------------
    traj = []
    for ob in objs:
        if ob["type"] != 1:                      # vehicles only
            continue
        v = ob["valid"]
        if v.sum() < 10:
            continue
        x, y, h = ob["x"][v], ob["y"][v], ob["heading"][v]
        if len(x) < 10:
            continue
        spd = np.hypot(np.diff(x), np.diff(y)) * HZ
        if not len(spd) or spd.max() <= MOVE_MS:
            continue                             # parked
        traj.append((ob["id"], x, y, h, spd))

    if len(centres):
        n_trav = np.zeros(len(centres), int)
        for _, x, y, _, _ in traj:
            d = np.hypot(x[:, None] - centres[None, :, 0],
                         y[:, None] - centres[None, :, 1])
            n_trav += (d.min(0) <= args.zone_radius).astype(int)
        active = centres[n_trav >= args.min_traversals]
    else:
        active = np.empty((0, 2))

    # --- pass 2: per-trajectory features ----------------------------------
    rows = []
    for vid, x, y, h, spd in traj:
        if len(active):
            d = np.hypot(x[:, None] - active[None, :, 0],
                         y[:, None] - active[None, :, 1])
            dmin_t = d.min(1)                    # per-step distance to nearest
            inzone = dmin_t <= args.zone_radius
            min_dist = float(dmin_t.min())
            n_zones = int((d.min(0) <= args.zone_radius).sum())
        else:
            inzone = np.zeros(len(x), bool)
            min_dist, n_zones = np.inf, 0

        # moving mask aligned to the per-step speed array
        mov = np.concatenate([[spd[0] > MOVE_MS], spd > MOVE_MS])
        involved = bool((inzone & mov).any())

        s_in  = spd[inzone[:-1] & (spd > 0)] if inzone[:-1].any() else np.array([])
        s_out = spd[~inzone[:-1]] if (~inzone[:-1]).any() else np.array([])

        dh = wrap(np.diff(h))
        tot_turn = float(np.abs(dh).sum())
        in_seg = inzone[:-1] & inzone[1:]
        turn_in = float(np.abs(dh[in_seg]).sum()) if in_seg.any() else 0.0
        # WHERE the heading change happens. A junction turn concentrates it
        # inside the zone; a curved road spreads it along the whole path. This
        # replaces road_rel_turn, which could not discriminate: at a junction
        # the nearest lane at the END is the NEW road, so the road bearing
        # change already contains the turn and the residual is ~0 either way
        # (measured: 0.172 for junction turns vs 0.531 for the rest -- backwards).
        turn_in_zone_frac = (turn_in / tot_turn) if tot_turn > 1e-6 else 0.0
        net_turn = float(dh.sum())
        b0 = lane_bearing_at((x[0],  y[0]),  lanes_xy, lanes_bear)
        b1 = lane_bearing_at((x[-1], y[-1]), lanes_xy, lanes_bear)
        road_turn = float(wrap(b1 - b0)) if np.isfinite(b0 * b1) else np.nan
        road_rel  = float(wrap(net_turn - road_turn)) if np.isfinite(road_turn) else np.nan

        rows.append(dict(
            scenario_id=sid, vehicle_id=int(vid),
            intersection=int(involved),
            n_active_zones_scene=int(len(active)),
            n_zones_traversed=n_zones,
            min_dist_to_zone=round(min_dist, 2) if np.isfinite(min_dist) else "",
            frac_steps_in_zone=round(float(inzone.mean()), 4),
            speed_mean=round(float(spd.mean()), 3),
            speed_max=round(float(spd.max()), 3),
            speed_in_zone=round(float(s_in.mean()), 3) if len(s_in) else "",
            speed_out_zone=round(float(s_out.mean()), 3) if len(s_out) else "",
            stop_frac=round(float((spd < MOVE_MS).mean()), 4),
            net_turn=round(net_turn, 4),
            abs_net_turn=round(abs(net_turn), 4),
            total_abs_turn=round(tot_turn, 4),
            turn_in_zone=round(turn_in, 4),
            turn_in_zone_frac=round(turn_in_zone_frac, 4),
            road_bearing_change=round(road_turn, 4) if np.isfinite(road_turn) else "",
            road_rel_turn=round(road_rel, 4) if np.isfinite(road_rel) else "",
            abs_road_rel_turn=round(abs(road_rel), 4) if np.isfinite(road_rel) else "",
        ))
    return rows, src, active, centres


def main():
    args = parse_args()
    files = sorted(Path(args.data_dir).glob("map_*.bin"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no map_*.bin in {args.data_dir}")
    print(f"[isect] {len(files)} maps  eps={args.zone_eps} R={args.zone_radius} "
          f"lane_crossings={bool(args.use_lane_crossings)}")

    import csv
    rows_all, tot_src = [], {"stop_sign": 0, "crosswalk": 0,
                             "lane_cross": 0, "lane_tjunc": 0}
    n_zone, n_act, bad, n_drop = 0, 0, 0, 0
    for i, fp in enumerate(files):
        try:
            m = read_map_binary(fp)
            r, src, active, centres = scene_rows(m, args)
            na, nz = len(active), len(centres)
        except Exception as e:
            bad += 1
            if bad <= 3:
                print(f"  FAIL {fp.name}: {type(e).__name__}: {e}")
            continue
        rows_all += r
        n_drop += src.pop("zones_dropped_1axis", 0)
        for k in tot_src:
            tot_src[k] += src.get(k, 0)
        n_zone += nz; n_act += na
        if args.progress and (i + 1) % args.progress == 0:
            print(f"  {i+1}/{len(files)}  rows={len(rows_all)}", flush=True)

    if not rows_all:
        raise SystemExit("no trajectories extracted")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
        w.writeheader(); w.writerows(rows_all)

    n = len(rows_all)
    inv = sum(r["intersection"] for r in rows_all)
    print(f"\n  maps ok={len(files)-bad} failed={bad}")
    print(f"  markers: {tot_src}")
    print(f"  zones dropped for having only ONE road axis: {n_drop}")
    print(f"  zones: {n_zone} found, {n_act} ACTIVE "
          f"({100*n_act/max(n_zone,1):.1f}% used by traffic)")
    print(f"  trajectories: {n}")
    print(f"  intersection-involved: {inv} ({100*inv/n:.1f}%)")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
