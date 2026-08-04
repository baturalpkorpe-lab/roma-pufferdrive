"""
conflict_metrics.py -- regime membership and driving-style parameters from raw
trajectory arrays. Shared core: works on GT tracks (from map binaries) and on
policy rollouts, so the human reference and the swept policy are measured by
exactly the same code.

WHY THIS EXISTS
The stop_frac clustering (fast / stop&go / mid-speed / turning) is built from
six EGO-KINEMATIC features -- distance, speed_mean, speed_max, speed_min,
net_turn, stop_frac (traj_cluster_view.py ALL_FEATURES). Every one of them is
an OUTCOME of the behaviour we then try to move with a role sweep. Inside a
speed-defined stratum the speed variance is compressed by construction, so the
sweep has almost nothing left to move -- which is why the effect is only
visible in the "fast" cluster, where the binding constraint is slack in both
directions.

So every stratum here is built ONLY from things the policy does not control:
map geometry, the presence and motion of OTHER agents, and the t=0 state.
Never episode aggregates of the ego. The style parameters below are the
RESPONSE, and each is identified in exactly one regime:

  A. conflict  (junction + a partner whose path crosses/merges with ego's)
       -> gap acceptance / assertiveness: critical gap from the go/yield
          decision, plus PET / minTTC / MRD / speed at the conflict point
  B. following (a persistent leader ahead, no conflict)
       -> desired time headway T, jam spacing s0
  C. free-flow (no leader, no conflict)
       -> desired speed, as a RATIO to what the scene affords

The ratio in C matters: "the speed margin in these maps is way open" is a
scene property, and dividing by a scene reference speed removes it from the
metric instead of leaving it to be absorbed by the stratum.

NOTE ON WHAT THE POLICY CAN SEE (verified in PufferDrive drive.h, Emerge-Lab
main):
  * init_grid_map() gates on `type > 3 && type < 7` with the comment
    "Only Road Edges, Lines, and Lanes in grid map". STOP_SIGN (7),
    CROSSWALK (8), SPEED_BUMP (9) and DRIVEWAY (10) never reach the
    observation.
  * There is no traffic-light / signal state anywhere in the env.
So right-of-way is INVISIBLE to the agent. Stop signs are still the best
available label for "equal priority, genuinely contested" -- but that is an
ANALYST-side label about the scene, not something the policy could have read.
Treat any behaviour difference across control types as emergent from geometry
and partner motion, never as sign compliance.

Conflict definitions follow Rahmani, Xu, Calvert & van Arem, "Automated
Vehicles at Unsignalized Intersections", TRR 2025 (DOI
10.1177/03611981251370343), with three deliberate departures documented at
their call sites:
  1. no "speed changed by > 3 m/s" gate -- that selects on the outcome,
  2. PET is not a gate (the window is 9.1 s, their threshold was 10 s),
  3. merging vs crossing from geometry, not lane ids (the road id is a
     SEGMENT id here, so "same lane after" is not computable).
"""

import numpy as np

HZ = 10.0
MOVE_MS = 1.0          # max speed below this = parked (matches traj_kinematics)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def _longest_valid_run(v):
    """Start and stop of the longest contiguous True run in v."""
    if not v.any():
        return 0, 0
    idx = np.flatnonzero(np.diff(np.concatenate([[0], v.view(np.int8), [0]])))
    starts, stops = idx[0::2], idx[1::2]
    k = int(np.argmax(stops - starts))
    return int(starts[k]), int(stops[k])


def vehicle_tracks(objects, move_ms=MOVE_MS, min_steps=15):
    """Moving vehicles as dicts on their longest contiguous valid run.

    Keys: id, i0, n, x, y, h, v, s (arc length), width, length.
    Indices are absolute step numbers, so i0 + k indexes the episode.

    The longest-contiguous-run restriction matters: differencing across a
    validity hole invents a speed spike, and every metric here is built on
    speed or arc length.
    """
    out = []
    for ob in objects:
        if ob["type"] != 1:                          # vehicles only
            continue
        v = np.asarray(ob["valid"], bool)
        i0, i1 = _longest_valid_run(v)
        n = i1 - i0
        if n < min_steps:
            continue
        x = np.asarray(ob["x"], float)[i0:i1]
        y = np.asarray(ob["y"], float)[i0:i1]
        h = np.asarray(ob["heading"], float)[i0:i1]
        spd = np.hypot(np.gradient(x), np.gradient(y)) * HZ
        if spd.max() <= move_ms:                     # parked
            continue
        step = np.hypot(np.diff(x), np.diff(y))
        out.append(dict(id=int(ob["id"]), i0=i0, n=n, x=x, y=y, h=h, v=spd,
                        s=np.concatenate([[0.0], np.cumsum(step)]),
                        width=float(ob["width"]), length=float(ob["length"])))
    return out


def _overlap(a, b):
    """Absolute step range where two tracks are both valid."""
    lo = max(a["i0"], b["i0"])
    hi = min(a["i0"] + a["n"], b["i0"] + b["n"])
    return lo, hi


# ---------------------------------------------------------------------------
# Conflict point
# ---------------------------------------------------------------------------

def conflict_point(a, b, buffer=2.0):
    """Where two paths conflict, or None.

    Crossing: the paths actually intersect -> the intersection itself.
    Merging: they never intersect because of the lateral offset inside a lane,
    so Rahmani et al. add a 1 m buffer each side and take the first contact
    between the buffers. Implemented as the earliest pair of path points
    within `buffer`, earliest by i+j so the result does not depend on which
    track is passed first.

    Returns dict(pt, ia, ib, geometric) where ia/ib are indices INTO THE
    TRACKS (not absolute steps) and `geometric` is "cross" if the paths truly
    intersect, else "buffer".
    """
    ax, ay, bx, by = a["x"], a["y"], b["x"], b["y"]

    # cheap reject: bounding boxes further apart than the buffer cannot touch
    if (ax.min() - bx.max() > buffer or bx.min() - ax.max() > buffer or
            ay.min() - by.max() > buffer or by.min() - ay.max() > buffer):
        return None

    # --- true segment intersection (crossing) ------------------------------
    p = np.stack([ax[:-1], ay[:-1]], 1)[:, None, :]      # (na,1,2)
    r = np.stack([np.diff(ax), np.diff(ay)], 1)[:, None, :]
    q = np.stack([bx[:-1], by[:-1]], 1)[None, :, :]      # (1,nb,2)
    s = np.stack([np.diff(bx), np.diff(by)], 1)[None, :, :]
    den = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / den
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / den
    hit = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    if hit.any():
        ia, ib = np.nonzero(hit)
        k = int(np.argmin(ia + ib))                  # earliest joint progress
        i, j = int(ia[k]), int(ib[k])
        pt = p[i, 0] + t[i, j] * r[i, 0]
        return dict(pt=pt, ia=i, ib=j, geometric="cross")

    # --- buffer contact (merging) -----------------------------------------
    d = np.hypot(ax[:, None] - bx[None, :], ay[:, None] - by[None, :])
    close = d <= buffer
    if not close.any():
        return None
    ia, ib = np.nonzero(close)
    k = int(np.argmin(ia + ib))
    i, j = int(ia[k]), int(ib[k])
    pt = np.array([0.5 * (ax[i] + bx[j]), 0.5 * (ay[i] + by[j])])
    return dict(pt=pt, ia=i, ib=j, geometric="buffer")


def approach_offset(a, b, cp, lookback=3.0):
    """Max LATERAL offset of the two upstream approach positions from the
    conflict-point axis.

    Rahmani et al. require merging vehicles to "start from different lanes
    before the intersection". Without lane ids, the geometric equivalent is that
    at least one vehicle approached from ACROSS the axis they end up sharing.

    This is what separates a merge from a QUEUE. Two cars following each other in
    one lane have almost the same path, so a 2 m buffer contact triggers
    immediately, and they were being counted as merging conflicts. On the 10k pool
    that inflated merging to 84% of all conflicts (17246 vs 3270 crossing,
    against a roughly even split in the paper) and gave merging car-following
    fingerprints: minTTC 8.9 s against their 6.1-6.3, MRD 1.14 against their 0.58.

    LATERAL offset, not distance. A follower sits far BEHIND along the axis but
    ON it, so a plain distance test passes as soon as the leader's recorded track
    starts later than the follower's -- measured on the synthetic queue, that
    scored 20 m purely from the lead vehicle's path not extending back that far.
    Only the across-axis component says "different lane".

    Measured between their SIMULTANEOUS positions, not as each one's distance
    from the conflict point. Two reasons. Simultaneous separation splits cleanly
    into a longitudinal part (following) and a lateral part (different lane),
    which is exactly the distinction wanted. And anchoring on the conflict point
    instead leaks longitudinal distance into the lateral term whenever the axis
    is tilted: a vehicle 36 m upstream on a straight path picked up a spurious
    4 m of "lateral" offset from a 6 deg axis tilt, enough to matter against a
    5 m threshold.
    """
    k = int(lookback * HZ)
    sa, sb = a["i0"] + cp["ia"], b["i0"] + cp["ib"]
    s_up = max(sa, sb) - k                           # before the LATER arrival
    ja = int(np.clip(s_up - a["i0"], 0, a["n"] - 1))
    jb = int(np.clip(s_up - b["i0"], 0, b["n"] - 1))

    # Axis = mean heading at that instant. Only ever called where the two are
    # within align_deg of each other, so the mean is meaningful.
    ux = np.cos(a["h"][ja]) + np.cos(b["h"][jb])
    uy = np.sin(a["h"][ja]) + np.sin(b["h"][jb])
    n = np.hypot(ux, uy)
    if n < 1e-9:                                     # exactly opposed: no axis
        return np.inf
    ux, uy = ux / n, uy / n

    dx = b["x"][jb] - a["x"][ja]
    dy = b["y"][jb] - a["y"][ja]
    return float(abs(dx * uy - dy * ux))             # across-axis component


def classify_conflict(a, b, cp, align_deg=30.0, cross_deg=45.0,
                      oncoming_deg=150.0, merge_sep=4.0, merge_hold=1.0,
                      min_approach_offset=5.0):
    """merging | crossing | None, from geometry alone. None = not a conflict.

    Rahmani et al. use lane identity ("different lanes before, SAME lane
    after" = merging). Not available here: the road id is a SEGMENT id, so one
    road is chopped into ~16 m pieces with unique ids and "same lane after"
    cannot be evaluated.

    Substitute: the heading difference at the conflict point. Merging vehicles
    are (almost) parallel there -- the same assumption Rahmani et al. rely on
    for TTC_merge -- while crossing vehicles meet at an angle. The ambiguous
    band between align_deg and cross_deg is broken by asking whether the two
    paths STAY together afterwards, which is what "same lane after" means.
    Post-point evidence is checked second on purpose: the episode is 9.1 s, so
    a conflict late in the window has little path left after it.

    The heading difference is NOT folded to an axis here. Folding (min(dh,
    pi-dh), which is right for counting road axes) makes a head-on pair at
    180 deg look like a parallel pair at 0 deg, so ordinary oncoming traffic on
    a two-way street would be labelled "merging" -- caught by a synthetic
    head-on test. Direction is the whole distinction between merging and
    passing someone.

    Oncoming pairs are rejected ONLY when the paths never actually intersected
    (a buffer contact, i.e. they passed in adjacent lanes). A true path
    intersection between near-opposed vehicles is the unprotected left turn
    across oncoming traffic -- the most safety-relevant conflict there is --
    and it stays in as a crossing.
    """
    dh = abs(float(wrap(a["h"][cp["ia"]] - b["h"][cp["ib"]])))
    deg = np.rad2deg(dh)

    if dh <= np.deg2rad(align_deg):
        # Parallel is necessary but NOT sufficient for a merge: a queue in one
        # lane is parallel too. Demand distinct approaches. Only merging needs
        # this -- a crossing requires a true path intersection at an angle, which
        # same-lane followers cannot produce, and the crossing metrics already
        # reproduce the paper (minTTC 5.03 vs their 4.71-5.08).
        if approach_offset(a, b, cp) < min_approach_offset:
            return None, deg                         # same lane: car-following
        return "merging", deg
    if dh >= np.deg2rad(oncoming_deg) and cp["geometric"] == "buffer":
        return None, deg                             # just passing, not merging
    if dh >= np.deg2rad(cross_deg):
        return "crossing", deg

    k = min(len(a["x"]) - cp["ia"], len(b["x"]) - cp["ib"])
    need = int(merge_hold * HZ)
    if k >= need:
        sep = np.hypot(a["x"][cp["ia"]:cp["ia"] + k] - b["x"][cp["ib"]:cp["ib"] + k],
                       a["y"][cp["ia"]:cp["ia"] + k] - b["y"][cp["ib"]:cp["ib"] + k])
        if (sep[:need] <= merge_sep).all():
            if approach_offset(a, b, cp) < min_approach_offset:
                return None, deg                     # same lane: car-following
            return "merging", deg
    return "crossing", deg


# ---------------------------------------------------------------------------
# Conflict metrics
# ---------------------------------------------------------------------------

def _arrival(tr, pt):
    """(index, absolute step) where the track is closest to pt."""
    i = int(np.argmin(np.hypot(tr["x"] - pt[0], tr["y"] - pt[1])))
    return i, tr["i0"] + i


def _dist_along(tr, i_cp):
    """Distance still to travel to the conflict point, per index.

    Arc length, not straight line: the follower's braking distance is the
    distance along its own path, and on a turning approach those differ a lot.
    """
    return tr["s"][i_cp] - tr["s"]


def conflict_metrics(a, b, cp, kind, tau_dec=3.0, clear_pad=1.0,
                     v_floor=0.5):
    """Safety / assertiveness metrics for one conflict, or None if undefined.

    Leader / follower are the REALISED passing order through the conflict
    point, as in Rahmani et al. That makes them an outcome, not a stratum --
    which is the point: `ego_went_first` is the cleanest possible role
    readout, a bounded binary decision that a disposition should control.

    All continuous metrics are evaluated over ONE window: from the start of
    the overlap until the leader clears the conflict point. Rahmani et al.
    say "until the second vehicle passes the conflict point", but for a
    crossing TTC = d_f / v_f is just the follower's own time-to-arrival, whose
    minimum over that window is ~0 by construction. Cutting the window at the
    moment the leader clears is the reading under which minTTC is a severity
    measure at all, and it reproduces their reported magnitudes (4-7 s).
    """
    lo, hi = _overlap(a, b)
    if hi - lo < 5:
        return None

    ia, sa = _arrival(a, cp["pt"])
    ib, sb = _arrival(b, cp["pt"])
    a_first = sa <= sb
    ld, fo = (a, b) if a_first else (b, a)
    i_ld, i_fo = (ia, ib) if a_first else (ib, ia)

    # leader clears when it is further than its own half-length + pad away
    clear_r = 0.5 * ld["length"] + clear_pad
    d_ld_pt = np.hypot(ld["x"] - cp["pt"][0], ld["y"] - cp["pt"][1])
    after = np.flatnonzero((np.arange(len(d_ld_pt)) >= i_ld) &
                           (d_ld_pt > clear_r))
    i_clear = int(after[0]) if len(after) else len(d_ld_pt) - 1
    s_clear = ld["i0"] + i_clear

    d_fo = _dist_along(fo, i_fo)
    d_ld = _dist_along(ld, i_ld)

    # --- window: overlap start -> leader clears ---------------------------
    w0, w1 = lo, min(s_clear, fo["i0"] + i_fo)
    if w1 <= w0:
        w1 = w0 + 1
    k = np.arange(w0, w1) - fo["i0"]
    k = k[(k >= 0) & (k < fo["n"]) & (d_fo[np.clip(k, 0, fo["n"] - 1)] > 0.1)]

    min_ttc, mrd = np.nan, np.nan
    if len(k):
        v_f = np.maximum(fo["v"][k], v_floor)
        d_f = d_fo[k]
        if kind == "merging":
            kl = np.clip(np.arange(w0, w1)[:len(k)] - ld["i0"], 0, ld["n"] - 1)
            dv = v_f - ld["v"][kl]
            ttc = np.where(dv > v_floor, d_f / np.maximum(dv, v_floor), np.inf)
        else:
            ttc = d_f / v_f
        ttc = ttc[np.isfinite(ttc)]
        if len(ttc):
            min_ttc = float(ttc.min())
        mrd = float((v_f ** 2 / (2.0 * d_f)).max())

    # --- PET: follower arrives minus leader clears ------------------------
    pet = (fo["i0"] + i_fo - s_clear) / HZ

    # --- decision moment: the START of the encounter ----------------------
    # Both projections are read at the SAME instant, and that instant is fixed
    # by co-presence, not by either vehicle's own progress.
    #
    # The previous version searched for the step where the FOLLOWER was tau_dec
    # of travel from the point. That pins t_follower at tau_dec by construction,
    # so TA collapsed to (tau_dec - t_leader): a function of the leader alone,
    # and the leader IS the outcome. Measured on the 10k pool it produced a
    # median TA of 2.99 s against tau_dec=3.0, only 1.1% of conflicts looking
    # contested, and a logistic fit that separated at 98.5% accuracy with the
    # slope running to -59. Circular, and it left no residual variance for a
    # role term to explain.
    #
    # Anchoring on any moment defined by the arrivals has the same defect: the
    # vehicle that arrives first is nearer at every such anchor, so sign(TA)
    # reproduces the outcome. The gap has to be read EARLY, while both are still
    # approaching on pre-negotiation speeds -- which is also where Rahmani et al.
    # find their informative cases, the ones where "the vehicle that ultimately
    # passed first was temporarily projected to arrive second".
    ta_dec, i_dec = np.nan, None
    for s in range(lo, min(sa, sb) + 1):
        ja, jb = s - fo["i0"], s - ld["i0"]
        if not (0 <= ja < fo["n"] and 0 <= jb < ld["n"]):
            continue
        if d_fo[ja] <= 0.5 or d_ld[jb] <= 0.5:       # one already arrived
            continue
        if fo["v"][ja] <= v_floor or ld["v"][jb] <= v_floor:
            continue                                  # stopped: no projection
        t_fo = d_fo[ja] / fo["v"][ja]
        t_ld = d_ld[jb] / ld["v"][jb]
        ta_dec = float(t_fo - t_ld)                   # >0: ego arrives later
        i_dec = ja
        break
    _ = tau_dec                                       # kept for the CSV column

    return dict(
        leader_id=ld["id"], follower_id=fo["id"],
        kind=kind, geometric=cp["geometric"],
        t_leader_arrive=round((ld["i0"] + i_ld) / HZ, 2),
        t_leader_clear=round(s_clear / HZ, 2),
        t_follower_arrive=round((fo["i0"] + i_fo) / HZ, 2),
        pet=round(float(pet), 3),
        min_ttc=round(min_ttc, 3) if np.isfinite(min_ttc) else "",
        mrd=round(mrd, 3) if np.isfinite(mrd) else "",
        ta_at_decision=round(ta_dec, 3) if np.isfinite(ta_dec) else "",
        t_decision=round((fo["i0"] + i_dec) / HZ, 2) if i_dec is not None else "",
        v_follower_at_point=round(float(fo["v"][i_fo]), 3),
        v_leader_at_point=round(float(ld["v"][i_ld]), 3),
        d_follower_at_decision=(round(float(d_fo[i_dec]), 2)
                                if i_dec is not None else ""),
        both_passed=int(sa < a["i0"] + a["n"] - 1 and sb < b["i0"] + b["n"] - 1),
    )


# ---------------------------------------------------------------------------
# Regime B: car following
# ---------------------------------------------------------------------------

def dense_scene(tracks, T=91):
    """Tracks re-expanded onto the full episode grid.

    (n_veh, T) arrays with NaN outside each track's valid run. The pairwise
    work below is O(n_veh^2 * T); on a dense Waymo scene that is a few hundred
    thousand elements, which is fine vectorised and hopeless as a Python loop.
    """
    n = len(tracks)
    X = np.full((n, T), np.nan)
    Y = np.full((n, T), np.nan)
    H = np.full((n, T), np.nan)
    V = np.full((n, T), np.nan)
    for k, tr in enumerate(tracks):
        a, b = tr["i0"], tr["i0"] + tr["n"]
        b = min(b, T)
        w = b - a
        X[k, a:b], Y[k, a:b] = tr["x"][:w], tr["y"][:w]
        H[k, a:b], V[k, a:b] = tr["h"][:w], tr["v"][:w]
    return dict(X=X, Y=Y, H=H, V=V, M=~np.isnan(X),
                ids=np.array([t["id"] for t in tracks], int),
                L=np.array([t["length"] for t in tracks], float),
                W=np.array([t["width"] for t in tracks], float))


def leaders_dense(D, max_range=60.0, lat=1.8, align_deg=25.0, min_gap=1.0):
    """(n_veh, T) index of each vehicle's leader at each step, or -1.

    Purely geometric, no lane graph. PufferDrive gives the policy road
    GEOMETRY only (lanes, lines, edges) and the road id is a SEGMENT id, so
    "same lane" is not computable. The standard trajectory-data substitute is a
    corridor in the ego BODY frame: ahead along the ego heading, within half a
    lane laterally, and pointing the same way. The heading test is what keeps
    oncoming and cross traffic out.
    """
    X, Y, H, M = D["X"], D["Y"], D["H"], D["M"]
    n, T = X.shape
    out = -np.ones((n, T), int)
    ch, sh = np.cos(H), np.sin(H)
    thr = np.deg2rad(align_deg)
    for e in range(n):
        dx, dy = X - X[e], Y - Y[e]                  # (n, T)
        s = dx * ch[e] + dy * sh[e]
        d = np.abs(-dx * sh[e] + dy * ch[e])
        al = np.abs(wrap(H - H[e]))
        ok = (M & M[e] & (s > min_gap) & (s <= max_range) &
              (d <= lat) & (al <= thr))
        ok[e] = False
        sm = np.where(ok, s, np.inf)
        best = np.argmin(sm, axis=0)
        out[e] = np.where(np.isfinite(sm[best, np.arange(T)]), best, -1)
    return out


def following_segments(leader_row, min_hold=3.0):
    """Contiguous runs with the SAME leader, at least min_hold seconds.

    Persistence is the filter that separates real car-following from a vehicle
    that merely swept through the corridor for a few frames at a junction.
    Indices are absolute episode steps.
    """
    need = int(min_hold * HZ)
    segs, t, T = [], 0, len(leader_row)
    while t < T:
        lid = leader_row[t]
        u = t
        while u < T and leader_row[u] == lid:
            u += 1
        if lid >= 0 and u - t >= need:
            segs.append((t, u, int(lid)))
        t = u
    return segs


def headway_params(D, e, segs, a_tol=0.5, v_min=2.0):
    """Desired time headway T and jam spacing s0 for vehicle e.

    T is read only where the pair is in STEADY STATE (|a| small): during a
    transient the gap is whatever the approach left behind, not what the driver
    wants. s0 comes from the opposite end -- steps where both are stopped.
    """
    X, Y, V, L = D["X"], D["Y"], D["V"], D["L"]
    acc = np.gradient(np.nan_to_num(V[e])) * HZ
    Ts, s0s = [], []
    for t0, t1, k in segs:
        t = np.arange(t0, t1)
        gap = (np.hypot(X[k, t] - X[e, t], Y[k, t] - Y[e, t])
               - 0.5 * (L[e] + L[k]))
        ok = np.isfinite(gap) & (gap > 0)
        steady = ok & (V[e, t] >= v_min) & (np.abs(acc[t]) < a_tol)
        jam = ok & (V[e, t] < MOVE_MS) & (V[k, t] < MOVE_MS)
        Ts.append(gap[steady] / V[e, t][steady])
        s0s.append(gap[jam])
    Ts = np.concatenate(Ts) if Ts else np.empty(0)
    s0s = np.concatenate(s0s) if s0s else np.empty(0)
    return (float(np.median(Ts)) if len(Ts) else np.nan,
            float(np.median(s0s)) if len(s0s) else np.nan,
            int(len(Ts)), int(len(s0s)))


# ---------------------------------------------------------------------------
# Regime C: free flow
# ---------------------------------------------------------------------------

def freeflow_mask(D, e, leader_row, zones, zone_radius=25.0, v_min=2.0):
    """Steps for vehicle e with nothing binding: no leader, outside every
    junction zone, moving."""
    m = D["M"][e] & (leader_row < 0) & (np.nan_to_num(D["V"][e]) > v_min)
    if len(zones):
        Z = np.asarray(zones, float).reshape(-1, 2)
        d = np.hypot(D["X"][e][:, None] - Z[None, :, 0],
                     D["Y"][e][:, None] - Z[None, :, 1])
        with np.errstate(invalid="ignore"):
            m &= np.nan_to_num(d.min(1), nan=np.inf) > zone_radius
    return m


def scene_reference_speed(D, masks, q=85.0):
    """What the scene AFFORDS: the q-th percentile of free-flow speed over
    every moving vehicle in it.

    Dividing desired speed by this is what removes "the speed margin in these
    maps is way open" from the metric. speed_mean never had a denominator, so
    a fast map and an assertive driver produced the same number.
    """
    pool = [D["V"][e][m] for e, m in enumerate(masks) if m.any()]
    if not pool:
        return np.nan
    v = np.concatenate(pool)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) else np.nan
