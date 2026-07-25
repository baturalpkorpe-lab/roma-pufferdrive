"""
traj_kinematics.py -- shared kinematic helpers + physical plausibility filter.

One home for the logic that was previously copy-pasted (and drifted) across
role_direct_cluster / role_regime_analysis / role_paired_sweep / renderers:

  - respawn segment cutting (TELEPORT_M)
  - step-level PHYSICAL PLAUSIBILITY masking: differentiated-position data
    produces impossible values (40+ m/s2 "accelerations", 60+ m/s "speeds")
    from residual respawn jumps, GT glitches and valid-gap boundaries.
    Policy: MASK the offending steps (never clip values -- clipping biases
    aggregates toward the cap; never drop a whole agent for one glitch),
    and drop the agent-episode only when too many steps are flagged
    (JUNK_FLAG_FRAC) -- that means the trajectory itself is corrupt.
  - GT trajectory features for trajectory-level stratification
    (distance / mean / max / min speed / net turn).

Physical anchors (10 Hz data):
  SPEED_MAX_MS  = 45 m/s  (162 km/h -- beyond anything in these scenes)
  ACCEL_MAX_MS2 = 10 m/s2 (emergency braking ~9; launch ~5)
  TURN_MAX_RADS = 2.0 rad/s (junk-GT headings ran ~1.9+ mean; real peaks <1)
"""

import numpy as np

TELEPORT_M     = 4.0    # per-step jump above this = respawn discontinuity
SPEED_MAX_MS   = 45.0   # m/s   -- physically implausible above
ACCEL_MAX_MS2  = 10.0   # m/s^2 -- physically impossible above
TURN_MAX_RADS  = 2.0    # rad/s -- corrupted-GT heading detector
JUNK_FLAG_FRAC = 0.2    # >20% flagged steps -> the trajectory is corrupt
MIN_STEPS      = 10     # min usable steps for any aggregate
MOVE_MS        = 1.0    # below this max speed = parked

# Braking-event segmentation (see event_kinematics)
BRAKE_ON_MS2     = 0.5  # |a| below this is coasting, not a braking action
BRAKE_MIN_STEPS  = 3    # an EVENT lasts >= 0.3 s (rejects single-step noise)
HARD_BRAKE_MS2   = 3.0  # naturalistic-driving "harsh braking" line
SEVERE_BRAKE_MS2 = 5.0  # "severe"; still well under the ACCEL_MAX_MS2 mask


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cut_at_teleport(x, y, t_max=None):
    """First index AFTER the first >TELEPORT_M single-step jump (= usable
    segment length). t_max caps it (e.g. GT horizon)."""
    step = np.hypot(np.diff(x), np.diff(y))
    jumps = np.where(step > TELEPORT_M)[0]
    t_end = int(jumps[0] + 1) if len(jumps) else len(x)
    return min(t_end, t_max) if t_max is not None else t_end


def plausible_steps(spd, dh_rate=None):
    """Step-level plausibility mask over per-step speeds (m/s) and optional
    per-step |heading-rate| (rad/s). True = keep."""
    ok = spd <= SPEED_MAX_MS
    if dh_rate is not None:
        ok &= np.abs(dh_rate) <= TURN_MAX_RADS
    return ok


def masked_kinematics(spd, dh, mask):
    """Aggregate speed/accel/jerk/turn over plausibility-masked steps.
    accel is additionally masked by ACCEL_MAX_MS2 (needs consecutive kept
    speeds, so it is computed on the masked series -- a coarse but unbiased
    approximation at these flag rates <~1%)."""
    s = spd[mask]
    d = dh[mask] if dh is not None else None
    outm = {
        "speed_mean": float(s.mean()) if len(s) else np.nan,
        "speed_max":  float(s.max()) if len(s) else np.nan,
        "speed_min":  float(s.min()) if len(s) else np.nan,
        "turn_abs":   float(np.abs(d).mean()) if d is not None and len(d) else np.nan,
    }
    acc = np.diff(s) * 10.0
    acc = acc[np.abs(acc) <= ACCEL_MAX_MS2]
    jrk = np.diff(acc) * 10.0
    outm["accel_abs"] = float(np.abs(acc).mean()) if len(acc) else np.nan
    # Split throttle vs braking: |a| conflates them, so accel_abs is a mushy
    # signal. accel_pos + decel_abs == accel_abs exactly (|a|=max(a,0)+max(-a,0)).
    outm["accel_pos"] = float(np.maximum(acc, 0).mean()) if len(acc) else np.nan
    outm["decel_abs"] = float(np.maximum(-acc, 0).mean()) if len(acc) else np.nan
    outm["jerk_abs"]  = float(np.abs(jrk).mean()) if len(jrk) else np.nan
    return outm


def accel_series(p_spd, t_idx=None):
    """Per-step acceleration and jerk with the TIME INDEX preserved.

    Why this exists rather than np.diff(spd[ok]): compacting the speed array
    before differencing silently differences ACROSS the holes. If step 40 is
    dropped, the pair (39, 40) of the compacted array spans 0.2 s of real time
    but is still multiplied by 10 as though it were 0.1 s -- manufacturing a
    ~2x acceleration at the boundary of every masked region. Harmless inside a
    mean; not harmless inside a p95, where those manufactured values land
    exactly in the tail being measured.

    Here a[i] is defined only when speed samples i and i+1 are BOTH plausible
    AND adjacent in real time. Everything else is NaN, which keeps the array
    aligned to the time axis instead of resampling it.

    p_spd : (N,) per-step speeds, m/s at 10 Hz.
    t_idx : (N,) original step index of each speed sample. Callers that
            already compacted their series (e.g. through a consecutive-valid
            GT mask) MUST pass the original indices -- np.where(pair)[0] --
            or the gaps stay invisible here too. Defaults to arange(N).

    Returns (acc, jrk, info). acc is (N-1,) m/s2 and jrk is (N-2,) m/s3, both
    NaN where undefined. acc is additionally NaN where |a| > ACCEL_MAX_MS2:
    those are broken-sim / respawn artifacts, never real driving, and the cap
    sits above the tire limit (~8 m/s2) so it cannot clip genuine braking.

    info["accel_mask_frac"] is the fraction of otherwise-defined accelerations
    that hit that cap. AUDIT IT PER CONDITION before trusting any tail
    statistic: masking deletes the largest samples, so a condition that breaks
    the sim more often is truncated harder and its tail is biased low relative
    to the others -- a confound that survives within-(map, vehicle) pairing.
    """
    p_spd = np.asarray(p_spd, dtype=float)
    n = len(p_spd)
    if n < 2:
        empty = np.full(max(n - 1, 0), np.nan)
        return empty, np.full(max(n - 2, 0), np.nan), {
            "n_accel": 0, "n_accel_defined": 0, "accel_mask_frac": np.nan}
    t_idx = np.arange(n) if t_idx is None else np.asarray(t_idx)

    spd_ok = p_spd <= SPEED_MAX_MS
    adj    = (np.diff(t_idx) == 1) & spd_ok[:-1] & spd_ok[1:]
    acc    = np.where(adj, np.diff(p_spd) * 10.0, np.nan)

    n_defined = int(adj.sum())
    over      = adj & (np.abs(np.nan_to_num(acc)) > ACCEL_MAX_MS2)
    n_masked  = int(over.sum())
    acc[over] = np.nan

    # Jerk needs two DEFINED, time-adjacent accelerations. a[i] and a[i+1]
    # both being defined already implies t_idx[i..i+2] are consecutive, so
    # finiteness is the whole condition.
    j_ok = np.isfinite(acc[:-1]) & np.isfinite(acc[1:])
    jrk  = np.where(j_ok, np.diff(acc) * 10.0, np.nan)

    return acc, jrk, {
        "n_accel":         int(np.isfinite(acc).sum()),
        "n_accel_defined": n_defined,
        "accel_mask_frac": (n_masked / n_defined) if n_defined else np.nan,
    }


def _nan(f, x, *a, **k):
    """Apply a nan-aware reducer, returning NaN on an all-NaN/empty input
    instead of warning."""
    x = x[np.isfinite(x)]
    return float(f(x, *a, **k)) if len(x) else np.nan


def event_kinematics(p_spd, t_idx=None):
    """Tail-and-event description of ONE agent-episode's acceleration.

    The averages in ego_kinematics structurally cannot express aggression.
    Over a fixed horizon,

        mean|a| = 10 * TV(speed) / N

    i.e. the TOTAL VARIATION of the speed profile divided by episode length.
    A 12->0 m/s stop delivered as -8 m/s2 over 1.5 s and the same stop
    delivered as -2 m/s2 over 6 s have identical TV, hence identical
    accel_abs / decel_abs (1.33 m/s2 apiece). And since accel_pos is speed
    GAINED / T and decel_abs is speed LOST / T, a trip that starts and ends at
    a similar speed has accel_pos ~ decel_abs ~ accel_abs/2 -- three columns
    carrying roughly one degree of freedom.

    Everything here separates that pair (decel_p95 8.0 vs 2.0, decel_max 8.0
    vs 2.0, hard_brake_rate 0.167 vs 0.0).

    Two families:

    TAIL/SHAPE -- the same series described by its extremes instead of its
    mean. decel_p95 is taken over ALL defined steps (zeros included), so it
    mixes how hard with how often; brake_peak_* below is the pure-intensity
    counterpart. jerk_spikiness = rms/mean is ~1 for a steady signal and grows
    with spikes. NOTE that with a discrete action grid jerk fires whenever the
    sampled action changes level, so it measures ACTION CHURN, not physical
    comfort -- name it accordingly in any writeup.

    EVENTS -- contiguous runs of a < -BRAKE_ON_MS2 lasting >= BRAKE_MIN_STEPS,
    described per event and then aggregated. abruptness = peak / mean within
    the event: 1.0 for a brake held at a constant level, larger when one spike
    sits inside an otherwise soft event. Event RATE is normalised per 100 m
    travelled, not per episode -- per-episode would hand a free "calm" score
    to any car that spent half the window stopped.

    Intensity fields are NaN for an agent that never braked; rate fields are
    0.0. That is deliberate: a paired comparison of brake_peak_mean should
    only use agents that braked in BOTH conditions, while a comparison of
    n_brake_events should keep the zeros.
    """
    p_spd = np.asarray(p_spd, dtype=float)
    acc, jrk, info = accel_series(p_spd, t_idx)
    out = dict(info)

    fin  = np.isfinite(acc)
    a    = acc[fin]
    dec  = np.maximum(-a, 0.0)          # braking magnitude, 0 when throttling
    thr  = np.maximum(a,  0.0)          # throttle magnitude, 0 when braking

    out["decel_p95"]  = _nan(np.percentile, dec, 95) if len(dec) else np.nan
    out["decel_max"]  = _nan(np.max, dec)
    out["accel_p95"]  = _nan(np.percentile, thr, 95) if len(thr) else np.nan
    out["accel_max"]  = _nan(np.max, thr)
    out["jerk_p95"]   = (_nan(np.percentile, np.abs(jrk), 95)
                         if np.isfinite(jrk).any() else np.nan)
    jf = jrk[np.isfinite(jrk)]
    out["jerk_rms"]   = float(np.sqrt((jf ** 2).mean())) if len(jf) else np.nan
    jm = float(np.abs(jf).mean()) if len(jf) else np.nan
    out["jerk_spikiness"] = (out["jerk_rms"] / jm
                             if jm and np.isfinite(jm) and jm > 1e-9 else np.nan)

    # Fisher kurtosis of the signed acceleration: low = one consistent driving
    # intensity, high = mostly calm with rare violent excursions.
    if len(a) > 3 and a.std() > 1e-9:
        out["accel_kurt"] = float(((a - a.mean()) ** 4).mean() / a.var() ** 2 - 3.0)
    else:
        out["accel_kurt"] = np.nan

    out["hard_brake_rate"]   = (float((dec > HARD_BRAKE_MS2).mean())
                                if len(dec) else np.nan)
    out["severe_brake_rate"] = (float((dec > SEVERE_BRAKE_MS2).mean())
                                if len(dec) else np.nan)

    # ---- Braking events ----------------------------------------------------
    braking = np.zeros(len(acc), dtype=bool)
    np.greater(-acc, BRAKE_ON_MS2, out=braking, where=np.isfinite(acc))
    edges  = np.flatnonzero(np.diff(np.concatenate(
        ([0], braking.astype(np.int8), [0]))))
    starts, ends = edges[0::2], edges[1::2]

    peaks, durs, dvs, abrupts = [], [], [], []
    for s, e in zip(starts, ends):
        if e - s < BRAKE_MIN_STEPS:
            continue
        seg = -acc[s:e]                     # all finite and > BRAKE_ON_MS2
        peaks.append(float(seg.max()))
        durs.append(len(seg) * 0.1)
        dvs.append(float(seg.sum()) * 0.1)  # m/s shed over the event
        abrupts.append(float(seg.max() / seg.mean()))

    dist = float(p_spd[p_spd <= SPEED_MAX_MS].sum()) * 0.1   # m travelled
    out["distance_m"]      = dist
    out["n_brake_events"]  = len(peaks)
    out["brake_per_100m"]  = (len(peaks) / (dist / 100.0)) if dist > 1.0 else np.nan
    out["brake_peak_mean"] = float(np.mean(peaks))   if peaks else np.nan
    out["brake_peak_max"]  = float(np.max(peaks))    if peaks else np.nan
    out["brake_dur_mean"]  = float(np.mean(durs))    if durs  else np.nan
    out["brake_dv_mean"]   = float(np.mean(dvs))     if dvs   else np.nan
    out["brake_abrupt_mean"] = float(np.mean(abrupts)) if abrupts else np.nan
    return out


def ego_kinematics(p_spd, p_dh, t_idx=None):
    """Direct ego kinematics from a focal's per-step speed (m/s) and per-step
    heading-change RATE (rad/s), with physical plausibility masking applied --
    the shared fix for the impossible accel/jerk artifacts. Returns
    speed_mean/max/min/std, accel_abs/pos/std, decel_abs, jerk_abs, turn_abs,
    n_steps (accel_pos=throttle, decel_abs=braking; accel_pos+decel_abs==accel_abs).
    Steps with speed > SPEED_MAX_MS are dropped; accels with |a| > ACCEL_MAX_MS2
    are dropped before jerk. Turn is NOT capped here (real sharp low-speed turns
    are legitimate; the caller filters corrupt GT separately).

    Acceleration now comes from accel_series (see there): differences are taken
    only between time-ADJACENT speed samples, where before the speed array was
    compacted first and differenced across the holes. Values move slightly --
    by roughly the mask rate, <~1% -- and the change is a correctness fix, not
    a redefinition.

    Pass t_idx when p_spd was already compacted by the caller, so the gaps stay
    visible. Every accel_abs/pos/decel/jerk aggregate here is a TIME AVERAGE
    and therefore fixed by the total variation of the speed profile; see
    event_kinematics for the tail and event statistics that are not."""
    p_spd = np.asarray(p_spd, dtype=float)
    ok = p_spd <= SPEED_MAX_MS
    s  = p_spd[ok]
    dh = p_dh[ok] if p_dh is not None else None
    acc, jrk, _ = accel_series(p_spd, t_idx)
    acc = acc[np.isfinite(acc)]
    jrk = jrk[np.isfinite(jrk)]
    return {
        "speed_mean": float(s.mean()) if len(s) else np.nan,
        "speed_max":  float(s.max()) if len(s) else np.nan,
        "speed_min":  float(s.min()) if len(s) else np.nan,
        "speed_std":  float(s.std()) if len(s) else np.nan,
        "accel_abs":  float(np.abs(acc).mean()) if len(acc) else np.nan,
        "accel_pos":  float(np.maximum(acc, 0).mean()) if len(acc) else np.nan,
        "decel_abs":  float(np.maximum(-acc, 0).mean()) if len(acc) else np.nan,
        "accel_std":  float(acc.std()) if len(acc) else np.nan,
        "jerk_abs":   float(np.abs(jrk).mean()) if len(jrk) else np.nan,
        "turn_abs":   float(np.abs(dh).mean()) if dh is not None and len(dh) else np.nan,
        "n_steps":    int(len(s)),
    }


def gt_traj_features(gx, gy, gh, valid):
    """Trajectory-stratification features for ONE vehicle's GT arrays (T,).

    Returns dict(distance, speed_mean, speed_max, speed_min, net_turn,
    n_steps, flag_frac) or None if unusable (too short / parked / corrupt).
    net_turn = |sum of wrapped per-step heading changes| in rad -- the ROUTE
    shape (did this seat have to take a corner/exit), robust to multi-wrap.
    """
    m = valid.astype(bool)
    pair = m[:-1] & m[1:]                       # consecutive-valid steps only
    if pair.sum() < MIN_STEPS:
        return None
    dx  = (gx[1:] - gx[:-1])[pair]
    dy  = (gy[1:] - gy[:-1])[pair]
    spd = np.hypot(dx, dy) * 10.0               # m/s at 10 Hz
    dh  = wrap_angle((gh[1:] - gh[:-1])[pair])  # rad/step
    ok  = plausible_steps(spd, dh_rate=dh * 10.0)

    flag_frac = float(1.0 - ok.mean())
    if flag_frac > JUNK_FLAG_FRAC:
        return None                             # corrupt GT (old junk maps)
    if ok.sum() < MIN_STEPS:
        return None
    s = spd[ok]
    if s.max() <= MOVE_MS:
        return None                             # parked
    return {
        "distance":   float((s / 10.0).sum()),  # m over kept steps
        "speed_mean": float(s.mean()),
        "speed_max":  float(s.max()),
        "speed_min":  float(s.min()),
        "net_turn":   float(abs(dh[ok].sum())), # rad, absolute (no L/R split)
        "stop_frac":  float((s < MOVE_MS).mean()),  # fraction of steps stopped
                                                     # (bimodal: hit a queue/light
                                                     # or free-flowed)
        "n_steps":    int(ok.sum()),
        "flag_frac":  round(flag_frac, 4),
    }


# ---------------------------------------------------------------------------
# Spatio-temporal path crossings (interaction feature)
# ---------------------------------------------------------------------------
# A trajectory "interacts" (a conflict) with another when their GT PATHS cross
# geometrically AND both cars pass the crossing point within CROSS_DT seconds
# of each other. Direction-agnostic but temporally gated, so it excludes
# oncoming/parallel cars (paths never cross) and same-lane followers (paths
# overlap, don't cross) -- only genuine conflicts (intersections, turns across
# traffic, merges) count. Likely bimodal (had a conflict or didn't).

CROSS_DT_S = 1.0    # seconds -- max time gap at the crossing point


def _time_segments(x, y, valid):
    """Consecutive-in-time valid segments of one GT path: endpoints P1,P2 and
    their integer step times t1,t2 (=step indices). None if too short."""
    v = np.asarray(valid, bool)
    seg = v[:-1] & v[1:]                          # consecutive valid steps
    a = np.where(seg)[0]
    if len(a) < 1:
        return None
    P1 = np.stack([x[a],   y[a]],   axis=1).astype(np.float64)   # (S,2)
    P2 = np.stack([x[a+1], y[a+1]], axis=1).astype(np.float64)
    return P1, P2, a.astype(np.float64), (a + 1).astype(np.float64)


def _min_cross_dt(sa, sb, dt_steps):
    """Min |t_a - t_b| over all geometric crossings of two time-tagged paths,
    if any crossing is within dt_steps; else None. Vectorized over segment
    pairs."""
    P1, P2, ta1, _ = sa
    Q1, Q2, tb1, _ = sb
    # cheap bbox reject
    if (max(P1[:, 0].max(), P2[:, 0].max()) < min(Q1[:, 0].min(), Q2[:, 0].min())
        or min(P1[:, 0].min(), P2[:, 0].min()) > max(Q1[:, 0].max(), Q2[:, 0].max())
        or max(P1[:, 1].max(), P2[:, 1].max()) < min(Q1[:, 1].min(), Q2[:, 1].min())
        or min(P1[:, 1].min(), P2[:, 1].min()) > max(Q1[:, 1].max(), Q2[:, 1].max())):
        return None
    r = (P2 - P1)[:, None, :]                     # (Na,1,2)
    s = (Q2 - Q1)[None, :, :]                      # (1,Nb,2)
    denom = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]     # (Na,Nb)
    with np.errstate(divide="ignore", invalid="ignore"):
        qp = Q1[None, :, :] - P1[:, None, :]       # (Na,Nb,2)
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / denom
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / denom
    hit = (denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    if not hit.any():
        return None
    tA = ta1[:, None] + t                          # crossing time on A (steps)
    tB = tb1[None, :] + u
    dt = np.abs(tA - tB)
    dt = np.where(hit, dt, np.inf)
    m = float(dt.min())
    return m if m <= dt_steps else None


def scene_crossings(gx, gy, valid, vids, dt_s=CROSS_DT_S, hz=10):
    """Per-vehicle conflict-crossing features for one scene's GT.

    gx, gy, valid: (V, T) arrays for the scene's vehicle slots; vids: (V,) ids.
    Returns dict vid -> {n_cross, min_cross_dt (s)}: number of DISTINCT other
    vehicles whose path this one crosses within dt_s, and the tightest timing.
    """
    dt_steps = dt_s * hz
    segs = {}
    for k in range(len(vids)):
        s = _time_segments(gx[k], gy[k], valid[k])
        if s is not None:
            segs[k] = s
    n     = {int(v): 0 for v in vids}
    mindt = {int(v): np.inf for v in vids}
    ks = list(segs)
    for ii in range(len(ks)):
        for jj in range(ii + 1, len(ks)):
            i, j = ks[ii], ks[jj]
            d = _min_cross_dt(segs[i], segs[j], dt_steps)
            if d is not None:
                vi, vj = int(vids[i]), int(vids[j])
                n[vi] += 1
                n[vj] += 1
                d_s = d / hz
                mindt[vi] = min(mindt[vi], d_s)
                mindt[vj] = min(mindt[vj], d_s)
    return {int(v): {"n_cross": n[int(v)],
                     "min_cross_dt": (None if np.isinf(mindt[int(v)])
                                      else round(mindt[int(v)], 3))}
            for v in vids}
