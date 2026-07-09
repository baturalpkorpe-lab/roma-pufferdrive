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
    outm["jerk_abs"]  = float(np.abs(jrk).mean()) if len(jrk) else np.nan
    return outm


def ego_kinematics(p_spd, p_dh):
    """Direct ego kinematics from a focal's per-step speed (m/s) and per-step
    heading-change RATE (rad/s), with physical plausibility masking applied --
    the shared fix for the impossible accel/jerk artifacts. Returns
    speed_mean/max/min/std, accel_abs/std, jerk_abs, turn_abs, n_steps.
    Steps with speed > SPEED_MAX_MS are dropped; accels with |a| > ACCEL_MAX_MS2
    are dropped before jerk. Turn is NOT capped here (real sharp low-speed turns
    are legitimate; the caller filters corrupt GT separately)."""
    ok = p_spd <= SPEED_MAX_MS
    s  = p_spd[ok]
    dh = p_dh[ok] if p_dh is not None else None
    acc = np.diff(s) * 10.0
    acc = acc[np.abs(acc) <= ACCEL_MAX_MS2]
    jrk = np.diff(acc) * 10.0
    return {
        "speed_mean": float(s.mean()) if len(s) else np.nan,
        "speed_max":  float(s.max()) if len(s) else np.nan,
        "speed_min":  float(s.min()) if len(s) else np.nan,
        "speed_std":  float(s.std()) if len(s) else np.nan,
        "accel_abs":  float(np.abs(acc).mean()) if len(acc) else np.nan,
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
        "n_steps":    int(ok.sum()),
        "flag_frac":  round(flag_frac, 4),
    }
