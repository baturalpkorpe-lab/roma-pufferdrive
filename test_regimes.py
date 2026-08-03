"""
test_regimes.py -- synthetic self-test for conflict_metrics.py: hand-built
geometry with answers known in closed form.

Run this BEFORE submitting regime_extract.py to the queue. The conflict
geometry is subtle enough that it already caught one real defect: a head-on
pair was classified as "merging" because the heading difference was folded to
an axis (min(dh, pi-dh)), which is correct for counting road axes and wrong
here -- it would have put every oncoming car on a two-way street into the
conflict regime.

    python test_regimes.py     # ~1 s, no data, no env
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conflict_metrics as CM

T = 91
HZ = CM.HZ
fails = []


def check(name, got, want, tol=None):
    ok = (abs(got - want) <= tol) if tol is not None else (got == want)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got={got} want={want}"
          f"{'' if tol is None else f' +-{tol}'}")
    if not ok:
        fails.append(name)


def mk(vid, x, y, h, i0=0, length=4.5, width=2.0):
    """Track dict as vehicle_tracks would build it."""
    x, y, h = map(np.asarray, (x, y, h))
    v = np.hypot(np.gradient(x), np.gradient(y)) * HZ
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return dict(id=vid, i0=i0, n=len(x), x=x, y=y, h=h, v=v, s=s,
                length=length, width=width)


def straight(vid, x0, y0, vx, vy, n=T, i0=0):
    t = np.arange(n) / HZ
    return mk(vid, x0 + vx * t, y0 + vy * t,
              np.full(n, np.arctan2(vy, vx)), i0=i0)


print("\n1. CROSSING at the origin, 90 deg, A arrives first")
# A goes +x at 10 m/s starting 50 m before origin -> reaches origin at t=5.0
# B goes +y at 10 m/s starting 60 m before origin -> reaches origin at t=6.0
A = straight(1, -50, 0, 10, 0)
B = straight(2, 0, -60, 0, 10)
cp = CM.conflict_point(A, B)
check("conflict found", cp is not None, True)
check("cp.x", float(cp["pt"][0]), 0.0, 0.5)
check("cp.y", float(cp["pt"][1]), 0.0, 0.5)
check("geometric", cp["geometric"], "cross")
kind, dh = CM.classify_conflict(A, B, cp)
check("kind", kind, "crossing")
check("heading diff deg", dh, 90.0, 5.0)
mt = CM.conflict_metrics(A, B, cp, kind)
check("leader is A", mt["leader_id"], 1)
check("follower is B", mt["follower_id"], 2)
check("t_leader_arrive", mt["t_leader_arrive"], 5.0, 0.15)
check("t_follower_arrive", mt["t_follower_arrive"], 6.0, 0.15)
# leader clears once >  half-length+1 = 3.25 m past origin -> ~0.33 s later
check("PET ~ 1.0 - clear time", mt["pet"], 0.67, 0.2)
# at the decision moment the follower is 3 s out, i.e. t=3.0, d=30 m;
# leader then is at x=-20, so 2 s out -> TA = 3 - 2 = +1
check("TA at decision", float(mt["ta_at_decision"]), 1.0, 0.25)
check("both passed", mt["both_passed"], 1)

print("\n2. MERGING: parallel paths 1.2 m apart converging, no intersection")
n = T
t = np.arange(n) / HZ
# A on y=0 heading +x; B on y=1.2 heading +x, slightly behind
A2 = straight(1, -40, 0.0, 12, 0)
B2 = straight(2, -55, 1.2, 12, 0)
cp2 = CM.conflict_point(A2, B2, buffer=2.0)
check("merging conflict found", cp2 is not None, True)
check("geometric is buffer", cp2["geometric"], "buffer")
kind2, dh2 = CM.classify_conflict(A2, B2, cp2)
check("kind", kind2, "merging")
check("heading diff ~0", dh2, 0.0, 5.0)

print("\n3. NO conflict: parallel, 30 m apart, never within buffer")
A3 = straight(1, -40, 0, 12, 0)
B3 = straight(2, -40, 30, 12, 0)
check("no conflict", CM.conflict_point(A3, B3, 2.0), None)

print("\n4. Oncoming traffic in ADJACENT lanes must be rejected, not 'merging'")
# 1.5 m apart, opposite directions: paths never intersect, so this is a buffer
# contact at ~180 deg -> ordinary two-way traffic, not a conflict.
A4 = straight(1, -40, 0.0, 12, 0)
B4 = straight(2, 40, 1.5, -12, 0)
cp4 = CM.conflict_point(A4, B4, 2.0)
check("buffer contact found", cp4 is not None and cp4["geometric"] == "buffer", True)
k4, d4 = CM.classify_conflict(A4, B4, cp4)
check("oncoming rejected", k4, None)
check("heading diff ~180 (not folded)", d4, 180.0, 5.0)

print("\n4b. Unprotected left turn across oncoming traffic STAYS a conflict")
# A heads +x; B comes from +x heading -x then turns left across A's path.
t = np.arange(T) / HZ
bx = np.where(t < 4.0, 40 - 10 * t, 0.0 - 0.0 * t)
by = np.where(t < 4.0, 3.0, 3.0 - 10 * (t - 4.0))
bh = np.where(t < 4.0, np.pi, -np.pi / 2)
A4b = straight(1, -50, 0.0, 12, 0)
B4b = mk(2, bx, by, bh)
cp4b = CM.conflict_point(A4b, B4b, 2.0)
check("true path intersection", cp4b is not None and cp4b["geometric"] == "cross", True)
k4b, d4b = CM.classify_conflict(A4b, B4b, cp4b)
check("kept as crossing", k4b, "crossing")

print("\n5. dense_scene + leaders_dense: B follows A at 20 m")
A5 = straight(1, 0, 0, 10, 0)
B5 = straight(2, -20, 0, 10, 0)
D = CM.dense_scene([A5, B5], T)
check("dense X shape", D["X"].shape, (2, T))
lead = CM.leaders_dense(D)
check("A has no leader", int(lead[0].max()), -1)
check("B's leader is index 0", int(np.bincount(lead[1][lead[1] >= 0]).argmax()), 0)
segs = CM.following_segments(lead[1])
check("one following segment", len(segs), 1)
Th, s0, nst, njam = CM.headway_params(D, 1, segs)
# gap = 20 - 4.5 = 15.5 m at 10 m/s -> T = 1.55 s
check("headway_T", Th, 1.55, 0.05)

print("\n6. leaders_dense rejects a car 10 m to the SIDE")
A6 = straight(1, 0, 10, 10, 0)
B6 = straight(2, -20, 0, 10, 0)
lead6 = CM.leaders_dense(CM.dense_scene([A6, B6], T))
check("no leader across 10 m lateral", int(lead6.max()), -1)

print("\n7. freeflow + scene reference speed")
D7 = CM.dense_scene([straight(1, 0, 0, 15, 0), straight(2, 0, 40, 8, 0)], T)
l7 = CM.leaders_dense(D7)
ff = [CM.freeflow_mask(D7, e, l7[e], np.empty((0, 2))) for e in range(2)]
check("veh0 free-flow all steps", int(ff[0].sum()), T)
vref = CM.scene_reference_speed(D7, ff)
check("scene ref speed ~15 (p85 of 15 and 8)", vref, 15.0, 1.0)

print("\n8. zone exclusion kills free-flow inside a junction")
ff8 = CM.freeflow_mask(D7, 0, l7[0], np.array([[60.0, 0.0]]), zone_radius=25.0)
check("some steps excluded by zone", int(ff8.sum()) < T, True)

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
