"""
verify_cluster_names.py -- is 0=fast 1=stop&go 2=mid-speed 3=turning correct?

The names are asserted in ~6 places (role_analysis.sbatch, render_role_grid*,
intersection_report, the handoff) and traced back to a comment claiming they
were "verified against the K=4 centroid plot" -- with no artifact in the repo
behind it. The handoff still lists this as open: "verify 0/1/2/3 <-> type by
ground truth, never by label." A wrong mapping silently mislabels every
per-cluster figure and every per-cluster policy.

trajectory_clusters_stopfrac.csv holds only labels (scenario_id, vehicle_id,
cluster, margin, cluster2, is_edge), so it cannot answer this alone. But
intersection_features.csv carries GROUND-TRUTH kinematics per (scenario_id,
vehicle_id) read straight out of the map binaries -- speed_mean, speed_max,
stop_frac, abs_net_turn. Joining them settles it from the data.

The clustering is K-means on (speed_mean, net_turn, stop_frac), so each name
implies a signature:
    stop&go    highest stop_frac
    turning    highest |net_turn|
    fast       highest speed among what remains
    mid-speed  the other one

Usage:
    python verify_cluster_names.py \
        --isect /scratch/$USER/intersection_features.csv \
        --clusters data/trajectory_clusters_stopfrac.csv
"""

import argparse

import numpy as np
import pandas as pd

# The mapping now used everywhere in the repo. The previous ASSUMED here had
# 0 and 2 swapped (0=mid-speed 1=fast), traced to a comment claiming the
# centroid plot verified it, with no artifact behind the claim. This is the
# hypothesis to TEST -- it is still not independently confirmed.
ASSUMED = {0: "fast", 1: "stop&go", 2: "mid-speed", 3: "turning"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--isect",    required=True)
    p.add_argument("--clusters", required=True)
    p.add_argument("--drop_edge", action="store_true",
                   help="exclude is_edge=1 rows (ambiguous assignments)")
    return p.parse_args()


def main():
    a = parse_args()
    d = pd.read_csv(a.isect)
    c = pd.read_csv(a.clusters)
    c.columns = [x.strip() for x in c.columns]
    d["scenario_id"] = d["scenario_id"].astype(str)
    c["scenario_id"] = c["scenario_id"].astype(str)
    if a.drop_edge and "is_edge" in c.columns:
        c = c[c["is_edge"] != 1]

    for nm, df in (("intersection_features", d), ("clusters", c)):
        dup = df.duplicated(["scenario_id", "vehicle_id"]).sum()
        if dup:
            raise SystemExit(f"  {nm}: {dup} duplicate keys -- fix before joining")

    m = d.merge(c[["scenario_id", "vehicle_id", "cluster"]],
                on=["scenario_id", "vehicle_id"], how="inner", validate="1:1")
    print(f"\n  joined {len(m)} trajectories with ground-truth kinematics\n")
    if not len(m):
        raise SystemExit("  EMPTY JOIN")

    cols = ["speed_mean", "speed_max", "stop_frac", "abs_net_turn"]
    for col in cols:
        m[col] = pd.to_numeric(m[col], errors="coerce")

    g = m.groupby("cluster")[cols].median().round(3)
    g["n"] = m.groupby("cluster").size()
    print("  GROUND-TRUTH MEDIANS PER CLUSTER")
    print(g.to_string())

    # --- infer the names from the data, in the order the signatures are
    #     least ambiguous: stop&go and turning are defined by an extreme, the
    #     remaining two only by their relative speed.
    inferred, left = {}, list(g.index)
    k = g.loc[left, "stop_frac"].idxmax();    inferred[k] = "stop&go";  left.remove(k)
    k = g.loc[left, "abs_net_turn"].idxmax(); inferred[k] = "turning";  left.remove(k)
    k = g.loc[left, "speed_mean"].idxmax();   inferred[k] = "fast";     left.remove(k)
    for k in left:
        inferred[k] = "mid-speed"

    print("\n  ASSUMED vs INFERRED-FROM-DATA")
    print(f"  {'cluster':>7}  {'assumed':<12} {'inferred':<12}  match")
    bad = 0
    for k in sorted(g.index):
        ok = ASSUMED.get(k) == inferred[k]
        bad += not ok
        print(f"  {k:>7}  {ASSUMED.get(k,'?'):<12} {inferred[k]:<12}  "
              f"{'OK' if ok else '*** MISMATCH ***'}")

    if bad:
        print(f"\n  {bad} LABEL(S) WRONG. Every per-cluster figure, name and")
        print( "  policy that used the assumed mapping is mislabelled. Correct")
        print( "  mapping is:")
        print("    " + ",".join(f"{k}:{inferred[k]}" for k in sorted(inferred)))
        print( "  Pass it via TYPE_NAMES=/--cluster_names and fix the defaults")
        print( "  in role_analysis.sbatch, render_role_grid*.sbatch and")
        print( "  intersection_report.py.")
    else:
        print("\n  All four match. The assumed mapping is CORRECT and is now")
        print("  verified from ground truth rather than inherited.")

    # separation check: are the two speed clusters actually distinct?
    print("\n  SEPARATION (does each name earn its distinction?)")
    for k in sorted(g.index):
        s = m[m["cluster"] == k]
        print(f"    {k} {inferred[k]:<11} speed p25/50/75 = "
              f"{s['speed_mean'].quantile(.25):5.2f} / "
              f"{s['speed_mean'].median():5.2f} / "
              f"{s['speed_mean'].quantile(.75):5.2f}   "
              f"stop_frac={s['stop_frac'].median():.3f}  "
              f"|turn|={s['abs_net_turn'].median():.3f}")
    print()


if __name__ == "__main__":
    main()
