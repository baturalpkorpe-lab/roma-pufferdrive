"""
intersection_report.py -- does the intersection label fix what it was meant to?

Cross-references intersection_features.csv against the frozen stop_frac
clustering (trajectory_clusters_stopfrac.csv) and answers, with numbers:

  1. How does the new intersection label distribute over the old 4 clusters?
  2. THE test. Inside the old "turning" cluster, does the label separate the
     two populations that were conflated -- a car turning AT A JUNCTION vs a
     car following a curved road? If it works, the non-intersection half of
     that cluster has a large |net_turn| but a small |road_rel_turn| (it was
     tracking the road), while the intersection half has both large.
  3. How many intersection trajectories have LOW turning? Those are the ones
     net_turn could never have found -- straight through a 4-way stop -- and
     they are the whole reason for labelling the environment rather than the
     trajectory shape.

Usage:
    python intersection_report.py \
        --isect /scratch/$USER/intersection_features.csv \
        --clusters data/trajectory_clusters_stopfrac.csv
"""

import argparse

import numpy as np
import pandas as pd

CLUSTER_NAME = {0: "fast", 1: "stop&go", 2: "mid-speed", 3: "turning"}
TURN_HI = 0.5      # rad; |net_turn| above this = "turning-shaped"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--isect",    required=True)
    p.add_argument("--clusters", required=True)
    p.add_argument("--turn_hi",  type=float, default=TURN_HI)
    return p.parse_args()


def q(s, name):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not len(s):
        return f"    {name:<26} (no data)"
    return (f"    {name:<26} n={len(s):>6}  median={s.median():7.3f}  "
            f"p25={s.quantile(.25):7.3f}  p75={s.quantile(.75):7.3f}")


def main():
    a = parse_args()
    d = pd.read_csv(a.isect)
    c = pd.read_csv(a.clusters)
    c.columns = [x.strip() for x in c.columns]

    d["scenario_id"] = d["scenario_id"].astype(str)
    c["scenario_id"] = c["scenario_id"].astype(str)

    # A (scenario_id, vehicle_id) must be unique on BOTH sides. If it is not,
    # pandas does a many-to-many merge and silently inflates the row count --
    # every number downstream would then be weighted by the duplication, which
    # is invisible unless you check. Fail loudly instead.
    for name, df in (("intersection_features", d), ("clusters", c)):
        dup = df.duplicated(["scenario_id", "vehicle_id"]).sum()
        if dup:
            raise SystemExit(
                f"  {name}: {dup} duplicate (scenario_id, vehicle_id) rows.\n"
                f"  A many-to-many join would inflate every statistic below.\n"
                f"  Deduplicate first, or check the file was not appended twice.")

    m = d.merge(c[["scenario_id", "vehicle_id", "cluster"]],
                on=["scenario_id", "vehicle_id"], how="inner", validate="1:1")

    print(f"\n  intersection rows : {len(d)}")
    print(f"  clustering rows   : {len(c)}")
    print(f"  JOINED            : {len(m)}  "
          f"({100*len(m)/max(len(d),1):.1f}% of intersection rows)")
    if not len(m):
        raise SystemExit("  EMPTY JOIN -- scenario_id/vehicle_id do not match. "
                         "The clustering is keyed to the map pool it was built "
                         "on; check both came from the same binaries.")

    print("\n" + "=" * 68)
    print("  1. INTERSECTION LABEL vs THE FROZEN stop_frac CLUSTERS")
    print("=" * 68)
    ct = pd.crosstab(m["cluster"], m["intersection"])
    for k in (0, 1):
        if k not in ct.columns:
            ct[k] = 0
    ct = ct[[0, 1]]
    ct.columns = ["not_intersection", "intersection"]
    ct["total"] = ct.sum(1)
    ct["pct_intersection"] = (100 * ct["intersection"] / ct["total"]).round(1)
    ct.index = [f"{i} {CLUSTER_NAME.get(i, '?')}" for i in ct.index]
    print(ct.to_string())

    print("\n" + "=" * 68)
    print("  2. THE TEST: inside the old 'turning' cluster, does the label")
    print("     separate junction turns from curved-road following?")
    print("=" * 68)
    t = m[m["cluster"] == 3]
    if not len(t):
        print("    no cluster-3 rows in the join")
    else:
        for lab, sub in [("INTERSECTION (real junction turn)", t[t["intersection"] == 1]),
                         ("NOT intersection (road-following?)", t[t["intersection"] == 0])]:
            print(f"\n    -- {lab}  n={len(sub)} ({100*len(sub)/len(t):.1f}%)")
            print(q(sub["abs_net_turn"],      "|net_turn| (rad)"))
            print(q(sub["abs_road_rel_turn"], "|road_rel_turn| (rad)"))
            print(q(sub["speed_mean"],        "speed_mean (m/s)"))
        hi = pd.to_numeric(t[t["intersection"] == 0]["abs_net_turn"], errors="coerce")
        rr = pd.to_numeric(t[t["intersection"] == 0]["abs_road_rel_turn"], errors="coerce")
        ok = hi.notna() & rr.notna()
        if ok.sum():
            road_follow = ((hi[ok] > a.turn_hi) & (rr[ok] < a.turn_hi)).mean()
            print(f"\n    Of the NON-intersection half of 'turning': "
                  f"{100*road_follow:.1f}% have |net_turn|>{a.turn_hi} but "
                  f"|road_rel_turn|<{a.turn_hi}")
            print( "    -> that is the curved-road false positive, quantified.")

    print("\n" + "=" * 68)
    print("  3. WHAT net_turn COULD NEVER HAVE FOUND")
    print("=" * 68)
    i1 = m[m["intersection"] == 1]
    nt = pd.to_numeric(i1["abs_net_turn"], errors="coerce")
    low = (nt < a.turn_hi).mean()
    print(f"    intersection trajectories with |net_turn| < {a.turn_hi}: "
          f"{100*low:.1f}%  (n={int((nt < a.turn_hi).sum())})")
    print( "    -> straight-through-a-junction: invisible to a turning metric,")
    print( "       but the driver still had to deal with the intersection.")
    print("\n    old cluster of those low-turn intersection trajectories:")
    sub = i1[nt < a.turn_hi]
    vc = sub["cluster"].value_counts().sort_index()
    for k, v in vc.items():
        print(f"      {k} {CLUSTER_NAME.get(k,'?'):<12} {v:>7}  "
              f"({100*v/len(sub):.1f}%)")

    print("\n" + "=" * 68)
    print("  4. PROPOSED SPLIT (intersection first, then speed)")
    print("=" * 68)
    for lab, sub in [("intersection", m[m["intersection"] == 1]),
                     ("no intersection", m[m["intersection"] == 0])]:
        s = pd.to_numeric(sub["speed_mean"], errors="coerce").dropna()
        sf = pd.to_numeric(sub["stop_frac"], errors="coerce").dropna()
        print(f"\n    {lab:<16} n={len(sub):>7} ({100*len(sub)/len(m):.1f}%)")
        if len(s):
            print(f"      speed_mean  median={s.median():.2f}  "
                  f"p25={s.quantile(.25):.2f} p75={s.quantile(.75):.2f}")
        if len(sf):
            print(f"      stop_frac   median={sf.median():.3f}")
    print()


if __name__ == "__main__":
    main()
