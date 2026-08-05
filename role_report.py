"""
role_report.py -- one table per arm, OLD analysis and NEW regime analysis side
by side, so an arm can be judged without opening five files.

WHY BOTH
The two analyses answer different questions and neither replaces the other.

  OLD (role_paired_sweep)  33 generic kinematic metrics, pooled over every
                           situation the agent was in, swept along a PC.
                           Comparable with every figure made before this, which
                           is why it stays.
  NEW (regime_rollout)     each style parameter read only in the regime that
                           IDENTIFIES it -- desired speed in free flow, time
                           headway while following, gap acceptance at conflicts
                           -- and compared against the human reference measured
                           by the same code.

The old one pools across situations, so an effect that exists only in one regime
is diluted by the others; the new one cannot be compared with the historical
figures. Read them together.

WHAT TO LOOK AT FIRST
  1. accel_mask_frac in the old table. If it is not ~0, the tail metrics were
     truncated differently per condition and nothing below it is comparable.
  2. coherence in the causal-axis table. Pre-registered target +1. NEGATIVE
     means forcing the role moves behaviour opposite to what the encoder's own
     correlation says.
  3. cos_pc1. Negative = the PC sweep moves that metric the wrong way, and
     gain_vs_pc1 says how much effect the swept axis is leaving on the table.
  4. the regime table against the GT column. A policy can match humans on one
     style parameter and be wildly off on another -- measured here: free-flow
     speed 0.81 vs 0.83 human, but time headway 0.94 s vs 2.06 s human.

Usage:
    python role_report.py --tag mifutagent_dim1_nodiv \
        --analysis_root /scratch/$USER/analysis \
        --rollout_dir   /scratch/$USER/regimes_rollout/dim1_nodiv \
        --gt_regimes    /scratch/$USER/regimes/regimes_gt.csv \
        --icc_csv       /scratch/$USER/role_icc/ALL_ARMS.csv
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

# The human reference, measured by the same code as the policy side.
GT_KEYS = ("v_freeflow_rel", "headway_T", "jam_s0")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--analysis_root", default="")
    p.add_argument("--rollout_dir", default="")
    p.add_argument("--gt_regimes", default="")
    p.add_argument("--gt_conflicts", default="")
    p.add_argument("--icc_csv", default="")
    p.add_argument("--top", type=int, default=10,
                   help="rows to show in the old/causal tables")
    p.add_argument("--by", default="control",
                   help="stratify section 5 by 'control' (right-of-way) or "
                        "'kind' (merging vs crossing)")
    p.add_argument("--out", default="")
    return p.parse_args()


def _med(df, col):
    if col not in df.columns:
        return np.nan, 0
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return (float(v.median()), len(v)) if len(v) else (np.nan, 0)


def section(title):
    print("\n" + "=" * 78)
    print("  " + title)
    print("=" * 78)


def old_analysis(root, tag, top):
    p = os.path.join(root, tag, "paired", "role_paired_tests.csv")
    if not os.path.exists(p):
        print(f"  (no role_paired_tests.csv under {root}/{tag}/paired)")
        return None
    df = pd.read_csv(p)
    audit = df[df["metric"] == "accel_mask_frac"]
    for _, r in audit.iterrows():
        flag = "OK" if abs(r["t_stat"]) < 3 else "*** NOT FLAT -- tail metrics not comparable"
        print(f"  AUDIT accel_mask_frac on {r['axis']}: "
              f"diff={r['mean_paired_diff']:+.2e} t={r['t_stat']:+.2f}  {flag}")
    d = df[df["metric"] != "accel_mask_frac"].copy()
    d["abs_t"] = d["t_stat"].abs()
    print()
    for ax in sorted(d["axis"].unique()):
        s = d[d["axis"] == ax].nlargest(top, "abs_t")
        print(f"  {ax}: strongest paired effects (alpha +2 vs -2)")
        print(s[["metric", "mean_paired_diff", "t_stat", "n_pairs"]]
              .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
        print()
    return df


def causal_axis(root, tag, top):
    p = os.path.join(root, tag, "causal_axis", "role_causal_axis.csv")
    if not os.path.exists(p):
        print(f"  (no role_causal_axis.csv under {root}/{tag}/causal_axis)")
        return None
    df = pd.read_csv(p).nlargest(top, "effect_causal_per_sigma")
    cols = [c for c in ("metric", "effect_causal_per_sigma", "cos_pc1",
                        "gain_vs_pc1", "r2", "obs_r_pc1", "coherence")
            if c in df.columns]
    print(df[cols].to_string(index=False,
                             float_format=lambda v: f"{v:8.3f}"))
    if "coherence" in df.columns:
        neg = df[df["coherence"] < 0]["metric"].tolist()
        print(f"\n  coherence target is +1. NEGATIVE for: "
              f"{neg if neg else 'none'}")
    if "cos_pc1" in df.columns:
        wrong = df[df["cos_pc1"] < 0]["metric"].tolist()
        print(f"  PC1 sweep moves the WRONG WAY for: "
              f"{wrong if wrong else 'none'}")
    return df


def regime_table(roll_dir, gt_regimes, gt_conflicts):
    if not roll_dir or not os.path.isdir(roll_dir):
        print(f"  (no rollout dir {roll_dir})")
        return None
    rows = []
    for f in sorted(glob.glob(os.path.join(roll_dir, "regimes_*.csv"))):
        t = os.path.basename(f)[8:-4]
        a = 0.0 if t == "natural" else float(re.sub(r"^[A-Za-z0-9_\-]*?(?=[+-]\d)", "", t))
        r = pd.read_csv(f)
        row = {"alpha": a, "n_traj": len(r)}
        for k in GT_KEYS:
            m, n = _med(r, k)
            row[k] = round(m, 3) if np.isfinite(m) else np.nan
            row["n_" + k] = n
        cf = os.path.join(roll_dir, f"conflicts_{t}.csv")
        if os.path.exists(cf):
            c = pd.read_csv(cf)
            row["n_conf"] = len(c) // 2
            for k in ("pet", "min_ttc", "mrd"):
                m, _ = _med(c, k)
                row[k] = round(m, 3) if np.isfinite(m) else np.nan
        rows.append(row)
    if not rows:
        print("  (no regimes_*.csv)")
        return None
    df = pd.DataFrame(rows).sort_values("alpha")

    if gt_regimes and os.path.exists(gt_regimes):
        g = pd.read_csv(gt_regimes)
        gt = {"alpha": "HUMAN", "n_traj": len(g)}
        for k in GT_KEYS:
            m, n = _med(g, k)
            gt[k] = round(m, 3) if np.isfinite(m) else np.nan
            gt["n_" + k] = n
        if gt_conflicts and os.path.exists(gt_conflicts):
            c = pd.read_csv(gt_conflicts)
            gt["n_conf"] = len(c) // 2
            for k in ("pet", "min_ttc", "mrd"):
                m, _ = _med(c, k)
                gt[k] = round(m, 3) if np.isfinite(m) else np.nan
        df = pd.concat([pd.DataFrame([gt]), df], ignore_index=True)

    print(df.to_string(index=False))
    print("\n  alpha=HUMAN is the ground-truth reference measured by the SAME")
    print("  code. A monotone column across alpha is a causal role effect;")
    print("  distance from the HUMAN row is a realism gap. They are different")
    print("  questions and an arm can pass one and fail the other.")
    return df


def conflict_breakdown(roll_dir, gt_conflicts, by="control"):
    """Conflict metrics split by right-of-way class -- the pre-registered
    stratification.

    Section 4 pools every junction type together, which hides the comparison the
    whole thing was built for: the role should have the most room where
    right-of-way is SYMMETRIC (all-way stop) and least where it is externally
    imposed (traffic light). On ground truth 23.5% of the go/yield decision is
    unexplained by geometry at all-way stops against 10.8% at signalised, so the
    headroom is there -- this table says whether the role uses it.
    """
    rows = []
    if gt_conflicts and os.path.exists(gt_conflicts):
        g = pd.read_csv(gt_conflicts)
        if by in g.columns:
            for k, sub in g.groupby(by):
                r = {"alpha": "HUMAN", by: k, "n_conf": len(sub) // 2}
                for m in ("pet", "min_ttc", "mrd"):
                    v, _ = _med(sub, m)
                    r[m] = round(v, 3) if np.isfinite(v) else np.nan
                rows.append(r)

    for f in sorted(glob.glob(os.path.join(roll_dir, "conflicts_*.csv"))):
        t = os.path.basename(f)[10:-4]
        a = 0.0 if t == "natural" else float(
            re.sub(r"^[A-Za-z0-9_\-]*?(?=[+-]\d)", "", t))
        c = pd.read_csv(f)
        if by not in c.columns:
            continue
        for k, sub in c.groupby(by):
            r = {"alpha": a, by: k, "n_conf": len(sub) // 2}
            for m in ("pet", "min_ttc", "mrd"):
                v, _ = _med(sub, m)
                r[m] = round(v, 3) if np.isfinite(v) else np.nan
            if "ego_went_first" in sub.columns:
                r["go_rate"] = round(float(
                    pd.to_numeric(sub["ego_went_first"],
                                  errors="coerce").mean()), 3)
            rows.append(r)

    if not rows:
        print(f"  (no '{by}' column in the conflict files -- regenerate with a "
              f"regime_rollout that emits it)")
        return None
    df = pd.DataFrame(rows)
    order = ["all_way_stop", "partial_stop", "signalised_likely", "uncontrolled"]
    df["_o"] = df[by].apply(lambda v: order.index(v) if v in order else 99)
    df["_a"] = df["alpha"].apply(lambda v: -99 if v == "HUMAN" else float(v))
    df = df.sort_values(["_o", "_a"]).drop(columns=["_o", "_a"])
    print(df.to_string(index=False))
    print("\n  Read DOWN each control block: does the role move the metrics")
    print("  more where right-of-way is symmetric (all_way_stop) than where a")
    print("  light decides (signalised_likely)? That ordering is the")
    print("  pre-registered prediction. go_rate near 0.5 is forced by the")
    print("  mirrored rows and is only a file check.")
    return df


def icc_row(path, tag):
    if not path or not os.path.exists(path):
        print(f"  (no {path})")
        return
    d = pd.read_csv(path)
    hit = d[d["arm"].astype(str).str.contains(tag, regex=False)]
    print((hit if len(hit) else d).to_string(index=False))
    if not len(hit):
        print(f"\n  '{tag}' not in the table -- showing every arm instead.")


def main():
    args = parse_args()
    print("#" * 78)
    print(f"#  ROLE REPORT -- {args.tag}")
    print("#" * 78)

    section("1. SCENE ICC (pre-registered map-independence metric)")
    icc_row(args.icc_csv, args.tag)

    section("2. OLD ANALYSIS -- paired PC sweep, 33 metrics pooled")
    if args.analysis_root:
        old_analysis(args.analysis_root, args.tag, args.top)

    section("3. CAUSAL AXIS -- is the swept direction the right one?")
    if args.analysis_root:
        causal_axis(args.analysis_root, args.tag, args.top)

    section("4. NEW ANALYSIS -- style parameters in their own regimes")
    regime_table(args.rollout_dir, args.gt_regimes, args.gt_conflicts)

    section("5. CONFLICTS BY RIGHT-OF-WAY -- the pre-registered stratification")
    if args.rollout_dir and os.path.isdir(args.rollout_dir):
        conflict_breakdown(args.rollout_dir, args.gt_conflicts, args.by)
    else:
        print(f"  (no rollout dir {args.rollout_dir})")

    print("\n" + "=" * 78)
    print("  READ IN THIS ORDER: the accel_mask_frac audit (section 2) gates")
    print("  everything below it; then coherence and cos_pc1 (section 3) say")
    print("  whether the swept axis was even the right dial; then section 4")
    print("  says what moved, in physical units, against the human reference.")
    print("=" * 78)


if __name__ == "__main__":
    main()
