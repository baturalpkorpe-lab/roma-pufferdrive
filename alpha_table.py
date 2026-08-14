"""alpha_table.py -- one table: every feature as a row, every alpha as a column.

role_report.py splits the same numbers across three sections with a different
column layout in each, which makes reading a sweep an exercise in cross-
referencing. This prints features DOWN and alpha ACROSS, with the human value
and a monotonicity score beside each row.

TWO COLUMNS THAT ARE NOT THE VALUES

  rho   Spearman rank correlation between alpha and the metric: does the metric
        move in a consistent ORDER as the dial turns, ignoring by how much.
        -1 or +1 = perfectly ordered, 0 = no ordering. With five alphas it can
        only take steps of 0.1; one out-of-order pair gives 0.9, two gives 0.7.
        Range says how far, rho says whether it was a response or a wobble --
        a wide range at rho ~ 0 is noise.

  n     smallest and largest number of observations behind the row, across
        alpha. A style parameter is read only on the steps where its regime is
        identified, and THE SWEEP CHANGES WHICH STEPS THOSE ARE: ep_nodiv at
        alpha=-2 crawls until it stops following anyone, and its headway sample
        fell to 386 against ~1800 at alpha=0. A median over a collapsing
        population is not a dial. If min and max differ by more than ~2x, treat
        that row's range as part selection effect.

    python alpha_table.py \
        --rollout_dir  /scratch/$USER/regimes_rollout/<tag> \
        --gt_regimes   /scratch/$USER/regimes/regimes_gt.csv \
        --gt_conflicts /scratch/$USER/regimes/conflicts_gt.csv
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

# (column, source) -- source 0 = regimes csv, 1 = conflicts csv
STYLE = [("headway_T", 0), ("jam_s0", 0), ("v_freeflow_rel", 0),
         ("pet", 1), ("min_ttc", 1), ("mrd", 1)]
KINEM = [("speed_ff", 0), ("speed_fol", 0), ("speed_zone", 0),
         ("accel_ff", 0), ("accel_fol", 0), ("accel_zone", 0)]
STOPC = [("vmin_zone", 0), ("stopfrac_zone", 0), ("rollfrac_zone", 0)]
CONF = ["pet", "min_ttc", "mrd"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", action="append", required=True,
                   help="repeatable. Given more than once the rows are POOLED "
                        "before the median is taken, which is not the same as "
                        "averaging each rep's median and is strictly better: "
                        "one median over 3x the rows has lower variance than "
                        "the mean of three medians, at no extra compute. The "
                        "reps must be of the SAME checkpoint.")
    p.add_argument("--gt_regimes", default="")
    p.add_argument("--gt_conflicts", default="")
    p.add_argument("--controls", default="all_way_stop,partial_stop",
                   help="right-of-way classes to break the conflict metrics "
                        "out by; '' to skip. signalised_likely is omitted by "
                        "default -- n is 30-60 per alpha, directional at best.")
    p.add_argument("--stop_compliance", action="store_true",
                   help="also show vmin/stopfrac/rollfrac inside junctions. "
                        "For a POLICY these measure an observation gap, not "
                        "driving: stop signs never enter the observation.")
    p.add_argument("--out", default="")
    return p.parse_args()


def med(df, col):
    if df is None or col not in df.columns:
        return np.nan, 0
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return (float(v.median()), len(v)) if len(v) else (np.nan, 0)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def alpha_of(tag):
    return 0.0 if tag == "natural" else float(
        re.sub(r"^[A-Za-z0-9_\-]*?(?=[+-]\d)", "", tag))


def load(dirs):
    """Alpha -> (regimes, conflicts), rows POOLED across dirs.

    Pooling reps rather than averaging their medians is the cheap half of the
    stability problem: the thin rows here are jam_s0 (~200-330 per alpha) and
    the per-class conflicts (~350), and three reps triples both for free.

    Caveat worth knowing: rollouts resample the map pool, so reps overlap in
    scenes. The pooled sample is therefore not three times as INDEPENDENT as
    one rep -- it is a bigger sample of the same scene distribution, which
    lowers the median's variance but does not widen scene coverage. For that,
    raise EPISODES and MAP_POOL.
    """
    acc = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "regimes_*.csv"))):
            t = os.path.basename(f)[8:-4]
            cf = os.path.join(d, "conflicts_%s.csv" % t)
            r = pd.read_csv(f)
            c = pd.read_csv(cf) if os.path.exists(cf) else None
            k = alpha_of(t)
            if k not in acc:
                acc[k] = ([r], [c] if c is not None else [])
            else:
                acc[k][0].append(r)
                if c is not None:
                    acc[k][1].append(c)
    out = {}
    for k, (rs, cs) in acc.items():
        out[k] = (pd.concat(rs, ignore_index=True),
                  pd.concat(cs, ignore_index=True) if cs else None)
    return dict(sorted(out.items()))


def main():
    a = parse_args()
    arms = load(a.rollout_dir)
    if not arms:
        raise SystemExit("no regimes_*.csv in %s" % ", ".join(a.rollout_dir))
    alphas = list(arms)

    hr = pd.read_csv(a.gt_regimes) if a.gt_regimes and os.path.exists(a.gt_regimes) else None
    hc = pd.read_csv(a.gt_conflicts) if a.gt_conflicts and os.path.exists(a.gt_conflicts) else None

    w = 8
    head = ("  %-17s" % "feature") + "".join("%*s" % (w, "%+g" % x if x else "0")
                                             for x in alphas)
    head += "  |%8s%8s  %s" % ("HUMAN", "rho", "n")
    print("=" * (len(head) + 4))
    names = [os.path.basename(x.rstrip("/")) for x in a.rollout_dir]
    print("  ALPHA TABLE -- %s%s"
          % (names[0], ("  + %d more, POOLED" % (len(names) - 1))
             if len(names) > 1 else ""))
    print("=" * (len(head) + 4))
    print(head)

    rows = []

    def block(title, items, sub=None, label=None):
        print("  " + "-" * (len(head) - 4))
        print("  %s" % title)
        for col, src in items:
            vals, ns = [], []
            for al in alphas:
                d = arms[al][src]
                if sub is not None and d is not None and "control" in d.columns:
                    d = d[d["control"] == sub]
                v, n = med(d, col)
                vals.append(v)
                ns.append(n if src == 0 else n // 2)
            if not any(np.isfinite(v) for v in vals):
                continue
            h, _ = med(hr if src == 0 else hc, col) if (hr is not None or hc is not None) else (np.nan, 0)
            if sub is not None and hc is not None and "control" in hc.columns:
                h, _ = med(hc[hc["control"] == sub], col)
            r = spearman(alphas, vals)
            nn = [x for x in ns if x]
            name = label + col if label else col
            print(("  %-17s" % name)
                  + "".join("%*.3f" % (w, v) if np.isfinite(v) else "%*s" % (w, "--")
                            for v in vals)
                  + "  |%8s%8s  %s" % (
                      "%.3f" % h if np.isfinite(h) else "--",
                      "%+.2f" % r if np.isfinite(r) else "--",
                      "%d-%d" % (min(nn), max(nn)) if nn else "0"))
            rows.append(dict(feature=name, human=h, rho=r,
                             **{("a%+g" % x): v for x, v in zip(alphas, vals)}))

    block("style parameters  (each read only where its regime identifies it)",
          STYLE)
    block("kinematics, per regime  (same quantity in three situations)", KINEM)
    if a.stop_compliance:
        block("stop compliance inside junctions  (an OBSERVATION gap for a "
              "policy:\n  stop signs never reach the observation)", STOPC)
    for c in [x.strip() for x in a.controls.split(",") if x.strip()]:
        block("conflicts at %s" % c, [(m, 1) for m in CONF], sub=c,
              label="")

    print("  " + "-" * (len(head) - 4))
    print("  rho: does the metric move in a consistent ORDER as alpha turns.")
    print("       -1/+1 perfectly ordered, 0 none. Five alphas quantise it to")
    print("       steps of 0.1; one swapped pair gives 0.9, two gives 0.7.")
    print("  n:   min-max observations behind the row. More than ~2x apart and")
    print("       the sweep is changing WHICH steps the metric is read on, so")
    print("       part of that row's range is selection, not style.")
    print("=" * (len(head) + 4))

    if a.out and rows:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print("\n  wrote %s" % a.out)


if __name__ == "__main__":
    main()
