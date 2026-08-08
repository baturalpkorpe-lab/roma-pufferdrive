"""ft_check.py -- pass/fail on a regime_rollout sweep, against pre-registered
thresholds. One screen, no table reading.

WHY THIS EXISTS
Two 3B-step runs were taken to completion before anyone measured whether the
role still moved behaviour. Both had a dead or near-dead dial from early on.
This turns "read six tables and squint" into a verdict you can act on at a
mid-run checkpoint.

THREE QUESTIONS, IN THIS ORDER

  1. DOES THE DIAL MOVE?      spread of each style parameter across alpha.
     Reference points, all measured with this same pipeline at 6 episodes:
         ep_nodiv (no perturbation)          headway range 1.150 s
         comply 0.05 + perturb  (arm A)                    0.274 s
         + FiLM                 (arm B)                    0.010 s, 0.037 s
     The empirical noise floor from four independent ep_nodiv-family rollouts
     is +-0.04 s on headway and +-0.05 s on minTTC, so anything under ~0.1 s
     of range is indistinguishable from no dial at all.

  2. IS IT STILL SAFE AT THE EXTREMES?   the point of perturbation is an agent
     that drives DIFFERENTLY but still safely and interactively. The safety
     surrogates must not collapse when the role is forced: minTTC and PET may
     not fall much below their alpha=0 value, and MRD (required deceleration)
     may not rise much above it. A dial that buys its range by driving into
     people has not solved the problem.

  3. IS IT MORE HUMAN?        distance from the HUMAN row, which is measured by
     the same code on ground truth.

WHAT IS NOT HERE
Collision and offroad RATES. regime_rollout does not emit them -- collisions
need pairwise box overlap and offroad needs road geometry, neither of which the
rollout currently loads. The conflict surrogates below (PET / minTTC / MRD) are
the available proxy and they are the standard ones. For true collision and
offroad rates use the WOSAC eval, which reports both.

    python ft_check.py --rollout_dir /scratch/$USER/regimes_rollout/<tag> \
        --gt_regimes   /scratch/$USER/regimes/regimes_gt.csv \
        --gt_conflicts /scratch/$USER/regimes/conflicts_gt.csv
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# Style parameters and the regime each is identified in. Ranges are read across
# alpha; the conflict metrics come from the conflicts_*.csv instead.
REG_COLS = ["headway_T", "v_freeflow_rel", "speed_ff", "speed_fol", "speed_zone"]
CONF_COLS = ["pet", "min_ttc", "mrd"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", required=True)
    p.add_argument("--gt_regimes", default="")
    p.add_argument("--gt_conflicts", default="")
    p.add_argument("--min_dial", type=float, default=0.5,
                   help="required headway spread across alpha, in seconds. "
                        "0.5 is under half of ep_nodiv's 1.150 and well clear "
                        "of the 0.04 noise floor.")
    p.add_argument("--safety_tol", type=float, default=0.20,
                   help="fractional degradation allowed at |alpha|=2 relative "
                        "to alpha=0 for the safety surrogates")
    p.add_argument("--ref_headway", type=float, default=1.02,
                   help="headway at alpha=0 to beat: ep_nodiv's value, so a "
                        "fine-tune is not allowed to trade realism for range")
    return p.parse_args()


def med(df, col):
    if df is None or col not in df.columns:
        return np.nan
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(v.median()) if len(v) else np.nan


def alpha_of(tag):
    """0 for 'natural', else the signed number after the axis name."""
    if tag == "natural":
        return 0.0
    return float(re.sub(r"^[A-Za-z0-9_\-]*?(?=[+-]\d)", "", tag))


def load(rollout_dir):
    rows = {}
    for f in sorted(glob.glob(os.path.join(rollout_dir, "regimes_*.csv"))):
        tag = os.path.basename(f)[8:-4]
        cf = os.path.join(rollout_dir, "conflicts_%s.csv" % tag)
        rows[alpha_of(tag)] = (
            pd.read_csv(f),
            pd.read_csv(cf) if os.path.exists(cf) else None,
        )
    return dict(sorted(rows.items()))


def main():
    a = parse_args()
    arms = load(a.rollout_dir)
    if not arms:
        raise SystemExit("no regimes_*.csv in %s" % a.rollout_dir)

    alphas = list(arms)
    print("=" * 72)
    print("  FT CHECK  --  %s" % a.rollout_dir)
    print("  alphas found: %s" % alphas)
    print("=" * 72)
    if len(alphas) < 3:
        print("\n  WARNING: fewer than 3 alphas. SWEEP=1 probably did not take,")
        print("           and no range below is meaningful.")

    human_reg = (pd.read_csv(a.gt_regimes)
                 if a.gt_regimes and os.path.exists(a.gt_regimes) else None)
    human_cf = (pd.read_csv(a.gt_conflicts)
                if a.gt_conflicts and os.path.exists(a.gt_conflicts) else None)

    verdicts = []

    # ---- 1. does the dial move? -------------------------------------------
    print("\n  1. DIAL  (spread across alpha; the whole point of the role)")
    print("     %-18s %9s %9s %9s   %s" %
          ("metric", "min", "max", "range", "human"))
    dial_range = {}
    for c in REG_COLS:
        vals = [med(arms[al][0], c) for al in alphas]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        rng = max(vals) - min(vals)
        dial_range[c] = rng
        h = med(human_reg, c) if human_reg is not None else np.nan
        print("     %-18s %9.3f %9.3f %9.3f   %s" %
              (c, min(vals), max(vals), rng,
               "%.3f" % h if np.isfinite(h) else "--"))

    hw = dial_range.get("headway_T", np.nan)
    ok_dial = np.isfinite(hw) and hw >= a.min_dial
    verdicts.append(("dial range on headway >= %.2f s" % a.min_dial, ok_dial,
                     "%.3f s" % hw if np.isfinite(hw) else "n/a"))
    print("\n     reference: ep_nodiv 1.150 | arm A 0.274 | arm B 0.010")
    print("     noise floor from four independent rollouts: +-0.04 s")

    # ---- 2. still safe at the extremes? -----------------------------------
    print("\n  2. SAFETY AT THE EXTREMES  (different behaviour, still safe)")
    if 0.0 not in arms:
        print("     no alpha=0 arm -- cannot compare extremes against natural")
    else:
        base = {c: med(arms[0.0][1], c) for c in CONF_COLS}
        ext = [al for al in alphas if abs(al) >= 2.0] or \
              [al for al in alphas if al != 0.0]
        print("     %-10s %9s %9s   %s" % ("metric", "alpha=0", "worst |a|",
                                           "human"))
        for c in CONF_COLS:
            vs = [med(arms[al][1], c) for al in ext]
            vs = [v for v in vs if np.isfinite(v)]
            if not np.isfinite(base.get(c, np.nan)) or not vs:
                continue
            # pet and min_ttc: bigger is safer. mrd: smaller is safer.
            worse = min(vs) if c in ("pet", "min_ttc") else max(vs)
            h = med(human_cf, c) if human_cf is not None else np.nan
            print("     %-10s %9.3f %9.3f   %s" %
                  (c, base[c], worse, "%.3f" % h if np.isfinite(h) else "--"))
            if c in ("pet", "min_ttc"):
                ok = worse >= base[c] * (1.0 - a.safety_tol)
            else:
                ok = worse <= base[c] * (1.0 + a.safety_tol)
            verdicts.append(("%s holds within %.0f%% at |alpha|=2"
                             % (c, 100 * a.safety_tol), ok,
                             "%.3f vs %.3f" % (worse, base[c])))
        print("\n     pet/min_ttc: higher is safer. mrd: lower is safer.")
        print("     A dial that buys its range by cutting people off fails here.")

    # ---- 3. more human at the natural role? -------------------------------
    print("\n  3. REALISM AT alpha=0  (not allowed to regress)")
    if 0.0 in arms:
        h0 = med(arms[0.0][0], "headway_T")
        print("     headway_T   %.3f    ep_nodiv %.3f    human %.3f" %
              (h0, a.ref_headway,
               med(human_reg, "headway_T") if human_reg is not None else np.nan))
        verdicts.append(("headway at alpha=0 >= ep_nodiv's %.2f s"
                         % a.ref_headway,
                         np.isfinite(h0) and h0 >= a.ref_headway,
                         "%.3f s" % h0 if np.isfinite(h0) else "n/a"))
        for c in ("speed_ff", "accel_ff"):
            v = med(arms[0.0][0], c)
            hv = med(human_reg, c) if human_reg is not None else np.nan
            if np.isfinite(v):
                print("     %-11s %.3f    %s" %
                      (c, v, ("human %.3f" % hv) if np.isfinite(hv) else ""))

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 72)
    for name, ok, detail in verdicts:
        print("  [%s]  %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    hard = [v for v in verdicts if not v[1]]
    print("=" * 72)
    if not hard:
        print("  VERDICT: continue. The dial moves and safety holds.")
    elif not ok_dial:
        print("  VERDICT: KILL THE RUN. No dial means nothing downstream is")
        print("           interpretable, and it will not appear later -- both")
        print("           previous arms were already flat well before 3B.")
    else:
        print("  VERDICT: the dial works but %d safety/realism check(s) failed."
              % len(hard))
        print("           Range bought at the cost of safety is not the result")
        print("           we want; lower the perturbation or raise the anchor.")
    print("=" * 72)
    return 0 if not hard else 1


if __name__ == "__main__":
    sys.exit(main())
