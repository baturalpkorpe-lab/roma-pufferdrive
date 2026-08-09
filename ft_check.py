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
    p.add_argument("--min_dial", type=float, default=0.25,
                   help="required headway spread across alpha, in seconds. "
                        "NOT 0.5: that was set against ep_nodiv's dim1_v4 "
                        "rollout (1.150), which is an OUTLIER. The same "
                        "checkpoint rolled out four times gave 0.477 / 0.492 / "
                        "0.544 / 1.150, and the 1.150 came from an alpha=-2 "
                        "median over 386 steady-following steps against ~1800 "
                        "at alpha=0. 0.25 is half of the honest ~0.50.")
    p.add_argument("--min_conflict_dial", type=float, default=0.30,
                   help="required min_ttc spread at ALL-WAY STOPS, in seconds. "
                        "This is the pre-registered metric: symmetric "
                        "right-of-way is where a driving style has room, and "
                        "conflict counts stay stable across alpha, so it does "
                        "not suffer the population drift that headway does.")
    p.add_argument("--min_rho", type=float, default=0.9,
                   help="|Spearman rho| between alpha and all-way-stop min_ttc. "
                        "A dial should be MONOTONE; a large range with no "
                        "ordering is noise.")
    p.add_argument("--safety_tol", type=float, default=0.20,
                   help="fractional degradation allowed at |alpha|=2 relative "
                        "to alpha=0 for the safety surrogates")
    p.add_argument("--ref_headway", type=float, default=1.02,
                   help="headway at alpha=0 to beat: ep_nodiv's value")
    p.add_argument("--noise", type=float, default=0.05,
                   help="measurement noise floor in seconds, from four "
                        "independent rollouts of one checkpoint. Realism is "
                        "only judged to have regressed beyond this.")
    return p.parse_args()


def spearman(x, y):
    """Rank correlation without scipy. +-1 = perfectly monotone."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


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
    # The n column is not decoration. A style parameter is read only on the
    # steps where its regime is identified, and the SWEEP CHANGES WHICH STEPS
    # THOSE ARE. ep_nodiv at alpha=-2 crawls so slowly it stops acquiring
    # leaders at all: n_headway_T fell to 386 against ~1800 at alpha=0, and the
    # median over that remnant read 2.011 s -- which then looked like a
    # human-level following gap and set a benchmark no honest run could meet.
    # A range measured across a collapsing population is not a dial.
    print("\n  1. DIAL  (spread across alpha; the whole point of the role)")
    print("     %-18s %9s %9s %9s %7s   %s" %
          ("metric", "min", "max", "range", "n_swing", "human"))
    dial_range, unstable = {}, []
    # Where the rollout carries a step-count column, sum it. Otherwise fall
    # back to how many AGENTS produced a value for the metric at all -- which
    # is what role_report's n_ columns report, and the quantity that collapsed
    # to 386 on ep_nodiv at alpha=-2.
    NCOL = {"speed_ff": "n_ff", "speed_fol": "n_fol", "speed_zone": "n_zone"}
    for c in REG_COLS:
        vals = [med(arms[al][0], c) for al in alphas]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        rng = max(vals) - min(vals)
        dial_range[c] = rng
        ns, nc = [], NCOL.get(c)
        for al in alphas:
            d = arms[al][0]
            if nc and nc in d.columns:
                v = pd.to_numeric(d[nc], errors="coerce").dropna()
                ns.append(float(v.sum()))
            elif c in d.columns:
                ns.append(float(pd.to_numeric(d[c], errors="coerce")
                                .notna().sum()))
        swing = (max(ns) / max(min(ns), 1.0)) if ns and all(
            np.isfinite(ns)) else np.nan
        if np.isfinite(swing) and swing > 2.0:
            unstable.append(c)
        h = med(human_reg, c) if human_reg is not None else np.nan
        print("     %-18s %9.3f %9.3f %9.3f %6s   %s" %
              (c, min(vals), max(vals), rng,
               ("%.1fx" % swing) if np.isfinite(swing) else "--",
               "%.3f" % h if np.isfinite(h) else "--"))
    if unstable:
        print("\n     POPULATION UNSTABLE (>2x sample swing across alpha): %s"
              % ", ".join(unstable))
        print("     Those ranges are part selection effect, not pure style.")
    hw = dial_range.get("headway_T", np.nan)
    ok_dial = np.isfinite(hw) and hw >= a.min_dial
    verdicts.append(("dial range on headway >= %.2f s" % a.min_dial, ok_dial,
                     "%.3f s" % hw if np.isfinite(hw) else "n/a"))
    print("\n     headway reference, ep_nodiv rolled out FOUR times on the same")
    print("     weights: 0.477 / 0.492 / 0.544 / 1.150. The 1.150 is an outlier")
    print("     (n=386 at alpha=-2). Honest reference ~0.50; arm A 0.274;")
    print("     arm B 0.010 and 0.037. Noise floor +-0.04 s.")

    # ---- 1b. the pre-registered dial: min_ttc at all-way stops ------------
    # Better than headway on two counts. Conflict counts stay stable across
    # alpha (409-430 on the fine-tune) where following-step counts do not, so
    # there is no population drift to confound the range. And symmetric
    # right-of-way is where a driving style has room by construction: on GT,
    # 23.5% of the go/yield decision at all-way stops is unexplained by
    # geometry, against 10.8% at signals.
    print("\n  1b. PRE-REGISTERED DIAL  (min_ttc at all-way stops)")
    aw_a, aw_v, aw_n = [], [], []
    for al in alphas:
        c = arms[al][1]
        if c is None or "control" not in c.columns:
            continue
        sub = c[c["control"] == "all_way_stop"]
        v = med(sub, "min_ttc")
        if np.isfinite(v):
            aw_a.append(al); aw_v.append(v); aw_n.append(len(sub) // 2)
    if len(aw_v) >= 3:
        rho = spearman(aw_a, aw_v)
        rng = max(aw_v) - min(aw_v)
        print("     alpha    " + "  ".join("%7.1f" % x for x in aw_a))
        print("     min_ttc  " + "  ".join("%7.3f" % x for x in aw_v))
        print("     n_conf   " + "  ".join("%7d" % x for x in aw_n))
        print("     range=%.3f s   spearman rho=%+.2f  (-1 or +1 = monotone)"
              % (rng, rho))
        verdicts.append(("all-way min_ttc range >= %.2f s" % a.min_conflict_dial,
                         rng >= a.min_conflict_dial, "%.3f s" % rng))
        verdicts.append(("all-way min_ttc is monotone (|rho| >= %.1f)" % a.min_rho,
                         np.isfinite(rho) and abs(rho) >= a.min_rho,
                         "rho=%+.2f" % rho if np.isfinite(rho) else "n/a"))
        print("     reference: ep_nodiv ~2.10 | ft_comply25 0.655 | arm A 0.413")
    else:
        print("     no 'control' column in the conflict files -- skipped")

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
        # Judged against the noise floor, not exactly. A run reading 1.000
        # against a reference of 1.020 has not regressed by anything the
        # pipeline can resolve -- four rollouts of one checkpoint spread by
        # +-0.04 s -- and flagging it as a failure is how this check first
        # produced a spurious KILL.
        verdicts.append(("headway at alpha=0 >= %.2f s (ref %.2f - noise %.2f)"
                         % (a.ref_headway - a.noise, a.ref_headway, a.noise),
                         np.isfinite(h0) and h0 >= a.ref_headway - a.noise,
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

    # The kill call needs BOTH dial measures to fail. Keying it on headway
    # alone produced a spurious KILL on a run whose all-way-stop min_ttc was
    # perfectly monotone over five points: headway is gap/speed, so an agent
    # that follows faster at the same time gap shows a large speed change and
    # no headway change at all.
    aw = [v for v in verdicts if v[0].startswith("all-way min_ttc range")]
    ok_conf = aw[0][1] if aw else None
    dead = (not ok_dial) and (ok_conf is False or ok_conf is None)

    if not hard:
        print("  VERDICT: continue. The dial moves and safety holds.")
    elif dead:
        print("  VERDICT: KILL THE RUN. Neither the headway dial nor the")
        print("           all-way-stop dial moves, so nothing downstream is")
        print("           interpretable and it will not appear later -- both")
        print("           failed arms were already flat well before 3B.")
    elif not ok_dial and ok_conf:
        print("  VERDICT: continue, with a caveat. The headway dial is weak but")
        print("           the pre-registered all-way-stop dial moves and is")
        print("           monotone. Headway is speed-invariant by construction,")
        print("           so check the speed_fol row before reading anything")
        print("           into its flatness.")
    else:
        print("  VERDICT: the dial works but %d safety/realism check(s) failed."
              % len(hard))
        print("           Range bought at the cost of safety is not the result")
        print("           we want; lower the perturbation or raise the anchor.")
    print("=" * 72)
    return 0 if not hard else 1


if __name__ == "__main__":
    sys.exit(main())
