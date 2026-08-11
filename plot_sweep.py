"""plot_sweep.py -- figures from regime_rollout sweeps.

Three figures, each answering a different question a reader will ask.

  dose_response.png   Does the dial move behaviour, and where does HUMAN sit?
                      One panel per feature, alpha across, one line per arm,
                      the human value as a dashed rule. This is the figure that
                      shows the dial and the realism gap at the same time --
                      a line with a slope is a working dial, and its distance
                      from the dashed rule is how far from human it still is.

  dial_range.png      How much does each arm's dial move each feature? Grouped
                      bars. This is the "why we moved on with X" figure: the
                      arms that fail are visibly flat.

  gap_to_human.png    At the natural role, how far is each arm from human, as
                      a fraction of the human value? Signed, so over- and
                      under-shoot are distinguishable -- speed_fol overshooting
                      LOW is a different problem from speed_ff overshooting
                      high, and an absolute-error bar would hide that.

Sample counts are drawn as marker size on the dose-response panels: a style
parameter is read only on the steps where its regime is identified, and the
sweep changes which steps those are. A point resting on 386 observations should
not look the same as one resting on 1800.

    python plot_sweep.py --out_dir /scratch/$USER/figs \
        --gt_regimes   /scratch/$USER/regimes/regimes_gt.csv \
        --gt_conflicts /scratch/$USER/regimes/conflicts_gt.csv \
        "ep_nodiv=/scratch/$USER/regimes_rollout/dim1_v4" \
        "collision=/scratch/$USER/regimes_rollout/collstop_final_rep1"
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, column, source) -- source 0 = regimes csv, 1 = conflicts csv,
# 2 = conflicts restricted to all-way stops
PANELS = [
    ("time headway (s)",       "headway_T",  0),
    ("jam spacing (m)",        "jam_s0",     0),
    ("free-flow speed (m/s)",  "speed_ff",   0),
    ("following speed (m/s)",  "speed_fol",  0),
    ("junction speed (m/s)",   "speed_zone", 0),
    ("|accel| free-flow",      "accel_ff",   0),
    ("PET (s)",                "pet",        1),
    ("minTTC (s)",             "min_ttc",    1),
    ("minTTC, all-way (s)",    "min_ttc",    2),
    ("MRD (m/s2)",             "mrd",        1),
]
COLORS = ["#2b6cb0", "#c0392b", "#2e9e5b", "#e07b1a", "#7b529e", "#555555"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("arms", nargs="+", help="LABEL=/path/to/rollout_dir")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gt_regimes", default="")
    p.add_argument("--gt_conflicts", default="")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def med_n(df, col):
    if df is None or col not in df.columns:
        return np.nan, 0
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return (float(v.median()), len(v)) if len(v) else (np.nan, 0)


def alpha_of(tag):
    return 0.0 if tag == "natural" else float(
        re.sub(r"^[A-Za-z0-9_\-]*?(?=[+-]\d)", "", tag))


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "regimes_*.csv"))):
        t = os.path.basename(f)[8:-4]
        cf = os.path.join(d, "conflicts_%s.csv" % t)
        out[alpha_of(t)] = (pd.read_csv(f),
                            pd.read_csv(cf) if os.path.exists(cf) else None)
    return dict(sorted(out.items()))


def pick(arms_at_alpha, src):
    """Return the dataframe a panel's source refers to."""
    reg, conf = arms_at_alpha
    if src == 0:
        return reg
    if src == 1:
        return conf
    if conf is None or "control" not in conf.columns:
        return None
    return conf[conf["control"] == "all_way_stop"]


def human_value(hr, hc, col, src):
    if src == 0:
        return med_n(hr, col)[0]
    if src == 1:
        return med_n(hc, col)[0]
    if hc is None or "control" not in hc.columns:
        return np.nan
    return med_n(hc[hc["control"] == "all_way_stop"], col)[0]


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    arms = []
    for spec in a.arms:
        if "=" not in spec:
            raise SystemExit("expected LABEL=path, got %r" % spec)
        lab, path = spec.split("=", 1)
        d = load(path)
        if not d:
            print("  skipping %s -- no regimes_*.csv in %s" % (lab, path))
            continue
        arms.append((lab, d))
    if not arms:
        raise SystemExit("no usable rollout dirs")

    hr = pd.read_csv(a.gt_regimes) if a.gt_regimes and os.path.exists(a.gt_regimes) else None
    hc = pd.read_csv(a.gt_conflicts) if a.gt_conflicts and os.path.exists(a.gt_conflicts) else None

    # ---- 1. dose response -------------------------------------------------
    n = len(PANELS)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for k, (title, col, src) in enumerate(PANELS):
        ax = axes[k]
        for i, (lab, d) in enumerate(arms):
            al = list(d)
            vs, ns = [], []
            for x in al:
                v, nn = med_n(pick(d[x], src), col)
                vs.append(v)
                ns.append(nn)
            if not any(np.isfinite(v) for v in vs):
                continue
            c = COLORS[i % len(COLORS)]
            ax.plot(al, vs, "-", color=c, lw=1.6, zorder=3,
                    label=lab if k == 0 else None)
            # marker area tracks sample count: a point on 386 observations
            # should not look like one on 1800
            mx = max(ns) or 1
            ax.scatter(al, vs, s=[18 + 60 * (x / mx) for x in ns],
                       color=c, zorder=4, edgecolor="white", linewidth=.6)
        h = human_value(hr, hc, col, src)
        if np.isfinite(h):
            ax.axhline(h, ls="--", lw=1.3, color="#111111", zorder=2)
            ax.annotate("human", (ax.get_xlim()[0], h), fontsize=7.5,
                        va="bottom", ha="left", color="#111111")
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=.25)
        ax.set_xlabel("role  alpha", fontsize=8)
        ax.tick_params(labelsize=8)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.legend(loc="lower right", fontsize=9, frameon=True,
               bbox_to_anchor=(0.99, 0.02))
    fig.suptitle("Dose response: does forcing the role move behaviour, "
                 "and where is human?", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p1 = os.path.join(a.out_dir, "dose_response.png")
    fig.savefig(p1, dpi=a.dpi)
    plt.close(fig)
    print("  wrote %s" % p1)

    # ---- 2. dial range ----------------------------------------------------
    labels = [t for t, _, _ in PANELS]
    fig, ax = plt.subplots(figsize=(max(8, 1.05 * len(PANELS)), 4.2))
    w = 0.8 / len(arms)
    for i, (lab, d) in enumerate(arms):
        rng = []
        for _, col, src in PANELS:
            vs = [med_n(pick(d[x], src), col)[0] for x in d]
            vs = [v for v in vs if np.isfinite(v)]
            rng.append(max(vs) - min(vs) if len(vs) > 1 else np.nan)
        ax.bar(np.arange(len(PANELS)) + i * w - 0.4 + w / 2, rng, w,
               label=lab, color=COLORS[i % len(COLORS)])
    ax.set_xticks(range(len(PANELS)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("swept range  (max - min over alpha)")
    ax.set_title("How far the role dial moves each feature", weight="bold")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(a.out_dir, "dial_range.png")
    fig.savefig(p2, dpi=a.dpi)
    plt.close(fig)
    print("  wrote %s" % p2)

    # ---- 3. signed gap to human at alpha=0 --------------------------------
    fig, ax = plt.subplots(figsize=(max(8, 1.05 * len(PANELS)), 4.2))
    for i, (lab, d) in enumerate(arms):
        if 0.0 not in d:
            continue
        gaps = []
        for _, col, src in PANELS:
            v = med_n(pick(d[0.0], src), col)[0]
            h = human_value(hr, hc, col, src)
            gaps.append((v - h) / h if np.isfinite(v) and np.isfinite(h)
                        and abs(h) > 1e-9 else np.nan)
        ax.bar(np.arange(len(PANELS)) + i * w - 0.4 + w / 2, gaps, w,
               label=lab, color=COLORS[i % len(COLORS)])
    ax.axhline(0, color="#111111", lw=1.2)
    ax.set_xticks(range(len(PANELS)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("(policy - human) / human   at alpha = 0")
    ax.set_title("Signed gap to human at the natural role  "
                 "(0 = human; sign shows which way it is wrong)",
                 weight="bold")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p3 = os.path.join(a.out_dir, "gap_to_human.png")
    fig.savefig(p3, dpi=a.dpi)
    plt.close(fig)
    print("  wrote %s" % p3)


if __name__ == "__main__":
    main()
