"""human_style_spread.py -- how much do DRIVERS differ, once the map is taken
out of it? Human against policy, on the same footing.

WHY THE OBVIOUS COMPARISON IS NOT ENOUGH
Comparing the pooled p10-p90 of regimes_gt.csv against a rollout's is
confounded twice over. regimes_gt covers all 10,000 maps while a rollout
covers a few hundred, so the human side sees more SITUATIONS; and a wider
range of situations produces a wider range of behaviour whether or not the
drivers differ at all. A driver in dense traffic has a different headway from
one on an empty road regardless of temperament.

What a driving-style claim needs is BETWEEN-DRIVER variation: how much do
drivers IN THE SAME SCENE differ from each other. That is scene-centred, so
every map effect cancels exactly, and it is the same decomposition
role_scene_icc uses on the role itself.

Three numbers per metric:

  pooled p90-p10     the naive spread. Reported so the difference from the
                     within-scene number is visible rather than hidden.
  within-scene sd    the honest one. Per scene, the sd across drivers in that
                     scene; then the median over scenes. Map effects cancel.
  ICC                between-scene variance / total. High = the metric is
                     mostly telling you which map you are on, so its pooled
                     spread is mostly map variety and not style at all.

READ IT AS: for a role to express a style range, that range has to exist in
the data the role describes. within-scene sd is the ceiling. If humans have
several times the policy's within-scene sd on a metric, anchoring the role to
human style could inherit that; if they do not, no amount of role work will
produce it.

    python human_style_spread.py \
        --gt      /scratch/$USER/regimes/regimes_gt.csv \
        --rollout /scratch/$USER/regimes_rollout/ft_comply25_final_rep2/regimes_natural.csv
"""

import argparse

import numpy as np
import pandas as pd

METRICS = ["headway_T", "jam_s0", "v_freeflow_rel",
           "speed_ff", "speed_fol", "speed_zone", "accel_ff"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, help="regimes_gt.csv")
    p.add_argument("--rollout", action="append", default=[],
                   help="a rollout regimes_*.csv; repeatable")
    p.add_argument("--min_per_scene", type=int, default=2,
                   help="scenes need at least this many drivers with the "
                        "metric before they can say anything about how "
                        "drivers differ")
    p.add_argument("--out", default="")
    return p.parse_args()


def stats(df, col, min_n):
    """pooled spread, within-scene sd, and scene ICC for one metric."""
    if col not in df.columns or "scenario_id" not in df.columns:
        return None
    d = df[["scenario_id", col]].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna()
    if len(d) < 50:
        return None

    v = d[col].to_numpy(float)
    pooled = float(np.percentile(v, 90) - np.percentile(v, 10))

    # within-scene: sd across drivers inside one scene, median over scenes
    g = d.groupby("scenario_id")[col]
    sizes = g.size()
    keep = sizes[sizes >= min_n].index
    sub = d[d["scenario_id"].isin(keep)]
    within = np.nan
    n_scenes = len(keep)
    if n_scenes:
        within = float(sub.groupby("scenario_id")[col].std().median())

    # ICC(1): between-scene share of total variance, unequal group sizes.
    # A metric with a high ICC is largely a property of the map, so its
    # pooled spread is map variety wearing a driver's clothes.
    icc = np.nan
    if n_scenes >= 5:
        grand = sub[col].mean()
        gm = sub.groupby("scenario_id")[col].agg(["mean", "count", "var"])
        k = gm["count"].to_numpy(float)
        n = k.sum()
        msb = float((k * (gm["mean"].to_numpy(float) - grand) ** 2).sum()
                    / max(len(gm) - 1, 1))
        msw = float((np.nan_to_num(gm["var"].to_numpy(float)) * (k - 1)).sum()
                    / max(n - len(gm), 1))
        k0 = (n - (k ** 2).sum() / n) / max(len(gm) - 1, 1)
        denom = msb + (k0 - 1) * msw
        if denom > 0:
            icc = float((msb - msw) / denom)

    return dict(n=len(d), pooled=pooled, within=within,
                n_scenes=n_scenes, icc=icc, median=float(np.median(v)))


def main():
    a = parse_args()
    sources = [("HUMAN", pd.read_csv(a.gt))]
    for r in a.rollout:
        sources.append((r.split("/")[-2][:18], pd.read_csv(r)))

    rows = []
    print("=" * 88)
    print("  BETWEEN-DRIVER STYLE SPREAD  (within-scene = the honest one)")
    print("=" * 88)
    for m in METRICS:
        got = [(name, stats(df, m, a.min_per_scene)) for name, df in sources]
        got = [(n, s) for n, s in got if s]
        if not got:
            continue
        print("\n  %s" % m)
        print("     %-18s %8s %10s %12s %8s %8s"
              % ("source", "median", "p90-p10", "within-sd", "scenes", "ICC"))
        for name, s in got:
            print("     %-18s %8.3f %10.3f %12.3f %8d %8s"
                  % (name, s["median"], s["pooled"], s["within"],
                     s["n_scenes"],
                     "%.3f" % s["icc"] if np.isfinite(s["icc"]) else "--"))
            rows.append(dict(metric=m, source=name, **s))
        # the ratio that decides whether there is headroom
        h = dict(got).get("HUMAN")
        for name, s in got:
            if name == "HUMAN" or not np.isfinite(s["within"]) or not h:
                continue
            if s["within"] > 1e-9:
                print("     -> human within-scene sd is %.2fx %s's"
                      % (h["within"] / s["within"], name))

    print("\n" + "=" * 88)
    print("  within-sd is the ceiling on the style range a role can express:")
    print("  a role cannot encode more variation than exists between drivers")
    print("  in the same situation. Compare THAT column, not p90-p10 -- the")
    print("  pooled figure mixes in how varied the maps were, and the two")
    print("  sides do not cover the same number of maps.")
    print("  A high ICC means the metric is mostly the map, not the driver.")
    print("=" * 88)

    if a.out and rows:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print("\n  wrote %s" % a.out)


if __name__ == "__main__":
    main()
