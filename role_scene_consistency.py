"""
role_scene_consistency.py -- turn "is the forced-role effect real per scene, or
only on average?" into a NUMBER, from an existing role_paired_agent.csv. No GPU,
no env -- pure pandas, runs on a login node in seconds.

The paired sweep gives, per (episode, sid, vid), each metric at every forcing
alpha. Per scene we take the ENDPOINT delta = metric(+alpha_max) -
metric(-alpha_min). The "consistency rate" = fraction of scenes whose delta has
the SAME SIGN as the population-mean delta:

  ~50%  = coin flip -> the effect is average-only; per scene it's a toss-up
          (this is what "it's just noise" would actually look like)
  ~100% = the dial does the intended thing in (nearly) every individual scene
          -> a genuine per-instance control knob, not just a mean shift

Also reported per (axis, metric):
  pop_mean_delta, paired t   the average effect (what the dose-response shows)
  consistency                sign-match rate vs the population direction
  monotone_rate              stricter: fraction monotone across ALL alphas
  median_|delta|             is the right-sign effect also sizeable
  z_vs_50                    how many SD the consistency sits above a coin flip

Endpoint sign-consistency is only meaningful for metrics with a genuine
MONOTONE population effect (speed, turn under PC1). For V-shaped metrics
(event_rate, jerk at the extremes) the +/-max endpoints roughly cancel, so a
~50% here means "no monotone per-scene effect", NOT "unreliable" -- read those
rows with the pop_mean/t column, not the consistency column.

Optional --traj_clusters stratifies every number by trajectory type, which is
the "useful in DIFFERENT scenarios" question.

Usage:
  python role_scene_consistency.py \
      --agent_csv /scratch/e452103/role_paired/dim2/role_paired_agent.csv \
      --out_dir   /scratch/e452103/role_paired/dim2 \
      [--traj_clusters /scratch/e452103/traj_atlas/trajectory_clusters.csv]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS  = ["speed_mean", "accel_abs", "jerk_abs", "turn_abs", "event_rate"]
KEY      = ["episode", "sid", "vid"]
ALPHA_RE = re.compile(r"(PC\d+)\|([+-]?\d+\.?\d*)")


def parse_cond(c):
    """'PC1|+2' -> ('PC1', 2.0); 'natural'/'base' -> (None, 0.0) baseline."""
    m = ALPHA_RE.match(str(c))
    return (m.group(1), float(m.group(2))) if m else (None, 0.0)


def consistency_row(a, metric):
    """a = rows for ONE axis (all alphas incl. the +/- extremes). Returns the
    per-scene endpoint-delta consistency stats for one metric, or None."""
    alphas = sorted(a["alpha"].unique())
    if len(alphas) < 2:
        return None
    amin, amax = alphas[0], alphas[-1]
    piv = a.pivot_table(index=KEY, columns="alpha", values=metric, aggfunc="mean")
    if amin not in piv.columns or amax not in piv.columns:
        return None

    delta = (piv[amax] - piv[amin]).dropna()
    n = len(delta)
    if n < 3:
        return None
    pop_mean = float(delta.mean())
    pop_sign = np.sign(pop_mean)
    sd = float(delta.std(ddof=1))
    t = pop_mean / (sd / np.sqrt(n)) if sd > 0 else np.nan

    nz = delta[delta != 0]                       # exclude exact ties (int events)
    n_nz = len(nz)
    consistency = float((np.sign(nz) == pop_sign).mean()) if n_nz else np.nan
    z50 = ((consistency - 0.5) / np.sqrt(0.25 / n_nz)
           if n_nz and np.isfinite(consistency) else np.nan)

    # monotone across ALL alphas in the population direction (stricter)
    cols = sorted(piv.columns)
    full = piv[cols].dropna()
    if len(full) and pop_sign != 0:
        d = np.diff(full.values, axis=1)
        mono = (np.all(d >= -1e-9, axis=1) if pop_sign > 0
                else np.all(d <= 1e-9, axis=1))
        monotone = float(mono.mean())
    else:
        monotone = np.nan

    return {"metric": metric, "n": n, "n_nonzero": n_nz,
            "pop_mean_delta": round(pop_mean, 4), "paired_t": round(float(t), 1),
            "consistency": round(consistency, 3) if np.isfinite(consistency) else np.nan,
            "monotone_rate": round(monotone, 3) if np.isfinite(monotone) else np.nan,
            "median_abs_delta": round(float(delta.abs().median()), 4),
            "z_vs_50": round(float(z50), 1) if np.isfinite(z50) else np.nan}


def table_for(df, axis, metrics):
    a = df[df["axis"] == axis]
    rows = [r for m in metrics
            if (r := consistency_row(a, m)) is not None]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_csv",     required=True)
    ap.add_argument("--traj_clusters", default="")
    ap.add_argument("--min_margin",    type=float, default=1.5)
    ap.add_argument("--cluster_names", default="",
                    help="optional 'id:name,id:name'")
    ap.add_argument("--out_dir",       required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.agent_csv)
    parsed = df["cond"].map(parse_cond)
    df["axis"]  = parsed.map(lambda t: t[0])
    df["alpha"] = parsed.map(lambda t: t[1])
    # baseline (natural/base, alpha 0) belongs to every axis -> replicate it in
    axes = sorted(df["axis"].dropna().unique())
    base = df[df["axis"].isna()].copy()
    metrics = [m for m in METRICS if m in df.columns]
    if not axes:
        raise SystemExit("no PC#|alpha conditions found in cond column")
    print(f"[consistency] axes={axes}  metrics={metrics}  rows={len(df)}")

    def with_base(sub):
        """attach the shared alpha=0 baseline rows so endpoints incl. natural
        are available if a sweep only stored one signed side (defensive)."""
        b = base[base[KEY].apply(tuple, axis=1).isin(
            sub[KEY].apply(tuple, axis=1))] if len(base) else base
        return pd.concat([sub, b], ignore_index=True) if len(b) else sub

    names = {}
    if args.cluster_names:
        names = {int(k): v for k, v in
                 (it.split(":") for it in args.cluster_names.split(","))}

    strata = [("ALL", df)]
    if args.traj_clusters:
        tc = pd.read_csv(args.traj_clusters)
        tc = tc[tc["margin"] > args.min_margin].rename(
            columns={"scenario_id": "sid", "vehicle_id": "vid",
                     "cluster": "traj"})
        merged = df.merge(tc[["sid", "vid", "traj"]], on=["sid", "vid"],
                          how="left")
        for k in sorted(merged["traj"].dropna().unique()):
            nm = names.get(int(k), f"traj{int(k)}")
            strata.append((f"C{int(k)}:{nm}", merged[merged["traj"] == k]))

    all_out = []
    for sname, sub in strata:
        for axis in axes:
            tbl = table_for(sub, axis, metrics)
            if tbl.empty:
                continue
            tbl.insert(0, "stratum", sname)
            tbl.insert(1, "axis", axis)
            all_out.append(tbl)
            print(f"\n=== {sname}  {axis}  (endpoint +/-{axis} sign-consistency) ===")
            print(tbl.drop(columns=["stratum", "axis"]).to_string(index=False))

    res = pd.concat(all_out, ignore_index=True)
    res.to_csv(out / "role_scene_consistency.csv", index=False)
    print(f"\n[consistency] wrote {out/'role_scene_consistency.csv'}")
    print("\nRead: consistency ~0.5 = per-scene coin-flip (average-only); "
          "->1.0 = the dial works in nearly every scene. Trust the consistency "
          "column only for metrics whose paired_t is large (a real monotone "
          "effect); for V-shaped metrics read pop_mean/paired_t instead.")


if __name__ == "__main__":
    main()
