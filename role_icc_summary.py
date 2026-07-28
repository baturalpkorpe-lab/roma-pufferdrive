"""
role_icc_summary.py -- collapse N repeats of role_scene_icc.py into ONE row.

A single scene-ICC run is not a result: the measured run-to-run spread is
sd 0.005-0.021 depending on the arm (the value of +-0.0035 that circulated
earlier came from two runs that happened to agree). So every arm gets >=3
repeats and is reported as mean +- sd.

Reads role_scene_icc.csv from each --dir and emits:

  role       mean / sd / sem / n over the repeats' role_ALL_DIMS
             (role_ALL_DIMS = pooled between-scene variance over all role dims
             divided by total, NOT a median -- the role dims share one unitless
             latent scale so their variances add meaningfully)
  beh_med    the behaviour reference: each repeat's median ICC over its
             behaviour metrics, then mean/sd of those medians. Median, not
             pooled, because the behaviour metrics are in incommensurable
             units (m/s vs events/100m) and their variances cannot be summed.
  speed_mean the HIGH-RELIABILITY reference. Prefer this over beh_med when
             judging "is the role more scene-determined than behaviour?": the
             median is dragged down by event metrics computed from a handful of
             events per agent, where measurement noise inflates within-scene
             variance and pushes ICC down for reasons that have nothing to do
             with scenes. speed_mean has many samples per agent.
  per-feature mean/sd across repeats for every quantity, so a single noisy
             metric cannot be mistaken for a finding.

n_beh is reported and checked: 7 / 25 / 33 non-NaN behaviour metrics correspond
to three different code eras. The ROLE number is unaffected by which list ran
(behaviour columns are only selected, never used to filter rows), but beh_med is
NOT comparable across eras, so a mismatch is called out loudly.

Usage:
    python role_icc_summary.py --tag mifuture_dim4 \
        --dirs /scratch/$USER/role_icc/mifuture_dim4_rep{1,2,3} \
        --out  /scratch/$USER/role_icc/mifuture_dim4_summary.csv \
        --append /scratch/$USER/role_icc/ALL_ARMS.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_N_BEH = 33          # the current traj_kinematics METRICS list


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True,
                   help="role_scene_icc.py output dirs (one per repeat)")
    p.add_argument("--tag", required=True, help="arm name for the summary row")
    p.add_argument("--out", default="", help="write the one-row summary CSV here")
    p.add_argument("--per_feature", default="",
                   help="write the per-quantity mean/sd table here")
    p.add_argument("--append", default="",
                   help="append the summary row to this cross-arm CSV "
                        "(creates it with a header if absent), so a new arm "
                        "lands next to the existing ones")
    return p.parse_args()


def main():
    args = parse_args()
    rows, frames, missing = [], [], []
    for d in args.dirs:
        f = Path(d) / "role_scene_icc.csv"
        if not f.is_file():
            missing.append(str(f))
            continue
        df = pd.read_csv(f)
        beh = df.loc[df["kind"] == "behaviour", "icc"]
        role = df.loc[df["quantity"] == "role_ALL_DIMS", "icc"]
        spd = df.loc[df["quantity"] == "speed_mean", "icc"]
        rows.append({
            "repeat":     Path(d).name,
            "role":       float(role.iloc[0]) if len(role) else np.nan,
            "beh_med":    float(np.nanmedian(beh)) if len(beh) else np.nan,
            "n_beh":      int(beh.notna().sum()),
            "speed_mean": float(spd.iloc[0]) if len(spd) else np.nan,
            "k_groups":   int(df["k_groups"].max()),
            "n_obs":      int(df["n_obs"].max()),
        })
        frames.append(df.assign(repeat=Path(d).name))

    if missing:
        print("MISSING role_scene_icc.csv (repeat did not finish):")
        for m in missing:
            print(f"  {m}")
    if not rows:
        raise SystemExit("no repeats produced role_scene_icc.csv -- nothing to "
                         "summarise. Check the role_icc_*.out logs.")

    per = pd.DataFrame(rows)
    print(f"\n=== {args.tag}: {len(per)} repeat(s) ===")
    print(per.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    n_beh = sorted(int(v) for v in per["n_beh"].unique())
    if len(n_beh) > 1:
        print(f"\n  *** WARNING: repeats disagree on behaviour metric count "
              f"{n_beh} -- they ran different code. The role numbers are still")
        print( "      comparable (behaviour columns never filter rows), but "
               "beh_med is not. Do not average those medians.")
    elif n_beh[0] != EXPECTED_N_BEH:
        print(f"\n  *** NOTE: n_beh={n_beh[0]}, expected {EXPECTED_N_BEH}. This "
              f"ran an older metric list, so beh_med is NOT comparable")
        print( "      against arms measured on the current list. role is fine.")

    # Averaging medians taken over different metric lists is precisely what
    # the warning above says not to do, so refuse rather than print a number
    # labelled with one era's count. The role figures are unaffected.
    mixed = len(n_beh) > 1
    r = per["role"].values
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
    summary = {
        "arm":            args.tag,
        "n_repeats":      len(per),
        "role_mean":      float(np.mean(r)),
        "role_sd":        sd,
        "role_sem":       (sd / len(r) ** 0.5) if len(r) > 1 else np.nan,
        "role_min":       float(np.min(r)),
        "role_max":       float(np.max(r)),
        "beh_med_mean":   np.nan if mixed else float(per["beh_med"].mean()),
        "beh_med_sd":     np.nan if mixed or len(per) == 1
                          else float(per["beh_med"].std(ddof=1)),
        "speed_mean_icc": float(per["speed_mean"].mean()),
        "n_beh":          -1 if mixed else int(per["n_beh"].iloc[0]),
        "scenes":         int(per["k_groups"].mean()),
        "vehicles":       int(per["n_obs"].mean()),
    }

    print(f"\n  role ICC          : {summary['role_mean']:.3f} "
          f"+- {summary['role_sd']:.3f} (sd)   sem {summary['role_sem']:.3f}   "
          f"n={summary['n_repeats']}")
    if mixed:
        print(f"  behaviour median  : NOT COMPUTED -- repeats used different "
              f"metric lists {n_beh}")
    else:
        print(f"  behaviour median  : {summary['beh_med_mean']:.3f}   "
              f"<- median over {summary['n_beh']} metrics, noisy reference")
    print(f"  speed_mean ICC    : {summary['speed_mean_icc']:.3f}   "
          f"<- high-reliability reference, prefer this one")
    gap = summary["role_mean"] - summary["speed_mean_icc"]
    print(f"  role - speed_mean : {gap:+.3f}   "
          + ("role is LESS scene-determined than speed" if gap < 0 else
             "role is MORE scene-determined than speed"))

    # -- per-quantity mean/sd across repeats --------------------------------
    allf = pd.concat(frames, ignore_index=True)
    pf = (allf.groupby(["kind", "quantity"])["icc"]
              .agg(mean="mean", sd=lambda s: s.std(ddof=1), n="size")
              .reset_index().sort_values(["kind", "mean"]))
    if args.per_feature:
        Path(args.per_feature).parent.mkdir(parents=True, exist_ok=True)
        pf.to_csv(args.per_feature, index=False)
        print(f"\n  per-quantity table -> {args.per_feature}")

    sdf = pd.DataFrame([summary])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        sdf.to_csv(args.out, index=False)
        print(f"  summary row        -> {args.out}")
    if args.append:
        ap = Path(args.append)
        ap.parent.mkdir(parents=True, exist_ok=True)
        if ap.is_file():
            old = pd.read_csv(ap)
            old = old[old["arm"] != args.tag]            # replace, do not duplicate
            sdf = pd.concat([old, sdf], ignore_index=True)
        sdf.sort_values("role_mean").to_csv(ap, index=False)
        print(f"  appended to        -> {ap}")
        print("\n=== all arms so far ===")
        print(sdf.sort_values("role_mean")[
            ["arm", "n_repeats", "role_mean", "role_sd", "beh_med_mean",
             "speed_mean_icc", "n_beh"]].to_string(index=False,
                                                   float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
