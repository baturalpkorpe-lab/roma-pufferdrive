"""
role_vs_gt.py -- is the role a DRIVING STYLE or is it re-encoding the TASK?

THE CONCERN
Eyeballing the dim-1 natural extremes suggests the role tracks how far the GT
vehicle travels: long trip -> high role, short trip -> low role. If that is what
it is, the latent is not "aggressive vs cautious", it is "long trip vs short
trip", and the behavioural-heterogeneity claim does not stand.

THE MECHANISM THAT WOULD EXPLAIN IT
PufferDrive's ego observation slice carries RELATIVE GOAL x/y (drive.h), and the
goal is the GT endpoint. So trip length is not something the encoder has to
infer from behaviour -- it is handed to it directly, every step. A role trained
to predict a behaviour summary can satisfy that by reading the goal vector,
because far goal => must drive fast => the behaviour summary follows.

WHY IT IS NOT SETTLED BY THE ICC
Scene ICC is 0.170 for the dim-1 role but 0.60-0.64 for speed_mean. If the role
were simply a monotone function of realised speed it would inherit speed's ICC.
It does not, which means the role is NOT absolute speed. Goal distance fits
better: it varies BETWEEN agents inside one scene, so it is agent-level
variance, exactly what a low ICC describes. Relative speed (v_freeflow_rel, the
regime-C style parameter) also fits. This script separates those two.

WHAT IT REPORTS
  1. raw correlations of the role with trip geometry, speed, and validity length
  2. the same WITHIN scene (scene-centred), which removes any map effect
  3. partial correlations: role vs relative speed CONTROLLING for goal distance,
     and role vs goal distance CONTROLLING for relative speed. Whichever
     survives is what the role is actually encoding.
  4. a matched contrast: among agents in the SAME scene with SIMILAR goal
     distance, does the role still separate fast from slow drivers? If not, the
     role is trip length and nothing else.

Usage:
    python role_vs_gt.py \
        --role_csv /scratch/$USER/role_natural/mifutagent_dim1_nodiv/natural_agents.csv \
        --regimes  /scratch/$USER/regimes/regimes_gt.csv
"""

import argparse

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--role_csv", required=True,
                   help="natural_agents.csv (sid, vid, pc1, ...)")
    p.add_argument("--regimes", required=True, help="regimes_gt.csv")
    p.add_argument("--role_col", default="pc1")
    p.add_argument("--out", default="")
    p.add_argument("--goal_tol", type=float, default=10.0,
                   help="metres: 'similar goal distance' for the matched test")
    return p.parse_args()


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan, int(m.sum())
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(m.sum())


def partial(a, b, c):
    """corr(a,b | c) -- the standard first-order partial correlation."""
    rab, n = pearson(a, b)
    rac, _ = pearson(a, c)
    rbc, _ = pearson(b, c)
    if not all(np.isfinite([rab, rac, rbc])):
        return np.nan, n
    den = np.sqrt(max(1 - rac ** 2, 1e-12) * max(1 - rbc ** 2, 1e-12))
    return float((rab - rac * rbc) / den), n


def scene_centre(df, cols, key="scenario_id"):
    """Subtract each scene's own mean: removes everything constant per map."""
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(key)[c].transform("mean")
    return out


def main():
    args = parse_args()
    r = pd.read_csv(args.role_csv)
    g = pd.read_csv(args.regimes)

    # natural_agents.csv uses sid/vid; regimes_gt.csv uses scenario_id/vehicle_id
    r = r.rename(columns={"sid": "scenario_id", "vid": "vehicle_id"})
    for c in ("scenario_id",):
        r[c] = r[c].astype(str)
        g[c] = g[c].astype(str)

    df = r.merge(g, on=["scenario_id", "vehicle_id"], how="inner",
                 suffixes=("_roll", "_gt"))
    if len(df) < 50:
        # scenario ids are sometimes truncated to 16 chars upstream
        r["scenario_id"] = r["scenario_id"].str[:16]
        g["scenario_id"] = g["scenario_id"].str[:16]
        df = r.merge(g, on=["scenario_id", "vehicle_id"], how="inner",
                     suffixes=("_roll", "_gt"))
    if not len(df):
        raise SystemExit("join produced no rows -- check the two id columns")

    role = args.role_col
    for c in ("gt_path_len", "gt_goal_dist", "v_freeflow_rel", "n_valid",
              "headway_T", "v_freeflow"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    spd = "speed_mean_roll" if "speed_mean_roll" in df.columns else "speed_mean"

    print(f"[role_vs_gt] {len(df)} agents joined, "
          f"{df['scenario_id'].nunique()} scenes, role={role}\n")

    cand = [c for c in ("gt_path_len", "gt_goal_dist", spd, "v_freeflow",
                        "v_freeflow_rel", "n_valid", "headway_T")
            if c in df.columns]

    print("  RAW correlation with the role")
    rows = []
    for c in cand:
        rr, n = pearson(df[role].to_numpy(float), df[c].to_numpy(float))
        rows.append(dict(var=c, r_raw=rr, n=n))
        print(f"    {c:<18} r={rr:+.3f}  n={n}")

    dc = scene_centre(df, [role] + cand)
    print("\n  WITHIN SCENE (scene-centred -- any map effect removed)")
    for i, c in enumerate(cand):
        rr, n = pearson(dc[role].to_numpy(float), dc[c].to_numpy(float))
        rows[i]["r_within"] = rr
        print(f"    {c:<18} r={rr:+.3f}  n={n}")

    if "gt_goal_dist" in df.columns and "v_freeflow_rel" in df.columns:
        print("\n  PARTIAL correlations (the decisive pair)")
        a = dc[role].to_numpy(float)
        gd = dc["gt_goal_dist"].to_numpy(float)
        vr = dc["v_freeflow_rel"].to_numpy(float)
        p1, n1 = partial(a, vr, gd)
        p2, n2 = partial(a, gd, vr)
        print(f"    role ~ v_freeflow_rel | goal_dist   r={p1:+.3f}  n={n1}")
        print(f"    role ~ goal_dist | v_freeflow_rel   r={p2:+.3f}  n={n2}")
        print("    -> whichever survives is what the role encodes. If the")
        print("       goal-distance partial stays strong and the relative-speed")
        print("       partial collapses, the latent is trip length.")

        # ---- matched contrast --------------------------------------------
        print(f"\n  MATCHED: same scene, goal distance within "
              f"{args.goal_tol:.0f} m")
        keep = df[np.isfinite(df["gt_goal_dist"]) & np.isfinite(df[role])]
        hi, lo = [], []
        for _, sub in keep.groupby("scenario_id"):
            if len(sub) < 2:
                continue
            v = sub.sort_values("gt_goal_dist")
            gdv = v["gt_goal_dist"].to_numpy()
            rv = v[role].to_numpy()
            sv = v[spd].to_numpy() if spd in v else None
            if sv is None:
                continue
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    if abs(gdv[j] - gdv[i]) > args.goal_tol:
                        break
                    if sv[i] == sv[j]:
                        continue
                    fast, slow = (i, j) if sv[i] > sv[j] else (j, i)
                    hi.append(rv[fast])
                    lo.append(rv[slow])
        if len(hi) >= 20:
            hi, lo = np.array(hi), np.array(lo)
            d = hi - lo
            print(f"    {len(d)} matched pairs; role(faster) - role(slower)")
            print(f"    mean={d.mean():+.4f}  median={np.median(d):+.4f}  "
                  f"P(role higher for faster)={float((d > 0).mean()):.3f}")
            print("    -> ~0.50 means the role is blind to speed once trip")
            print("       length is held fixed, i.e. it IS trip length.")
        else:
            print(f"    only {len(hi)} matched pairs -- widen --goal_tol")

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
