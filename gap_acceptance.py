"""
gap_acceptance.py -- the regime-A analysis: does a role move the go/yield
decision, and by how many seconds?

MODEL
    P(ego passes the conflict point first) = sigma(b0 + b1*TA [+ b2*role])

TA is `ta_at_decision`: the projected time by which EGO arrives LATER, read when
the follower is still --tau_dec seconds of travel from the conflict point, i.e.
BEFORE the interaction resolves. So b1 should be NEGATIVE -- the later you are
projected to arrive, the less likely you go first.

CRITICAL GAP
    t_c = -(b0 + b2*role) / b1
the TA at which the driver is indifferent, in seconds. A driver who goes first
even when projected to arrive 1 s late has t_c = +1 and is assertive; one who
yields even when projected 1 s early has t_c = -1.

WHAT t_c MEANS AND DOES NOT MEAN
On ground truth with no role term, t_c comes out at ~0 BY CONSTRUCTION, and that
is a validation rather than a finding. Every conflict contributes two mirrored
rows -- for the pair (A,B), row A has (TA, went) and row B has (-TA, 1-went) --
so the symmetric fit forces b0 ~ 0. t_c is only informative as a DIFFERENCE:
between role levels, between right-of-way classes, or between GT and a policy.
Read the deltas, never the absolute value on its own.

Because of that mirroring the rows are not independent, so the bootstrap
resamples CONFLICTS (both rows together), not rows. Fitting on rows and taking
the naive Hessian would understate the standard errors by about sqrt(2).

WHY THIS METRIC
`ego_went_first` is binary and bounded, so unlike speed it cannot be compressed
by the stratum it is measured in, and it is an outcome that never enters the
stratum definition. Contrast accel_abs = 10*TV(speed)/N, which cannot move under
a sweep at all.

Usage:
    python gap_acceptance.py --conflicts /scratch/$USER/regimes/conflicts_gt.csv
    python gap_acceptance.py --conflicts rollout.csv \
        --role_csv /scratch/$USER/analysis/<tag>/paired/natural_agents.csv \
        --role_col pc1
    python gap_acceptance.py --selftest      # no data needed
"""

import argparse

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--conflicts", help="conflicts_gt.csv or a rollout equivalent")
    p.add_argument("--role_csv", default="",
                   help="optional: scenario_id, vehicle_id, <role_col>")
    p.add_argument("--role_col", default="pc1",
                   help="role column. Taken from --role_csv if given, else used "
                        "directly if the conflicts file already has it -- which "
                        "regime_rollout output does, as role_dec_<d>.")
    p.add_argument("--out", default="", help="optional CSV of the fit table")
    p.add_argument("--boot", type=int, default=400)
    p.add_argument("--min_n", type=int, default=60,
                   help="skip groups with fewer conflicts than this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logistic regression (IRLS) -- no sklearn/statsmodels dependency
# ---------------------------------------------------------------------------

def logistic_fit(X, y, ridge=1e-6, iters=100):
    """Newton-Raphson. Tiny ridge keeps the Hessian invertible when a group is
    perfectly separated, which happens in small right-of-way strata."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9)
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p) - ridge * b
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def critical_gap(b, role=None):
    """t_c = -(b0 + b2*role) / b1, in seconds. NaN if the slope is degenerate."""
    if abs(b[1]) < 1e-8:
        return np.nan
    num = b[0] + (b[2] * role if role is not None and len(b) > 2 else 0.0)
    return float(-num / b[1])


def design(df, role_col=None):
    X = [np.ones(len(df)), df["ta_at_decision"].to_numpy(float)]
    if role_col:
        X.append(df[role_col].to_numpy(float))
    return np.stack(X, 1), df["ego_went_first"].to_numpy(float)


def fit_group(df, role_col, boot, rng):
    """Fit + conflict-level bootstrap. Returns a dict of estimates and CIs."""
    X, y = design(df, role_col)
    b = logistic_fit(X, y)
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))

    cids = df["cid"].to_numpy()
    uniq = np.unique(cids)
    idx_of = {c: np.flatnonzero(cids == c) for c in uniq}

    tc_bs, b1_bs, brole_bs = [], [], []
    for _ in range(boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_of[c] for c in pick])
        bb = logistic_fit(X[rows], y[rows])
        tc_bs.append(critical_gap(bb))
        b1_bs.append(bb[1])
        if role_col:
            brole_bs.append(bb[2])

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if not len(v):
            return (np.nan, np.nan)
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    out = dict(
        n_conflicts=len(uniq), n_rows=len(df),
        base_rate=round(float(y.mean()), 3),
        b_ta=round(float(b[1]), 4), b_ta_lo=round(ci(b1_bs)[0], 4),
        b_ta_hi=round(ci(b1_bs)[1], 4),
        t_c=round(critical_gap(b), 3),
        t_c_lo=round(ci(tc_bs)[0], 3), t_c_hi=round(ci(tc_bs)[1], 3),
        accuracy=round(float(((p > 0.5) == (y > 0.5)).mean()), 3),
    )
    if role_col:
        out["b_role"] = round(float(b[2]), 4)
        out["b_role_lo"] = round(ci(brole_bs)[0], 4)
        out["b_role_hi"] = round(ci(brole_bs)[1], 4)
        r = df[role_col].to_numpy(float)
        lo, hi = np.nanpercentile(r, [10, 90])
        out["t_c_role_p10"] = round(critical_gap(b, lo), 3)
        out["t_c_role_p90"] = round(critical_gap(b, hi), 3)
        out["d_t_c"] = round(out["t_c_role_p90"] - out["t_c_role_p10"], 3)
    return out


# ---------------------------------------------------------------------------

def load(args):
    df = pd.read_csv(args.conflicts)
    need = {"scenario_id", "vehicle_id", "other_id",
            "ego_went_first", "ta_at_decision"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"conflicts csv missing columns: {sorted(missing)}")
    df["ta_at_decision"] = pd.to_numeric(df["ta_at_decision"], errors="coerce")
    df = df[np.isfinite(df["ta_at_decision"])].copy()

    # conflict id: both mirrored rows of a pair share it, so the bootstrap can
    # resample them together
    lo = np.minimum(df["vehicle_id"], df["other_id"]).astype(str)
    hi = np.maximum(df["vehicle_id"], df["other_id"]).astype(str)
    df["cid"] = df["scenario_id"].astype(str) + "_" + lo + "_" + hi

    role_col = None
    if args.role_csv:
        r = pd.read_csv(args.role_csv)
        if args.role_col not in r.columns:
            raise SystemExit(f"--role_col {args.role_col} not in {args.role_csv}; "
                             f"columns: {list(r.columns)[:12]}")
        r = r[["scenario_id", "vehicle_id", args.role_col]].drop_duplicates(
            ["scenario_id", "vehicle_id"])
        # If the conflicts file already carries a column of this name, the merge
        # would silently produce <col>_x / <col>_y and every later lookup would
        # KeyError. The role CSV is the authority, so drop the local copy.
        if args.role_col in df.columns:
            df = df.drop(columns=[args.role_col])
        n0 = len(df)
        df = df.merge(r, on=["scenario_id", "vehicle_id"], how="inner")
        df = df[np.isfinite(pd.to_numeric(df[args.role_col], errors="coerce"))]
        print(f"[gap] role join kept {len(df)}/{n0} rows")
        role_col = args.role_col
    elif args.role_col in df.columns:
        # The rollout conflicts already carry role_dec_<d>: the role the policy
        # was ACTING ON at the decision step, which is a better covariate than
        # any episode mean joined in from outside. Use it directly rather than
        # forcing a self-join.
        df[args.role_col] = pd.to_numeric(df[args.role_col], errors="coerce")
        df = df[np.isfinite(df[args.role_col])]
        role_col = args.role_col
        print(f"[gap] using in-file role column '{role_col}' ({len(df)} rows)")
    return df, role_col


def selftest():
    """Recover a known critical gap from simulated decisions."""
    rng = np.random.default_rng(0)
    k, tc_true, n = 1.4, 0.8, 4000
    ta = rng.normal(0, 3, n)
    p = 1.0 / (1.0 + np.exp(k * (ta - tc_true)))       # sigma(-k(ta - tc))
    y = (rng.random(n) < p).astype(float)
    X = np.stack([np.ones(n), ta], 1)
    b = logistic_fit(X, y)
    tc = critical_gap(b)
    print(f"  true  k={k:.2f}  t_c={tc_true:.2f}")
    print(f"  fit   b_ta={b[1]:.3f} (expect ~{-k:.2f})   t_c={tc:.3f}")
    ok = abs(tc - tc_true) < 0.15 and abs(b[1] + k) < 0.15
    print("  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    args = parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if not args.conflicts:
        raise SystemExit("--conflicts is required (or use --selftest)")

    df, role_col = load(args)
    rng = np.random.default_rng(args.seed)
    print(f"[gap] {len(df)} rows, {df['cid'].nunique()} conflicts"
          f"{'  role=' + role_col if role_col else '  (no role term)'}")

    groups = [("ALL", df)]
    for col in ("kind", "control"):
        if col in df.columns:
            for v, sub in df.groupby(col):
                groups.append((f"{col}={v}", sub))

    rows = []
    for name, sub in groups:
        if sub["cid"].nunique() < args.min_n:
            continue
        r = fit_group(sub, role_col, args.boot, rng)
        r["group"] = name
        rows.append(r)

    if not rows:
        raise SystemExit("no group had enough conflicts to fit")
    out = pd.DataFrame(rows)
    cols = ["group", "n_conflicts", "base_rate", "b_ta", "b_ta_lo", "b_ta_hi",
            "t_c", "t_c_lo", "t_c_hi", "accuracy"]
    if role_col:
        cols += ["b_role", "b_role_lo", "b_role_hi",
                 "t_c_role_p10", "t_c_role_p90", "d_t_c"]
    out = out[cols]
    print()
    print(out.to_string(index=False))

    print("\n  READ IT AS:")
    print("   b_ta must be NEGATIVE -- later projected arrival, less likely to go.")
    print("   base_rate ~0.500 is forced by the mirrored rows; it is a file")
    print("   integrity check, not a result.")
    if role_col:
        print("   d_t_c is the headline: seconds of critical gap between the")
        print("   10th and 90th percentile of the role. If its bootstrap CI on")
        print("   b_role excludes 0, the role moves the go/yield decision.")
    else:
        print("   t_c ~0 here is EXPECTED on GT and validates the pipeline.")
        print("   Re-run with --role_csv to get the number that matters.")
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
