"""
role_within_scene_corr.py -- is the observational-vs-sweep sign flip caused by
the MAP, or by something lowering the ICC will never fix?

THE PLAIN-LANGUAGE VERSION
--------------------------
role_paired_observational.png correlates the role projection against behaviour
over ALL agents at once. The paired sweep instead forces one agent's role up
and down and watches what that same agent does. The two sometimes disagree in
SIGN, and we have been calling that disagreement "the map effect".

There are only two ways a cross-agent correlation can disagree with the causal
sweep:

  (A) BETWEEN scenes. Fast scenes hand out high roles AND high speeds, so
      pooling all agents shows "high role = fast" even if pushing any single
      agent's role up slows it down. This is the map effect. Lowering the
      scene ICC shrinks it.

  (B) WITHIN a scene. Even comparing agents standing next to each other in the
      SAME scene, the encoder gives high roles to the ones already driving
      fast -- it is describing them, not causing them. Lowering the scene ICC
      does NOTHING to this, because there is no scene difference involved.

This script separates the two by "scene-centering": inside every scene, it
subtracts that scene's own average from both the role projection and the
behaviour, then correlates the leftovers. That throws away every difference
BETWEEN scenes and keeps only differences BETWEEN AGENTS IN THE SAME SCENE.

  r_all     the original correlation over all agents (what the figure shows)
  r_within  the same correlation after scene-centering

Read it like this:

  r_all and r_within have DIFFERENT signs
      -> the flip was case (A). Between-scene structure was driving it. This
         is the map effect, and reducing scene ICC should shrink it.

  r_all and r_within have the SAME sign, and it still disagrees with the sweep
      -> case (B). The disagreement survives inside single scenes, so it is
         not about maps at all and no amount of ICC reduction will remove it.
         Pass --tests_csv to have that comparison made for you.

Inputs (both already produced by the existing pipeline):
  --agents_csv  role_icc/<run>/role_scene_icc_agents.csv   (sid, vid, role_*,
                pc1, pc2, and every behaviour metric -- one row per agent)
  --tests_csv   analysis/<tag>/paired/role_paired_tests.csv  (optional; supplies
                the CAUSAL direction from the sweep so the verdict is automatic)

No GPU, no env, no rollouts -- pure pandas over CSVs you already have.

Usage:
    python role_within_scene_corr.py \
        --agents_csv /scratch/$USER/role_icc/mifutagent_div_v33_rep1/role_scene_icc_agents.csv \
        --tests_csv  /scratch/$USER/analysis/mifutagent_dim4/paired/role_paired_tests.csv \
        --tag mifutagent_div \
        --out_dir    /scratch/$USER/role_icc/within_scene
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_AGENTS = 2          # a scene needs >=2 agents to have any within-scene info
MIN_PAIRS  = 30         # below this a correlation is noise
FLAT       = 1e-9       # a series with no spread has no correlation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agents_csv", required=True,
                   help="role_scene_icc_agents.csv from role_scene_icc.py")
    p.add_argument("--tests_csv", default="",
                   help="role_paired_tests.csv from the sweep -- supplies the "
                        "causal sign so the verdict can be decided for you")
    p.add_argument("--axes_csv", default="",
                   help="role_paired_axes.csv -- only needed if the agents CSV "
                        "has no pc* columns (it normally does)")
    p.add_argument("--tag", default="run", help="name for the outputs")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_agents", type=int, default=MIN_AGENTS)
    return p.parse_args()


def corr(x, y):
    """Pearson r, or nan when either side is flat / too few points."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_PAIRS:
        return np.nan, int(ok.sum())
    xs, ys = x[ok], y[ok]
    if xs.std() < FLAT or ys.std() < FLAT:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(xs, ys)[0, 1]), int(ok.sum())


def scene_center(df, cols, scene_col="sid"):
    """Subtract each scene's own mean from every column -- what is left is
    purely how an agent differs from its NEIGHBOURS in that same scene."""
    return df[cols] - df.groupby(scene_col)[cols].transform("mean")


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.agents_csv)
    if "sid" not in df.columns:
        raise SystemExit(f"no 'sid' column in {args.agents_csv} -- this needs "
                         f"role_scene_icc_agents.csv, not the warmup CSV "
                         f"(the warmup one has no scene id, so scenes cannot "
                         f"be separated).")

    # -- role projections: prefer the pc* columns the ICC run already wrote ---
    pc_cols = sorted([c for c in df.columns if c.startswith("pc")
                      and c[2:].isdigit()], key=lambda c: int(c[2:]))
    if not pc_cols:
        if not args.axes_csv:
            raise SystemExit("no pc* columns and no --axes_csv given")
        role_cols = sorted([c for c in df.columns if c.startswith("role_")
                            and c[5:].isdigit()], key=lambda c: int(c[5:]))
        ax = pd.read_csv(args.axes_csv)
        cc = [f"c{i}" for i in range(len(role_cols))]
        mu = ax[ax["name"] == "mu"][cc].values[0].astype(float)
        cen = df[role_cols].values.astype(float) - mu
        for _, r in ax[ax["name"] != "mu"].iterrows():
            df[str(r["name"]).lower()] = cen @ r[cc].values.astype(float)
        pc_cols = sorted([c for c in df.columns if c.startswith("pc")
                          and c[2:].isdigit()], key=lambda c: int(c[2:]))
        print(f"[within] projected onto axes from {args.axes_csv}")

    skip = set(pc_cols) | {"sid", "vid", "episode"} | \
           {c for c in df.columns if c.startswith("role_")}
    beh_cols = [c for c in df.columns if c not in skip
                and pd.api.types.is_numeric_dtype(df[c])]

    # -- keep only scenes that carry within-scene information ----------------
    sizes = df.groupby("sid")["vid"].transform("size")
    n_before, n_scenes_before = len(df), df["sid"].nunique()
    dfw = df[sizes >= args.min_agents].copy()
    print(f"[within] {n_before} agents / {n_scenes_before} scenes -> "
          f"{len(dfw)} agents / {dfw['sid'].nunique()} scenes with "
          f">={args.min_agents} agents "
          f"({len(dfw)/max(dfw['sid'].nunique(),1):.1f} per scene)")

    cen = scene_center(dfw, pc_cols + beh_cols)

    # -- causal direction from the sweep, if provided ------------------------
    causal = {}
    if args.tests_csv and Path(args.tests_csv).is_file():
        t = pd.read_csv(args.tests_csv)
        for _, r in t.iterrows():
            causal[(str(r["axis"]).lower(), str(r["metric"]))] = \
                float(r["mean_paired_diff"])
        print(f"[within] causal signs for {len(causal)} (axis, metric) pairs "
              f"from {args.tests_csv}")

    rows = []
    for pc in pc_cols:
        for m in beh_cols:
            r_all, n_all = corr(dfw[pc].values.astype(float),
                                dfw[m].values.astype(float))
            r_win, n_win = corr(cen[pc].values.astype(float),
                                cen[m].values.astype(float))
            c = causal.get((pc, m), np.nan)
            rows.append({"axis": pc, "metric": m, "r_all": r_all,
                         "r_within": r_win, "n": n_all,
                         "causal_diff": c,
                         "flip_all_vs_within": bool(
                             np.isfinite(r_all) and np.isfinite(r_win)
                             and np.sign(r_all) != np.sign(r_win))})
    res = pd.DataFrame(rows)

    # -- verdict per (axis, metric) -----------------------------------------
    def verdict(r):
        if not (np.isfinite(r["r_all"]) and np.isfinite(r["r_within"])):
            return "insufficient data"
        if not np.isfinite(r["causal_diff"]):
            return ("between-scene structure dominates"
                    if r["flip_all_vs_within"] else "no between-scene flip")
        s_c = np.sign(r["causal_diff"])
        s_a, s_w = np.sign(r["r_all"]), np.sign(r["r_within"])
        if s_a == s_c:
            return "observational already agrees with the sweep"
        if s_w == s_c:
            return "MAP EFFECT -- scene-centering fixes the disagreement"
        return "NOT the map -- disagreement survives inside single scenes"

    res["verdict"] = res.apply(verdict, axis=1)
    res.to_csv(out / f"{args.tag}_within_scene.csv", index=False)

    print(f"\n{'axis':5} {'metric':22} {'r_all':>7} {'r_within':>9} "
          f"{'causal':>8}  verdict")
    for _, r in res.iterrows():
        cs = f"{r['causal_diff']:+.3f}" if np.isfinite(r["causal_diff"]) else "   --"
        ra = f"{r['r_all']:+.3f}" if np.isfinite(r["r_all"]) else "   nan"
        rw = f"{r['r_within']:+.3f}" if np.isfinite(r["r_within"]) else "   nan"
        print(f"{r['axis']:5} {r['metric']:22} {ra:>7} {rw:>9} {cs:>8}  "
              f"{r['verdict']}")

    if causal.__len__():
        print("\n=== SUMMARY ===")
        for pc in pc_cols:
            sub = res[(res["axis"] == pc) & res["causal_diff"].notna()]
            if sub.empty:
                continue
            n_map = (sub["verdict"].str.startswith("MAP EFFECT")).sum()
            n_not = (sub["verdict"].str.startswith("NOT the map")).sum()
            n_ok = (sub["verdict"].str.startswith("observational")).sum()
            print(f"  {pc}: {n_ok} agree, {n_map} explained by the map, "
                  f"{n_not} NOT explained by the map (of {len(sub)})")
        print("\n  'NOT explained by the map' means the encoder describes the "
              "agent rather than\n  causing it, INSIDE one scene. Lowering the "
              "scene ICC cannot remove those.")

    # -- figure: r_all vs r_within, one panel per axis ----------------------
    fig, axs = plt.subplots(1, len(pc_cols), figsize=(5.2 * len(pc_cols), 5.0),
                            squeeze=False)
    for j, pc in enumerate(pc_cols):
        ax = axs[0][j]
        sub = res[res["axis"] == pc].dropna(subset=["r_all", "r_within"])
        ax.axhline(0, color="grey", lw=.8); ax.axvline(0, color="grey", lw=.8)
        ax.plot([-1, 1], [-1, 1], ls=":", color="grey", lw=.8)
        flip = sub["flip_all_vs_within"]
        ax.scatter(sub.loc[~flip, "r_all"], sub.loc[~flip, "r_within"],
                   s=28, color="#4477aa", label="same sign")
        ax.scatter(sub.loc[flip, "r_all"], sub.loc[flip, "r_within"],
                   s=44, color="#cc3311", label="sign flips when scene-centered")
        for _, r in sub.iterrows():
            ax.annotate(r["metric"], (r["r_all"], r["r_within"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"r over ALL agents  ({pc})")
        ax.set_ylabel("r WITHIN scene (scene-centered)")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_title(f"{args.tag} — {pc}", fontsize=10)
        ax.grid(alpha=.3); ax.legend(fontsize=7, loc="best")
    fig.suptitle("Points off the dotted line moved when between-scene "
                 "differences were removed.\nPoints in the top-left / "
                 "bottom-right quadrants CHANGED SIGN -- those correlations "
                 "were driven by which scene the agent was in.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / f"{args.tag}_within_scene.png", dpi=150)
    plt.close(fig)
    print(f"\n[within] -> {out}/{args.tag}_within_scene.csv + .png")


if __name__ == "__main__":
    main()
