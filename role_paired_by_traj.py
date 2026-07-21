"""
role_paired_by_traj.py -- re-stratify the paired forced-role sweep by
TRAJECTORY type instead of map regime. No GPU, no env: joins the existing
role_paired_agent.csv (keyed episode/sid/vid) with trajectory_clusters.csv
(keyed sid/vid) and recomputes all paired statistics per trajectory cluster.

Why: a map regime mislabels within-scene minorities (the exiting seat on a
highway map); the trajectory cluster describes the TASK each seat was given.
Replication of the PC1/PC2 dose-response across task types is a stronger
control than across map types.

Outputs in --out_dir (default = paired --in_dir):
  role_paired_traj_deltas.csv   paired deltas vs alpha=0 per (axis, alpha,
                                traj cluster, metric)
  role_paired_traj_tests.csv    alpha=+2 vs -2 same-(map,vehicle) paired diff
                                per (axis, traj cluster, metric)
  role_paired_traj_PC{d}.png    dose-response, one line per trajectory type

Usage:
    python role_paired_by_traj.py \
        --in_dir /scratch/e452103/role_paired/dim4 \
        --traj_clusters /scratch/e452103/traj_atlas/trajectory_clusters.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["speed_mean", "accel_abs", "accel_pos", "decel_abs", "jerk_abs",
           "turn_abs", "event_rate"]
METRIC_LABEL = {"speed_mean": "speed (m/s)", "accel_abs": "|accel| (m/s2)",
                "accel_pos": "throttle a+ (m/s2)", "decel_abs": "braking |a-| (m/s2)",
                "jerk_abs": "|jerk| (m/s3)", "turn_abs": "|turn| (rad/s)",
                "event_rate": "safety events / 91"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",        type=str, required=True,
                   help="Dir with role_paired_agent.csv")
    p.add_argument("--traj_clusters", type=str, required=True,
                   help="trajectory_clusters.csv from trajectory_atlas.py")
    p.add_argument("--out_dir",       type=str, default="",
                   help="Default = --in_dir")
    p.add_argument("--min_margin",    type=float, default=1.5,
                   help="Keep confidently-assigned trajectories only")
    p.add_argument("--cluster_names", type=str, default="",
                   help="Optional 'id:name,id:name' (from tatlas_names.txt)")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    names = {}
    if args.cluster_names:
        names = {int(k): v for k, v in
                 (it.split(":") for it in args.cluster_names.split(","))}

    df = pd.read_csv(in_dir / "role_paired_agent.csv")
    tc = pd.read_csv(args.traj_clusters)
    tc = tc[tc["margin"] > args.min_margin]
    tc = tc.rename(columns={"scenario_id": "sid", "vehicle_id": "vid",
                            "cluster": "traj_cluster"})
    n0 = len(df)
    df = df.merge(tc[["sid", "vid", "traj_cluster"]], on=["sid", "vid"],
                  how="inner")
    print(f"[traj-strat] {len(df)}/{n0} paired-sweep rows matched to a core "
          f"trajectory cluster")
    clusters = sorted(df["traj_cluster"].unique())
    print(f"[traj-strat] trajectory clusters present: {clusters}")

    # Only plot metrics the agent CSV actually has: older sweeps (pre accel
    # throttle/brake split) lack accel_pos/decel_abs, so intersect with the
    # columns present instead of KeyError-ing on the global METRICS list.
    metrics = [m for m in METRICS if m in df.columns]
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        print(f"[traj-strat] note: {missing} absent from this CSV (older "
              f"sweep) -- skipping them")

    # conditions: "base" is alpha=0; "PC{d}|{alpha}" otherwise
    pcs = sorted({m.group(1) for c in df["cond"].unique()
                  for m in [re.match(r"(PC\d+)\|", str(c))] if m})
    alphas = sorted({float(m.group(1)) for c in df["cond"].unique()
                     for m in [re.match(r"PC\d+\|([+-]?\d+\.?\d*)", str(c))]
                     if m} | {0.0})
    key = ["episode", "sid", "vid"]
    # alpha=0 baseline: old sweeps forced mu ("base"); the redesigned sweep
    # shifts the focal's own role, so its alpha=0 baseline is "natural".
    base_cond = "base" if (df["cond"] == "base").any() else "natural"
    base = df[df["cond"] == base_cond].set_index(key)

    delta_rows, test_rows = [], []
    for pc in pcs:
        cond_of = {al: (f"{pc}|{al:+g}" if al != 0.0 else base_cond)
                   for al in alphas}
        fig, axs = plt.subplots(1, len(metrics),
                                figsize=(3.1 * len(metrics), 3.8))
        for ax, met in zip(np.atleast_1d(axs), metrics):
            for tcid in clusters:
                xr, yr, er = [], [], []
                for al in alphas:
                    sub = df[df["cond"] == cond_of[al]].set_index(key)
                    j = sub.join(base, how="inner", lsuffix="", rsuffix="_b")
                    dd = (j[met] - j[f"{met}_b"])[j["traj_cluster"] == tcid]
                    dd = dd.dropna()
                    # 3 (not 5): rare-focal types (stop&go barely ever wins the
                    # farthest-driving focal pick) sit at ~4 pairs/alpha and
                    # were silently dropped from the figure entirely.
                    if len(dd) < 3:
                        continue
                    xr.append(al)
                    yr.append(float(dd.mean()))
                    er.append(1.96 * float(dd.std() / len(dd) ** 0.5))
                    delta_rows.append({"axis": pc, "alpha": al,
                                       "traj_cluster": tcid, "metric": met,
                                       "mean_delta": float(dd.mean()),
                                       "sem": float(dd.std()/len(dd)**0.5),
                                       "n_pairs": int(len(dd))})
                if xr:
                    ax.errorbar(xr, yr, yerr=er, marker="o", ms=3, capsize=2,
                                lw=1.2, alpha=0.85,
                                label=names.get(tcid, f"traj {tcid}"))
            # headline test per cluster: max vs min alpha, same (map, vehicle)
            lo, hi = min(alphas), max(alphas)
            s_lo = df[df["cond"] == cond_of[lo]].set_index(key)
            s_hi = df[df["cond"] == cond_of[hi]].set_index(key)
            j2 = s_hi.join(s_lo, how="inner", lsuffix="", rsuffix="_lo")
            for tcid in clusters:
                dd2 = (j2[met] - j2[f"{met}_lo"])[j2["traj_cluster"] == tcid]
                dd2 = dd2.dropna()
                if len(dd2) >= 3:
                    test_rows.append({
                        "axis": pc, "metric": met, "traj_cluster": tcid,
                        "hi": hi, "lo": lo,
                        "mean_paired_diff": float(dd2.mean()),
                        "sem": float(dd2.std() / len(dd2) ** 0.5),
                        "t_stat": float(dd2.mean() /
                                        (dd2.std() / len(dd2) ** 0.5)),
                        "n_pairs": int(len(dd2))})
            ax.axhline(0, color="grey", lw=0.7)
            ax.axvline(0, color="grey", lw=0.7, ls=":")
            ax.set_title(f"Δ {METRIC_LABEL[met]}", fontsize=8)
            ax.set_xlabel(f"{pc} (σ)", fontsize=8)
            ax.grid(alpha=0.3)
        h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=7, ncol=min(len(l), 6), loc="lower center")
        fig.suptitle(f"PAIRED dose-response along {pc}, stratified by "
                     f"TRAJECTORY type — Δ vs α=0 on the SAME (map, vehicle)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(out_dir / f"role_paired_traj_{pc}.png", dpi=140)
        plt.close(fig)

    pd.DataFrame(delta_rows).to_csv(out_dir / "role_paired_traj_deltas.csv",
                                    index=False)
    tdf = pd.DataFrame(test_rows)
    tdf.to_csv(out_dir / "role_paired_traj_tests.csv", index=False)

    print("\n[traj-strat] === alpha hi vs lo, SAME (map,vehicle), per "
          "trajectory type ===")
    for pc in pcs:
        for tcid in clusters:
            sub = tdf[(tdf["axis"] == pc) & (tdf["traj_cluster"] == tcid)]
            if sub.empty:
                continue
            parts = "  ".join(
                f"{r['metric']}={r['mean_paired_diff']:+.2f}(t={r['t_stat']:+.0f})"
                for _, r in sub.iterrows())
            n = int(sub["n_pairs"].iloc[0])
            print(f"[traj-strat]  {pc} {names.get(tcid, f'traj {tcid}'):>16} "
                  f"(n={n:>4}): {parts}")
    print(f"\n[traj-strat] outputs -> {out_dir}/role_paired_traj_*")


if __name__ == "__main__":
    main()
