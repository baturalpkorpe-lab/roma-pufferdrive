"""
trajectory_taxonomy.py -- the 2x2 turn x stop trajectory taxonomy.

Replaces K-means on the trajectory features with two thresholds on the only
two genuinely BIMODAL (gap-separated) features -- net_turn and stop_frac --
so every stratum is defined by a real distributional valley, not an arbitrary
cut through a continuum. Speed is a within-type COVARIATE, never a boundary.

    is_turn = net_turn  > --turn_thr   (~0.5 rad: took a corner)
    is_stop = stop_frac > --stop_thr   (~0.3: spent real time stopped)

    cluster 0  free-flow   (straight, no stop)
    cluster 1  stop&go     (straight, stops)
    cluster 2  turning     (turns,   no stop)
    cluster 3  turning stop&go (turns, stops)

Output = trajectory_clusters.csv in the SAME schema the downstream tools read
(scenario_id, vehicle_id, cluster, margin, cluster2, is_edge) so
role_paired_by_traj / render_role_compare consume it unchanged. There is no
K-means margin here, so "confidence" is distance from each threshold: a
trajectory near BOTH thresholds is the honest edge case (is_edge=1).

Also prints the 2x2 table with per-cell speed (the covariate) and draws a
diagnostic figure: the net_turn & stop_frac histograms with the chosen
thresholds, and the cells in (speed x ...) space.

No GPU. Usage:
    python trajectory_taxonomy.py \
        --features_csv /scratch/e452103/traj_atlas/trajectory_features.csv \
        --out_dir /scratch/e452103/traj_atlas/taxonomy
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {0: "free-flow", 1: "stop&go", 2: "turning", 3: "turning stop&go"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_csv", type=str, required=True,
                   help="trajectory_features.csv WITH stop_frac")
    p.add_argument("--out_dir",   type=str, required=True)
    p.add_argument("--turn_thr",  type=float, default=0.5,
                   help="net_turn (rad) above this = turning")
    p.add_argument("--stop_thr",  type=float, default=0.3,
                   help="stop_frac above this = stop&go")
    p.add_argument("--edge_band", type=float, default=0.15,
                   help="relative band around EITHER threshold that flags an "
                        "edge trajectory (fraction of the threshold)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.read_csv(args.features_csv)
    for c in ("net_turn", "stop_frac", "speed_mean"):
        if c not in df.columns:
            raise SystemExit(f"features_csv missing '{c}'")
    df = df.dropna(subset=["net_turn", "stop_frac", "speed_mean"]).reset_index(drop=True)

    is_turn = df["net_turn"].values  > args.turn_thr
    is_stop = df["stop_frac"].values > args.stop_thr
    cluster = (is_turn.astype(int) * 2 + is_stop.astype(int))   # 0..3

    # 2nd-nearest cell = flip whichever binary switch is CLOSEST to its
    # threshold; edge = within edge_band (relative) of either threshold.
    dt = np.abs(df["net_turn"].values  - args.turn_thr) / max(args.turn_thr, 1e-9)
    ds = np.abs(df["stop_frac"].values - args.stop_thr) / max(args.stop_thr, 1e-9)
    flip_turn = dt < ds
    cluster2 = np.where(flip_turn,
                        ((~is_turn).astype(int) * 2 + is_stop.astype(int)),
                        (is_turn.astype(int) * 2 + (~is_stop).astype(int)))
    is_edge = (dt < args.edge_band) | (ds < args.edge_band)
    # margin analogue: relative distance to the NEAREST threshold (bigger=safer)
    margin = np.minimum(dt, ds).round(3)

    pd.DataFrame({"scenario_id": df["scenario_id"].values,
                  "vehicle_id":  df["vehicle_id"].values,
                  "cluster": cluster, "margin": margin,
                  "cluster2": cluster2, "is_edge": is_edge.astype(int)}
                 ).to_csv(out / "trajectory_clusters.csv", index=False)

    # -- report: the 2x2 table with speed covariate ---------------------------
    print(f"[tax] thresholds: net_turn>{args.turn_thr}  stop_frac>{args.stop_thr}"
          f"   ({len(df)} trajectories, {is_edge.mean():.0%} edge)")
    print(f"[tax] {'cell':<18} {'n':>7} {'%':>5}  {'speed_mean':>10} "
          f"{'net_turn':>9} {'stop_frac':>9}")
    lines = []
    for c in range(4):
        m = cluster == c
        if not m.any():
            continue
        line = (f"[tax] {NAMES[c]:<18} {int(m.sum()):>7} {m.mean():>4.0%}  "
                f"{df['speed_mean'][m].mean():>10.2f} "
                f"{df['net_turn'][m].mean():>9.2f} "
                f"{df['stop_frac'][m].mean():>9.2f}")
        print(line); lines.append(line)
    with open(out / "taxonomy_table.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    # -- figure: threshold histograms + the 2x2 in (speed, net_turn) ----------
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    axs[0].hist(df["net_turn"], bins=80, range=(0, 3), color="#4477aa")
    axs[0].axvline(args.turn_thr, color="red", lw=2)
    axs[0].set_title("net_turn (bimodal: turn vs straight)")
    axs[0].set_xlabel("net_turn (rad)"); axs[0].set_yscale("log")

    axs[1].hist(df["stop_frac"], bins=60, range=(0, 1), color="#ee6677")
    axs[1].axvline(args.stop_thr, color="red", lw=2)
    axs[1].set_title("stop_frac (bimodal: stop vs free-flow)")
    axs[1].set_xlabel("stop_frac"); axs[1].set_yscale("log")

    cmap = plt.get_cmap("tab10")
    sub = np.random.default_rng(0).choice(len(df), min(12000, len(df)),
                                          replace=False)
    for c in range(4):
        m = (cluster[sub] == c)
        if m.any():
            axs[2].scatter(df["speed_mean"].values[sub][m],
                           df["net_turn"].values[sub][m],
                           s=5, alpha=0.3, color=cmap(c),
                           label=f"{NAMES[c]} ({int((cluster==c).sum())})")
    axs[2].axhline(args.turn_thr, color="grey", ls="--")
    axs[2].set_xlabel("speed_mean (m/s) -- the within-type covariate")
    axs[2].set_ylabel("net_turn (rad)")
    axs[2].set_title("2x2 taxonomy (speed varies WITHIN each type)")
    axs[2].legend(fontsize=8, markerscale=2)
    fig.suptitle("Turn x Stop trajectory taxonomy -- thresholds on the two "
                 "bimodal features; speed is a covariate", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "taxonomy.png", dpi=140)
    plt.close(fig)
    print(f"\n[tax] -> {out}/trajectory_clusters.csv + taxonomy.png + "
          f"taxonomy_table.txt")


if __name__ == "__main__":
    main()
