"""
traj_stopfrac_view.py -- is the old slow/mid trajectory continuum divisible by
stop_frac?

Recomputes the OLD trajectory PCA (5 features: distance, speed_mean, speed_max,
speed_min, net_turn -- NO stop_frac, same preprocessing as trajectory_atlas)
and plots stop_frac as a THIRD axis on top of the old PC1/PC2 layout. If
stop_frac is a real bimodal separator, the low-speed end (where slow & mid
overlapped in 2-D) lifts into a separate high-stop_frac sheet.

Panels:
  - 3-D scatter PC1 x PC2 x stop_frac (two view angles)
  - 2-D PC1 vs stop_frac (PC1 is ~the speed axis; shows the split directly)

Color: old cluster labels if --clusters_csv given (see whether the old slow/mid
clusters separate on the new axis), else stop_frac itself.

No GPU. Needs a trajectory_features.csv that HAS a stop_frac column (i.e. the
re-featurized one).

Usage:
    python traj_stopfrac_view.py \
        --features_csv /scratch/e452103/traj_atlas/trajectory_features.csv \
        --clusters_csv /scratch/e452103/traj_atlas/k4/trajectory_clusters.csv \
        --out /scratch/e452103/traj_atlas/stopfrac_view.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

OLD_FEATURES = ["distance", "speed_mean", "speed_max", "speed_min", "net_turn"]
LOG_FEATURES = ["distance", "net_turn"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_csv", type=str, required=True,
                   help="trajectory_features.csv WITH a stop_frac column")
    p.add_argument("--clusters_csv", type=str, default="",
                   help="old trajectory_clusters.csv for coloring (optional)")
    p.add_argument("--out",          type=str, required=True)
    p.add_argument("--max_points",   type=int, default=12000)
    return p.parse_args()


def main():
    args = parse_args()
    import pandas as pd
    df = pd.read_csv(args.features_csv)
    if "stop_frac" not in df.columns:
        raise SystemExit("features_csv has no stop_frac column -- re-run "
                         "trajectory_features.py first")
    df = df.dropna(subset=OLD_FEATURES + ["stop_frac"]).reset_index(drop=True)

    # OLD PCA: exactly the trajectory_atlas preprocessing on the 5 old features
    X = df[OLD_FEATURES].copy()
    for f in LOG_FEATURES:
        X[f] = np.log1p(X[f].clip(lower=0))
    Z = ((X - X.mean()) / X.std().replace(0, 1)).values
    _, svals, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    evr = (svals ** 2) / (svals ** 2).sum()
    pc1 = Z @ vt[0]
    pc2 = Z @ vt[1]
    sf  = df["stop_frac"].values

    # coloring
    labels = names = None
    if args.clusters_csv:
        tc = pd.read_csv(args.clusters_csv).rename(
            columns={"scenario_id": "sid", "vehicle_id": "vid"})
        key = df[["scenario_id", "vehicle_id"]].rename(
            columns={"scenario_id": "sid", "vehicle_id": "vid"})
        merged = key.merge(tc[["sid", "vid", "cluster"]], on=["sid", "vid"],
                           how="left")
        labels = merged["cluster"].values
        print(f"[view] joined old clusters: "
              f"{np.isfinite(pd.to_numeric(labels, errors='coerce')).sum()} "
              f"matched")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(pc1), min(args.max_points, len(pc1)), replace=False)
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(18, 6))
    # two 3-D views
    for k, (az, el) in enumerate([(-60, 18), (-120, 12)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        if labels is not None and np.isfinite(
                pd.to_numeric(labels, errors="coerce")).any():
            for c in sorted(set(int(v) for v in labels[idx]
                                if np.isfinite(v))):
                m = labels[idx] == c
                ax.scatter(pc1[idx][m], pc2[idx][m], sf[idx][m], s=4, alpha=0.4,
                           color=cmap(c % 10), label=f"C{c}")
        else:
            ax.scatter(pc1[idx], pc2[idx], sf[idx], s=4, alpha=0.4,
                       c=sf[idx], cmap="viridis")
        ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
        ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
        ax.set_zlabel("stop_frac")
        ax.view_init(elev=el, azim=az)
        if k == 0 and labels is not None:
            ax.legend(fontsize=8, markerscale=2, loc="upper left")
        ax.set_title(f"old PC1 x PC2 x stop_frac  (view {k+1})", fontsize=10)

    # 2-D PC1 vs stop_frac -- the direct "does the speed axis split on stops"
    ax = fig.add_subplot(1, 3, 3)
    if labels is not None and np.isfinite(
            pd.to_numeric(labels, errors="coerce")).any():
        for c in sorted(set(int(v) for v in labels[idx] if np.isfinite(v))):
            m = labels[idx] == c
            ax.scatter(pc1[idx][m], sf[idx][m], s=5, alpha=0.35,
                       color=cmap(c % 10), label=f"C{c}")
        ax.legend(fontsize=8, markerscale=2)
    else:
        ax.scatter(pc1[idx], sf[idx], s=5, alpha=0.35, c=sf[idx],
                   cmap="viridis")
    ax.set_xlabel(f"old PC1 ({evr[0]:.0%})  ~ speed axis")
    ax.set_ylabel("stop_frac")
    ax.set_title("PC1 vs stop_frac", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Is the old trajectory continuum divisible by stop_frac?",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    # quick numeric read: stop_frac at the slow end of PC1
    lo = pc1 < np.quantile(pc1, 0.33)
    print(f"[view] stop_frac  slow-PC1 third: mean={sf[lo].mean():.2f}  "
          f"frac>0.3={np.mean(sf[lo] > 0.3):.0%}   |   "
          f"fast two-thirds: mean={sf[~lo].mean():.2f}")
    print(f"[view] -> {out}")


if __name__ == "__main__":
    main()
