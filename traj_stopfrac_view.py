"""
traj_stopfrac_view.py -- is the trajectory speed-continuum divisible by a
candidate 3rd feature (stop_frac, or a MAP-based feature)?

Recomputes a trajectory PCA (--pca_features, same preprocessing as
trajectory_atlas) and plots a chosen feature (--z_col) as a THIRD axis over
PC1/PC2. If that feature is a real separator ORTHOGONAL to speed, it splits
trajectories that share the same PC1 (same speed) into distinct bands.

--z_col can be:
  a per-trajectory feature (stop_frac, net_turn, ...), or
  a per-SCENE map feature joined via --map_features_csv, referenced as
  m_<name> (e.g. m_n_moving = scene density, m_curviness, m_speed_std_across).
  A scene feature is broadcast to every trajectory in that scene, so a vertical
  band at fixed PC1 = "same speed, different map context" -- exactly the
  cross-speed separator we're hunting for.

Panels: two 3-D views (PC1 x PC2 x z) + a 2-D PC1 vs z (the direct read).
Color: cluster labels if --clusters_csv given, else z itself.

No GPU. Usage:
    # new clustering, stop_frac axis
    python traj_stopfrac_view.py \
        --features_csv /scratch/e452103/traj_atlas/trajectory_features.csv \
        --clusters_csv /scratch/e452103/traj_atlas/trajectory_clusters.csv \
        --pca_features distance,speed_mean,speed_max,speed_min,net_turn,stop_frac \
        --z_col stop_frac --out .../stopfrac_view_new.png

    # test scene DENSITY as a cross-speed separator
    python traj_stopfrac_view.py \
        --features_csv /scratch/e452103/traj_atlas/trajectory_features.csv \
        --map_features_csv /scratch/e452103/map_atlas/map_features.csv \
        --z_col m_n_moving --out .../density_view.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

LOGGABLE = {"distance", "net_turn", "m_road_len", "m_extent", "m_road_density",
            "m_n_vehicles", "m_n_moving"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_csv", type=str, required=True)
    p.add_argument("--clusters_csv", type=str, default="")
    p.add_argument("--map_features_csv", type=str, default="",
                   help="map_features.csv (per scenario) -> join scene features "
                        "as m_<name>")
    p.add_argument("--pca_features", type=str,
                   default="distance,speed_mean,speed_max,speed_min,net_turn",
                   help="features defining PC1/PC2 (default = old 5)")
    p.add_argument("--z_col",        type=str, default="stop_frac",
                   help="3rd-axis feature (per-traj col or m_<map feature>)")
    p.add_argument("--out",          type=str, required=True)
    p.add_argument("--max_points",   type=int, default=12000)
    return p.parse_args()


def main():
    args = parse_args()
    import pandas as pd
    df = pd.read_csv(args.features_csv)

    if args.map_features_csv:
        mf = pd.read_csv(args.map_features_csv)
        mcols = [c for c in mf.columns if c != "scenario_id"]
        mf = mf.rename(columns={c: f"m_{c}" for c in mcols})
        df = df.merge(mf, on="scenario_id", how="left")
        print(f"[view] joined {len(mcols)} map features as m_* "
              f"({[f'm_{c}' for c in mcols]})")

    feats = [f.strip() for f in args.pca_features.split(",")]
    miss  = [f for f in feats + [args.z_col] if f not in df.columns]
    if miss:
        raise SystemExit(f"missing columns {miss}; have {list(df.columns)}")
    df = df.dropna(subset=feats + [args.z_col]).reset_index(drop=True)

    # PCA (trajectory_atlas preprocessing: log the skewed ones, z-score)
    X = df[feats].copy()
    for f in feats:
        if f in LOGGABLE:
            X[f] = np.log1p(X[f].clip(lower=0))
    Z = ((X - X.mean()) / X.std().replace(0, 1)).values
    _, svals, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    evr = (svals ** 2) / (svals ** 2).sum()
    pc1, pc2 = Z @ vt[0], Z @ vt[1]
    # orient PC1 so + = faster (interpretable)
    if np.corrcoef(pc1, df["speed_mean"].values)[0, 1] < 0:
        pc1 = -pc1
    z = df[args.z_col].values.astype(float)

    labels = None
    if args.clusters_csv:
        tc = pd.read_csv(args.clusters_csv).rename(
            columns={"scenario_id": "sid", "vehicle_id": "vid"})
        key = df[["scenario_id", "vehicle_id"]].rename(
            columns={"scenario_id": "sid", "vehicle_id": "vid"})
        labels = key.merge(tc[["sid", "vid", "cluster"]], on=["sid", "vid"],
                           how="left")["cluster"].values

    rng = np.random.default_rng(0)
    idx = rng.choice(len(pc1), min(args.max_points, len(pc1)), replace=False)
    cmap = plt.get_cmap("tab10")

    def color_scatter(ax, xs, ys, zs=None):
        if labels is not None and np.isfinite(
                pd.to_numeric(labels, errors="coerce")).any():
            for c in sorted(set(int(v) for v in labels[idx] if np.isfinite(v))):
                m = labels[idx] == c
                a = (xs[m], ys[m]) if zs is None else (xs[m], ys[m], zs[m])
                ax.scatter(*a, s=4, alpha=0.4, color=cmap(c % 10), label=f"C{c}")
            return True
        a = (xs, ys) if zs is None else (xs, ys, zs)
        ax.scatter(*a, s=4, alpha=0.4, c=z[idx], cmap="viridis")
        return False

    fig = plt.figure(figsize=(18, 6))
    for k, (az, el) in enumerate([(-60, 18), (-120, 12)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        has_lab = color_scatter(ax, pc1[idx], pc2[idx], z[idx])
        ax.set_xlabel(f"PC1 ({evr[0]:.0%})"); ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
        ax.set_zlabel(args.z_col)
        ax.view_init(elev=el, azim=az)
        if k == 0 and has_lab:
            ax.legend(fontsize=8, markerscale=2, loc="upper left")
        ax.set_title(f"PC1 x PC2 x {args.z_col} (view {k+1})", fontsize=10)

    ax = fig.add_subplot(1, 3, 3)
    color_scatter(ax, pc1[idx], z[idx])
    ax.set_xlabel(f"PC1 ({evr[0]:.0%})  ~ speed axis")
    ax.set_ylabel(args.z_col)
    ax.set_title(f"PC1 vs {args.z_col}", fontsize=10)
    ax.grid(alpha=0.3)
    if labels is not None and np.isfinite(
            pd.to_numeric(labels, errors="coerce")).any():
        ax.legend(fontsize=8, markerscale=2)

    fig.suptitle(f"Is the trajectory continuum divisible by {args.z_col}?",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)

    # is z orthogonal to speed? corr with PC1 near 0 = a cross-speed separator
    r = float(np.corrcoef(pc1, z)[0, 1])
    mid = (pc1 > np.quantile(pc1, 0.4)) & (pc1 < np.quantile(pc1, 0.6))
    print(f"[view] {args.z_col}: corr with PC1(speed) = {r:+.2f} "
          f"({'cross-speed separator' if abs(r) < 0.3 else 'speed-correlated'})"
          f"; at MID speed: mean={z[mid].mean():.2f} std={z[mid].std():.2f}")
    print(f"[view] -> {out}")


if __name__ == "__main__":
    main()
