"""
traj_cluster_view.py -- view an EXISTING trajectory clustering in raw feature
space (not PCA), and dump many example trajectories from one cluster.

Reads a FROZEN clustering (trajectory_clusters.csv from trajectory_atlas.py)
and joins it back to trajectory_features.csv. It never re-runs K-means, so the
colors and cluster sizes match the tatlas_pca.png you already have exactly --
this only re-draws the same labels on different axes.

Two figures:

  cluster_feature_space.png
      A scatter-matrix of the RAW features (original units, no PCA) coloured by
      cluster, edges (is_edge==1) in black -- the same legend/colors as the PCA
      figure. The diagonal holds each feature's per-cluster histograms, so you
      see where the clusters actually separate (e.g. net_turn) and where they
      overlap (e.g. speed_mean once mid-speed and turning have merged).

  cluster<K>_examples.png
      A grid of N real GT trajectories from ONE cluster (needs
      trajectory_cache.pkl), start at the white dot, coloured by mean speed
      (blue slow -> red fast), axes in metres from start. Same rendering as
      tatlas_examples.png but for a single cluster and any N.

No GPU, no env, pure pandas/matplotlib -- runs on a login node in seconds.

Usage:
    python traj_cluster_view.py \
        --in_dir       /scratch/$USER/traj_atlas \
        --clusters_csv /scratch/$USER/traj_atlas/k3/trajectory_clusters.csv \
        --out_dir      /scratch/$USER/traj_atlas/k3 \
        --cluster 1 --n_examples 20
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same feature set + skew handling as trajectory_atlas.py.
ALL_FEATURES = ["distance", "speed_mean", "speed_max", "speed_min",
                "net_turn", "stop_frac"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, required=True,
                   help="Dir with trajectory_features.csv + trajectory_cache.pkl")
    p.add_argument("--clusters_csv", type=str, required=True,
                   help="Frozen trajectory_clusters.csv from trajectory_atlas.py "
                        "(the K you want to view, e.g. k3/)")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--features", type=str, default="speed_mean,net_turn,stop_frac,distance",
                   help="Raw features for the scatter-matrix axes (comma list). "
                        "Default = the interpretable 4. Use all 6 for the full "
                        "matrix.")
    p.add_argument("--cluster", type=int, default=1,
                   help="Which cluster to dump examples from (1 = orange in the "
                        "tab10 palette, matching the PCA figure)")
    p.add_argument("--n_examples", type=int, default=20)
    p.add_argument("--ex_cols", type=int, default=5,
                   help="Columns in the example grid (rows derived from N)")
    p.add_argument("--pick", type=str, default="core",
                   choices=["core", "random"],
                   help="core = the most UNAMBIGUOUS members (highest margin, "
                        "edges excluded) = most representative; random = a fair "
                        "sample incl. borderline ones")
    p.add_argument("--max_points", type=int, default=8000,
                   help="Points plotted in the scatter-matrix (stratified per "
                        "cluster so small clusters stay visible)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def short_name(centroid, all_centroids):
    """Rule-based cluster name from its raw-unit centroid (terciles across
    clusters), matching trajectory_atlas.suggest_name's spirit."""
    def tercile(key):
        vals = np.sort(all_centroids[key])
        v = centroid[key]
        if v <= vals[max(0, len(vals) // 3 - 1)]:
            return 0
        if v >= vals[-max(1, len(vals) // 3)]:
            return 2
        return 1
    tags = []
    if "speed_mean" in all_centroids:
        tags.append(["slow", "mid-speed", "fast"][tercile("speed_mean")])
    if "net_turn" in all_centroids:
        if tercile("net_turn") == 2:
            tags.append("turning")
        elif tercile("net_turn") == 0:
            tags.append("straight")
    if "stop_frac" in all_centroids and tercile("stop_frac") == 2:
        tags.append("stop&go")
    if "distance" in all_centroids and tercile("distance") == 2:
        tags.append("long")
    return " ".join(tags) or "cluster"


def main():
    args = parse_args()
    import pandas as pd

    in_dir = Path(args.in_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    feats = [f.strip() for f in args.features.split(",")]

    # --- Join frozen labels back to the features ---------------------------
    cl = pd.read_csv(args.clusters_csv)
    fe = pd.read_csv(in_dir / "trajectory_features.csv")
    need = {"scenario_id", "vehicle_id", "cluster"}
    if not need.issubset(cl.columns):
        raise SystemExit(f"{args.clusters_csv} missing {need - set(cl.columns)}")
    df = cl.merge(fe, on=["scenario_id", "vehicle_id"], how="left",
                  suffixes=("", "_f"))
    missing = [f for f in feats + ALL_FEATURES if f not in df.columns]
    if set(feats) & set(missing):
        raise SystemExit(f"features {set(feats)&set(missing)} not in the "
                         f"features CSV (have {list(fe.columns)})")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats)
    if "is_edge" not in df.columns:
        df["is_edge"] = 0

    Ks = sorted(df["cluster"].unique())
    cmap = plt.get_cmap("tab10")
    # Per-cluster centroids (raw units) for naming, over whatever features exist.
    name_feats = [f for f in ALL_FEATURES if f in df.columns]
    cents = df.groupby("cluster")[name_feats].mean()
    allc = {f: cents[f].values for f in name_feats}
    names = {int(c): short_name({f: cents.loc[c, f] for f in name_feats}, allc)
             for c in Ks}
    sizes = df["cluster"].value_counts().to_dict()
    print("[view] clusters:", {int(c): f"{names[int(c)]} ({sizes[int(c)]})"
                               for c in Ks})

    # --- Fig 1: raw-feature scatter matrix ---------------------------------
    core = df[df["is_edge"] == 0]
    edge = df[df["is_edge"] == 1]
    # Stratified downsample per cluster so green (small) is not swamped.
    per = max(1, args.max_points // max(len(Ks), 1))
    show_idx = []
    for c in Ks:
        idx = core.index[core["cluster"] == c].to_numpy()
        show_idx.append(rng.choice(idx, min(per, len(idx)), replace=False))
    show = core.loc[np.concatenate(show_idx)]
    e_show = edge.loc[rng.choice(edge.index.to_numpy(),
                                 min(per, len(edge)), replace=False)] \
             if len(edge) else edge

    n = len(feats)
    fig, axs = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n), squeeze=False)
    for i, fi in enumerate(feats):
        for j, fj in enumerate(feats):
            ax = axs[i][j]
            if i == j:
                # diagonal: per-cluster histogram of this feature
                lo, hi = df[fi].quantile([0.01, 0.99])
                bins = np.linspace(lo, hi, 40)
                for c in Ks:
                    v = df[df["cluster"] == c][fi]
                    ax.hist(v, bins=bins, color=cmap(int(c) % 10), alpha=0.5,
                            density=True)
                ax.set_yticks([])
            else:
                for c in Ks:
                    m = show["cluster"] == c
                    ax.scatter(show[m][fj], show[m][fi], s=3, alpha=0.35,
                               color=cmap(int(c) % 10), linewidths=0)
                if len(e_show):
                    ax.scatter(e_show[fj], e_show[fi], s=4, alpha=0.4,
                               color="black", linewidths=0)
            if i == n - 1:
                ax.set_xlabel(fj, fontsize=9)
            if j == 0:
                ax.set_ylabel(fi, fontsize=9)
            ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(int(c) % 10),
                          label=f"C{int(c)}: {names[int(c)]} ({sizes[int(c)]})")
               for c in Ks]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="black",
                              label=f"edge ({len(edge)})"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, markerscale=1.6, frameon=False)
    fig.suptitle("Trajectory clusters in raw feature space (no PCA) — "
                 "diagonal = per-cluster histograms", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(out / "cluster_feature_space.png", dpi=140)
    plt.close(fig)
    print(f"[view] wrote cluster_feature_space.png ({n}x{n} on {feats})")

    # --- Fig 2: N examples from one cluster --------------------------------
    try:
        with open(in_dir / "trajectory_cache.pkl", "rb") as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        print(f"[view] no trajectory_cache.pkl in {in_dir} -- skipping examples")
        return

    C = args.cluster
    sub = df[df["cluster"] == C].copy()
    if args.pick == "core" and "margin" in sub.columns:
        sub = sub[sub["is_edge"] == 0].sort_values("margin", ascending=False)
    else:
        sub = sub.sample(frac=1.0, random_state=args.seed)

    # Match rows to cache keys (scenario_id, vehicle_id).
    picked = []
    for _, r in sub.iterrows():
        key = (r["scenario_id"], int(r["vehicle_id"]))
        if key in cache:
            picked.append((key, cache[key]))
        if len(picked) >= args.n_examples:
            break
    if not picked:
        # vehicle_id may be stored as str in the cache keys; retry loosely.
        idset = {(str(s), str(v)): (s, v) for (s, v) in cache}
        for _, r in sub.iterrows():
            k2 = (str(r["scenario_id"]), str(int(r["vehicle_id"])))
            if k2 in idset:
                picked.append((idset[k2], cache[idset[k2]]))
            if len(picked) >= args.n_examples:
                break
    if not picked:
        print(f"[view] no cache entries matched cluster {C} "
              f"(cache has {len(cache)} keys) -- skipping examples")
        return

    ncol = args.ex_cols
    nrow = int(np.ceil(len(picked) / ncol))
    BG = "#0f0f1a"
    spd_cmap = plt.get_cmap("coolwarm")
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow),
                             squeeze=False)
    fig.patch.set_facecolor(BG)
    for a in range(nrow * ncol):
        ax = axes[a // ncol][a % ncol]
        ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#333344")
        if a >= len(picked):
            ax.set_visible(False)
            continue
        g = picked[a][1]
        x = np.asarray(g["x"]); y = np.asarray(g["y"])
        spd = float(np.mean(g["spd"])) if "spd" in g else 0.0
        ax.plot(x - x[0], y - y[0], lw=1.8,
                color=spd_cmap(min(spd / 25.0, 1.0)))
        ax.scatter([0], [0], color="white", s=16, zorder=5)
        ax.set_aspect("equal")
    fig.suptitle(f"C{C}: {names.get(C, '')} — {len(picked)} example GT "
                 f"trajectories ({args.pick})\n"
                 f"start at dot; color = mean speed (blue slow / red fast); "
                 f"axes = metres from start",
                 color="#e8e8f0", fontsize=11)
    fig.tight_layout()
    out_ex = out / f"cluster{C}_examples.png"
    fig.savefig(out_ex, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"[view] wrote {out_ex.name} ({len(picked)} trajectories)")


if __name__ == "__main__":
    main()
