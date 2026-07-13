"""
trajectory_atlas.py -- Phase B of the trajectory-level stratification.

Clusters the per-(scenario, vehicle) GT trajectory features from
trajectory_features.py into TASK types (the new strata for role analysis):
the fast cruiser on a mid-speed map lands with the highway cruisers; the
exiting/turning seat on a highway map lands with the turners.

Outputs in --out_dir (default = --in_dir):
  trajectory_clusters.csv   scenario_id, vehicle_id -> cluster + margin
  tatlas_model_choice.png   silhouette + inertia vs K
  tatlas_centroids.png      cluster x feature table (raw units + z color)
  tatlas_pca.png            PCA scatter colored by cluster
  tatlas_examples.png       REAL example trajectories per cluster
  tatlas_names.txt          rule-based name suggestions

Clustering here is SCAFFOLDING for stratified analysis, not a natural-kinds
claim (trajectory space is likely a continuum; K-means bins it -- fine for
strata, same status as the map atlas).

Usage:
    python trajectory_atlas.py --in_dir /scratch/e452103/traj_atlas [--k 4]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES     = ["distance", "speed_mean", "speed_max", "speed_min", "net_turn"]
LOG_FEATURES = ["distance", "net_turn"]      # right-skewed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",  type=str, default="traj_atlas",
                   help="Dir with trajectory_features.csv + trajectory_cache.pkl")
    p.add_argument("--out_dir", type=str, default="",
                   help="Output dir (default = --in_dir); use a per-K subdir "
                        "to compare Ks without overwriting")
    p.add_argument("--k",       type=int, default=0,
                   help="0 = best silhouette in --k_range")
    p.add_argument("--k_range", type=str, default="3,8")
    p.add_argument("--edge_margin", type=float, default=1.2,
                   help="margin (d2/d1) below this = an 'edge' trajectory "
                        "sitting between two types (1.0 = exactly on the "
                        "boundary; core trajectories are >1.5)")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def suggest_name(c, allc):
    """Rule-based name from a cluster's raw-unit centroid."""
    def tercile(key):
        vals = np.sort(allc[key])
        v = c[key]
        if v <= vals[max(0, len(vals) // 3 - 1)]:
            return 0
        if v >= vals[-max(1, len(vals) // 3)]:
            return 2
        return 1

    speed = ["slow", "mid-speed", "fast"][tercile("speed_mean")]
    tags = [speed]
    if tercile("net_turn") == 2:
        tags.append("turning")
    elif tercile("net_turn") == 0:
        tags.append("straight")
    if tercile("speed_min") == 0 and tercile("speed_max") >= 1:
        tags.append("stop&go")
    if tercile("distance") == 2:
        tags.append("long")
    return " ".join(tags)


def main():
    args    = parse_args()
    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    df = pd.read_csv(in_dir / "trajectory_features.csv")
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    df = df.reset_index(drop=True)
    print(f"[trajB] {len(df)} trajectories ({n0 - len(df)} dropped)")

    X = df[FEATURES].copy()
    for f in LOG_FEATURES:
        X[f] = np.log1p(X[f].clip(lower=0))
    mu, sd = X.mean(), X.std().replace(0, 1)
    Z = ((X - mu) / sd).values

    # -- model choice ----------------------------------------------------------
    k_lo, k_hi = (int(v) for v in args.k_range.split(","))
    ks = list(range(k_lo, k_hi + 1))
    rng = np.random.default_rng(args.seed)
    sub = rng.choice(len(Z), min(4000, len(Z)), replace=False)
    sils, inertias, models = [], [], {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(Z)
        sil = silhouette_score(Z[sub], km.labels_[sub])
        sils.append(sil)
        inertias.append(km.inertia_)
        models[k] = km
        print(f"[trajB] K={k}: silhouette={sil:.3f}  inertia={km.inertia_:,.0f}")
    K  = args.k if args.k else ks[int(np.argmax(sils))]
    km = models[K]
    print(f"[trajB] using K={K}" + ("" if args.k else " (best silhouette)"))

    labels = km.labels_
    dists  = km.transform(Z)
    order  = np.argsort(dists, axis=1)         # cluster ids, nearest-first
    d_sort = np.take_along_axis(dists, order, axis=1)
    margin = d_sort[:, 1] / np.maximum(d_sort[:, 0], 1e-9)
    cluster2 = order[:, 1]                       # 2nd-nearest cluster
    is_edge  = margin < args.edge_margin         # ambiguous / between two types

    pd.DataFrame({"scenario_id": df["scenario_id"].values,
                  "vehicle_id":  df["vehicle_id"].values,
                  "cluster": labels, "margin": margin.round(3),
                  "cluster2": cluster2, "is_edge": is_edge.astype(int)}
                 ).to_csv(out_dir / "trajectory_clusters.csv", index=False)
    print(f"[trajB] margin>1.5 (core trajectories): {(margin > 1.5).mean():.0%}")
    print(f"[trajB] edge trajectories (margin<{args.edge_margin}): "
          f"{is_edge.mean():.0%} ({int(is_edge.sum())})")
    # which two types each edge sits between (unordered pair)
    from collections import Counter
    pair_ct = Counter(tuple(sorted((int(a), int(b))))
                      for a, b in zip(labels[is_edge], cluster2[is_edge]))
    for (a, b), n in sorted(pair_ct.items(), key=lambda kv: -kv[1]):
        print(f"[trajB]   edge C{a}<->C{b}: {n}  "
              f"({n/max(int(is_edge.sum()),1):.0%} of edges)")

    # -- centroid table + names ------------------------------------------------
    raw = df[FEATURES].groupby(labels).mean()
    allc = {f: raw[f].values for f in FEATURES}
    names = [suggest_name({f: raw.loc[c, f] for f in FEATURES}, allc)
             for c in range(K)]
    sizes = np.bincount(labels, minlength=K)
    with open(out_dir / "tatlas_names.txt", "w") as f:
        for c in range(K):
            line = (f"cluster {c} (n={sizes[c]}, {sizes[c]/len(df):.0%}): "
                    f"'{names[c]}'  " +
                    "  ".join(f"{k}={raw.loc[c, k]:.2f}" for k in FEATURES))
            print("[trajB] " + line)
            f.write(line + "\n")

    # -- fig 1: model choice ---------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(ks, sils, "o-"); a1.axvline(K, color="red", ls="--", alpha=0.5)
    a1.set_xlabel("K"); a1.set_ylabel("silhouette")
    a2.plot(ks, inertias, "o-"); a2.axvline(K, color="red", ls="--", alpha=0.5)
    a2.set_xlabel("K"); a2.set_ylabel("inertia")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tatlas_model_choice.png", dpi=140)
    plt.close(fig)

    # -- fig 2: centroid heatmap ------------------------------------------------
    cz = raw[FEATURES].copy()
    for f in LOG_FEATURES:
        cz[f] = np.log1p(cz[f].clip(lower=0))
    cz = ((cz - mu) / sd).values
    fig, ax = plt.subplots(figsize=(1.4 * len(FEATURES) + 2, 0.9 * K + 1.5))
    im = ax.imshow(cz, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"C{c}: {names[c]}\n(n={sizes[c]})" for c in range(K)],
                       fontsize=8)
    for c in range(K):
        for j, f in enumerate(FEATURES):
            ax.text(j, c, f"{raw.loc[c, f]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(cz[c, j]) > 1 else "black")
    plt.colorbar(im, ax=ax, label="z-score (color) / raw units (text)")
    ax.set_title(f"Trajectory-type centroids (K={K})")
    fig.tight_layout()
    fig.savefig(out_dir / "tatlas_centroids.png", dpi=140)
    plt.close(fig)

    # -- fig 3: PCA scatter ------------------------------------------------------
    pca = PCA(n_components=2)
    Z2  = pca.fit_transform(Z)
    show = rng.choice(len(Z), min(8000, len(Z)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    core_show = show[~is_edge[show]]
    for c in range(K):
        m = labels[core_show] == c
        ax.scatter(Z2[core_show][m, 0], Z2[core_show][m, 1], s=4, alpha=0.4,
                   color=cmap(c % 10), label=f"C{c}: {names[c]} ({sizes[c]})")
    # edge trajectories (between two types) overlaid in black
    em = show[is_edge[show]]
    ax.scatter(Z2[em, 0], Z2[em, 1], s=6, alpha=0.5, color="black",
               label=f"edge (margin<{args.edge_margin}, {int(is_edge.sum())})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.legend(fontsize=8, markerscale=3)
    ax.set_title("Trajectory types in feature space (PCA) — edges in black")
    fig.tight_layout()
    fig.savefig(out_dir / "tatlas_pca.png", dpi=140)
    plt.close(fig)

    # -- fig 4: REAL example trajectories per cluster ---------------------------
    try:
        with open(in_dir / "trajectory_cache.pkl", "rb") as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        cache = {}
    if cache:
        key_to_idx = {(s, v): i for i, (s, v) in
                      enumerate(zip(df["scenario_id"].values,
                                    df["vehicle_id"].values))}
        n_ex = 6
        fig, axes = plt.subplots(K, n_ex, figsize=(2.6 * n_ex, 2.6 * K))
        axes = np.atleast_2d(axes)
        BG = "#0f0f1a"
        fig.patch.set_facecolor(BG)
        spd_cmap = plt.get_cmap("coolwarm")
        for c in range(K):
            cands = [(dists[key_to_idx[k], c], k) for k in cache
                     if k in key_to_idx and labels[key_to_idx[k]] == c]
            cands.sort(key=lambda t: t[0])
            for j in range(n_ex):
                ax = axes[c, j]
                ax.set_facecolor(BG)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#333344")
                if j >= len(cands):
                    ax.set_visible(False)
                    continue
                k = cands[j][1]
                g = cache[k]
                ax.plot(g["x"] - g["x"][0], g["y"] - g["y"][0], lw=1.8,
                        color=spd_cmap(min(g["spd"] / 25.0, 1.0)))
                ax.scatter([0], [0], color="white", s=18, zorder=5)
                ax.set_aspect("equal")
                if j == 0:
                    ax.set_ylabel(f"C{c}: {names[c]}", color="#e8e8f0",
                                  fontsize=8)
        fig.suptitle("Example GT trajectories per type (start at dot; color = "
                     "mean speed, blue slow / red fast; axes = meters from "
                     "start)", color="#e8e8f0", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "tatlas_examples.png", dpi=130, facecolor=BG)
        plt.close(fig)

    print(f"\n[trajB] atlas -> {out_dir}/tatlas_*.png + trajectory_clusters.csv")


if __name__ == "__main__":
    main()
