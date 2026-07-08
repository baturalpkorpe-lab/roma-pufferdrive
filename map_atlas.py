"""
map_atlas.py -- Phase B of the map-regime analysis.

Clusters the per-scenario features from map_features.py (Phase A) into
behavioral regimes and produces the "map atlas":

  map_clusters.csv        scenario_id -> cluster label + assignment margin
  atlas_model_choice.png  silhouette + inertia vs K (why this K)
  atlas_centroids.png     cluster x feature table in original units (nameable)
  atlas_pca.png           PCA scatter of all maps colored by cluster
  atlas_examples.png      rendered example maps per cluster (roads + GT paths)
  atlas_names.txt         rule-based name suggestions per cluster

The clustering is scaffolding for stratified role analysis (Phase C), not a
claim in itself: validity = the centroid table reads like nameable driving
regimes + the examples look coherent + assignments are stable.

Usage:
    python map_atlas.py --in_dir /scratch/e452103/map_atlas [--k 5]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Trimmed feature set (2026-07-08): dropped the redundant / low-signal features
# (n_moving~n_vehicles, speed_mean~speed_p85, road_len/extent/road_density = size
# collinear, turn_rate = flat across real regimes / junk-only). One feature per
# conceptual axis: density / speed / flow-spread / congestion / road geometry.
FEATURES = ["n_vehicles", "speed_p85", "speed_std_across", "stop_frac", "curviness"]
LOG_FEATURES = ["n_vehicles"]

BG, ROAD = "#0f0f1a", "#55556a"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",  type=str, default="map_atlas",
                   help="Directory holding map_features.csv + geometry_cache.pkl")
    p.add_argument("--out_dir", type=str, default="",
                   help="Where to write atlas outputs (default = --in_dir). "
                        "Use a per-K subdir so k=3 and k=5 don't overwrite.")
    p.add_argument("--k",       type=int, default=0,
                   help="Number of clusters. 0 = pick best silhouette in --k_range")
    p.add_argument("--k_range", type=str, default="3,8")
    p.add_argument("--junk_turn_rate", type=float, default=0.5,
                   help="Drop scenarios with turn_rate above this BEFORE "
                        "clustering (corrupted GT headings, ~1.94 rad/s). Real "
                        "regimes sit at 0.04-0.10, so 0.5 is a safe cutoff.")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def suggest_name(centroid_raw, all_centroids_raw):
    """Rule-based human-readable name from a cluster's raw-unit centroid."""
    def tercile(key):
        vals = np.sort(all_centroids_raw[key])
        v    = centroid_raw[key]
        if v <= vals[max(0, len(vals) // 3 - 1)]:
            return 0
        if v >= vals[-max(1, len(vals) // 3)]:
            return 2
        return 1

    speed = ["slow", "mid-speed", "fast"][tercile("speed_p85")]
    curv  = ["straight", "", "curvy"][tercile("curviness")]
    dens  = ["sparse", "", "dense"][tercile("n_vehicles")]
    tags  = [speed]
    if curv:
        tags.append(curv)
    if dens:
        tags.append(dens)
    if tercile("stop_frac") == 2:
        tags.append("stop-heavy")
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

    df = pd.read_csv(in_dir / "map_features.csv")
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"[phaseB] {len(df)} scenarios ({n0 - len(df)} dropped as degenerate)")

    # Quarantine junk maps (corrupted GT headings -> implausible turn_rate) BEFORE
    # clustering, since turn_rate is no longer a clustering feature to isolate them.
    n_pre = len(df)
    df = df[df["turn_rate"] <= args.junk_turn_rate].reset_index(drop=True)
    print(f"[phaseB] removed {n_pre - len(df)} junk maps "
          f"(turn_rate > {args.junk_turn_rate} rad/s); {len(df)} remain")

    # -- Preprocess: log-transform skewed features, then z-score everything --
    X = df[FEATURES].copy()
    for f in LOG_FEATURES:
        X[f] = np.log1p(X[f].clip(lower=0))
    mu, sd = X.mean(), X.std().replace(0, 1)
    Z = ((X - mu) / sd).values

    # -- Model choice: silhouette + inertia over K range ----------------------
    k_lo, k_hi = (int(v) for v in args.k_range.split(","))
    ks   = list(range(k_lo, k_hi + 1))
    sils, inertias, models = [], [], {}
    rng  = np.random.default_rng(args.seed)
    sub  = rng.choice(len(Z), min(4000, len(Z)), replace=False)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(Z)
        sil = silhouette_score(Z[sub], km.labels_[sub])
        sils.append(sil)
        inertias.append(km.inertia_)
        models[k] = km
        print(f"[phaseB] K={k}: silhouette={sil:.3f}  inertia={km.inertia_:,.0f}")

    K  = args.k if args.k else ks[int(np.argmax(sils))]
    km = models[K]
    print(f"[phaseB] using K={K}" + ("" if args.k else " (best silhouette)"))

    labels = km.labels_
    dists  = km.transform(Z)                      # (N, K) distances to centroids
    d_sort = np.sort(dists, axis=1)
    margin = d_sort[:, 1] / np.maximum(d_sort[:, 0], 1e-9)  # >1.5 = confident

    df_out = pd.DataFrame({"scenario_id": df["scenario_id"].values,
                           "cluster": labels, "margin": margin.round(3)})
    df_out.to_csv(out_dir / "map_clusters.csv", index=False)
    core_frac = (margin > 1.5).mean()
    print(f"[phaseB] margin>1.5 (core maps usable as clean strata): {core_frac:.0%}")

    # -- Raw-unit centroid table + names --------------------------------------
    raw = df[FEATURES].groupby(labels).mean()
    all_raw = {f: raw[f].values for f in FEATURES}
    names = [suggest_name({f: raw.loc[c, f] for f in FEATURES}, all_raw)
             for c in range(K)]
    sizes = np.bincount(labels, minlength=K)

    with open(out_dir / "atlas_names.txt", "w") as f:
        for c in range(K):
            line = (f"cluster {c} (n={sizes[c]}, {sizes[c]/len(df):.0%}): "
                    f"'{names[c]}'  " +
                    "  ".join(f"{k}={raw.loc[c, k]:.2f}" for k in FEATURES))
            print("[phaseB] " + line)
            f.write(line + "\n")

    # -- Figure 1: model choice ------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(ks, sils, "o-"); a1.axvline(K, color="red", ls="--", alpha=0.5)
    a1.set_xlabel("K"); a1.set_ylabel("silhouette"); a1.set_title("silhouette vs K")
    a2.plot(ks, inertias, "o-"); a2.axvline(K, color="red", ls="--", alpha=0.5)
    a2.set_xlabel("K"); a2.set_ylabel("inertia"); a2.set_title("inertia (elbow)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "atlas_model_choice.png", dpi=140)
    plt.close(fig)

    # -- Figure 2: centroid heatmap (z-scored, annotated with raw units) ------
    cz = (np.log1p(raw[LOG_FEATURES].clip(lower=0)).join(
              raw[[f for f in FEATURES if f not in LOG_FEATURES]])
          )[FEATURES]
    cz = ((cz - mu) / sd).values
    fig, ax = plt.subplots(figsize=(1.1 * len(FEATURES), 0.9 * K + 1.5))
    im = ax.imshow(cz, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"C{c}: {names[c]}\n(n={sizes[c]})" for c in range(K)],
                       fontsize=8)
    for c in range(K):
        for j, f in enumerate(FEATURES):
            ax.text(j, c, f"{raw.loc[c, f]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(cz[c, j]) > 1 else "black")
    plt.colorbar(im, ax=ax, label="z-score (color) / raw units (text)")
    ax.set_title(f"Map-regime centroids  (K={K})")
    fig.tight_layout()
    fig.savefig(out_dir / "atlas_centroids.png", dpi=140)
    plt.close(fig)

    # -- Figure 3: PCA scatter -------------------------------------------------
    pca  = PCA(n_components=2)
    Z2   = pca.fit_transform(Z)
    show = rng.choice(len(Z), min(8000, len(Z)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for c in range(K):
        m = labels[show] == c
        ax.scatter(Z2[show][m, 0], Z2[show][m, 1], s=4, alpha=0.4,
                   color=cmap(c % 10), label=f"C{c}: {names[c]} ({sizes[c]})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.legend(fontsize=8, markerscale=3)
    ax.set_title("Map regimes in feature space (PCA)")
    fig.tight_layout()
    fig.savefig(out_dir / "atlas_pca.png", dpi=140)
    plt.close(fig)

    # -- Figure 4: example maps per cluster ------------------------------------
    try:
        with open(in_dir / "geometry_cache.pkl", "rb") as f:
            geo = pickle.load(f)
    except FileNotFoundError:
        geo = {}
    if geo:
        sid_to_idx = {s: i for i, s in enumerate(df["scenario_id"].values)}
        n_ex = 3
        fig, axes = plt.subplots(K, n_ex, figsize=(4 * n_ex, 3.6 * K))
        axes = np.atleast_2d(axes)
        fig.patch.set_facecolor(BG)
        for c in range(K):
            cands = [(dists[sid_to_idx[s], c], s) for s in geo
                     if s in sid_to_idx and labels[sid_to_idx[s]] == c]
            cands.sort()
            for j in range(n_ex):
                ax = axes[c, j]
                ax.set_facecolor(BG)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#333344")
                if j >= len(cands):
                    ax.set_visible(False)
                    continue
                sid = cands[j][1]
                g   = geo[sid]
                if g["polys"]:
                    ax.add_collection(LineCollection(
                        g["polys"], colors=ROAD, linewidths=0.7))
                for px, py, spd in g["paths"]:
                    ax.plot(px, py, lw=1.4, alpha=0.9,
                            color=plt.get_cmap("coolwarm")(min(spd / 25, 1.0)))
                ax.autoscale()
                ax.set_aspect("equal")
                if j == 0:
                    ax.set_ylabel(f"C{c}: {names[c]}", color="#e8e8f0",
                                  fontsize=9)
        fig.suptitle("Example maps per regime (roads gray, GT paths colored "
                     "by speed: blue=slow red=fast)", color="#e8e8f0")
        fig.tight_layout()
        fig.savefig(out_dir / "atlas_examples.png", dpi=130, facecolor=BG)
        plt.close(fig)

    print(f"\n[phaseB] atlas written to {out_dir}/atlas_*.png + map_clusters.csv")


if __name__ == "__main__":
    main()
