"""
cross_cluster_contrast.py -- THE payoff table + figure: "role encodes whatever
the task leaves free."

For each per-cluster analysis (the cluster<K>_only filtered folders produced by
cluster_post.sbatch), it reads:
  * role_scene_consistency.csv  (ALL stratum) -> per (axis, metric):
        consistency (per-scene sign-match rate) + paired_t (mean effect)
  * role_paired_warmup.csv + role_paired_axes.csv -> the natural PC1/PC2 role
    distribution -> Sarle's bimodality coefficient BC (is the role a smooth
    continuum, BC<~0.55, or a collapsed binary switch, BC->1?)

and emits a single comparison across clusters:
  cross_cluster_contrast.csv    long table: cluster x axis x metric x {consistency, paired_t, pop_mean_delta}
  cross_cluster_bimodality.csv  cluster x {PC1_BC, PC2_BC, n}
  cross_cluster_contrast.png    heatmap of PC1 (and PC2) per-scene consistency,
                                clusters x metrics -- the money figure: each
                                cluster's role lights up a DIFFERENT metric.
  + a printed headline: for each cluster, which metric its PC1 controls
    (highest consistency among metrics with a real monotone effect |t|>=T_MIN).

No GPU, no env. Reads the checkpoint's frozen analysis CSVs only, so it is safe
to run while (re)training continues -- point it at the 2B-frozen outputs.

Usage:
  python cross_cluster_contrast.py \
      --root     /scratch/e452103/role_paired \
      --clusters "0:fast,1:stopgo,2:mid" \
      --suffix   _only \
      --out_dir  /scratch/e452103/role_paired/cross_cluster
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["speed_mean", "accel_pos", "decel_abs", "jerk_abs", "turn_abs",
           "event_rate"]
T_MIN = 3.0   # |paired_t| below this = no real monotone effect -> ignore consistency


def sarle_bc(x):
    """Sarle's bimodality coefficient (SAS sample-corrected). BC in (0,1];
    ~0.555 = uniform; >0.555 hints bimodal; ->1 = two spikes / collapsed switch.
    <0.4 = clean unimodal continuum."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return np.nan
    m = x.mean()
    d = x - m
    m2 = (d ** 2).mean()
    if m2 <= 0:
        return np.nan
    g1 = (d ** 3).mean() / m2 ** 1.5              # skewness
    g2 = (d ** 4).mean() / m2 ** 2 - 3.0          # excess kurtosis
    G1 = g1 * np.sqrt(n * (n - 1)) / (n - 2)
    G2 = ((n - 1) * ((n + 1) * g2 + 6)) / ((n - 2) * (n - 3))
    denom = G2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((G1 ** 2 + 1.0) / denom) if denom else np.nan


def load_axes(axes_csv, role_dim):
    df = pd.read_csv(axes_csv)
    cols = [f"c{i}" for i in range(role_dim)]
    axes = {}
    for _, r in df[df["name"] != "mu"].iterrows():
        axes[str(r["name"])] = r[cols].values.astype(np.float64)
    return axes


def cluster_bimodality(folder):
    """Project the natural warmup roles onto PC1/PC2 -> BC for each."""
    warm = folder / "role_paired_warmup.csv"
    axf  = folder / "role_paired_axes.csv"
    if not (warm.exists() and axf.exists()):
        return {}
    df = pd.read_csv(warm)
    role_cols = sorted([c for c in df.columns
                        if c.startswith("role_") and c[5:].isdigit()],
                       key=lambda c: int(c[5:]))
    if not role_cols:
        return {}
    R = df[role_cols].values.astype(np.float64)
    axes = load_axes(axf, len(role_cols))
    out = {"n": len(df)}
    for nm in ("PC1", "PC2"):
        if nm in axes:
            out[f"{nm}_BC"] = sarle_bc(R @ axes[nm])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dir containing the cluster<K><suffix> folders")
    ap.add_argument("--clusters", required=True,
                    help="'0:fast,1:stopgo,2:mid,3:turning' -- id:label")
    ap.add_argument("--suffix", default="_only",
                    help="folder suffix (cluster_post writes cluster<K>_only)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    specs = []
    for it in args.clusters.split(","):
        k, _, lab = it.partition(":")
        specs.append((int(k), lab or f"c{k}"))

    long_rows, bc_rows, missing = [], [], []
    for k, lab in specs:
        folder = root / f"cluster{k}{args.suffix}"
        cons = folder / "role_scene_consistency.csv"
        if not cons.exists():
            missing.append(str(folder))
            continue
        c = pd.read_csv(cons)
        allc = c[c["stratum"] == "ALL"] if "stratum" in c.columns else c
        for _, r in allc.iterrows():
            long_rows.append({
                "cluster": k, "label": lab, "axis": r["axis"],
                "metric": r["metric"], "consistency": r.get("consistency"),
                "paired_t": r.get("paired_t"),
                "pop_mean_delta": r.get("pop_mean_delta"),
                "n": r.get("n")})
        bc = cluster_bimodality(folder)
        bc.update({"cluster": k, "label": lab})
        bc_rows.append(bc)

    if not long_rows:
        raise SystemExit(f"no role_scene_consistency.csv found under {root} "
                         f"for {[s[0] for s in specs]}. Missing: {missing}")
    if missing:
        print(f"[warn] skipped (analysis not ready): {missing}")

    long = pd.DataFrame(long_rows)
    long.to_csv(out / "cross_cluster_contrast.csv", index=False)
    bc_df = pd.DataFrame(bc_rows)
    bc_df.to_csv(out / "cross_cluster_bimodality.csv", index=False)

    present = [(k, lab) for k, lab in specs
               if k in long["cluster"].values]
    metrics = [m for m in METRICS if m in long["metric"].values]

    # ---- headline: what does each cluster's PC1 control? ---------------------
    print("\n=== HEADLINE: what PC1 controls in each cluster "
          f"(highest per-scene consistency among metrics with |t|>={T_MIN}) ===")
    for k, lab in present:
        sub = long[(long["cluster"] == k) & (long["axis"] == "PC1")].copy()
        real = sub[sub["paired_t"].abs() >= T_MIN]
        pool = real if len(real) else sub
        if not len(pool):
            continue
        best = pool.loc[pool["consistency"].astype(float).idxmax()]
        bcv = next((r.get("PC1_BC") for r in bc_rows if r.get("cluster") == k), np.nan)
        print(f"  cluster{k} {lab:<8}: PC1 -> {best['metric']:<11} "
              f"(consistency {float(best['consistency']):.2f}, "
              f"t={float(best['paired_t']):+.0f})   PC1 role BC="
              f"{bcv:.2f}" if bcv == bcv else
              f"  cluster{k} {lab}: PC1 -> {best['metric']} "
              f"(consistency {float(best['consistency']):.2f})")

    # ---- money figure: PC1 (+PC2) consistency heatmap ------------------------
    for axis in ("PC1", "PC2"):
        piv = long[long["axis"] == axis].pivot_table(
            index="cluster", columns="metric", values="consistency",
            aggfunc="mean")
        if piv.empty:
            continue
        piv = piv.reindex(index=[k for k, _ in present],
                          columns=[m for m in metrics if m in piv.columns])
        tpiv = long[long["axis"] == axis].pivot_table(
            index="cluster", columns="metric", values="paired_t",
            aggfunc="mean").reindex(index=piv.index, columns=piv.columns)

        fig, ax = plt.subplots(figsize=(1.6 * len(piv.columns) + 2,
                                        1.0 * len(piv.index) + 2))
        data = piv.values.astype(float)
        im = ax.imshow(data, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"c{k}:{dict(present)[k]}" for k in piv.index],
                           fontsize=9)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isfinite(v):
                    continue
                tval = tpiv.values[i, j]
                star = "*" if abs(tval) >= T_MIN else ""
                ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center",
                        color="white" if v < 0.78 else "black", fontsize=9)
        ax.set_title(f"{axis} per-scene consistency by cluster x metric\n"
                     f"(* = real monotone effect, |t|>={T_MIN}; "
                     f"each task's role controls a DIFFERENT metric)",
                     fontsize=10)
        fig.colorbar(im, ax=ax, label="per-scene sign consistency", shrink=0.8)
        fig.tight_layout()
        p = out / f"cross_cluster_contrast_{axis}.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"[fig] {p}")

    print(f"\n[bimodality] PC1 role continuum vs collapse per cluster:")
    for r in bc_rows:
        print(f"  cluster{r.get('cluster')} {r.get('label',''):<8}: "
              f"PC1_BC={r.get('PC1_BC', float('nan')):.2f}  "
              f"PC2_BC={r.get('PC2_BC', float('nan')):.2f}  n={r.get('n','?')}"
              f"   (BC<0.55 = smooth continuum, ->1 = binary switch)")
    print(f"\n[cross] wrote -> {out}/cross_cluster_contrast.csv, "
          f"cross_cluster_bimodality.csv, *.png")


if __name__ == "__main__":
    main()
