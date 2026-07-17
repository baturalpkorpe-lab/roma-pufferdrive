"""
role_space_map.py -- ONE figure that shows the whole role story: the 2-D PCA
map of natural role vectors, colored by each agent's speed, with the PC1/PC2
dial directions drawn as arrows and the alpha = -2..+2 sweep positions marked.

Reader takeaway at a glance: where natural agents live in role space, that the
space is a continuum (no islands), that speed varies smoothly along PC1 (the
color gradient), and what "forcing alpha" means geometrically (the tick marks).

Input: any per-agent CSV with role_0..role_{D-1} columns + a speed column --
  /scratch/e452103/role_direct/dim4/role_direct_agent_data.csv   (speed_mean)
  /scratch/e452103/role_paired/dim4/role_paired_warmup.csv       (speed_mean)

No GPU, no env. Usage:
    python role_space_map.py \
        --agent_csv /scratch/e452103/role_direct/dim4/role_direct_agent_data.csv \
        --out /scratch/e452103/role_direct/dim4/role_space_map.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_csv", type=str, required=True,
                   help="CSV with role_* columns + the color feature column")
    p.add_argument("--color_by",  type=str, default="speed_mean",
                   help="column to color dots by + orient PC1 (e.g. turn_abs "
                        "for a turning cluster where the role controls steering)")
    p.add_argument("--speed_col", type=str, default=None,
                   help="deprecated alias for --color_by")
    p.add_argument("--out",       type=str, required=True)
    p.add_argument("--alphas",    type=str, default="-2,-1,0,1,2")
    p.add_argument("--max_points", type=int, default=20000)
    p.add_argument("--vmin",      type=float, default=None,
                   help="colorbar floor; unset = auto (2nd percentile of the "
                        "feature IN THIS CSV) -- normalizes per-dataset instead "
                        "of a fixed scale, so a low-speed cluster (e.g. "
                        "stop&go) still shows its own internal variation "
                        "instead of collapsing to one end of the colormap")
    p.add_argument("--vmax",      type=float, default=None,
                   help="colorbar cap; unset = auto (98th percentile of the "
                        "feature in this CSV)")
    return p.parse_args()


def main():
    args = parse_args()
    color_col = args.speed_col or args.color_by     # speed_col is a legacy alias
    import pandas as pd
    df = pd.read_csv(args.agent_csv)
    if color_col not in df.columns:
        raise SystemExit(f"no '{color_col}' column; have {list(df.columns)}")
    role_cols = sorted([c for c in df.columns if c.startswith("role_")
                        and c[5:].isdigit()], key=lambda c: int(c[5:]))
    lc = {c.lower(): c for c in df.columns}

    if role_cols:
        # Preferred: compute PCA from the raw role vectors.
        df = df.dropna(subset=role_cols + [color_col])
        R   = df[role_cols].values.astype(np.float64)
        cen = R - R.mean(axis=0)
        _, svals, vt = np.linalg.svd(cen, full_matrices=False)
        evr = (svals ** 2) / (svals ** 2).sum()
        p1 = cen @ vt[0]
        p2 = cen @ vt[1] if R.shape[1] > 1 else np.zeros(len(cen))
        print(f"[map] {len(df)} agents, role_dim={len(role_cols)} "
              f"(PCA from role vectors)")
    elif "pc1" in lc:
        # Fallback: use precomputed pc1/pc2 projections (role_paired_warmup.csv).
        need = [lc["pc1"]] + ([lc["pc2"]] if "pc2" in lc else []) + [color_col]
        df = df.dropna(subset=need)
        p1 = df[lc["pc1"]].values.astype(np.float64)
        p2 = (df[lc["pc2"]].values.astype(np.float64) if "pc2" in lc
              else np.zeros(len(df)))
        v1, v2 = p1.var(), p2.var()
        evr = np.array([v1 / (v1 + v2 + 1e-12), v2 / (v1 + v2 + 1e-12)])
        print(f"[map] {len(df)} agents (precomputed pc1/pc2; var% among the "
              f"retained PCs)")
    else:
        raise SystemExit(f"need role_* or pc1/pc2 columns in {args.agent_csv}")

    spd = df[color_col].values.astype(np.float64)
    # Orient PC1 so + points to the HIGH end of the color feature (SVD sign is
    # arbitrary, esp. for dim-2) -- keeps red on the right, dial reading right.
    ok = np.isfinite(p1) & np.isfinite(spd)
    if ok.sum() > 10 and np.corrcoef(p1[ok], spd[ok])[0, 1] < 0:
        p1 = -p1
    s1, s2 = p1.std(), p2.std()
    # Per-dataset normalization: clip both ends to this CSV's own 2nd/98th
    # percentile instead of a fixed/absolute scale. A low-speed cluster (e.g.
    # stop&go, speeds ~0-10) would otherwise all land near one end of a
    # colormap calibrated for the full-speed range (~0-25) and show no visible
    # gradient even though real relative variation exists within the cluster.
    vmin = args.vmin if args.vmin is not None else float(np.nanpercentile(spd, 2))
    vmax = args.vmax if args.vmax is not None else float(np.nanpercentile(spd, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    alphas = sorted(float(x) for x in args.alphas.split(","))

    rng = np.random.default_rng(0)
    idx = rng.choice(len(p1), min(args.max_points, len(p1)), replace=False)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    sc = ax.scatter(p1[idx], p2[idx], c=np.clip(spd[idx], vmin, vmax),
                    cmap="coolwarm", vmin=vmin, vmax=vmax,
                    s=5, alpha=0.35, rasterized=True)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(f"{color_col}  (colorbar range {vmin:.2f}..{vmax:.2f}, "
                 f"this dataset's own 2nd-98th percentile)")

    # dial axes: arrows from the population mean (origin in PC coordinates)
    span1 = max(abs(a) for a in alphas) * s1
    ax.annotate("", xy=(span1 * 1.08, 0), xytext=(-span1 * 1.08, 0),
                arrowprops=dict(arrowstyle="<|-|>", color="black", lw=2))
    for al in alphas:
        ax.plot([al * s1], [0], marker="|", ms=16, mew=2.5, color="black")
        ax.annotate(f"α={al:+g}", (al * s1, 0), xytext=(0, -18),
                    textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold")
    # label above the arrow, anchored at the tip and extending LEFT (into the
    # plot) so it never collides with the colorbar
    ax.annotate(f"PC1 dial ({evr[0]:.0%} var)",
                (span1, 0), xytext=(-4, 12), textcoords="offset points",
                fontsize=10, fontweight="bold", ha="right")
    if s2 > 0:
        span2 = max(abs(a) for a in alphas) * s2
        ax.annotate("", xy=(0, span2 * 1.08), xytext=(0, -span2 * 1.08),
                    arrowprops=dict(arrowstyle="<|-|>", color="#444444",
                                    lw=1.4, linestyle="--"))
        for al in alphas:
            if al:
                ax.plot([0], [al * s2], marker="_", ms=14, mew=2,
                        color="#444444")
        ax.annotate(f"PC2 ({evr[1]:.0%} var)", (0, span2 * 1.05),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9, color="#444444")
    ax.scatter([0], [0], marker="*", s=260, color="black", zorder=6,
               edgecolors="white", linewidths=1)
    ax.annotate("μ (population mean)", (0, 0), xytext=(10, 10),
                textcoords="offset points", fontsize=9)

    ax.set_xlabel(f"PC1 score (σ₁={s1:.2f})")
    ax.set_ylabel(f"PC2 score (σ₂={s2:.2f})" if s2 > 0 else "")
    ax.set_title(f"The role space: natural agents (one dot each), colored by "
                 f"{color_col}.\nForcing α slides an agent along the marked "
                 f"PC1 dial.", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[map] PC1 {evr[0]:.0%}  PC2 {evr[1] if len(evr)>1 else 0:.0%}  "
          f"-> {out}")


if __name__ == "__main__":
    main()
