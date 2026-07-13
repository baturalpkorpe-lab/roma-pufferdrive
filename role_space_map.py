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
                   help="CSV with role_* columns + a speed column")
    p.add_argument("--speed_col", type=str, default="speed_mean")
    p.add_argument("--out",       type=str, required=True)
    p.add_argument("--alphas",    type=str, default="-2,-1,0,1,2")
    p.add_argument("--max_points", type=int, default=20000)
    p.add_argument("--vmax",      type=float, default=25.0,
                   help="Speed colorbar cap (m/s)")
    return p.parse_args()


def main():
    args = parse_args()
    import pandas as pd
    df = pd.read_csv(args.agent_csv)
    role_cols = sorted([c for c in df.columns if c.startswith("role_")
                        and c[5:].isdigit()], key=lambda c: int(c[5:]))
    if not role_cols:
        raise SystemExit(f"no role_* columns in {args.agent_csv}")
    if args.speed_col not in df.columns:
        raise SystemExit(f"no '{args.speed_col}' column; have {list(df.columns)}")
    df = df.dropna(subset=role_cols + [args.speed_col])
    print(f"[map] {len(df)} agents, role_dim={len(role_cols)}")

    R   = df[role_cols].values.astype(np.float64)
    spd = df[args.speed_col].values.astype(np.float64)
    mu  = R.mean(axis=0)
    cen = R - mu
    _, svals, vt = np.linalg.svd(cen, full_matrices=False)
    evr = (svals ** 2) / (svals ** 2).sum()
    p1, p2 = cen @ vt[0], (cen @ vt[1] if R.shape[1] > 1
                           else np.zeros(len(cen)))
    # Orient PC1 so + points to the fast/assertive end (SVD sign is arbitrary,
    # esp. for dim-2) -- keeps red on the right and the dial reading correctly.
    ok = np.isfinite(p1) & np.isfinite(spd)
    if ok.sum() > 10 and np.corrcoef(p1[ok], spd[ok])[0, 1] < 0:
        p1 = -p1
    s1, s2 = p1.std(), (p2.std() if R.shape[1] > 1 else 0.0)
    alphas = sorted(float(x) for x in args.alphas.split(","))

    rng = np.random.default_rng(0)
    idx = rng.choice(len(R), min(args.max_points, len(R)), replace=False)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    sc = ax.scatter(p1[idx], p2[idx], c=np.clip(spd[idx], 0, args.vmax),
                    cmap="coolwarm", s=5, alpha=0.35, rasterized=True)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(f"{args.speed_col} (m/s, capped at {args.vmax:.0f})")

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
    ax.annotate(f"PC1 — assertiveness dial ({evr[0]:.0%} var)",
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
    ax.set_title("The role space: natural agents (one dot each), colored by "
                 "speed.\nA continuum, not clusters; forcing α slides an agent "
                 "along the marked dial.", fontsize=11)
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
