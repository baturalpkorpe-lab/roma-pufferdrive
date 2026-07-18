"""
role_grid_table.py -- turn a render_role_grid grid_summary.csv into the
PC1xPC2 INTERACTION table + figure that the single-axis sweeps cannot show.

The single PC1 / PC2 dose-responses each hold the OTHER axis at 0, so they
miss INTERACTIONS. E.g. cluster0's speed FALLS with PC2 at PC1=0 (the pc2
sweep) but RISES with PC2 at PC1=-2 (the g-2_+2 corner) -- because the role
space is a non-linear 2-D continuum and the linear PC2 axis has a PC1-
dependent effect. Only the grid corners (g+-2_+-2) probe off the axes.

Builds (from the render's grid_summary.csv -- no GPU):
  grid_table_<metric>.csv    3x3 table: rows PC1 in {-2,0,2}, cols PC2 in
                             {-2,0,2}, cell = mean metric (the natural cell is
                             PC1=0,PC2=0). One per metric.
  grid_interaction_<metric>.png  LEFT: metric vs PC2, one line per PC1 level
                             (non-parallel lines = interaction). RIGHT: the
                             3x3 as a heatmap.

CAVEAT: n per cell = the render's --scenes_per_type (default 3), so treat
magnitudes as ILLUSTRATIVE. For a statistically solid interaction, run a
paired grid sweep (many focals); this is the quick look at what you rendered.

Usage:
  python role_grid_table.py \
      --grid_summary /scratch/e452103/renders/role_grid_cluster0/type0/grid_summary.csv \
      --out_dir      /scratch/e452103/renders/role_grid_cluster0/type0 \
      --metric speed_mean
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEVELS = [-2.0, 0.0, 2.0]
METRICS = ["speed_mean", "accel_pos", "decel_abs", "jerk_abs", "turn_abs",
           "event_rate", "offroad_rate"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_summary", required=True)
    p.add_argument("--out_dir",      required=True)
    p.add_argument("--metric",       default="speed_mean",
                   help="metric for the interaction figure")
    p.add_argument("--traj",         type=int, default=None,
                   help="restrict to one traj_type (else pool all)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.grid_summary)
    if args.traj is not None and "traj_type" in df.columns:
        df = df[df["traj_type"] == args.traj]
    # keep only the 3x3 grid cells (PC1, PC2 both in {-2,0,2})
    df = df[df["pc1"].round().isin(LEVELS) & df["pc2"].round().isin(LEVELS)].copy()
    df["pc1"] = df["pc1"].round()
    df["pc2"] = df["pc2"].round()
    if df.empty:
        raise SystemExit("no grid cells found -- was --grid 1 used in the render?")
    metrics = [m for m in METRICS if m in df.columns]
    n_scenes = df.groupby(["pc1", "pc2"]).size().max()
    print(f"[grid] {len(df)} rows, ~{n_scenes} scenes/cell "
          f"(traj={'all' if args.traj is None else args.traj})")

    # -- 3x3 tables per metric -------------------------------------------------
    for m in metrics:
        piv = df.pivot_table(index="pc1", columns="pc2", values=m, aggfunc="mean")
        piv = piv.reindex(index=LEVELS, columns=LEVELS)
        piv.to_csv(out / f"grid_table_{m}.csv")
        print(f"\n=== {m}  (rows=PC1, cols=PC2) ===")
        print(piv.round(3).to_string())

    # -- interaction quantification (on the chosen metric) --------------------
    M = args.metric
    piv = df.pivot_table(index="pc1", columns="pc2", values=M,
                         aggfunc="mean").reindex(index=LEVELS, columns=LEVELS)
    # PC2 effect (+2 minus -2) at each PC1 level -> if it changes sign/size,
    # that IS the interaction.
    pc2_effect = piv[2.0] - piv[-2.0]
    print(f"\n=== INTERACTION on {M}: PC2 effect (PC2=+2 minus -2) per PC1 level ===")
    for lv in LEVELS:
        print(f"  at PC1={lv:+.0f}:  Δ{M}(PC2:-2->+2) = {pc2_effect[lv]:+.3f}")
    print(f"  interaction size = effect@PC1=-2 minus effect@PC1=+2 = "
          f"{pc2_effect[-2.0] - pc2_effect[2.0]:+.3f}  "
          f"(0 = additive/no interaction; large = strong interaction)")

    # -- figure: lines (interaction) + heatmap --------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("coolwarm")
    for i, lv in enumerate(LEVELS):
        axL.plot(LEVELS, piv.loc[lv].values, marker="o", lw=2,
                 color=cmap(i / 2), label=f"PC1={lv:+.0f}")
    axL.set_xlabel("PC2 (σ)"); axL.set_ylabel(M)
    axL.set_xticks(LEVELS)
    axL.set_title(f"{M} vs PC2, one line per PC1 level\n"
                  "non-parallel lines = PC1xPC2 interaction")
    axL.legend(); axL.grid(alpha=0.3)

    im = axR.imshow(piv.values, cmap="coolwarm", origin="lower", aspect="auto")
    axR.set_xticks(range(3)); axR.set_xticklabels([f"{v:+.0f}" for v in LEVELS])
    axR.set_yticks(range(3)); axR.set_yticklabels([f"{v:+.0f}" for v in LEVELS])
    axR.set_xlabel("PC2 (σ)"); axR.set_ylabel("PC1 (σ)")
    for a in range(3):
        for b in range(3):
            v = piv.values[a, b]
            if np.isfinite(v):
                axR.text(b, a, f"{v:.1f}", ha="center", va="center",
                         color="white" if abs(v - np.nanmean(piv.values)) >
                         0.5 * np.nanstd(piv.values) else "black", fontsize=10)
    plt.colorbar(im, ax=axR, label=M)
    axR.set_title(f"{M} over the PC1xPC2 grid")
    fig.suptitle(f"PC1xPC2 interaction  (n~{n_scenes} scenes/cell -- "
                 f"ILLUSTRATIVE; corners fastest? read the heatmap)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"grid_interaction_{M}.png", dpi=140)
    plt.close(fig)
    print(f"\n[grid] wrote grid_table_*.csv + grid_interaction_{M}.png -> {out}")


if __name__ == "__main__":
    main()
