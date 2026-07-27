"""
role_feature_corr.py -- two correlation heatmaps from any agent-level role
CSV (role_paired_warmup.csv / role_direct_agent_data.csv):
  role_role_corr.png     role dims against each other (redundancy check,
                          same number eval_roma.py's dim-correlation health
                          check reports, but computed here with no GPU/env)
  role_feature_corr.png  role dims against raw behavioral features
                          (speed_mean, accel_abs, jerk_abs, turn_abs,
                          event_rate, + d_speed/d_turn_rate/d_jerk/speed_ratio
                          /ade if the CSV has them)

No GPU, no env -- pure pandas over an existing CSV, runs on the login node.

Usage:
    python role_feature_corr.py \
        --agent_csv /scratch/e452103/role_paired/cluster1_cpu/role_paired_warmup.csv \
        --out_dir   /scratch/e452103/role_paired/cluster1_cpu
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_FEATS = ["speed_mean", "accel_abs", "jerk_abs", "turn_abs", "event_rate",
             "d_speed", "d_turn_rate", "d_jerk", "speed_ratio", "ade"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_csv", type=str, required=True)
    p.add_argument("--out_dir",   type=str, required=True)
    return p.parse_args()


def heatmap(mat, rows, cols, title, out_path):
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2, 0.6 * len(rows) + 2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.agent_csv)

    role_cols = sorted([c for c in df.columns if c.startswith("role_")],
                       key=lambda c: int(c.split("_")[1]))
    if not role_cols:
        raise SystemExit(f"no role_* columns in {args.agent_csv}")
    feat_cols = [c for c in RAW_FEATS if c in df.columns]
    print(f"role dims: {role_cols}")
    print(f"raw features found: {feat_cols}")

    # -- role x role intercorrelation --------------------------------------
    R  = df[role_cols].values.astype(np.float64)
    ok = np.isfinite(R).all(axis=1)
    # np.corrcoef of a SINGLE variable returns a 0-d array, not a 1x1 matrix,
    # so a role_dim=1 checkpoint used to die in heatmap() on mat[i, j].
    corr_rr = np.atleast_2d(np.corrcoef(R[ok].T))
    heatmap(corr_rr, role_cols, role_cols,
            f"Role dim inter-correlation  (n={int(ok.sum())})",
            out / "role_role_corr.png")
    off = np.abs(corr_rr - np.eye(len(role_cols)))
    max_off = float(off.max()) if len(role_cols) > 1 else 0.0
    print(f"max |off-diagonal r| = {max_off:.3f}"
          + ("  -- redundant dims" if max_off > 0.6 else "  -- fairly independent"))

    # -- role x raw feature correlation -------------------------------------
    if feat_cols:
        F = df[feat_cols].values.astype(np.float64)
        M = np.full((len(role_cols), len(feat_cols)), np.nan)
        for i in range(len(role_cols)):
            for j in range(len(feat_cols)):
                x, y = R[:, i], F[:, j]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() > 10 and x[m].std() > 0 and y[m].std() > 0:
                    M[i, j] = np.corrcoef(x[m], y[m])[0, 1]
        heatmap(M, role_cols, feat_cols,
                f"Role dims vs raw behavioral features  (n={int(ok.sum())})",
                out / "role_feature_corr.png")
    else:
        print("no raw feature columns found -- need speed_mean/accel_abs/... "
              "in the CSV (role_paired_warmup.csv has these by default)")


if __name__ == "__main__":
    main()
