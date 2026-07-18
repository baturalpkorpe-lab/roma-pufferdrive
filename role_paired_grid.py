"""
role_paired_grid.py -- statistically-solid PC1xPC2 INTERACTION.

Same paired natural-offset forcing as role_paired_sweep, but over the full
grid of (alpha1*sigma1*PC1 + alpha2*sigma2*PC2) conditions -- so every grid
cell has hundreds of paired focals and a real mean +- t, not the render
grid's n~3. Answers: is the PC2 effect on behaviour genuinely different at
different PC1 levels (a real interaction / non-linear role manifold), or is
the render-grid pattern just small-n noise / off-manifold extrapolation?

Outputs (to --out_dir):
  role_paired_grid_agent.csv   every (episode, sid, vid, pc1, pc2) + metrics
  role_paired_grid_cells.csv   per (pc1,pc2,metric): abs mean, paired delta vs
                               natural(0,0), paired t, n
  role_paired_grid_interaction.csv  per metric: PC2 effect (paired) at each
                               PC1 level + the 2x2 corner difference-of-
                               differences interaction contrast + its t
  grid_interaction_<metric>.png  abs-mean heatmap + delta-vs-PC2 lines per PC1

Reuses role_paired_sweep.rollout_condition (identical forcing/pairing). Axes
loaded from role_paired_axes.csv so PC1/PC2 match every other figure.

Usage: see slurm/role_paired_grid.sbatch.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import load_policy
from render_role_alpha import load_axes
from role_paired_sweep import rollout_condition, load_drive_config

METRICS = ["speed_mean", "accel_abs", "accel_pos", "decel_abs", "jerk_abs",
           "turn_abs", "event_rate"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--axes_csv",   required=True)
    p.add_argument("--clusters",   required=True,
                   help="map_clusters.csv for focal selection")
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--alphas",     default="-2,0,2",
                   help="grid levels applied to BOTH PC1 and PC2 (NxN conds)")
    p.add_argument("--episodes",   type=int, default=8)
    p.add_argument("--num_agents", type=int, default=2048)
    p.add_argument("--num_maps",   type=int, default=10000)
    p.add_argument("--min_margin", type=float, default=1.5)
    p.add_argument("--metric",     default="speed_mean")
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


def paired_t(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 3 or d.std(ddof=1) == 0:
        return (float(d.mean()) if len(d) else np.nan), np.nan, len(d)
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    alphas = sorted(float(x) for x in args.alphas.split(","))

    cl = pd.read_csv(args.clusters)
    cl = cl[cl["margin"] > args.min_margin]
    regime_of = dict(zip(cl["scenario_id"].astype(str), cl["cluster"]))

    from pufferlib.ocean.drive.drive import Drive
    env_cfg = dict(load_drive_config()["env"])
    env_cfg.update({"num_maps": args.num_maps, "num_agents": args.num_agents,
                    "map_dir": args.data_dir})
    env = Drive(**env_cfg)
    obs, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs.shape[-1], device)
    if role_dim < 2:
        raise SystemExit(f"role_dim={role_dim} -- need >=2 dims for a PC1xPC2 grid")
    mu, axes = load_axes(args.axes_csv, role_dim)
    if "PC1" not in axes or "PC2" not in axes:
        raise SystemExit(f"axes csv needs PC1+PC2, has {list(axes)}")
    u1, s1 = axes["PC1"]; u2, s2 = axes["PC2"]
    u1 = np.asarray(u1, dtype=np.float32); u2 = np.asarray(u2, dtype=np.float32)

    conds = []
    for a1 in alphas:
        for a2 in alphas:
            vec = None if (a1 == 0 and a2 == 0) else (a1 * s1 * u1 + a2 * s2 * u2)
            conds.append((a1, a2, vec))
    print(f"[grid] {len(conds)} conditions x {args.episodes} episodes "
          f"(natural-offset forcing)")

    all_rows = []
    for ep in range(args.episodes):
        env.resample_maps()
        for a1, a2, vec in conds:
            rows, *_ = rollout_condition(env, policy, device, regime_of, vec)
            for r in rows:
                r.update({"episode": ep, "pc1": a1, "pc2": a2})
            all_rows.extend(rows)
            print(f"[grid] ep{ep} ({a1:+g},{a2:+g}): {len(rows)} focals", flush=True)
    env.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "role_paired_grid_agent.csv", index=False)
    metrics = [m for m in METRICS if m in df.columns]
    key = ["episode", "sid", "vid"]

    # -- per-cell: abs mean + paired delta vs natural(0,0) + t -----------------
    base = df[(df.pc1 == 0) & (df.pc2 == 0)].set_index(key)
    cells = []
    for a1 in alphas:
        for a2 in alphas:
            sub = df[(df.pc1 == a1) & (df.pc2 == a2)].set_index(key)
            j = sub.join(base, how="inner", lsuffix="", rsuffix="_b")
            row = {"pc1": a1, "pc2": a2, "n": len(j)}
            for m in metrics:
                row[f"{m}_abs"] = float(sub[m].mean())
                dm, dt, _ = paired_t((j[m] - j[f"{m}_b"]).values)
                row[f"{m}_delta"] = round(dm, 4)
                row[f"{m}_t"] = round(dt, 1) if np.isfinite(dt) else np.nan
            cells.append(row)
    pd.DataFrame(cells).to_csv(out / "role_paired_grid_cells.csv", index=False)

    # -- interaction: PC2 effect per PC1 level + 2x2 corner DD test -------------
    amin, amax = min(alphas), max(alphas)
    inter_rows = []
    for m in metrics:
        piv = df.pivot_table(index=key, columns=["pc1", "pc2"], values=m)
        rec = {"metric": m}
        for a1 in alphas:                            # PC2 effect at this PC1 level
            if (a1, amax) in piv.columns and (a1, amin) in piv.columns:
                eff, t, n = paired_t((piv[(a1, amax)] - piv[(a1, amin)]).values)
                rec[f"pc2eff@pc1={a1:+g}"] = round(eff, 4)
                rec[f"pc2eff_t@pc1={a1:+g}"] = round(t, 1) if np.isfinite(t) else np.nan
        # 2x2 corner difference-of-differences (THE interaction test)
        need = [(amax, amax), (amax, amin), (amin, amax), (amin, amin)]
        if all(c in piv.columns for c in need):
            dd = ((piv[(amax, amax)] - piv[(amax, amin)])
                  - (piv[(amin, amax)] - piv[(amin, amin)]))
            ddm, ddt, ddn = paired_t(dd.values)
            rec["interaction_DD"] = round(ddm, 4)
            rec["interaction_t"] = round(ddt, 1) if np.isfinite(ddt) else np.nan
            rec["interaction_n"] = ddn
        inter_rows.append(rec)
    idf = pd.DataFrame(inter_rows)
    idf.to_csv(out / "role_paired_grid_interaction.csv", index=False)
    print("\n[grid] === interaction (PC2 effect per PC1 level + corner DD) ===")
    print(idf.to_string(index=False))

    # -- figure for the chosen metric -----------------------------------------
    M = args.metric
    cdf = pd.DataFrame(cells)
    abs_piv = cdf.pivot(index="pc1", columns="pc2", values=f"{M}_abs")
    del_piv = cdf.pivot(index="pc1", columns="pc2", values=f"{M}_delta")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("coolwarm")
    for i, a1 in enumerate(alphas):
        axL.plot(alphas, del_piv.loc[a1].values, marker="o", lw=2,
                 color=cmap(i / max(len(alphas) - 1, 1)), label=f"PC1={a1:+g}")
    axL.axhline(0, color="gray", lw=0.8)
    axL.set_xlabel("PC2 (σ)"); axL.set_ylabel(f"Δ{M} vs natural")
    axL.set_xticks(alphas)
    axL.set_title(f"Δ{M} vs PC2, one line per PC1 level\n"
                  "non-parallel = interaction (paired, so scene noise cancels)")
    axL.legend(); axL.grid(alpha=0.3)

    im = axR.imshow(abs_piv.values, cmap="coolwarm", origin="lower", aspect="auto")
    axR.set_xticks(range(len(alphas))); axR.set_xticklabels([f"{v:+g}" for v in alphas])
    axR.set_yticks(range(len(alphas))); axR.set_yticklabels([f"{v:+g}" for v in alphas])
    axR.set_xlabel("PC2 (σ)"); axR.set_ylabel("PC1 (σ)")
    vals = abs_piv.values
    for a in range(vals.shape[0]):
        for b in range(vals.shape[1]):
            if np.isfinite(vals[a, b]):
                axR.text(b, a, f"{vals[a, b]:.1f}", ha="center", va="center",
                         fontsize=9, color="black")
    plt.colorbar(im, ax=axR, label=f"{M} (abs mean)")
    it = idf[idf.metric == M]
    itt = it["interaction_t"].iloc[0] if len(it) and "interaction_t" in it else np.nan
    axR.set_title(f"{M} over the PC1xPC2 grid\ncorner interaction t={itt}")
    fig.suptitle(f"PAIRED PC1xPC2 interaction ({M}) -- n~{cells[0]['n']} pairs/cell",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"grid_interaction_{M}.png", dpi=140)
    plt.close(fig)
    print(f"\n[grid] wrote grid_interaction_{M}.png + *_cells.csv + "
          f"*_interaction.csv -> {out}")


if __name__ == "__main__":
    main()
