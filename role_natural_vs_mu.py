"""
role_natural_vs_mu.py -- does the agent's OWN role beat the population mean?

Compares, on the SAME (map, vehicle) pairs from role_paired_sweep's output:
  natural   focal keeps its own encoder role (unforced)
  base      focal forced to mu (the population-mean role, alpha=0)

Paired delta = natural - base, per metric, overall and per regime.

Reading:
  ~0 everywhere        -> forcing per se is benign; the agent's self-assigned
                          role adds little over a generic average driver.
  natural smoother /   -> the encoder's role assignment carries useful,
  safer than mu           scene-appropriate information (individuality works);
                          also quantifies how much the forced-mu baseline in
                          the dose-response figures differs from natural.

No GPU, no env -- reads <in_dir>/role_paired_agent.csv.

Usage:
    python role_natural_vs_mu.py --in_dir /scratch/e452103/role_paired/dim4
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["speed_mean", "accel_abs", "jerk_abs", "turn_abs", "event_rate"]
METRIC_LABEL = {"speed_mean": "speed (m/s)", "accel_abs": "|accel| (m/s2)",
                "jerk_abs": "|jerk| (m/s3)", "turn_abs": "|turn| (rad/s)",
                "event_rate": "safety events / 91"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, required=True,
                   help="Dir holding role_paired_agent.csv")
    p.add_argument("--regime_names", type=str, default="")
    args = p.parse_args()
    in_dir = Path(args.in_dir)

    import pandas as pd
    names = {}
    if args.regime_names:
        names = {int(k): v for k, v in
                 (it.split(":") for it in args.regime_names.split(","))}

    df = pd.read_csv(in_dir / "role_paired_agent.csv")
    key = ["episode", "sid", "vid"]
    nat = df[df["cond"] == "natural"].set_index(key)
    mu_ = df[df["cond"] == "base"].set_index(key)
    j = nat.join(mu_, how="inner", lsuffix="", rsuffix="_mu")
    print(f"[nat-vs-mu] {len(j)} same-(map,vehicle) pairs "
          f"(natural={len(nat)}, base={len(mu_)})")

    regimes = sorted(j["regime"].unique())
    rows = []
    print("\n[nat-vs-mu] === natural - forced-mu, paired on the same "
          "(map, vehicle) ===")
    for met in METRICS:
        dd = (j[met] - j[f"{met}_mu"]).dropna()
        t = float(dd.mean() / (dd.std() / len(dd) ** 0.5)) if len(dd) > 4 else np.nan
        rows.append({"metric": met, "regime": "all",
                     "mean_diff": float(dd.mean()),
                     "sem": float(dd.std() / len(dd) ** 0.5),
                     "t_stat": t, "n_pairs": int(len(dd)),
                     "mean_natural": float(j[met].mean()),
                     "mean_mu": float(j[f"{met}_mu"].mean())})
        print(f"[nat-vs-mu]   {met:<11}: diff={dd.mean():+.3f} ± "
              f"{1.96 * dd.std() / len(dd) ** 0.5:.3f}  t={t:+.1f}  "
              f"n={len(dd)}  (natural {j[met].mean():.2f} vs "
              f"mu {j[f'{met}_mu'].mean():.2f})")
        for rg in regimes:
            ddr = (j[met] - j[f"{met}_mu"])[j["regime"] == rg].dropna()
            if len(ddr) >= 5:
                rows.append({"metric": met, "regime": rg,
                             "mean_diff": float(ddr.mean()),
                             "sem": float(ddr.std() / len(ddr) ** 0.5),
                             "t_stat": float(ddr.mean() /
                                             (ddr.std() / len(ddr) ** 0.5)),
                             "n_pairs": int(len(ddr)),
                             "mean_natural": float(j[met][j["regime"] == rg].mean()),
                             "mean_mu": float(j[f"{met}_mu"][j["regime"] == rg].mean())})

    out = pd.DataFrame(rows)
    out.to_csv(in_dir / "role_natural_vs_mu.csv", index=False)

    # Figure: one panel per metric, bar per regime + overall, ±95% CI
    fig, axs = plt.subplots(1, len(METRICS), figsize=(3.0 * len(METRICS), 3.6))
    for ax, met in zip(np.atleast_1d(axs), METRICS):
        sub = out[out["metric"] == met]
        labs, vals, errs, cols = [], [], [], []
        for _, r in sub.iterrows():
            is_all = r["regime"] == "all"
            labs.append("ALL" if is_all else names.get(r["regime"],
                                                       f"reg {r['regime']}"))
            vals.append(r["mean_diff"])
            errs.append(1.96 * r["sem"])
            cols.append("#222222" if is_all else "#4477aa")
        x = np.arange(len(labs))
        ax.bar(x, vals, yerr=errs, capsize=3, color=cols)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"Δ {METRIC_LABEL[met]}\n(natural − forced-μ)", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Agent's OWN role vs population-mean role — paired on the "
                 "same (map, vehicle)", fontsize=11)
    fig.tight_layout()
    fig.savefig(in_dir / "role_natural_vs_mu.png", dpi=140)
    plt.close(fig)
    print(f"\n[nat-vs-mu] outputs -> {in_dir}/role_natural_vs_mu.csv + .png")


if __name__ == "__main__":
    main()
