"""
role_scene_icc.py -- how much of the role is the SCENE, and how much is the AGENT?

The question behind the per-cluster training pivot, asked directly instead of
inferred from downstream behaviour. Roll out naturally, collect one role vector
per (scene, vehicle), and decompose its variance:

    ICC = between-scene variance / (between-scene + within-scene)

    ICC -> 1  every agent in a scene gets nearly the SAME role; the role is a
              property of the map/situation. Forcing a role is then fighting the
              encoder, per-scene consistency is structurally capped, and one
              policy over the full dataset cannot express per-agent style.
    ICC -> 0  agents in the same scene get DIFFERENT roles; the role is a
              property of the driver. This is what the architecture is meant to
              produce.

THE CONTROL MATTERS MORE THAN THE NUMBER. Scenes genuinely differ -- a jammed
intersection really is slower than an empty arterial -- so behaviour itself has
a non-zero scene ICC. A role ICC of 0.6 means nothing on its own; it means a
lot if speed's ICC is 0.3, because then the role is MORE scene-determined than
the behaviour it exists to explain. Every behaviour metric in METRICS is
therefore run through the identical estimator and reported side by side.

Estimator: one-way random-effects ICC(1) with the unequal-group-size
correction (n0), scenes as groups. Roles are collapsed to one row per
(scenario, vehicle) BEFORE the decomposition -- the same map is re-dealt across
episodes and duplicate scene instances share a scenario_id, so leaving the
repeats in would count near-identical copies as independent within-scene
observations and bias ICC upward (the pseudoreplication issue).

No forcing, no cluster file needed: passes an empty regime map to
rollout_condition, which makes select_focals return nothing and the rollout
purely natural.

Usage (from /scratch/<user>/PufferDrive):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/<user>/PufferDrive \
    python $HOME/roma_pufferdrive/role_scene_icc.py \
        --checkpoint /scratch/<user>/checkpoints/roma_baseline_dim4/roma_dim4_final.pt \
        --data_dir   pufferlib/resources/drive/binaries/training \
        --out_dir    /scratch/<user>/role_icc/dim4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import load_policy
from role_paired_sweep import (
    load_drive_config, _squeeze, focal_metrics, rollout_condition,
    T, TELEPORT_M, METRICS,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--episodes",   type=int, default=3,
                   help="Map-pool draws. Each adds scenes; 3 is plenty since "
                        "ICC is estimated over thousands of vehicles.")
    p.add_argument("--num_agents", type=int, default=3072)
    p.add_argument("--num_maps",   type=int, default=10000)
    p.add_argument("--min_agents", type=int, default=2,
                   help="Scenes with fewer controlled vehicles than this carry "
                        "no within-scene information and are dropped.")
    p.add_argument("--device",     type=str, default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# ICC(1), one-way random effects, unequal group sizes
# ---------------------------------------------------------------------------

def icc1(values, groups):
    """
    values : (N,) float
    groups : (N,) group label per value (scenario id)

    Returns dict with icc, between/within variance components, k groups, N.

    Unequal group sizes need the n0 correction -- using a plain mean group size
    biases the estimate when a few scenes contribute many more vehicles than
    the rest, which is exactly the case here (scene occupancy varies a lot).
    ICC can come out slightly negative when between-scene variance is at or
    below noise; that is a real "no scene effect" answer, so it is reported
    raw and only clamped for the figure.
    """
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]

    uniq, inv = np.unique(groups, return_inverse=True)
    k = len(uniq)
    N = len(values)
    if k < 2 or N - k < 1:
        return {"icc": np.nan, "var_between": np.nan, "var_within": np.nan,
                "k_groups": k, "n_obs": N}

    counts = np.bincount(inv, minlength=k).astype(np.float64)
    sums   = np.bincount(inv, weights=values, minlength=k)
    gmeans = sums / counts
    grand  = values.mean()

    ss_between = float((counts * (gmeans - grand) ** 2).sum())
    ss_within  = float(((values - gmeans[inv]) ** 2).sum())
    ms_between = ss_between / (k - 1)
    ms_within  = ss_within / (N - k)

    n0 = (N - (counts ** 2).sum() / N) / (k - 1)
    denom = ms_between + (n0 - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom > 0 else np.nan

    # Variance components on a comparable scale
    var_between = max((ms_between - ms_within) / n0, 0.0)
    return {"icc": float(icc), "var_between": var_between,
            "var_within": float(ms_within), "k_groups": int(k), "n_obs": int(N)}


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect(env, policy, device, episodes):
    """Natural rollouts -> one row per (episode, scenario, vehicle) with the
    agent's mean role vector over its pre-teleport window + its behaviour."""
    rows = []
    for ep in range(episodes):
        if ep > 0:
            env.resample_maps()
        # Empty regime map -> select_focals returns [] -> nobody is forced.
        _, rl, xs, ys, hs, rews, gt = rollout_condition(env, policy, device, {}, None)

        sids   = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
        vids   = _squeeze(np.asarray(gt["id"])).reshape(-1)
        gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
        gvalid = _squeeze(gt["valid"]).astype(bool)
        is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
        T_gt   = gx.shape[1]
        step_d = np.hypot(np.diff(xs, axis=0), np.diff(ys, axis=0))

        n_ep = 0
        for a in range(env.num_agents):
            if not is_veh[a]:
                continue
            sid = str(sids[a])
            # Padding slots carry a synthetic "map*" id and no real scenario.
            if not sid or sid.lower().startswith("map"):
                continue
            m = focal_metrics(a, xs, ys, hs, rews, gx, gy, gvalid, T_gt)
            if m is None:
                continue
            jumps = np.where(step_d[:, a] > TELEPORT_M)[0]
            t_end = min(int(jumps[0] + 1) if len(jumps) else T, T_gt)
            role  = rl[:t_end, a].mean(axis=0)

            row = {"episode": ep, "sid": sid, "vid": int(vids[a])}
            row.update({f"role_{i}": float(role[i]) for i in range(role.shape[0])})
            row.update(m)
            rows.append(row)
            n_ep += 1
        print(f"[icc] episode {ep+1}/{episodes}: {n_ep} vehicles", flush=True)
    return rows


def main():
    args = parse_args()
    import pandas as pd
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    from pufferlib.ocean.drive.drive import Drive
    env_cfg = dict(load_drive_config()["env"])
    env_cfg.update({"num_maps": args.num_maps, "num_agents": args.num_agents,
                    "map_dir": args.data_dir})
    env = Drive(**env_cfg)
    obs_np, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs_np.shape[-1], device)
    if role_dim == 0:
        raise SystemExit("role_dim=0 checkpoint -- no role to decompose")
    print(f"[icc] role_dim={role_dim} checkpoint={args.checkpoint}")

    df = pd.DataFrame(collect(env, policy, device, args.episodes))
    if df.empty:
        raise SystemExit("no vehicles collected")
    role_cols = [f"role_{i}" for i in range(role_dim)]
    beh_cols  = [c for c in METRICS if c in df.columns]

    # --- Collapse repeats: one row per (scenario, vehicle) ------------------
    # The same map is re-dealt across episodes, and a single episode can hold
    # several duplicate instances of one scenario. Both produce near-identical
    # copies of the same vehicle; counted separately they would masquerade as
    # independent within-scene observations and push ICC up.
    n_raw = len(df)
    df = df.groupby(["sid", "vid"], as_index=False)[role_cols + beh_cols].mean()
    print(f"[icc] {n_raw} rows -> {len(df)} unique (scenario, vehicle) "
          f"after collapsing {n_raw - len(df)} repeats")

    # --- Keep scenes that carry within-scene information -------------------
    sizes = df.groupby("sid")["vid"].transform("size")
    dropped_scenes = int((sizes < args.min_agents).groupby(df["sid"]).first().sum())
    df = df[sizes >= args.min_agents].copy()
    n_scenes = df["sid"].nunique()
    print(f"[icc] {n_scenes} scenes with >={args.min_agents} vehicles "
          f"({dropped_scenes} single-vehicle scenes dropped), "
          f"{len(df)} vehicles, mean {len(df)/max(n_scenes,1):.1f} per scene")
    if n_scenes < 2:
        raise SystemExit("need >=2 multi-vehicle scenes for a variance decomposition")

    # --- PCA of the role space; PC1/PC2 are the axes every sweep forces ----
    R   = df[role_cols].values.astype(np.float64)
    cen = R - R.mean(axis=0)
    _, svals, vt = np.linalg.svd(cen, full_matrices=False)
    evr = (svals ** 2) / (svals ** 2).sum()
    n_pc = min(2, role_dim)
    for d in range(n_pc):
        df[f"pc{d+1}"] = cen @ vt[d]
    print(f"[icc] role PCA explained var: "
          f"{['%.0f%%' % (100*e) for e in evr[:n_pc]]}")

    # --- The decomposition -------------------------------------------------
    results = []
    for col, kind in ([(c, "role_dim") for c in role_cols]
                      + [(f"pc{d+1}", "role_pc") for d in range(n_pc)]
                      + [(c, "behaviour") for c in beh_cols]):
        r = icc1(df[col].values, df["sid"].values)
        r.update({"quantity": col, "kind": kind})
        results.append(r)

    # Multivariate summary: pooled across dims, so it is not hostage to which
    # single dim happens to carry the variance.
    tot_b = sum(r["var_between"] for r in results if r["kind"] == "role_dim")
    tot_w = sum(r["var_within"]  for r in results if r["kind"] == "role_dim")
    role_icc_all = tot_b / (tot_b + tot_w) if (tot_b + tot_w) > 0 else np.nan
    results.append({"quantity": "role_ALL_DIMS", "kind": "role_summary",
                    "icc": role_icc_all, "var_between": tot_b,
                    "var_within": tot_w, "k_groups": n_scenes, "n_obs": len(df)})

    res = pd.DataFrame(results)[["kind", "quantity", "icc", "var_between",
                                 "var_within", "k_groups", "n_obs"]]
    res.to_csv(out / "role_scene_icc.csv", index=False)
    df.to_csv(out / "role_scene_icc_agents.csv", index=False)

    # --- Report ------------------------------------------------------------
    print("\n" + "=" * 66)
    print("  SCENE ICC  (1.0 = role is the map, 0.0 = role is the driver)")
    print("=" * 66)
    for kind, label in [("role_dim", "role dims"), ("role_pc", "role PCs"),
                        ("role_summary", "ROLE OVERALL"), ("behaviour", "behaviour (control)")]:
        sub = res[res["kind"] == kind]
        if sub.empty:
            continue
        print(f"\n  {label}")
        for _, r in sub.iterrows():
            print(f"    {r['quantity']:<16} ICC = {r['icc']:+.3f}   "
                  f"(between {r['var_between']:.4g} / within {r['var_within']:.4g})")

    beh_icc = res[res["kind"] == "behaviour"]["icc"]
    beh_med = float(np.nanmedian(beh_icc)) if len(beh_icc) else np.nan
    print("\n" + "-" * 66)
    print(f"  role overall ICC      : {role_icc_all:+.3f}")
    print(f"  behaviour median ICC  : {beh_med:+.3f}   <- the honest reference")
    if np.isfinite(role_icc_all) and np.isfinite(beh_med):
        if role_icc_all > beh_med + 0.10:
            print("  VERDICT: the role is MORE scene-determined than the behaviour it\n"
                  "           explains. Map-based-role diagnosis confirmed.")
        elif role_icc_all < beh_med - 0.10:
            print("  VERDICT: the role varies WITHIN scenes more than behaviour does --\n"
                  "           it is agent-based already. The premise does not hold.")
        else:
            print("  VERDICT: role tracks behaviour's own scene structure. Not obviously\n"
                  "           map-driven; whatever caps per-scene consistency is elsewhere.")
    print("-" * 66)

    # --- Figure ------------------------------------------------------------
    plot = res[res["kind"] != "role_summary"]
    colors = {"role_dim": "#4c72b0", "role_pc": "#dd8452", "behaviour": "#55a868"}
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(plot)), 4.4))
    ax.bar(range(len(plot)), np.clip(plot["icc"].values, 0, 1),
           color=[colors[k] for k in plot["kind"]])
    if np.isfinite(role_icc_all):
        ax.axhline(role_icc_all, ls="--", lw=1.4, color="#4c72b0",
                   label=f"role overall {role_icc_all:.2f}")
    if np.isfinite(beh_med):
        ax.axhline(beh_med, ls="--", lw=1.4, color="#55a868",
                   label=f"behaviour median {beh_med:.2f}")
    ax.set_xticks(range(len(plot)))
    ax.set_xticklabels(plot["quantity"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("scene ICC")
    ax.set_ylim(0, 1)
    ax.set_title(f"How much of each quantity is the scene?  "
                 f"({n_scenes} scenes, {len(df)} vehicles)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "role_scene_icc.png", dpi=140)
    print(f"\n[icc] wrote role_scene_icc.csv / .png / _agents.csv -> {out}")


if __name__ == "__main__":
    main()
