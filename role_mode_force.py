"""
role_mode_force.py -- Test 3: the CAUSAL two-centroid forcing for a bimodal
role. Instead of a continuous alpha-sweep (meaningless for a discrete latent --
the intermediate alphas land in the empty valley between the two clumps), this
CLAMPS the focal's role to mode-A's centroid vs mode-B's centroid on the SAME
(map, vehicle), and compares. No interpolation through the hole; scene held
fixed, so no scene confound at all.

Centroids come from role_paired_warmup.csv: split its natural roles into two
modes along PC1 (KMeans k=2), then each centroid = that mode's MEAN full role
vector. Forcing is a constant clamp (z_focal = centroid, not z_nat + alpha*sig).

Per episode the SAME map pool serves both conditions (+ an optional natural
reference), so every focal is paired by (episode, sid, vid) with identical
action noise. Output:
  role_mode_force_agent.csv   per (episode, sid, vid, cond) behaviour
  role_mode_force_tests.csv   paired (A - B) per metric: mean, t, n
  printed verdict incl. the SAFETY signature (fast mode riskier => aggressive/
  cautious types; fast mode safer => competence/degradation)

Run from PufferDrive (CPU fine):
  python role_mode_force.py \
      --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim1/roma_dim1_final.pt \
      --warmup_csv /scratch/e452103/role_paired/dim1/role_paired_warmup.csv \
      --clusters   /scratch/e452103/map_atlas/k4/map_clusters.csv \
      --data_dir   pufferlib/resources/drive/binaries/training \
      --out_dir    /scratch/e452103/role_paired/dim1
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import load_policy
from role_regime_analysis import _squeeze
from role_paired_sweep import (T, ROLLOUT_SEED, select_focals, focal_metrics)

METRICS = ["speed_mean", "accel_abs", "accel_pos", "decel_abs", "jerk_abs",
           "turn_abs", "event_rate"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--warmup_csv", type=str, required=True,
                   help="role_paired_warmup.csv (has role_* + pc1) for centroids")
    p.add_argument("--clusters",   type=str, required=True,
                   help="map_clusters.csv -> which scenes to draw focals from")
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--episodes",   type=int, default=8)
    p.add_argument("--num_agents", type=int, default=2048)
    p.add_argument("--num_maps",   type=int, default=10000)
    p.add_argument("--min_margin", type=float, default=1.5)
    p.add_argument("--with_natural", type=int, default=1,
                   help="1 = also roll the unforced natural condition as a ref")
    p.add_argument("--device",     type=str, default="cpu")
    return p.parse_args()


def rollout_clamp(env, policy, device, regime_of, clamp_vec):
    """Like role_paired_sweep.rollout_condition but CONSTANT-CLAMPS each focal's
    role to clamp_vec (z_focal = clamp_vec) instead of shifting it. clamp_vec
    None = natural (nobody forced). Returns rows keyed by (sid, vid)."""
    B = env.num_agents
    obs_np, _ = env.reset()
    gt = env.get_ground_truth_trajectories()
    focals = select_focals(gt, regime_of)
    slots  = np.array([f[0] for f in focals], dtype=int)

    cv = (None if clamp_vec is None else
          torch.as_tensor(clamp_vec, dtype=torch.float32, device=device))
    torch.manual_seed(ROLLOUT_SEED)
    xs = np.zeros((T, B), np.float32); ys = np.zeros((T, B), np.float32)
    hs = np.zeros((T, B), np.float32); rews = np.zeros((T, B), np.float32)

    obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    state = policy.initial_state(B, device)
    for t in range(T):
        ag = env.get_global_agent_state()
        xs[t], ys[t], hs[t] = ag["x"], ag["y"], ag["heading"]
        with torch.no_grad():
            if cv is None or len(slots) == 0:
                logits, _, state, _ = policy(obs, state)
            else:
                _, _, _, ri0 = policy(obs, state)       # pass 1: natural roles
                forced = ri0["role_z"].clone()
                forced[slots] = cv                        # CONSTANT clamp
                logits, _, state, _ = policy(obs, state, forced_role=forced)
        action = Categorical(logits=logits.float()).sample()
        obs_np, rew_np, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
        rews[t] = np.asarray(rew_np).reshape(B)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

    gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
    gvalid = _squeeze(gt["valid"]).astype(bool)
    T_gt   = gx.shape[1]
    rows = []
    for slot, sid, vid, rg in focals:
        m = focal_metrics(slot, xs, ys, hs, rews, gx, gy, gvalid, T_gt)
        if m is None:
            continue
        m.update({"sid": sid, "vid": vid})
        rows.append(m)
    return rows


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    from sklearn.cluster import KMeans

    # -- centroids from the warmup role distribution ---------------------------
    w = pd.read_csv(args.warmup_csv)
    role_cols = sorted([c for c in w.columns if c.startswith("role_")],
                       key=lambda c: int(c.split("_")[1]))
    if not role_cols:
        raise SystemExit(f"{args.warmup_csv} has no role_* columns")
    split_on = "pc1" if "pc1" in w.columns else role_cols[0]
    lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(
        w[split_on].values.reshape(-1, 1))
    if w[split_on].values[lab == 0].mean() > w[split_on].values[lab == 1].mean():
        lab = 1 - lab                                    # mode 0 = lower PC1
    cA = w[role_cols][lab == 0].mean().values.astype(np.float32)   # low-PC1 mode
    cB = w[role_cols][lab == 1].mean().values.astype(np.float32)   # high-PC1 mode
    print(f"[modeforce] centroids from {len(w)} warmup agents "
          f"(n_A={int((lab==0).sum())}, n_B={int((lab==1).sum())})")
    print(f"[modeforce]   mode A (low PC1)  centroid = {np.round(cA,2)}")
    print(f"[modeforce]   mode B (high PC1) centroid = {np.round(cB,2)}")

    # -- env + policy + focal scenes -------------------------------------------
    cl = pd.read_csv(args.clusters)
    cl = cl[cl["margin"] > args.min_margin]
    regime_of = dict(zip(cl["scenario_id"].astype(str), cl["cluster"]))

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(num_maps=args.num_maps, num_agents=args.num_agents,
                map_dir=args.data_dir, episode_length=T, goal_speed=100, seed=1)
    obs_np, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs_np.shape[-1], device)
    if role_dim != len(role_cols):
        raise SystemExit(f"checkpoint role_dim={role_dim} != warmup role cols "
                         f"{len(role_cols)} -- wrong pairing")

    conds = [("modeA", cA), ("modeB", cB)]
    if args.with_natural:
        conds.append(("natural", None))

    all_rows = []
    for ep in range(args.episodes):
        env.resample_maps()                              # same pool serves all conds
        for cname, cvec in conds:
            rows = rollout_clamp(env, policy, device, regime_of, cvec)
            for r in rows:
                r.update({"episode": ep, "cond": cname})
            all_rows.extend(rows)
            print(f"[modeforce] ep{ep} {cname:>8}: {len(rows)} focals", flush=True)
    env.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "role_mode_force_agent.csv", index=False)
    metrics = [m for m in METRICS if m in df.columns]

    # -- paired A vs B on the SAME (episode, sid, vid) -------------------------
    key = ["episode", "sid", "vid"]
    a = df[df["cond"] == "modeA"].set_index(key)
    b = df[df["cond"] == "modeB"].set_index(key)
    j = a.join(b, how="inner", lsuffix="_A", rsuffix="_B")
    print(f"\n=== CAUSAL two-centroid contrast: mode B - mode A, "
          f"SAME (map, vehicle), n={len(j)} paired ===")
    tests = []
    for m in metrics:
        d = (j[f"{m}_B"] - j[f"{m}_A"]).dropna().values
        if len(d) < 5:
            continue
        mean = float(d.mean()); se = float(d.std(ddof=1)/np.sqrt(len(d)))
        t = mean/se if se > 0 else np.nan
        tests.append({"metric": m, "meanB_minus_A": round(mean, 4),
                      "paired_t": round(float(t), 1), "n": len(d)})
    tdf = pd.DataFrame(tests)
    print(tdf.to_string(index=False))
    tdf.to_csv(out / "role_mode_force_tests.csv", index=False)

    # -- verdict: are the modes causally distinct, and how? --------------------
    def diff(m):
        r = tdf[tdf["metric"] == m]
        return (float(r["meanB_minus_A"].iloc[0]), float(r["paired_t"].iloc[0])) \
            if len(r) else (np.nan, np.nan)
    ds, ts = diff("speed_mean")
    de, te = diff("event_rate")
    print("\n=== VERDICT ===")
    if np.isfinite(ts) and abs(ts) > 3:
        faster = "B" if ds > 0 else "A"
        print(f"modes ARE causally distinct on speed (mode {faster} faster by "
              f"{abs(ds):.2f} m/s, t={ts:+.1f}).")
        if np.isfinite(te) and abs(te) > 2:
            # does the faster mode also crash more (aggressive) or less (competent)?
            faster_more_events = (de > 0) == (ds > 0)
            print("  safety: faster mode has "
                  + ("MORE" if faster_more_events else "FEWER")
                  + f" events (event dB-A={de:+.2f}, t={te:+.1f}) => "
                  + ("AGGRESSIVE vs CAUTIOUS types"
                     if faster_more_events else "COMPETENT vs DEGRADED"))
        else:
            print(f"  safety: event diff not significant (t={te:+.1f}) -- the "
                  f"modes differ in speed/style but not risk.")
    else:
        print(f"modes NOT causally distinct on speed (t={ts:+.1f}). Clamping to "
              f"either centroid gives ~the same driving -> the bimodal split "
              f"was NOT a behavioural type (likely an encoder/scene artifact).")
    print(f"\n[modeforce] wrote role_mode_force_agent.csv + "
          f"role_mode_force_tests.csv -> {out}")


if __name__ == "__main__":
    main()
