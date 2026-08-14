"""
role_causal_axis.py -- which direction in role space actually CONTROLS behaviour?

THE QUESTION
------------
Every sweep so far has forced PC1/PC2 -- the top-variance directions of the
role distribution. But PCA runs on the ENCODER'S OUTPUT (how much agents differ
from each other), while the causal effect lives in the POLICY'S WEIGHTS (how
much the GRU reads each role direction). Those are different matrices and there
is no reason their principal directions coincide.

Worse, the measured per-PC scene ICCs say PC1 is the MOST scene-determined axis
in every arm (0.33-0.51 vs 0.13-0.30 for PC2). Agents differ most along the
direction that encodes which map they are in, because maps vary more than
driving styles do. So the top-variance axis is systematically the map axis --
the worst possible choice for a behaviour dial.

WHAT THIS MEASURES
------------------
The causal Jacobian of behaviour with respect to the role, estimated by
randomised intervention rather than by correlation:

  1. Roll the natural policy on a map pool  -> baseline behaviour b_i per focal.
  2. Roll the SAME map pool again, but give EACH focal its OWN random role
     shift  d_i = alpha_i * sigma * u_i   (u_i a random unit direction,
     alpha_i ~ U(-a, a)).                 -> perturbed behaviour b'_i.
  3. Least-squares fit, per metric:   (b'_i - b_i)  ~  d_i
     The fitted coefficient vector g is the direction in role space that moves
     that metric most per unit of role movement. That IS the causal axis.

Because every focal gets a DIFFERENT random shift, one perturbed rollout
identifies the whole gradient -- no need to sweep each direction separately.
And because it is paired against the natural rollout on the same (map,
vehicle), map noise cancels exactly as in role_paired_sweep.

This is NOT the behaviour-fitted direction that was rejected earlier. That one
was fitted to OBSERVATIONAL correlation across agents and was confounded by
scene. This is fitted to the measured RESPONSE TO AN INTERVENTION, which is an
experiment, not a correlation.

WHAT IT REPORTS
---------------
  cos_pc1 / cos_pc2   alignment of the causal direction with the swept axes.
                      |cos| is exactly the fraction of the achievable effect
                      that sweeping that PC delivers -- cos = 0.3 means PC1 is
                      a 30%-efficient dial, and a NEGATIVE cos means sweeping
                      PC1 moves the metric the WRONG WAY.
  gain_ratio          |effect along the causal axis| / |effect along PC1|.
  r2                  how linear the response is. Low r2 = the role acts
                      non-linearly and no single direction is a good dial.
  obs_r               the observational correlation, for contrast.

  Plus a POLICY SENSITIVITY report: the mean per-input-dimension weight norm
  the policy GRU assigns to the role slice vs the env-embedding slice. If the
  role slice is much smaller, the policy is barely reading the role at all and
  no choice of direction will give a strong dial -- that is an architecture
  problem, not an axis problem.

Usage (from the PufferDrive dir, like role_paired_sweep):
    python role_causal_axis.py \
        --checkpoint /scratch/$USER/checkpoints/roma_x/roma_dim4_final.pt \
        --data_dir   pufferlib/resources/drive/binaries/training \
        --out_dir    /scratch/$USER/analysis/x/causal_axis
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.distributions import Categorical

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import load_policy
from role_paired_sweep import (load_drive_config, _squeeze, focal_metrics,
                               select_focals, T, TELEPORT_M, ROLLOUT_SEED)
from traj_kinematics import METRICS, PLOT_METRICS


class AllScenes:
    """select_focals filters by regime; here every scene is eligible."""
    def get(self, key, default=None):
        return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--warmup_episodes", type=int, default=2,
                   help="natural rollouts used to estimate mu / sigma / PCs")
    p.add_argument("--episodes",   type=int, default=6,
                   help="paired (natural, perturbed) map-pool draws")
    p.add_argument("--alpha_max",  type=float, default=2.0,
                   help="per-focal shift is alpha*sigma*u with alpha ~ U(-a, a)")
    p.add_argument("--num_agents", type=int, default=3072)
    p.add_argument("--num_maps",   type=int, default=10000)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


def rollout(env, policy, device, shifts_by_slot=None, all_agents=False):
    """One episode. shifts_by_slot: dict slot -> (role_dim,) shift added to
    that agent's OWN live role every step (None = natural).
    Returns rows keyed by (sid, vid) plus the per-agent mean role.

    all_agents=True additionally returns EVERY vehicle's (sid, role, behaviour).
    select_focals yields one focal per scene, so focal rows cannot be
    scene-centred -- each scene would have a single member and centring would
    return exactly zero. The observational gradient therefore needs the full
    per-scene population, the same rows role_scene_icc decomposes."""
    B = env.num_agents
    obs_np, _ = env.reset()
    gt = env.get_ground_truth_trajectories()
    focals = select_focals(gt, AllScenes())

    fv = None
    if shifts_by_slot:
        fv = torch.zeros((B, policy.role_dim), dtype=torch.float32, device=device)
        for slot, vec in shifts_by_slot.items():
            fv[slot] = torch.as_tensor(vec, dtype=torch.float32, device=device)

    torch.manual_seed(ROLLOUT_SEED)
    xs = np.zeros((T, B), np.float32); ys = np.zeros((T, B), np.float32)
    hs = np.zeros((T, B), np.float32); rews = np.zeros((T, B), np.float32)
    rl = np.zeros((T, B, policy.role_dim), np.float32)

    obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    state = policy.initial_state(B, device)
    for t in range(T):
        ag = env.get_global_agent_state()
        xs[t], ys[t], hs[t] = ag["x"], ag["y"], ag["heading"]
        with torch.no_grad():
            if fv is None:
                logits, _, state, ri = policy(obs, state)
            else:
                _, _, _, ri0 = policy(obs, state)          # pass 1: natural
                logits, _, state, ri = policy(
                    obs, state, forced_role=ri0["role_z"] + fv)
        if ri.get("role_mean") is not None and ri["role_mean"].shape[-1]:
            rl[t] = ri["role_mean"].float().cpu().numpy()
        action = Categorical(logits=logits.float()).sample()
        obs_np, rew_np, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
        rews[t] = np.asarray(rew_np).reshape(B)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

    gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
    gvalid = _squeeze(gt["valid"]).astype(bool)
    T_gt   = gx.shape[1]
    step_d = np.hypot(np.diff(xs, axis=0), np.diff(ys, axis=0))

    everyone = []
    if all_agents:
        sids_all = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
        is_veh   = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
        for a in range(B):
            sid_a = str(sids_all[a])
            if not is_veh[a] or not sid_a or sid_a.lower().startswith("map"):
                continue
            m = focal_metrics(a, xs, ys, hs, rews, gx, gy, gvalid, T_gt)
            if m is None:
                continue
            jm = np.where(step_d[:, a] > TELEPORT_M)[0]
            te = min(int(jm[0] + 1) if len(jm) else T, T_gt)
            everyone.append((sid_a, rl[:te, a].mean(axis=0), m))

    rows = {}
    for slot, sid, vid, _rg in focals:
        m = focal_metrics(slot, xs, ys, hs, rews, gx, gy, gvalid, T_gt)
        if m is None:
            continue
        jumps = np.where(step_d[:, slot] > TELEPORT_M)[0]
        t_end = min(int(jumps[0] + 1) if len(jumps) else T, T_gt)
        m["_role"] = rl[:t_end, slot].mean(axis=0)
        m["_slot"] = slot
        rows[(sid, vid)] = m
    return (rows, focals, everyone) if all_agents else (rows, focals)


def policy_role_sensitivity(policy):
    """How much input weight does the GRU give the role vs the env embedding?
    weight_ih is (3*hidden, env_dim + role_dim); compare the mean per-input-dim
    column norm of each slice. This is an architecture readout, not a result:
    a tiny role share means no direction can be a strong dial."""
    gru = getattr(policy, "policy_gru", None)
    if gru is None or not hasattr(gru, "weight_ih"):
        return None
    W = gru.weight_ih.detach().float().cpu().numpy()      # (3H, in_dim)
    D = policy.role_dim
    if D == 0 or W.shape[1] <= D:
        return None
    col = np.linalg.norm(W, axis=0)                       # per input dim
    env_c, role_c = col[:-D], col[-D:]
    return {"env_mean": float(env_c.mean()), "role_mean": float(role_c.mean()),
            "ratio": float(role_c.mean() / (env_c.mean() + 1e-12)),
            "role_per_dim": role_c.tolist(),
            "n_env": int(len(env_c)), "n_role": int(D)}


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    import pandas as pd

    from pufferlib.ocean.drive.drive import Drive
    cfg = dict(load_drive_config()["env"])
    cfg.update({"num_maps": args.num_maps, "num_agents": args.num_agents,
                "map_dir": args.data_dir})
    env = Drive(**cfg)
    obs_np, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs_np.shape[-1], device)
    if role_dim < 2:
        raise SystemExit(f"role_dim={role_dim}: with a single axis there is no "
                         f"direction to choose -- PC1 IS the only direction.")

    # ---- policy sensitivity: is the role even being read? -----------------
    sens = policy_role_sensitivity(policy)
    print("=" * 68)
    print("  POLICY SENSITIVITY TO THE ROLE  (architecture readout)")
    print("=" * 68)
    if sens is None:
        print("  could not read policy_gru.weight_ih -- skipped")
    else:
        print(f"  mean |w| per env-embedding input dim : {sens['env_mean']:.4f}"
              f"  ({sens['n_env']} dims)")
        print(f"  mean |w| per ROLE input dim          : {sens['role_mean']:.4f}"
              f"  ({sens['n_role']} dims)")
        print(f"  role / env ratio                     : {sens['ratio']:.3f}")
        if sens["ratio"] < 0.5:
            print("  -> the policy weights the role BELOW the env embedding per "
                  "input dim.\n     No choice of direction fixes that; it is an "
                  "architecture/objective issue.")
        else:
            print("  -> the policy does read the role at a comparable weight "
                  "per dim.\n     A weak dial would then be about WHICH "
                  "direction is swept, not whether\n     the role is used.")

    # ---- warmup: natural roles -> mu, sigma, PC1/PC2 ----------------------
    vecs, pop = [], []
    for ep in range(args.warmup_episodes):
        if ep > 0:
            env.resample_maps()
        rows, _, everyone = rollout(env, policy, device, None, all_agents=True)
        vecs.extend(r["_role"] for r in rows.values())
        pop.extend(everyone)
        print(f"[causal] warmup {ep+1}/{args.warmup_episodes}: "
              f"{len(vecs)} focals", flush=True)
    R = np.asarray(vecs, dtype=np.float64)
    mu = R.mean(axis=0)
    cen = R - mu
    _, sv, vt = np.linalg.svd(cen, full_matrices=False)
    evr = (sv ** 2) / (sv ** 2).sum()
    sigma = float(cen.std())
    pc1, pc2 = vt[0], vt[1]
    print(f"[causal] role sigma={sigma:.4f}  PC1 var={100*evr[0]:.0f}%  "
          f"PC2 var={100*evr[1]:.0f}%")

    # ---- observational gradient, SCENE-CENTRED --------------------------
    # o_j = how the role covaries with metric j among agents in the SAME scene.
    # Comparing it with the causal gradient g_j gives the coherence measure:
    # does the role MEAN what it DOES? Scene-centring removes the between-scene
    # confound, so a disagreement here is not the map.
    o_hat = {}
    if len(pop) > 50:
        p_sid = np.array([q[0] for q in pop])
        p_rol = np.asarray([q[1] for q in pop], dtype=np.float64)
        p_beh = np.asarray([[q[2].get(m, np.nan) for m in METRICS]
                            for q in pop], dtype=np.float64)
        uniq, inv = np.unique(p_sid, return_inverse=True)
        counts = np.bincount(inv, minlength=len(uniq))
        keep = counts[inv] >= 2                 # a lone agent carries no
        p_rol, p_beh, inv = p_rol[keep], p_beh[keep], inv[keep]
        # subtract each scene's own mean from role and behaviour
        def _center(M):
            sums = np.zeros((len(uniq), M.shape[1]))
            np.add.at(sums, inv, np.nan_to_num(M))
            cnt = np.bincount(inv, minlength=len(uniq)).astype(float)[:, None]
            return M - (sums / np.maximum(cnt, 1))[inv]
        Rc = _center(p_rol)
        for j, met in enumerate(METRICS):
            y = p_beh[:, j]
            ok = np.isfinite(y)
            if ok.sum() < 50:
                continue
            yc = _center(y[:, None].copy())[:, 0]
            if not np.isfinite(yc[ok]).all() or yc[ok].std() < 1e-12:
                continue
            ov, *_ = np.linalg.lstsq(Rc[ok], yc[ok], rcond=None)
            n = np.linalg.norm(ov)
            if n > 1e-12:
                o_hat[met] = ov / n
        print(f"[causal] observational gradient from {len(p_rol)} agents in "
              f"multi-agent scenes ({len(o_hat)} metrics)")
    else:
        print("[causal] too few agents for a within-scene observational "
              "gradient -- coherence will be blank")

    # ---- paired randomised intervention -----------------------------------
    d_rows, y_rows, obs_role, obs_beh = [], [], [], []
    for ep in range(args.episodes):
        env.resample_maps()
        base, focals = rollout(env, policy, device, None)
        # one INDEPENDENT random shift per focal slot
        shifts = {}
        for slot, sid, vid, _ in focals:
            u = rng.normal(size=role_dim)
            u /= np.linalg.norm(u) + 1e-12
            a = rng.uniform(-args.alpha_max, args.alpha_max)
            shifts[slot] = a * sigma * u
        pert, _ = rollout(env, policy, device, shifts)

        for key, b0 in base.items():
            b1 = pert.get(key)
            if b1 is None or b0["_slot"] not in shifts:
                continue
            d_rows.append(shifts[b0["_slot"]])
            y_rows.append([b1.get(m, np.nan) - b0.get(m, np.nan) for m in METRICS])
            obs_role.append(b0["_role"])
            obs_beh.append([b0.get(m, np.nan) for m in METRICS])
        print(f"[causal] ep{ep+1}/{args.episodes}: "
              f"{len(d_rows)} paired interventions", flush=True)
    env.close()

    D = np.asarray(d_rows)                     # (N, role_dim) applied shifts
    Y = np.asarray(y_rows, dtype=np.float64)   # (N, n_metrics) behaviour deltas
    OR = np.asarray(obs_role); OB = np.asarray(obs_beh, dtype=np.float64)
    if len(D) < 50:
        raise SystemExit(f"only {len(D)} paired interventions -- too few to fit")
    print(f"\n[causal] fitting on {len(D)} paired interventions, "
          f"role_dim={role_dim}")

    res = []
    for j, met in enumerate(METRICS):
        y = Y[:, j]
        ok = np.isfinite(y) & np.isfinite(D).all(axis=1)
        if ok.sum() < 50 or np.nanstd(y[ok]) < 1e-12:
            continue
        Dm, ym = D[ok], y[ok]
        g, *_ = np.linalg.lstsq(Dm, ym, rcond=None)      # causal gradient
        pred = Dm @ g
        ss_res = float(((ym - pred) ** 2).sum())
        ss_tot = float(((ym - ym.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            continue
        u_g = g / gn
        c1, c2 = float(u_g @ pc1), float(u_g @ pc2)
        # observational correlation of the PC1 projection with this metric
        p1 = (OR - mu) @ pc1
        m_ok = np.isfinite(OB[:, j])
        r_obs = (float(np.corrcoef(p1[m_ok], OB[m_ok, j])[0, 1])
                 if m_ok.sum() > 30 and OB[m_ok, j].std() > 0 else np.nan)
        res.append({
            "metric": met, "r2": r2,
            "effect_causal_per_sigma": gn * sigma,
            "effect_pc1_per_sigma": float(g @ pc1) * sigma,
            "effect_pc2_per_sigma": float(g @ pc2) * sigma,
            "cos_pc1": c1, "cos_pc2": c2,
            "gain_vs_pc1": abs(gn) / (abs(float(g @ pc1)) + 1e-12),
            "obs_r_pc1": r_obs, "n": int(ok.sum()),
            # cos(causal, observational-within-scene): +1 the role means what
            # it does; -1 it means the exact opposite of what it does.
            "coherence": (float(u_g @ o_hat[met]) if met in o_hat else np.nan),
            **{f"g{i}": float(u_g[i]) for i in range(role_dim)},
        })
    rdf = pd.DataFrame(res)
    rdf.to_csv(out / "role_causal_axis.csv", index=False)

    show = [m for m in PLOT_METRICS if m in set(rdf["metric"])]
    sub = rdf[rdf["metric"].isin(show)].sort_values(
        "effect_causal_per_sigma", ascending=False)
    print("\n" + "=" * 92)
    print("  CAUSAL AXIS vs SWEPT AXIS   (per 1 sigma of role movement)")
    print("=" * 92)
    print(f"  {'metric':22}{'causal':>9}{'via PC1':>9}{'cos_pc1':>9}"
          f"{'cos_pc2':>9}{'gain':>7}{'r2':>7}{'obs r':>8}{'cohere':>8}")
    for _, r in sub.iterrows():
        print(f"  {r['metric']:22}{r['effect_causal_per_sigma']:9.3f}"
              f"{r['effect_pc1_per_sigma']:9.3f}{r['cos_pc1']:9.2f}"
              f"{r['cos_pc2']:9.2f}{r['gain_vs_pc1']:7.1f}x{r['r2']:7.2f}"
              f"{r['obs_r_pc1']:8.2f}{r['coherence']:8.2f}")
    print("\n  cos_pc1 is the fraction of the achievable effect that sweeping "
          "PC1 delivers.\n  NEGATIVE cos_pc1 = sweeping PC1 moves that metric "
          "the WRONG WAY.\n  Low r2 = the role acts non-linearly; no single "
          "direction is a good dial.")

    # ---- figure -----------------------------------------------------------
    if len(sub):
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
        y = np.arange(len(sub))
        axs[0].barh(y, sub["cos_pc1"], color=["#cc3311" if v < 0 else "#4477aa"
                                              for v in sub["cos_pc1"]])
        axs[0].set_yticks(y); axs[0].set_yticklabels(sub["metric"], fontsize=8)
        axs[0].axvline(0, color="k", lw=.8); axs[0].set_xlim(-1, 1)
        axs[0].set_xlabel("cos(causal axis, PC1)")
        axs[0].set_title("How good a dial is PC1?\n"
                         "red = sweeping PC1 moves it the WRONG WAY", fontsize=9)
        axs[0].grid(alpha=.3, axis="x")
        axs[1].barh(y, sub["r2"], color="#55a868")
        axs[1].set_yticks(y); axs[1].set_yticklabels([]); axs[1].set_xlim(0, 1)
        axs[1].set_xlabel("r2 of the linear causal fit")
        axs[1].set_title("Is the response linear at all?", fontsize=9)
        axs[1].grid(alpha=.3, axis="x")
        ttl = "Causal role axis vs swept PC1"
        if sens:
            ttl += f"   |   policy role/env input-weight ratio = {sens['ratio']:.2f}"
        fig.suptitle(ttl, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out / "role_causal_axis.png", dpi=150)
        plt.close(fig)

    if sens:
        pd.DataFrame([sens]).to_csv(out / "policy_role_sensitivity.csv",
                                    index=False)
    print(f"\n[causal] -> {out}/role_causal_axis.csv + .png"
          f" + policy_role_sensitivity.csv")


if __name__ == "__main__":
    main()
