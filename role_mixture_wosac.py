"""
role_mixture_wosac.py -- WOSAC for a MIXTURE of per-cluster policies, swept
along each policy's PC1.

Idea 2: every trajectory cluster has its own trained policy. Every VEHICLE is
driven by the policy for ITS trajectory cluster; the four policies interact in
one scene for the first time. Peds/cyclists are not clustered, so with
control_mode=control_vehicles they stay on the GT log.

PC1 SWEEP: the whole mixture is evaluated at five role-forcing levels along
PC1 -- alpha in {-1, -0.5, 0, +0.5, +1} x sigma -- so we can compare how the
mixture's realism/safety move as the roles are pushed. PC1 and sigma are
PER CLUSTER (each policy has its own role space and its own role_paired_axes.csv),
so an agent on policy k is shifted along policy k's OWN PC1:

    z_k(t) = z_k_natural(t) + alpha * sigma_k * PC1_k     (natural offset)

alpha=0 is exactly the natural mixture. A policy without an axes file (e.g. a
dedicated fallback) is never forced (natural at every alpha).

Cost -- SAME as a normal single-policy WOSAC per condition
----------------------------------------------------------
Agents are PARTITIONED by assigned policy: policy k runs on only its own agents
(a batch slice) each step, so the forward work per step is B rows total --
identical to one policy driving all B. The policy forward is fully per-agent
(no cross-agent op), so a slice yields bit-identical logits to the full batch.
The partition is fixed within a map batch (assignment is rebuilt only when maps
are re-dealt). Forced conditions (alpha != 0) use the two-pass natural-offset
scheme, so they cost 2x that -- alpha=0 is 1x.

Assignment (rebuilt per map batch from the GT). EVERY clustered trajectory has
a label -- no margin/confidence filter, a cluster is a cluster:
  - a vehicle in the trajectory clustering  -> that cluster's policy
  - peds/cyclists                            -> not policy-controlled (GT log,
    via control_vehicles); never clustered
  - a vehicle NOT in the clustering CSV ("unlabelled": dropped in feature
    extraction, usually barely-moving; control_vehicles' goal filter already
    drops parked cars) -> the FALLBACK policy. Reported; a small residual.

Metrics per alpha: the full WOSAC set (realism meta-score, kinematic /
interactive / map-based, every likelihood, ADE/minADE) PLUS the env's
collision_rate, offroad_rate and goal (completion) rate.

CAVEAT, on purpose: each cluster policy trained in a world where every OTHER
agent replays GT; in the mixture it drives among other LEARNED policies for the
first time. If the mixture underperforms one full-dataset policy, that shift is
the finding, not a bug.

Usage:
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/$USER/PufferDrive \
    python role_mixture_wosac.py \
        --ckpts /scratch/$USER/checkpoints/roma_cluster0_dim4/roma_dim4_final.pt,...,\
/scratch/$USER/checkpoints/roma_cluster3_dim4/roma_dim4_final_3B.pt \
        --axes  /scratch/$USER/role_paired/cluster0/role_paired_axes.csv,...,\
/scratch/$USER/role_paired/cluster3_3B/role_paired_axes.csv \
        --n_clusters 4 \
        --traj_clusters /scratch/$USER/traj_atlas/trajectory_clusters_stopfrac.csv \
        --data_dir pufferlib/resources/drive/binaries/training \
        --out_dir /scratch/$USER/wosac_mixture
"""

import argparse
import ast
import configparser
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import load_policy

T = 91
ENV_KEYS = ["score", "collision_rate", "offroad_rate", "completion_rate"]
WOSAC_KEYS = [
    "realism_meta_score", "kinematic_metrics", "interactive_metrics",
    "map_based_metrics", "ade", "min_ade",
    "likelihood_linear_speed", "likelihood_linear_acceleration",
    "likelihood_angular_speed", "likelihood_angular_acceleration",
    "likelihood_collision_indication", "likelihood_distance_to_nearest_object",
    "likelihood_time_to_collision", "likelihood_distance_to_road_edge",
    "likelihood_offroad_indication",
]


def load_drive_config():
    import pufferlib
    puffer_dir = os.path.dirname(pufferlib.__file__)
    p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    p.read([os.path.join(puffer_dir, "config", "default.ini"),
            os.path.join(puffer_dir, "config", "ocean", "drive.ini")])

    def _parse(v):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return {s: {k: _parse(v) for k, v in p[s].items()} for s in p.sections()}


def _squeeze(a):
    a = np.asarray(a)
    return a[:, 0] if a.ndim >= 2 and a.shape[1] == 1 else a


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", type=str, required=True,
                   help="Per-cluster checkpoints, comma-separated in cluster "
                        "order (cluster3 may be roma_dim4_final_3B.pt).")
    p.add_argument("--axes", type=str, required=True,
                   help="Per-cluster role_paired_axes.csv, comma-separated in "
                        "the SAME order as --ckpts. Supplies each policy's PC1 "
                        "direction + sigma for the sweep.")
    p.add_argument("--n_clusters",   type=int, default=4)
    p.add_argument("--traj_clusters", type=str, required=True)
    p.add_argument("--data_dir",     type=str, required=True)
    p.add_argument("--out_dir",      type=str, required=True)
    p.add_argument("--role_dim",     type=int, default=4)
    p.add_argument("--axis",         type=str, default="PC1")
    p.add_argument("--alphas",       type=str, default="-1,-0.5,0,0.5,1",
                   help="natural-offset multiples of sigma along the axis")
    p.add_argument("--fallback_ckpt", type=str, default="",
                   help="Policy for unlabelled/non-vehicle agents. Empty = use "
                        "the --fallback_cluster policy. A dedicated fallback is "
                        "never role-forced (no axes).")
    p.add_argument("--fallback_cluster", type=int, default=0)
    p.add_argument("--num_agents",   type=int, default=3072)
    p.add_argument("--num_maps",     type=int, default=10000)
    p.add_argument("--wosac_rollouts",    type=int, default=32)
    p.add_argument("--wosac_max_batches", type=int, default=70)
    p.add_argument("--control_mode", type=str, default="control_vehicles",
                   help="control_vehicles: only VEHICLES are policy-controlled, "
                        "peds/cyclists stay on the GT log.")
    p.add_argument("--device",       type=str, default="cuda")
    return p.parse_args()


def load_axis(axes_csv, role_dim, axis_name):
    """(unit_vector, sigma) for the named axis of a policy's role space."""
    import pandas as pd
    df = pd.read_csv(axes_csv)
    cols = [f"c{i}" for i in range(role_dim)]
    row = df[df["name"] == axis_name]
    if row.empty:
        raise SystemExit(f"{axes_csv} has no axis '{axis_name}' "
                         f"(names: {list(df['name'])})")
    vec = row[cols].values[0].astype(np.float64)
    sigma = float(row["sigma"].values[0])
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec, sigma


def load_traj_map(path):
    """Every clustered trajectory -> its cluster. No margin filter."""
    import pandas as pd
    tc = pd.read_csv(path)
    tc.columns = [c.strip() for c in tc.columns]
    return {(str(s)[:16], int(v)): int(c)
            for s, v, c in zip(tc["scenario_id"], tc["vehicle_id"], tc["cluster"])}


def build_assignment(gt, traj_of, n_clusters, fallback_idx):
    """Per-slot policy index for this batch. Labelled vehicle -> its cluster;
    everything else -> fallback_idx."""
    sids   = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
    vids   = _squeeze(np.asarray(gt["id"])).reshape(-1)
    is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
    B = len(sids)
    assign = np.full(B, fallback_idx, dtype=np.int64)
    stats = Counter()
    for a in range(B):
        if not is_veh[a]:
            stats["non_vehicle"] += 1
            continue
        c = traj_of.get((str(sids[a])[:16], int(vids[a])))
        if c is not None and 0 <= c < n_clusters:
            assign[a] = c
            stats[f"cluster{c}"] += 1
        else:
            stats["unlabelled_vehicle"] += 1
    return assign, stats


def collect_mixture(env, policies, assign, shifts, num_rollouts, device):
    """num_rollouts of the current allocation, driven by the assigned mixture
    with an optional per-policy role SHIFT. shifts[k] is a (role_dim,) tensor
    or None (natural). Agents partitioned by policy -> B rows/step total."""
    B = env.num_agents
    K = len(policies)
    traj = {k: np.zeros((B, num_rollouts, T), dtype=np.float32)
            for k in ("x", "y", "z", "heading")}
    traj["id"] = np.zeros((B, num_rollouts, T), dtype=np.int32)
    env_acc = {k: [] for k in ENV_KEYS}

    idx_by_pol = [torch.as_tensor(np.where(assign == k)[0],
                                  dtype=torch.long, device=device)
                  for k in range(K)]

    for r in range(num_rollouts):
        print(f"\r    rollout {r+1}/{num_rollouts}", end="", flush=True)
        obs_np, _ = env.reset()
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        states = [policies[k].initial_state(int(idx_by_pol[k].numel()), device)
                  if idx_by_pol[k].numel() else None for k in range(K)]
        for t in range(T):
            ag = env.get_global_agent_state()
            traj["x"][:, r, t]       = ag["x"]
            traj["y"][:, r, t]       = ag["y"]
            traj["z"][:, r, t]       = ag.get("z", np.zeros(B))
            traj["heading"][:, r, t] = ag["heading"]
            traj["id"][:, r, t]      = ag["id"]
            action = torch.zeros(B, dtype=torch.long, device=device)
            with torch.no_grad():
                for k in range(K):
                    idx = idx_by_pol[k]
                    if idx.numel() == 0:
                        continue
                    sh = shifts[k]
                    if sh is None:
                        lg, _, states[k], _ = policies[k](obs[idx], states[k])
                    else:
                        # two-pass natural offset: read the natural role, then
                        # re-run with it shifted (state advances once, on pass 2)
                        _, _, _, ri0 = policies[k](obs[idx], states[k])
                        forced = ri0["role_z"] + sh
                        lg, _, states[k], _ = policies[k](
                            obs[idx], states[k], forced_role=forced)
                    action[idx] = Categorical(logits=lg.float()).sample()
            obs_np, _, _, _, info = env.step(
                action.cpu().numpy().reshape(B, 1))
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            for it in (info if isinstance(info, list) else [info]):
                if isinstance(it, dict) and "score" in it:
                    for kk in ENV_KEYS:
                        if kk in it:
                            env_acc[kk].append(float(np.mean(it[kk])))
    print("\r" + " " * 30 + "\r", end="")
    env_metrics = {k: (float(np.mean(v)) if v else float("nan"))
                   for k, v in env_acc.items()}
    return traj, env_metrics


def main():
    args = parse_args()
    import pandas as pd

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]

    ini = load_drive_config()
    cfg = dict(ini["env"])
    cfg.update({
        "num_maps":      args.num_maps,
        "num_agents":    args.num_agents,
        "map_dir":       args.data_dir,
        "control_mode":  args.control_mode,
        "goal_behavior": ini["eval"]["wosac_goal_behavior"],
        "goal_radius":   ini["eval"]["wosac_goal_radius"],
    })
    from pufferlib.ocean.drive.drive import Drive
    env = Drive(**cfg)
    obs_np, _ = env.reset()
    obs_dim = obs_np.shape[-1]

    # --- Policies + per-cluster PC1 axes -----------------------------------
    ckpt_paths = [c.strip() for c in args.ckpts.split(",")]
    axes_paths = [a.strip() for a in args.axes.split(",")]
    if len(ckpt_paths) != args.n_clusters or len(axes_paths) != args.n_clusters:
        raise SystemExit(f"[mix] --ckpts ({len(ckpt_paths)}) and --axes "
                         f"({len(axes_paths)}) must each have n_clusters="
                         f"{args.n_clusters} entries")

    policies, axis_of = [], []
    for k, (ckpt, axf) in enumerate(zip(ckpt_paths, axes_paths)):
        if not os.path.isfile(ckpt):
            raise SystemExit(f"[mix] missing cluster-{k} checkpoint: {ckpt}")
        if not os.path.isfile(axf):
            raise SystemExit(f"[mix] missing cluster-{k} axes csv: {axf}")
        pol, _ = load_policy(ckpt, obs_dim, device, role_dim_override=args.role_dim)
        pol.eval()
        vec, sigma = load_axis(axf, args.role_dim, args.axis)
        policies.append(pol)
        axis_of.append((vec, sigma))
        print(f"[mix] cluster {k}: {ckpt}\n         axes {axf}  "
              f"{args.axis} sigma={sigma:.4f}")

    if args.fallback_ckpt:
        pol, _ = load_policy(args.fallback_ckpt, obs_dim, device,
                             role_dim_override=args.role_dim)
        pol.eval()
        policies.append(pol)
        axis_of.append(None)                 # dedicated fallback is never forced
        fallback_idx = len(policies) - 1
        print(f"[mix] fallback (dedicated, never forced): {args.fallback_ckpt}")
    else:
        fallback_idx = args.fallback_cluster
        print(f"[mix] fallback: reuse cluster-{fallback_idx} policy "
              f"(forced along its own PC1)")

    def shifts_for(alpha):
        if alpha == 0.0:
            return [None] * len(policies)
        out = []
        for ax in axis_of:
            if ax is None:
                out.append(None)
            else:
                vec, sigma = ax
                out.append(torch.as_tensor(alpha * sigma * vec,
                                           dtype=torch.float32, device=device))
        return out

    traj_of = load_traj_map(args.traj_clusters)
    print(f"[mix] {len(traj_of)} labelled trajectories | control_mode="
          f"{args.control_mode} | alphas={alphas}")
    print(f"[mix] {args.wosac_max_batches} batches x {len(alphas)} conditions "
          f"x {args.wosac_rollouts} rollouts")

    from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator
    evaluator = WOSACEvaluator({
        "eval":  {"wosac_init_steps": 10, "wosac_num_rollouts": args.wosac_rollouts},
        "train": {"device": str(device)},
    })

    per_alpha_rows = {a: [] for a in alphas}
    seen           = {a: set() for a in alphas}
    env_rows, assign_totals = [], Counter()

    for batch in range(args.wosac_max_batches):
        if batch > 0:
            env.resample_maps()
        env.reset()
        gt          = env.get_ground_truth_trajectories()
        agent_state = env.get_global_agent_state()
        road_edges  = env.get_road_edge_polylines()
        assign, stats = build_assignment(gt, traj_of, args.n_clusters, fallback_idx)
        assign_totals.update(stats)

        for a in alphas:
            print(f"  batch {batch+1}/{args.wosac_max_batches}  alpha={a:+.2f}",
                  flush=True)
            sim, env_m = collect_mixture(env, policies, assign, shifts_for(a),
                                         args.wosac_rollouts, device)
            env_m.update({"alpha": a, "batch": batch})
            env_rows.append(env_m)
            try:
                df = evaluator.compute_metrics(gt, sim, agent_state, road_edges,
                                               aggregate_results=False)
            except Exception as e:
                print(f"    [wosac metrics failed: {e}]")
                continue
            new = set(df.index.tolist()) - seen[a]
            if new:
                sub = df[df.index.isin(new)].copy()
                sub["alpha"] = a
                per_alpha_rows[a].append(sub)
                seen[a].update(new)

    if not any(per_alpha_rows.values()):
        raise SystemExit("[mix] no WOSAC results collected")

    env_df = pd.DataFrame(env_rows)
    pd.concat([pd.concat(v) for v in per_alpha_rows.values() if v]) \
        .to_csv(out / "mixture_wosac_by_scenario.csv")
    env_df.to_csv(out / "mixture_env_by_batch.csv", index=False)

    # --- Per-alpha aggregate (WOSAC + env safety) --------------------------
    rows = []
    for a in alphas:
        if not per_alpha_rows[a]:
            continue
        w = pd.concat(per_alpha_rows[a])
        r = {"alpha": a, "n_scenarios": len(w)}
        r.update({k: float(w[k].mean()) for k in WOSAC_KEYS if k in w.columns})
        e = env_df[env_df["alpha"] == a]
        for k in ENV_KEYS:
            if k in e.columns:
                r[k] = float(e[k].mean())
        r["goal_rate"] = r.get("completion_rate", float("nan"))
        rows.append(r)
    agg = pd.DataFrame(rows).sort_values("alpha")
    agg.to_csv(out / "mixture_wosac_by_alpha.csv", index=False)

    # --- Paired subset: scenarios present under EVERY alpha ----------------
    common = None
    for a in alphas:
        if per_alpha_rows[a]:
            ids = set(pd.concat(per_alpha_rows[a]).index.tolist())
            common = ids if common is None else (common & ids)
    if common:
        prows = []
        for a in alphas:
            if not per_alpha_rows[a]:
                continue
            w = pd.concat(per_alpha_rows[a])
            w = w[w.index.isin(common)]
            pr = {"alpha": a, "n_scenarios": len(w)}
            pr.update({k: float(w[k].mean()) for k in WOSAC_KEYS if k in w.columns})
            prows.append(pr)
        pd.DataFrame(prows).sort_values("alpha").to_csv(
            out / "mixture_wosac_by_alpha_paired.csv", index=False)

    # --- Report ------------------------------------------------------------
    tot = sum(assign_totals.values())
    fb  = assign_totals["non_vehicle"] + assign_totals["unlabelled_vehicle"]
    print("\n" + "=" * 66)
    print("  POLICY ASSIGNMENT  (summed over batches)")
    print("=" * 66)
    for k in range(args.n_clusters):
        n = assign_totals.get(f"cluster{k}", 0)
        print(f"  cluster {k} policy   : {n:>9,}  ({100*n/max(tot,1):.1f}%)")
    print(f"  -> fallback         : {fb:>9,}  ({100*fb/max(tot,1):.1f}%)  "
          f"[{assign_totals['unlabelled_vehicle']:,} unlabelled veh + "
          f"{assign_totals['non_vehicle']:,} non-veh]")
    if fb / max(tot, 1) > 0.5:
        print("  WARNING: >50% on the fallback -- barely a 4-way mixture.")

    print("\n" + "=" * 66)
    print(f"  MIXTURE WOSAC vs {args.axis} FORCING")
    print("=" * 66)
    show = ["alpha", "n_scenarios", "realism_meta_score", "kinematic_metrics",
            "interactive_metrics", "map_based_metrics", "min_ade",
            "collision_rate", "offroad_rate", "goal_rate"]
    show = [c for c in show if c in agg.columns]
    print(agg[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[mix] wrote mixture_wosac_by_alpha.csv / _paired.csv / "
          f"_by_scenario.csv / mixture_env_by_batch.csv -> {out}")


if __name__ == "__main__":
    main()
