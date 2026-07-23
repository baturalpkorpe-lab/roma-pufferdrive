"""
role_mixture_wosac.py -- WOSAC for a MIXTURE of per-cluster policies.

Idea 2: every trajectory cluster has its own trained policy. At scene init,
look up each vehicle's trajectory cluster and let THAT cluster's policy drive
it. Then run the normal WOSAC procedure -- except the scene is now driven by
four specialists at once instead of one generalist.

How it differs from role_cluster_wosac.py (idea 1): idea 1 scored ONE cluster
policy on ONLY its own cluster's vehicles (everyone else GT-replay). Here the
FULL WOSAC population is controlled (control_wosac, the standard eval mode),
and each agent is routed to the policy matching its trajectory type -- so the
four policies interact in the same scene for the first time.

Mechanics
---------
All K policies run on the full agent batch every step (each keeps its own GRU
state per agent, so each has tracked every agent's full history), and each
agent's action is gathered from the policy it was assigned:

    logits_k = policy_k(obs, state_k)  for k in 0..K-1     # (K, B, A)
    action[i] = sample( logits[ assign[i], i ] )

Cost is K x a normal WOSAC forward; memory is trivial (K small GRU states).

Assignment (rebuilt per map batch from the batch's GT):
  - a vehicle with a trajectory-cluster label -> that cluster's policy
  - an UNLABELLED vehicle, or a non-vehicle (ped/cyclist), or a below-margin
    edge trajectory -> the FALLBACK policy (--fallback_ckpt if given, else the
    --fallback_cluster policy). The fallback fraction is reported every run --
    if it is large the mixture is really a fallback-policy eval, so watch it.

THE SCIENTIFIC CAVEAT, on purpose: each cluster policy was trained in a world
where every OTHER agent replays GT. In the mixture it drives among other
LEARNED policies for the first time -- a real distribution shift. If the
mixture underperforms a single full-dataset policy, that shift is the finding,
not a bug. This is the honest test of whether per-cluster specialisation
survives contact with itself.

Usage (from the account that has the cluster checkpoints):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/$USER/PufferDrive \
    python role_mixture_wosac.py \
        --ckpt_template /scratch/$USER/checkpoints/roma_cluster{k}_dim4/roma_dim4_final.pt \
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
    p.add_argument("--ckpt_template", type=str, default="",
                   help="Checkpoint path with a literal {k} for the cluster "
                        "index, e.g. .../roma_cluster{k}_dim4/roma_dim4_final.pt. "
                        "Ignored if --ckpts is given.")
    p.add_argument("--ckpts", type=str, default="",
                   help="Explicit per-cluster checkpoints, comma-separated in "
                        "cluster order. Overrides --ckpt_template -- use when a "
                        "cluster's final checkpoint has an odd name (e.g. "
                        "cluster3 roma_dim4_final_3B.pt).")
    p.add_argument("--n_clusters",   type=int, default=4)
    p.add_argument("--traj_clusters", type=str, required=True,
                   help="trajectory_clusters_stopfrac.csv: scenario_id, "
                        "vehicle_id, cluster")
    p.add_argument("--data_dir",     type=str, required=True)
    p.add_argument("--out_dir",      type=str, required=True)
    p.add_argument("--role_dim",     type=int, default=4)
    p.add_argument("--fallback_ckpt", type=str, default="",
                   help="Policy for unlabelled/non-vehicle agents. Empty = use "
                        "the --fallback_cluster policy instead.")
    p.add_argument("--fallback_cluster", type=int, default=0,
                   help="If no --fallback_ckpt, unlabelled agents use this "
                        "cluster's policy.")
    p.add_argument("--num_agents",   type=int, default=3072)
    p.add_argument("--num_maps",     type=int, default=10000)
    p.add_argument("--wosac_rollouts",    type=int, default=32)
    p.add_argument("--wosac_max_batches", type=int, default=100)
    p.add_argument("--min_margin",   type=float, default=0.0,
                   help=">0 routes only CONFIDENT trajectories (margin>this) to "
                        "their cluster policy; the rest go to fallback. 0 = use "
                        "every labelled vehicle.")
    p.add_argument("--control_mode", type=str, default="control_wosac",
                   help="Standard WOSAC controls every valid agent -- that is "
                        "the point here (the mixture drives the whole scene).")
    p.add_argument("--device",       type=str, default="cuda")
    return p.parse_args()


def load_traj_map(path, min_margin):
    import pandas as pd
    tc = pd.read_csv(path)
    tc.columns = [c.strip() for c in tc.columns]
    if min_margin > 0 and "margin" in tc.columns:
        n0 = len(tc)
        tc = tc[tc["margin"] >= min_margin]
        print(f"[mix] margin>={min_margin}: kept {len(tc)}/{n0} trajectories")
    # scenario_id truncated to 16 chars to match the binary header + GT.
    return {(str(s)[:16], int(v)): int(c)
            for s, v, c in zip(tc["scenario_id"], tc["vehicle_id"], tc["cluster"])}


def build_assignment(gt, traj_of, n_clusters, fallback_idx):
    """Per-slot policy index for this batch's allocation. Vehicles with a valid
    cluster label -> that cluster; everything else -> fallback_idx."""
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


def collect_mixture(env, policies, assign, num_rollouts, device):
    """num_rollouts of the current allocation, driven by the assigned mixture.
    Returns (sim_traj, env_metrics)."""
    B = env.num_agents
    K = len(policies)
    traj = {k: np.zeros((B, num_rollouts, T), dtype=np.float32)
            for k in ("x", "y", "z", "heading")}
    traj["id"] = np.zeros((B, num_rollouts, T), dtype=np.int32)
    env_acc = {k: [] for k in ENV_KEYS}

    assign_t = torch.as_tensor(assign, dtype=torch.long, device=device)
    idxB     = torch.arange(B, device=device)

    for r in range(num_rollouts):
        print(f"\r    rollout {r+1}/{num_rollouts}", end="", flush=True)
        obs_np, _ = env.reset()
        obs    = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        states = [p.initial_state(B, device) for p in policies]
        for t in range(T):
            ag = env.get_global_agent_state()
            traj["x"][:, r, t]       = ag["x"]
            traj["y"][:, r, t]       = ag["y"]
            traj["z"][:, r, t]       = ag.get("z", np.zeros(B))
            traj["heading"][:, r, t] = ag["heading"]
            traj["id"][:, r, t]      = ag["id"]
            with torch.no_grad():
                logits = []
                for k in range(K):
                    lg, _, states[k], _ = policies[k](obs, states[k])
                    logits.append(lg.float())
                stacked = torch.stack(logits, dim=0)      # (K, B, A)
                chosen  = stacked[assign_t, idxB]         # (B, A): each agent's policy
            action = Categorical(logits=chosen).sample()
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

    # --- Resolve the K cluster checkpoints ---------------------------------
    if args.ckpts:
        ckpt_paths = [c.strip() for c in args.ckpts.split(",")]
        if len(ckpt_paths) != args.n_clusters:
            raise SystemExit(f"[mix] --ckpts has {len(ckpt_paths)} paths but "
                             f"--n_clusters={args.n_clusters}")
    elif args.ckpt_template:
        ckpt_paths = [args.ckpt_template.format(k=k) for k in range(args.n_clusters)]
    else:
        raise SystemExit("[mix] give --ckpts or --ckpt_template")

    # --- Load the K cluster policies (+ optional fallback) -----------------
    policies = []
    for k, ckpt in enumerate(ckpt_paths):
        if not os.path.isfile(ckpt):
            raise SystemExit(f"[mix] missing cluster-{k} checkpoint: {ckpt}")
        pol, rd = load_policy(ckpt, obs_dim, device, role_dim_override=args.role_dim)
        pol.eval()
        policies.append(pol)
        print(f"[mix] cluster {k}: {ckpt}  (role_dim={rd})")

    if args.fallback_ckpt:
        pol, _ = load_policy(args.fallback_ckpt, obs_dim, device,
                             role_dim_override=args.role_dim)
        pol.eval()
        policies.append(pol)
        fallback_idx = len(policies) - 1
        print(f"[mix] fallback (dedicated): {args.fallback_ckpt} -> idx {fallback_idx}")
    else:
        fallback_idx = args.fallback_cluster
        print(f"[mix] fallback: reuse cluster-{fallback_idx} policy for "
              f"unlabelled/non-vehicle agents")

    traj_of = load_traj_map(args.traj_clusters, args.min_margin)
    print(f"[mix] {len(traj_of)} labelled (scenario,vehicle) trajectories")
    print(f"[mix] control_mode={args.control_mode}  "
          f"{args.wosac_max_batches} batches x {args.wosac_rollouts} rollouts")

    from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator
    evaluator = WOSACEvaluator({
        "eval":  {"wosac_init_steps": 10, "wosac_num_rollouts": args.wosac_rollouts},
        "train": {"device": str(device)},
    })

    all_results, seen = [], set()
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

        sim, env_m = collect_mixture(env, policies, assign, args.wosac_rollouts, device)
        env_m["batch"] = batch
        env_rows.append(env_m)

        try:
            df = evaluator.compute_metrics(gt, sim, agent_state, road_edges,
                                           aggregate_results=False)
        except Exception as e:
            print(f"  [batch {batch}: wosac metrics failed: {e}]")
            continue
        new = set(df.index.tolist()) - seen
        if new:
            all_results.append(df[df.index.isin(new)])
            seen.update(new)

        if (batch + 1) % 10 == 0 and all_results:
            agg = pd.concat(all_results).mean()
            print(f"  Batch {batch+1}/{args.wosac_max_batches} | "
                  f"scenarios: {len(seen)} | realism: {agg['realism_meta_score']:.4f}"
                  f" | kin {agg['kinematic_metrics']:.3f}"
                  f" | int {agg['interactive_metrics']:.3f}"
                  f" | map {agg['map_based_metrics']:.3f}", flush=True)

    if not all_results:
        raise SystemExit("[mix] no WOSAC results collected")

    combined = pd.concat(all_results)
    agg = combined.mean()

    # --- Assignment breakdown ----------------------------------------------
    tot = sum(assign_totals.values())
    fb  = assign_totals["non_vehicle"] + assign_totals["unlabelled_vehicle"]
    print("\n" + "=" * 64)
    print("  POLICY ASSIGNMENT  (summed over batches)")
    print("=" * 64)
    for k in range(args.n_clusters):
        n = assign_totals.get(f"cluster{k}", 0)
        print(f"  cluster {k} policy   : {n:>9,}  ({100*n/max(tot,1):.1f}%)")
    print(f"  -> fallback         : {fb:>9,}  ({100*fb/max(tot,1):.1f}%)  "
          f"[{assign_totals['unlabelled_vehicle']:,} unlabelled veh + "
          f"{assign_totals['non_vehicle']:,} non-veh]")
    if fb / max(tot, 1) > 0.5:
        print("  WARNING: >50% on the fallback policy -- this is mostly a")
        print("           fallback-policy eval, not a 4-way mixture.")

    # --- WOSAC report ------------------------------------------------------
    print("\n" + "=" * 64)
    print("  MIXTURE WOSAC REALISM METRICS")
    print("=" * 64)
    print(f"  Scenarios evaluated   : {len(combined)}")
    print(f"  Realism meta-score    : {agg['realism_meta_score']:.4f}")
    print(f"  Kinematic metrics     : {agg['kinematic_metrics']:.4f}")
    print(f"  Interactive metrics   : {agg['interactive_metrics']:.4f}")
    print(f"  Map-based metrics     : {agg['map_based_metrics']:.4f}")
    if "min_ade" in agg:
        print(f"  minADE                : {agg['min_ade']:.4f} m")

    env_df = pd.DataFrame(env_rows)
    for k in ENV_KEYS:
        if k in env_df.columns:
            print(f"  {k:<21} : {env_df[k].mean():.4f}")

    # --- Save --------------------------------------------------------------
    summary = {k: float(agg[k]) for k in WOSAC_KEYS if k in agg.index}
    summary["scenarios"] = int(len(combined))
    for k in ENV_KEYS:
        if k in env_df.columns:
            summary[f"env_{k}"] = float(env_df[k].mean())
    for k in range(args.n_clusters):
        summary[f"assigned_cluster{k}"] = int(assign_totals.get(f"cluster{k}", 0))
    summary["assigned_fallback"] = int(fb)
    pd.DataFrame([summary]).to_csv(out / "mixture_wosac_summary.csv", index=False)
    combined.to_csv(out / "mixture_wosac_by_scenario.csv")
    env_df.to_csv(out / "mixture_env_by_batch.csv", index=False)
    print(f"\n[mix] wrote mixture_wosac_summary.csv / _by_scenario.csv / "
          f"mixture_env_by_batch.csv -> {out}")


if __name__ == "__main__":
    main()
