"""
role_cluster_wosac.py -- WOSAC + safety metrics for a PER-CLUSTER policy,
across a PC1 forcing sweep.

What this evaluates
-------------------
A cluster-K policy only ever learned to drive cluster-K vehicles, so it must
be scored in the world it was trained in: ONLY cluster-K vehicles are
policy-controlled, every other object replays its GT trajectory. That is
already baked into the cluster map binaries by make_cluster_maps.py
(mark_as_expert=0 on cluster-K vehicles, =1 on everything else), so pointing
--data_dir at cluster_maps/cluster<K> is what selects the evaluation
population -- NOT the control_mode setting. See --control_mode below.

Five conditions per run, each a natural-offset shift of the role along PC1:
    z(t) = z_natural(t) + alpha * sigma_PC1 * PC1        alpha in {-1,-.5,0,.5,1}
alpha=0 is exactly the natural policy. This is the same forcing scheme as
role_paired_sweep.py (a SHIFT of each agent's own live role, not a clamp), so
the numbers here sit on the same axis as the paired-sweep statistics.

Conditions are run INSIDE each map batch, not outside: every alpha therefore
sees the identical map pool and the identical scenarios within a batch, which
makes the across-alpha comparison paired. `paired_only` in the output
restricts to scenarios that produced metrics under ALL five alphas.

Metrics per condition: the full WOSAC set (realism meta-score, kinematic /
interactive / map-based components, every likelihood term, ADE / minADE) plus
the env's own collision_rate, offroad_rate, goal (completion) rate and score,
accumulated over the same rollouts.

TWO THINGS TO CHECK BEFORE TRUSTING ABSOLUTE NUMBERS
----------------------------------------------------
1. Expert dilution. Most agents in a cluster map replay GT exactly, so their
   sim matches GT perfectly and their WOSAC likelihoods are ~ideal. Unless
   WOSACEvaluator filters to controlled agents, the meta-score is inflated and
   is NOT comparable to a full-dataset baseline. It is still valid ACROSS
   alphas: the experts contribute the same constant to every condition, so the
   role effect survives, merely compressed. controlled_frac in the output says
   how much dilution there is.
2. Control mode. USE control_agents, NOT drive.ini's eval default. From
   drive.h should_control_agent():

       case CONTROL_WOSAC:
           // Valid types only, ignore expert flag and goal distance
           return (is_vehicle || is_ped_or_bike);

   CONTROL_WOSAC returns before the mark_as_expert check, so it controls EVERY
   vehicle, pedestrian and cyclist in the scene -- 100% leakage, which would
   make a per-cluster evaluation meaningless. control_agents falls through to
   the default branch, which does test mark_as_expert, and is the mode
   training used. control_sdc_only is worse still: it returns
   agent_idx == sdc_track_index, and make_cluster_maps blanks that index for
   foreign egos, so most maps would have nothing controlled at all.

   Note that control_agents ALSO applies a goal-distance filter
   (distance_to_goal >= MIN_DISTANCE_TO_GOAL) and an active_agent_count cap,
   so the controlled count is legitimately somewhat BELOW manifest n_control.
   Training used the same filter, so eval and training populations match.
   Verified empirically at runtime anyway: expert agents replay GT
   deterministically and are bit-identical across all 32 rollouts, while
   controlled agents sample actions and vary.

Usage:
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/$USER/PufferDrive \
    python $HOME/roma_pufferdrive/role_cluster_wosac.py \
        --checkpoint /scratch/$USER/checkpoints/roma_cluster3_dim4/roma_dim4_final.pt \
        --data_dir   /scratch/$USER/cluster_maps/cluster3 \
        --axes_csv   /scratch/$USER/role_paired/cluster3_3B/role_paired_axes.csv \
        --out_dir    /scratch/$USER/wosac_cluster/cluster3
"""

import argparse
import ast
import configparser
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import load_policy

T = 91

# Env-side metrics pulled out of the info dict (goal rate is completion_rate).
ENV_KEYS = ["score", "collision_rate", "offroad_rate", "completion_rate"]

# WOSAC columns copied straight through when present.
WOSAC_KEYS = [
    "realism_meta_score", "kinematic_metrics", "interactive_metrics",
    "map_based_metrics", "ade", "min_ade",
    "likelihood_linear_speed", "likelihood_linear_acceleration",
    "likelihood_angular_speed", "likelihood_angular_acceleration",
    "likelihood_collision_indication", "likelihood_distance_to_nearest_object",
    "likelihood_time_to_collision", "likelihood_distance_to_road_edge",
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir",   type=str, required=True,
                   help="cluster_maps/cluster<K> -- the binaries decide which "
                        "vehicles are controllable")
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--axes_csv",   type=str, required=True,
                   help="role_paired_axes.csv from role_paired_sweep.py for "
                        "THIS cluster: supplies the PC1 direction and its sigma")
    p.add_argument("--axis",       type=str, default="PC1")
    p.add_argument("--alphas",     type=str, default="-1,-0.5,0,0.5,1",
                   help="natural-offset multiples of sigma along the axis")
    p.add_argument("--role_dim",   type=int, default=0,
                   help="0 = read from the checkpoint")
    p.add_argument("--num_agents", type=int, default=3072)
    p.add_argument("--num_maps",   type=int, default=10000)
    p.add_argument("--wosac_rollouts",    type=int, default=32)
    p.add_argument("--wosac_max_batches", type=int, default=20,
                   help="Map batches. COST IS 5x A NORMAL WOSAC RUN (one pass "
                        "per alpha), so 20 here costs about what 100 costs in "
                        "eval_wosac.sbatch. Raise only with the wall clock.")
    p.add_argument("--control_mode", type=str, default="control_agents",
                   help="Env control mode. DEFAULTS TO control_agents, NOT to "
                        "drive.ini's eval.wosac_control_mode: CONTROL_WOSAC "
                        "ignores mark_as_expert and would control every agent "
                        "in the scene, destroying the per-cluster population. "
                        "Pass '' to fall back to the drive.ini eval value.")
    p.add_argument("--device",     type=str, default="cuda")
    return p.parse_args()


def load_axis(axes_csv, role_dim, axis_name):
    """Returns (unit_vector, sigma) for the named axis."""
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


def rollout_batch(env, policy, device, num_rollouts, shift_vec):
    """
    32 rollouts of the same map batch with a constant role SHIFT applied to
    every agent's own live role. Returns (sim_traj, env_metrics, ctrl_info).

    Uncontrolled agents are GT-replayed by the env, so shifting their role is
    a no-op -- no need to identify them up front.
    """
    B = env.num_agents
    traj = {k: np.zeros((B, num_rollouts, T), dtype=np.float32)
            for k in ("x", "y", "z", "heading")}
    traj["id"] = np.zeros((B, num_rollouts, T), dtype=np.int32)
    env_acc = {k: [] for k in ENV_KEYS}

    fv = (None if shift_vec is None else
          torch.as_tensor(shift_vec, dtype=torch.float32, device=device))

    for r in range(num_rollouts):
        print(f"\r    rollout {r+1}/{num_rollouts}", end="", flush=True)
        obs_np, _ = env.reset()
        obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        state = policy.initial_state(B, device)
        for t in range(T):
            ag = env.get_global_agent_state()
            traj["x"][:, r, t]       = ag["x"]
            traj["y"][:, r, t]       = ag["y"]
            traj["z"][:, r, t]       = ag.get("z", np.zeros(B))
            traj["heading"][:, r, t] = ag["heading"]
            traj["id"][:, r, t]      = ag["id"]
            with torch.no_grad():
                if fv is None:
                    logits, _, state, _ = policy(obs, state)
                else:
                    # Two-pass natural offset, identical to
                    # role_paired_sweep.rollout_condition: pass 1 reads each
                    # agent's own live role (its state update is discarded),
                    # pass 2 re-runs with that role shifted. Keeps individual
                    # variation instead of clamping everyone to one vector.
                    _, _, _, ri0 = policy(obs, state)
                    logits, _, state, _ = policy(
                        obs, state, forced_role=ri0["role_z"] + fv)
            action = Categorical(logits=logits.float()).sample()
            obs_np, _, _, _, info = env.step(
                action.cpu().numpy().reshape(B, 1))
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

            items = info if isinstance(info, list) else [info]
            for it in items:
                if isinstance(it, dict) and "score" in it:
                    for k in ENV_KEYS:
                        if k in it:
                            env_acc[k].append(float(np.mean(it[k])))
    print("\r" + " " * 30 + "\r", end="")

    # Controlled-agent detection: an expert replays GT deterministically and
    # is bit-identical across rollouts; a controlled agent samples actions and
    # moves differently every time. This is what verifies control_mode.
    spread = traj["x"].std(axis=1).max(axis=1) + traj["y"].std(axis=1).max(axis=1)
    moved  = (np.abs(traj["x"][:, 0, :] - traj["x"][:, 0, :1]).max(axis=1)
              + np.abs(traj["y"][:, 0, :] - traj["y"][:, 0, :1]).max(axis=1)) > 0.5
    ctrl = {
        "n_controlled_detected": int((spread > 1e-3).sum()),
        "n_moving": int(moved.sum()),
        "n_agents": int(B),
        "controlled_frac": float((spread > 1e-3).mean()),
    }
    env_metrics = {k: (float(np.mean(v)) if v else float("nan"))
                   for k, v in env_acc.items()}
    return traj, env_metrics, ctrl


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
    control_mode = args.control_mode or ini["eval"]["wosac_control_mode"]
    cfg.update({
        "num_maps":      args.num_maps,
        "num_agents":    args.num_agents,
        "map_dir":       args.data_dir,
        "control_mode":  control_mode,
        "goal_behavior": ini["eval"]["wosac_goal_behavior"],
        "goal_radius":   ini["eval"]["wosac_goal_radius"],
    })
    from pufferlib.ocean.drive.drive import Drive
    env = Drive(**cfg)
    obs_np, _ = env.reset()

    policy, role_dim = load_policy(args.checkpoint, obs_np.shape[-1], device,
                                   role_dim_override=args.role_dim)
    if role_dim == 0:
        raise SystemExit("role_dim=0 checkpoint -- no role to sweep")

    vec, sigma = load_axis(args.axes_csv, role_dim, args.axis)
    print(f"[cwosac] ckpt={args.checkpoint}")
    print(f"[cwosac] maps={args.data_dir}")
    print(f"[cwosac] control_mode={control_mode}  (drive.ini eval default: "
          f"{ini['eval']['wosac_control_mode']})")
    print(f"[cwosac] {args.axis} sigma={sigma:.4f}  alphas={alphas}")
    print(f"[cwosac] {args.wosac_max_batches} batches x {len(alphas)} conditions "
          f"x {args.wosac_rollouts} rollouts")

    from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator
    evaluator = WOSACEvaluator({
        "eval":  {"wosac_init_steps": 10,
                  "wosac_num_rollouts": args.wosac_rollouts},
        "train": {"device": str(device)},
    })

    # Per-alpha accumulators. Scenario dedup is PER ALPHA -- a shared set would
    # let the first condition claim every scenario and starve the rest.
    per_alpha_rows = {a: [] for a in alphas}
    seen           = {a: set() for a in alphas}
    env_rows, ctrl_rows = [], []

    for batch in range(args.wosac_max_batches):
        if batch > 0:
            env.resample_maps()
        # Conditions INSIDE the batch: identical map pool for every alpha.
        for a in alphas:
            shift = None if a == 0.0 else (a * sigma * vec)
            print(f"  batch {batch+1}/{args.wosac_max_batches}  alpha={a:+.2f}",
                  flush=True)
            env.reset()
            gt          = env.get_ground_truth_trajectories()
            agent_state = env.get_global_agent_state()
            road_edges  = env.get_road_edge_polylines()

            sim, env_m, ctrl = rollout_batch(
                env, policy, device, args.wosac_rollouts, shift)
            env_m.update({"alpha": a, "batch": batch})
            ctrl.update({"alpha": a, "batch": batch})
            env_rows.append(env_m); ctrl_rows.append(ctrl)

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
        raise SystemExit("no WOSAC results collected")

    # ---- Per-scenario table, then per-alpha aggregate ----------------------
    scen = pd.concat([pd.concat(v) for v in per_alpha_rows.values() if v])
    scen.to_csv(out / "wosac_by_scenario.csv")
    env_df  = pd.DataFrame(env_rows)
    ctrl_df = pd.DataFrame(ctrl_rows)
    env_df.to_csv(out / "env_metrics_by_batch.csv", index=False)

    rows = []
    for a in alphas:
        if not per_alpha_rows[a]:
            continue
        w = pd.concat(per_alpha_rows[a])
        r = {"alpha": a, "n_scenarios": len(w)}
        r.update({k: float(w[k].mean()) for k in WOSAC_KEYS if k in w.columns})
        e = env_df[env_df["alpha"] == a]
        r.update({k: float(e[k].mean()) for k in ENV_KEYS if k in e.columns})
        r["goal_rate"] = r.get("completion_rate", float("nan"))
        c = ctrl_df[ctrl_df["alpha"] == a]
        r["controlled_frac"]       = float(c["controlled_frac"].mean())
        r["n_controlled_detected"] = float(c["n_controlled_detected"].mean())
        rows.append(r)
    agg = pd.DataFrame(rows).sort_values("alpha")
    agg.to_csv(out / "wosac_by_alpha.csv", index=False)

    # ---- Paired subset: scenarios that survived under EVERY alpha ---------
    common = None
    for a in alphas:
        if not per_alpha_rows[a]:
            continue
        ids = set(pd.concat(per_alpha_rows[a]).index.tolist())
        common = ids if common is None else (common & ids)
    paired = pd.DataFrame()
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
        paired = pd.DataFrame(prows).sort_values("alpha")
        paired.to_csv(out / "wosac_by_alpha_paired.csv", index=False)

    # ---- Report -----------------------------------------------------------
    cf = float(ctrl_df["controlled_frac"].mean())
    nd = float(ctrl_df["n_controlled_detected"].mean())
    print("\n" + "=" * 70)
    print("  CONTROLLED-AGENT CHECK  (settles the control_mode question)")
    print("=" * 70)
    print(f"  control_mode used        : {control_mode}")
    print(f"  agents per env           : {args.num_agents}")
    print(f"  detected as CONTROLLED   : {nd:.0f}  ({100*cf:.1f}%)")
    print( "  the rest replay GT exactly (bit-identical across 32 rollouts)")
    print( "  -> expect somewhat BELOW manifest.csv n_control: control_agents")
    print( "     also applies the goal-distance filter and the active-agent cap,")
    print( "     exactly as training did.")
    if control_mode == "control_wosac":
        print("\n  *** WRONG MODE ***  drive.h CONTROL_WOSAC returns before the")
        print("      mark_as_expert check ('ignore expert flag'), so every")
        print("      vehicle/ped/cyclist is controlled and the cluster")
        print("      population is gone. Rerun with --control_mode control_agents.")
    if nd < 1:
        print("\n  WARNING: NOTHING was controlled. Expected if control_mode is")
        print("           control_sdc_only -- make_cluster_maps blanks")
        print("           sdc_track_index for foreign egos. Use control_agents.")
    print("\n" + "=" * 70)
    print(f"  RESULTS BY ALPHA  ({args.axis} natural-offset sweep)")
    print("=" * 70)
    show = ["alpha", "n_scenarios", "realism_meta_score", "kinematic_metrics",
            "interactive_metrics", "map_based_metrics", "min_ade",
            "collision_rate", "offroad_rate", "goal_rate"]
    show = [c for c in show if c in agg.columns]
    print(agg[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not paired.empty:
        print(f"\n  paired subset ({len(common)} scenarios present under every alpha):")
        pshow = [c for c in ["alpha", "realism_meta_score", "min_ade"]
                 if c in paired.columns]
        print(paired[pshow].to_string(index=False,
                                      float_format=lambda v: f"{v:.4f}"))
    print(f"\n  NOTE: {100*(1-cf):.0f}% of agents replay GT exactly, so absolute")
    print( "  realism is inflated vs a full-dataset baseline. The across-alpha")
    print( "  comparison is unaffected (same constant in every condition).")

    # ---- Figure -----------------------------------------------------------
    panels = [c for c in ["realism_meta_score", "kinematic_metrics",
                          "interactive_metrics", "map_based_metrics",
                          "min_ade", "collision_rate", "offroad_rate",
                          "goal_rate"] if c in agg.columns]
    if panels:
        ncol = 4
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.1 * nrow),
                                squeeze=False)
        for i, k in enumerate(panels):
            ax = axs[i // ncol][i % ncol]
            ax.plot(agg["alpha"], agg[k], "o-", color="#4c72b0")
            ax.axvline(0, ls=":", lw=1, color="grey")
            ax.set_title(k, fontsize=9)
            ax.set_xlabel(f"alpha ({args.axis}, x sigma)", fontsize=8)
            ax.tick_params(labelsize=8)
        for j in range(len(panels), nrow * ncol):
            axs[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"{Path(args.data_dir).name}  --  {args.axis} sweep  "
                     f"({int(nd)} controlled agents, {100*cf:.0f}% of env)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "wosac_alpha_sweep.png", dpi=140)

    print(f"\n[cwosac] wrote wosac_by_alpha.csv / _paired.csv / "
          f"wosac_by_scenario.csv / env_metrics_by_batch.csv / "
          f"wosac_alpha_sweep.png -> {out}")


if __name__ == "__main__":
    main()
