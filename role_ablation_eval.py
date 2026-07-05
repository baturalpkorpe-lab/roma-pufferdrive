"""
role_ablation_eval.py -- Is the role variable USEFUL? Eval-time ablation.

Rolls the same checkpoint through WOSAC-style evaluation under three role
conditions (nothing is retrained):

  natural    roles from the encoder (status quo)
  shuffled   each agent wears a SCENE-MATE's role (per-rollout permutation
             within each scenario: identical role distribution, scrambled
             assignment)
  collapsed  every agent forced to the population-mean role vector
             (heterogeneity removed; mean taken from the Phase C agent CSV)

Per condition it reports:
  - WOSAC realism metrics (meta score + all likelihoods)
  - within-scene across-agent speed spread vs the human GT spread
    (ratio ~1 = human-calibrated heterogeneity)

Interpretation:
  natural > collapsed on realism AND collapsed spread-ratio << 1
      -> roles causally produce human-calibrated heterogeneity: USEFUL.
  natural ~ collapsed everywhere
      -> the spread was sampling noise; roles are decorative.
  natural > shuffled
      -> correct role ASSIGNMENT matters, not just diversity.

Usage (from /scratch/e452103/PufferDrive):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/role_ablation_eval.py \
        --checkpoint /scratch/e452103/checkpoints/roma_baseline/roma_dim8_step3000238080.pt \
        --agent_data /scratch/e452103/role_regime/dim8/phaseC_agent_data.csv \
        --data_dir   pufferlib/resources/drive/binaries/training \
        --out_dir    /scratch/e452103/role_ablation/dim8
"""

import argparse
import ast
import configparser
import csv
import os
import time
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
from role_regime_analysis import _squeeze

T = 91
CONDITIONS = ["natural", "shuffled", "collapsed"]
TELEPORT_M = 8.0
GT_MOVE_MS = 1.0


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
    p.add_argument("--agent_data", type=str, required=True,
                   help="phaseC_agent_data.csv of the SAME checkpoint "
                        "(defines the population-mean role for 'collapsed')")
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--batches",    type=int, default=15,
                   help="Scenario batches per condition (~500 scenarios each)")
    p.add_argument("--rollouts",   type=int, default=32)
    p.add_argument("--num_agents", type=int, default=3072)
    p.add_argument("--num_maps",   type=int, default=10000)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


def population_mean_role(agent_csv):
    with open(agent_csv) as f:
        rows = list(csv.DictReader(f))
    role_cols = [c for c in rows[0].keys() if c.startswith("role_")]
    vals = np.array([[float(r[c]) for c in role_cols] for r in rows])
    return vals.mean(axis=0), len(role_cols)


def make_scene_perm(sids, rng):
    """Permutation of agent slots that stays WITHIN each scenario."""
    perm = np.arange(len(sids))
    for sid in np.unique(sids):
        idx = np.where(sids == sid)[0]
        if len(idx) > 1:
            perm[idx] = rng.permutation(idx)
    return perm


# ---------------------------------------------------------------------------
# Rollout collection under a role condition
# ---------------------------------------------------------------------------

def collect_condition(env, policy, condition, mean_role, device,
                      rollouts, rng):
    """WOSAC-format sim trajectories for one batch under one role condition.

    env.reset() reshuffles the slot<->scenario allocation, so the scene
    grouping (for the shuffled permutation and the spread stats) is re-read
    from THIS rollout's own reset, never from a stale earlier one.
    Returns (sim, gt_r0): the trajectories plus rollout-0's ground truth,
    slot-aligned with rollout 0 of sim."""
    B = env.num_agents
    sim = {k: np.zeros((B, rollouts, T), dtype=np.float32)
           for k in ("x", "y", "z", "heading")}
    sim["id"]    = np.zeros((B, rollouts, T), dtype=np.int32)
    sim["dones"] = np.zeros((B, rollouts, T), dtype=np.bool_)

    collapsed = None
    if condition == "collapsed":
        collapsed = torch.as_tensor(mean_role, dtype=torch.float32,
                                    device=device).unsqueeze(0).expand(B, -1)

    gt_r0 = None
    for r in range(rollouts):
        obs_np, _ = env.reset()
        gt_r   = env.get_ground_truth_trajectories()
        sids_r = _squeeze(np.asarray(gt_r["scenario_id"]).astype(str))
        if r == 0:
            gt_r0 = gt_r
        obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        state = policy.initial_state(B, device)
        trunc = np.zeros(B, dtype=bool)
        perm  = (make_scene_perm(sids_r, rng)
                 if condition == "shuffled" else None)
        perm_t = (torch.as_tensor(perm, dtype=torch.long, device=device)
                  if perm is not None else None)

        for t in range(T):
            ag = env.get_global_agent_state()
            sim["x"]      [:, r, t] = ag["x"]
            sim["y"]      [:, r, t] = ag["y"]
            sim["z"]      [:, r, t] = ag.get("z", np.zeros(B))
            sim["heading"][:, r, t] = ag["heading"]
            sim["id"]     [:, r, t] = ag["id"]
            sim["dones"]  [:, r, t] = trunc

            with torch.no_grad():
                if condition == "natural":
                    logits, _, state, _ = policy(obs, state)
                elif condition == "collapsed":
                    logits, _, state, _ = policy(obs, state,
                                                 forced_role=collapsed)
                else:  # shuffled: wear a scene-mate's role
                    _, _, _, ri = policy(obs, state)
                    logits, _, state, _ = policy(
                        obs, state, forced_role=ri["role_z"][perm_t])
            action = Categorical(logits=logits.float()).sample()
            obs_np, _, _, trunc, _ = env.step(
                action.cpu().numpy().reshape(B, 1))
            trunc = np.asarray(trunc).reshape(B)
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

    return sim, gt_r0


# ---------------------------------------------------------------------------
# Within-scene spread: sim vs human, from rollout 0 of a batch
# ---------------------------------------------------------------------------

def spread_stats(sim, gt, sids):
    gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
    valid  = _squeeze(gt["valid"]).astype(bool)
    is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
    T_gt   = gx.shape[1]

    out = []   # (std_human, std_sim) per usable scene
    for sid in np.unique(sids):
        if not sid:
            continue
        slots = np.where(sids == sid)[0]
        hu, si = [], []
        for a in slots:
            if not is_veh[a] or valid[a].sum() < 10:
                continue
            m    = valid[a]
            pair = m[:-1] & m[1:]
            if pair.sum() < 10:
                continue
            g_spd = np.hypot((gx[a, 1:] - gx[a, :-1])[pair],
                             (gy[a, 1:] - gy[a, :-1])[pair]) * 10
            if g_spd.max() <= GT_MOVE_MS:
                continue
            px = sim["x"][a, 0, :T_gt]
            py = sim["y"][a, 0, :T_gt]
            d  = np.hypot(np.diff(px), np.diff(py))
            d  = d[d < TELEPORT_M]              # mask respawn teleports
            if len(d) < 10:
                continue
            hu.append(g_spd.mean())
            si.append(d.mean() * 10)
        if len(hu) >= 3:
            out.append((float(np.std(hu)), float(np.std(si))))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mean_role, role_dim_csv = population_mean_role(args.agent_data)
    print(f"[ablation] population-mean role ({role_dim_csv} dims): "
          f"{np.round(mean_role, 3)}")

    import pandas as pd
    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

    ini = load_drive_config()
    cfg = dict(ini["env"])
    cfg.update({
        "num_maps":      args.num_maps,
        "num_agents":    args.num_agents,
        "map_dir":       args.data_dir,
        "control_mode":  ini["eval"]["wosac_control_mode"],
        "goal_behavior": 2,
        "goal_radius":   ini["eval"]["wosac_goal_radius"],
    })
    t0  = time.time()
    env = Drive(**cfg)
    obs_probe, _ = env.reset()
    print(f"[ablation] env created in {time.time()-t0:.0f}s")

    policy, role_dim = load_policy(args.checkpoint, obs_probe.shape[-1],
                                   device)
    if role_dim != role_dim_csv:
        raise SystemExit(f"checkpoint role_dim={role_dim} but agent_data has "
                         f"{role_dim_csv} role columns -- wrong pairing")
    print(f"[ablation] policy loaded (role_dim={role_dim})")

    evaluator = WOSACEvaluator({
        "eval":  {"wosac_init_steps": 10,
                  "wosac_num_rollouts": args.rollouts},
        "train": {"device": str(device)},
    })

    results = {c: {"dfs": [], "seen": set(), "spread": []}
               for c in CONDITIONS}

    for batch in range(args.batches):
        if batch > 0:
            env.resample_maps()
        env.reset()
        # Batch-level gt/agent_state/road go to the WOSAC evaluator, which
        # matches trajectories by vehicle id (robust to the reset reshuffle);
        # anything slot-aligned uses per-rollout data from collect_condition.
        gt          = env.get_ground_truth_trajectories()
        agent_state = env.get_global_agent_state()
        road_edges  = env.get_road_edge_polylines()

        for cond in CONDITIONS:
            t0  = time.time()
            sim, gt_r0 = collect_condition(env, policy, cond, mean_role,
                                           device, args.rollouts, rng)
            res = results[cond]
            # spread uses rollout 0's OWN ground truth: reset() reshuffles
            # the slot<->scenario allocation, so slot-alignment only holds
            # within the same reset generation.
            sids_r0 = _squeeze(np.asarray(gt_r0["scenario_id"]).astype(str))
            res["spread"].extend(spread_stats(sim, gt_r0, sids_r0))
            try:
                df  = evaluator.compute_metrics(gt, sim, agent_state,
                                                road_edges,
                                                aggregate_results=False)
                new = set(df.index.tolist()) - res["seen"]
                if new:
                    res["dfs"].append(df[df.index.isin(new)])
                    res["seen"].update(new)
            except Exception as e:
                if batch == 0:
                    import traceback
                    print(f"[ablation] compute_metrics failed ({cond}): {e}")
                    traceback.print_exc()
            print(f"[ablation] batch {batch+1}/{args.batches} {cond:>9}: "
                  f"{len(res['seen'])} scenarios, "
                  f"{time.time()-t0:.0f}s", flush=True)

    # ----- aggregate + report ------------------------------------------------
    KEYS = ["realism_meta_score", "kinematic_metrics", "interactive_metrics",
            "map_based_metrics", "min_ade", "likelihood_linear_speed",
            "likelihood_linear_acceleration", "likelihood_angular_speed",
            "likelihood_angular_acceleration",
            "likelihood_collision_indication",
            "likelihood_distance_to_nearest_object",
            "likelihood_time_to_collision",
            "likelihood_distance_to_road_edge",
            "likelihood_offroad_indication"]

    table = {}
    for cond in CONDITIONS:
        res = results[cond]
        row = {}
        if res["dfs"]:
            agg = pd.concat(res["dfs"]).mean()
            row.update({k: float(agg[k]) for k in KEYS if k in agg})
            row["scenarios"] = int(sum(len(d) for d in res["dfs"]))
        if res["spread"]:
            hu = np.mean([s[0] for s in res["spread"]])
            si = np.mean([s[1] for s in res["spread"]])
            row["spread_human"] = float(hu)
            row["spread_sim"]   = float(si)
            row["spread_ratio"] = float(si / hu)
            row["spread_scenes"] = len(res["spread"])
        table[cond] = row

    print("\n" + "=" * 74)
    print(f"  ROLE ABLATION  [{Path(args.checkpoint).stem}]")
    print("=" * 74)
    metrics_all = sorted({k for r in table.values() for k in r})
    hdr = f"  {'metric':<38}" + "".join(f"{c:>12}" for c in CONDITIONS)
    print(hdr)
    for k in metrics_all:
        line = f"  {k:<38}"
        for c in CONDITIONS:
            v = table[c].get(k)
            line += f"{v:>12.4f}" if isinstance(v, float) else f"{str(v):>12}"
        print(line)

    csv_path = out_dir / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + CONDITIONS)
        for k in metrics_all:
            w.writerow([k] + [table[c].get(k, "") for c in CONDITIONS])
    print(f"\n[ablation] saved -> {csv_path}")

    # ----- summary figure -----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    panels = [
        ("realism_meta_score", "WOSAC realism meta-score", None),
        ("likelihood_linear_speed", "speed likelihood", None),
        ("spread_ratio", "within-scene spread ratio\n(sim/human, 1.0 = "
         "calibrated)", 1.0),
    ]
    colors = {"natural": "#4477aa", "shuffled": "#ee8833",
              "collapsed": "#999999"}
    for ax, (key, title, refline) in zip(axes, panels):
        vals = [table[c].get(key, np.nan) for c in CONDITIONS]
        ax.bar(CONDITIONS, vals, color=[colors[c] for c in CONDITIONS])
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=9)
        if refline is not None:
            ax.axhline(refline, color="black", ls="--", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Role ablation -- {Path(args.checkpoint).stem}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_summary.png", dpi=140)
    plt.close(fig)
    print(f"[ablation] figure -> {out_dir / 'ablation_summary.png'}")

    env.close()


if __name__ == "__main__":
    main()
