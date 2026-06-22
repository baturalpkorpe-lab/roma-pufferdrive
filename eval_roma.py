"""
eval_roma.py — WOSAC realism evaluation for ROMA checkpoints.

Usage (from ~/PufferDrive):
    PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 eval_roma.py \
        --checkpoint roma_main/checkpoints/roma_dim8/roma_dim8_final.pt \
        --wosac_rollouts 32 --wosac_num_maps 10000 --wosac_max_batches 500 \
        --wandb
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from torch.distributions import Categorical


class WOSACPolicyAdapter:
    def __init__(self, policy, num_agents, device):
        self.policy     = policy
        self.num_agents = num_agents
        self.device     = device
        self._state     = None

    def reset_state(self):
        self._state = self.policy.initial_state(self.num_agents, self.device)

    def forward_eval(self, obs):
        with torch.no_grad():
            logits, _, new_state, _ = self.policy(obs, self._state)
        self._state = new_state
        return logits


def load_policy(checkpoint_path, role_dim, obs_dim, device):
    from roma_pufferdrive.roma.policy import RomaPolicy
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    key    = "policy_state" if "policy_state" in ckpt else "policy"
    policy = RomaPolicy(obs_dim=obs_dim, role_dim=role_dim)
    policy.load_state_dict(ckpt[key])
    policy.to(device)
    policy.eval()
    return policy


def collect_wosac_trajectories(env, adapter, num_rollouts, num_steps=91):
    num_agents = env.num_agents
    traj = {
        "x":       np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32),
        "y":       np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32),
        "z":       np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32),
        "heading": np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32),
        "id":      np.zeros((num_agents, num_rollouts, num_steps), dtype=np.int32),
    }
    for r in range(num_rollouts):
        print(f"\r  rollout {r+1}/{num_rollouts}", end="", flush=True)
        obs_np, _ = env.reset()
        adapter.reset_state()
        obs = torch.as_tensor(obs_np, dtype=torch.float32).to(adapter.device)
        for t in range(num_steps):
            ag = env.get_global_agent_state()
            traj["x"]      [:, r, t] = ag["x"]
            traj["y"]      [:, r, t] = ag["y"]
            traj["z"]      [:, r, t] = ag.get("z", np.zeros(num_agents))
            traj["heading"][:, r, t] = ag["heading"]
            traj["id"]     [:, r, t] = ag["id"]
            action = Categorical(logits=adapter.forward_eval(obs)).sample()
            obs_np, _, _, _, _ = env.step(action.cpu().numpy().reshape(num_agents, 1))
            obs = torch.as_tensor(obs_np, dtype=torch.float32).to(adapter.device)
    print()
    return traj


def wosac_metric_dict(agg, scenarios, prefix="wosac/"):
    return {
        f"{prefix}realism_meta_score":              agg["realism_meta_score"],
        f"{prefix}kinematic_metrics":               agg["kinematic_metrics"],
        f"{prefix}interactive_metrics":             agg["interactive_metrics"],
        f"{prefix}map_based_metrics":               agg["map_based_metrics"],
        f"{prefix}min_ade":                         agg["min_ade"],
        f"{prefix}likelihood_linear_speed":         agg["likelihood_linear_speed"],
        f"{prefix}likelihood_linear_acceleration":  agg["likelihood_linear_acceleration"],
        f"{prefix}likelihood_angular_speed":        agg["likelihood_angular_speed"],
        f"{prefix}likelihood_angular_acceleration": agg["likelihood_angular_acceleration"],
        f"{prefix}likelihood_collision":            agg["likelihood_collision_indication"],
        f"{prefix}likelihood_dist_obj":             agg["likelihood_distance_to_nearest_object"],
        f"{prefix}likelihood_ttc":                  agg["likelihood_time_to_collision"],
        f"{prefix}likelihood_dist_road_edge":       agg["likelihood_distance_to_road_edge"],
        f"{prefix}likelihood_offroad":              agg["likelihood_offroad_indication"],
        f"{prefix}scenarios_evaluated":             scenarios,
    }


def print_progress(agg, batch, max_batches, scenarios):
    print(f"  Batch {batch}/{max_batches} | scenarios: {scenarios} | "
          f"realism: {agg['realism_meta_score']:.4f}")
    print(f"      kinematic {agg['kinematic_metrics']:.4f} | "
          f"interactive {agg['interactive_metrics']:.4f} | "
          f"map_based {agg['map_based_metrics']:.4f}")
    print(f"      lin_spd {agg['likelihood_linear_speed']:.3f} | "
          f"lin_acc {agg['likelihood_linear_acceleration']:.3f} | "
          f"ang_spd {agg['likelihood_angular_speed']:.3f} | "
          f"ang_acc {agg['likelihood_angular_acceleration']:.3f}")
    print(f"      collision {agg['likelihood_collision_indication']:.3f} | "
          f"dist_obj {agg['likelihood_distance_to_nearest_object']:.3f} | "
          f"ttc {agg['likelihood_time_to_collision']:.3f} | "
          f"dist_edge {agg['likelihood_distance_to_road_edge']:.3f} | "
          f"offroad {agg['likelihood_offroad_indication']:.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",        type=str,   required=True)
    p.add_argument("--role_dim",          type=int,   default=8)
    p.add_argument("--obs_dim",           type=int,   default=1121)
    p.add_argument("--num_agents",        type=int,   default=3072)
    p.add_argument("--goal_speed",        type=float, default=100.0)
    p.add_argument("--device",            type=str,   default="cuda")
    p.add_argument("--map_dir",           type=str,   default="resources/drive/binaries/training")
    p.add_argument("--wosac_rollouts",    type=int,   default=32)
    p.add_argument("--wosac_num_maps",    type=int,   default=10000)
    p.add_argument("--wosac_max_batches", type=int,   default=500)
    p.add_argument("--output_dir",        type=str,   default="eval_results")
    p.add_argument("--wandb",             action="store_true")
    p.add_argument("--wandb_project",     type=str,   default="roma-pufferdrive")
    p.add_argument("--wandb_run_name",    type=str,   default=None)
    return p.parse_args()


def evaluate(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or Path(args.checkpoint).stem
        wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        wandb_run.define_metric("wosac/batch")
        wandb_run.define_metric("wosac/*", step_metric="wosac/batch")

    print(f"Checkpoint : {args.checkpoint}")
    policy = load_policy(args.checkpoint, args.role_dim, args.obs_dim, device)

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(
        num_maps       = args.wosac_num_maps,
        num_agents     = args.num_agents,
        map_dir        = args.map_dir,
        episode_length = 91,
        goal_speed     = args.goal_speed,
    )

    import pandas as pd
    from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

    wosac_config = {
        "eval": {"wosac_init_steps": 0, "wosac_num_rollouts": args.wosac_rollouts},
        "train": {"device": str(device)},
    }
    evaluator        = WOSACEvaluator(wosac_config)
    adapter          = WOSACPolicyAdapter(policy, env.num_agents, device)
    all_results      = []
    unique_scenarios = set()

    print(f"rollouts={args.wosac_rollouts} | num_maps={args.wosac_num_maps} | "
          f"max_batches={args.wosac_max_batches} | num_agents={env.num_agents}\n")

    for batch in range(args.wosac_max_batches):
        env.resample_maps()
        env.reset()
        gt          = env.get_ground_truth_trajectories()
        agent_state = env.get_global_agent_state()
        road_edges  = env.get_road_edge_polylines()

        sim = collect_wosac_trajectories(env, adapter, args.wosac_rollouts)

        try:
            df  = evaluator.compute_metrics(gt, sim, agent_state, road_edges,
                                            aggregate_results=False)
            new = set(df.index.tolist()) - unique_scenarios
            if new:
                all_results.append(df[df.index.isin(new)])
                unique_scenarios.update(new)
        except Exception:
            pass

        if (batch + 1) % 10 == 0:
            if all_results:
                agg = pd.concat(all_results).mean()
                print_progress(agg, batch + 1, args.wosac_max_batches,
                               len(unique_scenarios))
                if wandb_run:
                    d = wosac_metric_dict(agg, len(unique_scenarios))
                    d["wosac/batch"] = batch + 1
                    wandb_run.log(d)

    if not all_results:
        print("No WOSAC results collected.")
        return

    combined = pd.concat(all_results)
    agg      = combined.mean()

    print("\n" + "=" * 57)
    print("  WOSAC REALISM METRICS")
    print("=" * 57)
    print(f"  Scenarios evaluated       : {len(combined)}")
    print(f"  Rollouts per scenario     : {args.wosac_rollouts}")
    print(f"  Realism meta-score        : {agg['realism_meta_score']:.4f}")
    print(f"  Kinematic metrics         : {agg['kinematic_metrics']:.4f}")
    print(f"  Interactive metrics       : {agg['interactive_metrics']:.4f}")
    print(f"  Map-based metrics         : {agg['map_based_metrics']:.4f}")
    print()
    print(f"  ADE                       : {agg['ade']:.4f} m")
    print(f"  minADE                    : {agg['min_ade']:.4f} m")
    print()
    print(f"  likelihood_linear_speed   : {agg['likelihood_linear_speed']:.4f}")
    print(f"  likelihood_linear_accel   : {agg['likelihood_linear_acceleration']:.4f}")
    print(f"  likelihood_angular_speed  : {agg['likelihood_angular_speed']:.4f}")
    print(f"  likelihood_angular_accel  : {agg['likelihood_angular_acceleration']:.4f}")
    print(f"  likelihood_collision      : {agg['likelihood_collision_indication']:.4f}")
    print(f"  likelihood_dist_obj       : {agg['likelihood_distance_to_nearest_object']:.4f}")
    print(f"  likelihood_ttc            : {agg['likelihood_time_to_collision']:.4f}")
    print(f"  likelihood_dist_road_edge : {agg['likelihood_distance_to_road_edge']:.4f}")
    print(f"  likelihood_offroad        : {agg['likelihood_offroad_indication']:.4f}")
    print("=" * 57)

    if wandb_run:
        final = wosac_metric_dict(agg, len(combined))
        final["wosac/batch"] = args.wosac_max_batches
        wandb_run.log(final)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.output_dir) / f"wosac_{Path(args.checkpoint).stem}.csv"
    combined.to_csv(csv_path)
    print(f"\n  Results saved -> {csv_path}")


if __name__ == "__main__":
    evaluate(parse_args())
