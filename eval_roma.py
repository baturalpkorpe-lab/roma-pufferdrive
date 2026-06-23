"""
eval_roma.py — WOSAC realism evaluation for ROMA checkpoints.

Runs two things in one go:
  1. Lane-coverage diagnostic — parses the map binaries and reports whether the
     maps have lane centerlines through intersections/turns (decides whether a
     lane-centering reward would have guidance through junctions).
  2. WOSAC realism metrics — realism meta-score + all 9 sub-metrics.

Usage (from ~/PufferDrive):
    PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 eval_roma.py \
        --checkpoint roma_main/checkpoints/roma_dim8/roma_dim8_final.pt \
        --wosac_rollouts 32 --wosac_num_maps 10000 --wosac_max_batches 500 \
        --wandb

    # only run the lane diagnostic (no policy / WOSAC):
    python3 eval_roma.py --checkpoint x --lane_only
"""

import argparse
import glob
import math
import os
import struct
import numpy as np
import torch
from pathlib import Path
from torch.distributions import Categorical


# ===========================================================================
# Lane-coverage diagnostic — parses map binaries directly (no pufferlib).
# Answers: do the maps have lane centerlines through intersections/turns?
# ===========================================================================

VEHICLE, PEDESTRIAN, CYCLIST = 1, 2, 3
ROAD_LANE, ROAD_LINE, ROAD_EDGE = 4, 5, 6
_OBJECT_TYPES = {VEHICLE, PEDESTRIAN, CYCLIST}
_LANE_THRESHOLD = 4.0  # drive.h: agents > 4 m from any lane are "not aligned"


class _MapReader:
    """Cursor over a map binary (little-endian), matching drive.h load_map_binary."""
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def i32(self):
        v = struct.unpack_from("<i", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def f32(self, n):
        a = np.frombuffer(self.buf, dtype="<f4", count=n, offset=self.pos)
        self.pos += 4 * n
        return a

    def skip(self, nbytes):
        self.pos += nbytes


def _parse_map(path):
    """Return (lanes, vehicles): lanes=[(x,y)], vehicles=[(x,y,valid)]."""
    with open(path, "rb") as f:
        r = _MapReader(f.read())
    r.i32()                                  # sdc_track_index
    n_ttp = r.i32()
    r.skip(4 * n_ttp)                        # tracks_to_predict_indices
    num_objects = r.i32()
    num_roads = r.i32()

    lanes, vehicles = [], []
    for _ in range(num_objects + num_roads):
        r.i32()                             # scenario_id
        etype = r.i32()
        r.i32()                             # id
        size = r.i32()
        x = r.f32(size).copy()
        y = r.f32(size).copy()
        r.skip(4 * size)                    # z
        if etype in _OBJECT_TYPES:
            r.skip(4 * size * 4)            # vx, vy, vz, heading
            valid = np.frombuffer(r.f32(size).tobytes(), dtype="<i4").copy()
        else:
            valid = None
        r.skip(4 * 6)                       # width,length,height,goalx,goaly,goalz
        r.skip(4)                           # mark_as_expert
        if etype == ROAD_LANE:
            lanes.append((x, y))
        elif etype == VEHICLE:
            vehicles.append((x, y, valid))
    return lanes, vehicles


def _total_heading_change(x, y):
    if len(x) < 3:
        return 0.0
    h = np.arctan2(np.diff(y), np.diff(x))
    dh = (np.diff(h) + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(np.abs(dh)))


def _build_segments(lanes):
    starts, ends = [], []
    for x, y in lanes:
        if len(x) < 2:
            continue
        pts = np.stack([x, y], axis=1)
        starts.append(pts[:-1])
        ends.append(pts[1:])
    if not starts:
        return None, None
    return np.concatenate(starts, 0), np.concatenate(ends, 0)


def _min_dist_points_to_segments(P, A, B, chunk=512):
    out = np.empty(len(P), dtype=np.float32)
    AB = B - A
    denom = np.einsum("md,md->m", AB, AB) + 1e-9
    for i in range(0, len(P), chunk):
        p = P[i:i + chunk]
        ap = p[:, None, :] - A[None, :, :]
        t = np.clip(np.einsum("cmd,md->cm", ap, AB) / denom, 0.0, 1.0)
        proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]
        out[i:i + chunk] = np.linalg.norm(p[:, None, :] - proj, axis=2).min(axis=1)
    return out


def run_lane_diagnostic(map_dir, num_maps=30, curve_deg=30.0, turn_deg_per_step=3.0):
    """Print lane-centerline coverage report (entity counts, turn connectors,
    and how far human trajectories are from lanes split by straight/turning)."""
    files = sorted(glob.glob(os.path.join(map_dir, "*.bin")))[:num_maps]
    print("\n" + "=" * 57)
    print("  LANE-COVERAGE DIAGNOSTIC")
    print("=" * 57)
    if not files:
        print(f"  No .bin files in {map_dir}")
        return {}

    curve_thr = math.radians(curve_deg)
    turn_thr = math.radians(turn_deg_per_step)
    n_lanes_total = n_veh_total = 0
    lane_curv, lane_len = [], []
    d_straight, d_turn = [], []

    for path in files:
        try:
            lanes, vehicles = _parse_map(path)
        except Exception:
            continue  # skip a truncated / unexpected map file rather than crash
        n_lanes_total += len(lanes)
        n_veh_total += len(vehicles)
        for x, y in lanes:
            lane_curv.append(_total_heading_change(x, y))
            lane_len.append(float(np.sum(np.hypot(np.diff(x), np.diff(y)))))

        A, B = _build_segments(lanes)
        if A is None:
            continue
        pts, turning = [], []
        for x, y, valid in vehicles:
            if valid is None or valid.astype(bool).sum() < 3:
                continue
            v = valid.astype(bool)
            xy = np.stack([x, y], axis=1)
            h = np.arctan2(np.diff(y), np.diff(x))
            dh = np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi)
            for k in range(1, len(x) - 1):
                if v[k] and v[k - 1] and v[k + 1]:
                    pts.append(xy[k])
                    turning.append(dh[k - 1] > turn_thr)
        if not pts:
            continue
        d = _min_dist_points_to_segments(np.asarray(pts, dtype=np.float32), A, B)
        turning = np.asarray(turning, dtype=bool)
        d_straight.append(d[~turning])
        d_turn.append(d[turning])

    m = {}  # metrics dict (returned for wandb logging)
    m["lane/lanes_per_map"]    = n_lanes_total / len(files)
    m["lane/vehicles_per_map"] = n_veh_total / len(files)
    print(f"  maps parsed : {len(files)}")
    print(f"  ROAD_LANE   : {n_lanes_total} ({m['lane/lanes_per_map']:.0f}/map)")
    print(f"  VEHICLE     : {n_veh_total} ({m['lane/vehicles_per_map']:.0f}/map)")

    if lane_curv:
        lc, ll = np.array(lane_curv), np.array(lane_len)
        n_curved = int(np.sum(lc > curve_thr))
        m["lane/turn_connector_pct"] = 100 * n_curved / len(lc)
        m["lane/bend_median_deg"]    = math.degrees(np.median(lc))
        m["lane/bend_max_deg"]       = math.degrees(lc.max())
        m["lane/length_median_m"]    = float(np.median(ll))
        print(f"\n  turn connectors (>{curve_deg:.0f}deg bend): {n_curved} "
              f"({m['lane/turn_connector_pct']:.1f}% of lanes)")
        print(f"  lane bend: median {m['lane/bend_median_deg']:.1f}deg | "
              f"max {m['lane/bend_max_deg']:.1f}deg")
        print(f"  lane length: median {m['lane/length_median_m']:.1f}m")

    ds = np.concatenate(d_straight) if d_straight else np.array([])
    dt = np.concatenate(d_turn) if d_turn else np.array([])
    print("\n  human distance to nearest lane centerline:")
    for label, key, d in (("straight", "straight", ds), ("turning ", "turning", dt)):
        if len(d) == 0:
            print(f"    {label}: (none)")
            continue
        m[f"lane/{key}_dist_median_m"] = float(np.median(d))
        m[f"lane/{key}_dist_90pct_m"]  = float(np.percentile(d, 90))
        m[f"lane/{key}_pct_over_4m"]   = 100 * float(np.mean(d > _LANE_THRESHOLD))
        print(f"    {label}: n={len(d):>7} | median {m[f'lane/{key}_dist_median_m']:.2f}m | "
              f"90pct {m[f'lane/{key}_dist_90pct_m']:.2f}m | "
              f">{_LANE_THRESHOLD:.0f}m: {m[f'lane/{key}_pct_over_4m']:.1f}%")
    if len(dt):
        far = float(np.mean(dt > _LANE_THRESHOLD))
        m["lane/turns_covered"] = 1.0 if far < 0.15 else 0.0
        verdict = ("turns ARE covered - lane-centering reward works through junctions"
                   if far < 0.15 else
                   "turns POORLY covered - reward gives 0 through many turns")
        print(f"\n  => {100*far:.1f}% of human TURNING points are >4m from any lane")
        print(f"     {verdict}")
    print("=" * 57)
    return m


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
    # Lane-coverage diagnostic
    p.add_argument("--lane_maps",         type=int,   default=30,
                   help="Maps to analyze for the lane-coverage diagnostic. 0 = skip.")
    p.add_argument("--lane_only",         action="store_true",
                   help="Run only the lane diagnostic, then exit (no policy / WOSAC).")
    return p.parse_args()


def evaluate(args):
    # wandb is initialized first so the lane diagnostic can log to it too.
    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or Path(args.checkpoint).stem
        wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        wandb_run.define_metric("wosac/batch")
        wandb_run.define_metric("wosac/*", step_metric="wosac/batch")

    # --- Part 1: lane-coverage diagnostic (no policy / env needed) ---
    # Non-fatal: a diagnostic failure must never block the WOSAC eval.
    if args.lane_maps > 0:
        try:
            lane_metrics = run_lane_diagnostic(args.map_dir, args.lane_maps)
            if wandb_run and lane_metrics:
                wandb_run.log(lane_metrics)
        except Exception as e:
            print(f"[lane diagnostic skipped: {e}]")
    if args.lane_only:
        if wandb_run:
            wandb_run.finish()
        return

    # --- Part 2: WOSAC realism metrics ---
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"Checkpoint : {args.checkpoint}")
    policy = load_policy(args.checkpoint, args.role_dim, args.obs_dim, device)

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(
        num_maps             = args.wosac_num_maps,
        num_agents           = args.num_agents,
        map_dir              = args.map_dir,
        episode_length       = 91,
        goal_speed           = args.goal_speed,
        goal_target_distance = 30.0,   # drive.ini default; Drive() Python default is 10.0
        termination_mode     = 1,      # drive.ini default; Drive() Python default is None->0
        resample_frequency   = 910,    # CRITICAL: default 91 triggers auto-resample at every
                                       # rollout's last step (tick=91), changing maps mid-batch
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
