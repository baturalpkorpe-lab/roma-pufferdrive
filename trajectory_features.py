"""
trajectory_features.py -- Phase A of the TRAJECTORY-level stratification.

One feature row per (scenario_id, vehicle_id) from the HUMAN GT trajectory --
never the policy rollout (GT is exogenous: the strata must describe the task a
seat was given, not the behavior we later analyze). Uses ONLY reset-time data
(no policy, no GPU). Features (see traj_kinematics.gt_traj_features):

    distance     m driven over kept steps
    speed_mean   m/s
    speed_max    m/s
    speed_min    m/s (the stop detector)
    net_turn     |total heading change| rad -- route shape, no L/R distinction

Step-level physical plausibility masking is applied (speed/turn caps);
trajectories with >20% flagged steps are dropped as corrupt -- this REPLACES
the old map-level junk-cluster quarantine at the right granularity.

Also caches a sample of raw (x, y, speed) paths so trajectory_atlas.py can
draw real example trajectories per cluster.

Usage (from /scratch/e452103/PufferDrive so the C binding finds drive.ini):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/trajectory_features.py \
        --data_dir pufferlib/resources/drive/binaries/training \
        --out_dir  /scratch/e452103/traj_atlas
"""

import argparse
import ast
import configparser
import csv
import os
import pickle
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from traj_kinematics import gt_traj_features

FEATURES = ["distance", "speed_mean", "speed_max", "speed_min", "net_turn",
            "stop_frac"]
EXTRAS   = ["n_steps", "flag_frac"]           # bookkeeping, not for clustering


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
    p.add_argument("--data_dir",      type=str, required=True)
    p.add_argument("--out_dir",       type=str, default="traj_atlas")
    p.add_argument("--num_agents",    type=int, default=3072)
    p.add_argument("--num_maps",      type=int, default=10000)
    p.add_argument("--max_batches",   type=int, default=100)
    p.add_argument("--target_scenarios", type=int, default=9500,
                   help="Stop once this many unique scenarios are covered")
    p.add_argument("--cache_per_batch", type=int, default=8,
                   help="Trajectories per batch whose raw path is cached "
                        "for the atlas example plots")
    return p.parse_args()


def _squeeze(a):
    a = np.asarray(a)
    return a[:, 0] if a.ndim >= 2 and a.shape[1] == 1 else a


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path   = out_dir / "trajectory_features.csv"
    cache_path = out_dir / "trajectory_cache.pkl"

    from pufferlib.ocean.drive.drive import Drive
    env_cfg = dict(load_drive_config()["env"])
    env_cfg.update({"num_maps": args.num_maps, "num_agents": args.num_agents,
                    "map_dir": args.data_dir})
    t0 = time.time()
    env = Drive(**env_cfg)
    print(f"[trajA] env created in {time.time()-t0:.0f}s")

    rows       = {}     # (sid, vid) -> feature dict
    done_sids  = set()  # scenarios fully processed once
    traj_cache = {}     # (sid, vid) -> {"x", "y", "spd"} for example plots
    n_dropped  = 0      # parked / too-short / corrupt

    for batch in range(args.max_batches):
        if batch > 0:
            env.resample_maps()
        env.reset()

        gt   = env.get_ground_truth_trajectories()
        gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
        gh     = _squeeze(gt["heading"])
        valid  = _squeeze(gt["valid"]).astype(bool)
        is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
        sids   = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
        vids   = _squeeze(np.asarray(gt["id"])).reshape(-1)

        new, cached = 0, 0
        for a in range(gx.shape[0]):
            sid = sids[a]
            if (not sid or sid in done_sids or not is_veh[a]
                    or str(sid).lower().startswith("map")):
                continue
            key = (sid, int(vids[a]))
            if key in rows:
                continue
            f = gt_traj_features(gx[a], gy[a], gh[a], valid[a])
            if f is None:
                n_dropped += 1
                continue
            rows[key] = f
            new += 1
            if cached < args.cache_per_batch:
                m = valid[a]
                px, py = gx[a][m].astype(np.float32), gy[a][m].astype(np.float32)
                traj_cache[key] = {"x": px, "y": py,
                                   "spd": float(f["speed_mean"])}
                cached += 1
        done_sids.update(s for s in np.unique(sids) if s)

        print(f"[trajA] batch {batch+1:>3}: +{new:>5} new  "
              f"rows={len(rows):>6}  scenarios={len(done_sids):>5}  "
              f"dropped={n_dropped}", flush=True)
        if len(done_sids) >= args.target_scenarios:
            print(f"[trajA] target of {args.target_scenarios} scenarios reached")
            break

    env.close()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "vehicle_id"] + FEATURES + EXTRAS)
        for (sid, vid), feat in rows.items():
            w.writerow([sid, vid] + [feat[k] for k in FEATURES + EXTRAS])
    with open(cache_path, "wb") as f:
        pickle.dump(traj_cache, f)

    print(f"\n[trajA] {len(rows)} trajectories from {len(done_sids)} scenarios "
          f"-> {csv_path}")
    print(f"[trajA] {len(traj_cache)} raw paths cached -> {cache_path}")
    print(f"[trajA] dropped (parked/short/corrupt): {n_dropped}")


if __name__ == "__main__":
    main()
