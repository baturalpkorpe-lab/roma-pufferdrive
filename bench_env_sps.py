"""
bench_env_sps.py -- Benchmark raw Drive env stepping speed (no policy, no GPU).

Isolates SPS regressions: if this number changed between runs, the slowdown is
in the PufferDrive C build / env config, not in the training loop.

Prints which pufferlib install and binding .so are in use (with build date), so
two stacks can be compared unambiguously.

Usage:
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python bench_env_sps.py \
        --data_dir /scratch/e452103/PufferDrive/pufferlib/resources/drive/binaries/training
"""

import argparse
import datetime
import glob
import os
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--num_agents", type=int, default=3072)
    p.add_argument("--num_maps",   type=int, default=10000)
    p.add_argument("--steps",      type=int, default=2000)
    p.add_argument("--warmup",     type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()

    import pufferlib
    puffer_dir = os.path.dirname(pufferlib.__file__)
    print(f"pufferlib      : {puffer_dir}")
    for so in glob.glob(os.path.join(puffer_dir, "ocean", "drive", "binding*.so")):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(so))
        print(f"binding        : {so}")
        print(f"binding built  : {mtime:%Y-%m-%d %H:%M}")

    from pufferlib.ocean.drive.drive import Drive

    t0 = time.time()
    env = Drive(num_maps=args.num_maps, num_agents=args.num_agents,
                map_dir=args.data_dir)
    env.reset()
    print(f"env creation   : {time.time() - t0:.1f}s "
          f"({args.num_maps} maps, {args.num_agents} agents)")

    B   = args.num_agents
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 91, size=(args.steps + args.warmup, B, 1),
                           dtype=np.int32)

    for t in range(args.warmup):
        env.step(actions[t])

    t0 = time.time()
    for t in range(args.warmup, args.warmup + args.steps):
        env.step(actions[t])
    dt = time.time() - t0

    env_sps   = args.steps / dt
    agent_sps = env_sps * B
    print(f"\nenv.step calls : {args.steps} in {dt:.1f}s")
    print(f"env steps/s    : {env_sps:,.1f}")
    print(f"agent steps/s  : {agent_sps:,.0f}   <-- compare this across stacks")
    print("\nNote: training SPS is lower than this (policy forward + PPO update);"
          "\nbut if THIS number dropped ~40%, the env/C-build is the regression.")

    env.close()


if __name__ == "__main__":
    main()
