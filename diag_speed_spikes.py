"""
diag_speed_spikes.py -- diagnose the unrealistic speed_max / jerk_abs values in
role_direct_cluster.py.

Hypothesis: TELEPORT_M=8.0 m/step only catches jumps > 80 m/s, so respawns that
land 4-8 m away survive the segment cut and inject 40-80 m/s single-step spikes
(and splice two trajectory pieces -> huge accel/jerk). This script reproduces the
EXACT segment-cutting logic used for the features, finds agents whose in-segment
speed_max is implausible, and prints the per-step trace around the spike (step
distance, speed, GT-valid, reward) so we can see whether it is a lone respawn
discontinuity coinciding with a goal/reward event.

Run on the cluster (needs the C env + a checkpoint), short (~3 episodes):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/diag_speed_spikes.py \
        --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_final.pt \
        --data_dir   pufferlib/resources/drive/binaries/training
"""

import argparse
import ast
import configparser
import os

import numpy as np
import torch
from torch.distributions import Categorical

T = 91
TELEPORT_M = 8.0     # the current threshold in role_direct_cluster.py
MIN_STEPS  = 10
PLAUSIBLE_MS = 30.0  # ~108 km/h: above this, a per-step "speed" is suspect here


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


def load_policy(ckpt_path, obs_dim, device):
    from roma_pufferdrive.roma.policy import RomaPolicy
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {}) or {}
    policy = RomaPolicy(
        obs_dim=obs_dim, action_dim=91, role_dim=saved.get("role_dim", 8),
        role_hidden=saved.get("role_hidden", 64),
        policy_hidden=saved.get("policy_hidden", 128),
        var_floor=saved.get("var_floor", 1e-4), obs_window_len=8,
    ).to(device)
    key = "policy_state" if "policy_state" in ckpt else "policy"
    policy.load_state_dict(ckpt[key])
    policy.eval()
    return policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--episodes",   type=int, default=3)
    p.add_argument("--num_agents", type=int, default=3072)
    p.add_argument("--num_maps",   type=int, default=10000)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--n_examples", type=int, default=12,
                   help="How many worst offenders to print full traces for")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    from pufferlib.ocean.drive.drive import Drive
    env_cfg = dict(load_drive_config()["env"])
    env_cfg.update({"num_maps": args.num_maps, "num_agents": args.num_agents,
                    "map_dir": args.data_dir})
    env = Drive(**env_cfg)
    B = args.num_agents

    obs_np, _ = env.reset()
    policy = load_policy(args.checkpoint, obs_np.shape[-1], device)

    all_step_speeds = []          # every in-segment valid per-step speed (m/s)
    spike_agents    = []          # (speed_max, n_over, ep, a, t_star, traces)
    n_usable = 0

    for ep in range(args.episodes):
        if ep > 0:
            env.resample_maps()
        obs_np, _ = env.reset()
        gt     = env.get_ground_truth_trajectories()
        gvalid = _squeeze(gt["valid"]).astype(bool)
        is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
        T_gt   = _squeeze(gt["x"]).shape[1]

        xs = np.zeros((T, B), np.float32); ys = np.zeros((T, B), np.float32)
        rews = np.zeros((T, B), np.float32)
        obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        state = policy.initial_state(B, device)
        for t in range(T):
            ag = env.get_global_agent_state()
            xs[t], ys[t] = ag["x"], ag["y"]
            with torch.no_grad():
                logits, _, state, _ = policy(obs, state)
            action = Categorical(logits=logits.float()).sample()
            obs_np, rew_np, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
            rews[t] = np.asarray(rew_np).reshape(B)
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        step_d = np.sqrt(np.diff(xs, axis=0)**2 + np.diff(ys, axis=0)**2)  # (T-1,B)
        for a in range(B):
            if not is_veh[a]:
                continue
            # EXACT segment logic from role_direct_cluster.collect()
            jumps = np.where(step_d[:, a] > TELEPORT_M)[0]
            t_end = int(jumps[0] + 1) if len(jumps) else T
            t_end = min(t_end, T_gt)
            if t_end < MIN_STEPS + 1:
                continue
            v = gvalid[a, :t_end]
            pair = v[:-1] & v[1:]
            if pair.sum() < MIN_STEPS:
                continue
            n_usable += 1
            sd_seg = step_d[:t_end - 1, a][pair]      # step distances, valid pairs
            spd    = sd_seg * 10.0                     # m/s
            all_step_speeds.append(spd)
            if spd.max() > PLAUSIBLE_MS:
                # locate the spike in absolute step index
                idx_in_pair = int(np.argmax(spd))
                pair_pos    = np.where(pair)[0][idx_in_pair]   # step k -> k..k+1
                w = range(max(0, pair_pos - 3), min(t_end, pair_pos + 4))
                trace = [(k, float(step_d[k, a]) if k < t_end - 1 else np.nan,
                          bool(gvalid[a, k]), float(rews[k, a])) for k in w]
                spike_agents.append((float(spd.max()),
                                     int((spd > PLAUSIBLE_MS).sum()),
                                     ep, a, pair_pos, trace))

    env.close()

    speeds = np.concatenate(all_step_speeds)
    print("\n================ per-step speed distribution (m/s) ================")
    for q in [50, 90, 99, 99.9, 100]:
        print(f"  p{q:<5}: {np.percentile(speeds, q):6.2f}")
    for lo, hi in [(0, 30), (30, 40), (40, 60), (60, 80), (80, 1e9)]:
        n = int(((speeds >= lo) & (speeds < hi)).sum())
        print(f"  steps in [{lo:>3},{hi if hi<1e9 else 'inf':>4}) m/s: "
              f"{n:>8}  ({100*n/len(speeds):.3f}%)")

    print(f"\n[diag] usable agent-segments: {n_usable}")
    print(f"[diag] segments with in-segment speed > {PLAUSIBLE_MS:.0f} m/s "
          f"(survived the {TELEPORT_M:.0f} m teleport cut): {len(spike_agents)} "
          f"({100*len(spike_agents)/max(n_usable,1):.2f}%)")

    spike_agents.sort(reverse=True)
    print(f"\n============ {min(args.n_examples, len(spike_agents))} worst "
          f"offenders (trace around the spike step) ============")
    print("  each row: step  step_dist(m)  gt_valid  reward   [>>> = the spike]")
    for smax, n_over, ep, a, tstar, trace in spike_agents[:args.n_examples]:
        print(f"\n  ep{ep} agent {a}: in-seg speed_max={smax:.1f} m/s "
              f"(={smax/10:.2f} m/step), #steps>{PLAUSIBLE_MS:.0f}m/s = {n_over}")
        for (k, sd, val, rew) in trace:
            mark = "  <<< spike" if k == tstar else ""
            sds  = f"{sd:6.2f}" if np.isfinite(sd) else "   n/a"
            print(f"      t={k:>2}  d={sds}  valid={int(val)}  "
                  f"rew={rew:+.3f}{mark}")

    # what a tighter threshold would save
    for cap in [3.0, 3.5, 4.0]:
        surv = float((speeds > cap * 10).mean())
        print(f"\n[diag] if a step-jump > {cap:.1f} m ({cap*10:.0f} m/s) were "
              f"treated as a discontinuity: {100*surv:.3f}% of current per-step "
              f"speeds would be cut as artifacts")


if __name__ == "__main__":
    main()
