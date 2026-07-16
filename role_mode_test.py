"""
role_mode_test.py -- is the BIMODAL role two real DRIVER TYPES, or is the
encoder just reading the SCENE?

Splits natural (unforced) agents into two role modes along PC1 (KMeans k=2 on
the PC1 projection; axes loaded from the paired sweep's role_paired_axes.csv so
PC1 = the same direction as every other figure -- for dim-1, PC1 = role_0), and
asks whether the two modes differ in DRIVING once the scene is held fixed.

Tests (printed + written to role_mode_test.csv):
  RAW         mode A vs B behavioural means over ALL agents (confounded).
  CONFOUND    each mode's mean GT (human) speed -- if the modes differ in the
              HUMAN's speed, the split is largely tracking the scene/route,
              NOT a driving disposition.
  SAME-SCENE  within scenes that contain BOTH modes, the paired (A - B)
              behaviour difference across scenes, with a paired t. This is the
              headline: a difference that SURVIVES within-scene is a real type;
              one that VANISHES was a pure scene artifact. GT-speed is included
              in the same-scene panel too -- if even within a scene mode-A's
              humans drove faster, the routes still differ (sub-task confound);
              if GT-speed matches within-scene but the POLICY behaviour differs,
              that is the cleanest "real style" signal.

Observational still (natural roles are obs-conditioned) -- the fully causal
test is forcing the two mode centroids (Test 3). But same-scene pairing removes
the scene-LEVEL confound, which raw profiling cannot.

No training. CPU is fine (like role_natural_extremes). Run from PufferDrive:
  python role_mode_test.py \
      --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim1/roma_dim1_final.pt \
      --axes_csv   /scratch/e452103/role_paired/dim1/role_paired_axes.csv \
      --data_dir   pufferlib/resources/drive/binaries/training \
      --out_dir    /scratch/e452103/role_paired/dim1
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import load_policy
from role_natural_extremes import natural_rollout
from render_role_alpha import load_axes
from role_regime_analysis import _squeeze
from traj_kinematics import ego_kinematics, wrap_angle

T = 91
TELEPORT_M = 4.0
BEHAV = ["speed_mean", "accel_pos", "decel_abs", "jerk_abs", "turn_abs"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--axes_csv",   type=str, required=True)
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--warmup_episodes", type=int, default=6)
    p.add_argument("--map_pool",   type=int, default=10000)
    p.add_argument("--total_agents", type=int, default=2048)
    p.add_argument("--min_speed",  type=float, default=0.5,
                   help="m/s; skip agents that never really move")
    p.add_argument("--min_pair",   type=int, default=1,
                   help="min agents of EACH mode for a scene to enter the "
                        "same-scene test")
    p.add_argument("--seed_start", type=int, default=100)
    p.add_argument("--device",     type=str, default="cpu")
    return p.parse_args()


def gt_speed_of(gt, a):
    """Mean human GT speed (m/s) of agent a over its valid consecutive steps."""
    gx = _squeeze(gt["x"])[a].astype(np.float64)
    gy = _squeeze(gt["y"])[a].astype(np.float64)
    gv = _squeeze(gt["valid"])[a].astype(bool)
    pair = gv[:-1] & gv[1:]
    if pair.sum() < 3:
        return np.nan
    spd = np.hypot(np.diff(gx), np.diff(gy))[pair] * 10.0
    spd = spd[spd <= 45.0]                       # plausibility
    return float(spd.mean()) if len(spd) else np.nan


def collect(env, policy, device, args, mu, u1):
    """Per-agent natural (role, behaviour, gt_speed, sid, vid) rows."""
    rows, seen = [], set()
    for ep in range(args.warmup_episodes):
        if ep > 0:
            env.resample_maps()
        xs, ys, hs, roles, gt = natural_rollout(env, policy, device)
        sids   = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
        vids   = _squeeze(np.asarray(gt["id"])).reshape(-1)
        is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
        step_d = np.hypot(np.diff(xs, axis=0), np.diff(ys, axis=0))
        for a in range(env.num_agents):
            sid = str(sids[a])
            if (not is_veh[a] or not sid or sid.lower().startswith("map")
                    or (sid, int(vids[a])) in seen):
                continue
            jumps = np.where(step_d[:, a] > TELEPORT_M)[0]
            end   = int(jumps[0] + 1) if len(jumps) else T
            if end < 10:
                continue
            spd = step_d[:end - 1, a] * 10.0
            dh  = wrap_angle(np.diff(hs[:end, a])) * 10.0
            n   = min(len(spd), len(dh))
            k   = ego_kinematics(spd[:n], dh[:n])
            if not np.isfinite(k["speed_mean"]) or k["speed_mean"] < args.min_speed:
                continue
            pc1 = float((roles[:end, a].mean(axis=0) - mu) @ u1)
            rows.append({"sid": sid, "vid": int(vids[a]), "pc1": pc1,
                         "gt_speed": gt_speed_of(gt, a),
                         **{m: k[m] for m in BEHAV}})
            seen.add((sid, int(vids[a])))
        print(f"[mode] warmup {ep+1}/{args.warmup_episodes}: {len(rows)} agents",
              flush=True)
    return rows


def paired_within_scene(df, cols, min_pair):
    """For scenes with >= min_pair agents of EACH mode, the per-scene
    (meanA - meanB) difference, then paired across scenes. Returns a row per
    column with mean diff, paired t, n_scenes."""
    out = []
    scenes = []
    for sid, g in df.groupby("sid"):
        a = g[g["mode"] == 0]; b = g[g["mode"] == 1]
        if len(a) >= min_pair and len(b) >= min_pair:
            scenes.append({c: a[c].mean() - b[c].mean() for c in cols})
    if not scenes:
        return out, 0
    import pandas as pd
    sd = pd.DataFrame(scenes)
    n = len(sd)
    for c in cols:
        d = sd[c].dropna().values
        if len(d) < 3:
            continue
        m = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(len(d)))
        t = m / se if se > 0 else np.nan
        out.append({"metric": c, "within_scene_meanA_minus_B": round(m, 4),
                    "paired_t": round(float(t), 1), "n_scenes": len(d)})
    return out, n


def main():
    args = parse_args()
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from sklearn.cluster import KMeans

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(num_maps=args.map_pool, num_agents=args.total_agents,
                map_dir=args.data_dir, episode_length=T, goal_speed=100,
                seed=args.seed_start)
    obs_probe, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs_probe.shape[-1], device)
    if role_dim == 0:
        raise SystemExit("role_dim=0 checkpoint -- no role to split")
    mu, axes = load_axes(args.axes_csv, role_dim)
    if "PC1" not in axes:
        raise SystemExit(f"axes csv needs PC1, has {list(axes)}")
    u1, _ = axes["PC1"]

    rows = collect(env, policy, device, args, mu, u1)
    env.close()
    df = pd.DataFrame(rows)
    if len(df) < 20:
        raise SystemExit(f"only {len(df)} agents collected -- raise "
                         f"--warmup_episodes/--total_agents")

    # -- split into 2 modes along PC1 -----------------------------------------
    lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(
        df["pc1"].values.reshape(-1, 1))
    # order modes so mode 0 = LOWER PC1 (consistent labelling across runs)
    if df["pc1"].values[lab == 0].mean() > df["pc1"].values[lab == 1].mean():
        lab = 1 - lab
    df["mode"] = lab
    df.to_csv(out_dir / "role_mode_agents.csv", index=False)

    cols = BEHAV + ["gt_speed"]
    n0, n1 = int((lab == 0).sum()), int((lab == 1).sum())

    # -- RAW profile ----------------------------------------------------------
    raw = df.groupby("mode")[cols + ["pc1"]].mean()
    print("\n=== RAW mode profile (ALL agents; confounded) ===")
    print(f"mode 0 n={n0}   mode 1 n={n1}")
    print(raw.round(3).to_string())

    # -- CONFOUND: does GT (human) speed differ by mode? ----------------------
    g0, g1 = df.loc[lab == 0, "gt_speed"].mean(), df.loc[lab == 1, "gt_speed"].mean()
    print("\n=== CONFOUND CHECK: GT (human) speed per mode ===")
    print(f"mode0 gt_speed={g0:.2f}  mode1 gt_speed={g1:.2f}  "
          f"diff={g0 - g1:+.2f} m/s")
    print("  large GT-speed gap => the split largely tracks the SCENE/route, "
          "not a disposition.")

    # -- SAME-SCENE within-pair test (the headline) ---------------------------
    within, n_scenes = paired_within_scene(df, cols, args.min_pair)
    print(f"\n=== SAME-SCENE test: {n_scenes} scenes contain BOTH modes "
          f"(>={args.min_pair} each) ===")
    if within:
        wdf = pd.DataFrame(within)
        print(wdf.to_string(index=False))
        wdf.to_csv(out_dir / "role_mode_same_scene.csv", index=False)
        print("\n  |t|>~2 on speed/throttle/brake/jerk/turn => the modes differ "
              "WITHIN the same scene = a real driver-type distinction the scene "
              "cannot explain. If only gt_speed differs within-scene, the routes "
              "still differ (sub-task confound). If everything ~0, the modes were "
              "a pure scene artifact.")
    else:
        print("  no scenes had both modes present -- raise --warmup_episodes or "
              "lower --min_pair; the two modes may be nearly scene-disjoint "
              "(itself evidence the split is scene-driven).")

    print(f"\n[mode] wrote role_mode_agents.csv"
          + ("  + role_mode_same_scene.csv" if within else "")
          + f" -> {out_dir}")


if __name__ == "__main__":
    main()
