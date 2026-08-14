"""role_on_human.py -- what role does the encoder assign to a HUMAN driver,
and do humans get a spread of roles or all the same one?

WHY IT MATTERS
The role-conditioned BC anchor only works if the encoder assigns human drivers
a range of roles. A plain BC anchor pulls every agent toward the average human
and squeezes the style dial out; conditioning the anchor on the role fixes
that -- but only if `z` actually distinguishes one human from another. If every
human gets the same role, the conditioning is vacuous and the roles would have
to be fitted to human style directly instead.

Measured precondition (human_style_spread.py): human drivers in the SAME SCENE
differ substantially in following distance, so the variation is there to be
captured. This asks whether the encoder captures it.

THE PART THAT NEEDS VERIFYING, AND HOW IT VERIFIES ITSELF
The role encoder is observation-conditioned, so a human's role means "the role
this encoder emits while watching a human drive". That requires the env to step
those agents from their logged trajectories rather than from the policy. Which
control_mode does that is not something to take on faith -- so the script does
not.

Every agent's observed path is compared against its ground-truth path. Agents
whose paths match to within --ade_tol were replayed from logs, and their roles
are human roles. Agents that diverged were policy-driven and are dropped. If
almost nothing was replayed, the control mode was wrong and the script says so
instead of reporting numbers that look fine and mean nothing.

    python role_on_human.py \
        --checkpoint /scratch/$USER/checkpoints/<run>/roma_dim1_final.pt \
        --gt_regimes /scratch/$USER/regimes/regimes_gt.csv \
        --out_dir    /scratch/$USER/role_on_human/<tag>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import load_policy

T = 91


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir",
                   default="pufferlib/resources/drive/binaries/training")
    p.add_argument("--gt_regimes", default="",
                   help="regimes_gt.csv -- joined on (scenario_id, vehicle_id) "
                        "so the role can be correlated against the human's own "
                        "measured style parameters")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--map_pool", type=int, default=1000)
    p.add_argument("--total_agents", type=int, default=3072)
    p.add_argument("--control_mode", default="control_sdc_only",
                   help="the mode that steps non-controlled agents from logs. "
                        "If the replay check below fails, try control_wosac or "
                        "control_mixed_play -- the check is what tells you.")
    p.add_argument("--ade_tol", type=float, default=1.0,
                   help="metres of mean deviation from ground truth below "
                        "which an agent counts as log-replayed")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _sq(a):
    a = np.asarray(a)
    return a.reshape(-1) if a.ndim > 1 and 1 in a.shape else a


def as_agent_time(arr, B):
    """GT per-step arrays -> (B, T).

    They arrive with a leading or trailing singleton dimension, e.g.
    (1, B, T) or (B, T, 1), and in either orientation. Squeeze first, then
    orient on B -- guessing the layout without squeezing gave a (B, T, 1)
    that summed along the wrong axis and produced a (512, 91) count.
    """
    a = np.squeeze(np.asarray(arr, dtype=float))
    if a.ndim == 1:
        a = a.reshape(B, -1)
    elif a.ndim > 2:
        a = a.reshape(B, -1) if a.shape[0] == B else a.reshape(-1, B).T
    return a if a.shape[0] == B else a.T


def main():
    a = parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(num_maps=a.map_pool, num_agents=a.total_agents,
                map_dir=a.data_dir, episode_length=T,
                control_mode=a.control_mode)
    obs_np, _ = env.reset()
    policy, role_dim = load_policy(a.checkpoint, obs_np.shape[-1], device)
    if role_dim == 0:
        raise SystemExit("role_dim=0 -- nothing to measure")
    print("[human] control_mode=%s  role_dim=%d  agents=%d"
          % (a.control_mode, role_dim, env.num_agents))

    B = env.num_agents
    rows = []
    for ep in range(a.episodes):
        if ep > 0:
            env.resample_maps()
        obs_np, _ = env.reset()
        gt = env.get_ground_truth_trajectories()
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        state = policy.initial_state(B, device)

        xs = np.zeros((T, B)); ys = np.zeros((T, B))
        zs = np.zeros((T, B, role_dim), np.float32)
        for t in range(T):
            ag = env.get_global_agent_state()
            xs[t], ys[t] = ag["x"], ag["y"]
            with torch.no_grad():
                logits, _, state, ri = policy(obs, state)
            zs[t] = ri["role_z"].float().cpu().numpy()
            act = Categorical(logits=logits.float()).sample()
            obs_np, _, _, _, _ = env.step(act.cpu().numpy().reshape(B, 1))
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        if ep == 0:
            print("[human] raw GT shapes: " + "  ".join(
                "%s=%s" % (k, np.asarray(gt[k]).shape)
                for k in ("x", "y", "valid", "id", "is_vehicle")
                if k in gt))
        gx = as_agent_time(gt["x"], B)
        gy = as_agent_time(gt["y"], B)
        gv = as_agent_time(gt["valid"], B).astype(bool)
        assert gx.shape == gy.shape == gv.shape and gx.shape[0] == B, (
            "GT arrays did not normalise to (B, T): %s -- send this line and "
            "the raw shapes above" % (gx.shape,))
        sids = _sq(np.asarray(gt["scenario_id"]).astype(str))
        vids = _sq(np.asarray(gt["id"])).reshape(-1)
        isv = _sq(np.asarray(gt["is_vehicle"])).astype(bool)

        n = min(T, gx.shape[1])
        # ---- the verification: did the env replay the logs? --------------
        dev = np.hypot(xs[:n].T - gx[:, :n], ys[:n].T - gy[:, :n])
        m = gv[:, :n]
        cnt = m.sum(1)
        ade = np.where(cnt > 0, (dev * m).sum(1) / np.maximum(cnt, 1), np.inf)

        keep = isv & (cnt >= 15) & (ade < a.ade_tol)
        print("[human] ep %d: %d/%d vehicles replayed (ADE < %.1f m), "
              "median ADE of the rest = %.1f m"
              % (ep + 1, int(keep.sum()), int(isv.sum()), a.ade_tol,
                 float(np.median(ade[isv & ~keep])) if (isv & ~keep).any()
                 else 0.0))

        for i in np.flatnonzero(keep):
            zi = zs[:n, i][m[i]]
            if not len(zi):
                continue
            r = dict(scenario_id=str(sids[i]), vehicle_id=int(vids[i]),
                     episode=ep, n_steps=int(len(zi)), ade=round(float(ade[i]), 3))
            for d in range(role_dim):
                r["role_%d" % d] = round(float(zi[:, d].mean()), 5)
            rows.append(r)

    if not rows:
        raise SystemExit(
            "NOTHING WAS REPLAYED. control_mode=%s did not step agents from "
            "logs, so no role here would be a human's. Try --control_mode "
            "control_wosac or control_mixed_play." % a.control_mode)

    df = pd.DataFrame(rows).drop_duplicates(["scenario_id", "vehicle_id"])
    df.to_csv(out / "human_roles.csv", index=False)
    print("\n[human] %d human drivers with a role -> %s/human_roles.csv"
          % (len(df), out))
    if len(df) < 200:
        print("  WARNING: fewer than 200 drivers. The spread below is thin.")

    # ---- does the role vary between humans, and within a scene? ----------
    print("\n" + "=" * 70)
    print("  ROLE SPREAD ACROSS HUMAN DRIVERS")
    print("=" * 70)
    for d in range(role_dim):
        c = "role_%d" % d
        v = df[c].to_numpy(float)
        g = df.groupby("scenario_id")[c]
        sz = g.size()
        wi = g.std()[sz[sz >= 2].index]
        print("  dim %d: p10=%+.3f p50=%+.3f p90=%+.3f  sd=%.3f"
              % (d, np.percentile(v, 10), np.median(v),
                 np.percentile(v, 90), v.std()))
        print("         within-scene sd = %.3f over %d scenes"
              % (float(wi.median()) if len(wi) else np.nan, len(wi)))
    print("\n  A within-scene sd near zero means every human in a scene gets")
    print("  the same role: the conditioning would be vacuous and the roles")
    print("  would have to be fitted to human style directly instead.")

    # ---- does the role track the human's measured style? -----------------
    if a.gt_regimes and Path(a.gt_regimes).exists():
        g = pd.read_csv(a.gt_regimes)
        g["scenario_id"] = g["scenario_id"].astype(str)
        df["scenario_id"] = df["scenario_id"].astype(str)
        j = df.merge(g, on=["scenario_id", "vehicle_id"], how="inner")
        if len(j) < 50:
            j = df.assign(scenario_id=df.scenario_id.str[:16]).merge(
                g.assign(scenario_id=g.scenario_id.str[:16]),
                on=["scenario_id", "vehicle_id"], how="inner")
        print("\n" + "=" * 70)
        print("  ROLE vs THE HUMAN'S OWN STYLE  (%d joined)" % len(j))
        print("=" * 70)
        print("  %-16s %10s %14s" % ("style parameter", "r", "r within-scene"))
        for c in ("headway_T", "v_freeflow_rel", "speed_ff", "speed_fol",
                  "jam_s0"):
            if c not in j.columns:
                continue
            y = pd.to_numeric(j[c], errors="coerce")
            x = j["role_0"]
            k = np.isfinite(x) & np.isfinite(y)
            if k.sum() < 30:
                continue
            r = float(np.corrcoef(x[k], y[k])[0, 1])
            jj = j[k].copy()
            jj["_x"] = jj["role_0"] - jj.groupby("scenario_id")["role_0"].transform("mean")
            jj["_y"] = y[k] - y[k].groupby(jj["scenario_id"]).transform("mean")
            rw = (float(np.corrcoef(jj["_x"], jj["_y"])[0, 1])
                  if jj["_x"].std() > 1e-9 and jj["_y"].std() > 1e-9 else np.nan)
            print("  %-16s %+10.3f %+14s"
                  % (c, r, "%.3f" % rw if np.isfinite(rw) else "--"))
        print("\n  The within-scene column is the one that matters: a raw")
        print("  correlation can come entirely from both quantities tracking")
        print("  the map. If it survives scene-centring, the encoder is")
        print("  reading the DRIVER.")


if __name__ == "__main__":
    main()
