"""verify_bc_data.py -- produce and validate (observation, human action) pairs.

WHY
The BC anchor and the BC init both need the same thing: pairs of (obs_t, a_t)
where obs_t is an observation at a state a human drove through and a_t is what
that human did there. This script builds them on YOUR build and proves they are
real, rather than importing them from a branch whose environment differs.

WHY NOT THE dc/human_ll ROUTE
That branch has the machinery (_save_expert_data + binding.vec_collect_expert_data
+ infer_human_actions) but it forked 92 commits back, with 2889 changed lines in
drive.h and a factored MultiDiscrete([7,13]) action space against this build's
joint MultiDiscrete([7*13]). Its saved observations describe a different space
than the one collstop/paperrew were trained in. Merging it would invalidate
every checkpoint and every number in the results table.

Its infer_human_actions is also wrong FOR THIS BUILD. It inverts
    yaw_rate ~= v * tan(delta) / L        (assumes beta ~ 0, PRE-accel speed)
but drive.h:1595-1601 actually integrates
    v    <- v + a*dt
    beta =  tanh(0.5 * tan(delta))
    yaw  =  v * cos(beta) * tan(delta) / L        (POST-accel speed, beta kept)
cos(beta) falls to 0.795 at full lock, so that is a ~20% steering error exactly
where steering matters most -- in turns, at junctions, which is the problem you
are trying to fix.

WHAT THIS DOES INSTEAD
Grid-constrained closed-loop inversion against the env's own dynamics. At each
step, from the agent's ACTUAL state, pick the accel bin whose resulting speed
best matches the human's next speed, then the steer bin whose resulting heading
best matches the human's next heading, and emit the joint index
    a = accel_idx * 13 + steer_idx        (drive.h:1572, 1576, 3124)
Closed-loop, so inference error is corrected every step instead of accumulating,
and the label is by construction an action the policy can actually emit.

THREE CHECKS, EACH FAILING INDEPENDENTLY
  LAYOUT  the observation split this env emits vs the one policy.py assumes,
          from the binding constants. Decides how tau must read the vector.
  TRACK   step the env with the inferred actions and compare against
          get_ground_truth_trajectories(). End-to-end: it validates the
          inversion, the dynamics, the grid and the agent ordering at once.
          Pairs from a failing TRACK are mislabelled -- do not train on them.
  GRID    are the 91 bins used, or does human driving collapse into a handful?
          TRACK can pass while GRID fails; coarse-but-accurate is a real
          outcome and would make tau degenerate before it is trained.

On success it writes a real BC dataset, not a sample.

    python verify_bc_data.py --out_dir /scratch/$USER/bc_probe
"""

import argparse
import ast
import configparser
import os
import re
from pathlib import Path

import numpy as np

EPISODE_LEN = 91
N_ACCEL, N_STEER = 7, 13
N_ACTIONS = N_ACCEL * N_STEER
NOOP = 3 * N_STEER + 6          # accel 0.0, steer 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_dir", default="")
    p.add_argument("--map_pool", type=int, default=200)
    p.add_argument("--total_agents", type=int, default=512)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--control_mode", default="control_sdc_only")
    p.add_argument("--ade_tol", type=float, default=1.0)
    p.add_argument("--vehicle_length", type=float, default=4.5,
                   help="wheelbase L in the yaw-rate inversion. The env uses "
                        "per-agent length; this is the one approximation left, "
                        "and TRACK is what tells you whether it matters")
    p.add_argument("--dt", type=float, default=0.0, help="0 = read drive.ini")
    return p.parse_args()


def _sq(a):
    a = np.asarray(a)
    return a.reshape(-1) if a.ndim > 1 and 1 in a.shape else a


def as_agent_time(arr, B):
    """GT per-step arrays -> (B, T). Squeeze first, then orient on B."""
    a = np.squeeze(np.asarray(arr, dtype=float))
    if a.ndim == 1:
        a = a.reshape(B, -1)
    elif a.ndim > 2:
        a = a.reshape(B, -1) if a.shape[0] == B else a.reshape(-1, B).T
    return a if a.shape[0] == B else a.T


def wrap(d):
    return np.arctan2(np.sin(d), np.cos(d))


def load_drive_config():
    import pufferlib
    d = os.path.dirname(pufferlib.__file__)
    p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    p.read([os.path.join(d, "config", "default.ini"),
            os.path.join(d, "config", "ocean", "drive.ini")])

    def _parse(v):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return {s: {k: _parse(v) for k, v in p[s].items()} for s in p.sections()}


def load_action_grid():
    """Read the bin values out of drive.h rather than trusting a transcription.

    Falls back to linspace, which is what the header holds today, but reading
    wins if the grid is ever retuned.
    """
    import pufferlib
    h = Path(pufferlib.__file__).resolve().parent / "ocean" / "drive" / "drive.h"
    txt = h.read_text(errors="ignore") if h.exists() else ""
    out, src = {}, "read from drive.h"
    for name, n in (("ACCELERATION_VALUES", N_ACCEL),
                    ("STEERING_VALUES", N_STEER)):
        m = re.search(name + r"\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}", txt, re.S)
        vals = None
        if m:
            found = re.findall(r"-?\d+\.?\d*", m.group(1))
            if len(found) == n:
                vals = [float(x) for x in found]
        if vals is None:
            lo, hi = (-4.0, 4.0) if "ACCEL" in name else (-1.0, 1.0)
            vals, src = list(np.linspace(lo, hi, n)), "linspace fallback"
        out[name] = np.asarray(vals, dtype=float)
    return out["ACCELERATION_VALUES"], out["STEERING_VALUES"], src


def resolve_data_dir(cli):
    if cli:
        return cli
    import pufferlib
    pd = Path(pufferlib.__file__).resolve().parent
    for c in (pd / "resources" / "drive" / "binaries" / "training",
              Path("pufferlib/resources/drive/binaries/training")):
        if (c / "map_000.bin").exists():
            return str(c)
    raise SystemExit("map_000.bin not found; pass --data_dir")


def report_layout(env):
    """The observation split, from the binding constants.

    policy.py documents 1121 as 7 ego + 31x7 partner + 128x7 road + 1 padding.
    drive.py builds num_obs with NO padding term, so if ego_features is 8 the
    real split is 8 / 217 / 896 and policy.py's boundaries are shifted by one.
    That does not stop BC -- tau reads the vector the same way the policy does,
    so the pair stays consistent -- but you want to know which it is.
    """
    from pufferlib.ocean.drive import binding
    ego = getattr(env, "ego_features", None)
    consts = {
        "ego_features": ego,
        "MAX_AGENTS": getattr(binding, "MAX_AGENTS", None),
        "PARTNER_FEATURES": getattr(binding, "PARTNER_FEATURES", None),
        "MAX_ROAD_SEGMENT_OBSERVATIONS":
            getattr(binding, "MAX_ROAD_SEGMENT_OBSERVATIONS", None),
        "ROAD_FEATURES": getattr(binding, "ROAD_FEATURES", None),
    }
    print("\n" + "=" * 72)
    print("  LAYOUT -- what this env actually emits")
    print("=" * 72)
    for k, v in consts.items():
        print("  %-32s %s" % (k, v))
    try:
        mp = consts["MAX_AGENTS"] - 1
        p_end = ego + mp * consts["PARTNER_FEATURES"]
        tot = p_end + (consts["MAX_ROAD_SEGMENT_OBSERVATIONS"]
                       * consts["ROAD_FEATURES"])
        print("  computed num_obs                 %d  (env reports %d)"
              % (tot, env.num_obs))
        print("  real split                       ego[0:%d] partner[%d:%d] road[%d:%d]"
              % (ego, ego, p_end, p_end, tot))
        print("  policy.py assumes                ego[0:7] partner[7:224] "
              "road[224:1120] pad[1120:1121]")
        if ego != 7:
            print("  NOTE: ego_features is %s, not 7, so policy.py's boundaries"
                  % ego)
            print("        are shifted by one. Keep tau on the SAME split as the")
            print("        policy so the anchor stays consistent; realigning is a")
            print("        separate, train-from-scratch question.")
    except Exception as e:
        print("  (could not recompute: %s)" % e)


def main():
    a = parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import pufferlib
    from pufferlib.ocean.drive.drive import Drive

    data_dir = resolve_data_dir(a.data_dir)
    cfg = dict(load_drive_config()["env"])
    dt = a.dt or float(cfg.get("dt", 0.1))
    cfg.update({"num_maps": a.map_pool, "num_agents": a.total_agents,
                "map_dir": data_dir, "control_mode": a.control_mode})
    env = Drive(**cfg)
    obs_np, _ = env.reset()
    B = env.num_agents

    ACC, STEER, grid_src = load_action_grid()
    print("[env] pufferlib   : %s" % Path(pufferlib.__file__).resolve().parent)
    print("[env] maps        : %s" % data_dir)
    print("[env] agents=%d  obs_dim=%d  dt=%.3f  control_mode=%s"
          % (B, obs_np.shape[-1], dt, a.control_mode))
    print("[env] action grid : %s" % grid_src)
    print("       accel  " + " ".join("%.3f" % v for v in ACC))
    print("       steer  " + " ".join("%.3f" % v for v in STEER))
    print("       joint  a = accel_idx * %d + steer_idx" % N_STEER)

    report_layout(env)

    # beta and the yaw factor per steering bin -- exactly drive.h:1598-1601
    BETA = np.tanh(0.5 * np.tan(STEER))
    YAWF = np.cos(BETA) * np.tan(STEER)          # yaw = v_new * YAWF / L

    print("\n" + "=" * 72)
    print("  TRACK -- do the inferred actions reproduce the human path?")
    print("=" * 72)

    keep_obs, keep_act, keep_ade = [], [], []
    for ep in range(a.episodes):
        if ep > 0:
            env.resample_maps()
        obs_np, _ = env.reset()
        gt = env.get_ground_truth_trajectories()
        gx = as_agent_time(gt["x"], B)
        gy = as_agent_time(gt["y"], B)
        gh = as_agent_time(gt["heading"], B)
        gv = as_agent_time(gt["valid"], B).astype(bool)
        isv = _sq(np.asarray(gt.get("is_vehicle", np.ones(B)))).astype(bool)
        T = min(EPISODE_LEN, gx.shape[1])

        # human speed over [t, t+1] -- what the accel bin has to produce
        gspd = np.zeros_like(gx)
        gspd[:, :-1] = np.hypot(np.diff(gx, axis=1), np.diff(gy, axis=1)) / dt
        gspd[:, -1] = gspd[:, -2]

        xs = np.zeros((T, B))
        ys = np.zeros((T, B))
        ep_obs = np.zeros((T, B, obs_np.shape[-1]), np.float32)
        ep_act = np.zeros((T, B), np.int64)
        v_now = gspd[:, 0].copy()
        px = py = None

        for t in range(T):
            ag = env.get_global_agent_state()
            x_t = np.asarray(ag["x"], dtype=float)
            y_t = np.asarray(ag["y"], dtype=float)
            h_t = np.asarray(ag["heading"], dtype=float)
            xs[t], ys[t] = x_t, y_t
            if px is not None:                    # achieved speed: closed loop
                v_now = np.hypot(x_t - px, y_t - py) / dt
            px, py = x_t, y_t

            nt = min(t + 1, T - 1)
            v_tgt, h_tgt = gspd[:, t], gh[:, nt]

            # accel bin: whose resulting speed lands nearest the human's next
            ai = np.argmin(np.abs(v_now[:, None] + ACC[None, :] * dt
                                  - v_tgt[:, None]), axis=1)
            v_new = v_now + ACC[ai] * dt
            # steer bin: whose resulting heading lands nearest the human's next
            h_pred = h_t[:, None] + (v_new[:, None] * YAWF[None, :]
                                     / a.vehicle_length) * dt
            si = np.argmin(np.abs(wrap(h_pred - h_tgt[:, None])), axis=1)

            act = (ai * N_STEER + si).astype(np.int64)
            act[~(gv[:, t] & gv[:, nt])] = NOOP

            ep_obs[t] = obs_np
            ep_act[t] = act
            obs_np, _, _, _, _ = env.step(act.reshape(B, 1))

        dev = np.hypot(xs.T - gx[:, :T], ys.T - gy[:, :T])
        m = gv[:, :T]
        cnt = m.sum(1)
        ade = np.where(cnt > 0, (dev * m).sum(1) / np.maximum(cnt, 1), np.inf)
        pool = isv & (cnt >= 15)
        keep = pool & (ade < a.ade_tol)
        print("  ep %d: %4d/%4d tracked (ADE < %.1f m)   median ADE  "
              "all=%.2f  tracked=%.2f"
              % (ep + 1, int(keep.sum()), int(pool.sum()), a.ade_tol,
                 float(np.median(ade[pool])) if pool.any() else np.nan,
                 float(np.median(ade[keep])) if keep.any() else np.nan))
        if keep.any():
            km = m[keep][:, :T].T                  # (T, n_keep) validity mask
            keep_obs.append(ep_obs[:, keep][km])
            keep_act.append(ep_act[:, keep][km])
            keep_ade.append(ade[keep])

    n_agents = int(sum(len(x) for x in keep_ade))
    n_pairs = int(sum(len(x) for x in keep_act))
    track_ok = n_pairs > 0

    print("\n" + "=" * 72)
    print("  GRID -- is the 91-way action space actually used?")
    print("=" * 72)
    grid_ok = False
    if track_ok:
        acts = np.concatenate(keep_act)
        hist = np.bincount(acts, minlength=N_ACTIONS).astype(float)
        p = hist / hist.sum()
        nz = p[p > 0]
        perp = float(np.exp(-(nz * np.log(nz)).sum()))
        n90 = int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.90) + 1)
        print("  bins used            : %d / %d"
              % (int((hist > 0).sum()), N_ACTIONS))
        print("  effective bins       : %.1f  (exp of entropy)" % perp)
        print("  bins for 90%% of mass : %d" % n90)
        print("  top-5                : " + "  ".join(
            "a%d(ac%+.1f,st%+.2f)=%.1f%%"
            % (b, ACC[b // N_STEER], STEER[b % N_STEER], 100 * p[b])
            for b in np.argsort(p)[::-1][:5]))
        am = np.bincount(acts // N_STEER, minlength=N_ACCEL) / len(acts)
        sm = np.bincount(acts % N_STEER, minlength=N_STEER) / len(acts)
        print("  accel marginal       : " + " ".join("%.2f" % v for v in am))
        print("  steer marginal       : " + " ".join("%.2f" % v for v in sm))
        grid_ok = perp >= 5.0
    else:
        print("  skipped -- nothing tracked")

    if track_ok:
        np.savez_compressed(
            out / "bc_dataset.npz",
            obs=np.concatenate(keep_obs).astype(np.float32),
            act=np.concatenate(keep_act).astype(np.int16),
            ade=np.concatenate(keep_ade).astype(np.float32),
            accel_values=ACC, steer_values=STEER, n_steer=N_STEER)
        print("\n[save] %d pairs from %d agents -> %s/bc_dataset.npz"
              % (n_pairs, n_agents, out))

    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    print("  LAYOUT  see the block above -- it decides how tau reads obs")
    print("  TRACK   %s -- %d agents, %d pairs"
          % ("PASS" if track_ok else "FAIL", n_agents, n_pairs))
    if not track_ok:
        print("          Nothing reproduced its logged path. In order of")
        print("          likelihood: agent ordering differs between")
        print("          get_global_agent_state() and the GT arrays; the")
        print("          per-agent length differs enough from --vehicle_length")
        print("          to bias steering (try 3.5 and 5.5, see if ADE moves);")
        print("          this build's dynamics_model is not 'classic'.")
    print("  GRID    %s" % ("PASS" if grid_ok else
                            ("FAIL" if track_ok else "N/A")))
    if track_ok and not grid_ok:
        print("          Too few effective bins. tau would be near-")
        print("          deterministic and the KL anchor would carry no style")
        print("          information. Fix discretisation before training tau.")
    if track_ok and grid_ok:
        print("\n  BC data works. HR-PPO needed ~30 min of driving (~200")
        print("  scenarios x 91 steps ~ 18k pairs). You have %d." % n_pairs)


if __name__ == "__main__":
    main()
