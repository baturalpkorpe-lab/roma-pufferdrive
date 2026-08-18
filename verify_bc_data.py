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
best matches the distance to the human's next POSITION, then the steer bin whose
resulting course points at that position, and emit the joint index
    a = accel_idx * 13 + steer_idx        (drive.h:1572, 1576, 3124)
Position, not heading, is the target on purpose. Matching heading exactly still
lets position drift, because position is its integral and nothing pulls it back:
per-step residuals of 0.02 deg integrate over 80 steps into metres. Aiming at the
logged point closes the loop on the quantity ADE actually measures. The label is
by construction an action the policy can emit.

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
    p.add_argument("--ade_tol", type=float, default=1.0,
                   help="reporting only -- pairs are filtered by --keep_tol")
    p.add_argument("--keep_tol", type=float, default=0.5,
                   help="metres. A pair is kept while the agent is still this "
                        "close to the logged position. Beyond it the inferred "
                        "action is a correction manoeuvre, not the human's")
    p.add_argument("--lookahead", type=int, default=3,
                   help="steps ahead to aim at. 1 is deadbeat and oscillates "
                        "on a coarse actuator; 3 damps it (pure pursuit)")
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

    keep_obs, keep_act, keep_ade, keep_sid, keep_vid = [], [], [], [], []
    diag = {"res_v": [], "res_h": [], "sat": [], "h_off": [],
            "turn": [], "ade": [], "steer_ol": [], "v_err": [], "model_h": [], "yield": []}
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
        # Identity per agent, carried through to every kept pair. Without it
        # the pairs cannot be joined to regimes_gt.csv, and a role-CONDITIONED
        # tau needs exactly that join to get each human driver's style label.
        sid = _sq(np.asarray(gt["scenario_id"])).reshape(B)
        vid = _sq(np.asarray(gt["id"])).reshape(B)
        T = min(EPISODE_LEN, gx.shape[1])

        # human speed over [t, t+1] -- what the accel bin has to produce
        gspd = np.zeros_like(gx)
        gspd[:, :-1] = np.hypot(np.diff(gx, axis=1), np.diff(gy, axis=1)) / dt
        gspd[:, -1] = gspd[:, -2]

        # ---- open-loop residual: can the grid express the human's step AT
        # ALL, given a perfect starting state? This is computed from GT only,
        # with no env stepping, so it cannot be contaminated by drift. If it
        # is small, TRACK failures are runaway after divergence. If it is
        # large, the grid or the dynamics model cannot represent human
        # driving and no amount of closed-loop correction will help.
        v_in = np.concatenate([gspd[:, :1], gspd[:, :T - 1]], axis=1)   # (B,T)
        ai_o = np.argmin(np.abs(v_in[:, :, None] + ACC[None, None, :] * dt
                                - gspd[:, :T, None]), axis=2)
        vn_o = v_in + ACC[ai_o] * dt
        h_nx = np.concatenate([gh[:, 1:T], gh[:, T - 1:T]], axis=1)
        hp_o = gh[:, :T, None] + (vn_o[:, :, None] * YAWF[None, None, :]
                                  / a.vehicle_length) * dt
        eh_o = np.abs(wrap(hp_o - h_nx[:, :, None]))
        si_o = np.argmin(eh_o, axis=2)
        mv = gv[:, :T]
        diag["res_v"].append(np.abs(vn_o - gspd[:, :T])[mv])
        diag["res_h"].append(np.take_along_axis(eh_o, si_o[:, :, None], 2)[:, :, 0][mv])
        diag["steer_ol"].append(si_o[mv])

        xs = np.zeros((T, B))
        ys = np.zeros((T, B))
        ep_obs = np.zeros((T, B, obs_np.shape[-1]), np.float32)
        ep_act = np.zeros((T, B), np.int64)
        ep_keep = np.zeros((T, B), bool)
        # v_now is INTEGRATED, not estimated from positions. drive.h:1595 does
        # v <- v + a*dt on the instantaneous speed; hypot(dx,dy)/dt is the
        # AVERAGE speed over the interval, and mixing the two puts a half-step
        # bias into every accel choice that then compounds. Integrating the
        # accel we actually applied reproduces the env's own update exactly, so
        # v_now stays equal to the env's internal speed and targeting GT with
        # it is still closed-loop. v_pos below is kept only as a cross-check.
        v_now = gspd[:, 0].copy()
        px = py = hpred_prev = None
        PREF = np.abs(np.arange(N_STEER) - N_STEER // 2)[None, :]

        for t in range(T):
            ag = env.get_global_agent_state()
            x_t = np.asarray(ag["x"], dtype=float)
            y_t = np.asarray(ag["y"], dtype=float)
            h_t = np.asarray(ag["heading"], dtype=float)
            xs[t], ys[t] = x_t, y_t
            if px is not None:                     # cross-checks only
                diag["v_err"].append(
                    np.abs(np.hypot(x_t - px, y_t - py) / dt - v_prev)[gv[:, t]])
                # THE decisive one: last step we predicted this heading from
                # drive.h:1595-1601. Did the env actually do that? Open-loop is
                # exact and the closed loop is exact on a simulator that obeys
                # those lines, so if this is large the env does something else
                # -- steering rate limit, speed clamp, sub-stepping -- and no
                # inverter built on that model can track.
                diag["model_h"].append(np.abs(wrap(h_t - hpred_prev))[gv[:, t]])
            px, py = x_t, y_t

            # Target the human's next POSITION, not their next heading.
            # Heading error is corrected every step, but position is the
            # integral of heading and nothing pulled it back -- dead
            # reckoning. Per-step residuals of 0.02 deg integrate over 80
            # steps into metres of lateral drift, which is what the 2.1 m ADE
            # was. Aiming at the logged point closes the loop on position, so
            # drift cannot accumulate. Note the course is heading + slip: the
            # car travels along h + beta, not h (drive.h:1598).
            nt = min(t + 1, T - 1)
            la = min(t + max(1, a.lookahead), T - 1)
            v_tgt = np.hypot(gx[:, nt] - x_t, gy[:, nt] - y_t) / dt
            theta = np.arctan2(gy[:, la] - y_t, gx[:, la] - x_t)

            # accel bin: whose resulting speed covers that distance
            ai = np.argmin(np.abs(v_now[:, None] + ACC[None, :] * dt
                                  - v_tgt[:, None]), axis=1)
            v_new = v_now + ACC[ai] * dt
            # steer bin: whose resulting course points at the logged point
            course = (h_t[:, None] + (v_new[:, None] * YAWF[None, :]
                                      / a.vehicle_length) * dt
                      + BETA[None, :])
            err = np.abs(wrap(course - theta[:, None]))
            # At low speed yaw = v*YAWF/L collapses: every steer bin gives the
            # same heading and argmin breaks the tie on float noise, landing on
            # an extreme. That is not a steering decision, it is a parked car.
            # Among bins that are within EPS_H of the best, take the one
            # closest to centre. EPS_H sits below the open-loop p90 residual,
            # so it never overrides a genuine choice.
            ok = err <= err.min(axis=1, keepdims=True) + 1e-4
            si = np.where(ok, PREF, N_STEER + 1).argmin(axis=1)

            act = (ai * N_STEER + si).astype(np.int64)
            live = gv[:, t] & gv[:, nt]
            act[~live] = NOOP
            # integrate the accel the env is ACTUALLY about to apply, which is
            # the one in act -- NOOP overrides included
            v_now = v_now + ACC[act // N_STEER] * dt
            v_prev = v_now
            hpred_prev = h_t + (v_now * YAWF[act % N_STEER]
                                / a.vehicle_length) * dt
            if t == 0:
                diag["h_off"].append(wrap(h_t - gh[:, 0])[gv[:, 0]])
            diag["sat"].append(np.isin(si[live], [0, N_STEER - 1]))

            # per-STEP keep mask: this pair is usable while the agent is
            # still on the logged position. Filtering agents threw away every
            # step of an agent that drifted at step 60; filtering steps keeps
            # the first 60.
            ep_keep[t] = (np.hypot(x_t - gx[:, t], y_t - gy[:, t])
                          < a.keep_tol) & gv[:, t]
            ep_obs[t] = obs_np
            ep_act[t] = act
            obs_np, _, _, _, _ = env.step(act.reshape(B, 1))

        dev = np.hypot(xs.T - gx[:, :T], ys.T - gy[:, :T])
        m = gv[:, :T]
        cnt = m.sum(1)
        ade = np.where(cnt > 0, (dev * m).sum(1) / np.maximum(cnt, 1), np.inf)
        pool = isv & (cnt >= 15)
        keep = pool & (ade < a.ade_tol)
        # total |heading change| over the episode -- separates the straight
        # drivers from the turning ones, which is how the selection effect
        # shows itself
        turn = np.abs(wrap(np.diff(gh[:, :T], axis=1)) * m[:, 1:]).sum(1)
        diag["turn"].append(turn[pool])
        diag["ade"].append(ade[pool])
        print("  ep %d: %4d/%4d tracked (ADE < %.1f m)   median ADE  "
              "all=%.2f  tracked=%.2f"
              % (ep + 1, int(keep.sum()), int(pool.sum()), a.ade_tol,
                 float(np.median(ade[pool])) if pool.any() else np.nan,
                 float(np.median(ade[keep])) if keep.any() else np.nan))
        if ep_keep.any():
            keep_obs.append(ep_obs[ep_keep])
            keep_act.append(ep_act[ep_keep])
            keep_ade.append(ade[pool])
            # broadcast the per-agent ids over time, then apply the same mask
            keep_sid.append(np.broadcast_to(sid, (T, B))[ep_keep])
            keep_vid.append(np.broadcast_to(vid, (T, B))[ep_keep])
            diag["yield"].append(ep_keep[:, pool].sum(0))   # steps per agent

    n_agents = int(sum(int((y > 0).sum()) for y in diag["yield"]))
    n_pairs = int(sum(len(x) for x in keep_act))
    have_pairs = n_pairs > 0

    # ---- DIAGNOSE ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("  DIAGNOSE -- why did the untracked agents fail?")
    print("=" * 72)
    res_v = np.concatenate(diag["res_v"])
    res_h = np.concatenate(diag["res_h"])
    sat_cl = np.concatenate(diag["sat"])
    sat_ol = np.concatenate(diag["steer_ol"])
    h_off = np.concatenate(diag["h_off"])
    turn = np.concatenate(diag["turn"])
    ade_all = np.concatenate(diag["ade"])

    print("  OPEN LOOP (from GT states -- no drift possible)")
    print("    speed residual   median %.4f m/s   p90 %.4f"
          % (np.median(res_v), np.percentile(res_v, 90)))
    print("    heading residual median %.5f rad (%.3f deg)   p90 %.5f rad"
          % (np.median(res_h), np.degrees(np.median(res_h)),
             np.percentile(res_h, 90)))
    print("    steer saturated  %.1f%% of steps land on bin 0 or 12"
          % (100 * np.isin(sat_ol, [0, N_STEER - 1]).mean()))
    print("  CLOSED LOOP (what actually drove the env)")
    print("    steer saturated  %.1f%% of steps" % (100 * sat_cl.mean()))
    print("    heading offset at reset  median %+.5f  p10 %+.5f  p90 %+.5f rad"
          % (np.median(h_off), np.percentile(h_off, 10),
             np.percentile(h_off, 90)))
    print("      (a median near 0 with a WIDE spread would mean the agent")
    print("       ordering differs between the state and the GT arrays)")
    v_err = np.concatenate(diag["v_err"]) if diag["v_err"] else np.zeros(1)
    print("    integrated vs position-derived speed  median %.4f m/s  p90 %.4f"
          % (np.median(v_err), np.percentile(v_err, 90)))
    print("      (large = the env clips or otherwise departs from v += a*dt,")
    print("       so integrating its own update is no longer exact)")
    mh = np.concatenate(diag["model_h"]) if diag["model_h"] else np.zeros(1)
    print("    ONE-STEP MODEL residual, predicted vs actual heading")
    print("      median %.5f rad (%.3f deg)   p90 %.5f rad (%.3f deg)"
          % (np.median(mh), np.degrees(np.median(mh)),
             np.percentile(mh, 90), np.degrees(np.percentile(mh, 90))))
    print("      Compare against the OPEN LOOP heading residual above. If open")
    print("      loop is small and this is large, the grid can express human")
    print("      driving but drive.h:1595-1601 is not what the env executes.")
    print("  YIELD -- how many usable steps per agent, and from whom?")
    yld = np.concatenate(diag["yield"]) if diag["yield"] else np.zeros(len(turn))
    n = min(len(yld), len(turn))
    yld, turn2 = yld[:n], turn[:n]
    print("    steps kept per agent  median %d  p25 %d  p75 %d  (of %d)"
          % (np.median(yld), np.percentile(yld, 25),
             np.percentile(yld, 75), EPISODE_LEN))
    print("    agents contributing 0 steps: %.1f%%" % (100 * (yld == 0).mean()))
    q = np.quantile(turn2, [0.25, 0.5, 0.75]) if len(turn2) else [0, 0, 0]
    lab = ["straightest 25%", "q2", "q3", "most turning 25%"]
    b = np.digitize(turn2, q)
    tot = max(yld.sum(), 1)
    turn_share = 0.0
    for i in range(4):
        sel = b == i
        if sel.sum() < 5:
            continue
        share = 100 * yld[sel].sum() / tot
        if i == 3:
            turn_share = share
        print("    %-18s n=%4d  median steps %3d  share of pairs %5.1f%%  "
              "|dheading| %.2f rad"
              % (lab[i], int(sel.sum()), int(np.median(yld[sel])), share,
                 float(np.median(turn2[sel]))))
    print("\n    Share of pairs is the bias metric now, not tracked-fraction.")
    print("    Four balanced quartiles would be 25%% each. If the most-turning")
    print("    quartile is far below that, tau still under-sees turns and the")
    print("    anchor would be weakest exactly at junctions.")
    print("\n  Read it this way. Small open-loop residuals with a low")
    print("  open-loop saturation rate mean the grid CAN express human")
    print("  driving and the closed-loop failures are runaway after an early")
    print("  divergence. Large open-loop residuals mean the dynamics model or")
    print("  the grid cannot represent it and closed-loop correction is")
    print("  hopeless. A nonzero reset heading offset makes every step fight a")
    print("  constant bias, which is what one-sided steer saturation looks")
    print("  like. And if tracking falls off with turning, the kept pairs are")
    print("  a straight-line subset -- the worst possible BC set for junctions.")

    print("\n" + "=" * 72)
    print("  GRID -- is the 91-way action space actually used?")
    print("=" * 72)
    grid_ok = False
    if have_pairs:
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
        print("  skipped -- no pairs survived the keep filter")

    if have_pairs:
        np.savez_compressed(
            out / ("bc_dataset.npz" if (n_pairs >= 18000 and turn_share >= 15)
                   else "bc_dataset_BIASED.npz"),
            obs=np.concatenate(keep_obs).astype(np.float32),
            act=np.concatenate(keep_act).astype(np.int16),
            ade=np.concatenate(keep_ade).astype(np.float32),
            scenario_id=np.concatenate(keep_sid),
            vehicle_id=np.concatenate(keep_vid),
            accel_values=ACC, steer_values=STEER, n_steer=N_STEER)
        print("\n[save] %d pairs from %d agents, %.1f%% from the "
              "most-turning quartile -> %s"
              % (n_pairs, n_agents, turn_share,
                 out / ("bc_dataset.npz" if (n_pairs >= 18000
                                             and turn_share >= 15)
                        else "bc_dataset_BIASED.npz")))

    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    track_ok = n_pairs >= 18000 and turn_share >= 15.0
    print("  LAYOUT  see the block above -- it decides how tau reads obs")
    print("  YIELD   %s -- %d pairs, most-turning quartile contributes %.1f%%"
          % ("PASS" if track_ok else "FAIL", n_pairs, turn_share))
    print("          (bar: 18k pairs, HR-PPO's ~30 min; and >=15%% from the")
    print("           turning quartile against a balanced 25%%)")
    if n_pairs and not track_ok:
        if n_pairs < 18000:
            print("          Too few pairs. Raise --episodes or --map_pool, or")
            print("          loosen --keep_tol (0.5 m is strict).")
        if turn_share < 15.0:
            print("          Turning agents are under-represented. tau would")
            print("          under-see turns and the anchor would be weakest")
            print("          exactly at junctions. Raising --lookahead damps")
            print("          the controller and usually helps turns most.")
    if not n_pairs:
        print("          Nothing reproduced its logged path. In order of")
        print("          likelihood: agent ordering differs between")
        print("          get_global_agent_state() and the GT arrays; the")
        print("          per-agent length differs enough from --vehicle_length")
        print("          to bias steering (try 3.5 and 5.5, see if ADE moves);")
        print("          this build's dynamics_model is not 'classic'.")
    print("  GRID    %s" % ("PASS" if grid_ok else
                            ("FAIL" if have_pairs else "N/A")))
    if track_ok and not grid_ok:
        print("          Too few effective bins. tau would be near-")
        print("          deterministic and the KL anchor would carry no style")
        print("          information. Fix discretisation before training tau.")
    if track_ok and grid_ok:
        print("\n  BC data works. HR-PPO needed ~30 min of driving (~200")
        print("  scenarios x 91 steps ~ 18k pairs). You have %d." % n_pairs)


if __name__ == "__main__":
    main()
