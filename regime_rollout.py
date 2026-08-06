"""
regime_rollout.py -- the POLICY side of the regime analysis: roll the agent out,
measure the same three style parameters, and keep the role PER STEP.

This is the piece that turns the GT reference distributions into a measured role
effect. Everything here calls the same conflict_metrics functions the GT pass
used, so the human and the policy are never measured by two implementations.

WHAT IT ADDS OVER THE GT PASS
The role encoder emits a z at EVERY step -- it is observation-conditioned -- but
every analysis so far collapsed it to one mean per agent. Keeping it per step,
alongside per-step regime membership, makes four things measurable:

  1. Is the role a DISPOSITION or a REACTION? Split var(z) three ways: between
     agents, within-agent-between-regime, within-agent-within-regime. A driving
     style should be near-constant for one driver across situations. If z swings
     as the agent enters a junction, it is encoding the situation. This is the
     scene-ICC idea moved down a level, and it is printed as a summary here.
  2. The role AT THE DECISION MOMENT, rather than an episode mean, as the
     covariate for the gap-acceptance fit.
  3. Regime-conditional dose-response: sweep the role and read each style
     parameter only in the regime that identifies it.
  4. Lead vs lag: the MI loss targets FUTURE behaviour, so z should change
     BEFORE behaviour does. --dump_steps writes the per-step arrays for that.

FORCING CONVENTION
`--alphas` shifts each agent's OWN natural role along a PC axis:
    forced = z_natural + alpha * sigma * u
matching role_paired_sweep.py exactly (a shift, not an absolute point), so these
numbers stay comparable with role_paired_tests.csv and every existing figure.

ZONES COME FROM THE GT AUDIT
Junction zones are read from junction_control_audit.csv rather than re-derived,
so the policy is scored against the SAME zones, radii and right-of-way labels as
the human reference. Re-deriving would risk a silent drift between the two sides.

Usage (GPU if available, falls back to CPU):
    python regime_rollout.py \
        --checkpoint /scratch/$USER/checkpoints/<run>/roma_dim1_final.pt \
        --zones_csv  /scratch/$USER/regimes/junction_control_audit.csv \
        --data_dir   pufferlib/resources/drive/binaries/training \
        --out_dir    /scratch/$USER/regimes_rollout/<tag> \
        --episodes 4 --alphas 0
    # add --axes_csv .../role_paired_axes.csv --alphas -2,-1,0,1,2 to sweep
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conflict_metrics as CM
from render_topdown import load_policy

T = 91
TELEPORT_M = 4.0
ROLLOUT_SEED = 1234


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--zones_csv", required=True,
                   help="junction_control_audit.csv from the GT pass")
    p.add_argument("--data_dir", default="pufferlib/resources/drive/binaries/training")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--episodes", type=int, default=4,
                   help="map-pool resamples; more = more scenes covered")
    p.add_argument("--map_pool", type=int, default=1000)
    p.add_argument("--total_agents", type=int, default=3072)
    p.add_argument("--alphas", default="0",
                   help="comma list in sigma units, e.g. -2,-1,0,1,2. 0 = the "
                        "natural role (no shift), which is the reference arm.")
    p.add_argument("--axis", default="PC1")
    p.add_argument("--axes_csv", default="",
                   help="role_paired_axes.csv. OPTIONAL: without it the axis is "
                        "derived from this checkpoint's own natural rollouts, "
                        "which is safer -- role scales do NOT transfer between "
                        "runs.")
    p.add_argument("--axis_tol", type=float, default=10.0,
                   help="abort if --axes_csv sigma differs from this "
                        "checkpoint's measured natural sigma by more than this "
                        "factor. A stale axes file gave sigma=729 against a "
                        "measured 1.25 (580x), so alpha=-2 set the role to "
                        "about -1458 and the whole sweep was off-manifold.")
    p.add_argument("--skip_axis_check", action="store_true")
    p.add_argument("--causal_csv", default="",
                   help="role_causal_axis.csv. Sweep the MEASURED CAUSAL axis "
                        "instead of a PC. role_causal_axis showed PC1 has "
                        "NEGATIVE cos with the causal direction for speed in "
                        "both dim-4 arms -- sweeping it moves behaviour the "
                        "wrong way -- while the true axis carries 2.0-5.7x more "
                        "effect. At role_dim=1 this degrades to the same single "
                        "axis, harmlessly.")
    p.add_argument("--causal_metric", default="speed_mean",
                   help="which metric's causal axis to sweep from --causal_csv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dump_steps", action="store_true",
                   help="also write per-step z and regime arrays (npz) for the "
                        "lead/lag analysis; large")
    # regime parameters -- keep identical to the GT pass
    p.add_argument("--buffer", type=float, default=2.0)
    p.add_argument("--tau_dec", type=float, default=3.0)
    p.add_argument("--min_approach_offset", type=float, default=5.0)
    p.add_argument("--zone_radius", type=float, default=25.0)
    p.add_argument("--max_pairs", type=int, default=400)
    return p.parse_args()


def load_zones(path):
    """sid -> [dict(centre, R, control)] from the GT audit."""
    Z = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["control"] == "not_a_junction":
                continue
            try:
                Z[str(r["scenario_id"])].append(dict(
                    centre=np.array([float(r["cx"]), float(r["cy"])]),
                    R=float(r.get("zone_r") or 25.0),
                    control=r["control"]))
            except (ValueError, KeyError):
                continue
    return Z


def load_axis(path, name, role_dim):
    import pandas as pd
    df = pd.read_csv(path)
    cols = [f"c{i}" for i in range(role_dim)]
    mu = df[df["name"] == "mu"][cols].values[0].astype(np.float64)
    row = df[df["name"] == name]
    if not len(row):
        raise SystemExit(f"axis {name} not in {path}; "
                         f"have {sorted(df['name'].unique())}")
    return mu, row[cols].values[0].astype(np.float64), float(row["sigma"].iloc[0])


def rollout(env, policy, device, shift_vec=None):
    """One episode. Returns xs, ys, hs (T,B), z (T,B,D), and the gt dict.

    Two forward passes when forcing, exactly as role_paired_sweep does: the
    first recovers each agent's natural role so the shift is applied to its OWN
    role rather than replacing it with a global constant.
    """
    B = env.num_agents
    obs_np, _ = env.reset()
    gt = env.get_ground_truth_trajectories()
    torch.manual_seed(ROLLOUT_SEED)
    xs = np.zeros((T, B), np.float32)
    ys = np.zeros((T, B), np.float32)
    hs = np.zeros((T, B), np.float32)
    zs = None
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    state = policy.initial_state(B, device)
    fv = (None if shift_vec is None else
          torch.as_tensor(shift_vec, dtype=torch.float32, device=device))
    for t in range(T):
        ag = env.get_global_agent_state()
        xs[t], ys[t], hs[t] = ag["x"], ag["y"], ag["heading"]
        with torch.no_grad():
            if fv is None:
                logits, _, state, ri = policy(obs, state)
            else:
                _, _, _, ri0 = policy(obs, state)
                forced = ri0["role_z"].clone() + fv
                logits, _, state, ri = policy(obs, state, forced_role=forced)
        z = ri["role_z"].float().cpu().numpy()
        if zs is None:
            zs = np.zeros((T, B, z.shape[-1]), np.float32)
        zs[t] = z
        act = Categorical(logits=logits.float()).sample()
        obs_np, _, _, _, _ = env.step(act.cpu().numpy().reshape(B, 1))
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    return xs, ys, hs, zs, gt


def scene_rows(sid, tracks, zs, zones, args, episode=0):
    """Conflicts + per-trajectory regimes for ONE scene of a rollout.

    Mirrors regime_extract.scene(); the difference is that zones are given
    rather than derived, and every row carries role information.
    """
    if not tracks:
        return [], []
    D = CM.dense_scene(tracks, T)
    lead = CM.leaders_dense(D)
    centres = (np.array([z["centre"] for z in zones], float).reshape(-1, 2)
               if zones else np.empty((0, 2)))

    # per-step role for each track, on its own truncated window
    zt = {k: zs[:tr["n"], tr["slot"]] for k, tr in enumerate(tracks)}
    ndim = zs.shape[-1]

    conflicts = []
    if zones:
        # `seen` is per SCENE, not per zone. Resetting it inside the zone loop
        # emitted a pair once for every zone it appeared in -- the rollout came
        # out at 2766 rows over 763 conflicts, 3.6 per conflict instead of 2,
        # which both double-counts in any fit and breaks the conflict-level
        # bootstrap. regime_extract already scopes it correctly.
        seen, tested = set(), 0
        for z in zones:
            here = []
            for e, tr in enumerate(tracks):
                d = np.hypot(tr["x"] - z["centre"][0], tr["y"] - z["centre"][1])
                k = np.flatnonzero(d <= z["R"])
                if len(k):
                    here.append((e, int(k[0]), int(k[-1])))
            for i in range(len(here)):
                for j in range(i + 1, len(here)):
                    if tested >= args.max_pairs:
                        break
                    e, a0, a1 = here[i]
                    f, b0, b1 = here[j]
                    if min(a1, b1) < max(a0, b0):
                        continue
                    key = (min(e, f), max(e, f))
                    if key in seen:
                        continue
                    seen.add(key)
                    tested += 1
                    A, B_ = tracks[e], tracks[f]
                    cp = CM.conflict_point(A, B_, args.buffer)
                    if cp is None:
                        continue
                    kind, dh = CM.classify_conflict(
                        A, B_, cp, min_approach_offset=args.min_approach_offset)
                    if kind is None:
                        continue
                    mt = CM.conflict_metrics(A, B_, cp, kind,
                                             tau_dec=args.tau_dec)
                    if mt is None:
                        continue
                    for (ego, oth, ei) in ((A, B_, e), (B_, A, f)):
                        went = int(mt["leader_id"] == ego["id"])
                        ta = mt["ta_at_decision"]
                        if ta != "" and went:
                            ta = round(-float(ta), 3)
                        # the role the policy was ACTING ON at the decision
                        # step, not an episode average
                        idec = mt["t_decision"]
                        zrow = {}
                        if idec != "":
                            k = int(round(float(idec) * CM.HZ))
                            k = min(max(k, 0), ego["n"] - 1)
                            zrow = {f"role_dec_{d}": round(float(zt[ei][k, d]), 4)
                                    for d in range(ndim)}
                        conflicts.append(dict(
                            scenario_id=sid, episode=episode, vehicle_id=ego["id"],
                            other_id=oth["id"], ego_went_first=went,
                            kind=kind, control=z["control"],
                            heading_diff_deg=round(float(dh), 1),
                            ta_at_decision=ta, pet=mt["pet"],
                            min_ttc=mt["min_ttc"], mrd=mt["mrd"],
                            v_ego_at_point=(mt["v_leader_at_point"] if went
                                            else mt["v_follower_at_point"]),
                            t_decision=idec, both_passed=mt["both_passed"],
                            **zrow))

    # ---- per-trajectory regimes + style parameters + per-regime role -----
    ff = [CM.freeflow_mask(D, e, lead[e], centres, args.zone_radius)
          for e in range(len(tracks))]
    v_ref = CM.scene_reference_speed(D, ff)
    in_conf = {c["vehicle_id"] for c in conflicts}

    rows = []
    for e, tr in enumerate(tracks):
        segs = CM.following_segments(lead[e])
        Th, s0, n_st, n_jam = CM.headway_params(D, e, segs)
        n_free = int(ff[e][:tr["n"]].sum())
        fol = np.zeros(tr["n"], bool)
        for t0, t1, _ in segs:
            fol[max(t0, 0):min(t1, tr["n"])] = True
        v_free = float(np.nanmean(D["V"][e][ff[e]])) if n_free else np.nan

        fol_T = np.zeros(T, bool)
        for t0, t1, _ in segs:
            fol_T[max(t0, 0):min(t1, T)] = True
        zone_T = np.zeros(T, bool)
        if len(centres):
            dd = np.hypot(D["X"][e][:, None] - centres[None, :, 0],
                          D["Y"][e][:, None] - centres[None, :, 1])
            zone_T = np.nan_to_num(dd.min(1), nan=np.inf) <= args.zone_radius
        kin = CM.kinematics_by_regime(D, e, {"ff": ff[e], "fol": fol_T,
                                             "zone": zone_T})

        # which KIND of junction this vehicle was nearest to -- the column the
        # human side already carries, needed for the stop-compliance comparison
        z_ctrl = ""
        if len(centres):
            dd = np.hypot(D["X"][e][:, None] - centres[None, :, 0],
                          D["Y"][e][:, None] - centres[None, :, 1])
            if np.isfinite(dd).any():
                z_ctrl = zones[int(np.nanargmin(np.nanmin(dd, axis=0)))]["control"]

        zz = zt[e]
        def zmean(mask):
            return (zz[mask].mean(0) if mask.any()
                    else np.full(ndim, np.nan))
        z_all, z_ff, z_fo = zz.mean(0), zmean(ff[e][:tr["n"]]), zmean(fol)

        row = dict(
            scenario_id=sid, vehicle_id=tr["id"], n_steps=tr["n"],
            in_conflict=int(tr["id"] in in_conf),
            frac_following=round(float(fol.mean()), 4),
            frac_freeflow=round(float(n_free / max(tr["n"], 1)), 4),
            headway_T=round(Th, 3) if np.isfinite(Th) else "",
            jam_s0=round(s0, 3) if np.isfinite(s0) else "",
            n_steady_steps=n_st,
            v_freeflow=round(v_free, 3) if np.isfinite(v_free) else "",
            v_scene_ref=round(v_ref, 3) if np.isfinite(v_ref) else "",
            v_freeflow_rel=(round(v_free / v_ref, 4)
                            if np.isfinite(v_free) and np.isfinite(v_ref)
                            and v_ref > 0.1 else ""),
            nearest_zone_control=z_ctrl,
            **kin,
        )
        for d in range(ndim):
            row[f"role_{d}"] = round(float(z_all[d]), 4)
            row[f"role_sd_{d}"] = round(float(zz[:, d].std()), 4)
            row[f"role_ff_{d}"] = ("" if not np.isfinite(z_ff[d])
                                   else round(float(z_ff[d]), 4))
            row[f"role_fol_{d}"] = ("" if not np.isfinite(z_fo[d])
                                    else round(float(z_fo[d]), 4))
        rows.append(row)
    return conflicts, rows


def variance_split(rows, ndim):
    """between-agent / within-agent-between-regime / within-agent-within-regime.

    A driving STYLE should be near-constant for one agent across situations, so
    the between-agent share is the one that should dominate. A large
    between-regime share means the latent is tracking the situation instead.
    """
    print("\n  ROLE VARIANCE SPLIT (is it a disposition or a reaction?)")
    for d in range(ndim):
        allv, per = [], []
        for r in rows:
            a = r.get(f"role_{d}", "")
            ff = r.get(f"role_ff_{d}", "")
            fo = r.get(f"role_fol_{d}", "")
            if a == "":
                continue
            allv.append(float(a))
            got = [float(v) for v in (ff, fo) if v != ""]
            if len(got) == 2:
                per.append(got)
        if len(allv) < 30:
            print(f"    dim {d}: too few agents")
            continue
        v_between = float(np.var(allv))
        v_regime = (float(np.mean([np.var(g) for g in per])) if per else np.nan)
        v_within = float(np.mean([float(r[f"role_sd_{d}"]) ** 2 for r in rows
                                  if r.get(f"role_sd_{d}", "") != ""]))
        tot = v_between + (0 if not np.isfinite(v_regime) else v_regime)
        share = v_between / tot if tot > 0 else np.nan
        print(f"    dim {d}: between-agent={v_between:.4f}  "
              f"between-regime(within agent)={v_regime:.4f}  "
              f"step-level sd^2={v_within:.4f}")
        print(f"            disposition share = {share:.3f}  "
              f"(-> 1.0 = a stable style; -> 0.5 = as much situational "
              f"swing as driver difference)")


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (args.device != "cuda"
                                          or torch.cuda.is_available()) else "cpu")
    zones_by_sid = load_zones(args.zones_csv)
    print(f"[roll] zones for {len(zones_by_sid)} scenes  device={device}")

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(num_maps=args.map_pool, num_agents=args.total_agents,
                map_dir=args.data_dir, episode_length=T, goal_speed=100)
    obs_probe, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs_probe.shape[-1], device)
    if role_dim == 0:
        raise SystemExit("role_dim=0 checkpoint -- nothing to analyse")
    print(f"[roll] role_dim={role_dim}")

    alphas = [float(a) for a in args.alphas.replace(";", ",").split(",")
              if a.strip()]
    axis_u = axis_sg = None
    axis_name = args.axis
    if any(a != 0 for a in alphas):
        # Measure the axis from THIS checkpoint. Role scales do not transfer
        # between runs, and a stale axes file is silent: it just shifts the
        # role somewhere the policy has never been.
        from render_role_alpha import warmup_axes
        _, ax_w = warmup_axes(env, policy, device, episodes=1,
                              n_axes=max(1, role_dim))
        if args.axis not in ax_w:
            raise SystemExit(f"{args.axis} not derivable at role_dim={role_dim}; "
                             f"have {sorted(ax_w)}")
        u_w, sg_w = ax_w[args.axis]
        print(f"[roll] measured natural sigma({args.axis}) = {sg_w:.4f}")

        if args.causal_csv:
            # The causal axis is per-metric: g0..g{D-1} for the requested row.
            # Scale stays this checkpoint's measured sigma, so alpha keeps
            # meaning "sigmas of role movement" and the numbers stay comparable
            # with the PC sweeps.
            import pandas as pd
            cdf = pd.read_csv(args.causal_csv)
            row = cdf[cdf["metric"] == args.causal_metric]
            if not len(row):
                raise SystemExit(f"metric {args.causal_metric} not in "
                                 f"{args.causal_csv}; have "
                                 f"{sorted(cdf['metric'].unique())[:8]}...")
            g = row[[f"g{i}" for i in range(role_dim)]].values[0].astype(float)
            gn = float(np.linalg.norm(g))
            if gn < 1e-9:
                raise SystemExit("causal axis is all zeros for that metric")
            axis_u, axis_sg = g / gn, sg_w
            axis_name = f"CAUSAL-{args.causal_metric}"
            cos_pc = float(np.dot(axis_u, u_w))
            print(f"[roll] CAUSAL axis for '{args.causal_metric}': "
                  f"{np.round(axis_u, 3).tolist()}")
            print(f"[roll]   cos with {args.axis} = {cos_pc:+.3f}"
                  + ("   <- the PC sweep moves this metric the WRONG WAY"
                     if cos_pc < 0 else ""))
        elif args.axes_csv:
            _, axis_u, axis_sg = load_axis(args.axes_csv, args.axis, role_dim)
            axis_name = args.axis
            ratio = axis_sg / max(sg_w, 1e-12)
            print(f"[roll] axes_csv sigma = {axis_sg:.4f}  (ratio {ratio:.1f}x)")
            if not args.skip_axis_check and (ratio > args.axis_tol
                                             or ratio < 1.0 / args.axis_tol):
                raise SystemExit(
                    f"ABORT: --axes_csv sigma={axis_sg:.4g} but this "
                    f"checkpoint's measured natural sigma={sg_w:.4g} "
                    f"({ratio:.0f}x off).\n"
                    f"       That axes file almost certainly belongs to a "
                    f"DIFFERENT run -- role scales do not transfer.\n"
                    f"       Drop --axes_csv to derive the axis from this "
                    f"checkpoint, or pass --skip_axis_check if you mean it.")
        else:
            axis_u, axis_sg, axis_name = u_w, sg_w, args.axis
            print(f"[roll] axis derived from this checkpoint")
        print(f"[roll] sweeping {axis_name}  sigma={axis_sg:.4f}")

    from role_regime_analysis import _squeeze
    for al in alphas:
        tag = "natural" if al == 0 else f"{axis_name}{al:+g}"
        shift = None if al == 0 else al * axis_sg * axis_u
        C, R = [], []
        for ep in range(args.episodes):
            if ep > 0:
                env.resample_maps()
            xs, ys, hs, zs, gt = rollout(env, policy, device, shift)
            sids = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
            vids = _squeeze(np.asarray(gt["id"])).reshape(-1)
            isv = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
            keep = np.array([bool(isv[a]) and bool(str(sids[a]))
                             and not str(sids[a]).lower().startswith("map")
                             for a in range(len(vids))])
            by_sid = defaultdict(list)
            for a in np.flatnonzero(keep):
                by_sid[str(sids[a])].append(a)
            for sid, slots in by_sid.items():
                m = np.zeros(len(vids), bool)
                m[slots] = True
                tr = CM.tracks_from_arrays(xs, ys, hs, vids, m, TELEPORT_M)
                if not tr:
                    continue
                c, r = scene_rows(sid, tr, zs, zones_by_sid.get(sid, []), args, ep)
                C += c
                R += r
            print(f"[roll] {tag} ep {ep+1}/{args.episodes}: "
                  f"{len(R)} trajectories, {len(C)//2} conflicts", flush=True)
            if args.dump_steps and ep == 0:
                np.savez_compressed(out / f"steps_{tag}.npz", xs=xs, ys=ys,
                                    hs=hs, z=zs, sid=sids, vid=vids, keep=keep)

        for path, rows in ((out / f"conflicts_{tag}.csv", C),
                           (out / f"regimes_{tag}.csv", R)):
            if not rows:
                print(f"  WARNING nothing to write for {path.name}")
                continue
            keys = sorted({k for r in rows for k in r})
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        print(f"\n=== {tag}: {len(R)} trajectories, {len(C)//2} conflicts ===")
        if R:
            variance_split(R, role_dim)
            for key, name in (("v_freeflow_rel", "desired speed (rel)"),
                              ("headway_T", "time headway (s)")):
                v = np.array([float(r[key]) for r in R if r.get(key, "") != ""])
                if len(v):
                    print(f"    {name:<22} n={len(v):<6} med={np.median(v):.3f}")
        print(f"  wrote {out}/conflicts_{tag}.csv + regimes_{tag}.csv")


if __name__ == "__main__":
    main()
