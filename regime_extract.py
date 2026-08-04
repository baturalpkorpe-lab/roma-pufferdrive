"""
regime_extract.py -- assign every trajectory to behavioural REGIMES and
estimate the driving-style parameter that each regime identifies.

Replaces the stop_frac clustering for role analysis. That clustering is built
from six ego-kinematic features (traj_cluster_view.py ALL_FEATURES: distance,
speed_mean, speed_max, speed_min, net_turn, stop_frac), all of which are
OUTCOMES of the behaviour a role sweep is supposed to move -- so inside a
stratum the response variance is compressed by construction. Everything here is
built from map geometry, other agents, and the t=0 state only.

  A. conflict   junction + a partner whose path crosses or merges with ego's
                -> gap acceptance: the go/yield decision at a measured
                   projected time advantage, plus PET / minTTC / MRD
  B. following  a persistent leader in the ego's body-frame corridor
                -> desired time headway T, jam spacing s0
  C. free-flow  no leader, outside every junction zone
                -> desired speed, as a RATIO to the scene's own 85th
                   percentile free-flow speed

Regimes are PER-STEP, so one trajectory can contribute to several. That is a
feature, not a compromise: a hard per-trajectory cluster throws away the fact
that the same driver is unconstrained for 4 s and then negotiating for 3 s.
Per-trajectory rows carry the fractions plus a `primary` label (conflict wins
if present, else the largest fraction) for anyone who needs a hard label.

Two outputs:
  --out_conflicts   one row per (conflict, participant), keyed
                    (scenario_id, vehicle_id) so it joins straight onto role
                    vectors. `ego_went_first` is the go/yield outcome and
                    `ta_at_decision` is the covariate it should be modelled
                    against.
  --out_traj        one row per (scenario_id, vehicle_id): regime fractions
                    and the style parameters.

Run on GT first -- that is the human reference distribution every rollout
number has to be compared against. The same functions in conflict_metrics.py
take rollout arrays, so the policy side needs no second implementation.

Usage (CPU; login node fine at --limit ~300):
    python regime_extract.py \
        --data_dir /scratch/$USER/PufferDrive/pufferlib/resources/drive/binaries/training \
        --out_conflicts conflicts_gt.csv --out_traj regimes_gt.csv --limit 300
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

import conflict_metrics as CM
from junction_control_audit import labelled_zones
from map_binary import read_map_binary

T_STEPS = 91                     # map_binary.py: array_size is always 91


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_conflicts", default="conflicts_gt.csv")
    p.add_argument("--out_traj", default="regimes_gt.csv")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--zone_radius", type=float, default=25.0,
                   help="radius used for the free-flow exclusion only; the "
                        "conflict zones themselves carry their own adaptive R")
    p.add_argument("--clique_dist", type=float, default=45.0,
                   help="stop signs of ONE junction, mutual-proximity clique")
    p.add_argument("--zone_buffer", type=float, default=4.0)
    p.add_argument("--zone_radius_min", type=float, default=15.0)
    p.add_argument("--min_approach_offset", type=float, default=5.0,
                   help="across-axis separation, ~3 s upstream, required for a "
                        "MERGE rather than a same-lane queue. One lane width is "
                        "~3.2-3.7 m, so 3.5 is the permissive end of the "
                        "sensitivity range and 5.0 the strict end.")
    p.add_argument("--buffer", type=float, default=2.0,
                   help="merging conflict buffer, Rahmani et al. use 2 m total")
    p.add_argument("--tau_dec", type=float, default=3.0,
                   help="the decision moment: seconds of travel from the "
                        "conflict point at which the covariate is read")
    p.add_argument("--max_pairs", type=int, default=400,
                   help="per-scene cap on pairs tested, guards pathological "
                        "scenes")
    p.add_argument("--progress", type=int, default=200)
    return p.parse_args()


def scene(m, args):
    roads, objs = m["roads"], m["objects"]
    sid = m["scenario_id"]

    tracks = CM.vehicle_tracks(objs)
    if len(tracks) < 1:
        return [], [], 0, 0
    D = CM.dense_scene(tracks, T_STEPS)
    lead = CM.leaders_dense(D)

    # Clique-clustered zones carrying a right-of-way label, so every conflict
    # can be attributed to all-way / priority / signalised control. Single-link
    # was merging ~21% of junctions into their neighbours (46839 clique vs
    # 37028 single-link at the same 45 m on the 10k pool).
    Z = [z for z in labelled_zones(roads, args.clique_dist, args.zone_buffer,
                                   args.zone_radius_min)
         if z["control"] != "not_a_junction"]
    # a zone counts only where traffic actually goes through it -- a junction
    # nobody used is not a junction anyone had to deal with
    act = []
    for z in Z:
        d = np.hypot(D["X"] - z["centre"][0], D["Y"] - z["centre"][1])
        if int(np.nansum(np.nanmin(d, axis=1) <= z["R"])) >= 1:
            act.append(z)
    Z = act
    centres = (np.array([z["centre"] for z in Z], float).reshape(-1, 2)
               if Z else np.empty((0, 2)))

    # ---- conflicts: only pairs that share a junction zone ----------------
    # Gating on the zone is what keeps this tractable AND is the definition:
    # Rahmani et al. identify conflicts WITHIN identified intersection areas.
    conflicts, n_reject = [], 0
    if Z:
        inzone = []                       # per zone: track indices, with span
        for z in Z:
            d = np.hypot(D["X"] - z["centre"][0], D["Y"] - z["centre"][1])
            here = []
            for e in range(len(tracks)):
                k = np.flatnonzero(d[e] <= z["R"])
                if len(k):
                    here.append((e, int(k[0]), int(k[-1])))
            inzone.append(here)

        seen, tested = set(), 0
        for zi, here in enumerate(inzone):
            for i in range(len(here)):
                for j in range(i + 1, len(here)):
                    if tested >= args.max_pairs:
                        break
                    e, a0, a1 = here[i]
                    f, b0, b1 = here[j]
                    if min(a1, b1) < max(a0, b0):      # never co-present
                        continue
                    key = (min(e, f), max(e, f))
                    if key in seen:
                        continue
                    seen.add(key)
                    tested += 1
                    A, B = tracks[e], tracks[f]
                    cp = CM.conflict_point(A, B, args.buffer)
                    if cp is None:
                        continue
                    kind, dh = CM.classify_conflict(
                        A, B, cp, min_approach_offset=args.min_approach_offset)
                    if kind is None:
                        # oncoming traffic merely passing, or a same-lane
                        # queue (parallel paths are not a merge)
                        n_reject += 1
                        continue
                    asep = CM.approach_offset(A, B, cp)
                    mt = CM.conflict_metrics(A, B, cp, kind,
                                             tau_dec=args.tau_dec)
                    if mt is None:
                        continue
                    zone = Z[zi]
                    zx, zy = zone["centre"]
                    # two rows, one per participant, so the file joins onto
                    # per-agent role vectors without a pivot
                    for ego, other in ((A, B), (B, A)):
                        went = int(mt["leader_id"] == ego["id"])
                        ta = mt["ta_at_decision"]
                        # ta_at_decision is signed from the FOLLOWER's view;
                        # flip it for the leader so the covariate always means
                        # "projected time by which EGO arrives later"
                        if ta != "" and went:
                            ta = round(-float(ta), 3)
                        conflicts.append(dict(
                            scenario_id=sid, vehicle_id=ego["id"],
                            other_id=other["id"],
                            ego_went_first=went,
                            kind=kind, geometric=mt["geometric"],
                            heading_diff_deg=round(float(dh), 1),
                            approach_offset=round(float(asep), 2),
                            ta_at_decision=ta,
                            pet=mt["pet"], min_ttc=mt["min_ttc"],
                            mrd=mt["mrd"],
                            v_ego_at_point=(mt["v_leader_at_point"] if went
                                            else mt["v_follower_at_point"]),
                            v_other_at_point=(mt["v_follower_at_point"] if went
                                              else mt["v_leader_at_point"]),
                            t_decision=mt["t_decision"],
                            both_passed=mt["both_passed"],
                            zone_x=round(float(zx), 2),
                            zone_y=round(float(zy), 2),
                            control=zone["control"],
                            zone_n_stop=zone["n_stop"],
                        ))

    # ---- per-trajectory regimes + style parameters -----------------------
    ff = [CM.freeflow_mask(D, e, lead[e], centres, args.zone_radius)
          for e in range(len(tracks))]
    v_ref = CM.scene_reference_speed(D, ff)
    in_conf = {c["vehicle_id"] for c in conflicts}

    rows = []
    for e, tr in enumerate(tracks):
        segs = CM.following_segments(lead[e])
        Th, s0, n_st, n_jam = CM.headway_params(D, e, segs)
        n_valid = int(D["M"][e].sum())
        n_follow = int(sum(t1 - t0 for t0, t1, _ in segs))
        n_free = int(ff[e].sum())
        v_free = (float(np.nanmean(D["V"][e][ff[e]])) if n_free else np.nan)

        d_zone, z_ctrl = np.inf, ""
        if len(centres):
            dd = np.hypot(D["X"][e][:, None] - centres[None, :, 0],
                          D["Y"][e][:, None] - centres[None, :, 1])
            if np.isfinite(dd).any():
                d_zone = float(np.nanmin(dd))
                z_ctrl = Z[int(np.nanargmin(np.nanmin(dd, axis=0)))]["control"]

        fr = {"conflict": 1.0 if tr["id"] in in_conf else 0.0,
              "following": n_follow / max(n_valid, 1),
              "freeflow": n_free / max(n_valid, 1)}
        primary = ("conflict" if fr["conflict"] else
                   max(("following", "freeflow"), key=lambda k: fr[k])
                   if max(fr["following"], fr["freeflow"]) > 0.1 else "other")

        rows.append(dict(
            scenario_id=sid, vehicle_id=tr["id"], primary=primary,
            n_valid=n_valid,
            frac_following=round(fr["following"], 4),
            frac_freeflow=round(fr["freeflow"], 4),
            in_conflict=int(bool(fr["conflict"])),
            n_conflicts=sum(1 for c in conflicts if c["vehicle_id"] == tr["id"]),
            # regime B
            headway_T=round(Th, 3) if np.isfinite(Th) else "",
            jam_s0=round(s0, 3) if np.isfinite(s0) else "",
            n_steady_steps=n_st, n_jam_steps=n_jam,
            n_follow_segs=len(segs),
            # regime C
            v_freeflow=round(v_free, 3) if np.isfinite(v_free) else "",
            v_scene_ref=round(v_ref, 3) if np.isfinite(v_ref) else "",
            v_freeflow_rel=(round(v_free / v_ref, 4)
                            if np.isfinite(v_free) and np.isfinite(v_ref)
                            and v_ref > 0.1 else ""),
            # exogenous descriptors, safe to stratify on
            min_dist_to_zone=(round(d_zone, 2) if np.isfinite(d_zone) else ""),
            nearest_zone_control=z_ctrl,
            v_at_t0=round(float(D["V"][e][np.flatnonzero(D["M"][e])[0]]), 3),
            # GT trip geometry. The role encoder sees relative-goal x/y in the
            # ego observation slice (drive.h), and the goal is the GT endpoint,
            # so trip length is directly readable from the observation. If the
            # role tracks these it is encoding the TASK, not a driving style.
            gt_path_len=round(float(tr["s"][-1]), 2),
            gt_goal_dist=round(float(np.hypot(tr["x"][-1] - tr["x"][0],
                                              tr["y"][-1] - tr["y"][0])), 2),
            n_zones_scene=int(len(centres)),
        ))
    return conflicts, rows, len(tracks), n_reject


def main():
    args = parse_args()
    files = sorted(Path(args.data_dir).glob("map_*.bin"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no map_*.bin in {args.data_dir}")
    print(f"[regime] {len(files)} maps  R={args.zone_radius}m "
          f"tau_dec={args.tau_dec}s", flush=True)

    C, R, bad, n_tr, n_rej = [], [], 0, 0, 0
    for i, fp in enumerate(files):
        try:
            c, r, nt, nrj = scene(read_map_binary(fp), args)
        except Exception as e:
            bad += 1
            if bad <= 3:
                print(f"  FAIL {fp.name}: {type(e).__name__}: {e}")
            continue
        C += c
        R += r
        n_tr += nt
        n_rej += nrj
        if args.progress and (i + 1) % args.progress == 0:
            print(f"  {i+1}/{len(files)}  conflicts={len(C)} traj={len(R)}",
                  flush=True)

    if not R:
        raise SystemExit("no trajectories extracted")
    for path, rows in ((args.out_conflicts, C), (args.out_traj, R)):
        if not rows:
            print(f"  WARNING nothing to write for {path}")
            continue
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\n  maps ok={len(files)-bad} failed={bad}  trajectories={n_tr}")
    print(f"\n  CONFLICTS: {len(C)} participant-rows "
          f"({len(C)//2} conflicts)")
    kc = Counter(c["kind"] for c in C)
    for k, n in kc.most_common():
        print(f"    {k:<10} {n//2:>7} conflicts")
    print("\n  by RIGHT-OF-WAY class (the pre-registered stratification):")
    for k, n in Counter(c["control"] for c in C).most_common():
        sub = [c for c in C if c["control"] == k]
        mg = sum(1 for c in sub if c["kind"] == "merging") // 2
        print(f"    {k:<20} {n//2:>7} conflicts  "
              f"(merging {mg}, crossing {n//2 - mg})")
    both = sum(1 for c in C if c["both_passed"]) // 2
    print(f"    both vehicles passed the point inside the 9.1 s window: {both}")
    ta = [float(c["ta_at_decision"]) for c in C if c["ta_at_decision"] != ""]
    if ta:
        ta = np.abs(np.asarray(ta))
        print(f"    |TA| at decision: med={np.median(ta):.2f}s  "
              f"contested (<1s)={100*(ta<1).mean():.1f}%  n={len(ta)}")

    print(f"\n  REGIMES (primary label, {len(R)} trajectories)")
    for k, n in Counter(r["primary"] for r in R).most_common():
        print(f"    {k:<10} {n:>7}  {100*n/len(R):>5.1f}%")

    def stat(key):
        v = np.array([float(r[key]) for r in R if r[key] != ""])
        return (f"n={len(v):<6} med={np.median(v):.3f} "
                f"iqr=[{np.percentile(v,25):.3f},{np.percentile(v,75):.3f}]"
                ) if len(v) else "n=0"

    print(f"\n  STYLE PARAMETERS (GT reference distributions)")
    print(f"    headway_T (s)      {stat('headway_T')}")
    print(f"    jam_s0 (m)         {stat('jam_s0')}")
    print(f"    v_freeflow_rel     {stat('v_freeflow_rel')}")
    print(f"\n  wrote {args.out_conflicts} + {args.out_traj}")


if __name__ == "__main__":
    main()
