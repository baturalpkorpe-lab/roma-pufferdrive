"""
render_role_conditions.py -- Controlled role-condition videos (Phase C follow-up).

For each map regime: pick 2 core maps, pick one FOCAL agent (the one whose
human counterpart drives the most), and roll the SAME scene three times with
the focal agent's role forced to that regime's cluster-mean role vectors
(conformist / middle / runaway, identical k-means as the Phase C spider
charts). All other agents keep their natural encoder roles; torch RNG is
re-seeded identically per condition, so the ONLY difference between the three
rollouts is the focal agent's role vector.

Mechanics: the env loads a POOL of maps (--map_pool) with agent slots spread
across them like in training; target scenes are found inside the live
allocation and rendered by filtering slots/roads/GT via scenario_id.
Two hard-won env facts encoded here:
  1) a num_maps=1 env can never change its map (neither the seed nor
     resample_maps() have any effect on it), and
  2) env.reset() RESHUFFLES the slot<->scenario allocation, so slot indices
     are meaningless across resets -- the focal agent is identified by
     scenario_id + GT vehicle id and re-located after every reset (with
     reset retries until the scene is dealt back in).

Outputs per (regime, map):
  3 videos  r{regime}_map{seed}_{conformist|middle|runaway}.mp4
            focal agent = large colored box + trail, others gray,
            human GT paths dashed
  1 overlay overlay_r{regime}_map{seed}.png
            the focal agent's three trajectories on one image -- path
            divergence at a glance

Default: 4 regimes x 2 maps x 3 conditions = 24 videos + 8 overlays.

Usage (from /scratch/e452103/PufferDrive):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/render_role_conditions.py \
        --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim4/roma_dim4_step3000238080.pt \
        --clusters   /scratch/e452103/map_atlas/map_clusters.csv \
        --agent_data /scratch/e452103/role_regime/dim4/phaseC_agent_data.csv \
        --data_dir   pufferlib/resources/drive/binaries/training \
        --out_dir    /scratch/e452103/renders/role_conditions_dim4
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection, PolyCollection
from torch.distributions import Categorical

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import (vehicle_corners, load_policy, normalize_polylines,
                            BG, GTC)
from role_regime_analysis import _squeeze
from map_features import split_polylines

T = 91
JUMP_THRESH = 8.0          # metres -- respawn teleports between segments
MIN_VIEW_SPAN = 70.0        # metres -- keep agents readable when movement is small
ROAD = "#8a8aa0"            # brighter than render_topdown default for dark BG
ROAD_LW = 1.2
# "natural" = focal keeps its OWN encoder role (unforced, recomputed every
# frame from its observation) -- the honest "what does a trained agent do"
# baseline. The other three force the focal to a cluster-mean role.
FORCED_CONDITIONS = ["conformist", "middle", "runaway"]
CONDITIONS = ["natural"] + FORCED_CONDITIONS
COND_COLORS = {"natural": "#dcdce6", "conformist": "#4477aa",
               "middle": "#ee8833", "runaway": "#33aa44"}
REGIME_NAMES = {0: "quiet_local", 1: "fast_corridors",
                2: "parking_low_speed", 4: "congested_urban"}
ROLLOUT_SEED = 1234   # identical action noise across the 3 conditions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--clusters",   type=str, required=True)
    p.add_argument("--agent_data", type=str, required=True,
                   help="phaseC_agent_data.csv of the SAME checkpoint")
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--out_dir",    type=str, required=True)
    p.add_argument("--maps_per_regime", type=int, default=2)
    p.add_argument("--map_pool",   type=int, default=128,
                   help="Maps loaded into the env; target scenes are found "
                        "inside the live allocation (num_maps=1 cannot "
                        "change its map -- neither seed nor resample)")
    p.add_argument("--total_agents", type=int, default=768,
                   help="Agent slots spread across the map pool "
                        "(~6 per scene, like training)")
    p.add_argument("--min_margin", type=float, default=1.5)
    p.add_argument("--seed_start", type=int, default=100)
    p.add_argument("--max_resamples", type=int, default=10,
                   help="Pool reshuffles if some regime is missing")
    p.add_argument("--max_relocate", type=int, default=80,
                   help="env.reset() retries to re-deal a target scene before "
                        "giving up on it (resets are cheap; higher = fewer "
                        "scenes skipped for bad luck)")
    p.add_argument("--keep_ghosts", action="store_true",
                   help="Keep drawing background agents after their human GT "
                        "ends (default: hide them, so cars don't drive past "
                        "the end of their real trajectory)")
    p.add_argument("--min_focal_drive", type=float, default=5.0,
                   help="Min metres the focal's human must drive for a scene "
                        "to qualify. 5 m so slow regimes (parking, regime 2) "
                        "are not filtered out (10 m rejected almost all).")
    p.add_argument("--policy_background", action="store_true",
                   help="Drive background agents with the policy (old default) "
                        "instead of replaying their logged GT. GT replay "
                        "(default) removes respawn-splitting and policy "
                        "misbehaviour of context cars.")
    p.add_argument("--goal_radius", type=float, default=2.0,
                   help="Metres for the on-screen 'REACHED' flag (matches ini)")
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--fps",        type=int, default=10)
    p.add_argument("--dpi",        type=int, default=110)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Role conditions: identical k-means to the Phase C spider charts
# ---------------------------------------------------------------------------

def cluster_role_conditions(agent_csv):
    import pandas as pd
    from sklearn.cluster import KMeans
    df = pd.read_csv(agent_csv)
    role_cols = [c for c in df.columns if c.startswith("role_")]
    out = {}
    for rg, sub in df.groupby("regime"):
        sub = sub.dropna(subset=["d_speed"])
        km  = KMeans(n_clusters=3, n_init=10, random_state=42)
        lab = km.fit_predict(sub[role_cols].values)
        d_speed = np.array([sub["d_speed"][lab == c].mean() for c in range(3)])
        order   = np.argsort(d_speed)             # conformist -> runaway
        out[int(rg)] = {FORCED_CONDITIONS[rank]:
                        (km.cluster_centers_[c], float(d_speed[c]))
                        for rank, c in enumerate(order)}
    return out, len(role_cols)


# ---------------------------------------------------------------------------
# Scene selection inside the live allocation.
# The env loads a POOL of maps (num_maps=map_pool) and spreads the agent
# slots across them, exactly like training. A num_maps=1 env can NEVER
# change its map (neither the seed nor resample_maps() do anything), so we
# find our target scenes among the pool and render them by filtering the
# scene's agent slots / road polylines / GT rows via scenario_id.
# ---------------------------------------------------------------------------

def scan_allocation(env, regime_of, need, found, used_sids, min_drive=5.0):
    """Group the live allocation's agent slots by scenario; return all new
    usable target scenes as (sid, regime, focal_vehicle_id, human_m).

    IMPORTANT: env.reset() reshuffles the slot<->scenario allocation, so
    slot indices are worthless across resets. The focal agent is identified
    by scenario_id + GT vehicle id and re-located after every reset.
    Reserves accepted scenes in `found`/`used_sids`."""
    env.reset()
    gt     = env.get_ground_truth_trajectories()
    sids   = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
    ids    = _squeeze(np.asarray(gt["id"])).reshape(-1)
    gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
    valid  = _squeeze(gt["valid"]).astype(bool)
    is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)

    accepted = []
    for sid in np.unique(sids):
        if not sid or sid in used_sids:
            continue
        if str(sid).lower().startswith("map"):
            continue                        # placeholder ids, not real scenarios
        rg = regime_of.get(sid)
        if rg is None or rg not in need or len(found[rg]) >= need[rg]:
            continue
        slots = np.where(sids == sid)[0]
        lens  = np.zeros(len(slots))
        for i, a in enumerate(slots):
            if not is_veh[a] or valid[a].sum() < 10:
                continue
            px, py = gx[a][valid[a]], gy[a][valid[a]]
            lens[i] = np.hypot(np.diff(px), np.diff(py)).sum()
        # min_drive is low (default 5 m) so slow regimes (parking / low-speed
        # = regime 2) are not filtered out for barely moving -- a 10 m bar
        # rejected almost every parking scene, so regime 2 never rendered.
        if lens.max() < min_drive:
            continue
        focal_vid = int(ids[slots[int(np.argmax(lens))]])
        accepted.append((sid, int(rg), focal_vid, float(lens.max())))
        found[rg].append(sid)
        used_sids.add(sid)
    return accepted


def longest_segment(x, y, jump=JUMP_THRESH):
    """Longest contiguous run with no >jump m single-step teleports."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) <= 1:
        return x, y
    cuts = np.concatenate([[0], np.where(np.hypot(np.diff(x), np.diff(y)) > jump)[0] + 1,
                           [len(x)]])
    best = slice(0, len(x))
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a > best.stop - best.start:
            best = slice(a, b)
    return x[best], y[best]


def agent_extent(xs, ys, jump=JUMP_THRESH, min_span=MIN_VIEW_SPAN):
    """Axis-aligned bounds from agent motion; teleports cannot blow up the frame."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.ndim == 1:
        xs, ys = xs[:, None], ys[:, None]
    segs = []
    for b in range(xs.shape[1]):
        px, py = longest_segment(xs[:, b], ys[:, b], jump)
        if len(px):
            segs.append(np.stack([px, py], axis=-1))
    if not segs:
        fx = xs[np.isfinite(xs)]
        fy = ys[np.isfinite(ys)]
        if fx.size == 0:
            h = min_span / 2
            return -h, h, -h, h
        x0, x1 = np.percentile(fx, [2, 98])
        y0, y1 = np.percentile(fy, [2, 98])
    else:
        allp = np.concatenate(segs, axis=0)
        x0, x1 = np.percentile(allp[:, 0], [2, 98])
        y0, y1 = np.percentile(allp[:, 1], [2, 98])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, min_span) / 2
    return float(cx - half), float(cx + half), float(cy - half), float(cy + half)


def filter_polys_near(polys, x0, x1, y0, y1, margin=50.0):
    """Keep road segments that overlap the agent bbox (+ margin)."""
    if not polys:
        return []
    mx, my = margin, margin
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    out = []
    for p in np.asarray(polys, dtype=object):
        p = np.asarray(p, dtype=np.float64)
        if p.shape[0] < 2:
            continue
        if (p[:, 0].max() >= x0 and p[:, 0].min() <= x1
                and p[:, 1].max() >= y0 and p[:, 1].min() <= y1):
            out.append(p)
    return out


def roads_for_scene(road_edges, sid, xs, ys):
    """Road polylines for one scenario, clipped to where agents actually drive."""
    x0, x1, y0, y1 = agent_extent(xs, ys)
    polys = []
    if road_edges is None:
        return polys
    try:
        by_sid = split_polylines(road_edges)
        polys  = list(by_sid.get(sid, []))
        if not polys:
            pref = sid[:10]
            for k, v in by_sid.items():
                if (k == sid or k.startswith(pref) or pref.startswith(k[:10])
                        or sid.startswith(k)):
                    polys.extend(v)
        if not polys:
            all_polys = [p for ps in by_sid.values() for p in ps]
            polys = filter_polys_near(all_polys, x0, x1, y0, y1, margin=80.0)
    except Exception as e:
        print(f"  [road filter: {e}]")
        polys = normalize_polylines(road_edges)
    nearby = filter_polys_near(polys, x0, x1, y0, y1, margin=60.0)
    return nearby if nearby else polys


def hide_after_frame(view, keep_ghosts=False):
    """Per-agent last frame to draw. A background agent is hidden once its
    human GT ends (else the policy drives it aimlessly forever); agents with
    no valid GT are hidden entirely. keep_ghosts -> draw everything."""
    n = view["xs"].shape[1]
    gt = view.get("gt")
    if keep_ghosts or gt is None:
        return None
    v   = np.asarray(gt["valid"]).astype(bool)      # (n, T_gt)
    out = np.full(n, T - 1, dtype=int)
    for a in range(n):
        idx = np.where(v[a])[0]
        out[a] = int(idx[-1]) if len(idx) else -1   # -1 = never draw
    return out


def _obb_collision(x1, y1, h1, l1, w1, x2, y2, h2, l2, w2):
    """Oriented-box overlap test for two vehicles. Reimplements the env's own
    check_aabb_collision (drive.h) exactly: 4-axis SAT using each car's own
    length/width directions."""
    c1, s1 = np.cos(h1), np.sin(h1)
    c2, s2 = np.cos(h2), np.sin(h2)
    hl1, hw1 = l1 / 2, w1 / 2
    hl2, hw2 = l2 / 2, w2 / 2
    corners1 = np.array([
        [x1 + hl1 * c1 - hw1 * s1, y1 + hl1 * s1 + hw1 * c1],
        [x1 + hl1 * c1 + hw1 * s1, y1 + hl1 * s1 - hw1 * c1],
        [x1 - hl1 * c1 - hw1 * s1, y1 - hl1 * s1 + hw1 * c1],
        [x1 - hl1 * c1 + hw1 * s1, y1 - hl1 * s1 - hw1 * c1],
    ])
    corners2 = np.array([
        [x2 + hl2 * c2 - hw2 * s2, y2 + hl2 * s2 + hw2 * c2],
        [x2 + hl2 * c2 + hw2 * s2, y2 + hl2 * s2 - hw2 * c2],
        [x2 - hl2 * c2 - hw2 * s2, y2 - hl2 * s2 + hw2 * c2],
        [x2 - hl2 * c2 + hw2 * s2, y2 - hl2 * s2 - hw2 * c2],
    ])
    for ax in ((c1, s1), (-s1, c1), (c2, s2), (-s2, c2)):
        ax = np.asarray(ax)
        p1, p2 = corners1 @ ax, corners2 @ ax
        if p1.max() < p2.min() or p2.max() < p1.min():
            return False
    return True


def focal_collision_frames(data, end):
    """Per-frame vehicle-collision flag for the focal agent over its
    pre-respawn segment [0, end).

    rollout_forced() discards env.step()'s reward/info -- the C reward is an
    additive mix of collision/offroad/jerk/goal terms, so it can't be
    disentangled into "did a collision happen this step" after the fact.
    Instead this reimplements the env's own OBB/SAT test (drive.h
    check_aabb_collision, incl. its 15m dist^2<225 prefilter) directly on the
    x/y/heading/length/width the rollout already recorded for every agent --
    no env access, no C changes.

    Scoped to agents in this scene (data['slots']); does NOT detect offroad
    events (those need road/lane geometry, out of scope here)."""
    slots = data["slots"]
    local_focal = int(np.where(slots == data["focal"])[0][0])
    xs, ys, hs = data["xs"][:, slots], data["ys"][:, slots], data["hs"][:, slots]
    length, width = data["length"][slots], data["width"][slots]
    n = xs.shape[1]
    flags = np.zeros(end, dtype=bool)
    for t in range(end):
        x1, y1, h1 = xs[t, local_focal], ys[t, local_focal], hs[t, local_focal]
        if not (np.isfinite(x1) and np.isfinite(y1)):
            continue
        for a in range(n):
            if a == local_focal:
                continue
            x2, y2, h2 = xs[t, a], ys[t, a], hs[t, a]
            if not (np.isfinite(x2) and np.isfinite(y2)):
                continue
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 > 225.0:
                continue
            if _obb_collision(x1, y1, h1, length[local_focal], width[local_focal],
                              x2, y2, h2, length[a], width[a]):
                flags[t] = True
                break
    return flags


def scene_view(data, slots, sid):
    """Subset a full-allocation rollout to one scenario: its agent slots,
    its road polylines, its GT rows."""
    view = {
        "xs": data["xs"][:, slots], "ys": data["ys"][:, slots],
        "hs": data["hs"][:, slots],
        "length": data["length"][slots], "width": data["width"][slots],
        "road_polys": [], "gt": None,
    }
    view["road_polys"] = roads_for_scene(data["road_edges"], sid,
                                         view["xs"], view["ys"])
    if isinstance(data["gt"], dict):
        try:
            g = {"x":     _squeeze(data["gt"]["x"])[slots],
                 "y":     _squeeze(data["gt"]["y"])[slots],
                 "valid": _squeeze(data["gt"]["valid"])[slots]}
            if "heading" in data["gt"]:
                g["heading"] = _squeeze(data["gt"]["heading"])[slots]
            view["gt"] = g
        except Exception:
            pass
    return view


# ---------------------------------------------------------------------------
# One controlled rollout: forced role on the focal agent only
# ---------------------------------------------------------------------------

def rollout_forced(env, policy, sid, focal_vid, cond_vec, device,
                   max_tries=25, shift=False):
    """One rollout with the focal agent's role forced.

    shift=False (default): focal's role REPLACED by cond_vec (constant clamp,
    the original conditions-video behavior). shift=True: cond_vec is ADDED to
    the focal's own live per-step role (natural-offset scheme -- preserves the
    agent's individuality/time-variation; used by render_role_alpha).

    env.reset() reshuffles the slot<->scenario allocation, so the target
    scene is re-located AFTER the reset: its slots by scenario_id, the focal
    agent by its GT vehicle id. Resets are retried until the scene is dealt
    in. Returns None if it never appears within max_tries."""
    B = env.num_agents
    obs_np = gt = road = slots = None
    focal  = -1
    for attempt in range(max_tries):
        if attempt > 0 and attempt % 25 == 0:
            env.resample_maps()   # fresh live subset when the target is elusive
        obs_np, _ = env.reset()
        gt   = env.get_ground_truth_trajectories()
        sids = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
        ids  = _squeeze(np.asarray(gt["id"])).reshape(-1)
        cand = np.where(sids == sid)[0]
        hit  = cand[ids[cand] == focal_vid] if len(cand) else []
        if len(hit):
            focal = int(hit[0])
            # DEDUP: the allocator can instantiate the SAME scenario in TWO
            # env instances (instance count > map pool), so every vehicle --
            # including the focal -- appears in the slot list twice. The
            # duplicate focal is unforced and, drawn from GT, looks like a
            # "clone" driving the focal's intended route. Keep, per vehicle
            # id, the slot nearest the focal's (= its own instance block).
            # Cross-instance vehicles are invisible to the ego's observations,
            # so this is purely a view/rendering fix.
            ids_c = ids[cand]
            if len(np.unique(ids_c)) < len(ids_c):
                keep = [int(cand[ids_c == u][
                            np.argmin(np.abs(cand[ids_c == u] - focal))])
                        for u in np.unique(ids_c)]
                print(f"    (scene dealt into {len(cand)} slots -- duplicate "
                      f"instance detected; keeping the focal's "
                      f"{len(keep)}-slot copy)")
                cand = np.array(sorted(keep), dtype=int)
            slots = cand
            if attempt > 0:
                print(f"    (scene re-dealt after {attempt + 1} resets)")
            break
    else:
        return None
    try:
        road = env.get_road_edge_polylines()
    except Exception as e:
        print(f"  [road data: {e}]")

    torch.manual_seed(ROLLOUT_SEED)              # identical noise per condition
    xs = np.zeros((T, B), dtype=np.float32)
    ys = np.zeros((T, B), dtype=np.float32)
    hs = np.zeros((T, B), dtype=np.float32)
    length = np.full(B, 4.5, dtype=np.float32)
    width  = np.full(B, 2.0, dtype=np.float32)

    # cond_vec is None for the "natural" condition (focal keeps its own role).
    cond = (None if cond_vec is None
            else torch.as_tensor(cond_vec, dtype=torch.float32, device=device))
    obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    state = policy.initial_state(B, device)

    for t in range(T):
        ag = env.get_global_agent_state()
        xs[t], ys[t], hs[t] = ag["x"], ag["y"], ag["heading"]
        if t == 0:
            for k, arr in (("length", length), ("width", width)):
                try:
                    v = np.asarray(ag[k], dtype=np.float32).reshape(-1)
                    if v.shape[0] == B and (v > 0).all():
                        arr[:] = v
                except Exception:
                    pass
        with torch.no_grad():
            if cond is None:
                # natural: nobody forced, focal uses its own live role
                logits, _, state, _ = policy(obs, state)
            else:
                # pass 1: natural roles for everyone (state NOT advanced)
                _, _, _, ri = policy(obs, state)
                forced = ri["role_z"].clone()
                if shift:
                    forced[focal] = forced[focal] + cond   # offset own role
                else:
                    forced[focal] = cond                   # constant clamp
                # pass 2: same input state, focal role replaced
                logits, _, state, _ = policy(obs, state, forced_role=forced)
        action = Categorical(logits=logits.float()).sample()
        obs_np, _, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

    return {"xs": xs, "ys": ys, "hs": hs, "length": length, "width": width,
            "road_edges": road, "gt": gt, "slots": slots, "focal": focal}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def focal_goal(gt, focal):
    """The focal agent's goal = the last valid point of its human GT
    trajectory (that is what 'reaching the goal' means in this env).
    Returns (gx, gy) or None."""
    if not isinstance(gt, dict) or focal is None:
        return None
    try:
        gx = _squeeze(gt["x"])[focal]
        gy = _squeeze(gt["y"])[focal]
        gv = _squeeze(gt["valid"])[focal].astype(bool)
        idx = np.where(gv)[0]
        if len(idx):
            return float(gx[idx[-1]]), float(gy[idx[-1]])
    except Exception:
        pass
    return None


def scene_setup(ax, polys, gt, x0, x1, y0, y1, focal=None,
                focal_color="#ffd24a"):
    ax.set_facecolor(BG)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.tick_params(colors="#888899", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#333344")
    if polys:
        ax.add_collection(LineCollection(polys, colors=ROAD, linewidths=ROAD_LW,
                                         zorder=1))
    if isinstance(gt, dict):
        try:
            gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])
            gv     = _squeeze(gt["valid"]).astype(bool)
            segs = []
            for a in range(gx.shape[0]):
                if a == focal:
                    continue                      # focal route drawn brighter below
                pts = np.stack([gx[a][gv[a]], gy[a][gv[a]]], axis=-1)
                if len(pts) >= 2 and np.hypot(*np.diff(pts, axis=0).T).sum() > 2:
                    segs.append(pts)
            if segs:
                ax.add_collection(LineCollection(
                    segs, colors=GTC, linewidths=0.7,
                    linestyles=(0, (3, 3)), alpha=0.35))
            # focal's intended human route, brighter, + its GOAL as a star
            if focal is not None:
                fpts = np.stack([gx[focal][gv[focal]], gy[focal][gv[focal]]],
                                axis=-1)
                if len(fpts) >= 2:
                    ax.plot(fpts[:, 0], fpts[:, 1], color=focal_color, lw=1.3,
                            ls=(0, (4, 2)), alpha=0.55, zorder=2)
        except Exception:
            pass
    g = focal_goal(gt, focal)
    if g is not None:
        ax.scatter([g[0]], [g[1]], marker="*", s=340, color=focal_color,
                   edgecolors="black", linewidths=0.8, zorder=8)
        ax.annotate("goal", (g[0], g[1]), color=focal_color, fontsize=8,
                    xytext=(4, 4), textcoords="offset points", zorder=8)


def include_point(x0, x1, y0, y1, pt, pad=6.0):
    """Grow a bbox to contain pt (the goal), so the goal star is never
    off-frame even when the agent never reaches it."""
    if pt is None:
        return x0, x1, y0, y1
    gx, gy = pt
    if not (np.isfinite(gx) and np.isfinite(gy)):
        return x0, x1, y0, y1
    return (min(x0, gx - pad), max(x1, gx + pad),
            min(y0, gy - pad), max(y1, gy + pad))


def bbox_of(polys, xs, ys):
    """Frame agents first; add only nearby roads so the view stays tight."""
    x0, x1, y0, y1 = agent_extent(xs, ys)
    if polys:
        nearby = filter_polys_near(polys, x0, x1, y0, y1, margin=40.0)
        if nearby:
            allp = np.concatenate(nearby, axis=0)
            x0 = min(x0, float(allp[:, 0].min()))
            x1 = max(x1, float(allp[:, 0].max()))
            y0 = min(y0, float(allp[:, 1].min()))
            y1 = max(y1, float(allp[:, 1].max()))
    mx = max(0.08 * (x1 - x0 + 1), 8.0)
    my = max(0.08 * (y1 - y0 + 1), 8.0)
    return x0 - mx, x1 + mx, y0 - my, y1 + my


def goal_distance_track(data, focal, goal_radius=2.0):
    """Per-frame distance (m) from the focal agent to its goal, computed only
    over the pre-respawn segment (post-teleport coords are meaningless). Also
    returns the closest approach and whether it entered goal_radius."""
    goal = focal_goal(data["gt"], focal)
    T_n  = data["xs"].shape[0]
    dist = np.full(T_n, np.nan, dtype=np.float64)
    if goal is None:
        return dist, np.nan, -1, False
    fx = data["xs"][:, focal].astype(np.float64)
    fy = data["ys"][:, focal].astype(np.float64)
    # cut at first respawn teleport
    step = np.hypot(np.diff(fx), np.diff(fy))
    tp   = np.where(step > JUMP_THRESH)[0]
    end  = int(tp[0] + 1) if len(tp) else T_n
    d    = np.hypot(fx[:end] - goal[0], fy[:end] - goal[1])
    dist[:end] = d
    dmin = float(np.nanmin(d)) if len(d) else np.nan
    fmin = int(np.nanargmin(d)) if len(d) else -1
    return dist, dmin, fmin, bool(dmin <= goal_radius)


def render_condition_video(data, focal, color, title, out_path, fps, dpi,
                           gt_background=True, goal_radius=2.0):
    """data: a scene_view() dict; focal: index within the view's slots.
    gt_background: draw non-focal agents from their logged GT (clean, no
    policy weirdness / respawn splitting) instead of their policy rollout."""
    xs, ys, hs = data["xs"], data["ys"], data["hs"]
    B = xs.shape[1]
    polys = data["road_polys"]
    x0, x1, y0, y1 = bbox_of(polys, xs, ys)
    x0, x1, y0, y1 = include_point(x0, x1, y0, y1,
                                   focal_goal(data["gt"], focal))

    # Background source: GT (default) or policy. GT gives clean context cars
    # that never respawn/split and never drive aimlessly past their route.
    gt = data.get("gt") if gt_background else None
    if gt is not None and "heading" in gt:
        bxs, bys = np.asarray(gt["x"]), np.asarray(gt["y"])
        bhs      = np.asarray(gt["heading"])
        bvalid   = np.asarray(gt["valid"]).astype(bool)
        T_bg     = bxs.shape[1]
    else:
        gt_background = False
        bxs = bys = bhs = bvalid = None

    dist, dmin, fmin, reached = goal_distance_track(data, focal, goal_radius)

    w, h = x1 - x0, y1 - y0
    fw = 10.0 if w >= h else max(10.0 * w / h, 4)
    fh = 10.0 if h >= w else max(10.0 * h / w, 4)
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(BG)
    scene_setup(ax, polys, data["gt"], x0, x1, y0, y1, focal=focal)
    ax.set_title(title, color="#e8e8f0", fontsize=10)

    others = PolyCollection([], facecolors=(0.72, 0.72, 0.78, 1.0),
                            edgecolors="#1a1a22", linewidths=0.45, zorder=4)
    focal_c = PolyCollection([], facecolors=color, edgecolors="#ffffff",
                             linewidths=1.0, zorder=6)
    trail = LineCollection([], colors=color, linewidths=2.4, alpha=0.9,
                           zorder=5)
    ax.add_collection(others)
    ax.add_collection(focal_c)
    ax.add_collection(trail)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="#e8e8f0",
                  fontsize=10, va="top", family="monospace")

    # Policy-background only: hide a context car once its human GT ends.
    hide_after = None if gt_background else data.get("hide_after")

    def update(t):
        if gt_background:
            tb  = min(t, T_bg - 1)
            idx = np.array([a for a in range(B) if a != focal
                            and bvalid[a, tb]
                            and np.isfinite(bxs[a, tb])], dtype=int)
            verts = (list(vehicle_corners(
                bxs[idx, tb], bys[idx, tb], bhs[idx, tb],
                data["length"][idx], data["width"][idx])) if len(idx) else [])
        else:
            ok  = (np.isfinite(xs[t]) & np.isfinite(ys[t]) & np.isfinite(hs[t]))
            idx = np.array(
                [a for a in range(B) if a != focal and ok[a]
                 and (hide_after is None or t <= hide_after[a])], dtype=int)
            verts = (list(vehicle_corners(
                xs[t, idx], ys[t, idx], hs[t, idx],
                data["length"][idx], data["width"][idx])) if len(idx) else [])
        others.set_verts(verts)
        focal_c.set_verts(list(vehicle_corners(
            xs[t, focal:focal+1], ys[t, focal:focal+1], hs[t, focal:focal+1],
            data["length"][focal:focal+1] * 1.25,
            data["width"][focal:focal+1] * 1.25)))
        s = max(0, t - 40)
        px, py = longest_segment(xs[s:t+1, focal], ys[s:t+1, focal])
        trail.set_segments([np.stack([px, py], axis=-1)] if len(px) >= 2 else [])
        d = dist[t]
        if np.isnan(d):
            gtxt = "goal: (respawned)"
        elif d <= goal_radius:
            gtxt = f"goal: {d:5.1f} m  REACHED"
        else:
            gtxt = f"goal: {d:5.1f} m"
        txt.set_text(f"t = {t:2d}/{T-1}  ({t/10:.1f}s)\n{gtxt}")
        return others, focal_c, trail, txt

    ani = animation.FuncAnimation(fig, update, frames=T, blit=False)
    writer = (animation.FFMpegWriter(fps=fps, bitrate=2500)
              if animation.FFMpegWriter.isAvailable()
              else animation.PillowWriter(fps=fps))
    if not animation.FFMpegWriter.isAvailable():
        out_path = out_path.with_suffix(".gif")
    ani.save(str(out_path), writer=writer, dpi=dpi,
             savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    reach_str = (f"reached goal @ frame {fmin} ({dmin:.1f} m)" if reached
                 else f"closest {dmin:.1f} m @ frame {fmin} (never within "
                      f"{goal_radius:.0f} m)")
    print(f"  saved {out_path}  |  {reach_str}", flush=True)
    return dist, dmin, fmin, reached


def first_segment(x, y, jump=JUMP_THRESH):
    """Path from the start up to the first respawn teleport -- the honest
    comparison object (the same journey, from the same starting state)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) <= 1:
        return x, y
    cuts = np.where(np.hypot(np.diff(x), np.diff(y)) > jump)[0]
    end  = int(cuts[0] + 1) if len(cuts) else len(x)
    return x[:end], y[:end]


def render_overlay(datas, title, out_path, dpi):
    """datas: {condition: scene_view dict carrying its own 'focal' index}.
    One image, three focal paths, each cut at its first respawn teleport."""
    ref = datas[CONDITIONS[0]]

    # Cut BEFORE the bbox -- post-respawn coordinates can be km away.
    cuts = {c: first_segment(d["xs"][:, d["focal"]], d["ys"][:, d["focal"]])
            for c, d in datas.items()}

    polys = ref["road_polys"]
    all_x = np.concatenate([c[0] for c in cuts.values()])
    all_y = np.concatenate([c[1] for c in cuts.values()])
    x0, x1, y0, y1 = bbox_of(polys, all_x[:, None], all_y[:, None])
    x0, x1, y0, y1 = include_point(x0, x1, y0, y1,
                                   focal_goal(ref["gt"], ref["focal"]))

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor(BG)
    scene_setup(ax, polys, ref["gt"], x0, x1, y0, y1, focal=ref["focal"])
    for cond in CONDITIONS:
        px, py = cuts[cond]
        ax.plot(px, py, color=COND_COLORS[cond], lw=2.4,
                alpha=0.95, label=cond, zorder=5)
        if len(px):
            ax.scatter(px[-1], py[-1], color=COND_COLORS[cond], s=90,
                       marker="X", zorder=6, edgecolors="white", linewidths=0.7)
    ax.scatter(ref["xs"][0, ref["focal"]], ref["ys"][0, ref["focal"]],
               color="white", s=70, zorder=6, label="start")
    ax.legend(fontsize=9, loc="best")
    ax.set_title(title, color="#e8e8f0", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    cl = pd.read_csv(args.clusters)
    cl = cl[(cl["margin"] > args.min_margin) & (cl["cluster"].isin(REGIME_NAMES))]
    regime_of = dict(zip(cl["scenario_id"].astype(str), cl["cluster"]))

    conds, role_dim = cluster_role_conditions(args.agent_data)
    for rg, cc in sorted(conds.items()):
        print(f"[conditions] regime {rg} ({REGIME_NAMES.get(rg)}): " +
              "  ".join(f"{n}: dv={dv:+.1f}" for n, (_, dv) in cc.items()))

    # One env holding a POOL of maps; agent slots are spread across the pool
    # like in training. Target scenes are found inside the live allocation
    # and rendered by scenario_id filtering. env.reset() replays the same
    # allocation (WOSAC relies on this), so the 3 conditions per scene are
    # exactly comparable; resample_maps() reshuffles if a regime is missing.
    from pufferlib.ocean.drive.drive import Drive
    # goal_speed=100 disables the respawn speed gate (the env default ~20 m/s
    # blocks fast agents from ever registering goal-reach even at 0 m), so
    # aggressive roles can respawn on reaching their goal in the render.
    env = Drive(num_maps=args.map_pool, num_agents=args.total_agents,
                map_dir=args.data_dir, episode_length=T, goal_speed=100,
                seed=args.seed_start)
    obs_probe, _ = env.reset()
    policy, ckpt_dim = load_policy(args.checkpoint, obs_probe.shape[-1],
                                   device)
    if ckpt_dim != role_dim:
        raise SystemExit(f"checkpoint role_dim={ckpt_dim} but agent_data has "
                         f"{role_dim} role columns -- wrong pairing")

    need      = {rg: args.maps_per_regime for rg in REGIME_NAMES}
    found     = {rg: [] for rg in REGIME_NAMES}
    used_sids = set()
    n_target  = args.maps_per_regime * len(REGIME_NAMES)
    n_done    = 0

    for rnd in range(args.max_resamples + 1):
        if rnd > 0:
            print(f"\n[scan] regimes still missing -- resampling the map "
                  f"pool (round {rnd}) ...")
            env.resample_maps()
        scenes = scan_allocation(env, regime_of, need, found, used_sids,
                                 min_drive=args.min_focal_drive)
        print(f"[scan] round {rnd}: {len(scenes)} new target scenes found "
              f"in the {args.map_pool}-map allocation", flush=True)

        for sid, rg, focal_vid, human_m in scenes:
            print(f"\n[scene {n_done + 1}/{n_target}] regime {rg} "
                  f"({REGIME_NAMES[rg]}) map {sid[:10]}: focal vehicle id "
                  f"{focal_vid} (human drives {human_m:.0f} m)", flush=True)

            # Phase 1 -- run ALL three rollouts first (the expensive part).
            # Nothing is written to disk yet, so a scene that fails to re-deal
            # on any condition leaves NO partial files behind.
            views, ok = {}, True
            for cond in CONDITIONS:
                vec, dv = (None, None) if cond == "natural" else conds[rg][cond]
                dv_str = "own role" if dv is None else f"dv={dv:+.1f} m/s"
                print(f"  rollout {cond:>10} ({dv_str})", flush=True)
                data = rollout_forced(env, policy, sid, focal_vid, vec,
                                      device, max_tries=args.max_relocate)
                if data is None:
                    print(f"  [skip] scene {sid[:10]} not re-dealt within "
                          f"{args.max_relocate} resets on '{cond}' -- "
                          f"discarding this scene entirely (no files written)")
                    found[rg].remove(sid)     # free the regime slot
                    ok = False
                    break
                view = scene_view(data, data["slots"], sid)
                view["focal"] = int(np.where(
                    data["slots"] == data["focal"])[0][0])
                view["hide_after"] = hide_after_frame(view, args.keep_ghosts)
                views[cond] = view
            if not ok:
                continue

            # Phase 2 -- all three succeeded: write videos + overlay. If a
            # render error slips through, delete every file for this scene so
            # a partial triplet can never persist.
            try:
                tracks = {}          # cond -> per-frame distance-to-goal array
                for cond in CONDITIONS:
                    vec, dv = (None, None) if cond == "natural" \
                              else conds[rg][cond]
                    tag   = ("own live role" if dv is None
                             else f"{dv:+.1f} m/s vs human")
                    name  = f"r{rg}_{REGIME_NAMES[rg]}_{sid[:10]}_{cond}.mp4"
                    title = (f"{REGIME_NAMES[rg]} | map {sid[:10]} | focal "
                             f"role = {cond} ({tag})")
                    dist, dmin, fmin, reached = render_condition_video(
                        views[cond], views[cond]["focal"], COND_COLORS[cond],
                        title, out_dir / name, args.fps, args.dpi,
                        gt_background=not args.policy_background,
                        goal_radius=args.goal_radius)
                    tracks[cond] = dist
                render_overlay(views,
                               f"{REGIME_NAMES[rg]} | map {sid[:10]} | same "
                               f"scene, same noise -- only the focal role "
                               f"differs",
                               out_dir / f"overlay_r{rg}_{sid[:10]}.png",
                               args.dpi)
                # frame-by-frame distance-to-goal log for this scene
                dist_csv = out_dir / f"goaldist_r{rg}_{sid[:10]}.csv"
                with open(dist_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["frame"] + CONDITIONS)
                    for t in range(T):
                        w.writerow([t] + [
                            "" if np.isnan(tracks[c][t])
                            else round(float(tracks[c][t]), 2)
                            for c in CONDITIONS])
                print(f"  goal-distance track -> {dist_csv.name}")
                n_done += 1
            except Exception as e:
                print(f"  [render error on {sid[:10]}: {e}] -- removing any "
                      f"partial files for this scene")
                for f in list(out_dir.glob(f"*{sid[:10]}*")):
                    try:
                        f.unlink()
                    except OSError:
                        pass
                found[rg].remove(sid)
                import traceback; traceback.print_exc()

        if n_done >= n_target:
            break

    env.close()
    missing = {REGIME_NAMES[rg]: need[rg] - len(found[rg])
               for rg in need if len(found[rg]) < need[rg]}
    if missing:
        print(f"\n[scan] note: not all regimes filled after "
              f"{args.max_resamples} resamples, missing: {missing}")
    print(f"\n[render] done -> {out_dir}  ({n_done} scenes)")


if __name__ == "__main__":
    main()
