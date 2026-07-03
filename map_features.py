"""
map_features.py -- Phase A of the map-regime analysis.

Extracts one feature vector per scenario from the 10k map pool, using ONLY
data available at env.reset() (no policy, no stepping):

  Human-behavior features (ground-truth trajectories, moving vehicles only):
    n_vehicles        vehicles in the scene
    n_moving          vehicles that actually drive (max speed > 1 m/s)
    speed_p85         85th percentile of speed samples -- what the map ALLOWS
    speed_mean        mean speed of moving vehicles
    speed_std_across  across-vehicle spread of mean speeds (mixed vs uniform flow)
    stop_frac         fraction of samples < 1 m/s (lights / congestion)
    turn_rate         mean |dheading|/s while moving (how much turning happens)

  Geometry features (road-edge polylines):
    road_len          total road-edge length (m)
    curviness         total turning per meter of road edge (rad/m)
    extent            scene bounding-box diagonal (m)
    road_density      road_len / bbox area (grid-like urban vs single corridor)

Scenarios are keyed by Waymo scenario_id; the env is resampled batch by batch
until --target_unique scenarios are covered (or --max_batches reached).
Also caches raw geometry + GT paths for a subset of scenarios so Phase B
(map_atlas.py) can draw example maps per cluster.

Usage (from /scratch/e452103/PufferDrive so the C binding finds drive.ini):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python $HOME/roma_pufferdrive/map_features.py \
        --data_dir pufferlib/resources/drive/binaries/training \
        --out_dir /scratch/e452103/map_atlas
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

MIN_MOVE_SPEED = 1.0   # m/s -- below this a vehicle counts as parked
TURN_MIN_SPEED = 0.5   # m/s -- ignore heading noise while (nearly) stopped

FEATURES = ["n_vehicles", "n_moving", "speed_p85", "speed_mean",
            "speed_std_across", "stop_frac", "turn_rate",
            "road_len", "curviness", "extent", "road_density"]


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
    p.add_argument("--out_dir",       type=str, default="map_atlas")
    p.add_argument("--num_agents",    type=int, default=3072)
    p.add_argument("--num_maps",      type=int, default=10000)
    p.add_argument("--max_batches",   type=int, default=100)
    p.add_argument("--target_unique", type=int, default=9500,
                   help="Stop once this many unique scenarios are featurized")
    p.add_argument("--cache_per_batch", type=int, default=4,
                   help="Scenarios per batch whose raw geometry is cached "
                        "for Phase B example plots")
    return p.parse_args()


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _squeeze(a):
    a = np.asarray(a)
    return a[:, 0] if a.ndim >= 2 and a.shape[1] == 1 else a


# ---------------------------------------------------------------------------
# Per-scenario features
# ---------------------------------------------------------------------------

def gt_features(gx, gy, gh, valid):
    """Behavior features from (V, T) GT arrays of one scenario's vehicles."""
    speeds_all, per_veh_mean, turn_rates = [], [], []
    n_moving = 0

    for v in range(gx.shape[0]):
        m    = valid[v].astype(bool)
        pair = m[:-1] & m[1:]                     # consecutive valid samples
        if pair.sum() < 3:
            continue
        dx  = (gx[v, 1:] - gx[v, :-1])[pair]
        dy  = (gy[v, 1:] - gy[v, :-1])[pair]
        spd = np.hypot(dx, dy) * 10.0             # m/s at 10 Hz
        if spd.max() <= MIN_MOVE_SPEED:
            continue                               # parked
        n_moving += 1
        speeds_all.append(spd)
        per_veh_mean.append(spd.mean())
        dh = np.abs(wrap_angle((gh[v, 1:] - gh[v, :-1])[pair])) * 10.0  # rad/s
        dh = dh[spd > TURN_MIN_SPEED]              # heading noise while stopped
        if len(dh):
            turn_rates.append(dh.mean())

    if n_moving == 0:
        return None
    s = np.concatenate(speeds_all)
    return {
        "n_vehicles":       gx.shape[0],
        "n_moving":         n_moving,
        "speed_p85":        float(np.percentile(s, 85)),
        "speed_mean":       float(np.mean(per_veh_mean)),
        "speed_std_across": float(np.std(per_veh_mean)),
        "stop_frac":        float((s < MIN_MOVE_SPEED).mean()),
        "turn_rate":        float(np.mean(turn_rates)) if turn_rates else 0.0,
    }


def road_features(polys):
    """Geometry features from a scenario's road-edge polylines [(K,2), ...]."""
    if not polys:
        return None
    total_len, total_turn = 0.0, 0.0
    pts_all = []
    for p in polys:
        seg = np.diff(p, axis=0)
        L   = np.hypot(seg[:, 0], seg[:, 1])
        total_len += L.sum()
        if len(p) >= 3:
            h = np.arctan2(seg[:, 1], seg[:, 0])
            total_turn += np.abs(wrap_angle(np.diff(h))).sum()
        pts_all.append(p)
    if total_len < 1.0:
        return None
    pts  = np.concatenate(pts_all, axis=0)
    w    = max(pts[:, 0].max() - pts[:, 0].min(), 1.0)
    h    = max(pts[:, 1].max() - pts[:, 1].min(), 1.0)
    return {
        "road_len":     float(total_len),
        "curviness":    float(total_turn / total_len),
        "extent":       float(np.hypot(w, h)),
        "road_density": float(total_len / (w * h)),
    }


def split_polylines(road):
    """PufferDrive road dict {x, y, lengths, scenario_id} -> {sid: [(K,2)...]}."""
    fx  = np.asarray(road["x"], dtype=np.float64).reshape(-1)
    fy  = np.asarray(road["y"], dtype=np.float64).reshape(-1)
    lens = np.asarray(road["lengths"], dtype=np.int64).reshape(-1)
    sids = np.asarray(road["scenario_id"]).astype(str).reshape(-1)
    out, start = {}, 0
    for L, sid in zip(lens, sids):
        end = start + int(L)
        if L >= 2 and sid:
            pts = np.stack([fx[start:end], fy[start:end]], axis=-1)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) >= 2:
                out.setdefault(sid, []).append(pts)
        start = end
    return out


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path   = out_dir / "map_features.csv"
    cache_path = out_dir / "geometry_cache.pkl"

    from pufferlib.ocean.drive.drive import Drive
    env_cfg = dict(load_drive_config()["env"])
    env_cfg.update({"num_maps": args.num_maps, "num_agents": args.num_agents,
                    "map_dir": args.data_dir})

    t0 = time.time()
    env = Drive(**env_cfg)
    print(f"[phaseA] env created in {time.time()-t0:.0f}s "
          f"({args.num_maps} maps, {args.num_agents} agents)")

    rows      = {}    # sid -> feature dict
    geo_cache = {}    # sid -> {"polys": [...], "gt": (V,T,2) moving paths, "spd": (V,)}
    n_degenerate = 0

    for batch in range(args.max_batches):
        if batch > 0:
            env.resample_maps()
        env.reset()

        gt   = env.get_ground_truth_trajectories()
        road = env.get_road_edge_polylines()

        gx, gy, gh = _squeeze(gt["x"]), _squeeze(gt["y"]), _squeeze(gt["heading"])
        valid      = _squeeze(gt["valid"])
        is_veh     = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
        sid_agent  = _squeeze(np.asarray(gt["scenario_id"]).astype(str))

        polys_by_sid = split_polylines(road)

        new = 0
        for sid in np.unique(sid_agent):
            if not sid or sid in rows:
                continue
            sel = (sid_agent == sid) & is_veh
            if sel.sum() == 0:
                continue
            bf = gt_features(gx[sel], gy[sel], gh[sel], valid[sel])
            rf = road_features(polys_by_sid.get(sid, []))
            if bf is None or rf is None:
                n_degenerate += 1
                continue
            rows[sid] = {**bf, **rf}
            new += 1

            # Cache raw geometry for a few scenarios per batch (Phase B plots)
            if new <= args.cache_per_batch:
                keep = []
                m_any = valid[sel].astype(bool)
                for v in np.where(sel)[0]:
                    mv = valid[v].astype(bool)
                    if mv.sum() < 3:
                        continue
                    px, py = gx[v][mv], gy[v][mv]
                    d = np.hypot(np.diff(px), np.diff(py))
                    if d.sum() > 2.0:                       # moving vehicle
                        keep.append((px.astype(np.float32),
                                     py.astype(np.float32),
                                     float(d.sum() / max(len(d), 1) * 10)))
                geo_cache[sid] = {"polys": polys_by_sid.get(sid, []),
                                  "paths": keep}

        print(f"[phaseA] batch {batch+1:>3}: +{new:>4} new  "
              f"unique={len(rows):>5}  degenerate={n_degenerate}", flush=True)

        if len(rows) >= args.target_unique:
            print(f"[phaseA] target of {args.target_unique} reached")
            break

    env.close()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id"] + FEATURES)
        for sid, feat in rows.items():
            w.writerow([sid] + [feat[k] for k in FEATURES])
    with open(cache_path, "wb") as f:
        pickle.dump(geo_cache, f)

    print(f"\n[phaseA] {len(rows)} scenarios -> {csv_path}")
    print(f"[phaseA] {len(geo_cache)} scenarios' geometry cached -> {cache_path}")
    print(f"[phaseA] degenerate (no moving vehicles / no roads): {n_degenerate}")


if __name__ == "__main__":
    main()
