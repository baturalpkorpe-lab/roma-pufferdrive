"""
inspect_map_lanes.py — does the map have lane centerlines through intersections/turns?

Parses PufferDrive map binaries directly (no pufferlib import needed) and reports:
  1. Entity counts by type.
  2. ROAD_LANE curvature  -> how many lanes are curved "turn connectors".
  3. THE key test: for real human vehicle trajectories, how far is each point
     from the nearest lane centerline -- split into STRAIGHT vs TURNING points.
     If human turning points stay < 4 m from a lane, our lane-centering reward
     works through turns. If turning points are routinely > 4 m, turns are
     unmapped (the reward would give 0, no guidance, through junctions).

Binary format (from drive.h load_map_binary), all little-endian:
  int sdc_track_index
  int num_tracks_to_predict ; int[num_tracks_to_predict]
  int num_objects ; int num_roads
  repeat (num_objects + num_roads):
    int scenario_id, int type, int id, int array_size
    float x[size], float y[size], float z[size]
    if type in {1,2,3}:  float vx,vy,vz,heading [size each] ; int valid[size]
    float width,length,height, goalx,goaly,goalz ; int mark_as_expert

Usage (from PufferDrive root on Delft):
    python3 inspect_map_lanes.py --map_dir resources/drive/binaries/training --num_maps 30
"""

import argparse
import glob
import math
import os
import struct
import numpy as np

VEHICLE, PEDESTRIAN, CYCLIST = 1, 2, 3
ROAD_LANE, ROAD_LINE, ROAD_EDGE = 4, 5, 6
OBJECT_TYPES = {VEHICLE, PEDESTRIAN, CYCLIST}
TYPE_NAME = {0: "UNSET", 1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST",
             4: "ROAD_LANE", 5: "ROAD_LINE", 6: "ROAD_EDGE"}

LANE_THRESHOLD = 4.0  # drive.h: agents > 4 m from any lane are "not aligned"


class Reader:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def i32(self):
        v = struct.unpack_from("<i", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def f32(self, n):
        a = np.frombuffer(self.buf, dtype="<f4", count=n, offset=self.pos)
        self.pos += 4 * n
        return a

    def skip(self, nbytes):
        self.pos += nbytes


def parse_map(path):
    """Return (lanes, vehicles): lanes=[(x,y),...], vehicles=[(x,y,valid),...]."""
    with open(path, "rb") as f:
        r = Reader(f.read())
    r.i32()                                   # sdc_track_index
    n_ttp = r.i32()
    r.skip(4 * n_ttp)                         # tracks_to_predict_indices
    num_objects = r.i32()
    num_roads = r.i32()

    lanes, vehicles = [], []
    for _ in range(num_objects + num_roads):
        r.i32()                              # scenario_id
        etype = r.i32()
        r.i32()                              # id
        size = r.i32()
        x = r.f32(size).copy()
        y = r.f32(size).copy()
        r.skip(4 * size)                     # z
        if etype in OBJECT_TYPES:
            r.skip(4 * size * 4)             # vx, vy, vz, heading
            valid = r.f32(size)              # read as raw then view as int
            valid = np.frombuffer(valid.tobytes(), dtype="<i4").copy()
        else:
            valid = None
        r.skip(4 * 6)                        # width,length,height,goalx,goaly,goalz
        r.skip(4)                            # mark_as_expert

        if etype == ROAD_LANE:
            lanes.append((x, y))
        elif etype == VEHICLE:
            vehicles.append((x, y, valid))
    return lanes, vehicles


def total_heading_change(x, y):
    if len(x) < 3:
        return 0.0
    h = np.arctan2(np.diff(y), np.diff(x))
    dh = (np.diff(h) + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(np.abs(dh)))


def polyline_len(x, y):
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def build_segments(lanes):
    """Stack all lane segments into start (M,2) and end (M,2) arrays."""
    starts, ends = [], []
    for x, y in lanes:
        if len(x) < 2:
            continue
        pts = np.stack([x, y], axis=1)
        starts.append(pts[:-1])
        ends.append(pts[1:])
    if not starts:
        return None, None
    return np.concatenate(starts, 0), np.concatenate(ends, 0)


def min_dist_points_to_segments(P, A, B, chunk=512):
    """Min distance from each point in P (N,2) to any segment [A,B] (M,2)."""
    out = np.empty(len(P), dtype=np.float32)
    AB = B - A                                   # (M,2)
    denom = np.einsum("md,md->m", AB, AB) + 1e-9  # (M,)
    for i in range(0, len(P), chunk):
        p = P[i:i + chunk]                       # (c,2)
        ap = p[:, None, :] - A[None, :, :]       # (c,M,2)
        t = np.einsum("cmd,md->cm", ap, AB) / denom
        t = np.clip(t, 0.0, 1.0)
        proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]
        d = np.linalg.norm(p[:, None, :] - proj, axis=2)   # (c,M)
        out[i:i + chunk] = d.min(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_dir", default="resources/drive/binaries/training")
    ap.add_argument("--num_maps", type=int, default=30)
    ap.add_argument("--curve_deg", type=float, default=30.0,
                    help="Lane with > this total bend counts as a turn connector.")
    ap.add_argument("--turn_deg_per_step", type=float, default=3.0,
                    help="Human heading change/step above this = a 'turning' point.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.map_dir, "*.bin")))[:args.num_maps]
    if not files:
        print(f"No .bin files in {args.map_dir}")
        return

    curve_thr = math.radians(args.curve_deg)
    turn_thr = math.radians(args.turn_deg_per_step)

    type_counts = {}
    lane_curv, lane_len = [], []
    # human-point lane distances, split by straight/turning
    d_straight, d_turn = [], []

    for path in files:
        lanes, vehicles = parse_map(path)
        type_counts[ROAD_LANE] = type_counts.get(ROAD_LANE, 0) + len(lanes)
        type_counts[VEHICLE] = type_counts.get(VEHICLE, 0) + len(vehicles)

        for x, y in lanes:
            lane_curv.append(total_heading_change(x, y))
            lane_len.append(polyline_len(x, y))

        A, B = build_segments(lanes)
        if A is None:
            continue

        # gather valid human points + per-point turning flag
        pts, turning = [], []
        for x, y, valid in vehicles:
            if valid is None:
                continue
            v = valid.astype(bool)
            if v.sum() < 3:
                continue
            xy = np.stack([x, y], axis=1)
            h = np.arctan2(np.diff(y), np.diff(x))
            dh = np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi)   # len size-2
            # align a turning flag to interior points
            for k in range(1, len(x) - 1):
                if v[k] and v[k - 1] and v[k + 1]:
                    pts.append(xy[k])
                    turning.append(dh[k - 1] > turn_thr)
        if not pts:
            continue
        P = np.asarray(pts, dtype=np.float32)
        turning = np.asarray(turning, dtype=bool)
        d = min_dist_points_to_segments(P, A, B)
        d_straight.append(d[~turning])
        d_turn.append(d[turning])

    print(f"Parsed {len(files)} maps from {args.map_dir}\n")
    print("Entity counts (totals):")
    for t in (VEHICLE, ROAD_LANE):
        print(f"  {TYPE_NAME[t]:<11} {type_counts.get(t, 0)}  "
              f"({type_counts.get(t, 0) / len(files):.0f}/map)")

    lc, ll = np.array(lane_curv), np.array(lane_len)
    n = len(lc)
    n_curved = int(np.sum(lc > curve_thr))
    print(f"\nROAD_LANE curvature ({n} lanes):")
    print(f"  turn connectors (> {args.curve_deg:.0f}deg bend): {n_curved} "
          f"({100 * n_curved / max(n,1):.1f}%)")
    print(f"  median bend {math.degrees(np.median(lc)):.1f}deg | "
          f"90pct {math.degrees(np.percentile(lc, 90)):.1f}deg | "
          f"max {math.degrees(lc.max()):.1f}deg")
    print(f"  lane length: median {np.median(ll):.1f}m | min {ll.min():.1f}m")

    ds = np.concatenate(d_straight) if d_straight else np.array([])
    dt = np.concatenate(d_turn) if d_turn else np.array([])
    print(f"\nHuman trajectory distance to nearest lane centerline:")
    for label, d in (("STRAIGHT points", ds), ("TURNING  points", dt)):
        if len(d) == 0:
            print(f"  {label}: (none)")
            continue
        frac_far = 100 * np.mean(d > LANE_THRESHOLD)
        print(f"  {label}: n={len(d):>7} | median {np.median(d):.2f}m | "
              f"90pct {np.percentile(d, 90):.2f}m | "
              f">{LANE_THRESHOLD:.0f}m: {frac_far:.1f}%")

    if len(dt):
        far = np.mean(dt > LANE_THRESHOLD)
        verdict = ("turns ARE covered (reward works through junctions)"
                   if far < 0.15 else
                   "turns are POORLY covered (reward gives 0 through many turns)")
        print(f"\n  => {100*far:.1f}% of human TURNING points are >4m from any lane "
              f"-> {verdict}")


if __name__ == "__main__":
    main()
