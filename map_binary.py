"""
map_binary.py -- reader for PufferDrive map_*.bin, the exact inverse of
drive.py:save_map_binary.

Why this exists: the processed Waymo JSONs are not on the cluster, but every
piece of map semantics we need survived into the binaries. save_map_binary
remaps Waymo's map_element_id into a small type code and writes it per road
element, so the 10k binaries already carry stop signs, crosswalks and lane
CENTERLINES -- none of which the env exposes (get_road_edge_polylines gives
edges only).

Layout, read straight off the writer (little-endian, tightly packed, no
alignment padding since every field is a separate 4-byte pack):

  16s  scenario_id                 (NUL/garbage padded, truncated to 16 bytes)
  i    sdc_track_index
  i    n_tracks_to_predict
  i * n_tracks_to_predict          track indices
  i    num_objects
  i    num_roads
  objects[num_objects]:
    i  unique_map_id
    i  type            1=vehicle 2=pedestrian 3=cyclist
    i  id
    i  array_size      always 91
    f*91 x   f*91 y   f*91 z
    f*91 vx  f*91 vy  f*91 vz
    f*91 heading
    i*91 valid
    f  width  f length  f height
    f  goal_x f goal_y  f goal_z
    i  mark_as_expert
  roads[num_roads]:
    i  unique_map_id
    i  type            see ROAD_TYPE below
    i  id
    i  size            polyline point count (decimated by simplify_polyline)
    f*size x   f*size y   f*size z
    f  width  f length  f height
    f  goal_x f goal_y  f goal_z
    i  mark_as_expert

ROAD TYPE CODES -- decoded from the remap chain in save_map_binary:
    map_element_id 0-3  -> 4   lane centerline (undefined/freeway/surface/bike)
                   5-13 -> 5   road line (lane markings)
                   14-16-> 6   road edge (boundary/median)
                   17   -> 7   STOP SIGN      (single point)
                   18   -> 8   CROSSWALK      (polygon)
                   19   -> 9   speed bump
                   20   -> 10  driveway
    ("lane" and "road_edge" string types are pre-mapped to 2 and 15, which
     then fall into the 4 and 6 buckets respectively.)

Usage as a module:
    from map_binary import read_map_binary, ROAD_TYPE
    m = read_map_binary("map_000.bin")
    stops = [r for r in m["roads"] if r["type"] == 7]

Usage as a CLI census over a map directory:
    python map_binary.py --data_dir <dir of map_*.bin> --n 200
"""

import argparse
import struct
from collections import Counter
from pathlib import Path

import numpy as np

TRAJ_LEN = 91

ROAD_TYPE = {
    4:  "lane_center",
    5:  "road_line",
    6:  "road_edge",
    7:  "stop_sign",
    8:  "crosswalk",
    9:  "speed_bump",
    10: "driveway",
}
OBJ_TYPE = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}

# The two that directly mark an intersection-like place.
INTERSECTION_TYPES = (7, 8)


class _R:
    """Sequential little-endian reader over a bytes buffer."""

    def __init__(self, buf):
        self.b = buf
        self.o = 0

    def i(self, n=1):
        v = struct.unpack_from(f"<{n}i", self.b, self.o)
        self.o += 4 * n
        return v[0] if n == 1 else np.asarray(v, dtype=np.int32)

    def f(self, n=1):
        v = struct.unpack_from(f"<{n}f", self.b, self.o)
        self.o += 4 * n
        return v[0] if n == 1 else np.asarray(v, dtype=np.float32)

    def s(self, n):
        v = struct.unpack_from(f"<{n}s", self.b, self.o)[0]
        self.o += n
        return v


def read_map_binary(path, objects=True):
    """Parse one map_*.bin. objects=False skips the (large) trajectory blocks
    and returns roads only -- much faster when you just want map geometry."""
    r = _R(Path(path).read_bytes())

    raw_sid = r.s(16)
    # The writer packs with '16s', so a shorter id is NUL-padded; a longer one
    # is truncated. Cut at the first NUL and keep only printable ASCII.
    sid = raw_sid.split(b"\x00", 1)[0].decode("utf-8", "replace")
    sid = "".join(c for c in sid if 32 <= ord(c) < 127)

    sdc = r.i()
    n_ttp = r.i()
    ttp = [r.i() for _ in range(n_ttp)]
    n_obj = r.i()
    n_road = r.i()

    objs = []
    for _ in range(n_obj):
        r.i()                      # unique_map_id
        otype = r.i()
        oid = r.i()
        asz = r.i()
        if objects:
            x, y, z = r.f(asz), r.f(asz), r.f(asz)
            vx, vy, vz = r.f(asz), r.f(asz), r.f(asz)
            head = r.f(asz)
            valid = r.i(asz)
            w, ln, h = r.f(), r.f(), r.f()
            gx, gy, gz = r.f(), r.f(), r.f()
            mae = r.i()
            objs.append(dict(type=otype, id=oid, x=x, y=y, z=z,
                             vx=vx, vy=vy, vz=vz, heading=head,
                             valid=valid.astype(bool), width=w, length=ln,
                             height=h, goal=(gx, gy, gz), mark_as_expert=mae))
        else:
            # 6 float arrays + heading + valid + 6 scalars + 1 int
            r.o += 4 * (asz * 8) + 4 * 6 + 4

    roads = []
    for _ in range(n_road):
        r.i()                      # unique_map_id
        rtype = r.i()
        rid = r.i()
        size = r.i()
        x, y, z = r.f(size), r.f(size), r.f(size)
        w, ln, h = r.f(), r.f(), r.f()
        r.f(); r.f(); r.f()        # goalPosition, unused for roads
        mae = r.i()
        roads.append(dict(type=rtype, id=rid, n=size,
                          x=np.atleast_1d(x), y=np.atleast_1d(y),
                          z=np.atleast_1d(z),
                          width=w, length=ln, height=h, mark_as_expert=mae))

    if r.o != len(r.b):
        raise ValueError(f"{path}: parsed {r.o} bytes of {len(r.b)} -- layout "
                         f"mismatch (n_obj={n_obj} n_road={n_road})")
    return dict(scenario_id=sid, sdc_track_index=sdc, tracks_to_predict=ttp,
                objects=objs, roads=roads)


# ---------------------------------------------------------------------------
# CLI census
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n", type=int, default=200, help="how many maps to scan")
    args = p.parse_args()

    files = sorted(Path(args.data_dir).glob("map_*.bin"))[:args.n]
    if not files:
        raise SystemExit(f"no map_*.bin in {args.data_dir}")

    tcount = Counter()
    per_scene_stop, per_scene_cross, per_scene_lane = [], [], []
    npts = {}
    id_unique, id_total = 0, 0
    lane_lengths = []
    bad = 0

    for fp in files:
        try:
            m = read_map_binary(fp, objects=False)
        except Exception as e:
            bad += 1
            if bad <= 3:
                print(f"  PARSE FAIL {fp.name}: {e}")
            continue
        rs = m["roads"]
        for rd in rs:
            tcount[rd["type"]] += 1
            npts.setdefault(rd["type"], []).append(rd["n"])
        per_scene_stop.append(sum(1 for rd in rs if rd["type"] == 7))
        per_scene_cross.append(sum(1 for rd in rs if rd["type"] == 8))
        per_scene_lane.append(sum(1 for rd in rs if rd["type"] == 4))
        ids = [rd["id"] for rd in rs]
        id_total += len(ids); id_unique += len(set(ids))
        for rd in rs:
            if rd["type"] == 4 and rd["n"] >= 2:
                lane_lengths.append(float(
                    np.hypot(np.diff(rd["x"]), np.diff(rd["y"])).sum()))

    ok = len(files) - bad
    print(f"\n  scanned {ok}/{len(files)} maps ({bad} parse failures)\n")
    print(f"  {'type':>4}  {'name':<14} {'count':>8}  {'pts/elem med':>12}")
    for t, c in sorted(tcount.items()):
        med = int(np.median(npts[t])) if npts.get(t) else 0
        print(f"  {t:>4}  {ROAD_TYPE.get(t, '?'):<14} {c:>8}  {med:>12}")

    def stat(v, lab):
        v = np.asarray(v)
        print(f"    {lab:<22} mean={v.mean():6.2f}  median={np.median(v):5.1f}  "
              f"max={v.max():4d}  scenes with >=1: {100*(v>0).mean():5.1f}%")

    print("\n  PER-SCENE COUNTS")
    stat(per_scene_stop,  "stop signs")
    stat(per_scene_cross, "crosswalks")
    stat(per_scene_lane,  "lane centerlines")

    print("\n  ROAD ID GRANULARITY")
    print(f"    {id_total} elements, {id_unique} unique ids")
    verdict = ("UNIQUE per element: id is a SEGMENT id, not a road id"
               if id_unique == id_total else
               "ids REPEAT: id groups elements, usable as a road id")
    print(f"    -> {verdict}")
    if lane_lengths:
        L = np.asarray(lane_lengths)
        print(f"\n  LANE CENTERLINE LENGTHS (m): median={np.median(L):.1f} "
              f"p10={np.percentile(L,10):.1f} p90={np.percentile(L,90):.1f}")
        print("    Short median = one road is chopped into many pieces, so a")
        print("    change of lane id does NOT by itself mean 'turned onto")
        print("    another road'.")


if __name__ == "__main__":
    main()
