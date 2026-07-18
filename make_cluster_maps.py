"""
Per-cluster training map builder for cluster-conditioned ROMA training.

For each trajectory cluster K of the stop_frac K-means clustering
(trajectory_clusters_stopfrac.csv: scenario_id, vehicle_id -> cluster),
every scene containing >= --min_control cluster-K vehicles is re-emitted as
a binary map in which ONLY those vehicles are policy-controllable. Every
other object (other vehicles, pedestrians, cyclists) gets mark_as_expert=1:
the env then replays it along its GT trajectory (drive.h:
should_control_agent() skips mark_as_expert entities, move_expert() replays
them each step; init_mode=create_all_valid keeps them in the scene).

NO C changes and NO training-code changes: the flag lives inside the map
binary (drive.py save_map_binary), and training just points --data_dir at
the per-cluster directory. Maps are renumbered map_000.bin.. sequentially
per cluster (the C binding loads "%s/map_%03d.bin").

Respawn note: respawn_agent() resets the SAME entity onto its OWN traj[0],
so an agent never re-deals onto an out-of-cluster trajectory.

Output:
  <out_root>/cluster<K>/map_000.bin ...   renumbered binaries
  <out_root>/cluster<K>/manifest.csv      map_id, scenario_id, src_json, n_control
  summary printed with the exact --num_maps per cluster.

Run with PYTHONPATH including a PufferDrive checkout (imports save_map_binary):
  PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
  python make_cluster_maps.py \
      --clusters_csv /scratch/e452103/traj_atlas/trajectory_clusters_stopfrac.csv \
      --json_dir     <dir with the processed Waymo JSONs> \
      --out_root     /scratch/e452103/cluster_maps
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from pufferlib.ocean.drive.drive import save_map_binary

_SID_RE = re.compile(r'"scenario_id"\s*:\s*"([^"]+)"')

# Pool worker globals (set by _init_worker; inherited via fork on Linux).
_INDEX = None
_OUT_ROOT = None
_NEUTRALIZE_SDC = True


def _init_worker(index, out_root, neutralize_sdc):
    global _INDEX, _OUT_ROOT, _NEUTRALIZE_SDC
    _INDEX = index
    _OUT_ROOT = out_root
    _NEUTRALIZE_SDC = neutralize_sdc


def quick_sid(path):
    """Scenario id without a full json parse when possible.

    The binary header truncates scenario_id to 16 bytes (struct '16s'), and
    the cluster CSV ids came from that header -- so match on [:16].
    """
    try:
        with open(path, "r") as f:
            head = f.read(4096)
            m = _SID_RE.search(head)
            if m:
                return m.group(1)[:16]
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            m = _SID_RE.search(f.read())
            if m:
                return m.group(1)[:16]
        with open(path, "r") as f:  # slow fallback: key buried mid-file
            return str(json.load(f).get("scenario_id", ""))[:16]
    except Exception:
        return ""


def _is_vehicle(obj):
    t = obj.get("type", 1)
    return t == "vehicle" or t == 1


def build_one(task):
    """Write one source scene into every cluster dir that includes it.

    task = (json_path, [(cluster, new_map_id), ...])
    Returns manifest rows [(cluster, map_id, sid, src, n_control), ...].
    """
    path, jobs = task
    with open(path, "r") as f:
        md = json.load(f)
    sid = str(md.get("scenario_id", ""))[:16]
    objects = md.get("objects", [])
    meta = md.setdefault("metadata", {})
    orig_sdc = meta.get("sdc_track_index", -1)
    rows = []
    for k, map_id in jobs:
        members = _INDEX[sid][k]
        n_control = 0
        # Unconditional re-flag of EVERY object: cluster-K vehicles stay
        # controllable, everything else (incl. peds/cyclists -- drive.ini
        # uses control_agents, which would otherwise control them too)
        # becomes GT expert replay.
        for obj in objects:
            if _is_vehicle(obj) and int(obj.get("id", -1)) in members:
                obj["mark_as_expert"] = 0
                n_control += 1
            else:
                obj["mark_as_expert"] = 1
        # SDC (ego) leakage fix: the C env's set_active_agents() force-activates
        # the scenario's sdc_track_index UNCONDITIONALLY -- it never checks
        # mark_as_expert -- so a foreign-cluster ego is controlled anyway (~18%
        # of controlled agents, verified). If the SDC is NOT a cluster-K member,
        # blank its index so the env cannot force-control it. Kept per-k because
        # a cluster-K ego is legitimately controllable.
        sdc_flag = -1  # -1 = no sdc, 0 = foreign sdc neutralized, 1 = cluster-K sdc kept
        if 0 <= orig_sdc < len(objects):
            sdc_is_k = (_is_vehicle(objects[orig_sdc])
                        and int(objects[orig_sdc].get("id", -1)) in members)
            if sdc_is_k or not _NEUTRALIZE_SDC:
                meta["sdc_track_index"] = orig_sdc
                sdc_flag = 1 if sdc_is_k else 0
            else:
                meta["sdc_track_index"] = -1
                sdc_flag = 0
        # ALWAYS write, even on an id mismatch (n_control == 0): the C binding
        # loads map_%03d.bin by index, so a hole in the numbering is a crash,
        # while a 0-controllable map is merely skipped by binding.shared().
        out = Path(_OUT_ROOT) / f"cluster{k}" / f"map_{map_id:03d}.bin"
        save_map_binary(md, str(out), map_id)
        rows.append((k, map_id, sid, path.name, n_control, sdc_flag))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters_csv", type=str,
                    default="/scratch/e452103/traj_atlas/trajectory_clusters_stopfrac.csv",
                    help="frozen stop_frac clustering (scenario_id, vehicle_id, cluster)")
    ap.add_argument("--json_dir", type=str, required=True,
                    help="dir with the processed Waymo scenario JSONs "
                         "(the same set the training binaries were built from)")
    ap.add_argument("--out_root", type=str,
                    default="/scratch/e452103/cluster_maps")
    ap.add_argument("--clusters", type=str, default="0,1,2,3")
    ap.add_argument("--min_control", type=int, default=1,
                    help="min cluster-K vehicles for a scene to enter cluster K's set")
    ap.add_argument("--drop_edge", action="store_true",
                    help="exclude is_edge==1 vehicles from the control sets")
    ap.add_argument("--keep_foreign_sdc", action="store_true",
                    help="OLD behaviour: leave each scene's ego (SDC) index "
                         "intact even when the ego is not a cluster-K vehicle. "
                         "Default (off) blanks the SDC index for foreign egos so "
                         "the env can't force-control them (~18%% leakage fix).")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    clusters = [int(c) for c in args.clusters.split(",")]

    # -- 1. cluster index: sid -> {K: set(vehicle_ids)} -----------------------
    index = defaultdict(lambda: defaultdict(set))
    n_rows = n_edge_dropped = 0
    with open(args.clusters_csv) as f:
        for row in csv.DictReader(f):
            k = int(row["cluster"])
            if k not in clusters:
                continue
            if args.drop_edge and int(row.get("is_edge", 0) or 0) == 1:
                n_edge_dropped += 1
                continue
            index[str(row["scenario_id"])[:16]][k].add(int(row["vehicle_id"]))
            n_rows += 1
    index = {s: dict(d) for s, d in index.items()}
    print(f"[cmaps] {args.clusters_csv}: {n_rows} labeled vehicles in "
          f"{len(index)} scenes" +
          (f" ({n_edge_dropped} edge vehicles dropped)" if args.drop_edge else ""))

    # -- 2. pass 1: sid of every source JSON (cheap regex, parse fallback) ----
    json_files = sorted(Path(args.json_dir).glob("*.json"))
    if not json_files:
        sys.exit(f"[cmaps] no *.json in {args.json_dir}")
    with Pool(args.workers) as pool:
        sids = pool.map(quick_sid, json_files, chunksize=64)

    # -- 3. assign sequential per-cluster map ids ------------------------------
    next_id = {k: 0 for k in clusters}
    tasks, sid_seen = [], set()
    for path, sid in zip(json_files, sids):
        if not sid or sid not in index or sid in sid_seen:
            continue
        sid_seen.add(sid)
        jobs = [(k, next_id[k]) for k in clusters
                if len(index[sid].get(k, ())) >= args.min_control]
        for k, _ in jobs:
            next_id[k] += 1
        if jobs:
            tasks.append((path, jobs))
    unmatched = len(index) - len(sid_seen)
    if unmatched:
        print(f"[cmaps] WARNING: {unmatched} CSV scenes not found among the "
              f"JSONs (different map pool?)")

    # -- 4. pass 2: patch + write binaries ------------------------------------
    out_root = Path(args.out_root)
    for k in clusters:
        (out_root / f"cluster{k}").mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(args.workers, initializer=_init_worker,
              initargs=(index, str(out_root), not args.keep_foreign_sdc)) as pool:
        for out in pool.imap_unordered(build_one, tasks, chunksize=16):
            rows.extend(out)

    # -- 5. manifests + summary ------------------------------------------------
    by_k = defaultdict(list)
    for r in rows:
        by_k[r[0]].append(r)
    print(f"\n[cmaps] {'cluster':>8} {'scenes':>7} {'ctl_vehicles':>13} {'ctl/scene':>10}")
    for k in clusters:
        rs = sorted(by_k[k], key=lambda r: r[1])
        with open(out_root / f"cluster{k}" / "manifest.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["map_id", "scenario_id", "src_json", "n_control", "sdc_flag"])
            for _, mid, sid, src, nc, sf in rs:
                w.writerow([mid, sid, src, nc, sf])
        n_scenes = len(rs)
        n_ctl = sum(r[4] for r in rs)
        n_sdc_neutralized = sum(1 for r in rs if r[5] == 0)
        n_sdc_kept = sum(1 for r in rs if r[5] == 1)
        n_empty = sum(1 for r in rs if r[4] < args.min_control)
        if n_empty:
            print(f"[cmaps] WARNING cluster {k}: {n_empty} scenes have "
                  f"n_control < {args.min_control} (CSV vehicle ids didn't "
                  f"match the JSON objects) -- written anyway to keep the "
                  f"map numbering contiguous; the env skips them at sampling.")
        print(f"[cmaps] {k:>8} {n_scenes:>7} {n_ctl:>13} "
              f"{(n_ctl / max(n_scenes, 1)):>10.2f}")
        sdc_msg = (f"kept {n_sdc_kept} cluster-{k} egos, neutralized "
                   f"{n_sdc_neutralized} foreign egos"
                   if not args.keep_foreign_sdc
                   else f"kept ALL egos ({n_sdc_kept} cluster-{k} + "
                        f"{n_sdc_neutralized} foreign; --keep_foreign_sdc)")
        print(f"         SDC: {sdc_msg}")
        print(f"         -> sbatch --export=ALL,CLUSTER={k} slurm/train_cluster.sbatch"
              f"   (num_maps auto = {n_scenes})")
    print(f"\n[cmaps] done -> {out_root}/cluster<K>/map_XXX.bin + manifest.csv")


if __name__ == "__main__":
    main()
