"""
probe_map_semantics.py -- WHAT map information do we actually have?

Read-only reconnaissance for the intersection-aware trajectory clustering. It
answers, with evidence rather than assumption, the questions that decide which
clustering designs are even possible:

  1. Do the processed Waymo JSONs carry road ELEMENT TYPES (lane / road_edge /
     stop_sign / crosswalk / speed_bump)? Stop signs and crosswalks are direct
     intersection markers.
  2. Is there traffic-light state (tl_states / dynamic_map_states)?
  3. Is there a road/lane ID -- and is it a per-ROAD id or a per-SEGMENT id?
     The distinction decides whether "changed road id" can mean "turned onto a
     different road". Measured, not guessed: an id that segments one straight
     road shows up as many short collinear elements sharing a corridor.
  4. Are lane centerlines present (needed for lane-topology junction
     detection), or only road EDGES (which the env exposes)?
  5. What are the 7 ROAD_FEAT channels the policy sees? Which are continuous
     geometry and which are categorical (a type channel would let the POLICY
     see stop signs, and would let us detect junctions from obs alone)?
  6. What does the env API expose beyond get_road_edge_polylines?

Nothing here trains, steps, or writes to the dataset. Run it once and paste
the report.

Usage (from the PufferDrive root so the C binding finds drive.ini):
    python probe_map_semantics.py --json_dir <processed waymo jsons> \
        --data_dir pufferlib/resources/drive/binaries/training

--json_dir may be omitted; common locations are searched. --data_dir may be
omitted to skip the env half (JSON half needs no GPU and no pufferlib).
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir", type=str, default=None,
                   help="Directory of processed Waymo scenario JSONs. If unset, "
                        "a few common locations are searched.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Map binary dir for the env probe. Unset = skip env.")
    p.add_argument("--n_json",   type=int, default=5,
                   help="How many JSONs to sample for the schema report.")
    p.add_argument("--num_agents", type=int, default=256)
    p.add_argument("--num_maps",   type=int, default=64)
    p.add_argument("--out", type=str, default="map_semantics_report.txt")
    return p.parse_args()


def hr(t):
    return "\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70


def find_jsons(explicit):
    if explicit:
        f = sorted(glob.glob(os.path.join(explicit, "*.json")))
        return f, explicit
    cands = [
        os.path.expanduser("~/waymo_json"),
        os.path.expanduser("~/data/waymo"),
        "/scratch/$USER/waymo_json",
        os.path.expandvars("/scratch/$USER/waymo_json"),
        os.path.expandvars("/scratch/$USER/waymo_processed"),
        os.path.expandvars("/scratch/$USER/data/processed"),
        os.path.expandvars("/scratch/$USER/PufferDrive/data"),
    ]
    for c in cands:
        f = sorted(glob.glob(os.path.join(c, "*.json")))
        if f:
            return f, c
    return [], None


# ---------------------------------------------------------------------------
# 1-4: the JSON side
# ---------------------------------------------------------------------------

def probe_json(files, n, log):
    log(hr("A. PROCESSED-JSON SCHEMA"))
    if not files:
        log("  NO JSONs FOUND. Pass --json_dir. Without them we cannot know")
        log("  whether stop signs / lane types / traffic lights exist at all.")
        return None

    log(f"  found {len(files)} json files; sampling {min(n, len(files))}")
    road_key = None
    type_counter = Counter()
    id_fields = Counter()
    geom_len_by_type = defaultdict(list)
    tl_report = []
    top_keys = Counter()

    for fp in files[:n]:
        with open(fp) as f:
            d = json.load(f)
        top_keys.update(d.keys())

        # which key holds the map elements?
        for k in ("roads", "map_features", "road_graph", "roadgraph"):
            if k in d and isinstance(d[k], list) and d[k]:
                road_key = road_key or k
        # traffic lights
        for k in ("tl_states", "dynamic_map_states", "traffic_lights",
                  "tl_state", "dynamic_states"):
            if k in d:
                v = d[k]
                tl_report.append((k, type(v).__name__,
                                  len(v) if hasattr(v, "__len__") else "?"))

        if road_key and road_key in d:
            for el in d[road_key]:
                if not isinstance(el, dict):
                    continue
                t = el.get("type", "<no type key>")
                type_counter[str(t)] += 1
                for idk in ("id", "map_element_id", "lane_id", "road_id",
                            "feature_id", "index"):
                    if idk in el:
                        id_fields[idk] += 1
                g = el.get("geometry", el.get("points", el.get("polyline")))
                if isinstance(g, list):
                    geom_len_by_type[str(t)].append(len(g))

    log(f"\n  top-level keys seen: {sorted(top_keys)}")
    log(f"  map-element key    : {road_key or 'NONE FOUND'}")

    if not road_key:
        log("  -> No road/map-feature array. Intersection detection would have")
        log("     to come from trajectories + road-edge geometry only.")
        return None

    log(f"\n  ROAD ELEMENT TYPES (counts over sampled scenes):")
    for t, c in type_counter.most_common():
        L = geom_len_by_type.get(t, [])
        pts = f"pts/elem med={int(np.median(L))} max={max(L)}" if L else ""
        log(f"    {t:<24} {c:>7}   {pts}")

    log(f"\n  ID FIELDS on elements: {dict(id_fields) or 'NONE'}")

    # --- the road_id question: per-ROAD or per-SEGMENT? --------------------
    log(hr("B. IS THE ROAD/LANE ID PER-ROAD OR PER-SEGMENT?"))
    log("  Test: for the element type with the most instances, measure how")
    log("  long each element is and whether ids repeat. Many SHORT elements")
    log("  with unique ids = the id segments a road into pieces, so 'id")
    log("  changed' does NOT mean 'turned onto another road'.")
    with open(files[0]) as f:
        d0 = json.load(f)
    els = d0.get(road_key, [])
    by_type = defaultdict(list)
    for el in els:
        if isinstance(el, dict):
            by_type[str(el.get("type", "?"))].append(el)
    for t, group in sorted(by_type.items(), key=lambda kv: -len(kv[1]))[:4]:
        lens, ids = [], []
        for el in group:
            g = el.get("geometry", el.get("points", el.get("polyline")))
            if isinstance(g, list) and len(g) >= 2:
                try:
                    xy = np.array([[p["x"], p["y"]] for p in g], float)
                except (TypeError, KeyError):
                    xy = np.asarray(g, float)[:, :2]
                lens.append(float(np.hypot(*np.diff(xy, axis=0).T).sum()))
            for idk in ("id", "map_element_id", "lane_id", "road_id"):
                if idk in el:
                    ids.append(el[idk]); break
        if lens:
            log(f"\n    type={t}  n_elements={len(group)}")
            log(f"      element length (m): med={np.median(lens):.1f} "
                f"p10={np.percentile(lens,10):.1f} p90={np.percentile(lens,90):.1f}")
            if ids:
                log(f"      ids: {len(ids)} total, {len(set(ids))} unique "
                    f"-> {'UNIQUE per element (segment id)' if len(set(ids))==len(ids) else 'REPEATS (groups elements = road id)'}")

    if tl_report:
        log(hr("C. TRAFFIC LIGHTS"))
        for k, tp, n_ in tl_report:
            log(f"    key={k}  type={tp}  len={n_}")
        for fp in files[:1]:
            with open(fp) as f:
                d = json.load(f)
            for k, _, _ in tl_report[:1]:
                v = d[k]
                log(f"    sample of {k}: {json.dumps(v)[:600]}")
    else:
        log(hr("C. TRAFFIC LIGHTS"))
        log("    none of tl_states / dynamic_map_states / traffic_lights present")

    return road_key


# ---------------------------------------------------------------------------
# 5-6: the env side
# ---------------------------------------------------------------------------

def probe_env(args, log):
    log(hr("D. ENV API + THE 7 ROAD OBS CHANNELS"))
    try:
        import ast, configparser, pufferlib
        from pufferlib.ocean.drive.drive import Drive
    except Exception as e:
        log(f"  env import failed ({e}) -- skipping. Run from the PufferDrive root.")
        return

    pd_ = os.path.dirname(pufferlib.__file__)
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cp.read([os.path.join(pd_, "config", "default.ini"),
             os.path.join(pd_, "config", "ocean", "drive.ini")])

    def _p(v):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    cfg = dict((k, _p(v)) for k, v in cp["env"].items()) if "env" in cp else {}
    cfg.update(num_maps=args.num_maps, num_agents=args.num_agents,
               map_dir=args.data_dir)
    env = Drive(**cfg)
    obs, _ = env.reset()

    pub = [a for a in dir(env) if not a.startswith("_") and callable(getattr(env, a, None))]
    log(f"  env public methods: {pub}")

    for name in pub:
        if not name.startswith("get_"):
            continue
        try:
            v = getattr(env, name)()
        except Exception as e:
            log(f"    {name}() -> raised {type(e).__name__}")
            continue
        if isinstance(v, dict):
            log(f"    {name}() -> dict keys={list(v.keys())}")
            for k, vv in list(v.items())[:14]:
                a = np.asarray(vv)
                log(f"        {k:<22} shape={a.shape} dtype={a.dtype} "
                    f"uniq={min(np.unique(a).size, 9999)}")
        else:
            a = np.asarray(v)
            log(f"    {name}() -> array shape={a.shape} dtype={a.dtype}")

    # --- the 7 road channels ------------------------------------------------
    EGO, NP_, PF, MR, RF = 7, 31, 7, 128, 7
    road = np.asarray(obs)[:, EGO + NP_ * PF:].reshape(-1, MR, RF)
    log(f"\n  road block reshaped to (agents, {MR} roads, {RF} feats) = {road.shape}")
    log(f"  {'ch':>3} {'min':>10} {'max':>10} {'mean':>10} {'n_uniq':>8}  verdict")
    for c in range(RF):
        v = road[:, :, c].ravel()
        v = v[np.isfinite(v)]
        u = np.unique(v)
        verdict = ("CATEGORICAL <-- candidate type/id channel"
                   if u.size <= 24 else "continuous (geometry)")
        log(f"  {c:>3} {v.min():>10.3f} {v.max():>10.3f} {v.mean():>10.3f} "
            f"{u.size:>8}  {verdict}")
        if u.size <= 24:
            log(f"       values: {np.round(u, 4).tolist()}")
            nz = road[:, :, c].ravel()
            cnt = Counter(np.round(nz, 4).tolist())
            log(f"       counts: {dict(cnt.most_common(24))}")

    log("\n  NOTE: a categorical channel here means the POLICY already sees the")
    log("  element type, and junctions could be detected from obs directly.")
    log("  All-continuous means road type is NOT in the observation and must")
    log("  come from the JSON (or be inferred from geometry).")
    env.close()


def main():
    args = parse_args()
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(str(s))

    files, where = find_jsons(args.json_dir)
    if where:
        log(f"[probe] json dir: {where}")
    probe_json(files, args.n_json, log)
    if args.data_dir:
        probe_env(args, log)
    else:
        log(hr("D. ENV PROBE SKIPPED (--data_dir not given)"))

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    log(f"\n[probe] report written to {args.out}")


if __name__ == "__main__":
    main()
