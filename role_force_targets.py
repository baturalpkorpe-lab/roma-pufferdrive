"""
role_force_targets.py -- force SPECIFIC (sid, vid) agents' roles by a chosen
alpha along PC1 (natural-offset: z(t) = z_nat(t) + alpha*sigma*PC1) and render
natural-vs-forced so the behavioural change is visible.

Re-locating a specific scene in a 10k-map pool is a ~3.4%/deal lottery (the
role_natural_extremes lesson). This sidesteps it: it SCANS the binaries for the
target scenes (each binary's 16-byte header IS the scenario_id), copies just
those into a mini map dir, and runs the env on that -- so every target scene is
always present and re-deal succeeds on the first reset.

Usage (from PufferDrive; targets = 'sidprefix:vid:alpha' comma-separated):
  python role_force_targets.py \
      --targets    "9b77f20fee:1746:-1,43a1ab3d25:2819:1,7f1fd16402:3:1" \
      --checkpoint /scratch/e452103/checkpoints/roma_baseline_dim2/roma_dim2_final.pt \
      --axes_csv   /scratch/e452103/role_paired/dim2/role_paired_axes.csv \
      --data_dir   pufferlib/resources/drive/binaries/training \
      --out_dir    /scratch/e452103/renders/force_targets
"""

import argparse
import struct
import shutil
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import load_policy, BG
from render_role_conditions import (rollout_forced, scene_view, hide_after_frame,
                                    render_condition_video, scene_setup, bbox_of,
                                    include_point, first_segment, focal_goal)
from render_role_alpha import load_axes, focal_numbers

T = 91


def parse_targets(s):
    """'sidprefix:vid:alpha,...' -> [(prefix, vid, alpha)]."""
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        sid, vid, alpha = item.split(":")
        out.append((sid.strip(), int(vid), float(alpha)))
    return out


def find_map_files(data_dir, prefixes):
    """{prefix: (map_file, full_sid)} by reading each binary's 16-byte header."""
    found = {}
    for fp in sorted(Path(data_dir).glob("map_*.bin")):
        with open(fp, "rb") as f:
            raw = f.read(16)
        if len(raw) < 16:
            continue
        sid = struct.unpack("16s", raw)[0].rstrip(b"\x00").decode("utf-8", "ignore")
        for p in prefixes:
            if p not in found and (sid.startswith(p) or p.startswith(sid)):
                found[p] = (str(fp), sid)
        if len(found) == len(prefixes):
            break
    return found


def overlay(views, mets, sid, vid, alpha, out_path, dpi):
    """Natural (blue) vs forced (red) focal path on the SAME scene."""
    nat, frc = views["natural"], views["forced"]
    f = nat["focal"]
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG)
    px, py = first_segment(nat["xs"][:, f], nat["ys"][:, f])
    x0, x1, y0, y1 = bbox_of(nat["road_polys"], px[:, None], py[:, None])
    x0, x1, y0, y1 = include_point(x0, x1, y0, y1, focal_goal(nat["gt"], f))
    scene_setup(ax, nat["road_polys"], nat["gt"], x0, x1, y0, y1, focal=f,
                focal_color="#dddddd")
    for name, color in [("natural", "#4477dd"), ("forced", "#dd4444")]:
        v = views[name]; fi = v["focal"]
        qx, qy = first_segment(v["xs"][:, fi], v["ys"][:, fi])
        m = mets[name]
        ax.plot(qx, qy, color=color, lw=2.6, zorder=6,
                label=f"{name}: v={m['speed_mean']:.1f}  turn={m['turn_abs']:.2f}  "
                      f"jerk={m['jerk_abs']:.1f}  coll={m['event_rate']}")
        if len(qx):
            ax.scatter(qx[-1], qy[-1], color=color, s=110, marker="X",
                       zorder=7, edgecolors="white", linewidths=0.8)
    ax.legend(fontsize=9, loc="best")
    ax.set_title(f"{sid[:10]} v{vid} | PC1 forced α={alpha:+g} (natural-offset)\n"
                 f"blue = its own natural role, red = role shifted {alpha:+g}σ along PC1",
                 color="#e8e8f0", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets",    required=True,
                    help="'sidprefix:vid:alpha,...' e.g. 9b77f20fee:1746:-1")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--axes_csv",   required=True)
    ap.add_argument("--data_dir",   required=True,
                    help="binaries dir to SCAN for the target scenes")
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--num_agents", type=int, default=256)
    ap.add_argument("--max_relocate", type=int, default=60)
    ap.add_argument("--goal_radius", type=float, default=2.0)
    ap.add_argument("--device",     type=str, default="cpu")
    ap.add_argument("--fps",        type=int, default=10)
    ap.add_argument("--dpi",        type=int, default=110)
    args = ap.parse_args()
    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    targets = parse_targets(args.targets)
    prefixes = [t[0] for t in targets]

    # -- 1. find + copy target scenes into a mini map dir ----------------------
    found = find_map_files(args.data_dir, prefixes)
    missing = [p for p in prefixes if p not in found]
    if missing:
        raise SystemExit(f"scenes not found in {args.data_dir}: {missing}")
    mini = out / "mini_maps"
    if mini.exists():
        shutil.rmtree(mini)
    mini.mkdir(parents=True)
    full_sid = {}
    for i, p in enumerate(prefixes):
        src, sid = found[p]
        shutil.copy(src, mini / f"map_{i:03d}.bin")
        full_sid[p] = sid
    print(f"[force] mini map dir ({len(prefixes)} scenes) -> {mini}")

    # -- 2. policy + PC1 axis --------------------------------------------------
    from pufferlib.ocean.drive.drive import Drive
    env = Drive(num_maps=len(prefixes), num_agents=args.num_agents,
                map_dir=str(mini), episode_length=T, goal_speed=100, seed=1)
    obs, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs.shape[-1], device)
    if role_dim == 0:
        raise SystemExit("role_dim=0 checkpoint -- nothing to force")
    mu, axes = load_axes(args.axes_csv, role_dim)
    if "PC1" not in axes:
        raise SystemExit(f"axes csv needs PC1, has {list(axes)}")
    u1, s1 = axes["PC1"]
    u1 = np.asarray(u1, dtype=np.float32)

    # -- 3. per target: natural + forced render + overlay ----------------------
    for pref, vid, alpha in targets:
        sid = full_sid[pref]
        cond = (alpha * s1 * u1).astype(np.float32)
        print(f"\n[force] {sid[:10]} v{vid}: PC1 α={alpha:+g}  "
              f"(shift {np.round(cond, 2)})", flush=True)
        views, mets, ok = {}, {}, True
        for name, cvec in [("natural", None), ("forced", cond)]:
            data = rollout_forced(env, policy, sid, vid, cvec, device,
                                  max_tries=args.max_relocate, shift=True)
            if data is None:
                print(f"  [skip] {sid[:10]} v{vid} not re-dealt"); ok = False; break
            m = focal_numbers(data)
            view = scene_view(data, data["slots"], sid)
            view["focal"] = int(np.where(data["slots"] == data["focal"])[0][0])
            view["hide_after"] = hide_after_frame(view)
            views[name] = view; mets[name] = m
            color = "#4477dd" if name == "natural" else "#dd4444"
            fn = out / f"force_{sid[:10]}_v{vid}_{name}_a{alpha:+g}.mp4"
            render_condition_video(
                view, view["focal"], color,
                f"{sid[:10]} v{vid} | {name} | PC1 α={alpha:+g} | "
                f"v={m['speed_mean']:.1f} turn={m['turn_abs']:.2f} "
                f"jerk={m['jerk_abs']:.1f} coll={m['event_rate']}",
                fn, args.fps, args.dpi, goal_radius=args.goal_radius)
        if ok and len(views) == 2:
            overlay(views, mets, sid, vid, alpha,
                    out / f"force_overlay_{sid[:10]}_v{vid}_a{alpha:+g}.png", args.dpi)
    env.close()
    print(f"\n[force] done -> {out}")


if __name__ == "__main__":
    main()
