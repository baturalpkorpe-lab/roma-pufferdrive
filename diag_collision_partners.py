"""
diag_collision_partners.py -- name the INVISIBLE collision partner in a scene.

Mirrors the condition-video pipeline EXACTLY (rollout_forced -> duplicate-
instance dedup -> scene_view -> the same OBB/SAT collision test the video's
red flag uses) and then, for every collision frame of the focal, reports WHO
the partner was and whether the video would have DRAWN it at that moment:

  * partner GT vehicle id + box size,
  * the partner's hide_after frame: the video hides a background agent once
    its human GT ends, and NEVER draws agents with no valid GT -- but the
    collision test still counts their (policy-driven) bodies,
  * whether each collision happened while the partner was hidden
    -> the "car turns red next to nothing" case.

Also lists every invisible body (never-drawn / GT-expired agent) with its
position, so you can see what the renderer is not showing.

Run from the PufferDrive checkout (so pufferlib's drive.ini resolves):
  cd /scratch/e452103/PufferDrive
  PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
  python ~/roma_pufferdrive/diag_collision_partners.py \
      --scenario   1abcfdd308 \
      --checkpoint /scratch/e452103/checkpoints/roma_cluster2_dim4/roma_dim4_final.pt \
      --data_dir   /scratch/e452103/cluster_maps/cluster2 \
      --num_agents 256
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_topdown import load_policy
from render_role_conditions import (rollout_forced, scene_view,
                                    hide_after_frame, _obb_collision,
                                    JUMP_THRESH)
from render_role_alpha import load_axes
from role_regime_analysis import _squeeze
from role_force_targets import find_map_files

T = 91


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario",   required=True,
                    help="scenario_id prefix, e.g. 1abcfdd308")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir",   required=True,
                    help="binaries dir to SCAN for the target scene")
    ap.add_argument("--num_agents", type=int, default=256)
    ap.add_argument("--device",     type=str, default="cpu")
    ap.add_argument("--seed",       type=int, default=1)
    ap.add_argument("--out_dir",    type=str, default="diag_collision")
    ap.add_argument("--max_relocate", type=int, default=40)
    ap.add_argument("--axes_csv",   type=str, default=None,
                    help="role_paired_axes.csv; enables forced PC1/PC2 to match "
                         "a render. If omitted, uses the natural rollout.")
    ap.add_argument("--pc1",        type=float, default=-1.0,
                    help="forced PC1 level (matches the render's PC1=..)")
    ap.add_argument("--pc2",        type=float, default=0.0,
                    help="forced PC2 level (matches the render's PC2=..)")
    args = ap.parse_args()
    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # -- 1. locate the scene binary by header, copy to a 1-map mini dir --------
    found = find_map_files(args.data_dir, [args.scenario])
    if args.scenario not in found:
        raise SystemExit(f"scenario '{args.scenario}' not found in {args.data_dir}")
    src, full_sid = found[args.scenario]
    mini = out / "mini_maps"
    if mini.exists():
        shutil.rmtree(mini)
    mini.mkdir(parents=True)
    shutil.copy(src, mini / "map_000.bin")
    print(f"[diag] scene {full_sid} <- {src}")

    # -- 2. env + policy + focal pick (longest-driving GT vehicle, like the
    #       sweep's select_focals / scan_allocation) ---------------------------
    from pufferlib.ocean.drive.drive import Drive
    env = Drive(num_maps=1, num_agents=args.num_agents,
                map_dir=str(mini), episode_length=T, seed=args.seed)
    obs_np, _ = env.reset()
    policy, role_dim = load_policy(args.checkpoint, obs_np.shape[-1], device)

    gt    = env.get_ground_truth_trajectories()
    sids  = _squeeze(np.asarray(gt["scenario_id"]).astype(str))
    ids   = _squeeze(np.asarray(gt["id"])).reshape(-1)
    valid = _squeeze(gt["valid"]).astype(bool)
    is_veh = np.asarray(gt["is_vehicle"]).reshape(-1).astype(bool)
    gx, gy = _squeeze(gt["x"]), _squeeze(gt["y"])

    slots0 = np.where(sids == full_sid)[0]
    if not len(slots0):
        slots0 = np.array([i for i, s in enumerate(sids)
                           if s.startswith(args.scenario)])
    if not len(slots0):
        raise SystemExit(f"scenario {full_sid} not in the allocation?!")
    uniq = np.unique(ids[slots0])
    print(f"[diag] role_dim={role_dim}  scene slots={len(slots0)} "
          f"({len(uniq)} unique vehicles -> ~{len(slots0)//max(len(uniq),1)} "
          f"parallel instances of the scene)")

    lens = np.zeros(len(slots0))
    for i, a in enumerate(slots0):
        if not is_veh[a] or valid[a].sum() < 10:
            continue
        px, py = gx[a][valid[a]], gy[a][valid[a]]
        lens[i] = np.hypot(np.diff(px), np.diff(py)).sum()
    focal_vid = int(ids[slots0[int(np.argmax(lens))]])
    print(f"[diag] focal = longest-driving vehicle: vid {focal_vid} "
          f"({lens.max():.0f} m of human GT)")

    # -- 3. the VIDEO'S OWN rollout path -- forced PC1/PC2 to match the render
    cond_vec = None
    if args.axes_csv:
        mu, axes = load_axes(args.axes_csv, role_dim)
        cond = mu.copy()
        for nm, lvl in (("PC1", args.pc1), ("PC2", args.pc2)):
            if nm in axes and lvl != 0.0:
                u, s = axes[nm]
                cond = cond + lvl * s * u
        cond_vec = cond.astype(np.float32)
        print(f"[diag] forcing focal role to PC1={args.pc1:+g} "
              f"PC2={args.pc2:+g} (matches the render)")
    else:
        print("[diag] natural rollout (no forcing) -- pass --axes_csv to match "
              "a forced render exactly")

    data = rollout_forced(env, policy, full_sid, focal_vid, cond_vec, device,
                          max_tries=args.max_relocate)
    env.close()
    if data is None:
        raise SystemExit("scene never re-dealt -- try a different --seed")

    view = scene_view(data, data["slots"], full_sid)
    vids = ids[data["slots"]]
    f    = int(np.where(data["slots"] == data["focal"])[0][0])
    hide = hide_after_frame(view)          # per-agent last DRAWN frame; -1=never
    xs, ys, hs = view["xs"], view["ys"], view["hs"]
    L, W = view["length"], view["width"]
    n = xs.shape[1]

    # the video stops flagging at the focal's first respawn teleport
    jump = np.sqrt(np.diff(xs[:, f])**2 + np.diff(ys[:, f])**2) > JUMP_THRESH
    tp   = np.where(jump)[0]
    end  = int(tp[0] + 1) if len(tp) else T

    # -- 4. invisible bodies in this scene view --------------------------------
    if hide is not None:
        never   = [a for a in range(n) if a != f and hide[a] < 0]
        expired = [a for a in range(n) if a != f and 0 <= hide[a] < T - 1]
        print(f"\n[diag] scene view: {n} agents | invisible bodies: "
              f"{len(never)} NEVER drawn (no valid GT), "
              f"{len(expired)} hidden mid-episode (GT ends before t=90)")
        for a in never[:10]:
            ok = np.isfinite(xs[:, a]) & np.isfinite(ys[:, a])
            if ok.any():
                print(f"         never-drawn vid {int(vids[a])}: sits near "
                      f"({np.median(xs[ok, a]):.1f}, {np.median(ys[ok, a]):.1f})"
                      f"  size {L[a]:.1f}x{W[a]:.1f}")
        for a in expired[:10]:
            print(f"         vid {int(vids[a])}: drawn until t={hide[a]}, then "
                  f"HIDDEN but still simulated at "
                  f"({xs[min(hide[a]+5, T-1), a]:.1f}, "
                  f"{ys[min(hide[a]+5, T-1), a]:.1f})")

    # -- 5. focal's collisions, partner + visibility ---------------------------
    events = {}
    for t in range(end):
        if not (np.isfinite(xs[t, f]) and np.isfinite(ys[t, f])):
            continue
        for b in range(n):
            if b == f or not (np.isfinite(xs[t, b]) and np.isfinite(ys[t, b])):
                continue
            if (xs[t, f]-xs[t, b])**2 + (ys[t, f]-ys[t, b])**2 > 225.0:
                continue
            if _obb_collision(xs[t, f], ys[t, f], hs[t, f], L[f], W[f],
                              xs[t, b], ys[t, b], hs[t, b], L[b], W[b]):
                events.setdefault(b, []).append(t)

    print(f"\n[diag] focal vid {focal_vid}: "
          f"{sum(len(v) for v in events.values())} collision frame(s) "
          f"in [0, {end}) -- the exact frames the video flags red")
    if not events:
        print("       (none in this natural rollout -- the video's forced "
              "condition\n        may have taken a slightly different path; "
              "the invisible bodies\n        listed above are still what it "
              "can hit)")
    for b, ts in sorted(events.items(), key=lambda kv: kv[1][0]):
        t0 = ts[0]
        if hide is None:
            vis = "?"
        elif hide[b] < 0:
            vis = "NEVER DRAWN (no valid GT) -- INVISIBLE partner"
        elif t0 > hide[b]:
            vis = f"HIDDEN since t={hide[b]} (GT ended) -- INVISIBLE partner"
        else:
            vis = "visible when it happened"
        print(f"         t={t0:2d}..{ts[-1]:2d} ({len(ts)} fr)  partner "
              f"vid {int(vids[b])}  size {L[b]:.1f}x{W[b]:.1f}  at "
              f"({xs[t0, b]:.1f}, {ys[t0, b]:.1f})  -> {vis}")

    # per-frame trace around each collision: focal vs partner, separation, and
    # whether both are inside the drawn view (occlusion check).
    for b, ts in sorted(events.items(), key=lambda kv: kv[1][0]):
        t0 = ts[0]
        print(f"\n[diag] trace focal vid {focal_vid} vs partner vid "
              f"{int(vids[b])} around collision t={t0}:")
        for t in range(max(0, t0 - 4), min(end, ts[-1] + 3)):
            d = float(np.hypot(xs[t, f]-xs[t, b], ys[t, f]-ys[t, b]))
            hit = "  <== BOXES OVERLAP (red flag)" if t in ts else ""
            print(f"         t={t:2d}  focal ({xs[t, f]:7.1f},{ys[t, f]:6.1f})  "
                  f"partner ({xs[t, b]:7.1f},{ys[t, b]:6.1f})  gap={d:4.1f} m{hit}")

    hid = [b for b in events if hide is not None
           and (hide[b] < 0 or events[b][0] > hide[b])]
    if hid:
        print("\n[diag] VERDICT: the red flag comes from body(ies) the video "
              "does NOT draw\n        (GT-expired or GT-less agents are hidden "
              "on screen but still simulated\n        and still counted by the "
              "collision test). Not a map wall, not a pedestrian.")
    elif events:
        print("\n[diag] VERDICT: all partners were visible cars -- check the "
              "video frame again.")
    print(f"\n[diag] done -> {out}")


if __name__ == "__main__":
    main()
