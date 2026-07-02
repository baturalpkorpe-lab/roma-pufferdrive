"""
render_topdown.py -- Headless top-down scenario videos for a ROMA checkpoint.

No raylib, no X server, no GLX: renders entirely with matplotlib (Agg) from the
same env API that WOSAC uses (get_global_agent_state / get_road_edge_polylines),
so it runs on any login or compute node.

Output: one MP4 (if ffmpeg is on PATH, e.g. `conda install -c conda-forge ffmpeg`)
or animated GIF (pillow fallback, zero extra deps) per map, in --out_dir.
Vehicles are drawn as oriented rectangles colored by their role vector
(projection on the leading principal direction of the episode's roles);
near-static agents are gray. Optionally overlays ground-truth human
trajectories as dashed lines.

Modes:
  free    roles come from the role encoder (default)
  sweep   forced role sweep, group d sweeps dim d from -1 to +1 (and mirrored),
          matching run_render_rollouts' layout -- shows role causality
  forced  every agent gets the constant vector from --force_vec

Usage (baseline stack on DelftBlue):
    PYTHONPATH=$HOME/roma_pufferdrive:/scratch/e452103/PufferDrive \
    python render_topdown.py \
        --checkpoint /scratch/e452103/checkpoints/roma_baseline/roma_dim8_step3000238080.pt \
        --data_dir /scratch/e452103/PufferDrive/pufferlib/resources/drive/binaries/training \
        --out_dir /scratch/e452103/renders/topdown --n_maps 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.collections import LineCollection, PolyCollection
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))

T = 91  # episode length / frames


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir",   type=str, required=True,
                   help="Directory containing map_*.bin binaries")
    p.add_argument("--out_dir",    type=str, default="renders_topdown")
    p.add_argument("--n_maps",     type=int, default=4)
    p.add_argument("--seed",       type=int, default=100,
                   help="Map i uses seed --seed + i")
    p.add_argument("--num_agents", type=int, default=32)
    p.add_argument("--mode",       type=str, default="free",
                   choices=["free", "sweep", "forced"])
    p.add_argument("--force_vec",  type=str, default=None,
                   help="Comma-separated role vector for --mode forced, "
                        "e.g. '1,0,0,0,0,0,0,0'")
    p.add_argument("--role_dim",   type=int, default=0,
                   help="0 = read from checkpoint args (recommended)")
    p.add_argument("--device",     type=str, default="cpu",
                   help="cpu is fine: 32 agents x 91 steps per map")
    p.add_argument("--fps",        type=int, default=10)
    p.add_argument("--dpi",        type=int, default=110)
    p.add_argument("--format",     type=str, default="auto",
                   choices=["auto", "mp4", "gif"])
    p.add_argument("--gt",         type=int, default=1,
                   help="1 = overlay ground-truth human trajectories (dashed)")
    p.add_argument("--trail",      type=int, default=20,
                   help="Trajectory trail length in frames (0 = off)")
    p.add_argument("--debug_api",  type=int, default=0,
                   help="1 = print shapes/types of env API returns and exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Policy loading (self-contained: reads hyperparams from the checkpoint)
# ---------------------------------------------------------------------------

def load_policy(ckpt_path, obs_dim, device, role_dim_override=0):
    from roma_pufferdrive.roma.policy import RomaPolicy

    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {}) or {}
    role_dim = role_dim_override or saved.get("role_dim", 8)

    policy = RomaPolicy(
        obs_dim        = obs_dim,
        action_dim     = 91,
        role_dim       = role_dim,
        role_hidden    = saved.get("role_hidden",   64),
        policy_hidden  = saved.get("policy_hidden", 128),
        var_floor      = saved.get("var_floor",     1e-4),
        obs_window_len = 8,
    ).to(device)
    key = "policy_state" if "policy_state" in ckpt else "policy"
    policy.load_state_dict(ckpt[key])
    policy.eval()
    return policy, role_dim


# ---------------------------------------------------------------------------
# Rollout: record everything first, render later (no env/policy in render)
# ---------------------------------------------------------------------------

def make_sweep_fn(role_dim, B, device):
    """Group d (4 agents) sweeps dim d: 2 agents -1 -> +1, 2 agents +1 -> -1."""
    def _sweep(t):
        roles = torch.zeros(B, role_dim)
        alpha = -1.0 + 2.0 * t / (T - 1)
        for d in range(role_dim):
            base = d * 4
            if base + 3 >= B:
                break
            roles[base,     d] =  alpha
            roles[base + 1, d] =  alpha
            roles[base + 2, d] = -alpha
            roles[base + 3, d] = -alpha
        return roles.to(device)
    return _sweep


def rollout(env, policy, role_dim, device, forced_fn=None):
    B = env.num_agents
    obs_np, _ = env.reset()

    # Static scene data -- grab right after reset, like the WOSAC collectors.
    road_edges = None
    gt         = None
    try:
        road_edges = env.get_road_edge_polylines()
    except Exception as e:
        print(f"  [road edges unavailable: {e}]")
    try:
        gt = env.get_ground_truth_trajectories()
    except Exception as e:
        print(f"  [ground truth unavailable: {e}]")

    xs   = np.zeros((T, B), dtype=np.float32)
    ys   = np.zeros((T, B), dtype=np.float32)
    hs   = np.zeros((T, B), dtype=np.float32)
    rl   = np.zeros((T, B, role_dim), dtype=np.float32)
    length = np.full(B, 4.5, dtype=np.float32)
    width  = np.full(B, 2.0, dtype=np.float32)

    obs   = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    state = policy.initial_state(B, device)

    for t in range(T):
        ag = env.get_global_agent_state()
        xs[t] = ag["x"]
        ys[t] = ag["y"]
        hs[t] = ag["heading"]
        if t == 0:
            for k_env, arr in (("length", length), ("width", width)):
                try:
                    v = np.asarray(ag[k_env], dtype=np.float32).reshape(-1)
                    if v.shape[0] == B and np.isfinite(v).all() and (v > 0).all():
                        arr[:] = v
                except Exception:
                    pass

        forced = forced_fn(t) if forced_fn is not None else None
        with torch.no_grad():
            logits, _, state, role_info = policy(obs, state, forced_role=forced)
        rl[t] = (forced if forced is not None
                 else role_info["role_mean"]).float().cpu().numpy()

        action = Categorical(logits=logits.float()).sample()
        obs_np, _, _, _, _ = env.step(action.cpu().numpy().reshape(B, 1))
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

    return {"xs": xs, "ys": ys, "hs": hs, "roles": rl,
            "length": length, "width": width,
            "road_edges": road_edges, "gt": gt}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def normalize_polylines(road_edges):
    """Best-effort conversion of get_road_edge_polylines() output into a list
    of (K, 2) float arrays. Splits on NaN rows. Returns [] if unparseable."""
    def _split_nan(arr):
        arr  = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return []
        arr  = arr[:, :2]
        bad  = ~np.isfinite(arr).all(axis=1)
        outs, start = [], 0
        for i in np.where(bad)[0].tolist() + [len(arr)]:
            if i - start >= 2:
                outs.append(arr[start:i])
            start = i + 1
        return outs

    polys = []
    try:
        if road_edges is None:
            return []
        if isinstance(road_edges, dict):
            # PufferDrive format: flattened x/y coords + per-polyline lengths
            # (see Drive.get_road_edge_polylines in pufferlib/ocean/drive/drive.py)
            if {"x", "y", "lengths"} <= set(road_edges.keys()):
                fx = np.asarray(road_edges["x"], dtype=np.float64).reshape(-1)
                fy = np.asarray(road_edges["y"], dtype=np.float64).reshape(-1)
                lengths = np.asarray(road_edges["lengths"],
                                     dtype=np.int64).reshape(-1)
                start = 0
                for L in lengths:
                    end = start + int(L)
                    pts = np.stack([fx[start:end], fy[start:end]], axis=-1)
                    pts = pts[np.isfinite(pts).all(axis=1)]
                    if len(pts) >= 2:
                        polys.append(pts)
                    start = end
                return polys
            road_edges = list(road_edges.values())
        if isinstance(road_edges, np.ndarray):
            if road_edges.ndim == 3:
                road_edges = list(road_edges)
            else:
                road_edges = [road_edges]
        for item in road_edges:
            polys.extend(_split_nan(item))
    except Exception as e:
        print(f"  [road polyline parse failed: {e}]")
        return []
    return polys


def vehicle_corners(x, y, h, length, width):
    """(N,) arrays -> (N, 4, 2) rectangle corners oriented by heading."""
    c, s = np.cos(h), np.sin(h)
    dx = np.stack([ length/2,  length/2, -length/2, -length/2], axis=-1)
    dy = np.stack([ width/2,  -width/2,  -width/2,   width/2 ], axis=-1)
    cx = x[:, None] + c[:, None]*dx - s[:, None]*dy
    cy = y[:, None] + s[:, None]*dx + c[:, None]*dy
    return np.stack([cx, cy], axis=-1)


def role_color_scalar(roles):
    """(T, B, D) roles -> (T, B) scalar in [0,1]: projection on the leading
    principal direction of episode-mean roles, robustly normalized."""
    Tn, B, D = roles.shape
    ep_mean  = roles.mean(axis=0)                     # (B, D)
    centered = ep_mean - ep_mean.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0]                                   # (D,)
    except np.linalg.LinAlgError:
        pc1 = np.zeros(D); pc1[0] = 1.0
    proj = roles @ pc1                                # (T, B)
    lo, hi = np.percentile(proj, [2, 98])
    if hi - lo < 1e-6:
        return np.full((Tn, B), 0.5, dtype=np.float32)
    return np.clip((proj - lo) / (hi - lo), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

BG, ROAD, GTC = "#0f0f1a", "#55556a", "#c9c9d8"


def render_video(data, out_path, fps, dpi, trail, want_gt):
    xs, ys, hs = data["xs"], data["ys"], data["hs"]
    B = xs.shape[1]

    polys = normalize_polylines(data["road_edges"])

    # Scene bounding box: prefer road geometry, fall back to agent positions.
    if polys:
        allp = np.concatenate(polys, axis=0)
        x0, x1 = allp[:, 0].min(), allp[:, 0].max()
        y0, y1 = allp[:, 1].min(), allp[:, 1].max()
    else:
        x0, x1 = np.percentile(xs, [2, 98])
        y0, y1 = np.percentile(ys, [2, 98])
    mx, my = 0.03 * (x1 - x0 + 1), 0.03 * (y1 - y0 + 1)
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my

    # Moving vs parked/padding (for coloring only -- everything gets drawn).
    path_len = np.sqrt(np.diff(xs, axis=0)**2 + np.diff(ys, axis=0)**2).sum(axis=0)
    moving   = path_len > 2.0                          # metres over the episode

    col_scalar = role_color_scalar(data["roles"])      # (T, B)
    try:
        cmap = matplotlib.colormaps["coolwarm"]        # matplotlib >= 3.6
    except AttributeError:
        cmap = cm.get_cmap("coolwarm")                 # removed in 3.9

    w, h = x1 - x0, y1 - y0
    fig_w = 10.0 if w >= h else 10.0 * w / h
    fig_h = 10.0 if h >= w else 10.0 * h / w
    fig, ax = plt.subplots(figsize=(max(fig_w, 4), max(fig_h, 4)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.tick_params(colors="#888899", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#333344")

    if polys:
        ax.add_collection(LineCollection(polys, colors=ROAD, linewidths=0.8))

    # Ground-truth human trajectories (dashed), best effort.
    if want_gt and isinstance(data["gt"], dict):
        try:
            gx = np.asarray(data["gt"]["x"], dtype=np.float64)
            gy = np.asarray(data["gt"]["y"], dtype=np.float64)
            gv = data["gt"].get("valid")
            gv = None if gv is None else np.asarray(gv)
            while gx.ndim > 2:                         # squeeze rollout dims
                gx, gy = gx[:, 0], gy[:, 0]
                gv = gv[:, 0] if gv is not None else None
            if gx.ndim == 1:
                gx, gy = gx[None, :], gy[None, :]
                gv = gv[None, :] if gv is not None else None
            if gv is not None and gv.shape == gx.shape:
                # Invalid GT points are stored as zeros -- mask them out.
                gx = np.where(gv.astype(bool), gx, np.nan)
                gy = np.where(gv.astype(bool), gy, np.nan)
            segs = []
            for i in range(gx.shape[0]):
                pts = np.stack([gx[i], gy[i]], axis=-1)
                pts = pts[np.isfinite(pts).all(axis=1)]
                if len(pts) < 2:
                    continue
                d = np.sqrt(np.diff(pts[:, 0])**2 + np.diff(pts[:, 1])**2).sum()
                inside = ((pts[:, 0] > x0) & (pts[:, 0] < x1)
                          & (pts[:, 1] > y0) & (pts[:, 1] < y1)).mean() > 0.5
                if d > 2.0 and inside:
                    segs.append(pts)
            if segs:
                ax.add_collection(LineCollection(
                    segs, colors=GTC, linewidths=0.7,
                    linestyles=(0, (3, 3)), alpha=0.35))
        except Exception as e:
            print(f"  [gt overlay skipped: {e}]")

    cars   = PolyCollection([], edgecolors="#000000", linewidths=0.4, zorder=5)
    trails = LineCollection([], linewidths=1.0, alpha=0.5, zorder=4)
    ax.add_collection(cars)
    ax.add_collection(trails)
    step_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="#e8e8f0",
                       fontsize=10, va="top", family="monospace")
    title = Path(str(out_path)).stem.replace("_", " ")
    ax.set_title(title, color="#e8e8f0", fontsize=11)

    def in_view(t):
        return ((xs[t] > x0) & (xs[t] < x1) & (ys[t] > y0) & (ys[t] < y1)
                & np.isfinite(xs[t]) & np.isfinite(ys[t]))

    def update(t):
        vis = in_view(t)
        idx = np.where(vis)[0]
        corners = vehicle_corners(xs[t, idx], ys[t, idx], hs[t, idx],
                                  data["length"][idx], data["width"][idx])
        cars.set_verts(list(corners))
        fc = np.array([cmap(col_scalar[t, i]) if moving[i]
                       else (0.55, 0.55, 0.60, 1.0) for i in idx])
        cars.set_facecolors(fc if len(fc) else np.zeros((0, 4)))

        if trail > 0 and t > 0:
            s = max(0, t - trail)
            segs, cols = [], []
            for i in idx:
                if not moving[i]:
                    continue
                # Break the trail at respawn teleports (>8 m in one step).
                px, py = xs[s:t+1, i], ys[s:t+1, i]
                jump = np.sqrt(np.diff(px)**2 + np.diff(py)**2) > 8.0
                cut  = np.where(jump)[0]
                start = cut[-1] + 1 if len(cut) else 0
                if t + 1 - s - start >= 2:
                    segs.append(np.stack([px[start:], py[start:]], axis=-1))
                    cols.append(cmap(col_scalar[t, i]))
            trails.set_segments(segs)
            trails.set_colors(cols if cols else [(0, 0, 0, 0)])
        step_txt.set_text(f"t = {t:2d}/{T-1}   ({t/10:.1f}s)")
        return cars, trails, step_txt

    ani = animation.FuncAnimation(fig, update, frames=T, blit=False)

    if str(out_path).endswith(".mp4"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    else:
        writer = animation.PillowWriter(fps=fps)
    ani.save(str(out_path), writer=writer, dpi=dpi,
             savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    n_mov = int(moving.sum())
    print(f"  Saved: {out_path}  ({n_mov}/{B} moving agents)")


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

    if args.format == "auto":
        ext = ".mp4" if animation.FFMpegWriter.isAvailable() else ".gif"
        if ext == ".gif":
            print("[render] ffmpeg not found -> GIF output "
                  "(conda install -c conda-forge ffmpeg for MP4)")
    else:
        ext = "." + args.format
        if ext == ".mp4" and not animation.FFMpegWriter.isAvailable():
            print("[render] WARNING: --format mp4 but ffmpeg not on PATH; "
                  "this will fail. conda install -c conda-forge ffmpeg")

    from pufferlib.ocean.drive.drive import Drive

    policy = None
    for i in range(args.n_maps):
        seed = args.seed + i
        print(f"\nMap {i+1}/{args.n_maps} (seed={seed}, mode={args.mode})")
        env = Drive(num_maps=1, num_agents=args.num_agents,
                    map_dir=args.data_dir, episode_length=T, seed=seed)

        if policy is None:
            obs_probe, _ = env.reset()
            obs_dim = obs_probe.shape[-1]
            policy, role_dim = load_policy(args.checkpoint, obs_dim, device,
                                           args.role_dim)
            print(f"[render] obs_dim={obs_dim}  role_dim={role_dim}")

        if args.debug_api:
            env.reset()
            ag = env.get_global_agent_state()
            print("agent_state keys:", {k: np.asarray(v).shape for k, v in ag.items()})
            re_ = env.get_road_edge_polylines()
            print("road_edges type:", type(re_),
                  getattr(re_, "shape", f"len={len(re_)}" if hasattr(re_, "__len__") else "?"))
            try:
                gt = env.get_ground_truth_trajectories()
                print("gt keys:", {k: np.asarray(v).shape for k, v in gt.items()})
            except Exception as e:
                print("gt unavailable:", e)
            env.close()
            return

        if args.mode == "sweep":
            forced_fn = make_sweep_fn(role_dim, args.num_agents, device)
        elif args.mode == "forced":
            if not args.force_vec:
                raise SystemExit("--mode forced requires --force_vec, "
                                 "e.g. --force_vec 1,0,0,0,0,0,0,0")
            vec = torch.tensor([float(v) for v in args.force_vec.split(",")],
                               dtype=torch.float32, device=device)
            if vec.numel() != role_dim:
                raise SystemExit(f"--force_vec has {vec.numel()} values, "
                                 f"role_dim is {role_dim}")
            fixed = vec.unsqueeze(0).expand(args.num_agents, -1)
            forced_fn = lambda t: fixed
        else:
            forced_fn = None

        data = rollout(env, policy, role_dim, device, forced_fn)
        env.close()

        out_path = out_dir / f"map{i}_seed{seed}_{args.mode}{ext}"
        render_video(data, out_path, args.fps, args.dpi, args.trail,
                     want_gt=bool(args.gt))

    print(f"\n[render] Done -> {out_dir}")


if __name__ == "__main__":
    main()
