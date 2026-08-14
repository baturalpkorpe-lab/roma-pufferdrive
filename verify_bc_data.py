"""verify_bc_data.py -- can this env hand us (observation, human action) pairs?

WHY IT MATTERS
The BC anchor and the BC init both need exactly one thing, and it is the same
thing: pairs of (obs_t, a_t) where obs_t is an observation taken at a state a
human actually drove through, and a_t is what that human did there. Nothing
downstream -- training tau, the KL term, the role-conditioned anchor -- can
start until this is settled. So settle it first and settle it cheaply.

WHY IT IS NOT OBVIOUSLY BLOCKED
role_on_human.py established that log-replayed UNCONTROLLED agents get no
observation row: drive.py sets `num_agents = agent_offsets[-1]`, sizing the obs
tensor from the CONTROLLED set, so the cars created by "create_all_valid" move
from their logs but are invisible to us. That blocker does not automatically
apply here. Under control_sdc_only the SDC IS controlled, so it does have an
obs row. If the env can tell us the SDC's logged action, we step the SDC with
it, the SDC retraces its own logged path, and every step of that rollout is a
labelled training pair. This script tests that and only that.

WHAT IT CHECKS, IN ORDER -- three independent failure modes
  1. API    Does this build expose human/expert actions at all, and under what
            name? Printed unconditionally: if the answer is no, you still walk
            away knowing the exact surface the env does offer, which is what
            you need in order to ask for the right thing.
  2. TRACK  Step the controlled agents with those actions and compare the path
            they take against get_ground_truth_trajectories(). If the agent
            does not reproduce the human's path, the actions are not the
            human's actions -- inverse dynamics is wrong, or the discrete grid
            is too coarse to express human control, or the dynamics model has
            drifted. Pairs collected under a failing TRACK are mislabelled and
            training on them is worse than not training.
  3. GRID   Are the 91 discrete bins actually used? If human driving collapses
            into a handful of bins, the discretisation has already destroyed
            the style variation the anchor is supposed to import, and tau will
            be degenerate before it is trained. TRACK can pass while GRID
            fails -- coarse-but-accurate is a real outcome.

The verdict block at the bottom reports each separately, because each has a
different fix.

    python verify_bc_data.py --out_dir /scratch/$USER/bc_probe

Run it on the guid stack (the C env with prep_human_data / infer_human_actions
compiled in). On the baseline stack expect stage 1 to come back empty -- that
is a valid, informative result, not a crash.
"""

import argparse
import ast
import configparser
import os
from pathlib import Path

import numpy as np

EPISODE_LEN = 91   # WOMD timesteps
N_ACTIONS   = 91   # discrete action grid (coincidentally the same number)


# --- copied, deliberately, from role_on_human.py --------------------------
# This script has to run on the guid stack, where role_on_human.py does not
# exist (it lives on baseline) and where importing it would drag in
# render_topdown -> torch -> matplotlib for two numpy helpers. Everything
# here is numpy-only so it runs on any stack. The orientation logic below is
# the hard-won part (see 29167a9, "squeeze GT arrays before orienting them"):
# guessing the layout without squeezing first silently sums the wrong axis.

def _sq(a):
    a = np.asarray(a)
    return a.reshape(-1) if a.ndim > 1 and 1 in a.shape else a


def as_agent_time(arr, B):
    """GT per-step arrays -> (B, T).

    They arrive with a leading or trailing singleton dimension, e.g. (1, B, T)
    or (B, T, 1), and in either orientation. Squeeze first, then orient on B.
    """
    a = np.squeeze(np.asarray(arr, dtype=float))
    if a.ndim == 1:
        a = a.reshape(B, -1)
    elif a.ndim > 2:
        a = a.reshape(B, -1) if a.shape[0] == B else a.reshape(-1, B).T
    return a if a.shape[0] == B else a.T


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="",
                   help="map binaries. Default: derived from the installed "
                        "pufferlib package, so it works from any cwd and "
                        "always matches the stack you activated")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--map_pool", type=int, default=200)
    p.add_argument("--total_agents", type=int, default=512)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--control_mode", default="control_sdc_only")
    p.add_argument("--ade_tol", type=float, default=1.0,
                   help="metres of mean deviation from ground truth below "
                        "which an agent counts as having tracked the human")
    p.add_argument("--no_prep_flag", action="store_true",
                   help="skip prep_human_data=True (use if it makes Drive "
                        "raise, to see what the env exposes without it)")
    return p.parse_args()


def resolve_data_dir(cli):
    """Locate the map binaries.

    A relative default only resolved from the clone root, which is not where
    you run this from. Derive it from the INSTALLED pufferlib instead -- the
    same package load_drive_config() reads drive.ini out of -- so the maps
    always belong to the stack you activated rather than to whichever
    directory you happened to be standing in.
    """
    if cli:
        return cli
    import pufferlib
    pd = Path(pufferlib.__file__).resolve().parent
    cands = [pd / "resources" / "drive" / "binaries" / "training",
             pd.parent / "pufferlib" / "resources" / "drive" / "binaries" / "training",
             Path("pufferlib/resources/drive/binaries/training")]
    for c in cands:
        if (c / "map_000.bin").exists():
            return str(c)
    raise SystemExit(
        "Could not find map_000.bin. Tried:\n  " +
        "\n  ".join(str(c) for c in cands) +
        "\nPass --data_dir explicitly. Find it with:\n"
        "  find /scratch/$USER -name 'map_000.bin' -printf '%h\\n' 2>/dev/null | head")


def load_drive_config():
    """Same reader train_roma.py uses -- drive.ini [env] section, parsed."""
    import pufferlib
    puffer_dir  = os.path.dirname(pufferlib.__file__)
    default_ini = os.path.join(puffer_dir, "config", "default.ini")
    drive_ini   = os.path.join(puffer_dir, "config", "ocean", "drive.ini")
    p = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    p.read([default_ini, drive_ini])

    def _parse(v):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v

    return {s: {k: _parse(v) for k, v in p[s].items()} for s in p.sections()}


# ---------------------------------------------------------------------------
# Stage 1 -- what does this build actually expose?
# ---------------------------------------------------------------------------

METHOD_CANDIDATES = [
    "get_expert_actions", "get_human_actions", "get_expert_action",
    "get_human_action", "sample_expert_data", "infer_human_actions",
    "get_expert_trajectories", "expert_actions", "human_actions",
]
ATTR_CANDIDATES = [
    "expert_actions", "human_actions", "expert_action", "human_action",
    "expert_actions_np", "human_actions_np",
]
GT_KEY_CANDIDATES = [
    "action", "actions", "expert_action", "expert_actions",
    "human_action", "human_actions", "action_idx",
]


class ExpertSource:
    """Wraps whatever the env turned out to offer behind get(t) -> (B,).

    mode 'array'   the accessor handed back the whole episode up front. That
                   array belongs to the CURRENT episode, so it must be re-read
                   after every reset/resample -- see refresh().
    mode 'perstep' it hands back one action per agent, re-read every step.
    """

    def __init__(self, name, mode, kind, arr=None):
        self.name, self.mode, self.kind, self.arr = name, mode, kind, arr

    def _read(self, env, gt):
        if self.kind == "gt":
            return gt[self.name.split("'")[1]]
        v = getattr(env, self.name)
        return v() if callable(v) else v

    def refresh(self, env, B, gt):
        """Re-read the episode's actions. No-op for per-step sources."""
        if self.mode == "array":
            self.arr = as_agent_time(np.squeeze(np.asarray(self._read(env, gt))), B)
        return self

    def get(self, t, B, env=None, gt=None):
        if self.mode == "array":
            return np.asarray(self.arr[:, t]).reshape(B).astype(np.int64)
        return np.asarray(_sq(np.asarray(self._read(env, gt)))).reshape(B).astype(np.int64)


def _shape_of(x):
    try:
        return str(np.asarray(x).shape)
    except Exception:
        return type(x).__name__


def dump_api(env):
    """Print the env surface. This runs whether or not resolution succeeds --
    a failed probe that tells you what IS there is still a useful probe."""
    names = sorted(n for n in dir(env) if not n.startswith("__"))
    hits  = [n for n in names
             if any(k in n.lower()
                    for k in ("expert", "human", "log", "gt", "ground",
                              "action", "demo", "replay", "prep"))]
    print("\n[api] members matching expert/human/log/action/replay/prep:")
    for n in hits:
        try:
            v = getattr(env, n)
        except Exception as e:
            print("        %-34s <raised %s>" % (n, type(e).__name__))
            continue
        kind = "method" if callable(v) else _shape_of(v)
        print("        %-34s %s" % (n, kind))
    if not hits:
        print("        (none)")
    print("\n[api] full member list:")
    for i in range(0, len(names), 4):
        print("        " + "  ".join("%-26s" % n for n in names[i:i + 4]))


def resolve_expert(env, B, gt):
    """Try every plausible accessor. Returns ExpertSource or None."""
    def _accept(name, raw, kind):
        a = np.squeeze(np.asarray(raw))
        if a.ndim >= 2 and B in a.shape and EPISODE_LEN in a.shape:
            arr = as_agent_time(a, B)                    # -> (B, T)
            print("[probe] %-28s -> (B,T) array %s  ACCEPTED" % (name, arr.shape))
            return ExpertSource(name, "array", kind, arr=arr)
        if a.size == B and kind != "gt":
            # A GT-dict entry sized (B,) is one value per agent for the whole
            # episode, not a per-step action -- never a valid per-step source.
            print("[probe] %-28s -> per-step (B,)     ACCEPTED" % name)
            return ExpertSource(name, "perstep", kind)
        print("[probe] %-28s -> shape %s, neither (B,T) nor per-step (B,) -- skipped"
              % (name, a.shape))
        return None

    for name in METHOD_CANDIDATES:
        fn = getattr(env, name, None)
        if fn is None or not callable(fn):
            continue
        try:
            raw = fn()
        except TypeError as e:
            print("[probe] %-28s callable but needs args (%s)" % (name, e))
            continue
        except Exception as e:
            print("[probe] %-28s raised %s: %s" % (name, type(e).__name__, e))
            continue
        if raw is None:
            print("[probe] %-28s returned None" % name)
            continue
        src = _accept(name, raw, "env")
        if src:
            return src

    for name in ATTR_CANDIDATES:
        v = getattr(env, name, None)
        if v is None or callable(v):
            continue
        src = _accept(name, v, "env")
        if src:
            return src

    for k in GT_KEY_CANDIDATES:
        if k in gt:
            src = _accept("ground_truth['%s']" % k, gt[k], "gt")
            if src:
                return src

    return None


# ---------------------------------------------------------------------------

def main():
    a = parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import pufferlib
    from pufferlib.ocean.drive.drive import Drive
    data_dir = resolve_data_dir(a.data_dir)
    print("[env] pufferlib   : %s" % Path(pufferlib.__file__).resolve().parent)
    print("[env] map binaries: %s" % data_dir)

    env_cfg = dict(load_drive_config()["env"])
    env_cfg.update({
        "num_maps":       a.map_pool,
        "num_agents":     a.total_agents,
        "map_dir":        data_dir,
        "episode_length": EPISODE_LEN,
        "control_mode":   a.control_mode,
    })
    if not a.no_prep_flag:
        env_cfg["prep_human_data"] = True

    prep_accepted = not a.no_prep_flag
    try:
        env = Drive(**env_cfg)
    except TypeError as e:
        print("[env] Drive(**cfg) rejected a kwarg: %s" % e)
        print("[env] retrying WITHOUT prep_human_data. This is itself a")
        print("[env] finding: a build with no prep_human_data kwarg is not")
        print("[env] the guid stack, and stage 1 is expected to come back")
        print("[env] empty on it. Carried into the verdict.")
        prep_accepted = False
        env_cfg.pop("prep_human_data", None)
        env = Drive(**env_cfg)

    obs_np, _ = env.reset()
    B = env.num_agents
    print("[env] control_mode=%s  agents=%d  obs_dim=%d  prep_human_data=%s"
          % (a.control_mode, B, obs_np.shape[-1],
             "accepted" if prep_accepted else "NOT A KWARG ON THIS BUILD"))

    gt = env.get_ground_truth_trajectories()
    print("[env] ground-truth keys: %s" % sorted(gt.keys()))
    print("[env] agent-state keys : %s" % sorted(env.get_global_agent_state().keys()))

    dump_api(env)

    print("\n" + "=" * 72)
    print("  STAGE 1 -- RESOLVE THE HUMAN ACTION SOURCE")
    print("=" * 72)
    src = resolve_expert(env, B, gt)
    if src is None:
        print("\nNO HUMAN ACTION SOURCE FOUND.")
        print("Stages 2 and 3 cannot run. This is the answer, not a crash:")
        print("this build does not hand over the logged action, so the")
        print("teacher-forcing route to BC data is closed on THIS stack.")
        if not prep_accepted:
            print("\nAnd the two findings agree. Drive() here has no")
            print("prep_human_data kwarg, so this is not the guid build --")
            print("an empty stage 1 is the expected result, not a surprise.")
            print("Rerun on the clone whose C env has it. Find that clone:")
            print("  grep -rl prep_human_data /scratch/$USER/*/pufferlib/ocean/drive/")
        else:
            print("\nNote that prep_human_data WAS accepted here, so this")
            print("build has the flag but exposes no accessor under any name")
            print("I tried. The [api] dump is the important part of this run.")
        print("\nThe fallback that needs nothing from the C env: derive the")
        print("actions by inverse dynamics on the GT x/y/heading printed")
        print("above. Stage 2 is still exactly the test that validates it.")
        print("\nSend the [api] dump; it names everything that IS here.")
        return

    print("\nUsing: %s (mode=%s)" % (src.name, src.mode))

    # ---- Stage 2 -- do those actions reproduce the human's path? ----------
    print("\n" + "=" * 72)
    print("  STAGE 2 -- DO THOSE ACTIONS REPRODUCE THE HUMAN PATH?")
    print("=" * 72)

    all_obs, all_act, all_keep_ade = [], [], []
    for ep in range(a.episodes):
        if ep > 0:
            env.resample_maps()
        obs_np, _ = env.reset()
        gt = env.get_ground_truth_trajectories()
        src.refresh(env, B, gt)   # the action array is per-episode data

        xs = np.zeros((EPISODE_LEN, B))
        ys = np.zeros((EPISODE_LEN, B))
        ep_obs = np.zeros((EPISODE_LEN, B, obs_np.shape[-1]), np.float32)
        ep_act = np.zeros((EPISODE_LEN, B), np.int64)

        for t in range(EPISODE_LEN):
            ag = env.get_global_agent_state()
            xs[t], ys[t] = ag["x"], ag["y"]
            act = src.get(t, B, env, gt)
            ep_obs[t] = obs_np
            ep_act[t] = act
            obs_np, _, _, _, _ = env.step(act.reshape(B, 1))

        gx = as_agent_time(gt["x"], B)
        gy = as_agent_time(gt["y"], B)
        gv = as_agent_time(gt["valid"], B).astype(bool)
        isv = _sq(np.asarray(gt["is_vehicle"])).astype(bool)

        n   = min(EPISODE_LEN, gx.shape[1])
        dev = np.hypot(xs[:n].T - gx[:, :n], ys[:n].T - gy[:, :n])
        m   = gv[:, :n]
        cnt = m.sum(1)
        ade = np.where(cnt > 0, (dev * m).sum(1) / np.maximum(cnt, 1), np.inf)

        keep = isv & (cnt >= 15) & (ade < a.ade_tol)
        pool = isv & (cnt >= 15)
        print("  ep %d: %4d/%4d vehicles tracked (ADE < %.1f m)   "
              "median ADE all=%.2f m  tracked=%.2f m"
              % (ep + 1, int(keep.sum()), int(pool.sum()), a.ade_tol,
                 float(np.median(ade[pool])) if pool.any() else np.nan,
                 float(np.median(ade[keep])) if keep.any() else np.nan))

        all_obs.append(ep_obs[:n][:, keep])
        all_act.append(ep_act[:n][:, keep])
        all_keep_ade.append(ade[keep])

    n_tracked = sum(o.shape[1] for o in all_obs)
    n_pairs   = sum(int(o.shape[0] * o.shape[1]) for o in all_obs)
    track_ok  = n_tracked > 0

    # ---- Stage 3 -- is the 91-way grid actually used? ---------------------
    print("\n" + "=" * 72)
    print("  STAGE 3 -- IS THE 91-WAY ACTION GRID ACTUALLY USED?")
    print("=" * 72)
    grid_ok = False
    if track_ok:
        acts = np.concatenate([x.reshape(-1) for x in all_act])
        acts = acts[(acts >= 0) & (acts < N_ACTIONS)]
        hist = np.bincount(acts, minlength=N_ACTIONS).astype(float)
        p    = hist / max(hist.sum(), 1)
        nz   = p[p > 0]
        ent  = float(-(nz * np.log(nz)).sum())
        perp = float(np.exp(ent))
        srt  = np.sort(p)[::-1]
        n90  = int(np.searchsorted(np.cumsum(srt), 0.90) + 1)
        top  = np.argsort(p)[::-1][:5]
        print("  bins used           : %d / %d" % (int((hist > 0).sum()), N_ACTIONS))
        print("  entropy             : %.3f nats (max %.3f)"
              % (ent, float(np.log(N_ACTIONS))))
        print("  effective bins      : %.1f  (exp of entropy)" % perp)
        print("  bins for 90%% of mass: %d" % n90)
        print("  top-5 bins          : " + "  ".join(
            "%d=%.1f%%" % (b, 100 * p[b]) for b in top))
        grid_ok = perp >= 5.0
    else:
        print("  skipped -- nothing tracked in stage 2")

    # ---- Save whatever we got --------------------------------------------
    if track_ok:
        obs_cat = np.concatenate([o.reshape(-1, o.shape[-1]) for o in all_obs])
        act_cat = np.concatenate([x.reshape(-1) for x in all_act])
        k = min(len(act_cat), 20_000)   # 20k x 1121 floats ~ 90 MB; enough to eyeball
        np.savez_compressed(out / "bc_pairs_sample.npz",
                            obs=obs_cat[:k], act=act_cat[:k],
                            ade=np.concatenate(all_keep_ade))
        print("\n[save] %d pairs (%d saved) -> %s/bc_pairs_sample.npz"
              % (n_pairs, k, out))

    # ---- Verdict ----------------------------------------------------------
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    print("  1 API    PASS -- human actions available as %s" % src.name)
    print("  2 TRACK  %s -- %d agents reproduced their logged path, %d pairs"
          % ("PASS" if track_ok else "FAIL", n_tracked, n_pairs))
    if not track_ok:
        print("           The actions exist but do not steer the agent back")
        print("           onto the human's path. Do NOT train on these pairs.")
        print("           Causes, in order of likelihood: the discrete grid")
        print("           cannot express human control finely enough; the")
        print("           inverse dynamics assumes a different vehicle model;")
        print("           the actions are indexed by a different agent order")
        print("           than get_global_agent_state(). Check the last one")
        print("           first -- it is the cheapest and the most common.")
    print("  3 GRID   %s" % ("PASS" if grid_ok else
                             ("FAIL" if track_ok else "N/A")))
    if track_ok and not grid_ok:
        print("           Human driving collapses into too few bins. tau would")
        print("           be near-deterministic, the KL anchor would carry no")
        print("           style information, and the role-conditioned version")
        print("           is pointless. Fix the discretisation before tau.")
    if track_ok and grid_ok:
        print("\n  BC data collection works. tau can be trained from this path.")
        print("  Scale check: HR-PPO needed ~30 min of driving (200 scenarios")
        print("  x 91 steps). You have %d pairs from %d maps here."
              % (n_pairs, a.map_pool))


if __name__ == "__main__":
    main()
