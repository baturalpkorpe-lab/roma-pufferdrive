"""train_bc.py -- train tau, the frozen human reference policy for the anchor.

WHAT TAU IS FOR
Two things, and it is the same network for both:
  PRIOR   initialise PPO from these weights instead of from scratch.
  ANCHOR  freeze it, and during PPO add lambda * KL(tau(.|o) || pi(.|o))
          evaluated on the states the POLICY visits. That is the whole
          difference from a BC loss: a BC loss lives on dataset states and is
          open-loop; the anchor asks "what would a human do from HERE",
          including at states no human ever reached.

WHY FEEDFORWARD
tau is a RomaPolicy with role_dim=0, always called with a fresh initial_state,
so it is Markov: tau(a | o). That matches the reference implementation --
HR-PPO's reg_ppo.py calls self.reg_policy.get_distribution(observations) with
no hidden state -- and it means the KL is a plain per-step categorical KL.
Using RomaPolicy rather than a fresh MLP is deliberate: it reads the
observation through the same ego/partner/road encoders and the same split the
policy uses, including the off-by-one (ego_features is really 8, not 7). If tau
read the vector differently the anchor would be comparing two different
readings of the same observation.

INPUT
bc_dataset.npz from verify_bc_data.py -- obs (N, 1121) and act (N,) with
act = accel_idx * 13 + steer_idx, on states verified to be within --keep_tol
of a logged human position.

THE NUMBER THAT MATTERS
Held-out action accuracy. HR-PPO's BC reference reached 97-99% open-loop. Well
below that and tau is not a usable anchor, and the KL term would be pulling the
policy toward noise. Top-5 and the accel/steer marginals are reported too: a
tau that gets the joint index wrong but the accel right is a steering problem,
and the split tells you which.

    python train_bc.py --data /scratch/$USER/bc_probe_tight/bc_dataset.npz \
                       --out  /scratch/$USER/checkpoints/tau/tau.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roma_pufferdrive.roma.policy import RomaPolicy

N_STEER = 13


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="bc_dataset.npz")
    p.add_argument("--out", required=True, help="checkpoint path for tau")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--policy_hidden", type=int, default=128)
    p.add_argument("--gt_regimes", default="",
                   help="regimes_gt.csv. Given it, tau becomes ROLE-CONDITIONED: "
                        "tau(a|o,z) with z the driver's own measured style. An "
                        "unconditioned anchor pulls every alpha toward one "
                        "reference and narrows the dial (measured: 0.576 s at "
                        "lambda=0.02 down to 0.168 at 0.1). A conditioned one "
                        "pulls OUTWARD at the extremes instead.")
    p.add_argument("--style_scope", default="within_scene",
                   choices=["within_scene", "global"],
                   help="within_scene isolates the DRIVER from the scene, but "
                        "needs >1 driver per scenario. A dataset built with "
                        "control_sdc_only has exactly one, so centring makes z "
                        "identically zero -- rebuild the data with "
                        "--control_mode control_vehicles instead. 'global' is "
                        "the escape hatch: usable, but it conflates driver with "
                        "traffic density (corr to the true driver effect 0.52 "
                        "vs 0.94 for within-scene).")
    p.add_argument("--style_col", default="headway_T",
                   help="which measured style becomes z. headway_T is the one "
                        "with real within-scene human spread (sd 0.858 vs the "
                        "policy's 0.32-0.36)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_cpu", action="store_true",
                   help="run on CPU anyway. Off by default so a missing GPU "
                        "fails loudly instead of silently taking 40x longer")
    p.add_argument("--patience", type=int, default=15,
                   help="stop after this many epochs with no val improvement")
    return p.parse_args()


def attach_style(d, obs, act, gt_regimes, style_col, scope="within_scene"):
    """Join each pair to its driver's measured style, and standardise it.

    z is the driver's style WITHIN ITS OWN SCENE, not against the global mean.
    That matters: most of the variation in raw headway is the scene (traffic
    density, geometry), and only the within-scene part is the driver. Scene
    ICC on this quantity is 0.26, so ~three quarters of it is genuinely the
    person -- which is exactly the part a style-conditioned anchor should
    carry, and the part the role dial is trying to reproduce.

    Standardising also puts z on the same scale the policy's alpha uses, so
    forcing alpha into tau and into pi means the same thing in both.
    """
    import pandas as pd
    if "scenario_id" not in d:
        raise SystemExit(
            "This dataset has no scenario_id/vehicle_id, so it cannot be joined\n"
            "to regimes_gt.csv. Regenerate it with the current verify_bc_data.py\n"
            "(the ids were added for exactly this), then rerun.")
    df = pd.DataFrame({"scenario_id": np.asarray(d["scenario_id"]).astype(str),
                       "vehicle_id": np.asarray(d["vehicle_id"]).astype(np.int64),
                       "row": np.arange(len(act))})
    g = pd.read_csv(gt_regimes)
    if style_col not in g.columns:
        raise SystemExit("--style_col %r not in %s. Columns: %s"
                         % (style_col, gt_regimes, sorted(g.columns)[:15]))
    g = g[["scenario_id", "vehicle_id", style_col]].copy()
    g["scenario_id"] = g["scenario_id"].astype(str)
    g["vehicle_id"] = pd.to_numeric(g["vehicle_id"], errors="coerce")
    g = g.dropna(subset=["vehicle_id"])
    g["vehicle_id"] = g["vehicle_id"].astype(np.int64)

    j = df.merge(g, on=["scenario_id", "vehicle_id"], how="inner")
    if len(j) < 0.05 * len(df):
        # role_on_human.py hit the same thing: the env returns an int
        # scenario_id while regimes_gt.csv may carry a longer string form.
        j = df.assign(scenario_id=df.scenario_id.str[:16]).merge(
            g.assign(scenario_id=g.scenario_id.str[:16]),
            on=["scenario_id", "vehicle_id"], how="inner")
    if len(j) < 1000:
        raise SystemExit(
            "only %d pairs joined to %s -- the scenario_id forms do not match.\n"
            "Compare a few: dataset %r vs csv %r"
            % (len(j), gt_regimes, df.scenario_id.iloc[0], g.scenario_id.iloc[0]))

    y = pd.to_numeric(j[style_col], errors="coerce")
    j = j[np.isfinite(y)]
    y = y[np.isfinite(y)]
    # How many distinct drivers per scenario does this dataset actually have?
    # With one, within-scene centring is identically zero and tau would be
    # conditioned on a constant -- which is exactly what happened the first
    # time (p10=p50=p90=+0.00, gain 0.24 against 1.43 unconditioned).
    per_scene = j.groupby("scenario_id")["vehicle_id"].nunique()
    n_drivers = float(per_scene.median())
    print("[bc] drivers per scenario: median %.1f  max %d  (%d scenarios)"
          % (n_drivers, int(per_scene.max()), len(per_scene)))

    if scope == "within_scene":
        if n_drivers < 1.5:
            raise SystemExit("\n".join([
                    "within-scene centring needs more than one driver per",
                    "scenario, and this dataset has a median of %.1f." % n_drivers,
                    "Subtracting the group mean would make z identically zero,",
                    "so tau would be conditioned on a constant.",
                    "",
                    "The cause is control_sdc_only: one SDC per map. Rebuild",
                    "the pairs with several drivers per scene:",
                    "  python verify_bc_data.py --out_dir <dir> --keep_tol 0.2 \\",
                    "      --control_mode control_vehicles --total_agents 1024",
                    "",
                    "Or pass --style_scope global to use each driver's style",
                    "against the population instead. That works, but it",
                    "conflates the driver with the scene (corr to the true",
                    "driver effect 0.52, against 0.94 for within-scene).",
                ]))
        centred = y - y.groupby(j["scenario_id"]).transform("mean")
    else:
        centred = y - y.mean()
    zv = (centred / (centred.std() + 1e-8)).to_numpy(np.float32)

    keep = j["row"].to_numpy()
    print("[bc] style join   : %d / %d pairs carry a %s label (%.0f%%)"
          % (len(keep), len(df), style_col, 100 * len(keep) / len(df)))
    print("[bc] z (within-scene, standardised): p10 %+.2f  p50 %+.2f  p90 %+.2f"
          % (np.percentile(zv, 10), np.median(zv), np.percentile(zv, 90)))
    print("[bc] tau is ROLE-CONDITIONED: tau(a|o,z)  scope=%s" % scope)
    return (obs[keep], act[keep],
            torch.from_numpy(zv).float().unsqueeze(-1))


def evaluate(tau, obs, act, device, z=None, bs=8192):
    """Held-out accuracy, plus the accel/steer split of it."""
    tau.eval()
    n = len(act)
    loss = hit = hit5 = hit_a = hit_s = 0
    with torch.no_grad():
        for i in range(0, n, bs):
            o = obs[i:i + bs].to(device, non_blocking=True)
            y = act[i:i + bs].to(device, non_blocking=True)
            fr = None if z is None else z[i:i + bs].to(device, non_blocking=True)
            logits, _, _, _ = tau(o, tau.initial_state(len(o), device),
                                  forced_role=fr)
            logits = logits.float()
            loss += F.cross_entropy(logits, y, reduction="sum").item()
            pred = logits.argmax(-1)
            hit += (pred == y).sum().item()
            hit5 += (logits.topk(5, -1).indices == y[:, None]).any(-1).sum().item()
            hit_a += (pred // N_STEER == y // N_STEER).sum().item()
            hit_s += (pred % N_STEER == y % N_STEER).sum().item()
    tau.train()
    return loss / n, hit / n, hit5 / n, hit_a / n, hit_s / n


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device)
    if device.type == "cpu" and not a.allow_cpu:
        raise SystemExit("\n".join([
            "No CUDA device. This is GPU work -- 289k x 1121 through the",
            "encoders for 20 epochs -- and login nodes have no GPU, so this",
            "would grind for tens of minutes on a node that is not meant for",
            "compute. Submit it instead:",
            "  sbatch --export=ALL,DATA=%s slurm/train_bc.sbatch" % a.data,
            "Pass --allow_cpu if you really mean to run it here.",
        ]))

    d = np.load(a.data)
    obs = torch.from_numpy(d["obs"]).float()
    act = torch.from_numpy(d["act"]).long()
    n, obs_dim = obs.shape
    z, role_dim = None, 0
    if a.gt_regimes:
        obs, act, z = attach_style(d, obs, act, a.gt_regimes, a.style_col,
                                   a.style_scope)
        n, obs_dim = obs.shape
        role_dim = 1
    n_actions = int(d["accel_values"].shape[0] * d["steer_values"].shape[0])
    print("[bc] %s" % a.data)
    print("[bc] %d pairs  obs_dim=%d  n_actions=%d  device=%s"
          % (n, obs_dim, n_actions, device))

    # A held-out split drawn at random is optimistic here: consecutive steps of
    # one agent are near-duplicates, so a random split leaks. Split on a
    # contiguous tail instead -- different agents and different maps.
    n_val = int(n * a.val_frac)
    tr_o, tr_a = obs[:n - n_val], act[:n - n_val]
    va_o, va_a = obs[n - n_val:], act[n - n_val:]
    tr_z = None if z is None else z[:n - n_val]
    va_z = None if z is None else z[n - n_val:]
    print("[bc] train %d  val %d (contiguous tail, not a random split)"
          % (len(tr_a), len(va_a)))

    # The marginal action distribution is the baseline that matters, because
    # the anchor consumes tau's DISTRIBUTION, not its argmax: KL(tau||pi) pulls
    # pi toward the probability mass tau assigns. A tau that rarely wins the
    # argmax but puts the mass in the right place is a good anchor; a
    # confidently wrong one with high accuracy is a bad one. So score against
    # the entropy of the marginal -- that is what a tau which learned nothing
    # state-conditional would achieve.
    cnt = torch.bincount(act, minlength=n_actions).float()
    pm = cnt / cnt.sum()
    nzm = pm[pm > 0]
    h0 = float(-(nzm * nzm.log()).sum())
    print("[bc] majority class   : %.3f (action %d)"
          % ((cnt.max() / n).item(), int(cnt.argmax())))
    print("[bc] marginal entropy : %.4f nats = %.1f effective bins"
          % (h0, float(np.exp(h0))))
    print("[bc] a tau that learns nothing conditional scores val_loss %.4f" % h0)

    tau = RomaPolicy(obs_dim=obs_dim, action_dim=n_actions, role_dim=role_dim,
                     policy_hidden=a.policy_hidden).to(device)
    print("[bc] tau params: %d"
          % sum(p.numel() for p in tau.parameters() if p.requires_grad))
    opt = Adam(tau.parameters(), lr=a.lr)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    best, bad = float("inf"), 0
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(len(tr_a))
        tot = 0.0
        for i in range(0, len(perm), a.batch_size):
            mb = perm[i:i + a.batch_size]
            o = tr_o[mb].to(device, non_blocking=True)
            y = tr_a[mb].to(device, non_blocking=True)
            fr = None if tr_z is None else tr_z[mb].to(device, non_blocking=True)
            logits, _, _, _ = tau(o, tau.initial_state(len(o), device),
                                  forced_role=fr)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tau.parameters(), 0.5)
            opt.step()
            tot += loss.item() * len(mb)
        vl, acc, acc5, acc_a, acc_s = evaluate(tau, va_o, va_a, device, va_z)
        print("[bc] epoch %3d  train %.4f  val %.4f | gain %+.4f nats  "
              "eff.bins %5.1f | acc %.4f  top5 %.4f  accel %.4f  steer %.4f"
              % (ep, tot / len(tr_a), vl, h0 - vl, float(np.exp(vl)),
                 acc, acc5, acc_a, acc_s))
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save({"state_dict": tau.state_dict(), "obs_dim": obs_dim,
                        "action_dim": n_actions, "role_dim": role_dim,
                        "style_col": a.style_col if role_dim else None,
                        "policy_hidden": a.policy_hidden,
                        "val_loss": vl, "val_acc": acc, "epoch": ep,
                        "marginal_entropy": h0, "gain_nats": h0 - vl,
                        "data": str(a.data)}, a.out)
        else:
            bad += 1
            if bad >= a.patience:
                print("[bc] no val improvement for %d epochs -- stopping" % bad)
                break

    ck = torch.load(a.out, map_location="cpu")
    gain = ck["gain_nats"]
    print("")
    print("[bc] best epoch %d  val_loss %.4f  gain %+.4f nats  acc %.4f -> %s"
          % (ck["epoch"], ck["val_loss"], gain, ck["val_acc"], a.out))
    print("[bc] tau narrowed the action distribution from %.1f effective bins "
          "to %.1f"
          % (float(np.exp(ck["marginal_entropy"])), float(np.exp(ck["val_loss"]))))
    if gain < 0.3:
        print("[bc] WARNING: under 0.3 nats of conditional information. tau is")
        print("     barely better than the marginal, so KL(tau||pi) would mostly")
        print("     pull pi toward a fixed action histogram rather than toward")
        print("     state-appropriate behaviour. Not worth anchoring to.")
    elif ck["epoch"] >= a.epochs:
        print("[bc] NOTE: stopped at the epoch cap rather than by patience, so")
        print("     val_loss was probably still falling. Raise --epochs.")
    if ck["val_acc"] < 0.5:
        print("[bc] Low argmax accuracy is EXPECTED here and is not the gate.")
        print("     At dt=0.1 the per-step acceleration is mostly noise around a")
        print("     slow intent, and the accel bins are 1.33 m/s^2 apart, so the")
        print("     conditional mean genuinely is near zero -- predicting it is")
        print("     the right answer, not a failure. The anchor consumes tau's")
        print("     DISTRIBUTION, not its argmax, so read `gain` instead.")
        print("     To move the argmax, tau has to stop being Markov: the policy")
        print("     is recurrent for exactly this reason, and a recurrent tau")
        print("     needs the sequence indices verify_bc_data.py does not yet")
        print("     save (it flattens t-major, so order is lost).")


if __name__ == "__main__":
    main()
