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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--policy_hidden", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_cpu", action="store_true",
                   help="run on CPU anyway. Off by default so a missing GPU "
                        "fails loudly instead of silently taking 40x longer")
    p.add_argument("--patience", type=int, default=5,
                   help="stop after this many epochs with no val improvement")
    return p.parse_args()


def evaluate(tau, obs, act, device, bs=8192):
    """Held-out accuracy, plus the accel/steer split of it."""
    tau.eval()
    n = len(act)
    loss = hit = hit5 = hit_a = hit_s = 0
    with torch.no_grad():
        for i in range(0, n, bs):
            o = obs[i:i + bs].to(device, non_blocking=True)
            y = act[i:i + bs].to(device, non_blocking=True)
            logits, _, _, _ = tau(o, tau.initial_state(len(o), device))
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
    print("[bc] train %d  val %d (contiguous tail, not a random split)"
          % (len(tr_a), len(va_a)))

    # class balance of the target, for context on the accuracy number
    top = torch.bincount(act, minlength=n_actions).float()
    print("[bc] majority-class baseline: %.3f (action %d)"
          % ((top.max() / n).item(), int(top.argmax())))

    tau = RomaPolicy(obs_dim=obs_dim, action_dim=n_actions, role_dim=0,
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
            logits, _, _, _ = tau(o, tau.initial_state(len(o), device))
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tau.parameters(), 0.5)
            opt.step()
            tot += loss.item() * len(mb)
        vl, acc, acc5, acc_a, acc_s = evaluate(tau, va_o, va_a, device)
        print("[bc] epoch %2d  train_loss %.4f  val_loss %.4f  "
              "acc %.4f  top5 %.4f  accel %.4f  steer %.4f"
              % (ep, tot / len(tr_a), vl, acc, acc5, acc_a, acc_s))
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save({"state_dict": tau.state_dict(), "obs_dim": obs_dim,
                        "action_dim": n_actions, "role_dim": 0,
                        "policy_hidden": a.policy_hidden,
                        "val_loss": vl, "val_acc": acc, "epoch": ep,
                        "data": str(a.data)}, a.out)
        else:
            bad += 1
            if bad >= a.patience:
                print("[bc] no val improvement for %d epochs -- stopping" % bad)
                break

    ck = torch.load(a.out, map_location="cpu")
    print("\n[bc] best epoch %d  val_loss %.4f  val_acc %.4f -> %s"
          % (ck["epoch"], ck["val_loss"], ck["val_acc"], a.out))
    if ck["val_acc"] < 0.90:
        print("[bc] WARNING: HR-PPO's BC reference reached 0.97-0.99 open-loop.")
        print("     Below ~0.90 tau is not a usable anchor -- the KL term would")
        print("     pull the policy toward noise. Check the accel vs steer")
        print("     split above: if accel is high and steer is low, the")
        print("     steering labels are the problem, not the observation.")


if __name__ == "__main__":
    main()
