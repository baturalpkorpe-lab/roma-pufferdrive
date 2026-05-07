"""
train_roma.py
=============
Train the ROMA policy on PufferDrive.

CPU TESTING (quick sanity check):
    PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 roma_pufferdrive/train_roma.py \
        --role_dim 1 --num_maps 100 --total_steps 2000000 \
        --num_agents 16 --device cpu --save_interval 1000000 \
        --save_dir roma_pufferdrive/checkpoints/roma_dim1_test

DELFT / SUPERCOMPUTER (full training with wandb logging):
    PYTHONPATH=/path/to/roma_pufferdrive python3 roma_pufferdrive/train_roma.py \
        --role_dim 1 --num_maps 10000 --total_steps 1000000000 \
        --num_agents 64 --device cuda --run_eval \
        --wandb_project roma-pufferdrive --wandb_entity YOUR_WANDB_USERNAME \
        --save_dir roma_pufferdrive/checkpoints/roma_dim1

DELFT without wandb (offline CSV only):
    PYTHONPATH=/path/to/roma_pufferdrive python3 roma_pufferdrive/train_roma.py \
        --role_dim 1 --num_maps 10000 --total_steps 1000000000 \
        --num_agents 64 --device cuda --run_eval --wandb_offline \
        --save_dir roma_pufferdrive/checkpoints/roma_dim1

Wandb tracks: all losses, scores, hyperparameters, learning curves.
CSV fallback always runs regardless of wandb status.
After training, evaluation (WOSAC + environment metrics) runs
automatically if --run_eval is set.
"""

import argparse
import csv
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from roma.policy     import RomaPolicy
from roma.aux_losses import RomaAuxLoss


# ---------------------------------------------------------------------------
# Wandb setup — optional, falls back to CSV if unavailable
# ---------------------------------------------------------------------------

def init_wandb(args):
    """
    Initialize wandb if available and requested.
    Returns wandb run object or None if wandb is not available/configured.

    Saeed can run with:
        --wandb_project roma-pufferdrive --wandb_entity his_username
    Or set WANDB_API_KEY environment variable before running.
    Or use --wandb_offline to log locally without an account.
    """
    if not args.wandb_project:
        print("[ROMA] wandb disabled (no --wandb_project provided). Using CSV only.")
        return None

    try:
        import wandb
        import os

        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
            print("[ROMA] wandb running in offline mode. Run 'wandb sync' later to upload.")

        run = wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity or None,
            name    = f"roma_dim{args.role_dim}_maps{args.num_maps}_{args.seed}",
            config  = vars(args),
            tags    = [f"role_dim_{args.role_dim}", f"maps_{args.num_maps}"],
        )
        print(f"[ROMA] wandb initialized: {run.url}")
        return run

    except ImportError:
        print("[ROMA] wandb not installed. Install with: uv pip install wandb")
        print("[ROMA] Falling back to CSV logging only.")
        return None
    except Exception as e:
        print(f"[ROMA] wandb init failed: {e}")
        print("[ROMA] Falling back to CSV logging only.")
        return None


def log_metrics(wandb_run, metrics_dict, step):
    """Log metrics to wandb and/or print. Never crashes if wandb is unavailable."""
    if wandb_run is not None:
        try:
            wandb_run.log(metrics_dict, step=step)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    T    = len(rewards)
    adv  = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        nv    = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        last  = delta + gamma * lam * (1 - dones[t]) * last
        adv[t] = last
    return adv


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Environment
    p.add_argument("--data_dir",      type=str,   default="resources/drive/binaries/training")
    p.add_argument("--num_agents",    type=int,   default=64,
                   help="Agents per scene. 64 on GPU, 16 on CPU.")
    p.add_argument("--num_maps",      type=int,   default=10000,
                   help="Scenarios loaded. 10000 for full training, 100 for CPU test.")
    p.add_argument("--device",        type=str,   default="cuda",
                   help="cuda or cpu. Auto-falls back to cpu if cuda unavailable.")

    # Role
    p.add_argument("--role_dim",      type=int,   default=1,
                   help="Role vector dimension. 1=original ROMA, 8=proposed extension.")
    p.add_argument("--role_hidden",   type=int,   default=64)
    p.add_argument("--policy_hidden", type=int,   default=128)
    p.add_argument("--var_floor",     type=float, default=1e-4)
    p.add_argument("--mi_weight",     type=float, default=1.0)
    p.add_argument("--div_weight",    type=float, default=0.1,
                   help="Weight on cosine diversity loss. Safe in [-1,+1] range.")

    # PPO
    p.add_argument("--total_steps",   type=int,   default=1_000_000_000,
                   help="Total training steps. 1B for full run, 2M for CPU test.")
    p.add_argument("--rollout_steps", type=int,   default=256)
    p.add_argument("--ppo_epochs",    type=int,   default=4)
    p.add_argument("--num_minibatch", type=int,   default=4)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae_lambda",    type=float, default=0.95)
    p.add_argument("--clip_coef",     type=float, default=0.2)
    p.add_argument("--ent_coef",      type=float, default=0.01)
    p.add_argument("--vf_coef",       type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    # Logging / saving
    p.add_argument("--save_dir",      type=str,   default="roma_pufferdrive/checkpoints/roma")
    p.add_argument("--save_interval", type=int,   default=100_000_000,
                   help="Save checkpoint every N steps. 100M for 1B run, 1M for CPU test.")
    p.add_argument("--log_interval",  type=int,   default=50_000)
    p.add_argument("--seed",          type=int,   default=0)

    # Wandb
    p.add_argument("--wandb_project", type=str,   default=None,
                   help="Wandb project name. If not set, wandb is disabled.")
    p.add_argument("--wandb_entity",  type=str,   default=None,
                   help="Wandb username or team. Optional.")
    p.add_argument("--wandb_offline", action="store_true",
                   help="Run wandb in offline mode. Logs saved locally, sync later.")

    # Post-training evaluation
    p.add_argument("--run_eval",          action="store_true",
                   help="Run full WOSAC + environment evaluation after training.")
    p.add_argument("--eval_episodes",     type=int,   default=30)
    p.add_argument("--wosac_rollouts",    type=int,   default=32)
    p.add_argument("--wosac_num_maps",    type=int,   default=10000)
    p.add_argument("--wosac_max_batches", type=int,   default=500)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------

def run_evaluation(args, policy, device, wandb_run=None):
    """
    Run environment metrics + WOSAC evaluation after training.
    Results saved to CSV. If wandb_run is provided, logs there too.
    """
    print("\n" + "=" * 60)
    print("  POST-TRAINING EVALUATION")
    print("=" * 60)

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(
        num_maps       = args.wosac_num_maps,
        num_agents     = args.num_agents,
        map_dir        = args.data_dir,
        episode_length = 91,
    )
    policy.eval()

    # --- Environment metrics ---
    all_scores, all_collisions    = [], []
    all_offroads, all_completions = [], []
    all_returns                   = []

    print(f"\nEnvironment metrics ({args.eval_episodes} episodes)...")
    obs_np, _ = env.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

    for ep in range(args.eval_episodes):
        if ep > 0:
            obs_np, _ = env.reset()
            obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
        state = policy.initial_state(args.num_agents, device)

        for _ in range(91):
            with torch.no_grad():
                logits, _, state, _ = policy(obs, state)
            action = Categorical(logits=logits).sample()
            obs_np, _, _, _, info = env.step(action.cpu().numpy().reshape(args.num_agents, 1))
            obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

            if isinstance(info, list):
                for item in info:
                    if isinstance(item, dict) and "score" in item:
                        all_scores.append(item["score"])
                        all_collisions.append(item["collision_rate"])
                        all_offroads.append(item["offroad_rate"])
                        all_completions.append(item["completion_rate"])
                        all_returns.append(item["episode_return"])
            elif isinstance(info, dict) and "score" in info:
                all_scores.append(float(np.mean(info["score"])))
                all_collisions.append(float(np.mean(info["collision_rate"])))
                all_offroads.append(float(np.mean(info["offroad_rate"])))
                all_completions.append(float(np.mean(info["completion_rate"])))
                all_returns.append(float(np.mean(info["episode_return"])))

    env_metrics = {
        "eval/score":           np.mean(all_scores),
        "eval/collision_rate":  np.mean(all_collisions),
        "eval/offroad_rate":    np.mean(all_offroads),
        "eval/completion_rate": np.mean(all_completions),
        "eval/mean_return":     np.mean(all_returns),
    }
    for k, v in env_metrics.items():
        print(f"  {k.split('/')[-1]:<20}: {v:.4f}")

    log_metrics(wandb_run, env_metrics, step=args.total_steps)

    env_csv = Path(args.save_dir) / "eval_env_metrics.csv"
    with open(env_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score", "collision_rate", "offroad_rate",
                    "completion_rate", "episode_return"])
        for s, c, o, cp, r in zip(all_scores, all_collisions,
                                   all_offroads, all_completions, all_returns):
            w.writerow([s, c, o, cp, r])
    print(f"  Saved -> {env_csv}")

    # --- WOSAC metrics ---
    print(f"\nWOSAC realism metrics ({args.wosac_rollouts} rollouts, "
          f"up to {args.wosac_max_batches} batches)...")

    try:
        import pandas as pd
        from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator
        from torch.distributions import Categorical as Cat

        wosac_config = {
            "eval": {
                "wosac_init_steps":   0,
                "wosac_num_rollouts": args.wosac_rollouts,
            },
            "train": {"device": str(device)},
        }
        evaluator        = WOSACEvaluator(wosac_config)
        all_results      = []
        unique_scenarios = set()
        B = args.num_agents
        R = args.wosac_rollouts

        for batch in range(args.wosac_max_batches):
            env.resample_maps()
            env.reset()
            gt          = env.get_ground_truth_trajectories()
            agent_state = env.get_global_agent_state()
            road_edges  = env.get_road_edge_polylines()

            sim = {k: np.zeros((B, R, 91), dtype=np.float32)
                   for k in ["x", "y", "z", "heading"]}
            sim["id"] = np.zeros((B, R, 91), dtype=np.int32)

            for r in range(R):
                obs_np, _ = env.reset()
                obs   = torch.tensor(obs_np, dtype=torch.float32, device=device)
                state = policy.initial_state(B, device)
                for t in range(91):
                    ag = env.get_global_agent_state()
                    for k in ["x", "y", "z", "heading"]:
                        sim[k][:, r, t] = ag[k]
                    sim["id"][:, r, t] = ag["id"]
                    with torch.no_grad():
                        logits, _, state, _ = policy(obs, state)
                    action = Cat(logits=logits).sample()
                    obs_np, _, _, _, _ = env.step(
                        action.cpu().numpy().reshape(B, 1))
                    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

            try:
                df  = evaluator.compute_metrics(
                    gt, sim, agent_state, road_edges, aggregate_results=False)
                new = set(df.index.tolist()) - unique_scenarios
                if new:
                    all_results.append(df[df.index.isin(new)])
                    unique_scenarios.update(new)
            except Exception:
                pass

            if (batch + 1) % 10 == 0:
                score = pd.concat(all_results)["realism_meta_score"].mean() \
                        if all_results else 0.0
                print(f"  Batch {batch+1}/{args.wosac_max_batches} | "
                      f"scenarios: {len(unique_scenarios)} | realism: {score:.4f}")

        if all_results:
            combined = pd.concat(all_results)
            agg      = combined.mean()

            wosac_metrics = {
                "eval/wosac_realism_score":    agg["realism_meta_score"],
                "eval/wosac_kinematic":        agg["kinematic_metrics"],
                "eval/wosac_interactive":      agg["interactive_metrics"],
                "eval/wosac_map_based":        agg["map_based_metrics"],
                "eval/wosac_min_ade":          agg["min_ade"],
                "eval/wosac_offroad":          agg["likelihood_offroad_indication"],
                "eval/wosac_collision":        agg["likelihood_collision_indication"],
                "eval/wosac_ttc":              agg["likelihood_time_to_collision"],
                "eval/wosac_linear_speed":     agg["likelihood_linear_speed"],
                "eval/wosac_angular_speed":    agg["likelihood_angular_speed"],
                "eval/wosac_scenarios":        len(combined),
            }

            print(f"\n  Scenarios evaluated       : {len(combined)}")
            print(f"  Realism meta-score        : {agg['realism_meta_score']:.4f}")
            print(f"  Kinematic metrics         : {agg['kinematic_metrics']:.4f}")
            print(f"  Interactive metrics       : {agg['interactive_metrics']:.4f}")
            print(f"  Map-based metrics         : {agg['map_based_metrics']:.4f}")
            print(f"  minADE (m)                : {agg['min_ade']:.4f}")
            print(f"  likelihood_offroad        : {agg['likelihood_offroad_indication']:.4f}")
            print(f"  likelihood_collision      : {agg['likelihood_collision_indication']:.4f}")
            print(f"  likelihood_ttc            : {agg['likelihood_time_to_collision']:.4f}")
            print(f"  likelihood_linear_speed   : {agg['likelihood_linear_speed']:.4f}")
            print(f"  likelihood_angular_speed  : {agg['likelihood_angular_speed']:.4f}")

            log_metrics(wandb_run, wosac_metrics, step=args.total_steps)

            wosac_csv = Path(args.save_dir) / "eval_wosac_metrics.csv"
            combined.to_csv(wosac_csv)
            print(f"  Saved -> {wosac_csv}")

    except Exception as e:
        print(f"  WOSAC evaluation failed: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device — auto-fallback to CPU if CUDA unavailable
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[ROMA] WARNING: CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"[ROMA] Device        : {device}")
    print(f"[ROMA] role_dim      : {args.role_dim}")
    print(f"[ROMA] num_maps      : {args.num_maps}")
    print(f"[ROMA] num_agents    : {args.num_agents}")
    print(f"[ROMA] total_steps   : {args.total_steps:,}")
    print(f"[ROMA] mi_weight     : {args.mi_weight}")
    print(f"[ROMA] div_weight    : {args.div_weight}")

    # Init wandb (optional)
    wandb_run = init_wandb(args)

    from pufferlib.ocean.drive.drive import Drive
    env = Drive(
        num_maps       = args.num_maps,
        num_agents     = args.num_agents,
        map_dir        = args.data_dir,
        episode_length = 91,
    )

    # Auto-detect obs_dim
    obs_probe, _ = env.reset()
    obs_dim      = obs_probe.shape[-1]
    action_dim   = 91
    print(f"[ROMA] obs_dim       : {obs_dim} (auto-detected)")

    # Build policy using structured encoders from roma/policy.py
    policy = RomaPolicy(
        obs_dim        = obs_dim,
        action_dim     = action_dim,
        role_dim       = args.role_dim,
        role_hidden    = args.role_hidden,
        policy_hidden  = args.policy_hidden,
        var_floor      = args.var_floor,
        obs_window_len = 8,
    ).to(device)

    aux_loss_fn = RomaAuxLoss(
        role_dim   = args.role_dim,
        obs_dim    = obs_dim,
        mi_weight  = args.mi_weight,
        div_weight = args.div_weight,
    ).to(device)

    optimizer = Adam(
        list(policy.parameters()) + list(aux_loss_fn.parameters()),
        lr=args.lr,
    )
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # CSV logger — always runs regardless of wandb
    log_csv = Path(args.save_dir) / "training_log.csv"
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "step", "sps", "policy_loss", "value_loss",
            "mi_loss", "div_loss", "score", "mean_return"
        ])
    print(f"[ROMA] CSV log       : {log_csv}")

    B           = args.num_agents
    global_step = 0
    next_save   = args.save_interval
    ep_scores   = deque(maxlen=100)
    ep_returns  = deque(maxlen=100)
    start_time  = time.time()

    N        = args.rollout_steps * B
    b_obs    = torch.zeros(N, obs_dim,       device=device)
    b_act    = torch.zeros(N, dtype=torch.long, device=device)
    b_lp     = torch.zeros(N,                device=device)
    b_rew    = torch.zeros(N,                device=device)
    b_don    = torch.zeros(N,                device=device)
    b_val    = torch.zeros(N,                device=device)
    b_rz     = torch.zeros(N, args.role_dim, device=device)
    b_rmean  = torch.zeros(N, args.role_dim, device=device)
    b_rlv    = torch.zeros(N, args.role_dim, device=device)
    b_obswin = torch.zeros(N, 8, obs_dim,    device=device)

    state = policy.initial_state(B, device)
    obs   = torch.tensor(obs_probe, dtype=torch.float32, device=device)

    print(f"[ROMA] Training for {args.total_steps:,} steps ...")

    while global_step < args.total_steps:

        # ---- Rollout ----
        policy.eval()
        ptr = 0
        with torch.no_grad():
            for _ in range(args.rollout_steps):
                logits, value, state, role_info = policy(obs, state)
                dist    = Categorical(logits=logits)
                action  = dist.sample()
                logprob = dist.log_prob(action)

                actions_np = action.cpu().numpy().reshape(B, 1)
                obs_np, rew_np, term_np, trunc_np, info = env.step(actions_np)

                obs  = torch.tensor(obs_np,  dtype=torch.float32, device=device)
                rew  = torch.tensor(rew_np,  dtype=torch.float32, device=device)
                done = torch.tensor((term_np | trunc_np),
                                    dtype=torch.float32, device=device)

                b_obs   [ptr:ptr+B] = obs
                b_act   [ptr:ptr+B] = action
                b_lp    [ptr:ptr+B] = logprob
                b_rew   [ptr:ptr+B] = rew
                b_don   [ptr:ptr+B] = done
                b_val   [ptr:ptr+B] = value.squeeze(-1)
                b_rz    [ptr:ptr+B] = role_info["role_z"]
                b_rmean [ptr:ptr+B] = role_info["role_mean"]
                b_rlv   [ptr:ptr+B] = role_info["role_log_var"]
                b_obswin[ptr:ptr+B] = role_info["obs_window"]
                ptr += B
                global_step += B

                rm = done.bool()
                if rm.any():
                    state = (
                        torch.where(rm.unsqueeze(-1),
                                    torch.zeros_like(state[0]), state[0]),
                        torch.where(rm.unsqueeze(-1),
                                    torch.zeros_like(state[1]), state[1]),
                        torch.where(rm.unsqueeze(-1).unsqueeze(-1),
                                    torch.zeros_like(state[2]), state[2]),
                    )

                if isinstance(info, list):
                    for item in info:
                        if isinstance(item, dict):
                            if "score" in item:
                                ep_scores.append(float(np.mean(item["score"])))
                            if "episode_return" in item:
                                ep_returns.append(
                                    float(np.mean(item["episode_return"])))
                elif isinstance(info, dict):
                    if "score" in info:
                        ep_scores.append(float(np.mean(info["score"])))
                    if "episode_return" in info:
                        ep_returns.append(float(np.mean(info["episode_return"])))

        # ---- GAE ----
        with torch.no_grad():
            _, last_val, _, _ = policy(obs, state)
        adv     = compute_gae(b_rew[:ptr], b_val[:ptr], b_don[:ptr],
                               last_val.squeeze(-1).mean(), args.gamma, args.gae_lambda)
        returns = adv + b_val[:ptr]
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---- PPO update ----
        policy.train()
        aux_loss_fn.train()
        idx     = torch.randperm(ptr, device=device)
        mb_size = max(1, ptr // args.num_minibatch)

        pl  = vl  = torch.tensor(0.0)
        aux = {"mi_loss": torch.tensor(0.0), "div_loss": torch.tensor(0.0)}

        for _ in range(args.ppo_epochs):
            for start in range(0, ptr, mb_size):
                mb       = idx[start:start + mb_size]
                mb_state = policy.initial_state(mb.size(0), device)
                logits, value, _, role_info = policy(b_obs[mb], mb_state)

                dist        = Categorical(logits=logits)
                new_logprob = dist.log_prob(b_act[mb])
                entropy     = dist.entropy()

                ratio = (new_logprob - b_lp[mb]).exp()
                s1 = ratio * adv[mb]
                s2 = ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef) * adv[mb]
                pl  = -torch.min(s1, s2).mean()
                vl  = F.mse_loss(value.squeeze(-1), returns[mb])
                aux = aux_loss_fn(b_rz[mb], b_rmean[mb], b_rlv[mb], b_obswin[mb])

                loss = (pl
                        + args.vf_coef * vl
                        - args.ent_coef * entropy.mean()
                        + aux["aux_loss"])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

        # ---- Logging ----
        if global_step % args.log_interval < B * args.rollout_steps:
            sps   = global_step / (time.time() - start_time)
            score = np.mean(ep_scores)  if ep_scores  else 0.0
            ret   = np.mean(ep_returns) if ep_returns else 0.0

            print(f"[ROMA step {global_step:>12,}] "
                  f"sps={sps:>7,.0f}  "
                  f"policy_loss={pl.item():.4f}  "
                  f"value_loss={vl.item():.4f}  "
                  f"mi_loss={aux['mi_loss'].item():.4f}  "
                  f"div_loss={aux['div_loss'].item():.4f}  "
                  f"score={score:.3f}  return={ret:.3f}")

            # CSV log
            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    global_step, round(sps, 1),
                    round(pl.item(), 6), round(vl.item(), 6),
                    round(aux["mi_loss"].item(), 6),
                    round(aux["div_loss"].item(), 6),
                    round(score, 4), round(ret, 4),
                ])

            # Wandb log
            log_metrics(wandb_run, {
                "train/policy_loss": pl.item(),
                "train/value_loss":  vl.item(),
                "train/mi_loss":     aux["mi_loss"].item(),
                "train/div_loss":    aux["div_loss"].item(),
                "train/score":       score,
                "train/mean_return": ret,
                "train/sps":         sps,
            }, step=global_step)

        # ---- Checkpoint ----
        if global_step >= next_save:
            ckpt = Path(args.save_dir) / \
                   f"roma_dim{args.role_dim}_step{global_step}.pt"
            torch.save({
                "global_step"   : global_step,
                "policy_state"  : policy.state_dict(),
                "aux_loss_state": aux_loss_fn.state_dict(),
                "args"          : vars(args),
            }, ckpt)
            print(f"[ROMA] Saved -> {ckpt}")
            next_save += args.save_interval

    # ---- Final save ----
    final = Path(args.save_dir) / f"roma_dim{args.role_dim}_final.pt"
    torch.save({
        "global_step" : global_step,
        "policy_state": policy.state_dict(),
        "args"        : vars(args),
    }, final)
    print(f"[ROMA] Training complete. Saved -> {final}")

    # ---- Post-training evaluation ----
    if args.run_eval:
        run_evaluation(args, policy, device, wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    train(parse_args())
