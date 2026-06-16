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
        --role_dim 8 --num_maps 10000 --total_steps 5000000000 \
        --num_agents 3072 --device cuda --run_eval \
        --wandb_project roma-pufferdrive --wandb_entity YOUR_WANDB_USERNAME \
        --save_dir roma_pufferdrive/checkpoints/roma_dim8

DELFT without wandb (offline CSV only):
    PYTHONPATH=/path/to/roma_pufferdrive python3 roma_pufferdrive/train_roma.py \
        --role_dim 8 --num_maps 10000 --total_steps 5000000000 \
        --num_agents 3072 --device cuda --run_eval --wandb_offline \
        --save_dir roma_pufferdrive/checkpoints/roma_dim8

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
from roma_pufferdrive.roma.policy     import RomaPolicy
from roma_pufferdrive.roma.aux_losses import RomaAuxLoss


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

def compute_gae(rewards, values, dones, last_vals, gamma=0.99, lam=0.95):
    # rewards, values, dones: (T, B) — time × agents
    # last_vals: (B,) — per-agent bootstrap value at end of rollout
    T   = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_vals)
    for t in reversed(range(T)):
        nv    = last_vals if t == T - 1 else values[t + 1]
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
    p.add_argument("--num_agents",    type=int,   default=3072,
                   help="Total agents across parallel envs. 3072 on GPU "
                        "(~535 parallel C envs, ~6 agents each), 16 on CPU.")
    p.add_argument("--num_maps",      type=int,   default=10000,
                   help="Scenarios loaded. 10000 for full training, 100 for CPU test.")
    p.add_argument("--device",        type=str,   default="cuda",
                   help="cuda or cpu. Auto-falls back to cpu if cuda unavailable.")
    p.add_argument("--reward_vehicle_collision",  type=float, default=-0.5,
                   help="Penalty per step for vehicle collision. drive.ini default: -0.5.")
    p.add_argument("--reward_offroad_collision",  type=float, default=-0.5,
                   help="Penalty per step for going off-road. drive.ini default: -0.5.")
    p.add_argument("--goal_speed",               type=float, default=100.0,
                   help="Max target speed m/s toward goal. 100=uncapped (drive.ini default).")
    p.add_argument("--reward_goal_post_respawn", type=float, default=0.25,
                   help="Reward for reaching goal after respawn. drive.ini default: 0.25.")
    p.add_argument("--goal_target_distance",     type=float, default=30.0,
                   help="Target distance for new goals. drive.ini default: 30.0.")
    p.add_argument("--resample_frequency",       type=int,   default=910,
                   help="Steps between map resamples. drive.ini default: 910 (10 episodes).")
    p.add_argument("--termination_mode",         type=int,   default=1,
                   help="0=terminate all at episode_length, 1=terminate per agent on reset. drive.ini default: 1.")

    # Role
    p.add_argument("--role_dim",      type=int,   default=8,
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
    p.add_argument("--num_minibatch", type=int,   default=12,
                   help="Minibatches per PPO epoch. 12 keeps the minibatch at "
                        "~65k samples with 3072 agents — larger minibatches OOM "
                        "the A100 through the road-attention activations.")
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

    # Periodic WOSAC evaluation during training (lite settings).
    # Inspired by PufferDrive kj/guidance_reward, which runs WOSAC realism
    # eval every eval_interval epochs during training.
    p.add_argument("--wosac_interval",      type=int, default=200_000_000,
                   help="Run a lite WOSAC eval every N training steps. 0 disables.")
    p.add_argument("--wosac_eval_maps",     type=int, default=200,
                   help="Map pool size for the periodic (lite) WOSAC eval.")
    p.add_argument("--wosac_eval_rollouts", type=int, default=8,
                   help="Rollouts per scene for the periodic (lite) WOSAC eval.")
    p.add_argument("--wosac_eval_batches",  type=int, default=20,
                   help="Max scenario batches for the periodic (lite) WOSAC eval.")

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
        num_maps                  = args.wosac_num_maps,
        num_agents                = args.num_agents,
        map_dir                   = args.data_dir,
        episode_length            = 91,
        reward_vehicle_collision  = args.reward_vehicle_collision,
        reward_offroad_collision  = args.reward_offroad_collision,
        goal_speed                = args.goal_speed,
        reward_goal_post_respawn  = args.reward_goal_post_respawn,
        goal_target_distance      = args.goal_target_distance,
        resample_frequency        = args.resample_frequency,
        termination_mode          = args.termination_mode,
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
    run_wosac_eval(args, policy, device, wandb_run,
                   global_step=args.total_steps, env=env, save_csv=True)


def run_wosac_eval(args, policy, device, wandb_run=None, global_step=None,
                   env=None, num_maps=None, rollouts=None, max_batches=None,
                   save_csv=False):
    """
    WOSAC realism evaluation.

    Used in two modes:
      - post-training (from run_evaluation): full settings, pass `env` and
        save_csv=True
      - periodically during training: lite settings (num_maps / rollouts /
        max_batches overrides); creates and closes its own env so the
        training env is untouched
    """
    rollouts    = rollouts    or args.wosac_rollouts
    max_batches = max_batches or args.wosac_max_batches
    num_maps    = num_maps    or args.wosac_num_maps
    step        = global_step if global_step is not None else args.total_steps
    own_env     = env is None

    print(f"\nWOSAC realism metrics ({rollouts} rollouts, "
          f"up to {max_batches} batches, step {step:,})...")

    try:
        import pandas as pd
        from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator
        from torch.distributions import Categorical as Cat

        if own_env:
            from pufferlib.ocean.drive.drive import Drive
            env = Drive(
                num_maps                  = num_maps,
                num_agents                = args.num_agents,
                map_dir                   = args.data_dir,
                episode_length            = 91,
                reward_vehicle_collision  = args.reward_vehicle_collision,
                reward_offroad_collision  = args.reward_offroad_collision,
                goal_speed                = args.goal_speed,
            )
        policy.eval()

        wosac_config = {
            "eval": {
                "wosac_init_steps":   0,
                "wosac_num_rollouts": rollouts,
            },
            "train": {"device": str(device)},
        }
        evaluator        = WOSACEvaluator(wosac_config)
        all_results      = []
        unique_scenarios = set()
        B = args.num_agents
        R = rollouts

        for batch in range(max_batches):
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
                print(f"  Batch {batch+1}/{max_batches} | "
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

            log_metrics(wandb_run, wosac_metrics, step=step)

            if save_csv:
                wosac_csv = Path(args.save_dir) / "eval_wosac_metrics.csv"
                combined.to_csv(wosac_csv)
                print(f"  Saved -> {wosac_csv}")

    except Exception as e:
        print(f"  WOSAC evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Periodic mode owns its env — release it so the training env and
        # GPU memory are unaffected.
        if own_env and env is not None:
            try:
                env.close()
            except Exception:
                pass


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

    if device.type == "cuda":
        # TF32 tensor cores on Ampere+ (A100): up to ~8x faster float32
        # matmuls at negligible precision cost for RL workloads.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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
        num_maps                  = args.num_maps,
        num_agents                = args.num_agents,
        map_dir                   = args.data_dir,
        episode_length            = 91,
        reward_vehicle_collision  = args.reward_vehicle_collision,
        reward_offroad_collision  = args.reward_offroad_collision,
        goal_speed                = args.goal_speed,
        reward_goal_post_respawn  = args.reward_goal_post_respawn,
        goal_target_distance      = args.goal_target_distance,
        resample_frequency        = args.resample_frequency,
        termination_mode          = args.termination_mode,
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
        emb_dim    = policy.env_embed_dim,
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
    next_wosac  = args.wosac_interval if args.wosac_interval > 0 else float("inf")
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
    # Env embedding window (128-dim) instead of raw obs window (1121-dim):
    # ~9x less GPU memory and memory traffic per rollout step.
    b_embwin = torch.zeros(N, 8, policy.env_embed_dim, device=device)
    # Hidden states captured before each forward pass — used during PPO update
    # to avoid restarting GRU from zeros on shuffled minibatches.
    b_role_h   = torch.zeros(N, args.role_hidden,   device=device)
    b_policy_h = torch.zeros(N, args.policy_hidden, device=device)

    state = policy.initial_state(B, device)
    obs   = torch.as_tensor(obs_probe, dtype=torch.float32).to(device)

    # Pinned staging buffers for CPU<->GPU transfers (CUDA only).
    # Pinned memory makes host-to-device copies faster and allows them to
    # run asynchronously (non_blocking=True) instead of stalling the GPU.
    use_pin = device.type == "cuda"
    if use_pin:
        act_pin  = torch.zeros(B, 1,        dtype=torch.int32,   pin_memory=True)
        obs_pin  = torch.zeros(B, obs_dim,  dtype=torch.float32, pin_memory=True)
        rew_pin  = torch.zeros(B,           dtype=torch.float32, pin_memory=True)
        done_pin = torch.zeros(B,           dtype=torch.float32, pin_memory=True)

    print(f"[ROMA] Training for {args.total_steps:,} steps ...")

    while global_step < args.total_steps:

        # ---- Rollout ----
        policy.eval()
        ptr = 0
        with torch.no_grad():
            for _ in range(args.rollout_steps):
                # Store the obs the policy acts on BEFORE stepping the env —
                # otherwise b_obs would hold obs_{t+1} while logprob/value
                # belong to obs_t, corrupting the PPO ratio.
                b_obs[ptr:ptr+B]      = obs
                b_role_h[ptr:ptr+B]   = state[0]
                b_policy_h[ptr:ptr+B] = state[1]
                logits, value, state, role_info = policy(obs, state)
                dist    = Categorical(logits=logits)
                action  = dist.sample()
                logprob = dist.log_prob(action)

                if use_pin:
                    # GPU -> pinned host memory (faster than pageable)
                    act_pin.copy_(action.unsqueeze(-1))
                    actions_np = act_pin.numpy()
                else:
                    actions_np = action.cpu().numpy().reshape(B, 1)
                obs_np, rew_np, term_np, trunc_np, info = env.step(actions_np)

                done_np = (term_np | trunc_np)
                if use_pin:
                    # numpy -> pinned host buffer -> async copy to GPU
                    obs_pin.copy_(torch.from_numpy(obs_np))
                    rew_pin.copy_(torch.from_numpy(rew_np))
                    done_pin.copy_(torch.from_numpy(done_np.astype(np.float32)))
                    obs  = obs_pin.to(device,  non_blocking=True)
                    rew  = rew_pin.to(device,  non_blocking=True)
                    done = done_pin.to(device, non_blocking=True)
                else:
                    obs  = torch.as_tensor(obs_np,  dtype=torch.float32).to(device)
                    rew  = torch.as_tensor(rew_np,  dtype=torch.float32).to(device)
                    done = torch.as_tensor(done_np, dtype=torch.float32).to(device)

                b_act   [ptr:ptr+B] = action
                b_lp    [ptr:ptr+B] = logprob
                b_rew   [ptr:ptr+B] = rew
                b_don   [ptr:ptr+B] = done
                b_val   [ptr:ptr+B] = value.squeeze(-1)
                b_embwin[ptr:ptr+B] = role_info["emb_window"]
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
        T_steps   = args.rollout_steps
        last_vals = last_val.squeeze(-1)                     # (B,) — one per agent
        adv_2d    = compute_gae(
            b_rew[:ptr].reshape(T_steps, B),
            b_val[:ptr].reshape(T_steps, B),
            b_don[:ptr].reshape(T_steps, B),
            last_vals, args.gamma, args.gae_lambda,
        )                                                    # (T, B)
        adv     = adv_2d.reshape(-1)                         # back to (N,)
        returns = adv + b_val[:ptr]
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---- LR annealing ----
        frac = 1.0 - global_step / args.total_steps
        lr_now = args.lr * frac
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

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
                # The emb window slot of the state is unused during the PPO
                # forward (the MI target comes from b_embwin) — a per-batch
                # zeros tensor avoids keeping a full N-sized dummy buffer.
                mb_state = (b_role_h[mb], b_policy_h[mb],
                            torch.zeros(len(mb), policy.obs_window_len,
                                        policy.env_embed_dim, device=device))
                logits, value, _, role_info = policy(b_obs[mb], mb_state)

                dist        = Categorical(logits=logits)
                new_logprob = dist.log_prob(b_act[mb])
                entropy     = dist.entropy()

                ratio = (new_logprob - b_lp[mb]).exp()
                s1 = ratio * adv[mb]
                s2 = ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef) * adv[mb]
                pl  = -torch.min(s1, s2).mean()
                vl  = F.mse_loss(value.squeeze(-1), returns[mb])
                # Use the freshly recomputed role variables (they carry a
                # computation graph) — the rollout-buffer copies were created
                # under no_grad, so the aux losses would otherwise send zero
                # gradient to the role encoder.
                aux = aux_loss_fn(role_info["role_z"], role_info["role_mean"],
                                  role_info["role_log_var"], b_embwin[mb])

                loss = (pl
                        + args.vf_coef * vl
                        - args.ent_coef * entropy.mean()
                        + aux["aux_loss"])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(aux_loss_fn.parameters()),
                    args.max_grad_norm,
                )
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

        # ---- Periodic WOSAC evaluation (lite) ----
        if global_step >= next_wosac:
            print(f"[ROMA] Periodic WOSAC eval at step {global_step:,} ...")
            run_wosac_eval(
                args, policy, device, wandb_run,
                global_step = global_step,
                num_maps    = args.wosac_eval_maps,
                rollouts    = args.wosac_eval_rollouts,
                max_batches = args.wosac_eval_batches,
            )
            next_wosac += args.wosac_interval

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
