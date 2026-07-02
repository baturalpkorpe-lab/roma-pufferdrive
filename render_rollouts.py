"""
render_rollouts.py -- Standalone MP4 render for a trained ROMA checkpoint.

Renders 4 maps x 2 runs (free roles + forced role sweep) -> 8 MP4 videos plus
role heatmap PNGs, reusing run_render_rollouts() from train_roma.
Works on both the baseline and main stacks.

Headless cluster usage (raylib needs a display -> virtual framebuffer):
    xvfb-run -a python render_rollouts.py --checkpoint <ckpt.pt> \
        --data_dir <map binaries dir> --save_dir eval_results
MP4s land in <save_dir>/render_rollouts/.
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir",   type=str, required=True,
                   help="Directory containing map_*.bin binaries")
    p.add_argument("--save_dir",   type=str, default="eval_results",
                   help="Videos go to <save_dir>/render_rollouts/")
    p.add_argument("--role_dim",   type=int, default=8)
    p.add_argument("--device",     type=str, default="cpu",
                   help="cpu is fine: 32 agents x 91 steps x 8 episodes")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    from pufferlib.ocean.drive.drive import Drive
    probe = Drive(num_maps=1, num_agents=32, map_dir=args.data_dir)
    obs_np, _ = probe.reset()
    obs_dim   = obs_np.shape[-1]
    probe.close()
    del probe
    print(f"[render] obs_dim auto-detected: {obs_dim}")

    from eval_roma import load_policy
    policy = load_policy(args.checkpoint, args.role_dim, obs_dim, device)
    print(f"[render] policy loaded: {args.checkpoint}")

    from roma_pufferdrive.train_roma import run_render_rollouts
    run_render_rollouts(args, policy, device, wandb_run=None)


if __name__ == "__main__":
    main()
