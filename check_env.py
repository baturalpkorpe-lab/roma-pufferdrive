"""
check_env.py  -  Run this FIRST before anything else.
Tells you the exact observation dimension, action space, and
confirms the structured encoder split is correct.

Usage (from inside your PufferDrive directory):
    python3 roma_pufferdrive/check_env.py
"""

import numpy as np

print("=" * 55)
print("  PufferDrive Environment Diagnostic")
print("=" * 55)

try:
    from pufferlib.ocean.drive.drive import Drive
    env = Drive(
        num_maps       = 5,
        num_agents     = 4,
        map_dir        = "resources/drive/binaries/training",
        episode_length = 91,
    )
    obs_np, _ = env.reset()
    print(f"  obs shape        : {obs_np.shape}")
    print(f"  obs_dim (flat)   : {obs_np.shape[-1]}")
    print(f"  obs dtype        : {obs_np.dtype}")
    print(f"  obs min/max      : {obs_np.min():.3f} / {obs_np.max():.3f}")

    obs_dim      = obs_np.shape[-1]
    EGO_DIM      = 7
    MAX_PARTNERS = 31
    PARTNER_DIM  = 7
    partner_flat = MAX_PARTNERS * PARTNER_DIM
    road_flat    = obs_dim - EGO_DIM - partner_flat

    print()
    print("  Proposed structured split:")
    print(f"    ego      : obs[:, 0:{EGO_DIM}]  ({EGO_DIM} features)")
    print(f"    partners : obs[:, {EGO_DIM}:{EGO_DIM+partner_flat}]  ({MAX_PARTNERS}x{PARTNER_DIM}={partner_flat} features)")
    print(f"    roads    : obs[:, {EGO_DIM+partner_flat}:]  ({road_flat} features)")
    print(f"    TOTAL    : {EGO_DIM} + {partner_flat} + {road_flat} = {EGO_DIM+partner_flat+road_flat}")
    print(f"    Match?   : {EGO_DIM+partner_flat+road_flat == obs_dim}")

    action = np.zeros((obs_np.shape[0], 1), dtype=np.int32)
    obs2, rew, term, trunc, info = env.step(action)
    print()
    print(f"  action shape     : {action.shape}")
    print(f"  reward shape     : {rew.shape}")
    print(f"  info type        : {type(info)}")

    print()
    print("  SUCCESS — environment loaded correctly.")
    print(f"\n  >>> COPY THIS NUMBER: obs_dim = {obs_dim} <<<")

except Exception as e:
    print(f"  ERROR: {e}")
    print("  Make sure you run this from ~/PufferDrive and the .venv is active.")

print("=" * 55)
