"""
roma/policy_flat.py
===================
ROMA policy with a flat MLP observation encoder.

Instead of the structured approach (EgoEncoder + PartnerEncoder with attention
+ RoadEncoder with attention), this version feeds the entire 1120-dim
observation through a single MLP and compresses it to a 64-dim embedding.

Used to compare against the structured encoder in roma/policy.py.

Architecture:
    obs (1120 dims)
        → FlatEncoder: Linear(1120→256)→ReLU→Linear(256→64)→ReLU
        → env_embed (64 dims)
        → RoleEncoder (GRU): env_embed → role_z (role_dim dims)
        → PolicyGRU: [env_embed + role_z] → hidden (policy_hidden dims)
        → Actor:  hidden → 91 action logits
        → Critic: hidden → 1 value
"""

import torch
import torch.nn as nn
from roma_pufferdrive.roma.role_encoder import RoleEncoder


class FlatEncoder(nn.Module):
    """Single MLP that compresses full observation to a fixed embedding."""
    def __init__(self, obs_dim=1120, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, embed_dim), nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs[:, :1120])


class RomaPolicyFlat(nn.Module):
    """ROMA policy using a flat MLP encoder instead of structured attention encoders."""

    OBS_DIM    = 1120
    ACTION_DIM = 91

    def __init__(self, obs_dim=1121, action_dim=91, role_dim=8, role_hidden=64,
                 policy_hidden=128, var_floor=1e-4, obs_window_len=8,
                 embed_dim=64):
        super().__init__()
        self.obs_dim        = obs_dim
        self.action_dim     = action_dim
        self.role_dim       = role_dim
        self.policy_hidden  = policy_hidden
        self.obs_window_len = obs_window_len
        self.embed_dim      = embed_dim
        self.env_embed_dim  = embed_dim  # same attribute name as RomaPolicy

        self.flat_enc     = FlatEncoder(self.OBS_DIM, embed_dim)
        self.role_encoder = RoleEncoder(embed_dim, role_dim, role_hidden, var_floor)
        self.policy_gru   = nn.GRUCell(embed_dim + role_dim, policy_hidden)
        self.actor        = nn.Linear(policy_hidden, action_dim)
        self.critic       = nn.Linear(policy_hidden, 1)

    def initial_state(self, batch_size, device):
        role_h   = torch.zeros(batch_size, self.role_encoder.hidden_dim, device=device)
        policy_h = torch.zeros(batch_size, self.policy_hidden,           device=device)
        # Window of past env embeddings (64-dim) instead of raw obs
        # (1121-dim) — far less memory and a learned behavioural
        # trajectory for the MI loss.
        emb_win  = torch.zeros(batch_size, self.obs_window_len, self.env_embed_dim, device=device)
        return (role_h, policy_h, emb_win)

    def forward(self, obs, state):
        role_h, policy_h, emb_win = state

        env_emb = self.flat_enc(obs)
        role_z, role_mean, role_log_var, new_role_h = self.role_encoder(env_emb, role_h)

        policy_input = torch.cat([env_emb, role_z], dim=-1)
        new_policy_h = self.policy_gru(policy_input, policy_h)

        logits  = self.actor(new_policy_h)
        value   = self.critic(new_policy_h)

        # Slide the window with the current env embedding (detached — the
        # window is an MI-loss target, gradients should not flow through it)
        new_emb_win = torch.cat([emb_win[:, 1:, :], env_emb.detach().unsqueeze(1)], dim=1)
        new_state   = (new_role_h, new_policy_h, new_emb_win)
        role_info   = {
            "role_z"      : role_z,
            "role_mean"   : role_mean,
            "role_log_var": role_log_var,
            "emb_window"  : new_emb_win,
        }
        return logits, value, new_state, role_info

    def get_value(self, obs, state):
        _, value, new_state, _ = self.forward(obs, state)
        return value, new_state
