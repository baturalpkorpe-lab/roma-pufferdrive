"""
roma/policy.py
==============
Full ROMA policy using structured observation encoders.

Observation layout (flat vector, confirmed by check_env.py):
  obs_dim = 1121 total

  [0   : 7]      — ego vehicle state         (7 features)
  [7   : 224]    — up to 31 partner vehicles  (31 × 7 = 217 features)
  [224 : 1120]   — 128 road geometry points   (128 × 7 = 896 features)
  [1120: 1121]   — 1 padding/flag byte        (dropped before road encoder)

The three sub-encoders (EgoEncoder, PartnerEncoder, RoadEncoder) each
process their slice and produce a fixed-size embedding.  The embeddings
are concatenated → 32+32+64 = 128 dims → fed into the policy GRU
together with the role vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from roma.role_encoder import RoleEncoder


# ---------------------------------------------------------------------------
# Sub-encoders
# ---------------------------------------------------------------------------

class EgoEncoder(nn.Module):
    """Encode the ego vehicle state vector."""
    def __init__(self, ego_dim=7, out_dim=32):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(ego_dim, 32), nn.ReLU(),
            nn.Linear(32, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class PartnerEncoder(nn.Module):
    """
    Encode up to max_partners surrounding vehicles with max-pooling
    so the representation is permutation-invariant.
    """
    def __init__(self, partner_feat=7, out_dim=32, max_partners=31):
        super().__init__()
        self.max_partners = max_partners
        self.partner_feat = partner_feat
        self.out_dim      = out_dim
        self.net = nn.Sequential(
            nn.Linear(partner_feat, 32), nn.ReLU(),
            nn.Linear(32, out_dim),      nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, max_partners * partner_feat)
        B = x.size(0)
        x = x.view(B, self.max_partners, self.partner_feat)
        x = self.net(x)                  # (B, max_partners, out_dim)
        return x.max(dim=1).values       # (B, out_dim)  — permutation invariant


class RoadEncoder(nn.Module):
    """
    Encode nearby road geometry points with max-pooling.
    max_roads is derived from whatever is left in the observation.
    """
    def __init__(self, road_feat=7, out_dim=64, max_roads=232):
        super().__init__()
        self.max_roads = max_roads
        self.road_feat = road_feat
        self.out_dim   = out_dim
        self.net = nn.Sequential(
            nn.Linear(road_feat, 32), nn.ReLU(),
            nn.Linear(32, out_dim),   nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, max_roads * road_feat)
        B = x.size(0)
        x = x.view(B, self.max_roads, self.road_feat)
        x = self.net(x)                  # (B, max_roads, out_dim)
        return x.max(dim=1).values       # (B, out_dim)


# ---------------------------------------------------------------------------
# Full ROMA policy
# ---------------------------------------------------------------------------

class RomaPolicy(nn.Module):
    """
    ROMA policy with structured observation encoders and a recurrent role encoder.

    Args:
        obs_dim        : total flat observation dimension (set from check_env.py output)
        action_dim     : number of discrete actions (91 for PufferDrive)
        role_dim       : dimension of the latent role vector
        role_hidden    : hidden size of the role encoder GRU
        policy_hidden  : hidden size of the policy GRU
        var_floor      : minimum variance for the role distribution
        obs_window_len : number of past observations kept for MI loss
        ego_dim        : number of ego-state features
        partner_feat   : features per surrounding vehicle
        max_partners   : maximum number of surrounding vehicles
        road_feat      : features per road point
    """

    # Confirmed by check_env.py — do not change these.
    EGO_DIM      = 7
    PARTNER_FEAT = 7
    MAX_PARTNERS = 31   # 31 × 7 = 217
    ROAD_FEAT    = 7
    MAX_ROADS    = 128  # 128 × 7 = 896
    # obs[1120] is 1 padding byte — dropped before road encoder

    def __init__(
        self,
        obs_dim        = 1121,   # confirmed by check_env.py
        action_dim     = 91,
        role_dim       = 8,
        role_hidden    = 64,
        policy_hidden  = 128,
        var_floor      = 1e-4,
        obs_window_len = 8,
        ego_dim        = 7,
    ):
        super().__init__()
        self.obs_dim        = obs_dim
        self.action_dim     = action_dim
        self.role_dim       = role_dim
        self.policy_hidden  = policy_hidden
        self.obs_window_len = obs_window_len
        self.ego_dim        = ego_dim

        # Sub-encoders: 32 + 32 + 64 = 128 dims total
        self.ego_enc     = EgoEncoder(ego_dim, out_dim=32)
        self.partner_enc = PartnerEncoder(self.PARTNER_FEAT, out_dim=32,
                                          max_partners=self.MAX_PARTNERS)
        # Road encoder uses 128 points × 7 features — the trailing padding
        # byte (index 1120) is sliced off inside _split_obs before this runs.
        self.road_enc    = RoadEncoder(self.ROAD_FEAT, out_dim=64,
                                       max_roads=self.MAX_ROADS)
        env_embed_dim = 32 + 32 + 64  # = 128

        # Role encoder (GRU-based, produces role_z conditioned on obs history)
        self.role_encoder = RoleEncoder(obs_dim, role_dim, role_hidden, var_floor)

        # Policy GRU: takes [env_embedding || role_z] → hidden → actor/critic
        self.policy_gru = nn.GRUCell(env_embed_dim + role_dim, policy_hidden)
        self.actor  = nn.Linear(policy_hidden, action_dim)
        self.critic = nn.Linear(policy_hidden, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def initial_state(self, batch_size, device):
        """Return zero initial state for a batch of agents."""
        role_h   = torch.zeros(batch_size, self.role_encoder.hidden_dim, device=device)
        policy_h = torch.zeros(batch_size, self.policy_hidden,           device=device)
        obs_win  = torch.zeros(batch_size, self.obs_window_len,
                               self.obs_dim, device=device)
        return (role_h, policy_h, obs_win)

    def _split_obs(self, obs):
        """Split flat observation into ego / partners / roads slices.

        Layout (all indices confirmed by check_env.py):
          [0:7]    ego   (7 features)
          [7:224]  partners (31 × 7 = 217 features)
          [224:1120] roads (128 × 7 = 896 features)
          [1120]   padding — dropped
        """
        ego      = obs[:, :self.EGO_DIM]                              # (B, 7)
        p_end    = self.EGO_DIM + self.MAX_PARTNERS * self.PARTNER_FEAT  # 224
        partners = obs[:, self.EGO_DIM : p_end]                       # (B, 217)
        roads    = obs[:, p_end : p_end + self.MAX_ROADS * self.ROAD_FEAT]  # (B, 896) — drops [1120]
        return ego, partners, roads

    def _env_embed(self, obs):
        """Run structured sub-encoders and concatenate embeddings."""
        ego, partners, roads = self._split_obs(obs)
        e = self.ego_enc(ego)          # (B, 32)
        p = self.partner_enc(partners) # (B, 32)
        r = self.road_enc(roads)       # (B, 64)
        return torch.cat([e, p, r], dim=-1)  # (B, 128)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, obs, state):
        """
        Args:
            obs   : (B, obs_dim) current observation
            state : tuple (role_h, policy_h, obs_win)

        Returns:
            logits    : (B, action_dim)
            value     : (B, 1)
            new_state : updated (role_h, policy_h, obs_win)
            role_info : dict with role_z, role_mean, role_log_var, obs_window
        """
        role_h, policy_h, obs_win = state

        # 1. Role encoder
        role_z, role_mean, role_log_var, new_role_h = self.role_encoder(obs, role_h)

        # 2. Environment encoder (structured)
        env_emb = self._env_embed(obs)   # (B, 128)

        # 3. Policy GRU
        policy_input = torch.cat([env_emb, role_z], dim=-1)  # (B, 128+role_dim)
        new_policy_h = self.policy_gru(policy_input, policy_h)

        # 4. Actor / critic heads
        logits = self.actor(new_policy_h)   # (B, action_dim)
        value  = self.critic(new_policy_h)  # (B, 1)

        # 5. Slide observation window (oldest obs dropped, newest appended)
        new_obs_win = torch.cat([obs_win[:, 1:, :], obs.unsqueeze(1)], dim=1)

        new_state = (new_role_h, new_policy_h, new_obs_win)
        role_info = {
            "role_z"      : role_z,
            "role_mean"   : role_mean,
            "role_log_var": role_log_var,
            "obs_window"  : new_obs_win,
        }
        return logits, value, new_state, role_info

    def get_value(self, obs, state):
        """Convenience wrapper for value-only calls."""
        _, value, new_state, _ = self.forward(obs, state)
        return value, new_state
