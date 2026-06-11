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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from roma_pufferdrive.roma.role_encoder import RoleEncoder


class EgoEncoder(nn.Module):
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
    def __init__(self, partner_feat=7, out_dim=32, max_partners=31, embed_dim=64):
        super().__init__()
        self.max_partners = max_partners
        self.partner_feat = partner_feat
        self.out_dim      = out_dim
        self.embed_dim    = embed_dim
        self.scale        = embed_dim ** 0.5

        self.input_proj = nn.Sequential(
            nn.Linear(partner_feat, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),    nn.ReLU(),
        )
        self.query   = nn.Parameter(torch.zeros(embed_dim))
        nn.init.xavier_uniform_(self.query.unsqueeze(0))
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        B     = x.size(0)
        x     = x.view(B, self.max_partners, self.partner_feat)
        keys  = self.input_proj(x)
        q     = self.query.unsqueeze(0).unsqueeze(0)
        scores   = (keys * q).sum(dim=-1) / self.scale
        weights  = F.softmax(scores, dim=-1)
        attended = (weights.unsqueeze(-1) * keys).sum(dim=1)
        return self.out_proj(attended)


class RoadEncoder(nn.Module):
    def __init__(self, road_feat=7, out_dim=64, max_roads=128, embed_dim=128):
        super().__init__()
        self.max_roads = max_roads
        self.road_feat = road_feat
        self.out_dim   = out_dim
        self.embed_dim = embed_dim
        self.scale     = embed_dim ** 0.5

        self.input_proj = nn.Sequential(
            nn.Linear(road_feat, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )
        self.query    = nn.Parameter(torch.zeros(embed_dim))
        nn.init.xavier_uniform_(self.query.unsqueeze(0))
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        B     = x.size(0)
        x     = x.view(B, self.max_roads, self.road_feat)
        keys  = self.input_proj(x)
        q     = self.query.unsqueeze(0).unsqueeze(0)
        scores   = (keys * q).sum(dim=-1) / self.scale
        weights  = F.softmax(scores, dim=-1)
        attended = (weights.unsqueeze(-1) * keys).sum(dim=1)
        return self.out_proj(attended)


class RomaPolicy(nn.Module):
    EGO_DIM      = 7
    PARTNER_FEAT = 7
    MAX_PARTNERS = 31
    ROAD_FEAT    = 7
    MAX_ROADS    = 128

    def __init__(self, obs_dim=1121, action_dim=91, role_dim=8, role_hidden=64,
                 policy_hidden=128, var_floor=1e-4, obs_window_len=8, ego_dim=7):
        super().__init__()
        self.obs_dim        = obs_dim
        self.action_dim     = action_dim
        self.role_dim       = role_dim
        self.policy_hidden  = policy_hidden
        self.obs_window_len = obs_window_len
        self.ego_dim        = ego_dim

        self.ego_enc     = EgoEncoder(ego_dim, out_dim=32)
        self.partner_enc = PartnerEncoder(self.PARTNER_FEAT, out_dim=32, max_partners=self.MAX_PARTNERS)
        self.road_enc    = RoadEncoder(self.ROAD_FEAT, out_dim=64, max_roads=self.MAX_ROADS)
        env_embed_dim    = 32 + 32 + 64
        self.env_embed_dim = env_embed_dim

        self.role_encoder = RoleEncoder(env_embed_dim, role_dim, role_hidden, var_floor)
        self.policy_gru   = nn.GRUCell(env_embed_dim + role_dim, policy_hidden)
        self.actor        = nn.Linear(policy_hidden, action_dim)
        self.critic       = nn.Linear(policy_hidden, 1)

    def initial_state(self, batch_size, device):
        role_h   = torch.zeros(batch_size, self.role_encoder.hidden_dim, device=device)
        policy_h = torch.zeros(batch_size, self.policy_hidden, device=device)
        # Window of past env embeddings (128-dim each) instead of raw obs
        # (1121-dim) — 9x less memory and a structured, learned behavioural
        # trajectory for the MI loss.
        emb_win  = torch.zeros(batch_size, self.obs_window_len, self.env_embed_dim, device=device)
        return (role_h, policy_h, emb_win)

    def _split_obs(self, obs):
        ego      = obs[:, :self.EGO_DIM]
        p_end    = self.EGO_DIM + self.MAX_PARTNERS * self.PARTNER_FEAT
        partners = obs[:, self.EGO_DIM:p_end]
        roads    = obs[:, p_end:p_end + self.MAX_ROADS * self.ROAD_FEAT]
        return ego, partners, roads

    def _env_embed(self, obs):
        ego, partners, roads = self._split_obs(obs)
        e = self.ego_enc(ego)
        p = self.partner_enc(partners)
        r = self.road_enc(roads)
        return torch.cat([e, p, r], dim=-1)

    def forward(self, obs, state):
        role_h, policy_h, emb_win = state
        env_emb  = self._env_embed(obs)
        role_z, role_mean, role_log_var, new_role_h = self.role_encoder(env_emb, role_h)
        policy_input = torch.cat([env_emb, role_z], dim=-1)
        new_policy_h = self.policy_gru(policy_input, policy_h)
        logits   = self.actor(new_policy_h)
        value    = self.critic(new_policy_h)
        # Slide the window with the current env embedding (detached — the
        # window is an MI-loss target, gradients should not flow through it)
        new_emb_win = torch.cat([emb_win[:, 1:, :], env_emb.detach().unsqueeze(1)], dim=1)
        new_state = (new_role_h, new_policy_h, new_emb_win)
        role_info = {
            "role_z"      : role_z,
            "role_mean"   : role_mean,
            "role_log_var": role_log_var,
            "emb_window"  : new_emb_win,
        }
        return logits, value, new_state, role_info

    def get_value(self, obs, state):
        _, value, new_state, _ = self.forward(obs, state)
        return value, new_state
