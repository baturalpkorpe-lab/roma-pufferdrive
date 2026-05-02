import torch
import torch.nn as nn
import torch.nn.functional as F
from roma.role_encoder import RoleEncoder


class EgoEncoder(nn.Module):
    def __init__(self, ego_dim=7, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ego_dim, 32), nn.ReLU(),
            nn.Linear(32, out_dim), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class PartnerEncoder(nn.Module):
    def __init__(self, partner_feat=7, out_dim=32, max_partners=31):
        super().__init__()
        self.max_partners = max_partners
        self.partner_feat = partner_feat
        self.net = nn.Sequential(
            nn.Linear(partner_feat, 32), nn.ReLU(),
            nn.Linear(32, out_dim), nn.ReLU(),
        )
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.max_partners, self.partner_feat)
        x = self.net(x)
        return x.max(dim=1).values


class RoadEncoder(nn.Module):
    def __init__(self, road_feat=7, out_dim=32, max_roads=232):
        super().__init__()
        self.max_roads = max_roads
        self.road_feat = road_feat
        self.net = nn.Sequential(
            nn.Linear(road_feat, 32), nn.ReLU(),
            nn.Linear(32, out_dim), nn.ReLU(),
        )
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.max_roads, self.road_feat)
        x = self.net(x)
        return x.max(dim=1).values


class RomaPolicy(nn.Module):
    MAX_PARTNERS = 31
    PARTNER_DIM  = 7
    MAX_ROADS    = 232
    ROAD_DIM     = 7

    def __init__(self, obs_dim=1848, action_dim=91, role_dim=8,
                 role_hidden=64, policy_hidden=128, var_floor=1e-4,
                 obs_window_len=8, ego_dim=7):
        super().__init__()
        self.obs_dim        = obs_dim
        self.action_dim     = action_dim
        self.role_dim       = role_dim
        self.policy_hidden  = policy_hidden
        self.obs_window_len = obs_window_len
        self.ego_dim        = ego_dim

        self.ego_enc     = EgoEncoder(ego_dim, 32)
        self.partner_enc = PartnerEncoder(self.PARTNER_DIM, 32, self.MAX_PARTNERS)
        self.road_enc    = RoadEncoder(self.ROAD_DIM, 64, self.MAX_ROADS)

        self.role_encoder = RoleEncoder(obs_dim, role_dim, role_hidden, var_floor)

        self.policy_gru = nn.GRUCell(128 + role_dim, policy_hidden)
        self.actor  = nn.Linear(policy_hidden, action_dim)
        self.critic = nn.Linear(policy_hidden, 1)

    def initial_state(self, batch_size, device):
        role_h   = torch.zeros(batch_size, self.role_encoder.hidden_dim, device=device)
        policy_h = torch.zeros(batch_size, self.policy_hidden, device=device)
        obs_win  = torch.zeros(batch_size, self.obs_window_len, self.obs_dim, device=device)
        return (role_h, policy_h, obs_win)

    def _split_obs(self, obs):
        ego      = obs[:, :self.ego_dim]
        p_start  = self.ego_dim
        p_end    = p_start + self.MAX_PARTNERS * self.PARTNER_DIM
        partners = obs[:, p_start:p_end]
        roads    = obs[:, p_end:]
        return ego, partners, roads

    def _env_embed(self, obs):
        ego, partners, roads = self._split_obs(obs)
        e = self.ego_enc(ego)
        p = self.partner_enc(partners)
        r = self.road_enc(roads)
        return torch.cat([e, p, r], dim=-1)

    def forward(self, obs, state):
        role_h, policy_h, obs_win = state
        role_z, role_mean, role_log_var, new_role_h = self.role_encoder(obs, role_h)
        env_emb      = self._env_embed(obs)
        policy_input = torch.cat([env_emb, role_z], dim=-1)
        new_policy_h = self.policy_gru(policy_input, policy_h)
        logits = self.actor(new_policy_h)
        value  = self.critic(new_policy_h)
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
        _, value, new_state, _ = self.forward(obs, state)
        return value, new_state
