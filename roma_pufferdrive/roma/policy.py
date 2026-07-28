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
                 policy_hidden=128, var_floor=1e-4, obs_window_len=8, ego_dim=7,
                 role_partner_dim=None, role_road_dim=None, role_film=False):
        super().__init__()
        self.obs_dim        = obs_dim
        self.action_dim     = action_dim
        self.role_dim       = role_dim
        self.role_hidden    = role_hidden
        self.policy_hidden  = policy_hidden
        self.obs_window_len = obs_window_len
        self.ego_dim        = ego_dim

        self.ego_enc     = EgoEncoder(ego_dim, out_dim=32)
        self.partner_enc = PartnerEncoder(self.PARTNER_FEAT, out_dim=32, max_partners=self.MAX_PARTNERS)
        self.road_enc    = RoadEncoder(self.ROAD_FEAT, out_dim=64, max_roads=self.MAX_ROADS)
        env_embed_dim    = 32 + 32 + 64
        self.env_embed_dim = env_embed_dim

        # --- The role encoder's own view of the three streams ---------------
        # The POLICY GRU always receives the full 128-dim env embedding, so
        # resizing here never costs the policy the road detail it needs to stay
        # on the road. Only the role encoder's input is re-weighted.
        #
        # Why this exists: road is 64 of the 128 dims the role encoder reads,
        # and road geometry is a near-deterministic function of the scene, so
        # half the role's input bandwidth is a channel for "which map is this".
        # That is the mechanism behind the scene-determined role the per-cluster
        # training pivot was working around. Shrinking road here keeps the role
        # road-AWARE (an agent should still be able to hold back on a curve)
        # without letting the map dominate what the role is.
        #
        # The projections are plain Linear on purpose: the restriction that
        # matters is one of rank/capacity, not of nonlinearity, and the role
        # encoder's own fc_obs supplies the ReLU immediately after.
        #
        # Defaults (None) reproduce the original layout exactly and build no
        # extra modules, so existing checkpoints load unchanged.
        e_out = self.ego_enc.out_dim
        p_out = self.partner_enc.out_dim
        r_out = self.road_enc.out_dim
        self.role_partner_dim = p_out if role_partner_dim is None else role_partner_dim
        self.role_road_dim    = r_out if role_road_dim    is None else role_road_dim
        if self.role_road_dim < 0 or self.role_partner_dim <= 0:
            raise ValueError("role_partner_dim must be > 0 and role_road_dim >= 0 "
                             f"(got {self.role_partner_dim}, {self.role_road_dim})")

        self.role_partner_proj = (nn.Linear(p_out, self.role_partner_dim)
                                  if self.role_partner_dim != p_out else None)
        self.role_road_proj = (nn.Linear(r_out, self.role_road_dim)
                               if self.role_road_dim > 0 and self.role_road_dim != r_out
                               else None)
        self.role_in_dim = e_out + self.role_partner_dim + self.role_road_dim

        # role_dim == 0 -> NO-ROLE ablation (ported from baseline_role_0_dim so
        # the baseline branch can LOAD dim-0 checkpoints, e.g. for side-by-side
        # renders): the role encoder is removed and the policy GRU sees only
        # the env embedding. role_dim > 0 behavior is unchanged.
        self.use_role = role_dim > 0
        if self.use_role:
            self.role_encoder = RoleEncoder(self.role_in_dim, role_dim, role_hidden, var_floor)
        else:
            self.role_encoder = None
        # --- FiLM: let the role MODULATE the env features, not just offset ---
        # Concatenation makes the role an additive bias in the GRU's first
        # layer: h = W_env@e + W_z@z. The role can shift the pre-activation and
        # nothing more, and it is 4 inputs against 128 -- structurally
        # outnumbered 32:1 even at equal per-dimension weight. FiLM instead has
        # the role emit a per-feature scale and shift,
        #     e' = e * (1 + gamma(z)) + beta(z),
        # so it controls HOW every env feature is used. The concatenation is
        # KEPT as well, so the GRU input width is unchanged and the only new
        # parameters are this one Linear.
        #
        # Zero-init is load-bearing: gamma = beta = 0 at step 0 makes e' == e
        # exactly, so a FiLM run starts from the identical function a non-FiLM
        # run starts from and cannot destabilise early training.
        #
        # role_film=False builds NO module, so the state dict is unchanged and
        # every existing checkpoint still loads.
        self.role_film = bool(role_film) and self.use_role
        if self.role_film:
            self.film = nn.Linear(role_dim, 2 * env_embed_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.policy_gru   = nn.GRUCell(env_embed_dim + role_dim, policy_hidden)
        self.actor        = nn.Linear(policy_hidden, action_dim)
        self.critic       = nn.Linear(policy_hidden, 1)

    def initial_state(self, batch_size, device):
        # role_h keeps its shape even in the no-role case (never read then)
        role_h   = torch.zeros(batch_size, self.role_hidden, device=device)
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

    def _env_parts(self, obs):
        ego, partners, roads = self._split_obs(obs)
        return self.ego_enc(ego), self.partner_enc(partners), self.road_enc(roads)

    def _env_embed(self, obs):
        # Concat order [ego | partner | road] is load-bearing: the MI loss
        # slices this window by prefix to drop road from its target
        # (aux_losses.RomaAuxLoss.mi_emb_dim). Do not reorder.
        return torch.cat(self._env_parts(obs), dim=-1)

    def _role_input(self, e, p, r):
        parts = [e, p if self.role_partner_proj is None else self.role_partner_proj(p)]
        if self.role_road_dim > 0:
            parts.append(r if self.role_road_proj is None else self.role_road_proj(r))
        return torch.cat(parts, dim=-1)

    def forward(self, obs, state, forced_role=None):
        role_h, policy_h, emb_win = state
        e, p, r  = self._env_parts(obs)
        env_emb  = torch.cat([e, p, r], dim=-1)
        if self.use_role:
            role_in = self._role_input(e, p, r)
            role_z, role_mean, role_log_var, new_role_h = self.role_encoder(role_in, role_h)
            if forced_role is not None:
                role_z = forced_role
            # FiLM runs on the role the policy ACTUALLY acts on, so a forced
            # role is modulated too -- otherwise the sweep would bypass the
            # very mechanism it is meant to exercise.
            if self.role_film:
                gamma, beta = self.film(role_z).chunk(2, dim=-1)
                env_for_policy = env_emb * (1.0 + gamma) + beta
            else:
                env_for_policy = env_emb
            policy_input = torch.cat([env_for_policy, role_z], dim=-1)
        else:
            # no-role ablation: policy sees only the env embedding
            empty        = env_emb.new_zeros(env_emb.shape[0], 0)
            role_z       = role_mean = role_log_var = empty
            new_role_h   = role_h
            policy_input = env_emb
        new_policy_h = self.policy_gru(policy_input, policy_h)
        logits   = self.actor(new_policy_h)
        value    = self.critic(new_policy_h)
        # Slide the window with the current env embedding (detached — the
        # window is an MI-loss target, gradients should not flow through it).
        # NOTE: env_emb here is the UNMODULATED embedding on purpose. Feeding
        # the FiLM-modulated one would make the MI target a function of the
        # role, so forcing a role would move its own target and the loss could
        # be minimised by warping the target instead of by encoding behaviour.
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
