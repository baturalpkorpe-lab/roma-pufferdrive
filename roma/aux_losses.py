"""
roma/aux_losses.py
==================
ROMA auxiliary losses: MI loss + diversity loss.

MI Loss (Mutual Information):
    Forces the role vector to encode meaningful behavioural information.
    The role encoder predicts a behaviour summary extracted from the last
    8 observations. If the role contains no behavioural information, this
    loss is high.

Diversity Loss (Simplified):
    Pushes agents to have different role vectors from each other.
    We use pairwise cosine similarity between role means — agents with
    identical roles have similarity=1, maximally different agents have
    similarity=-1. We minimise average pairwise similarity.

    This is a simplified approximation of the original ROMA diversity loss
    (Wang et al., ICML 2020), which uses a trainable dissimilarity model
    d_φ and a trajectory encoder q_ξ to measure pairwise trajectory
    dissimilarity. The full implementation is planned for Phase 5.

    Advantages of this approach over KL-based diversity:
      - Naturally bounded in [-1, 1] — no clipping needed
      - Always interpretable:
            div_loss = +1.0  → all agents have identical roles (collapse)
            div_loss =  0.0  → agent roles are uncorrelated
            div_loss = -1.0  → agents are maximally diverse
      - No arbitrary hyperparameter for clip threshold
      - No population mean dependency that causes gradient instability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviourExtractor(nn.Module):
    """
    Encodes the last `window` observations into a behaviour summary vector.
    This is the target that the MI loss tries to predict from the role vector.
    """
    def __init__(self, obs_dim, behaviour_dim=32, window=8):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(
            nn.Linear(obs_dim * window, 128),
            nn.ReLU(),
            nn.Linear(128, behaviour_dim),
        )

    def forward(self, obs_window):
        B = obs_window.size(0)
        flat = obs_window.reshape(B, -1)
        return self.net(flat)


class MIDecoder(nn.Module):
    """
    Predicts the behaviour summary from the role vector.
    Low MSE loss = the role encodes real behavioural information.
    """
    def __init__(self, role_dim, behaviour_dim=32, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(role_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, behaviour_dim),
        )

    def forward(self, role_z):
        return self.net(role_z)


class RomaAuxLoss(nn.Module):
    """
    Combined ROMA auxiliary loss: MI loss + diversity loss.

    Args:
        role_dim      : dimension of the role vector
        obs_dim       : flat observation dimension
        behaviour_dim : dimension of the behaviour summary vector
        hidden_dim    : hidden size of the MI decoder
        window        : number of past observations for behaviour extraction
        mi_weight     : weight on the MI loss (default 1.0, from ROMA paper)
        div_weight    : weight on the diversity loss (default 0.1)
    """
    def __init__(self, role_dim, obs_dim, behaviour_dim=32, hidden_dim=64,
                 window=8, mi_weight=1.0, div_weight=0.1):
        super().__init__()
        self.mi_weight  = mi_weight
        self.div_weight = div_weight
        self.behaviour_extractor = BehaviourExtractor(obs_dim, behaviour_dim, window)
        self.mi_decoder          = MIDecoder(role_dim, behaviour_dim, hidden_dim)

    def mi_loss(self, role_z, obs_window):
        """
        MSE between predicted and actual behaviour summary.
        Forces role vector to encode real behavioural information.
        Range: [0, ∞) — lower is better.
        """
        behaviour_target = self.behaviour_extractor(obs_window).detach()
        behaviour_pred   = self.mi_decoder(role_z)
        return F.mse_loss(behaviour_pred, behaviour_target)

    def diversity_loss(self, role_mean):
        """
        Average pairwise cosine similarity between agent role means.

        We normalise each role mean to unit length, then compute the
        full pairwise cosine similarity matrix. We average the
        off-diagonal entries (excluding self-similarity).

        Range: [-1, +1]
            +1 = all agents have identical roles (collapse — bad)
             0 = roles are uncorrelated
            -1 = agents are maximally diverse (ideal)

        Minimising this loss pushes agents apart in role space.
        No clipping needed — the range is naturally bounded.
        """
        B = role_mean.size(0)
        if B < 2:
            return torch.tensor(0.0, device=role_mean.device)

        # Normalise to unit sphere
        normed = F.normalize(role_mean, dim=-1)   # (B, role_dim)

        # Full pairwise cosine similarity matrix
        sim_matrix = normed @ normed.T             # (B, B)  values in [-1, 1]

        # Mask out diagonal (self-similarity = 1, not informative)
        mask = 1.0 - torch.eye(B, device=role_mean.device)

        # Average off-diagonal similarity
        avg_sim = (sim_matrix * mask).sum() / mask.sum()

        return avg_sim

    def forward(self, role_z, role_mean, role_log_var, obs_window):
        """
        Args:
            role_z       : sampled role vector (B, role_dim)
            role_mean    : role distribution mean (B, role_dim)
            role_log_var : role distribution log variance (B, role_dim)
            obs_window   : last 8 observations (B, 8, obs_dim)

        Returns:
            dict with mi_loss, div_loss, aux_loss
        """
        l_mi  = self.mi_loss(role_z, obs_window)
        l_div = self.diversity_loss(role_mean)
        aux   = self.mi_weight * l_mi + self.div_weight * l_div

        return {"mi_loss": l_mi, "div_loss": l_div, "aux_loss": aux}
