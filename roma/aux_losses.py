"""
roma/aux_losses.py
==================
ROMA auxiliary losses: MI loss + diversity loss.

MI Loss (Mutual Information):
    Forces the role vector to encode meaningful behavioural information.
    The role encoder predicts a behaviour summary extracted from the last
    8 observations. If the role contains no behavioural information, this
    loss is high.

    This is a simplified approximation of the full ROMA MI loss
    (Wang et al., ICML 2020), which uses a GRU trajectory encoder q_ξ
    to estimate I(ρ; τ | o). Our BehaviourExtractor + MIDecoder serves
    the same purpose with less complexity.

    Future improvement (Phase 5):
    R3DM (Goel et al., ICML 2025) shows that linking roles to FUTURE
    expected behaviour via a learned dynamics model significantly improves
    role differentiation. This would replace the current past-observation
    based approach.

Diversity Loss:
    Pushes agents to have different role vectors from each other.

    Current implementation uses two approaches depending on role_dim:

    role_dim=1 — Normalised negative variance:
        Cosine similarity between scalars is always ±1 with no useful
        gradient. Instead we maximise variance of scalar roles across
        agents, normalised by mean absolute role value for scale invariance.
        Range: (-∞, 0], lower is more diverse.

    role_dim≥2 — Pairwise cosine similarity:
        Average pairwise cosine similarity between role mean vectors.
        Naturally bounded in [-1, +1]:
            +1 = all agents identical (collapse)
             0 = roles uncorrelated
            -1 = agents maximally diverse
        No clipping needed.

    This is a simplified approximation of the original ROMA diversity loss
    (Wang et al., ICML 2020), which uses a trainable dissimilarity model
    d_φ and trajectory encoder q_ξ. Full implementation planned for Phase 5.

    Literature context (searched May 2026):
    - DiCo (Bettini et al., ICML 2024): Controls diversity to a desired
      target value by dynamically scaling heterogeneous policy components.
      More principled than our loss — eliminates need for div_weight tuning.
      Reference: https://arxiv.org/html/2405.15054v1
    - R3DM (Goel et al., ICML 2025): Diversity via contrastive learning on
      past trajectories + dynamics model for future behaviour prediction.
      Outperforms ROMA on SMAC by up to 20% win rate improvement.
      Reference: https://arxiv.org/pdf/2505.24265
    - Trajectory prediction diversity (2024): Diversity loss and off-road
      loss are complementary in driving — optimising one helps the other.
      Reference: https://arxiv.org/html/2411.19747v1

    Planned improvements for Phase 5:
    1. Implement DiCo-style diversity control to a target value
    2. Add dynamics model for future-behaviour-based role differentiation
    3. Add semantic alignment losses correlating role dims with speed,
       steering, and proximity to other vehicles
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
        self.role_dim   = role_dim
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
        Diversity loss — pushes agents to have different role vectors.

        For role_dim=1:
            Cosine similarity between scalars is always +1 or -1 with no
            gradient in between. Instead we use negative variance across
            agents — maximising variance = agents spread across scalar space.
            Range: (-∞, 0] — 0 means all agents identical (bad),
            more negative means more diverse (better).
            We normalise by the mean absolute value to keep it scaled.

        For role_dim≥2:
            Average pairwise cosine similarity between role means.
            Range: [-1, +1]
                +1 = all agents identical (collapse — bad)
                 0 = roles uncorrelated
                -1 = agents maximally diverse (ideal)

        Both approaches need no clipping — naturally bounded.
        """
        B = role_mean.size(0)
        if B < 2:
            return torch.tensor(0.0, device=role_mean.device)

        if self.role_dim == 1:
            # For scalar roles: maximise variance across agents
            # Normalise by mean abs value so scale doesn't depend on role magnitude
            role_vals = role_mean.squeeze(-1)           # (B,)
            var       = role_vals.var()
            scale     = role_vals.abs().mean().detach() + 1e-8
            return -(var / scale)                       # negative = we minimise this

        else:
            # For multidim roles: minimise average pairwise cosine similarity
            normed     = F.normalize(role_mean, dim=-1) # (B, role_dim)
            sim_matrix = normed @ normed.T               # (B, B)
            mask       = 1.0 - torch.eye(B, device=role_mean.device)
            return (sim_matrix * mask).sum() / mask.sum()

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