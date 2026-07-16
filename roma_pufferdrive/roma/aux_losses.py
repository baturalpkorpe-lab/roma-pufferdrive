"""
roma/aux_losses.py
==================
ROMA auxiliary losses: MI loss + diversity loss.

MI Loss (Mutual Information) — FUTURE-prediction variant (this branch):
    Forces the role vector to be PREDICTIVE of the agent's upcoming
    behaviour: z_t must predict a behaviour summary extracted from the
    NEXT `window` env embeddings (t+1..t+window), not the past ones.
    R3DM-inspired (Goel et al., ICML 2025): linking roles to FUTURE
    expected behaviour improves role differentiation vs the original
    past-window target (baseline branch), which risks encoding "what
    already happened / what the scene looks like" instead of intent.

    Mechanically identical to the baseline loss except the caller passes
    the future window (train_roma.py indexes the rollout buffer at t+H)
    plus a validity mask: samples whose future crosses the rollout tail
    or an episode reset are excluded from the loss.

    This remains a simplified approximation of the full ROMA MI loss
    (Wang et al., ICML 2020), which uses a GRU trajectory encoder q_ξ
    to estimate I(ρ; τ | o). Our BehaviourExtractor + MIDecoder serves
    the same purpose with less complexity.

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
    Encodes the last `window` env embeddings into a behaviour summary vector.
    This is the target that the MI loss tries to predict from the role vector.

    Input is the policy's env embedding (ego + partner + road encoders,
    128-dim) across the window — a structured, learned behavioural
    trajectory — instead of raw 1121-dim observations. Two agents in the
    same scene that drive differently produce different env-embedding
    sequences (the ego encoder captures their own speed/heading), so the
    MI target now distinguishes driving style, not just scene content.
    """
    def __init__(self, emb_dim, behaviour_dim=32, window=8):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(
            nn.Linear(emb_dim * window, 64),
            nn.ReLU(),
            nn.Linear(64, behaviour_dim),
        )

    def forward(self, emb_window):
        B = emb_window.size(0)
        flat = emb_window.reshape(B, -1)
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
        emb_dim       : env embedding dimension (policy.env_embed_dim)
        behaviour_dim : dimension of the behaviour summary vector
        hidden_dim    : hidden size of the MI decoder
        window        : number of past env embeddings for behaviour extraction
        mi_weight     : weight on the MI loss (default 1.0, from ROMA paper)
        div_weight    : weight on the diversity loss (default 0.1)
    """
    def __init__(self, role_dim, emb_dim, behaviour_dim=32, hidden_dim=64,
                 window=8, mi_weight=1.0, div_weight=0.1):
        super().__init__()
        self.role_dim   = role_dim
        self.mi_weight  = mi_weight
        self.div_weight = div_weight
        self.behaviour_extractor = BehaviourExtractor(emb_dim, behaviour_dim, window)
        self.mi_decoder          = MIDecoder(role_dim, behaviour_dim, hidden_dim)

    def mi_loss(self, role_z, emb_window, mi_mask=None):
        """
        MSE between predicted and actual behaviour summary.
        Forces role vector to encode real behavioural information.
        Range: [0, ∞) — lower is better.

        mi_mask (B,) bool: samples to include. Future-window targets are
        invalid where the future crosses the rollout tail or an episode
        reset; those samples are excluded (loss averaged over valid only).
        """
        behaviour_target = self.behaviour_extractor(emb_window).detach()
        behaviour_pred   = self.mi_decoder(role_z)
        if mi_mask is None:
            return F.mse_loss(behaviour_pred, behaviour_target)
        if not bool(mi_mask.any()):
            # keep graph + dtype/device; contributes zero gradient
            return (behaviour_pred.sum() * 0.0)
        per = ((behaviour_pred - behaviour_target) ** 2).mean(dim=-1)
        return per[mi_mask].mean()

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
            # For scalar roles: maximise the SPREAD across agents, but as a
            # scale-INVARIANT quantity so it can't be gamed by inflating role
            # magnitude. The old form -(var/scale) used variance (~magnitude^2)
            # over mean|abs| (~magnitude^1), so the ratio grew linearly with
            # magnitude -> the optimizer blew the role up to +/-inf (div_loss
            # -> -1000s) and the MI target became unfittable. Using std (not
            # var) makes numerator and denominator both ~magnitude^1, i.e. a
            # coefficient of variation, bounded ~O(1) and magnitude-invariant
            # like the cosine path for role_dim>=2.
            role_vals = role_mean.squeeze(-1)           # (B,)
            std       = role_vals.std()
            scale     = role_vals.abs().mean().detach() + 1e-6
            return -(std / scale)                       # negative = we minimise this

        else:
            # For multidim roles: minimise average pairwise cosine similarity.
            #
            # Closed form instead of the naive (B, B) similarity matrix:
            #   sum_{i != j} n_i . n_j = ||sum_i n_i||^2 - sum_i ||n_i||^2
            #                          = ||sum_i n_i||^2 - B   (unit vectors)
            # The naive normed @ normed.T needs O(B^2) memory — at the
            # production minibatch size (B ~ 65k+) that is a >16 GB tensor
            # and an instant CUDA OOM. This form is O(B*d) and identical
            # in value and gradient.
            normed = F.normalize(role_mean, dim=-1)      # (B, role_dim)
            s      = normed.sum(dim=0)                   # (role_dim,)
            total  = (s * s).sum() - B                   # sum of off-diagonal sims
            return total / (B * (B - 1))

    def forward(self, role_z, role_mean, role_log_var, emb_window,
                mi_mask=None):
        """
        Args:
            role_z       : sampled role vector (B, role_dim)
            role_mean    : role distribution mean (B, role_dim)
            role_log_var : role distribution log variance (B, role_dim)
            emb_window   : `window` env embeddings (B, window, emb_dim) —
                           on this branch the FUTURE window t+1..t+window
                           (the caller indexes the rollout buffer at t+H)
            mi_mask      : (B,) bool — valid future targets (see mi_loss)

        Returns:
            dict with mi_loss, div_loss, aux_loss
        """
        l_mi  = self.mi_loss(role_z, emb_window, mi_mask)
        l_div = self.diversity_loss(role_mean)
        aux   = self.mi_weight * l_mi + self.div_weight * l_div

        return {"mi_loss": l_mi, "div_loss": l_div, "aux_loss": aux}