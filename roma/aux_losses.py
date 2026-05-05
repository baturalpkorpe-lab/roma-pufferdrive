import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviourExtractor(nn.Module):
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
    ROMA auxiliary losses: MI loss + diversity loss.

    Fix applied (div_loss explosion):
      The original diversity loss was -KL with no bound, causing it to
      diverge to -infinity as training progressed. Three fixes:
        1. Clip div_loss to [-div_clip, 0] so it cannot explode
        2. Add mean regularization — penalize role means from drifting far from 0
        3. Reduce default div_weight from 0.05 to 0.005

    The clipping value div_clip=10.0 means the diversity loss contributes
    at most 10.0 * div_weight to the total loss, keeping it commensurate
    with the MI loss (~0.008) and PPO loss (~0.02-0.05).
    """
    def __init__(self, role_dim, obs_dim, behaviour_dim=32, hidden_dim=64,
                 window=8, mi_weight=1.0, div_weight=5e-3, div_clip=10.0):
        super().__init__()
        self.mi_weight  = mi_weight
        self.div_weight = div_weight
        self.div_clip   = div_clip          # max magnitude of diversity loss
        self.behaviour_extractor = BehaviourExtractor(obs_dim, behaviour_dim, window)
        self.mi_decoder          = MIDecoder(role_dim, behaviour_dim, hidden_dim)

    @staticmethod
    def _kl_gaussian(mu1, logvar1, mu2, logvar2):
        var1 = logvar1.exp()
        var2 = logvar2.exp()
        kl = 0.5 * (
            logvar2 - logvar1
            + var1 / (var2 + 1e-8)
            + (mu1 - mu2).pow(2) / (var2 + 1e-8)
            - 1.0
        )
        return kl.sum(-1).mean()

    def mi_loss(self, role_z, obs_window):
        behaviour_target = self.behaviour_extractor(obs_window).detach()
        behaviour_pred   = self.mi_decoder(role_z)
        return F.mse_loss(behaviour_pred, behaviour_target)

    def diversity_loss(self, role_mean, role_log_var):
        B = role_mean.size(0)
        if B < 2:
            return torch.tensor(0.0, device=role_mean.device)

        mean_mu     = role_mean.mean(0, keepdim=True).expand(B, -1)
        mean_logvar = role_log_var.mean(0, keepdim=True).expand(B, -1)
        kl = self._kl_gaussian(role_mean, role_log_var, mean_mu, mean_logvar)

        # Fix: clip so diversity loss stays in [-div_clip, 0]
        # Without this, -kl diverges to -infinity as agents spread apart
        div = torch.clamp(-kl, min=-self.div_clip, max=0.0)

        # Fix: penalize role means drifting far from zero (keeps role space bounded)
        mean_reg = 0.01 * role_mean.pow(2).mean()

        return div + mean_reg

    def forward(self, role_z, role_mean, role_log_var, obs_window):
        l_mi  = self.mi_loss(role_z, obs_window)
        l_div = self.diversity_loss(role_mean, role_log_var)
        aux   = self.mi_weight * l_mi + self.div_weight * l_div
        return {"mi_loss": l_mi, "div_loss": l_div, "aux_loss": aux}
