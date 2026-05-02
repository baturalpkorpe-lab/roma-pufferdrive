import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden, out_dim, layers=2):
    dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
    mods = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


class RoleEncoder(nn.Module):
    def __init__(self, obs_dim, role_dim=8, hidden_dim=64, var_floor=1e-4):
        super().__init__()
        self.obs_dim    = obs_dim
        self.role_dim   = role_dim
        self.hidden_dim = hidden_dim
        self.var_floor  = var_floor

        self.fc_obs = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.mu_head     = nn.Linear(hidden_dim, role_dim)
        self.logvar_head = nn.Linear(hidden_dim, role_dim)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs, hidden):
        x          = self.fc_obs(obs)
        new_hidden = self.gru(x, hidden)
        role_mean    = self.mu_head(new_hidden)
        role_log_var = self.logvar_head(new_hidden)
        role_var     = F.softplus(role_log_var) + self.var_floor
        role_log_var = torch.log(role_var)
        if self.training:
            eps    = torch.randn_like(role_mean)
            role_z = role_mean + eps * role_var.sqrt()
        else:
            role_z = role_mean
        return role_z, role_mean, role_log_var, new_hidden
