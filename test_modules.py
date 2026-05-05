"""
test_modules.py
===============
Unit tests for ROMA on PufferDrive.
Run this before every training run to verify all components are healthy.

Usage:
    cd ~/PufferDrive/roma_pufferdrive
    python3 test_modules.py

All 12 tests should pass. If OBS_DIM tests fail, run check_env.py
and update OBS_DIM below to match your environment.
"""

import torch
import unittest

# ---------------------------------------------------------------------------
# Single source of truth for observation dimension.
# Change this if check_env.py reports a different value.
# ---------------------------------------------------------------------------
OBS_DIM = 1121  # confirmed by check_env.py


class TestRoleEncoder(unittest.TestCase):
    def setUp(self):
        from roma.role_encoder import RoleEncoder
        self.role_dim = 8
        self.hidden   = 64
        self.B        = 16
        self.enc      = RoleEncoder(OBS_DIM, self.role_dim, self.hidden)
        self.device   = torch.device("cpu")

    def test_init_hidden_shape(self):
        h = self.enc.init_hidden(self.B, self.device)
        self.assertEqual(h.shape, (self.B, self.hidden))

    def test_forward_shapes(self):
        obs = torch.randn(self.B, OBS_DIM)
        h   = self.enc.init_hidden(self.B, self.device)
        self.enc.train()
        role_z, mu, logvar, new_h = self.enc(obs, h)
        self.assertEqual(role_z.shape,  (self.B, self.role_dim))
        self.assertEqual(mu.shape,      (self.B, self.role_dim))
        self.assertEqual(logvar.shape,  (self.B, self.role_dim))
        self.assertEqual(new_h.shape,   (self.B, self.hidden))

    def test_var_floor(self):
        obs = torch.randn(self.B, OBS_DIM)
        h   = self.enc.init_hidden(self.B, self.device)
        self.enc.train()
        _, _, logvar, _ = self.enc(obs, h)
        var = logvar.exp()
        self.assertTrue((var >= self.enc.var_floor).all())

    def test_eval_deterministic(self):
        obs = torch.randn(self.B, OBS_DIM)
        h   = self.enc.init_hidden(self.B, self.device)
        self.enc.eval()
        with torch.no_grad():
            role_z, mu, _, _ = self.enc(obs, h)
        self.assertTrue(torch.allclose(role_z, mu))


class TestAuxLoss(unittest.TestCase):
    def setUp(self):
        from roma.aux_losses import RomaAuxLoss
        self.role_dim = 8
        self.B        = 16
        self.W        = 8
        self.fn       = RomaAuxLoss(
            role_dim=self.role_dim, obs_dim=OBS_DIM,
            behaviour_dim=32, window=self.W,
        )

    def test_forward_keys(self):
        role_z    = torch.randn(self.B, self.role_dim)
        role_mean = torch.randn(self.B, self.role_dim)
        role_lv   = torch.randn(self.B, self.role_dim)
        obs_win   = torch.randn(self.B, self.W, OBS_DIM)
        self.fn.train()
        out = self.fn(role_z, role_mean, role_lv, obs_win)
        for key in ("mi_loss", "div_loss", "aux_loss"):
            self.assertIn(key, out)

    def test_scalar_losses(self):
        role_z    = torch.randn(self.B, self.role_dim)
        role_mean = torch.randn(self.B, self.role_dim)
        role_lv   = torch.zeros(self.B, self.role_dim)
        obs_win   = torch.randn(self.B, self.W, OBS_DIM)
        self.fn.train()
        out = self.fn(role_z, role_mean, role_lv, obs_win)
        for key in ("mi_loss", "div_loss", "aux_loss"):
            self.assertEqual(out[key].shape, torch.Size([]))
            self.assertFalse(torch.isnan(out[key]))

    def test_diversity_single_agent(self):
        out = self.fn(
            torch.randn(1, self.role_dim),
            torch.randn(1, self.role_dim),
            torch.zeros(1, self.role_dim),
            torch.randn(1, self.W, OBS_DIM),
        )
        self.assertAlmostEqual(out["div_loss"].item(), 0.0, places=5)


class TestRomaPolicy(unittest.TestCase):
    def setUp(self):
        from roma.policy import RomaPolicy
        self.action_dim = 91
        self.role_dim   = 8
        self.B          = 4
        self.policy     = RomaPolicy(
            obs_dim    = OBS_DIM,
            action_dim = self.action_dim,
            role_dim   = self.role_dim,
        )
        self.device = torch.device("cpu")

    def test_initial_state_shapes(self):
        state = self.policy.initial_state(self.B, self.device)
        role_h, policy_h, obs_win = state
        self.assertEqual(role_h.shape,   (self.B, 64))
        self.assertEqual(policy_h.shape, (self.B, 128))
        self.assertEqual(obs_win.shape,  (self.B, 8, OBS_DIM))

    def test_forward_shapes(self):
        obs   = torch.randn(self.B, OBS_DIM)
        state = self.policy.initial_state(self.B, self.device)
        self.policy.train()
        logits, value, new_state, role_info = self.policy(obs, state)
        self.assertEqual(logits.shape, (self.B, self.action_dim))
        self.assertEqual(value.shape,  (self.B, 1))
        for k in ("role_z", "role_mean", "role_log_var"):
            self.assertEqual(role_info[k].shape, (self.B, self.role_dim))
        self.assertEqual(role_info["obs_window"].shape, (self.B, 8, OBS_DIM))

    def test_no_nan(self):
        obs   = torch.randn(self.B, OBS_DIM)
        state = self.policy.initial_state(self.B, self.device)
        self.policy.train()
        logits, value, _, role_info = self.policy(obs, state)
        for name, t in [("logits", logits), ("value", value),
                         ("role_z", role_info["role_z"])]:
            self.assertFalse(torch.isnan(t).any(), f"NaN in {name}")

    def test_obs_window_shift(self):
        obs   = torch.ones(self.B, OBS_DIM) * 99.0
        state = self.policy.initial_state(self.B, self.device)
        self.policy.eval()
        with torch.no_grad():
            _, _, _, role_info = self.policy(obs, state)
        self.assertTrue(
            torch.allclose(role_info["obs_window"][:, -1, :], obs),
            "Last obs_window slot should equal current observation"
        )


class TestObsSplit(unittest.TestCase):
    """Verify the structured encoder correctly splits the observation vector.

    Confirmed layout (check_env.py):
      [0:7]      ego      — 7 features
      [7:224]    partners — 31×7 = 217 features
      [224:1120] roads    — 128×7 = 896 features
      [1120]     padding  — 1 byte, dropped
    """

    def test_split_dimensions(self):
        from roma.policy import RomaPolicy
        B      = 2
        obs    = torch.randn(B, OBS_DIM)
        policy = RomaPolicy(obs_dim=OBS_DIM)
        ego, partners, roads = policy._split_obs(obs)

        self.assertEqual(ego.shape,      (B, 7),   "ego shape wrong")
        self.assertEqual(partners.shape, (B, 217), "partner shape wrong")
        # 128 road points × 7 features = 896  (padding byte at [1120] is dropped)
        self.assertEqual(roads.shape,    (B, 896), "road shape wrong")

    def test_split_covers_obs(self):
        """ego + partners + roads should cover 1120 of 1121 bytes (padding dropped)."""
        from roma.policy import RomaPolicy
        B      = 1
        obs    = torch.arange(OBS_DIM, dtype=torch.float32).unsqueeze(0)
        policy = RomaPolicy(obs_dim=OBS_DIM)
        ego, partners, roads = policy._split_obs(obs)
        total = ego.shape[1] + partners.shape[1] + roads.shape[1]
        self.assertEqual(total, OBS_DIM - 1,
                         "Should cover 1120 features (1121 minus 1 padding byte)")


if __name__ == "__main__":
    print(f"Running ROMA module tests (OBS_DIM={OBS_DIM}) ...\n")
    unittest.main(verbosity=2)
