import torch
import unittest


class TestRoleEncoder(unittest.TestCase):
    def setUp(self):
        from roma.role_encoder import RoleEncoder
        self.obs_dim  = 1848
        self.role_dim = 8
        self.hidden   = 64
        self.B        = 16
        self.enc = RoleEncoder(self.obs_dim, self.role_dim, self.hidden)
        self.device = torch.device("cpu")

    def test_init_hidden_shape(self):
        h = self.enc.init_hidden(self.B, self.device)
        self.assertEqual(h.shape, (self.B, self.hidden))

    def test_forward_shapes(self):
        obs = torch.randn(self.B, self.obs_dim)
        h   = self.enc.init_hidden(self.B, self.device)
        self.enc.train()
        role_z, mu, logvar, new_h = self.enc(obs, h)
        self.assertEqual(role_z.shape,  (self.B, self.role_dim))
        self.assertEqual(mu.shape,      (self.B, self.role_dim))
        self.assertEqual(logvar.shape,  (self.B, self.role_dim))
        self.assertEqual(new_h.shape,   (self.B, self.hidden))

    def test_var_floor(self):
        obs = torch.randn(self.B, self.obs_dim)
        h   = self.enc.init_hidden(self.B, self.device)
        self.enc.train()
        _, _, logvar, _ = self.enc(obs, h)
        var = logvar.exp()
        self.assertTrue((var >= self.enc.var_floor).all())

    def test_eval_deterministic(self):
        obs = torch.randn(self.B, self.obs_dim)
        h   = self.enc.init_hidden(self.B, self.device)
        self.enc.eval()
        with torch.no_grad():
            role_z, mu, _, _ = self.enc(obs, h)
        self.assertTrue(torch.allclose(role_z, mu))


class TestAuxLoss(unittest.TestCase):
    def setUp(self):
        from roma.aux_losses import RomaAuxLoss
        self.obs_dim  = 1848
        self.role_dim = 8
        self.B        = 16
        self.W        = 8
        self.fn = RomaAuxLoss(
            role_dim=self.role_dim, obs_dim=self.obs_dim,
            behaviour_dim=32, window=self.W,
        )

    def test_forward_keys(self):
        role_z    = torch.randn(self.B, self.role_dim)
        role_mean = torch.randn(self.B, self.role_dim)
        role_lv   = torch.randn(self.B, self.role_dim)
        obs_win   = torch.randn(self.B, self.W, self.obs_dim)
        self.fn.train()
        out = self.fn(role_z, role_mean, role_lv, obs_win)
        for key in ("mi_loss", "div_loss", "aux_loss"):
            self.assertIn(key, out)

    def test_scalar_losses(self):
        role_z    = torch.randn(self.B, self.role_dim)
        role_mean = torch.randn(self.B, self.role_dim)
        role_lv   = torch.zeros(self.B, self.role_dim)
        obs_win   = torch.randn(self.B, self.W, self.obs_dim)
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
            torch.randn(1, self.W, self.obs_dim),
        )
        self.assertAlmostEqual(out["div_loss"].item(), 0.0, places=5)


class TestRomaPolicy(unittest.TestCase):
    def setUp(self):
        from roma.policy import RomaPolicy
        self.obs_dim    = 1848
        self.action_dim = 91
        self.role_dim   = 8
        self.B          = 4
        self.policy = RomaPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            role_dim=self.role_dim,
        )
        self.device = torch.device("cpu")

    def test_initial_state_shapes(self):
        state = self.policy.initial_state(self.B, self.device)
        role_h, policy_h, obs_win = state
        self.assertEqual(role_h.shape,   (self.B, 64))
        self.assertEqual(policy_h.shape, (self.B, 128))
        self.assertEqual(obs_win.shape,  (self.B, 8, self.obs_dim))

    def test_forward_shapes(self):
        obs   = torch.randn(self.B, self.obs_dim)
        state = self.policy.initial_state(self.B, self.device)
        self.policy.train()
        logits, value, new_state, role_info = self.policy(obs, state)
        self.assertEqual(logits.shape, (self.B, self.action_dim))
        self.assertEqual(value.shape,  (self.B, 1))
        for k in ("role_z", "role_mean", "role_log_var"):
            self.assertEqual(role_info[k].shape, (self.B, self.role_dim))
        self.assertEqual(role_info["obs_window"].shape, (self.B, 8, self.obs_dim))

    def test_no_nan(self):
        obs   = torch.randn(self.B, self.obs_dim)
        state = self.policy.initial_state(self.B, self.device)
        self.policy.train()
        logits, value, _, role_info = self.policy(obs, state)
        for name, t in [("logits", logits), ("value", value),
                         ("role_z", role_info["role_z"])]:
            self.assertFalse(torch.isnan(t).any())

    def test_obs_window_shift(self):
        obs    = torch.ones(self.B, self.obs_dim) * 99.0
        state  = self.policy.initial_state(self.B, self.device)
        self.policy.eval()
        with torch.no_grad():
            _, _, new_state, role_info = self.policy(obs, state)
        self.assertTrue(torch.allclose(role_info["obs_window"][:, -1, :], obs))


class TestObsSplit(unittest.TestCase):
    def test_classic_split(self):
        from roma.policy import RomaPolicy
        B   = 2
        obs = torch.randn(B, 1848)
        p   = RomaPolicy(obs_dim=1848, ego_dim=7)
        ego, partners, roads = p._split_obs(obs)
        self.assertEqual(ego.shape,      (B, 7))
        self.assertEqual(partners.shape, (B, 217))
        self.assertEqual(roads.shape,    (B, 1624))


if __name__ == "__main__":
    print("Running ROMA module tests ...\n")
    unittest.main(verbosity=2)
