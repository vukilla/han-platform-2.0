import math
import unittest

import numpy as np

from xmimic.rewards.unified import HumanXRewardConfig, RewardTerm, compute_reward_terms


class TestUnifiedRewards(unittest.TestCase):
    def test_contact_reward_weighted_abs(self):
        cfg = HumanXRewardConfig(
            contact=RewardTerm(gamma=1.0, lambda_=1.0),
            contact_weights=[1.0, 2.0, 3.0],
            regularization=0.0,
        )
        scg = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        shat = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        # abs diff = [0, 1, 1]; weighted sum = 2*1 + 3*1 = 5
        expected = math.exp(-5.0)
        terms = compute_reward_terms(obs={"contact": scg}, targets={"contact": shat}, config=cfg)
        self.assertAlmostEqual(terms["contact"], expected, places=6)

    def test_contact_weights_must_match_dim(self):
        cfg = HumanXRewardConfig(
            contact=RewardTerm(gamma=1.0, lambda_=1.0),
            contact_weights=[1.0, 2.0],
            regularization=0.0,
        )
        scg = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_reward_terms(obs={"contact": scg}, targets={"contact": scg}, config=cfg)

    def test_regularization_action_rate(self):
        cfg = HumanXRewardConfig(
            regularization=0.5,
            reg_terms={"action_l2": 2.0, "action_rate": 3.0},
        )
        action = np.array([1.0, -1.0], dtype=np.float32)
        prev = np.array([0.0, 0.0], dtype=np.float32)
        # action_l2: mean([1,1]) = 1
        # action_rate: mean([(1-0)^2, (-1-0)^2]) = 1
        # reg_total = 2*1 + 3*1 = 5
        # term = -0.5 * 5 = -2.5
        terms = compute_reward_terms(obs={"action": action, "prev_action": prev}, targets={}, config=cfg)
        self.assertAlmostEqual(terms["regularization"], -2.5, places=6)

    def test_relative_pos_uses_mean_squared_l2_norm(self):
        cfg = HumanXRewardConfig(relative_pos=RewardTerm(gamma=1.0, lambda_=1.0), regularization=0.0)
        a = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        b = np.zeros_like(a)
        # squared norms: [1, 4], mean = 2.5
        expected = math.exp(-2.5)
        terms = compute_reward_terms(obs={"relative_pos": a}, targets={"relative_pos": b}, config=cfg)
        self.assertAlmostEqual(terms["relative_pos"], expected, places=6)

    def test_relative_pos_derived_from_body_pos_and_object_pos(self):
        cfg = HumanXRewardConfig(relative_pos=RewardTerm(gamma=1.0, lambda_=1.0), regularization=0.0)
        body = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        obj = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # relative vectors are body - obj, so identical to `body` here.
        expected = math.exp(-2.5)
        terms = compute_reward_terms(
            obs={"body_pos": body, "object_pos": obj},
            targets={"body_pos": np.zeros_like(body), "object_pos": obj},
            config=cfg,
        )
        self.assertAlmostEqual(terms["relative_pos"], expected, places=6)

    def test_relative_pos_derived_from_key_body_pos_and_object_pos(self):
        cfg = HumanXRewardConfig(relative_pos=RewardTerm(gamma=1.0, lambda_=1.0), regularization=0.0)
        key_body = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        obj = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        expected = math.exp(-2.5)
        terms = compute_reward_terms(
            obs={"key_body_pos": key_body, "object_pos": obj},
            targets={"key_body_pos": np.zeros_like(key_body), "object_pos": obj},
            config=cfg,
        )
        self.assertAlmostEqual(terms["relative_pos"], expected, places=6)

    def test_relative_pos_prefers_key_body_pos_over_body_pos_when_both_present(self):
        cfg = HumanXRewardConfig(relative_pos=RewardTerm(gamma=1.0, lambda_=1.0), regularization=0.0)
        obj = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        key_body = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # squared norm=1
        body = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)  # squared norm=100

        terms = compute_reward_terms(
            obs={"key_body_pos": key_body, "body_pos": body, "object_pos": obj},
            targets={"key_body_pos": np.zeros_like(key_body), "body_pos": np.zeros_like(body), "object_pos": obj},
            config=cfg,
        )
        self.assertAlmostEqual(terms["relative_pos"], math.exp(-1.0), places=6)

    def test_regularization_additional_table_iv_terms(self):
        cfg = HumanXRewardConfig(
            regularization=0.5,
            reg_terms={"feet_orientation": 2.0, "feet_slippage": 3.0, "dof_limit": 4.0, "torque_limit": 5.0},
        )
        obs = {
            "feet_orientation": np.array([2.0, 0.0], dtype=np.float32),  # mean(square)=2
            "feet_slippage": np.array([1.0], dtype=np.float32),  # mean=1
            "dof_limit": np.array([0.25, 0.75], dtype=np.float32),  # mean=0.5
            "torque_limit": np.array([0.1, 0.3], dtype=np.float32),  # mean=0.2
        }
        # reg_total = 2*2 + 3*1 + 4*0.5 + 5*0.2 = 4 + 3 + 2 + 1 = 10
        # term = -0.5 * 10 = -5
        terms = compute_reward_terms(obs=obs, targets={}, config=cfg)
        self.assertAlmostEqual(terms["regularization"], -5.0, places=6)

    def test_regularization_torque_waist_and_termination(self):
        cfg = HumanXRewardConfig(
            regularization=0.5,
            reg_terms={"torque": 2.0, "waist_dof": 3.0, "termination": 4.0},
        )
        obs = {
            "torque": np.array([1.0, -1.0], dtype=np.float32),  # mean(square)=1
            "waist_dof": np.array([2.0], dtype=np.float32),  # mean(square)=4
            "termination": np.array([1.0], dtype=np.float32),  # mean=1
        }
        # reg_total = 2*1 + 3*4 + 4*1 = 2 + 12 + 4 = 18
        # term = -0.5 * 18 = -9
        terms = compute_reward_terms(obs=obs, targets={}, config=cfg)
        self.assertAlmostEqual(terms["regularization"], -9.0, places=6)


if __name__ == "__main__":
    unittest.main()
