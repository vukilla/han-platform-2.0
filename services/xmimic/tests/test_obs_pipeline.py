import unittest

import numpy as np

from xmimic.obs_pipeline import ObservationPipeline, humanx_student_obs_config, humanx_teacher_obs_config
from xmimic.robot_spec import RobotSpec


class TestObsConfigs(unittest.TestCase):
    def _robot(self) -> RobotSpec:
        return RobotSpec(
            name="testbot",
            urdf="",
            root_body="base",
            joint_names=["j0", "j1", "j2", "j3"],
            contact_bodies=["hand_l", "hand_r"],
            key_bodies=["hand_l", "hand_r", "head"],
        )

    def test_teacher_schema_matches_table_iv(self):
        robot = self._robot()
        cfg = humanx_teacher_obs_config(robot, history=4)
        self.assertEqual(
            [f.name for f in cfg.fields],
            [
                "base_ang_vel",
                "gravity",
                "dof_pos",
                "dof_vel",
                "action",
                "pd_error",
                "ref_body_pos",
                "delta_body_pos",
                "object_pos",
            ],
        )
        self.assertEqual(
            [f.name for f in cfg.history_fields or []],
            ["base_ang_vel", "gravity", "dof_pos", "dof_vel", "action"],
        )
        # dim: current + history stack of subset (Table IV "History Terms")
        self.assertEqual(cfg.dim, 115)

    def test_student_nep_schema_matches_table_iv(self):
        robot = self._robot()
        cfg = humanx_student_obs_config(robot, num_skills=5, mode="nep", history=4)
        self.assertEqual(
            [f.name for f in cfg.fields],
            ["base_ang_vel", "gravity", "dof_pos", "dof_vel", "action", "pd_error", "skill_label"],
        )
        self.assertEqual(
            [f.name for f in cfg.history_fields or []],
            ["base_ang_vel", "gravity", "dof_pos", "dof_vel", "action"],
        )
        self.assertEqual(cfg.dim, 99)

    def test_student_mocap_schema_matches_table_iv(self):
        robot = self._robot()
        cfg = humanx_student_obs_config(robot, num_skills=5, mode="mocap", history=4)
        self.assertEqual(
            [f.name for f in cfg.fields],
            ["base_ang_vel", "gravity", "dof_pos", "dof_vel", "action", "pd_error", "object_pos", "skill_label"],
        )
        self.assertEqual(cfg.dim, 102)

    def test_pipeline_accepts_common_aliases(self):
        robot = self._robot()
        cfg = humanx_teacher_obs_config(robot, history=4)
        pipe = ObservationPipeline(cfg)
        dof = len(robot.joint_names)
        key = len(robot.key_bodies)
        obs = {
            "base_ang_vel": np.zeros((3,), dtype=np.float32),
            "projected_gravity": np.zeros((3,), dtype=np.float32),  # alias for gravity
            "dof_pos": np.zeros((dof,), dtype=np.float32),
            "dof_vel": np.zeros((dof,), dtype=np.float32),
            "prev_action": np.zeros((dof,), dtype=np.float32),  # alias for action
            "dof_pos_error": np.zeros((dof,), dtype=np.float32),  # alias for pd_error
            "ref_key_body_pos": np.zeros((key * 3,), dtype=np.float32),  # alias for ref_body_pos
            "delta_key_body_pos": np.zeros((key * 3,), dtype=np.float32),  # alias for delta_body_pos
            "object_pos": np.zeros((3,), dtype=np.float32),
        }
        vec = None
        for _ in range(cfg.history):
            vec = pipe.build(obs, update_norm=False)
        assert vec is not None
        self.assertEqual(vec.shape[0], cfg.dim)


if __name__ == "__main__":
    unittest.main()
