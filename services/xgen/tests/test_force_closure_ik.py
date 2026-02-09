import unittest

import numpy as np
from pathlib import Path

from xgen.interaction.force_closure_refine import _build_ik_chain, refine_contact_ik
from xgen.interaction.contact_synth import refine_contact_phase

try:  # pragma: no cover
    import ikpy  # noqa: F401

    _IKPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _IKPY_AVAILABLE = False


class TestForceClosureIK(unittest.TestCase):
    @unittest.skipUnless(_IKPY_AVAILABLE, "ikpy not installed")
    def test_refine_contact_ik_moves_tip_to_target(self):
        repo_root = Path(__file__).resolve().parents[3]
        urdf = str(repo_root / "assets" / "robots" / "generic_humanoid.urdf")
        # Minimal joint set on the root->left_hand chain for this URDF.
        joint_names = [
            "torso_yaw",
            "torso_pitch",
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_shoulder_yaw",
            "left_elbow",
            "left_wrist",
        ]
        qpos = np.zeros((1, len(joint_names)), dtype=np.float32)
        target = np.array([[0.4, 0.3, 0.5]], dtype=np.float32)

        refined = refine_contact_ik(
            urdf_path=urdf,
            joint_names=joint_names,
            robot_qpos=qpos,
            contact_indices=np.array([0], dtype=int),
            tip_targets={"left_hand": target},
            max_joint_delta=10.0,  # allow full IK update in one step for test determinism
        )

        chain = _build_ik_chain(urdf, tip_link="left_hand")
        jidx = {n: i for i, n in enumerate(joint_names)}
        qvec = np.zeros((len(chain.links),), dtype=np.float32)
        for li, link in enumerate(chain.links):
            if li == 0:
                continue
            if link.name in jidx:
                qvec[li] = float(refined[0, jidx[link.name]])

        tip = chain.forward_kinematics(qvec)[:3, 3]
        self.assertLess(float(np.linalg.norm(tip - target[0])), 1e-2)

    @unittest.skipUnless(_IKPY_AVAILABLE, "ikpy not installed")
    def test_refine_contact_phase_can_refine_robot_qpos(self):
        repo_root = Path(__file__).resolve().parents[3]
        urdf = str(repo_root / "assets" / "robots" / "generic_humanoid.urdf")
        joint_names = [
            "torso_yaw",
            "torso_pitch",
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_shoulder_yaw",
            "left_elbow",
            "left_wrist",
        ]
        robot_qpos = np.zeros((1, len(joint_names)), dtype=np.float32)
        target = np.array([[0.4, 0.3, 0.5]], dtype=np.float32)

        object_pose = np.zeros((1, 7), dtype=np.float32)
        anchors = np.zeros((1, 3), dtype=np.float32)
        contact = np.array([0], dtype=int)

        result = refine_contact_phase(
            object_pose,
            anchors,
            contact,
            robot_urdf_path=urdf,
            robot_joint_names=joint_names,
            robot_qpos=robot_qpos,
            tip_targets={"left_hand": target},
            max_joint_delta=10.0,
        )
        assert result.robot_qpos is not None

        chain = _build_ik_chain(urdf, tip_link="left_hand")
        jidx = {n: i for i, n in enumerate(joint_names)}
        qvec = np.zeros((len(chain.links),), dtype=np.float32)
        for li, link in enumerate(chain.links):
            if li == 0:
                continue
            if link.name in jidx:
                qvec[li] = float(result.robot_qpos[0, jidx[link.name]])
        tip = chain.forward_kinematics(qvec)[:3, 3]
        self.assertLess(float(np.linalg.norm(tip - target[0])), 1e-2)


if __name__ == "__main__":
    unittest.main()
