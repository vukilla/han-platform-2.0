from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from xgen.retarget import validate_retarget, RetargetResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate retargeted motions against SMPL keypoints")
    parser.add_argument("--landmarks", required=True, help="NPZ file with landmark arrays")
    parser.add_argument("--qpos", required=True, help="NPZ file with robot_qpos")
    parser.add_argument("--root", required=True, help="NPZ file with root_pose")
    parser.add_argument("--joint-names", required=True, help="JSON list of joint names")
    parser.add_argument("--out", required=True, help="Output JSON report path")
    args = parser.parse_args()

    landmarks_npz = np.load(args.landmarks)
    landmarks = {k: landmarks_npz[k] for k in landmarks_npz.files}
    qpos = np.load(args.qpos)["robot_qpos"] if args.qpos.endswith(".npz") else np.load(args.qpos)
    root = np.load(args.root)["root_pose"] if args.root.endswith(".npz") else np.load(args.root)
    joint_names = json.loads(Path(args.joint_names).read_text(encoding="utf-8"))

    result = RetargetResult(robot_qpos=qpos, root_pose=root, joint_names=joint_names)
    report = validate_retarget(landmarks, result)

    payload = report.__dict__
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
