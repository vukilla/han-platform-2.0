#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _bootstrap_imports() -> None:
    # Allow running directly from the monorepo without installing the package.
    here = Path(__file__).resolve()
    svc_root = here.parents[1]  # services/xmimic
    sys.path.insert(0, str(svc_root / "xmimic"))


def _parse_urdf(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()
    joints = {j.get("name"): j.get("type") for j in root.findall("joint")}
    links = {l.get("name") for l in root.findall("link")}
    return joints, links


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate robot_spec.yaml against URDF joint/link names.")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "robot_spec.yaml",
        help="Path to robot_spec.yaml (default: services/xmimic/configs/robot_spec.yaml).",
    )
    parser.add_argument("--urdf", type=Path, default=None, help="Override URDF path (otherwise use spec.urdf).")
    args = parser.parse_args()

    _bootstrap_imports()
    # Import the module directly to avoid importing the package __init__ (which may require heavy deps).
    from robot_spec import load_robot_spec

    spec = load_robot_spec(args.spec)
    urdf_path = args.urdf or Path(spec.urdf)
    if not urdf_path.is_absolute():
        # Resolve relative URDF paths from repo root.
        repo_root = Path(__file__).resolve().parents[3]
        urdf_path = (repo_root / urdf_path).resolve()

    if not urdf_path.exists():
        print(f"URDF not found: {urdf_path}")
        return 2

    joints, links = _parse_urdf(urdf_path)
    missing_joints = [name for name in spec.joint_names if name not in joints]
    missing_links = [name for name in (spec.contact_bodies + spec.key_bodies) if name not in links]

    ok = True
    if missing_joints:
        ok = False
        print("Missing joints in URDF:")
        for name in missing_joints:
            print(f"  - {name}")
    if missing_links:
        ok = False
        print("Missing links in URDF:")
        for name in missing_links:
            print(f"  - {name}")

    if ok:
        print("OK: robot_spec matches URDF for joints and required link names.")
        print(f"Spec: {args.spec}")
        print(f"URDF: {urdf_path}")
        print(f"Joints: {len(spec.joint_names)} (spec) / {len(joints)} (urdf)")
        print(f"Links: {len(links)} (urdf)")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
