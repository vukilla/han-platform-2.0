from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class RobotSpec:
    name: str
    urdf: str
    root_body: str
    joint_names: List[str]
    contact_bodies: List[str]
    key_bodies: List[str]


def load_robot_spec(path: str | Path) -> RobotSpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return RobotSpec(
        name=str(payload.get("name", "generic_humanoid")),
        urdf=str(payload.get("urdf", "")),
        root_body=str(payload.get("root_body", "base")),
        joint_names=list(payload.get("joint_names", [])),
        contact_bodies=list(payload.get("contact_bodies", [])),
        key_bodies=list(payload.get("key_bodies", [])),
    )
