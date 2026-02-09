# Contact Refinement Pipeline

Combines anchor propagation with force-closure refinement and optional QP force solve.

Module: `services/xgen/xgen/interaction/contact_synth.py`

## Usage
```python
from xgen.interaction import refine_contact_phase

result = refine_contact_phase(object_pose, anchors, contact_indices, contact_points)
refined_pose = result.object_pose
forces = result.contact_forces
```

## Optional: CPU Robot IK Refinement (Pre-Physics)

If you also have a retargeted robot trajectory, you can refine contact frames to better match
end-effector targets (e.g. hands at object contact points) before any physics simulation:

```python
result = refine_contact_phase(
    object_pose,
    anchors,
    contact_indices,
    robot_urdf_path="assets/robots/generic_humanoid.urdf",
    robot_joint_names=joint_names,
    robot_qpos=robot_qpos,
    tip_targets={"left_hand": left_targets, "right_hand": right_targets},
    # Multi-tip SQP-style solve (recommended when refining both hands together):
    ik_method="sqp",
    posture_weight=1e-3,
)
refined_robot_qpos = result.robot_qpos
```
