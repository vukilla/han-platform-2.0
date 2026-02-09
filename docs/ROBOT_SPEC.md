# Robot Spec

We need a consistent robot definition for retargeting, contact graph labeling, and reward computation.
This repo now includes a lightweight spec format and loader.

## Files
- `services/xmimic/xmimic/robot_spec.py`
- `services/xmimic/configs/robot_spec.yaml`
- `assets/robots/generic_humanoid.urdf` (v0, not tuned)

## Fields
- `name`: human-readable robot name
- `urdf`: path to URDF asset
- `root_body`: root body name used by the sim
- `joint_names`: deterministic joint order used for q/qdot
- `contact_bodies`: bodies used for contact-graph labels
- `key_bodies`: bodies used for relative-motion rewards

## Next steps
- Validate spec vs URDF:
  - `python services/xmimic/scripts/robot_spec_validate.py`
- Replace `generic_humanoid.urdf` with a real asset once licensed/available.
- Update joint order to match the chosen URDF.
- Align contact bodies with hand/foot collision geometries.
