# Dataset Clip Schema

Each clip is stored as a compressed `.npz` with these keys:

- `robot_qpos` (T, nq)
- `robot_qvel` (T, nq)
- `root_pose` (T, 7) position + quaternion
- `root_vel` (T, 3) linear velocity
- `object_pose` (T, 7) position + quaternion
- `object_vel` (T, 3) linear velocity
- `contact_graph` (T, J)
- `phase` (T,) enum: 0 pre-contact, 1 contact, 2 post-contact

Metadata is stored alongside in `*.metadata.json`:
- `skill_id`, `pattern_id`
- `anchor_type`
- `augmentation` params
- `ts_contact_start`, `ts_contact_end`
- `fps`

See `services/xgen/xgen/export/schema.py` and `services/xgen/xgen/export/clip_export.py`.
