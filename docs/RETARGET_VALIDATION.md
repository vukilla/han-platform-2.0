# Retargeting Validation

This validator checks three critical failure modes:

1. End-effector alignment (wrist vs SMPL keypoints).
2. Joint limit violations (per-joint bounds).
3. Foot skating (high speed while foot is near ground).

## API
Use `xgen.retarget.validate_retarget(landmarks, result)`.

## CLI
```
PYTHONPATH=services/xgen python services/xgen/scripts/retarget_validate.py \
  --landmarks /path/landmarks.npz \
  --qpos /path/qpos.npz \
  --root /path/root.npz \
  --joint-names /path/joint_names.json \
  --out /path/retarget_report.json
```

Notes:
- Foot skating is computed if ankle/foot landmarks are present.
- Joint limits default to [-pi, pi] unless provided programmatically.
