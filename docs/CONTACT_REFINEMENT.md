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
