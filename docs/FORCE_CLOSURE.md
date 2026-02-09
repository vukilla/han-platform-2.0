# Force-Closure QP

We added a QP-based force-closure solver for contact forces.

Module: `services/xgen/xgen/interaction/force_closure_refine.py`

## Usage
```python
from xgen.interaction import solve_force_closure_qp

forces = solve_force_closure_qp(contact_points, contact_normals, desired_wrench, friction_coeff=0.5)
```
