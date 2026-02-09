# Non-contact Simulation

Forward/backward non-contact simulation helpers with optional damping inversion.

Module: `services/xgen/xgen/interaction/noncontact_sim.py`

## Usage
```python
from xgen.interaction import simulate_noncontact_forward, simulate_noncontact_backward

forward = simulate_noncontact_forward(object_pose, contact_indices, fps)
backward = simulate_noncontact_backward(object_pose, contact_indices, fps, damping=0.05)
```
