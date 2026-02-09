# Inverse-Dynamics Contact Inference

We estimate external torques/forces using inverse dynamics:

```
tau_ext = tau_cmd - (M qddot + C qdot + G)
```

Implementation: `services/xmimic/xmimic/contacts/inverse_dynamics.py`

## Usage
```python
from xmimic.contacts import compute_external_torques, solve_contact_forces

res = compute_external_torques(urdf_path, q, qdot, qddot, tau_cmd)
forces = solve_contact_forces(urdf_path, q, qdot, qddot, tau_cmd, contact_jacobian)
```
