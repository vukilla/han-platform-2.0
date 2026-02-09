# XGen Technical Spec

## Pipeline overview
1. Pose extraction + retargeting
2. Contact and non-contact synthesis with physical priors
3. Augmentation

## Module boundaries
- pose/
  - Human pose → SMPL
- retarget/
  - SMPL → robot joint trajectories
- interaction/
  - segment.py (ts/te segmentation)
  - anchors.py (palms midpoint vs single body part)
  - object_init.py (mesh + initial pose)
  - contact_synth.py (anchor-object propagation)
  - force_closure_refine.py
  - noncontact_sim.py (forward/backward sim)
  - stitch.py (smooth interpolation)
- augment/
  - mesh scaling/substitution
  - contact trajectory transforms
  - non-contact velocity randomization
- export/
  - dataset clip writer + contact graph labels

## Critical details
- Phase segmentation: t < ts pre-contact, ts ≤ t ≤ te contact, t > te post-contact.
- Contact phase representation uses anchor definition.
- Object initialization supports manual MVP selection and later estimators.
- Contact graph labels are required for XMimic imitation reward.
- Non-contact synthesis uses forward and backward simulation; reverse sim inverts damping.

## Output format
Each clip exports clip.npz with:
- robot_qpos[t, nq]
- robot_qvel[t, nq]
- root_pose[t, 7]
- object_pose[t, 7]
- contact_graph[t, J]
- phase[t] (enum)
- metadata.json with augmentation tags, demo provenance, ts/te, anchor type
