# Interaction Termination (IT)

The IT heuristic terminates episodes during **contact frames** when the
object-to-key-body relative error exceeds a threshold, with some probability.

## API
- `xmimic.generalization.interaction_termination(relative_error, config, in_contact=True)`
- `xmimic.generalization.contact_relative_error(object_pos, key_body_pos, ref_relative)`

## PPO integration
`train/ppo.py` checks `info["relative_error"]` and optional `info["in_contact"]` to
terminate episodes when contact imitation degrades.
