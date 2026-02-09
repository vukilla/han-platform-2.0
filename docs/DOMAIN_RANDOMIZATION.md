# Domain Randomization + External Pushes

`xmimic.generalization` now supports:
- parameter randomization via `domain_randomization`
- scheduled external forces via `should_apply_external_force` + `sample_external_force`

The PPO loop will call `env.apply_external_force(...)` if the environment exposes it.
