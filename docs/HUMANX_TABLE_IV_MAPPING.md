# HumanX Table IV Mapping (Observations + Rewards)

Source of truth: Table IV from the HumanX paper PDF.
Recommended local path (not committed to git): `docs/references/HumanX.pdf`

Rendered screenshot used for this mapping (not committed to git):
- `tmp/pdfs/humanx/obs_rewards_zoom.png`

Implementation entrypoints:
- Observations: `services/xmimic/xmimic/obs_pipeline.py`
- Rewards: `services/xmimic/xmimic/rewards/unified.py`

## Observations (Table IV)

Table IV separates **Current Terms** from **History Terms**. In code, that maps to:
- `ObservationConfig.fields`: current terms (ordered exactly)
- `ObservationConfig.history_fields`: history terms (ordered exactly)
- `ObservationConfig.history = H`: number of frames stacked

The observation vector emitted by `ObservationPipeline` is:
`[current_terms] + concat_{t-H..t-1}([history_terms])`

### Canonical field names

Table IV uses these rows. We use the same names in the observation schema.

Notes:
- `gravity`: in practice this is typically **projected gravity in the base frame**. We keep the field named `gravity` but accept `projected_gravity` as an alias.
- `action`: in Table IV this is commonly the **previous action** fed back as an input. We keep the field named `action` but accept `prev_action` as an alias.
- `pd_error`: we treat this as a per-DoF PD error proxy. We keep the field named `pd_error` but accept `dof_pos_error` as an alias.

### Current Terms

Teacher (`pi_tea`) vs Student (`pi_stu`) parity, line-by-line:
- `base_ang_vel`: teacher yes, student yes
- `gravity`: teacher yes, student yes
- `dof_pos`: teacher yes, student yes
- `dof_vel`: teacher yes, student yes
- `action`: teacher yes, student yes
- `pd_error`: teacher yes, student yes
- `ref_body_pos`: teacher yes, student no
- `delta_body_pos`: teacher yes, student no
- `object_pos`: teacher yes, student optional (`*`)
- `object_rot`: teacher optional (`*`), student optional (`*`)
- `target_object_pos`: teacher optional (`*`), student optional (`*`)
- `target_object_rot`: teacher optional (`*`), student optional (`*`)
- `skill_label`: teacher no (per-skill teacher policy), student yes
- `history`: teacher yes, student yes (implemented via `history_fields` stacking, not a standalone scalar)

### History Terms

History terms are identical for teacher and student:
- `base_ang_vel`
- `gravity`
- `dof_pos`
- `dof_vel`
- `action`

### Enforced schemas (teacher vs student, NEP vs MoCap)

We encode the Table IV schemas as explicit builders:
- `humanx_teacher_obs_config(...)`
- `humanx_student_obs_config(..., mode="nep"|"mocap")`

Expected behavior:
- **Teacher** always includes `object_pos` (privileged state).
- **Teacher** does **not** include `skill_label` in Table IV (one teacher per skill). This repo keeps an opt-in `include_skill_label=True` for experiments.
- **Student NEP** includes no object observations.
- **Student MoCap** includes `object_pos`, and can optionally include `object_rot` (Table IV marks it optional).

Dropout behavior:
- Table IV does not show an `object_valid` mask.
- In this repo, the schema excludes a mask by default; dropout should be implemented by the environment (e.g., staling/holding last MoCap sample).
- `object_valid` remains available only as an opt-in debug hook (`mocap_dropout_mask=True`).

Exact ordering is enforced by unit tests (`services/xmimic/tests/test_obs_pipeline.py`).

For `services/xmimic/configs/robot_spec.yaml` (DoF=24, key bodies=5), the observation dimensions are:
- Teacher: `135 + 78 * history` (+4 if `object_rot` is enabled; +3/+4 for target object pose)
- Student NEP: `(102 + num_skills) + 78 * history`
- Student MoCap: `(105 + num_skills) + 78 * history` (+4 if `object_rot` is enabled)

## Rewards (Table IV)

Table IV lists rewards as:
- Mimic terms (applied to teacher and student)
- Regularization terms (applied to teacher and student)

In code (`HumanXRewardConfig`), these map to:

### Mimic Terms (`r_body`, `r_obj`, `r_rel`, `r_c`)

All are implemented in `services/xmimic/xmimic/rewards/unified.py`.

Table IV row -> reward key:
- Body Pos -> `body_pos`
- Body Rot -> `body_rot`
- Body Vel -> `body_vel`
- Body Ang Vel -> `body_ang_vel`
- Body AMP -> `amp` (optional hook; env must provide AMP score)
- DoF Pos -> `dof_pos`
- DoF Vel -> `dof_vel`
- Relative Motion -> `relative_pos` (+ optional `relative_rot`)
- Object Pos -> `object_pos`
- Object Rot (`*`) -> `object_rot`
- Contact Graph -> `contact`

Relative motion (paper Eq. 10) requires vectors `u_t` from key bodies to the object position.
Implementation detail:
- If `relative_pos` is not provided, we derive it automatically when `key_body_pos` (preferred) or `body_pos` and `object_pos` are present:
  - `u = key_body_pos - object_pos` (or `body_pos - object_pos` fallback)
  - `u_ref` computed from the corresponding reference signals.

Error metric parity:
- Paper Eq. 10 uses a **squared** L2 norm: `||u - u_hat||_2^2` (averaged across key bodies, and time if present).

### Reg Terms (`r_reg`)

Table IV row -> suggested `reg_terms` key -> expected `obs` key:
- Torque -> `torque` -> `torque`
- Action Rate -> `action_rate` -> `action` + `prev_action`
- Waist DoF -> `waist_dof` -> `waist_dof`
- Feet Orientation -> `feet_orientation` -> `feet_orientation` or `feet_orientation_error`
- Feet Slippage -> `feet_slippage` -> `feet_slippage` or `feet_slip`
- DoF Limit -> `dof_limit` -> `dof_limit` or `dof_limit_violation`
- Torque Limit -> `torque_limit` -> `torque_limit` or `torque_limit_violation`
- Termination -> `termination` -> `termination`

All reg terms are optional, but schema parity requires we support the full set even if an early environment stub does not emit all signals yet.

## Tests (schema enforcement)

Unit tests that enforce the Table IV mapping:
- `services/xmimic/tests/test_obs_pipeline.py`
- `services/xmimic/tests/test_rewards_unified.py`
