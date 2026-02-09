# Data Schema

## Core tables

### users
- id
- email
- name
- created_at

### projects
- id
- name
- description
- owner_id
- created_at

### demos
- id
- project_id
- uploader_id
- video_uri
- fps
- duration
- robot_model
- object_id
- status

### demo_annotations
- demo_id
- ts_contact_start
- ts_contact_end
- anchor_type
- key_bodies[]
- notes

### xgen_jobs
- id
- demo_id
- status
- started_at
- finished_at
- params_json
- logs_uri
- error

### datasets
- id
- project_id
- source_demo_id
- version
- status
- summary_json

### dataset_clips
- dataset_id
- clip_id
- uri_npz
- uri_preview_mp4
- augmentation_tags[]
- stats_json

### xmimic_jobs
- id
- dataset_id
- mode (nep|mocap)
- status
- params_json
- logs_uri

### policies
- id
- xmimic_job_id
- checkpoint_uri
- exported_at
- metadata_json

### eval_runs
- id
- policy_id
- env_task
- sr
- gsr
- eo
- eh
- report_uri
- videos_uri

### quality_scores
- entity_type (demo|clip|dataset)
- entity_id
- score
- breakdown_json
- validator_status

### reward_events
- user_id
- entity_type
- entity_id
- points
- reason
- created_at
