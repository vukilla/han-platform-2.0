# User Flows

## Flow A: Upload demo → annotate → XGen dataset
1. User creates a project.
2. Uploads monocular video demo.
3. Annotates contact start/end and anchor type.
4. Runs XGen pipeline.
5. Downloads dataset with preview clips.

## Flow B: Demo → XGen → XMimic training
1. Follow Flow A through dataset creation.
2. Start XMimic training (teacher + student).
3. View evaluation report and download checkpoint.

## Flow C (Later): Teleop/sim gameplay
1. Teleop or simulator collects episodes.
2. Auto-upload episodes to project.
3. Validator review and quality scoring.
4. Rewards issued to contributors.
