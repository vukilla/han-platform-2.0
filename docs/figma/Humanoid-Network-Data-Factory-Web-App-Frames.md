# Humanoid Network - Web App (Figma Frames)

## Frame list (in order)
1. Landing
2. Auth
3. Dashboard
4. New Demo Wizard (multi-step)
5. Job Progress Page
6. Dataset Detail
7. Policy Training Page (XMimic)
8. Evaluation Report Page
9. Contributor / Rewards Page

## Design system (minimum)
- Buttons: primary, outline, ghost
- Forms: input, select, file upload
- Stepper: horizontal and vertical
- Status badges: queued/running/completed/failed
- Timeline component
- Metric cards
- Tables: clip list, reward ledger

## Notes
- Each frame maps to an API endpoint in docs/notion/API-Contract-OpenAPI.yaml
- Each job stage maps to a UI badge/state
- Each artifact (dataset, preview, checkpoint, report) requires a DB record and storage URI
