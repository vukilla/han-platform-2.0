# Quality & Validation Spec

## Automated checks (MVP)
- Completeness: missing frames, corrupted files
- Physical plausibility
  - Object teleport spikes
  - Joint limit violations
  - Impossible contact transitions
- Task outcome signals

## Validator workflow
- Admin review queue
- Approve/reject
- Override score
- Signed validator decision stored in DB

## Rewards (MVP)
- Points-based
  - +X for accepted demo
  - +Y for clips used in training
  - +Z for clips that improve eval metrics

## Future hardening
- Token staking/slashing
- Multi-validator consensus
- Reputation-weighted scoring
