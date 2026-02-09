# Runtime Safety

Safety is non-negotiable for real robot deployment. This repo includes a minimal safety scaffold
that you can connect to your real-time runtime loop.

## Files
- `services/xmimic/xmimic/deploy/safety.py`

## Features
- Torque limit clipping
- Joint limit checks
- Fall detection (pitch/roll threshold)
- Watchdog timeout

## Next steps
- Wire into `RuntimeLoop` before sending torques.
- Add hardware E-stop integration (GPIO / SDK).
- Add staged test checklist (harness, spotter, low-torque mode).
