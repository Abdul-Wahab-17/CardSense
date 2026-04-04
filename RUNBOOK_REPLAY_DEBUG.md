# RUNBOOK: Replay Debug

## Symptom
- Replay output missing events or ordering inconsistent.

## Dashboard Links
- /api/games/{game_id}/events?source=db

## Diagnostic Commands
- Verify events count and timestamps.
- Check reducer offsets table for last committed message.

## Immediate Mitigation
1. Re-run replay with lower speed_multiplier.
2. Inspect event persistence flag (FEATURE_REPLAY_API=1).

## Permanent Fix Workflow
- Validate idempotency keys and event ordering.
- Review reducer offset updates.
