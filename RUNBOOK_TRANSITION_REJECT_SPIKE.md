# RUNBOOK: Transition Reject Spike

## Symptom
- illegal_transition_reject_rate spikes above baseline.

## Dashboard Links
- /api/metrics: transitions.illegal_transition_reject_rate
- /api/games/{game_id}/diagnostics

## Diagnostic Commands
- Inspect recent events via /api/games/{game_id}/events
- Check transition reasons in transition metrics.

## Immediate Mitigation
1. Verify camera alignment and detection quality.
2. Temporarily disable strict rejection (FEATURE_STRICT_TRANSITION_REJECT=0).

## Permanent Fix Workflow
- Tune fusion/transition thresholds.
- Improve calibration for table/seat regions.
