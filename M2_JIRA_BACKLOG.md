# M2 Jira-Style Backlog (One-Go Execution)

Last updated: 2026-04-08

## How to use

- Issue key format: `M2-Sx-yyy` where x is sprint number
- Story points are approximate and sized for 2-week sprints
- Every sprint must finish all P0 + pass its exit gate

## Sprint 1 (State correctness)

- `M2-S1-001` (8 SP): Implement reconcile and transition event application in reducer path.
  - Depends on: none
  - Files: `backend/app/match_engine.py`, `backend/app/main.py`
- `M2-S1-002` (5 SP): Wire reconciliation config variables into runtime logic.
  - Depends on: `M2-S1-001`
  - Files: `backend/app/config.py`, `backend/app/fusion.py`, `backend/app/game_state.py`
- `M2-S1-003` (3 SP): Replace trick close constant with `variant.trick_size`.
  - Depends on: `M2-S1-001`
  - Files: `backend/app/match_engine.py`, `backend/app/game_state.py`
- `M2-S1-004` (5 SP): Normalize turn progression using `play_direction` and `seat_order`.
  - Depends on: `M2-S1-003`
  - Files: `backend/app/match_engine.py`, `backend/app/game_state.py`
- `M2-S1-005` (5 SP): Collapse lifecycle truth to one source.
  - Depends on: `M2-S1-001`
  - Files: `backend/app/main.py`, `backend/app/match_engine.py`, `backend/app/game_state.py`
- `M2-S1-006` (3 SP): Drift diagnostics canonical vs match state.
  - Depends on: `M2-S1-005`
- `M2-S1-007` (3 SP): Transition reject counters by reason code/card.
  - Depends on: `M2-S1-001`
- `M2-S1-008` (2 SP): Replay-safe reconciliation flag (`allow_won_pile_replay`).
  - Depends on: `M2-S1-001`

## Sprint 2 (Two-camera production flow)

- `M2-S2-001` (5 SP): Enforce camera binding before ingest (flag-gated).
  - Depends on: `M2-S1-005`
  - Files: `backend/app/main.py`, `backend/app/models.py`
- `M2-S2-002` (5 SP): Add camera telemetry computation and API exposure.
  - Depends on: `M2-S2-001`
  - Files: `backend/app/main.py`
- `M2-S2-003` (8 SP): Frontend bind/calibration/readiness UX flow.
  - Depends on: `M2-S2-001`
  - Files: `frontend/app.jsx`
- `M2-S2-004` (2 SP): Send target_model_type/device_id/session_id from edge client.
  - Depends on: none
  - Files: `scripts/edge_camera_client.py`
- `M2-S2-005` (3 SP): Stale-camera cautious recommendation blocker.
  - Depends on: `M2-S2-002`
- `M2-S2-006` (3 SP): Setup diagnostics summary endpoint.
  - Depends on: `M2-S2-002`
- `M2-S2-007` (2 SP): Auto-rebind recovery on reconnect.
  - Depends on: `M2-S2-003`

## Sprint 3 (Auth/session hardening)

- `M2-S3-001` (5 SP): Refresh flow role load from DB.
  - Depends on: none
  - Files: `backend/app/auth_store.py`, `backend/app/main.py`
- `M2-S3-002` (5 SP): Harden token claims (`jti`, strict `exp/iat`, `iss`, `aud`).
  - Depends on: `M2-S3-001`
  - Files: `backend/app/security.py`
- `M2-S3-003` (3 SP): Production secret policy startup guard.
  - Depends on: `M2-S3-002`
  - Files: `backend/app/config.py`, `backend/app/main.py`
- `M2-S3-004` (8 SP): Frontend auth flows + headers + WS token.
  - Depends on: `M2-S3-002`
  - Files: `frontend/app.jsx`
- `M2-S3-005` (5 SP): Logout revocation checks API + WS reconnect.
  - Depends on: `M2-S3-002`
  - Files: `backend/app/main.py`, `backend/app/security.py`
- `M2-S3-006` (3 SP): Account status transitions active/blocked.
  - Depends on: `M2-S3-001`
  - Files: `backend/app/auth_store.py`
- `M2-S3-007` (3 SP): Brute-force guardrails.
  - Depends on: `M2-S3-001`
- `M2-S3-008` (2 SP): Admin role-assignment endpoint.
  - Depends on: `M2-S3-001`

## Sprint 4 (Distributed inference)

- `M2-S4-001` (8 SP): Real model artifact resolution for filesystem/http/s3.
  - Depends on: none
  - Files: `worker/worker.py`
- `M2-S4-002` (3 SP): Checksum validation before serving.
  - Depends on: `M2-S4-001`
  - Files: `worker/worker.py`
- `M2-S4-003` (8 SP): Strict hand/table route split + filter/requeue.
  - Depends on: `M2-S2-004`
  - Files: `backend/app/main.py`, `worker/worker.py`
- `M2-S4-004` (3 SP): Truthful worker capability publication.
  - Depends on: `M2-S4-001`
  - Files: `worker/worker.py`
- `M2-S4-005` (3 SP): Fetch retry/backoff and startup fail-fast reason codes.
  - Depends on: `M2-S4-001`
- `M2-S4-006` (3 SP): Capability-aware queue depth metrics.
  - Depends on: `M2-S4-004`
- `M2-S4-007` (2 SP): Warm standby worker role.
  - Depends on: `M2-S4-004`

## Sprint 5 (Recommendation v2)

- `M2-S5-001` (13 SP): Implement full recommendation v2 pipeline.
  - Depends on: `M2-S1-005`
  - Files: `backend/app/rules.py`, `backend/app/belief.py`, `backend/app/simulator.py`, `backend/app/main.py`
- `M2-S5-002` (5 SP): Add `rule_validity_score`, `state_stability_score`, `info_value_score`.
  - Depends on: `M2-S5-001`
- `M2-S5-003` (3 SP): Enforce legal-only recommended output invariant.
  - Depends on: `M2-S5-001`
- `M2-S5-004` (3 SP): Variant policy packs and simulation settings.
  - Depends on: `M2-S5-001`
- `M2-S5-005` (5 SP): Confidence calibration from live outcomes.
  - Depends on: `M2-S5-002`
- `M2-S5-006` (2 SP): Explanation templates by uncertainty condition.
  - Depends on: `M2-S5-001`

## Sprint 6 (Production launch readiness)

- `M2-S6-001` (5 SP): SLO alerting for latency/reject/stale/queue lag.
  - Depends on: `M2-S2-002`, `M2-S5-001`
  - Files: `ops/alerts.yml`, `ops/prometheus.yml`
- `M2-S6-002` (2 SP): Runbook-linked alert metadata.
  - Depends on: `M2-S6-001`
- `M2-S6-003` (5 SP): Canary rollout + rollback automation.
  - Depends on: `M2-S6-001`
  - Files: `scripts/start.sh`, deploy pipeline configs
- `M2-S6-004` (5 SP): Data retention/privacy/backup-restore drills.
  - Depends on: `M2-S3-003`
- `M2-S6-005` (3 SP): Cost dashboards and autoscale tuning.
  - Depends on: `M2-S6-001`
- `M2-S6-006` (3 SP): Chaos tests for redis/worker/stream delays.
  - Depends on: `M2-S6-001`
  - Files: `tests/test_chaos_placeholder.py`
- `M2-S6-007` (2 SP): Post-launch experiment framework.
  - Depends on: `M2-S6-004`

## Always-on board

- `M2-ALWAYS-001` (P0): backward compatibility for existing ingest clients.
- `M2-ALWAYS-002` (P0): block silent state corruption.
- `M2-ALWAYS-003` (P0): block security regressions at CI.
- `M2-ALWAYS-004` (P1): metrics parity for all new features.
- `M2-ALWAYS-005` (P1): docs updates (`.env.example`, endpoint contract, runbooks).
- `M2-ALWAYS-006` (P2): UI polish and convenience tooling.

## Dependency summary

1. Sprint 1 is a hard prerequisite for sprints 2 and 5.
2. Sprint 3 can run in parallel with sprint 2 after sprint 1 lifecycle unification.
3. Sprint 4 depends on sprint 2 frame contract readiness.
4. Sprint 6 depends on telemetry from sprint 2 and recommendation v2 outputs from sprint 5.

## Exit-gate checklist template

Use this checklist at sprint close:

- [ ] All sprint P0 issues done
- [ ] Exit-gate tests passing
- [ ] Backward compatibility checks passing
- [ ] Security regression suite passing
- [ ] Metrics and docs updated
- [ ] Staging deploy completed
- [ ] Canary verify passed
