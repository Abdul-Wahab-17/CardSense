# M2 Production Sprint Board (Deployable Every 2 Weeks)

Last updated: 2026-04-08

## Delivery setup

- Sprint length: 2 weeks
- Branch model: `m2` feature branches -> `m2` integration -> staging -> canary
- Rule: every sprint must ship P0 items and pass exit gate before closing

## Sprint 1: State correctness becomes authoritative

Goal: make game truth deterministic before adding UX/security complexity.

### P0

1. Implement `reconcile_card_locations()` and `apply_transition_events()` in `backend/app/match_engine.py` and wire from reducer path in `backend/app/main.py`.
2. Activate config variables in runtime logic:
   - `hand_occlusion_ttl_ms`
   - `table_occlusion_ttl_ms`
   - `hand_to_table_grace_ms`
   - `max_transition_gap_ms`
   - `max_table_growth_per_event`
3. Replace hardcoded trick close (`4`) with `variant.trick_size`.
4. Use `play_direction` and `seat_order` for all turn progression.
5. Make one lifecycle source of truth (no parallel divergent lifecycle maps).

### P1

1. Add drift diagnostics: canonical state vs match state.
2. Add transition reject reason counters by code and card.

### P2

1. Add replay-safe reconciliation mode (`allow_won_pile_replay` flag).

### Exit gate

- `0` illegal `won_pile -> hand/table` transitions in tests.
- All trick/turn tests pass for clockwise and counterclockwise.
- Reducer still works with existing clients.

## Sprint 2: Two-camera production flow

Goal: make phone+table setup reliable for real users.

### P0

1. Enforce camera binding (`/api/games/{id}/cameras/bind`) before ingest when feature flag is on.
2. Compute and expose camera telemetry fields:
   - `camera_online`
   - `last_frame_ts_ms`
   - `frame_rate_actual`
   - `latency_ms_p50`
   - `latency_ms_p95`
   - `drop_rate`
3. Add bind + calibration + readiness flow in `frontend/app.jsx`.
4. Extend edge client (`scripts/edge_camera_client.py`) to send:
   - `target_model_type`
   - `device_id`
   - `session_id`

### P1

1. Add stale-camera blocker: recommendation becomes `cautious` if one camera is stale.
2. Add setup diagnostics endpoint summary for onboarding.

### P2

1. Auto-rebind recovery on reconnect.

### Exit gate

- Two devices, same `game_id`, stable state convergence in soak test (30 min).
- No frame accepted for unbound camera when flag enabled.
- Camera telemetry updates in near real-time.

## Sprint 3: Auth and session hardening

Goal: secure real-user access and role correctness.

### P0

1. Fix refresh flow to load roles from DB (no hardcoded operator role).
2. Add token claims hardening in `backend/app/security.py`:
   - `jti`
   - strict `exp/iat`
   - issuer/audience checks
3. Enforce production secret policy:
   - fail startup in prod if default HS secret is used.
4. Add frontend login + token storage + Authorization headers + WS token.
5. Add logout revocation checks across API + WS reconnect.

### P1

1. Add account status transitions (`active/blocked`) in `backend/app/auth_store.py`.
2. Add brute-force guardrails (rate limit / cooldown).

### P2

1. Add admin endpoint for role assignment.

### Exit gate

- Register/login/refresh/logout/me all pass integration tests.
- Blocked user cannot refresh or access protected routes.
- WS rejects missing/invalid token when auth is enabled.

## Sprint 4: Multi-machine inference done properly

Goal: make distributed worker routing + remote model use real, not placeholder.

### P0

1. Implement real `resolve_model_artifact()` in `worker/worker.py` for:
   - `MODEL_PROVIDER=filesystem|http|s3`
   - `MODEL_REF_HAND`
   - `MODEL_REF_TABLE`
   - `MODEL_REGISTRY_URL`
   - `MODEL_REGISTRY_TOKEN`
   - `MODEL_SHA256_HAND`
   - `MODEL_SHA256_TABLE`
2. Verify checksum before worker enters serving loop.
3. Split routing into real processing paths:
   - stream-per-role (`frames_hand`, `frames_table`) or equivalent strict filter+requeue policy.
4. Publish truthful worker capabilities:
   - `supports_hand`
   - `supports_table`
   - `model_version`

### P1

1. Add model fetch retry/backoff and startup fail-fast reason codes.
2. Add capability-aware queue depth metrics.

### P2

1. Add warm standby worker role.

### Exit gate

- Remote model artifact fetch + checksum verification tested.
- Hand/table workers process only intended traffic.
- Mixed worker fleet scales under load without misrouting.

## Sprint 5: Recommendation v2 quality + explainability

Goal: recommendations become rule-correct and dependable.

### P0

1. Implement v2 pipeline:
   - `compute_legal_actions(match_state, seat)`
   - `build_belief_state(match_state, history, config)`
   - `sample_hidden_worlds_v2(...)`
   - `simulate_action_batch(...)`
   - `score_actions_v2(...)`
   - `select_recommendation(...)`
   - `explain_recommendation(...)`
2. Add new scores:
   - `rule_validity_score`
   - `state_stability_score`
   - `info_value_score`
3. Guarantee returned actions are always legal.

### P1

1. Add per-variant policy packs and simulation settings.
2. Add confidence calibration from live outcome telemetry.

### P2

1. Add explanation templates by uncertainty condition.

### Exit gate

- `0%` illegal recommended actions in test corpus.
- Deterministic output for fixed seed.
- Confidence degrades correctly on low state quality.

## Sprint 6: Production readiness and launch

Goal: operationally ready for real users.

### P0

1. SLOs and alerting:
   - recommendation latency p95
   - transition reject spike
   - stale camera rate
   - queue lag
2. Add runbook-linked alert metadata.
3. Canary rollout + rollback automation.
4. Data retention + privacy checks + backup/restore drills.

### P1

1. Cost dashboards and autoscale tuning.
2. Chaos tests: redis restart, worker crash, delayed stream.

### P2

1. Post-launch experiment framework.

### Exit gate

- 7-day staging soak with no Sev-1.
- Canary success criteria met.
- Rollback drill executed successfully.

## Cross-sprint priority board (always-on)

### P0 always

1. Backward compatibility for existing frame ingest clients.
2. No silent state corruption.
3. Security regressions blocked at CI.

### P1 always

1. Metrics parity for new features.
2. Documentation updates (`.env.example`, endpoint contract, runbooks).

### P2 always

1. UI polish and convenience tools.

## Production-progress scorecard per sprint

- After Sprint 1: `~75%` (core correctness)
- After Sprint 2: `~82%` (real two-camera usability)
- After Sprint 3: `~88%` (real-user security)
- After Sprint 4: `~93%` (true distributed inference)
- After Sprint 5: `~96%` (high-quality recommendations)
- After Sprint 6: `~99%` (operational production readiness)
