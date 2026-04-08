# CardSense Master Completion Plan (Near-100% Target)

Last updated: 2026-04-08

## M2 production execution artifacts

- Sprint board: `M2_SPRINT_BOARD.md`
- Jira-style backlog: `M2_JIRA_BACKLOG.md`
- Execution mode: all six sprints defined as deployable increments with mandatory P0 exit gates.

## Execution snapshot (implemented in this pass)

- ✅ Worker: calibrated confidence, quality flags, out-of-order metric, model-native top-k path when exposed.
- ✅ Fusion: explicit score components, entropy gating, provenance reliability weights, calibration-aware seat mapping, occlusion tracking.
- ✅ Game state: strict trick metadata (`trick_id`, open/close timestamps, seat-card map), reason-coded transition rejection, state versioning.
- ✅ Rules: variant-profile-driven legal action logic (`must_trump_if_void`, `trump_suit`, profile fields).
- ✅ Belief: seat evidence ledger and Bayesian update helper, mass/entropy diagnostics.
- ✅ Simulator: deterministic seeded rollouts, policy split hooks, EV decomposition (`ev_now`, `ev_future`, `ev_partner_synergy`, `ev_information_gain`).
- ✅ API: diagnostics, calibration set/get, belief, simulation debug, replay endpoints.
- ✅ Persistence: SQLite-backed `events_raw`, `state_transitions`, `recommendations`.
- ✅ Frontend: table-state, belief, explainability, diagnostics, and calibration workflow panels.
- ✅ Tests: added required unit tests for rules/belief/fusion/transitions/simulator determinism.

This document is the implementation contract for taking CardSense from current MVP-plus to near-production completeness for Mendikot/Thulla assistance.

## 1) Product objective and hard constraints

### 1.1 Objective
Build a real-time game assistant that:
1. Reads two camera streams (`camera_type = hand|table`).
2. Builds a legal, temporally stable game state (`CanonicalGameState`).
3. Recommends best legal move with confidence, rationale, and uncertainty.

### 1.2 Non-negotiable constraints
1. Only player-legal information may be used.
2. Never output certainty when vision/state confidence is low.
3. Recommendation must always include: `best_action`, `confidence`, `why`, `uncertainty_notes`.
4. System must be low-latency and fault-tolerant under dropped/duplicate/stale frames.

---

## 2) Current code inventory (source of truth)

### 2.1 Backend modules
- `backend/app/main.py`
- `backend/app/models.py`
- `backend/app/fusion.py`
- `backend/app/game_state.py`
- `backend/app/rules.py`
- `backend/app/belief.py`
- `backend/app/simulator.py`
- `backend/app/thulla.py`
- `backend/app/config.py`

### 2.2 Worker module
- `worker/worker.py`

### 2.3 Frontend module
- `frontend/app.jsx`

---

## 3) Required tracking model (all entities + variable names)

This section defines exactly what must be tracked continuously.

### 3.1 Per-frame tracking
Track in ingest payload (`FrameIn`):
- `game_id`
- `camera_id`
- `camera_type`
- `frame_id`
- `frame_b64`
- `ts_ms`

Track in worker runtime:
- `last_processed_ts_by_camera: dict[str, int]`
- `stats.processed_frames`
- `stats.skipped_stale`
- `stats.error_frames`
- `stats.frames_with_detections`
- `stats.total_detections`
- `stats.avg_latency_ms`

### 3.2 Per-detection tracking
Track in `Detection`:
- `cls_id`
- `cls_name`
- `conf`
- `card_label_top1`
- `card_conf_top1`
- `card_probs_topk`
- `det_conf`
- `bbox.{x1,y1,x2,y2}`

### 3.3 Per-track tracking (`fusion.Track`)
- `track_id`
- `bbox`
- `last_ts`
- `hits`
- `posterior: Dict[str, float]`
- `det_conf_avg`
- `source_workers: Dict[str, int]`
- `source_models: Dict[str, int]`
- `miss_count`
- `state` (`active|freeze|expire`)

### 3.4 Per-game state tracking (`CanonicalGameState`)
- `game_id`
- `hand_cards`
- `table_trick_cards`
- `played_history`
- `won_piles`
- `turn.leader`
- `turn.to_play`
- `turn.lead_suit`
- `rules.must_follow_suit`
- `rules.trump_mode`
- `uncertainty.hand_conf`
- `uncertainty.table_conf`
- `uncertainty.state_conf`
- `uncertainty.notes`
- `updated_ts_ms`

### 3.5 Per-recommendation tracking
- `best_action.card`
- `best_action.ev`
- `best_action.confidence`
- `alternatives[]`
- `why[]`
- `uncertainty_notes[]`
- `recommendation_mode`
- `freshness_ts_ms`
- `action_scores[]`

Runtime metrics in `main.py`:
- `recommendation_metrics.calls`
- `recommendation_metrics.total_ms`
- `recommendation_metrics.max_ms`

---

## 4) All methods used to help gameplay (pipeline)

### 4.1 Vision inference method (`worker/worker.py`)
1. Decode frame via `decode_b64_to_image`.
2. Multi-pass inference via `predict_with_fallback`:
- pass 0: original
- pass 1: rotate 90
- pass 2: rotate 270
- pass 3: rotate 180
- pass 4: upscaled 2x
3. If all passes fail, run low-threshold probe (`DEBUG_CONF`).
4. Emit event with detections + `debug.quality` + `debug.passes` + `debug.probe_top`.

### 4.2 Ingest and dedupe method (`main.py`)
1. Strict validate `DetectionEvent` with `model_validate(..., strict=True)`.
2. Acquire reducer lock (`reduce_lock:{game_id}`) to serialize reducer path.
3. Dedupe by `frame_id` using Redis key `seen_frame:{game_id}:{frame_id}`.
4. Drop duplicates by setting `event["dedupe_skipped"] = True`.

### 4.3 Fusion method (`fusion.py`)
1. In-frame dedupe by IOU (`frame_dedupe_iou`).
2. Associate detection to track by IOU (`track_match_iou`).
3. Update track posterior using weighted merge:
- prior weight `0.70`
- observation base weight `0.30`
- provenance multipliers from `source_workers` and `source_models`
4. Hysteresis filter:
- require `hits >= hysteresis_hits_threshold` OR `top_prob >= hysteresis_prob_threshold`.
5. Track lifecycle:
- increment `miss_count`
- set `state=freeze` at `freeze_miss_threshold`
- set `state=expire` at `expire_miss_threshold`
6. Cross-camera uniqueness:
- compare same label across hand and table
- keep higher confidence side
- record conflict rows
7. Deck-count guardrail:
- enforce one instance per label in final hand/table snapshot.

### 4.4 Game-state reduction method (`game_state.py`)
1. Compute `transition_score` using:
- hand->table sanity
- table->hand penalty
- table jump penalty (`len(next_table) > len(prev_table)+1`)
2. Reject update if `transition_score < 0.45`.
3. Apply accepted state:
- update `hand_cards`
- update `table_trick_cards`
- derive `lead_suit`
- advance `to_play` based on table length
4. Trick close when `len(table_trick_cards) >= 4`:
- append to `played_history`
- compute winner via `_resolve_trick_winner`
- move cards into `won_piles`
- set next `leader` and `to_play` to winner

### 4.5 Legal action method (`rules.py`)
1. If `turn.to_play != "me"`, return no legal actions.
2. If no lead suit or `must_follow_suit == False`, return all hand cards.
3. If lead suit exists, return only suit-following cards.
4. If void and trump mode is forcing (`must_trump_if_void`/`force_trump_if_void`), return trump suit cards (current default suit `S`).

### 4.6 Belief method (`belief.py`)
1. Build full deck from `RANKS x SUITS`.
2. Remove observed cards (`hand + table + history + won piles`).
3. Assign baseline uniform belief across opponent seats `SEATS = [left, across, right]`.
4. Apply seat-specific off-suit evidence penalties (`seat_void_suit_penalty`).
5. Normalize per-seat distributions.
6. Sample hidden worlds using seat target counts (13 - played_by_seat).

### 4.7 Decision simulation method (`simulator.py`)
1. Enumerate actions using `legal_actions(state)`.
2. Sample worlds with `sample_worlds`.
3. For each action and world:
- estimate trick win vs sampled opponent/partner plays
- estimate partner support
- estimate future value with preserve bias
4. Aggregate:
- `p_win_current_trick`
- `expected_future_trick_value`
- `partner_support_value`
- `risk` (std dev)
- `total_ev`
5. Sort by `total_ev` descending.

### 4.8 Advisor method (`main.py:_build_advisor_response`)
1. Build action scores.
2. Compute confidence:
- `confidence = clamp((1.0 - best.risk) * state.uncertainty.state_conf)`
3. Set mode:
- `normal` if confidence >= 0.60 else `cautious`
4. Compose reasons and uncertainty notes.
5. Return `AdvisorResponse`.

---

## 5) Complete parameter inventory (must be documented and tunable)

### 5.1 Worker env/config parameters
- `REDIS_URL`
- `FRAME_STREAM`
- `DETECTION_STREAM`
- `FRAME_CONSUMER_GROUP`
- `WORKER_NAME`
- `YOLO_MODEL`
- `YOLO_MODEL_URL`
- `MODEL_CACHE_DIR`
- `CONF_THRES`
- `DEBUG_CONF`
- `MAX_FRAME_AGE_MS`
- `SLOW_ACCURATE_MODE`
- `MAX_QUEUE_LEN`
- `WORKER_PROCESSES`
- `WORKER_LOG_STREAM`
- `WORKER_STATUS_PREFIX`
- `WORKER_HEARTBEAT_SEC`
- `MODEL_SMOKE_TEST_URLS`
- `MODEL_SMOKE_MIN_DETECTIONS`

### 5.2 Worker inference parameters (`infer`)
- `conf`
- `imgsz` (`1280` when slow-accurate else `960`)
- `augment`
- `agnostic_nms`
- `max_det` (`100`)
- `verbose` (`False`)

### 5.3 Backend settings (`Settings`)
- `redis_url`
- `frame_stream`
- `detection_stream`
- `frame_consumer_group`
- `backend_consumer_group`
- `backend_consumer_name`
- `max_queue_len`
- `static_dir`
- `hand_size`
- `min_card_conf`
- `hand_memory_events`
- `min_card_votes`
- `card_labels`
- `frame_id_dedupe_ttl_sec`

### 5.4 Fusion parameters (`FusionEngine`)
- `track_match_iou`
- `frame_dedupe_iou`
- `track_stale_ms`
- `hysteresis_hits_threshold`
- `hysteresis_prob_threshold`
- `freeze_miss_threshold`
- `expire_miss_threshold`

### 5.5 Transition parameters (`GameStateEngine`)
- transition reject threshold (`0.45`)
- table progression jump penalty
- table->hand penalty
- no-turn-context penalty

### 5.6 Simulation parameters (`simulator.py`)
- `rollout_count` default (`96`)
- world count (`n=max(24, rollout_count // 2)`)
- `same_suit_bonus = 0.08`
- win uncertainty blend (`0.85` + random `[0.0, 0.15]`)
- partner support baseline (`0.45`) and noise (`+-0.08`)
- preserve bonus (`+0.15` for low card, `-0.05` for high card)
- future noise (`+-0.10`)
- EV weights:
  - `0.55 * p_win_current_trick`
  - `0.30 * expected_future_trick_value`
  - `0.15 * partner_support_value`
  - `-0.20 * risk`

### 5.7 Recommendation thresholds
- mode threshold: `confidence >= 0.60` => `normal`
- cautious triggers:
  - low `state.uncertainty.table_conf`
  - low `state.uncertainty.hand_conf`
  - empty/weak belief mass

---

## 6) Mandatory enhancements to reach near-100%

### 6.1 Vision completeness tasks
1. Implement true top-k extraction from model logits/probs.
- Update worker variable names:
  - `topk_k = 5`
  - `topk_indices`
  - `topk_scores`
  - `card_probs_topk = [[label_i, prob_i], ...]`
2. Add confidence calibration:
- `calibration_temperature`
- `calibrated_card_conf_top1`
3. Add detection quality flags:
- `is_blurry`
- `is_dark`
- `is_overexposed`

### 6.2 Fusion completeness tasks
1. Replace implicit joint scoring with explicit per-candidate fields:
- `vision_conf`
- `temporal_stability`
- `legal_transition_score`
- `uniqueness_score`
- `joint_score`
2. Add posterior entropy gating:
- `posterior_entropy`
- `posterior_entropy_threshold`
3. Add provenance-weight controls:
- `worker_reliability_weight`
- `model_reliability_weight`

### 6.3 Table understanding completeness tasks
1. Add calibration API and persistence:
- `table_homography`
- `seat_regions`
- `trick_region`
- `won_pile_regions`
2. Replace x-order seat assignment with calibrated regions.
3. Add occlusion handling:
- `occluded_tracks`
- `occlusion_timeout_ms`

### 6.4 Game state completeness tasks
1. Enforce strict trick state machine with variables:
- `trick_id`
- `trick_open_ts_ms`
- `trick_close_ts_ms`
- `trick_cards_by_seat`
2. Validate transitions with typed reasons:
- `reason_code`
- `reason_detail`
3. Add state versioning:
- `state_version`
- `last_transition_id`

### 6.5 Rules completeness tasks
1. Formalize variant profile object:
- `variant_name`
- `must_follow_suit`
- `must_trump_if_void`
- `trump_suit`
- `rank_order_map`
2. Ensure `legal_actions` is entirely driven by profile, not hardcoded suit assumptions.

### 6.6 Belief completeness tasks
1. Add seat-level evidence ledger:
- `seat_void_suits[seat]`
- `seat_strength_estimate[seat][suit]`
2. Add Bayesian update function:
- `update_belief_from_play_event(play_event, prior_belief)`
3. Add belief normalization + sanity checks:
- `belief_mass_per_seat`
- `belief_entropy_per_seat`

### 6.7 Simulation completeness tasks
1. Split policies:
- `opponent_policy(world, state)`
- `partner_policy(world, state)`
- `self_policy(action, state)`
2. Add deterministic seeds for repeatability:
- `simulation_seed`
- `rng = random.Random(simulation_seed)`
3. Add richer EV decomposition:
- `ev_now`
- `ev_future`
- `ev_partner_synergy`
- `ev_information_gain`

### 6.8 API completeness tasks
Add/upgrade endpoints:
1. `GET /api/games/{game_id}/diagnostics` with:
- `fusion_metrics`
- `transition_metrics`
- `recommendation_metrics`
- `belief_metrics`
2. `POST /api/games/{game_id}/calibration/table`
3. `GET /api/games/{game_id}/belief`
4. `GET /api/games/{game_id}/simulation/debug`

### 6.9 Frontend completeness tasks
1. Add dedicated table-state panel:
- show `table_trick_cards` by seat
- show `won_piles`
2. Add belief panel:
- top hidden-card probabilities by seat
3. Add recommendation explainability panel:
- per-action EV breakdown and risk bars
4. Add calibration UI:
- mark seat regions and trick/won-pile zones

### 6.10 Storage and replay completeness tasks
1. Persist event log in DB (SQLite/Postgres):
- table `events_raw`
- table `state_transitions`
- table `recommendations`
2. Add replay worker:
- `replay_game(game_id, speed_multiplier)`

---

## 7) Exact file-by-file execution plan

### 7.1 `worker/worker.py`
Implement:
1. True `card_probs_topk` extraction.
2. Calibration and quality flags.
3. Keep existing monotonic per-camera guard (`last_processed_ts_by_camera`) and export count metric:
- `stats.skipped_out_of_order`.

### 7.2 `backend/app/fusion.py`
Implement:
1. Explicit joint-score object per output card.
2. Entropy gating and thresholds.
3. Provenance reliability registry:
- `source_reliability_worker`
- `source_reliability_model`.

### 7.3 `backend/app/game_state.py`
Implement:
1. Strict trick state machine variables.
2. Transition reason codes.
3. Variant-aware trick winner logic using profile.

### 7.4 `backend/app/rules.py`
Implement:
1. Profile-driven legality.
2. Full seat/turn guardrails.
3. Variant test vectors.

### 7.5 `backend/app/belief.py`
Implement:
1. Bayesian evidence update from `played_history` sequence.
2. Seat suit-void probabilities.
3. Belief diagnostics exports.

### 7.6 `backend/app/simulator.py`
Implement:
1. Deterministic seeded rollouts.
2. Policy pluggability.
3. EV decomposition fields.

### 7.7 `backend/app/main.py`
Implement:
1. New diagnostics endpoints.
2. Calibration endpoints.
3. Replay endpoints and persistent event reads.

### 7.8 `frontend/app.jsx`
Implement:
1. Calibration workflow.
2. Belief and EV breakdown visualizations.
3. Transition rejection/conflict panels.

---

## 8) Test plan (must pass for near-100%)

### 8.1 Unit tests
1. `test_rules_legal_actions_variant_profiles`
2. `test_belief_void_suit_update`
3. `test_fusion_uniqueness_conflict_resolution`
4. `test_transition_reject_table_to_hand`
5. `test_simulator_deterministic_seed`

### 8.2 Integration tests
1. Two-camera ingest -> fused state convergence.
2. Duplicate `frame_id` dropped exactly once.
3. Out-of-order frames skipped via monotonic guard.
4. Trick close updates `played_history`, `won_piles`, `leader`, `to_play`.
5. Recommendation endpoint returns non-empty `why` and `uncertainty_notes`.

### 8.3 Performance/SLO tests
1. `p95` recommendation latency < 300 ms (warm).
2. `p99` ingest-to-state latency < 700 ms.
3. Worker stale skip rate acceptable under overloaded input.

### 8.4 Offline replay evaluation
1. Use saved annotated game sequences.
2. Evaluate action recommendation agreement with strong baseline.
3. Track false certainty rate (`normal` mode under low vision confidence should be low).

---

## 9) Metrics contract (required)

Expose from `/api/metrics` and diagnostics:
- `recommendation_latency_ms.avg`
- `recommendation_latency_ms.p95`
- `recommendation_latency_ms.p99`
- `fusion.label_flip_rate`
- `fusion.track_stability_avg`
- `fusion.uniqueness_conflicts`
- `transitions.transition_accept_rate`
- `transitions.illegal_transition_reject_rate`
- `state_conf_avg`
- `hand_conf_avg`
- `table_conf_avg`
- `cautious_mode_rate`

Add counters:
- `metrics_out_of_order_frames_skipped`
- `metrics_deduped_frames`
- `metrics_transition_reject_reason_code[reason]`

---

## 10) Definition of near-100% complete

The app is near-complete when all are true:
1. Real top-k detection probabilities are used end-to-end.
2. Fusion uses explicit joint scoring and entropy-aware gating.
3. Table camera is calibrated to explicit zones.
4. Game state transitions are strict and explainable with reason codes.
5. Legal action generator is variant-profile-driven.
6. Belief model is seat-evidence-driven (not mostly uniform).
7. Simulation is deterministic-capable and policy-pluggable.
8. Recommendation UX shows EV breakdown + uncertainty clearly.
9. Replay and diagnostics are persistent and queryable.
10. SLO + quality metrics meet targets in continuous runs.

---

## 11) Execution order (recommended)

1. Worker top-k + calibration outputs.
2. Fusion explicit joint scoring + entropy gating.
3. Table calibration subsystem.
4. Strict game-state machine + transition reason codes.
5. Variant profile + full legal action engine.
6. Belief Bayesian upgrades.
7. Simulator determinism + richer EV model.
8. Diagnostics/replay persistence.
9. Frontend explainability + calibration UI.
10. Full metric/SLO validation and tuning.

---

## 12) Notes for implementation discipline

1. Keep backward compatibility for API contracts where possible.
2. Every new variable in this doc must be either:
- added to code, or
- explicitly marked deprecated/removed in a changelog section.
3. Do not merge phases without tests for that phase.
4. Track all threshold changes in one location with comments and rationale.

---

## 13) Production-grade plan (same codebase)

This section is the execution plan to make CardSense production-grade on top of the near-100% feature plan above.

### 13.1 Production architecture target

Deploy as 6 logical services:
1. `api-service` (FastAPI web/API + websocket)
2. `reducer-service` (single-writer state reduction loop)
3. `worker-service` (N scalable YOLO inference workers)
4. `redis-service` (stream queue + ephemeral coordination)
5. `db-service` (PostgreSQL, replacing SQLite for prod writes)
6. `frontend-service` (static UI + reverse proxy)

Keep module names but split process roles with env flags:
- `APP_ROLE=api|reducer|all`
- `WORKER_ENABLED=1|0`

### 13.2 Hard production requirements

1. No single point of failure for API layer.
2. Exactly-once-like state application per `(game_id, frame_id)` in reducer path.
3. Durable storage for events/transitions/recommendations.
4. Full authentication and authorization.
5. Full observability (logs + metrics + traces) and alerting.
6. Safe deploy/rollback strategy with zero-downtime targets.

### 13.3 Data durability and consistency plan

Move primary persistence from SQLite to Postgres.

Variables to add:
- `DATABASE_URL`
- `DB_POOL_MIN_SIZE`
- `DB_POOL_MAX_SIZE`
- `DB_STATEMENT_TIMEOUT_MS`
- `DB_CONNECT_TIMEOUT_MS`

Tables (required):
1. `events_raw`
2. `state_transitions`
3. `recommendations`
4. `game_configs`
5. `reducer_offsets`
6. `deploy_audit_log`

Idempotency keys:
- `idempotency_event_key = "{game_id}:{frame_id}"`
- `idempotency_transition_key = "{game_id}:{last_transition_id}"`
- `idempotency_recommendation_key = "{game_id}:{freshness_ts_ms}:{best_action.card}"`

### 13.4 Reducer correctness plan

Reducer must run as dedicated process (`APP_ROLE=reducer`) with strict locking.

Add variables:
- `REDUCER_LOCK_TTL_SEC` (default `5`)
- `REDUCER_LOCK_RETRY_MS` (default `20`)
- `REDUCER_BATCH_SIZE` (default `50`)
- `REDUCER_MAX_EVENT_LAG_MS` (default `2000`)

Required checks:
1. Reject stale event if `event.ts_ms < state.updated_ts_ms - REDUCER_MAX_EVENT_LAG_MS`.
2. Reject duplicate if `idempotency_event_key` already committed.
3. Write transition + recommendation in DB transaction boundary.

### 13.5 Security plan

Add authn/authz to all mutable endpoints.

Variables:
- `AUTH_ENABLED=1`
- `JWT_ISSUER`
- `JWT_AUDIENCE`
- `JWT_PUBLIC_KEY_PEM`
- `API_KEY_HASHES`
- `ADMIN_ROLE_NAME=admin`
- `OPERATOR_ROLE_NAME=operator`

Policies:
1. `POST/DELETE` game/config/calibration/replay endpoints require `operator`.
2. Worker status/log endpoints require authenticated user.
3. WebSocket requires token at connect time.
4. CORS allowlist via `CORS_ALLOWED_ORIGINS`.

### 13.6 Privacy and data governance plan

Variables:
- `STORE_RAW_FRAMES=0|1`
- `FRAME_RETENTION_DAYS`
- `EVENT_RETENTION_DAYS`
- `RECOMMENDATION_RETENTION_DAYS`
- `PII_REDACTION_ENABLED=1`

Policies:
1. Default `STORE_RAW_FRAMES=0` in production.
2. TTL purge jobs for old events/transitions/recommendations.
3. Export/delete by `game_id` for operator compliance workflows.

### 13.7 Observability plan

Metrics backend:
- Prometheus + Grafana (or OpenTelemetry collector).

Variables:
- `METRICS_ENABLED=1`
- `METRICS_NAMESPACE=cardsense`
- `TRACE_ENABLED=1`
- `TRACE_SAMPLE_RATE=0.1`
- `LOG_LEVEL=INFO`
- `LOG_JSON=1`

Required metric names:
1. `cardsense_recommendation_latency_ms_bucket`
2. `cardsense_reducer_event_lag_ms`
3. `cardsense_transition_reject_total{reason_code}`
4. `cardsense_fusion_label_flip_rate`
5. `cardsense_worker_skipped_out_of_order_total`
6. `cardsense_ws_connected_clients`
7. `cardsense_db_write_latency_ms`
8. `cardsense_queue_depth_frames`
9. `cardsense_queue_depth_detections`
10. `cardsense_cautious_mode_rate`

### 13.8 Alerting plan

Create production alerts:
1. `recommendation_latency_p95 > 300ms` for 10m
2. `transition_reject_rate > 0.15` for 15m
3. `worker_error_frames > threshold` for 10m
4. `queue_depth_frames increasing for 10m`
5. `ws_connected_clients drop > 50% in 5m`
6. `db_write_failure_rate > 1%`

### 13.9 Performance and scaling plan

Variables:
- `WORKER_AUTOSCALE_MIN`
- `WORKER_AUTOSCALE_MAX`
- `TARGET_QUEUE_DEPTH_PER_WORKER`
- `FRAME_INGEST_RATE_LIMIT_FPS`
- `WS_BROADCAST_BATCH_SIZE`

Capacity methods:
1. Scale workers on `queue_depth_frames`.
2. Separate API and reducer pods/processes.
3. Apply per-camera ingest rate limiting.
4. Backpressure: reject ingest when queue depth exceeds hard cap.

### 13.10 Deployment and release plan

Use CI/CD with staging gate before production.

Variables:
- `DEPLOY_ENV=dev|staging|prod`
- `RELEASE_VERSION`
- `CANARY_PERCENT`
- `FEATURE_FLAG_*`

Release steps:
1. Build immutable image tagged by git SHA.
2. Run unit + integration + replay regression suite.
3. Deploy canary (5-10% traffic).
4. Watch SLO/alert board for 30-60 min.
5. Roll forward to 100% or rollback automatically.

### 13.11 Feature flags (required)

Add flags:
- `FEATURE_CALIBRATED_SEAT_MAPPING`
- `FEATURE_STRICT_TRANSITION_REJECT`
- `FEATURE_POLICY_SIMULATOR_V2`
- `FEATURE_BELIEF_BAYES_UPDATE`
- `FEATURE_DB_EVENT_PERSISTENCE`
- `FEATURE_REPLAY_API`

Each flag requires:
1. default value
2. owner
3. removal date
4. rollback behavior

### 13.12 Production test matrix

Must pass in CI and staging:
1. Unit tests
2. Integration tests (dual-camera stream)
3. Contract tests (API/WS schemas)
4. Replay regression tests
5. Load tests (normal + overload)
6. Chaos tests (worker crash, Redis restart, DB failover)
7. Security tests (auth bypass, token expiry, role checks)

### 13.13 Runbooks (required)

Create operator runbooks:
1. `RUNBOOK_QUEUE_BACKLOG.md`
2. `RUNBOOK_WORKER_DEGRADATION.md`
3. `RUNBOOK_TRANSITION_REJECT_SPIKE.md`
4. `RUNBOOK_DB_LATENCY_SPIKE.md`
5. `RUNBOOK_REPLAY_DEBUG.md`
6. `RUNBOOK_CANARY_ROLLBACK.md`

Each runbook must include:
1. symptom
2. dashboard links
3. diagnostic commands
4. immediate mitigation
5. permanent fix workflow

### 13.14 Production acceptance gates (go/no-go)

Go-live only if all pass:
1. `recommendation_latency_ms.p95 <= 300` and `p99 <= 700`.
2. `illegal_transition_reject_rate` stable and explainable.
3. 24h soak test with no data-loss in reducer/DB path.
4. Replay determinism: same `simulation_seed` => stable ranked outputs.
5. Security checklist signed (auth, role checks, secret handling).
6. Rollback drill completed successfully in staging.

### 13.15 Final completion criteria (production-grade)

App is production-grade when:
1. Features are complete (Sections 1-12).
2. Reliability, security, observability, and ops controls in this section are implemented.
3. SLOs are met continuously with alert coverage.
4. On-call can diagnose and recover incidents using runbooks.
