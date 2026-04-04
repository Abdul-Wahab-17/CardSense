# RUNBOOK: DB Latency Spike

## Symptom
- cardsense_db_write_latency_ms increasing; persistence lag.

## Dashboard Links
- /api/metrics: cardsense_db_write_latency_ms

## Diagnostic Commands
- Inspect DB connection pool, storage metrics.

## Immediate Mitigation
1. Increase DB pool size.
2. Reduce write volume (disable raw frames, lower retention).
3. Temporarily disable persistence (FEATURE_DB_EVENT_PERSISTENCE=0).

## Permanent Fix Workflow
- Move to managed Postgres with tuned indexes.
- Batch writes where possible.
