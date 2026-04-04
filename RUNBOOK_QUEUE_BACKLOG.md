# RUNBOOK: Queue Backlog

## Symptom
- Frame queue depth steadily increasing; latency rising.

## Dashboard Links
- /api/metrics: cardsense_queue_depth_frames, cardsense_queue_depth_detections

## Diagnostic Commands
- Check queue depth via /api/metrics
- Inspect worker status via /api/workers/status

## Immediate Mitigation
1. Scale worker processes/pods.
2. Reduce ingest rate (FRAME_INGEST_RATE_LIMIT_FPS).
3. Temporarily drop frames (increase backpressure).

## Permanent Fix Workflow
- Increase worker capacity or tune model speed.
- Verify consumer group health and Redis throughput.
