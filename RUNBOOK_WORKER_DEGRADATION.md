# RUNBOOK: Worker Degradation

## Symptom
- Rising worker error frames, no detections, or frequent stale skips.

## Dashboard Links
- /api/workers/status
- /api/workers/logs

## Diagnostic Commands
- Inspect worker logs for model errors.
- Validate model files and smoke-test URLs.

## Immediate Mitigation
1. Restart unhealthy worker pods.
2. Switch to known-good model weights.
3. Lower CONF_THRES to recover detections.

## Permanent Fix Workflow
- Fix model download issues and update model cache strategy.
- Add monitoring for YOLO init failures.
