# RUNBOOK: Canary Rollback

## Symptom
- Canary deployment violates SLOs or causes regressions.

## Dashboard Links
- SLO panels and /api/metrics

## Diagnostic Commands
- Compare metrics between canary and stable.

## Immediate Mitigation
1. Set CANARY_PERCENT=0.
2. Roll back to previous release image.

## Permanent Fix Workflow
- Identify regression, add test coverage, and re-deploy.
