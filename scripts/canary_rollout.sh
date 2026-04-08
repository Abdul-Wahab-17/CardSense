#!/usr/bin/env sh
set -eu

# Usage: ./scripts/canary_rollout.sh [percent]
TARGET_PERCENT="${1:-10}"

if [ "$TARGET_PERCENT" -lt 0 ] || [ "$TARGET_PERCENT" -gt 100 ]; then
  echo "canary percent must be between 0 and 100"
  exit 1
fi

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
  echo ".env not found; create it from .env.example first"
  exit 1
fi

if grep -q '^CANARY_PERCENT=' "$ENV_FILE"; then
  sed -i "s/^CANARY_PERCENT=.*/CANARY_PERCENT=${TARGET_PERCENT}/" "$ENV_FILE"
else
  printf "\nCANARY_PERCENT=%s\n" "$TARGET_PERCENT" >> "$ENV_FILE"
fi

echo "Set CANARY_PERCENT=${TARGET_PERCENT} in ${ENV_FILE}"
echo "Next step: redeploy backend and verify /api/features + alert baseline"
