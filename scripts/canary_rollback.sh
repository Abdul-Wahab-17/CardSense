#!/usr/bin/env sh
set -eu

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
  echo ".env not found; create it from .env.example first"
  exit 1
fi

if grep -q '^CANARY_PERCENT=' "$ENV_FILE"; then
  sed -i 's/^CANARY_PERCENT=.*/CANARY_PERCENT=0/' "$ENV_FILE"
else
  printf '\nCANARY_PERCENT=0\n' >> "$ENV_FILE"
fi

echo "Canary rolled back: CANARY_PERCENT=0"
echo "Next step: redeploy and validate rollback runbook checks"
