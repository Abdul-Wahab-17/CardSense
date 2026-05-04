#!/usr/bin/env bash
# Smoke test for the new /api/blackjack endpoints.
# Runs an end-to-end happy-path round via curl. No camera needed.
#
# Usage: ./scripts/smoke_blackjack.sh
#   Optional env: BACKEND=http://localhost:8000
#                 USERNAME=demo PASSWORD=demo1234

set -euo pipefail

BACKEND="${BACKEND:-http://localhost:8000}"
EMAIL="${EMAIL:-demo@example.com}"
PASSWORD="${PASSWORD:-demo1234}"
GAME_ID="${GAME_ID:-bj-smoke-$RANDOM}"

green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
hdr()   { printf "\n\033[1;36m=== %s ===\033[0m\n" "$*"; }

hdr "1. Health check"
curl -fsS "$BACKEND/api/health" | head -c 200; echo

hdr "2. Register + login"
# Try register; ignore if already exists
curl -fsS -X POST "$BACKEND/api/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\",\"display_name\":\"Smoke Test\"}" \
  >/dev/null 2>&1 || true

LOGIN=$(curl -fsS -X POST "$BACKEND/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}")
TOKEN=$(echo "$LOGIN" | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
green "Got token: ${TOKEN:0:20}..."

AUTH=(-H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json")

hdr "3. Bootstrap operator role (first user is admin by default)"
# Look up our user_id from /api/auth/me, then self-promote (no-op if already admin).
ME=$(curl -fsS "$BACKEND/api/auth/me" "${AUTH[@]}")
USER_ID=$(echo "$ME" | python -c "import sys,json; print(json.load(sys.stdin).get('user_id',''))")
if [ -n "$USER_ID" ]; then
  curl -fsS -X POST "$BACKEND/api/auth/users/$USER_ID/roles" "${AUTH[@]}" \
    -d '{"roles":["operator","admin"]}' >/dev/null 2>&1 || true
fi

hdr "4. Start blackjack game ($GAME_ID)"
curl -fsS -X POST "$BACKEND/api/blackjack/games/$GAME_ID/start" "${AUTH[@]}" \
  -d '{"decks_in_shoe":1,"bet":1.0}' | python -m json.tool

hdr "5. Deal: player A+K (BJ!) vs dealer 10+7"
curl -fsS -X POST "$BACKEND/api/blackjack/games/$GAME_ID/deal" "${AUTH[@]}" \
  -d '{"player_cards":["AS","KH"],"dealer_cards":["10D","7C"]}' | python -m json.tool

hdr "6. State after deal — should show player_blackjack outcome"
curl -fsS "$BACKEND/api/blackjack/games/$GAME_ID/state" | python -m json.tool

hdr "7. New round + recommendation flow"
curl -fsS -X POST "$BACKEND/api/blackjack/games/$GAME_ID/new_round" "${AUTH[@]}" \
  -d '{"bet":1.0}' >/dev/null

curl -fsS -X POST "$BACKEND/api/blackjack/games/$GAME_ID/deal" "${AUTH[@]}" \
  -d '{"player_cards":["10S","6H"],"dealer_cards":["10D","5C"]}' >/dev/null

green "Player 16 vs dealer 10 — recommendation:"
curl -fsS "$BACKEND/api/blackjack/games/$GAME_ID/state?llm=true" | python -m json.tool

hdr "8. Hit, observe a 5 → player auto-stands at 21"
curl -fsS -X POST "$BACKEND/api/blackjack/games/$GAME_ID/observe" "${AUTH[@]}" \
  -d '{"label":"5C","target":"player"}' | python -m json.tool

hdr "9. Dealer hidden card was 5 (15 total, must hit). Observe a 4 → dealer stands at 19"
curl -fsS -X POST "$BACKEND/api/blackjack/games/$GAME_ID/observe" "${AUTH[@]}" \
  -d '{"label":"4D","target":"dealer"}' | python -m json.tool

hdr "10. History"
curl -fsS "$BACKEND/api/blackjack/games/$GAME_ID/history" | python -m json.tool

hdr "11. Tipsi review (LLM — falls back to 'not configured' if no key)"
curl -fsS -X POST "$BACKEND/api/blackjack/games/$GAME_ID/tipsi" "${AUTH[@]}" | python -m json.tool

green "Smoke test complete. Game id: $GAME_ID"
