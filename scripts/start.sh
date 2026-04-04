#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WORKERS="${WORKERS:-2}"
ENABLE_SPLIT_REDUCER="${ENABLE_SPLIT_REDUCER:-0}"
START_EDGE_CLIENT="${START_EDGE_CLIENT:-0}"
CAMERA_ID="${CAMERA_ID:-cam-1}"
CAMERA_SOURCE="${CAMERA_SOURCE:-0}"
CAMERA_FPS="${CAMERA_FPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
MODEL_URL="${YOLO_MODEL_URL:-https://raw.githubusercontent.com/noorkhokhar99/Playing-Cards-Detection-with-YoloV8/main/yolov8s_playing_cards.pt}"
MODEL_PATH="$ROOT_DIR/models/playing_cards_best.pt"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but not installed."
  exit 1
fi

mkdir -p "$ROOT_DIR/models"
if [ ! -s "$MODEL_PATH" ]; then
  echo "Model not found. Downloading to $MODEL_PATH"
  curl -L --fail --retry 2 "$MODEL_URL" -o "$MODEL_PATH"
fi

echo "Starting services with $WORKERS worker container(s)..."
SERVICES="$(docker compose config --services)"

if echo "$SERVICES" | grep -qx "worker-service"; then
  WORKER_SERVICE="worker-service"
  CORE_SERVICES=(redis db api-service worker-service frontend-service)
  if [ "$ENABLE_SPLIT_REDUCER" = "1" ]; then
    CORE_SERVICES+=(reducer-service)
  fi
elif echo "$SERVICES" | grep -qx "worker"; then
  WORKER_SERVICE="worker"
  CORE_SERVICES=(redis backend worker)
else
  echo "Could not find worker service in docker compose config."
  echo "Available services:"
  echo "$SERVICES"
  exit 1
fi

docker compose up --build -d "${CORE_SERVICES[@]}" --scale "${WORKER_SERVICE}=$WORKERS"

echo "Waiting for backend health at $BACKEND_URL/api/health ..."
for _ in $(seq 1 60); do
  if curl -fsS "$BACKEND_URL/api/health" >/dev/null 2>&1; then
    echo "Backend is healthy."
    break
  fi
  sleep 1
done

if [ "$START_EDGE_CLIENT" = "1" ]; then
  if ! command -v python >/dev/null 2>&1; then
    echo "Python is required to run edge client."
    exit 1
  fi

  echo "Starting edge camera client..."
  nohup python "$ROOT_DIR/scripts/edge_camera_client.py" \
    --base-url "$BACKEND_URL" \
    --camera-id "$CAMERA_ID" \
    --source "$CAMERA_SOURCE" \
    --fps "$CAMERA_FPS" \
    --batch-size "$BATCH_SIZE" \
    >/tmp/cardsense-edge.log 2>&1 &

  echo "Edge client started (log: /tmp/cardsense-edge.log)."
fi

echo "Done."
echo "Dashboard: $BACKEND_URL"
echo "Logs: docker compose logs -f"
echo "Stop: docker compose down"
