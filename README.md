# CardSense Distributed MVP (Queue + Parallel YOLO Workers)

This repository now includes a working distributed backend/frontend stack for real-time card detection, with a Thulla hand-view mode.

## What is implemented

- FastAPI backend with:
  - `POST /api/cameras/register`
  - `POST /api/frames` (frame ingest queue)
  - `POST /api/frames/batch` (parallel frame ingest in one call)
  - `GET /api/detections/latest`
  - `GET /api/hands/latest`
  - `GET /api/hands/{camera_id}`
  - `GET /api/hands/{camera_id}/advice`
  - `GET /api/health`
  - `WS /ws/detections` for live frontend updates
- Redis Streams queue for frame jobs + detection events
- Parallel worker service running lightweight YOLO card model (`yolov8n-playing-cards`) with multi-process execution per machine
- React frontend dashboard with Thulla hand slots (13 cards) + live detections (served by backend)
- Thulla helper suggestions (discard candidates, suit strength, sequence/pair hints)
- Edge camera client script to push webcam/RTSP frames

## Folder structure

- `backend/app/main.py` -> API + websocket + detection broadcast consumer
- `worker/worker.py` -> queue consumer + YOLO inference
- `frontend/` -> dashboard
- `scripts/edge_camera_client.py` -> sends video frames to backend
- `docker-compose.yml` -> local distributed stack

## Quick start (Docker)

### One-command start

Run:

```bash
./scripts/start.sh
```

Optional environment variables:

- `WORKERS=4` (number of worker containers)
- `START_EDGE_CLIENT=1` (auto-start local camera sender)
- `CAMERA_ID=cam-1`
- `CAMERA_SOURCE=0` (or RTSP URL)
- `CAMERA_FPS=10`
- `BATCH_SIZE=4`

Example:

```bash
WORKERS=4 START_EDGE_CLIENT=1 CAMERA_SOURCE=0 ./scripts/start.sh
```

1. Start services:

```bash
docker compose up --build
```

2. Open dashboard:

- http://localhost:8000

3. In the dashboard:

- Register camera (`cam-1`)
- Click **Start webcam stream**

This starts sending browser webcam frames to `/api/frames/batch`, so detections and Thulla help appear live.

4. (Optional) Start an external edge client from host (or another machine):

```bash
python scripts/edge_camera_client.py --base-url http://localhost:8000 --camera-id cam-1 --source 0 --fps 10
```

> Use RTSP source with: `--source rtsp://...`

On first run, the worker auto-downloads a prebuilt card-rank/suit model:

- `YOLO_MODEL_URL=https://raw.githubusercontent.com/noorkhokhar99/Playing-Cards-Detection-with-YoloV8/main/yolov8s_playing_cards.pt`

This returns card-type classes (rank/suit), not a single generic `card` class.

A local copy has been installed at:

- `/home/roger/Source/Python/ProjectDL/models/playing_cards_best.pt`

`docker-compose` mounts `./models` into worker containers as `/app/models` and uses this file directly.

## Scale parallel frame processing

1) Scale machine-level workers:

Run more inference workers:

```bash
docker compose up --scale worker=4
```

Each worker consumes from the same Redis Stream consumer group for parallel processing.

2) Scale process-level workers inside each machine:

- Set `WORKER_PROCESSES=2` (or higher) per worker container.
- Each process gets its own consumer name, so frames are processed concurrently.

## Multi-machine deployment

- Deploy `redis` + `backend` on a reachable host.
- On each worker machine, run only worker container with `REDIS_URL` pointing to shared Redis.
- On edge devices, run `scripts/edge_camera_client.py` pointing to backend URL.

Example worker run on remote machine:

```bash
docker run --rm \
  -e REDIS_URL=redis://<redis-host>:6379/0 \
  -e FRAME_STREAM=frames \
  -e DETECTION_STREAM=detections \
  -e FRAME_CONSUMER_GROUP=yolo_workers \
  -e YOLO_MODEL=yolov8n.pt \
  cardsense-worker:latest
```

## Notes

- The playing-card model auto-downloads on first worker boot.
- You can replace with your trained model later by changing `YOLO_MODEL`.
- Model class naming depends on the downloaded card model. Verify labels and set `CARD_LABELS` if needed.
- For strict card-only labels, set `CARD_LABELS` in `.env` to comma-separated class names from your card model.
- For best helper quality, use card-like class names (`AS`, `10H`, `KD`, `7C`) in your model.

## Troubleshooting: empty detections

If you only see `detections: []`:

- Ensure a card fully fills a good portion of the webcam frame.
- Keep card face upright or rotate slowly while scanning.
- Use good light and hold one card close to camera first (8-12 inches).
- Lower confidence threshold (`CONF_THRES=0.05`) and restart workers.
- Restart with rebuilt worker image after dependency/config changes:

```bash
docker compose build worker && docker compose up -d --force-recreate worker
```

If frontend capture settings were updated, recreate backend too:

```bash
docker compose up -d --force-recreate backend
```

Live Debug panel now shows:

- frame brightness and blur score
- frame age (`age_ms`) to confirm near-real-time processing
- per-pass rotation/scale candidate counts
- low-threshold probe top classes

Rebuild both worker and backend to enable it:

```bash
docker compose build worker backend && docker compose up -d --force-recreate worker backend
```
