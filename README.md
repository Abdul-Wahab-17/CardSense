# CardSense Distributed MVP (Queue + Parallel YOLO Workers)

This repository now includes a working distributed backend/frontend skeleton for real-time card detection.

## What is implemented

- FastAPI backend with:
  - `POST /api/cameras/register`
  - `POST /api/frames` (frame ingest queue)
  - `GET /api/detections/latest`
  - `GET /api/health`
  - `WS /ws/detections` for live frontend updates
- Redis Streams queue for frame jobs + detection events
- Parallel worker service running lightweight YOLO (`yolov8n.pt`)
- Simple frontend dashboard (served by backend)
- Edge camera client script to push webcam/RTSP frames

## Folder structure

- `backend/app/main.py` -> API + websocket + detection broadcast consumer
- `worker/worker.py` -> queue consumer + YOLO inference
- `frontend/` -> dashboard
- `scripts/edge_camera_client.py` -> sends video frames to backend
- `docker-compose.yml` -> local distributed stack

## Quick start (Docker)

1. Start services:

```bash
docker compose up --build
```

2. Open dashboard:

- http://localhost:8000

3. Start an edge client from host (or another machine):

```bash
python scripts/edge_camera_client.py --base-url http://localhost:8000 --camera-id cam-1 --source 0 --fps 10
```

> Use RTSP source with: `--source rtsp://...`

## Scale parallel workers

Run more inference workers:

```bash
docker compose up --scale worker=4
```

Each worker consumes from the same Redis Stream consumer group for parallel processing.

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

- `yolov8n.pt` auto-downloads on first worker boot.
- You can replace with your trained model later by changing `YOLO_MODEL`.
- Current model classes are generic COCO classes. Card-specific accuracy will depend on your custom model.
