# Operating CardSense

Tuning, scaling and troubleshooting notes. See the [README](../README.md) for setup.

## Scale frame processing

All workers read from the same Redis Stream consumer group (`yolo_workers`), so adding workers adds throughput with no other changes.

**More worker containers:**

```bash
docker compose up -d --scale worker-service=4
```

**More processes per container:** set `WORKER_PROCESSES=2` (or higher). Each process registers its own consumer name.

## Multi-machine deployment

- Run `redis` and `api-service` on a reachable host.
- On each worker machine, run only the worker image with `REDIS_URL` pointing at the shared Redis.
- On edge devices, run `scripts/edge_camera_client.py` against the API URL.

```bash
docker run --rm \
  -e REDIS_URL=redis://<redis-host>:6379/0 \
  -e FRAME_STREAM=frames \
  -e DETECTION_STREAM=detections \
  -e FRAME_CONSUMER_GROUP=yolo_workers \
  -e YOLO_DEVICE=cpu \
  cardsense-worker:latest
```

## Detection model

On first start, the worker downloads the model at `YOLO_MODEL_URL` into `MODEL_CACHE_DIR`. You can mount your own weights instead (for example `./models/your_model.pt`, which compose mounts at `/app/models`) and set `YOLO_MODEL=/app/models/your_model.pt`.

Labels must be card classes. Names like `AS`, `10H` and `KD`, or `ace_spades`, are normalised. To restrict detections to specific classes, set `CARD_LABELS` to a comma-separated list.

## Troubleshooting: empty detections

If you only see `detections: []`:

- Ensure a card fully fills a good portion of the webcam frame.
- Keep card face upright or rotate slowly while scanning.
- Use good light and hold one card close to camera first (8-12 inches).
- Lower confidence threshold (`CONF_THRES=0.05`) and restart workers.
- Restart with rebuilt worker image after dependency/config changes:

```bash
docker compose build worker-service && docker compose up -d --force-recreate worker-service
```

If frontend capture settings were updated, recreate backend too:

```bash
docker compose up -d --force-recreate api-service
```

Live Debug panel now shows:

- frame brightness and blur score
- frame age (`age_ms`) to confirm near-real-time processing
- per-pass rotation/scale candidate counts
- low-threshold probe top classes

If detection quality is inconsistent, use **Test Mode (Manual Hand)** in the UI to validate recommendation flow independently of model quality.

## LLM advisor setup

Set these in `.env` (or environment):

- `ADVISOR_MODE=hybrid` (or `llm`)
- `LLM_ENABLED=1`
- `LLM_API_KEY=<your key>`
- `LLM_MODEL=<model id>`
- optional: `LLM_RULES_TEXT=<your variant rules/instructions>`

Behavior:

- `rules`: deterministic simulator only.
- `hybrid`: deterministic scorer + LLM explanation/selection (legality-checked).
- `llm`: LLM-first with deterministic fallback when LLM is unavailable.

Temporal consensus tuning (for N-of-M frame stability):

- `HAND_CONSENSUS_WINDOW_FRAMES`, `HAND_CONSENSUS_MIN_SAMPLES`, `HAND_CONSENSUS_MIN_RATIO`
- `FUSION_CONSENSUS_WINDOW_FRAMES`, `FUSION_CONSENSUS_MIN_SAMPLES`, `FUSION_CONSENSUS_MIN_RATIO`

Worker can run in slower but more accurate mode:

- `SLOW_ACCURATE_MODE=1` (default)

Rebuild both worker and backend to enable it:

```bash
docker compose build worker-service api-service && docker compose up -d --force-recreate worker-service api-service
```
