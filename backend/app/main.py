import asyncio
import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .models import CameraRegister, DetectionEvent, FrameBatchIn, FrameIn, HandAdvice, HandCard, HandOverrideIn, HandState
from .redis_client import redis_client
from .thulla import build_thulla_advice


class DetectionHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._last_events = defaultdict(lambda: deque(maxlen=50))
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def remember(self, event: dict) -> None:
        camera_id = event.get("camera_id", "unknown")
        self._last_events[camera_id].append(event)

    async def latest(self) -> list[dict]:
        out = []
        for camera_events in self._last_events.values():
            if camera_events:
                out.append(camera_events[-1])
        return out

    async def broadcast(self, event: dict) -> None:
        stale = []
        async with self._lock:
            for ws in self._clients:
                try:
                    await ws.send_json(event)
                except Exception:
                    stale.append(ws)

            for ws in stale:
                self._clients.discard(ws)


class HandStateStore:
    def __init__(self) -> None:
        self._memory = defaultdict(lambda: deque(maxlen=settings.hand_memory_events))
        self._latest_ts = defaultdict(int)
        self._overrides: dict[str, HandState] = {}
        labels = [label.strip() for label in settings.card_labels.split(",") if label.strip()]
        self._card_labels = set(labels)

    def update(self, event: DetectionEvent) -> HandState:
        camera_id = event.camera_id
        self._latest_ts[camera_id] = max(self._latest_ts[camera_id], event.ts_ms)

        for detection in event.detections:
            if detection.conf < settings.min_card_conf:
                continue
            if self._card_labels and detection.cls_name not in self._card_labels:
                continue
            self._memory[camera_id].append((detection.cls_name, float(detection.conf)))

        return self._build_state(camera_id)

    def latest(self, camera_id: str) -> HandState:
        if camera_id in self._overrides:
            return self._overrides[camera_id]

        return self._build_state(camera_id)

    def _build_state(self, camera_id: str) -> HandState:
        stats: dict[str, dict[str, float]] = {}
        for label, conf in self._memory[camera_id]:
            label_stat = stats.setdefault(label, {"count": 0.0, "sum_conf": 0.0, "max_conf": 0.0})
            label_stat["count"] += 1.0
            label_stat["sum_conf"] += conf
            label_stat["max_conf"] = max(label_stat["max_conf"], conf)

        scored_cards: list[tuple[str, float, float, float]] = []
        for label, value in stats.items():
            count = value["count"]
            if count < settings.min_card_votes:
                continue
            avg_conf = value["sum_conf"] / max(count, 1.0)
            score = value["sum_conf"] + (0.03 * count)
            scored_cards.append((label, score, avg_conf, value["max_conf"]))

        scored_cards.sort(key=lambda item: item[1], reverse=True)
        top_cards = scored_cards[: settings.hand_size]
        cards = [HandCard(label=label, confidence=round(max(avg_conf, max_conf), 3)) for label, _, avg_conf, max_conf in top_cards]

        return HandState(
            camera_id=camera_id,
            updated_ts_ms=self._latest_ts[camera_id],
            max_cards=settings.hand_size,
            cards=cards,
        )

    def latest_all(self) -> list[HandState]:
        camera_ids = set(self._memory.keys()) | set(self._overrides.keys())
        return [self.latest(camera_id) for camera_id in camera_ids]

    def set_override(self, camera_id: str, cards: list[str]) -> HandState:
        ts_ms = int(time.time() * 1000)
        state = HandState(
            camera_id=camera_id,
            updated_ts_ms=ts_ms,
            max_cards=settings.hand_size,
            cards=[HandCard(label=card, confidence=1.0) for card in cards[: settings.hand_size]],
        )
        self._overrides[camera_id] = state
        return state

    def clear_override(self, camera_id: str) -> None:
        self._overrides.pop(camera_id, None)


hub = DetectionHub()
hand_state_store = HandStateStore()
consume_task: asyncio.Task | None = None


async def ensure_group(stream_name: str, group: str) -> None:
    try:
        await redis_client.xgroup_create(stream_name, group, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def consume_detections() -> None:
    await ensure_group(settings.detection_stream, settings.backend_consumer_group)

    while True:
        response = await redis_client.xreadgroup(
            groupname=settings.backend_consumer_group,
            consumername=settings.backend_consumer_name,
            streams={settings.detection_stream: ">"},
            count=50,
            block=1000,
        )

        if not response:
            continue

        for stream_name, messages in response:
            for msg_id, payload in messages:
                try:
                    raw = payload.get("event", "{}")
                    event = json.loads(raw)
                    parsed = DetectionEvent.model_validate(event)
                    hand_state = hand_state_store.update(parsed)
                    advice = build_thulla_advice(hand_state)
                    event["hand_state"] = hand_state.model_dump()
                    event["advice"] = advice.model_dump()
                    await hub.remember(event)
                    await hub.broadcast(event)
                finally:
                    await redis_client.xack(stream_name, settings.backend_consumer_group, msg_id)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global consume_task
    await ensure_group(settings.detection_stream, settings.backend_consumer_group)
    consume_task = asyncio.create_task(consume_detections())
    try:
        yield
    finally:
        if consume_task:
            consume_task.cancel()
            with suppress(asyncio.CancelledError):
                await consume_task
        await redis_client.close()


app = FastAPI(title="CardSense Distributed Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    pong = await redis_client.ping()
    return {"ok": True, "redis": pong}


@app.post("/api/cameras/register")
async def register_camera(body: CameraRegister) -> dict:
    camera_key = f"camera:{body.camera_id}"
    await redis_client.hset(
        camera_key,
        mapping={
            "camera_id": body.camera_id,
            "source": body.source or "unknown",
        },
    )
    return {"ok": True, "camera_id": body.camera_id}


@app.post("/api/frames")
async def ingest_frame(body: FrameIn) -> dict:
    camera_key = f"camera:{body.camera_id}"
    if not await redis_client.exists(camera_key):
        raise HTTPException(status_code=404, detail="Camera not registered")

    payload = {
        "camera_id": body.camera_id,
        "ts_ms": str(body.ts_ms),
        "frame_b64": body.frame_b64,
    }
    msg_id = await redis_client.xadd(
        settings.frame_stream,
        payload,
        maxlen=settings.max_queue_len,
        approximate=True,
    )
    return {"ok": True, "queued": msg_id}


@app.post("/api/frames/batch")
async def ingest_frame_batch(body: FrameBatchIn) -> dict:
    queued = []
    for frame in body.frames:
        camera_key = f"camera:{frame.camera_id}"
        if not await redis_client.exists(camera_key):
            raise HTTPException(status_code=404, detail=f"Camera not registered: {frame.camera_id}")

    pipeline = redis_client.pipeline(transaction=False)
    for frame in body.frames:
        payload = {
            "camera_id": frame.camera_id,
            "ts_ms": str(frame.ts_ms),
            "frame_b64": frame.frame_b64,
        }
        pipeline.xadd(
            settings.frame_stream,
            payload,
            maxlen=settings.max_queue_len,
            approximate=True,
        )

    queued = await pipeline.execute()

    return {"ok": True, "queued_count": len(queued), "queued": queued}


@app.get("/api/detections/latest")
async def latest_detections() -> dict:
    return {"events": await hub.latest()}


@app.get("/api/hands/latest")
async def latest_hands() -> dict:
    return {"hands": [hand.model_dump() for hand in hand_state_store.latest_all()]}


@app.get("/api/hands/{camera_id}")
async def latest_hand(camera_id: str) -> dict:
    return {"hand": hand_state_store.latest(camera_id).model_dump()}


@app.get("/api/hands/{camera_id}/advice")
async def hand_advice(camera_id: str) -> dict:
    hand = hand_state_store.latest(camera_id)
    advice: HandAdvice = build_thulla_advice(hand)
    return {"advice": advice.model_dump(), "hand": hand.model_dump()}


@app.post("/api/hands/{camera_id}/override")
async def set_hand_override(camera_id: str, body: HandOverrideIn) -> dict:
    hand = hand_state_store.set_override(camera_id, body.cards)
    advice: HandAdvice = build_thulla_advice(hand)
    event = {
        "camera_id": camera_id,
        "ts_ms": hand.updated_ts_ms,
        "model_name": "manual_override",
        "worker_name": "manual",
        "latency_ms": 0,
        "detections": [
            {
                "cls_id": -1,
                "cls_name": card.label,
                "conf": card.confidence,
                "bbox": {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0},
            }
            for card in hand.cards
        ],
        "hand_state": hand.model_dump(),
        "advice": advice.model_dump(),
        "debug": {"mode": "manual_override"},
    }
    await hub.remember(event)
    await hub.broadcast(event)
    return {"ok": True, "hand": hand.model_dump(), "advice": advice.model_dump()}


@app.delete("/api/hands/{camera_id}/override")
async def clear_hand_override(camera_id: str) -> dict:
    hand_state_store.clear_override(camera_id)
    return {"ok": True}


@app.websocket("/ws/detections")
async def detections_ws(ws: WebSocket):
    await hub.connect(ws)
    try:
        warmup = await hub.latest()
        for event in warmup:
            await ws.send_json(event)

        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.disconnect(ws)


frontend_path = Path(settings.static_dir)
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
