import asyncio
import json
from collections import defaultdict, deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .models import CameraRegister, DetectionEvent, FrameIn
from .redis_client import redis_client


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


hub = DetectionHub()
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
                    DetectionEvent.model_validate(event)
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


@app.get("/api/detections/latest")
async def latest_detections() -> dict:
    return {"events": await hub.latest()}


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
