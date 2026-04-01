import json
import os
import time
import uuid

import redis
from ultralytics import YOLO

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
FRAME_STREAM = os.getenv("FRAME_STREAM", "frames")
DETECTION_STREAM = os.getenv("DETECTION_STREAM", "detections")
FRAME_GROUP = os.getenv("FRAME_CONSUMER_GROUP", "yolo_workers")
WORKER_NAME = os.getenv("WORKER_NAME", f"worker-{uuid.uuid4().hex[:8]}")
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
MAX_QUEUE_LEN = int(os.getenv("MAX_QUEUE_LEN", "5000"))
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))


def ensure_group(r: redis.Redis, stream: str, group: str) -> None:
    try:
        r.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def decode_b64_to_file(frame_b64: str) -> str:
    import base64
    from tempfile import NamedTemporaryFile

    image_bytes = base64.b64decode(frame_b64)
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        return tmp.name


def run() -> None:
    print(f"Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    ensure_group(r, FRAME_STREAM, FRAME_GROUP)

    print(f"Worker {WORKER_NAME} ready. Waiting for frames...")

    while True:
        response = r.xreadgroup(
            groupname=FRAME_GROUP,
            consumername=WORKER_NAME,
            streams={FRAME_STREAM: ">"},
            count=8,
            block=1000,
        )

        if not response:
            continue

        for stream_name, messages in response:
            for msg_id, payload in messages:
                started = time.time()
                image_path = None
                try:
                    camera_id = payload["camera_id"]
                    ts_ms = int(payload["ts_ms"])
                    frame_b64 = payload["frame_b64"]

                    image_path = decode_b64_to_file(frame_b64)
                    result = model.predict(source=image_path, conf=CONF_THRES, verbose=False)[0]

                    detections = []
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                        detections.append(
                            {
                                "cls_id": cls_id,
                                "cls_name": result.names.get(cls_id, str(cls_id)),
                                "conf": conf,
                                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            }
                        )

                    event = {
                        "camera_id": camera_id,
                        "ts_ms": ts_ms,
                        "model_name": MODEL_NAME,
                        "latency_ms": round((time.time() - started) * 1000, 2),
                        "detections": detections,
                    }

                    r.xadd(
                        DETECTION_STREAM,
                        {"event": json.dumps(event)},
                        maxlen=MAX_QUEUE_LEN,
                        approximate=True,
                    )
                except Exception as exc:
                    print(f"[{WORKER_NAME}] Failed message {msg_id}: {exc}")
                finally:
                    if image_path:
                        try:
                            os.remove(image_path)
                        except OSError:
                            pass
                    r.xack(stream_name, FRAME_GROUP, msg_id)


if __name__ == "__main__":
    run()
