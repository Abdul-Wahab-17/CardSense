import json
import os
import time
import uuid
from multiprocessing import Process
from pathlib import Path

import cv2
import numpy as np
import redis
import requests
from ultralytics import YOLO

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
FRAME_STREAM = os.getenv("FRAME_STREAM", "frames")
DETECTION_STREAM = os.getenv("DETECTION_STREAM", "detections")
FRAME_GROUP = os.getenv("FRAME_CONSUMER_GROUP", "yolo_workers")
WORKER_NAME = os.getenv("WORKER_NAME", f"worker-{uuid.uuid4().hex[:8]}")
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
MODEL_URL = os.getenv(
    "YOLO_MODEL_URL",
    "https://raw.githubusercontent.com/noorkhokhar99/Playing-Cards-Detection-with-YoloV8/main/yolov8s_playing_cards.pt",
)
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/cardsense-models")
MAX_QUEUE_LEN = int(os.getenv("MAX_QUEUE_LEN", "5000"))
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
WORKER_PROCESSES = int(os.getenv("WORKER_PROCESSES", "1"))
DEBUG_CONF = float(os.getenv("DEBUG_CONF", "0.01"))
MAX_FRAME_AGE_MS = int(os.getenv("MAX_FRAME_AGE_MS", "2500"))


def ensure_group(r: redis.Redis, stream: str, group: str) -> None:
    try:
        r.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def decode_b64_to_image(frame_b64: str):
    import base64

    image_bytes = base64.b64decode(frame_b64)
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode frame")
    return image


def ensure_model() -> str:
    model_path = MODEL_NAME
    if model_path.startswith("http://") or model_path.startswith("https://"):
        return model_path

    if os.path.exists(model_path):
        return model_path

    cache_dir = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "playing_cards_best.pt"

    if target.exists() and target.stat().st_size > 0:
        return str(target)

    if not MODEL_URL:
        raise RuntimeError(f"Model not found at {model_path} and YOLO_MODEL_URL is empty")

    print(f"Downloading card model from: {MODEL_URL}")
    with requests.get(MODEL_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(target, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)

    return str(target)


def map_bbox_to_original(x1: float, y1: float, x2: float, y2: float, angle: int, width: int, height: int) -> tuple[float, float, float, float]:
    if angle == 0:
        return x1, y1, x2, y2

    def transform_point(x: float, y: float) -> tuple[float, float]:
        if angle == 90:
            return y, height - x
        if angle == 180:
            return width - x, height - y
        if angle == 270:
            return width - y, x
        return x, y

    p1 = transform_point(x1, y1)
    p2 = transform_point(x2, y2)
    nx1, nx2 = sorted([p1[0], p2[0]])
    ny1, ny2 = sorted([p1[1], p2[1]])
    return nx1, ny1, nx2, ny2


def extract_detections(result, angle: int, width: int, height: int):
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = map_bbox_to_original(x1, y1, x2, y2, angle, width, height)
        detections.append(
            {
                "cls_id": cls_id,
                "cls_name": result.names.get(cls_id, str(cls_id)),
                "conf": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )
    return detections


def predict_with_fallback(model: YOLO, image, conf: float):
    height, width = image.shape[:2]
    upscaled = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    pass_stats = []
    passes = [
        (0, image),
        (90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        (270, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        (180, cv2.rotate(image, cv2.ROTATE_180)),
        (0, upscaled),
    ]

    for pass_idx, (angle, candidate) in enumerate(passes):
        result = model.predict(source=candidate, conf=conf, imgsz=960, verbose=False)[0]
        if candidate.shape[:2] != (height, width):
            ch, cw = candidate.shape[:2]
            scale_x = cw / width
            scale_y = ch / height
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls.item())
                box_conf = float(box.conf.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "cls_id": cls_id,
                        "cls_name": result.names.get(cls_id, str(cls_id)),
                        "conf": box_conf,
                        "bbox": {
                            "x1": x1 / scale_x,
                            "y1": y1 / scale_y,
                            "x2": x2 / scale_x,
                            "y2": y2 / scale_y,
                        },
                    }
                )
        else:
            detections = extract_detections(result, angle, width, height)

        pass_stats.append(
            {
                "pass": pass_idx,
                "angle": angle,
                "shape": [int(candidate.shape[1]), int(candidate.shape[0])],
                "count": len(detections),
            }
        )

        if detections:
            detections.sort(key=lambda det: det["conf"], reverse=True)
            return detections, {"selected_pass": pass_idx, "passes": pass_stats}

    probe_result = model.predict(source=image, conf=DEBUG_CONF, imgsz=960, verbose=False)[0]
    probe_detections = extract_detections(probe_result, 0, width, height)
    probe_detections.sort(key=lambda det: det["conf"], reverse=True)

    return [], {
        "selected_pass": None,
        "passes": pass_stats,
        "probe_count": len(probe_detections),
        "probe_top": [
            {
                "cls_name": det["cls_name"],
                "conf": round(float(det["conf"]), 4),
            }
            for det in probe_detections[:5]
        ],
    }


def image_quality(image) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "brightness": round(brightness, 2),
        "blur_var": round(blur, 2),
    }


def run_worker(consumer_name: str) -> None:
    resolved_model = ensure_model()
    print(f"[{consumer_name}] Loading model: {resolved_model}")
    model = YOLO(resolved_model)

    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    ensure_group(r, FRAME_STREAM, FRAME_GROUP)

    print(f"Worker {consumer_name} ready. Waiting for frames...")

    while True:
        response = r.xreadgroup(
            groupname=FRAME_GROUP,
            consumername=consumer_name,
            streams={FRAME_STREAM: ">"},
            count=64,
            block=1000,
        )

        if not response:
            continue

        for stream_name, messages in response:
            latest_by_camera: dict[str, tuple[str, dict]] = {}
            for msg_id, payload in messages:
                camera_id = payload.get("camera_id", "unknown")
                latest_by_camera[camera_id] = (msg_id, payload)

            latest_ids = {msg_id for msg_id, _ in latest_by_camera.values()}
            for msg_id, _ in messages:
                if msg_id not in latest_ids:
                    r.xack(stream_name, FRAME_GROUP, msg_id)

            for msg_id, payload in latest_by_camera.values():
                started = time.time()
                try:
                    camera_id = payload["camera_id"]
                    ts_ms = int(payload["ts_ms"])
                    frame_b64 = payload["frame_b64"]
                    age_ms = int(time.time() * 1000) - ts_ms

                    if age_ms > MAX_FRAME_AGE_MS:
                        continue

                    image = decode_b64_to_image(frame_b64)
                    detections, debug_details = predict_with_fallback(model, image, CONF_THRES)
                    quality = image_quality(image)

                    event = {
                        "camera_id": camera_id,
                        "ts_ms": ts_ms,
                        "model_name": MODEL_NAME,
                        "worker_name": consumer_name,
                        "latency_ms": round((time.time() - started) * 1000, 2),
                        "detections": detections,
                        "debug": {
                            "quality": quality,
                            "conf_thres": CONF_THRES,
                            "debug_conf": DEBUG_CONF,
                            "age_ms": age_ms,
                            "max_frame_age_ms": MAX_FRAME_AGE_MS,
                            **debug_details,
                        },
                    }

                    r.xadd(
                        DETECTION_STREAM,
                        {"event": json.dumps(event)},
                        maxlen=MAX_QUEUE_LEN,
                        approximate=True,
                    )
                except Exception as exc:
                    print(f"[{consumer_name}] Failed message {msg_id}: {exc}")
                finally:
                    r.xack(stream_name, FRAME_GROUP, msg_id)


def run() -> None:
    process_count = max(WORKER_PROCESSES, 1)
    if process_count == 1:
        run_worker(WORKER_NAME)
        return

    print(f"Starting {process_count} worker processes on this machine")
    processes: list[Process] = []
    for index in range(process_count):
        consumer_name = f"{WORKER_NAME}-p{index + 1}"
        process = Process(target=run_worker, args=(consumer_name,), daemon=False)
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    run()
