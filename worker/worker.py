import json
import os
import time
import uuid
from multiprocessing import Process
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import redis
import requests
from ultralytics import YOLO


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

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
CALIBRATION_TEMPERATURE = float(os.getenv("CALIBRATION_TEMPERATURE", "1.0"))
MAX_FRAME_AGE_MS = int(os.getenv("MAX_FRAME_AGE_MS", "2500"))
SLOW_ACCURATE_MODE = os.getenv("SLOW_ACCURATE_MODE", "1") == "1"
WORKER_LOG_STREAM = os.getenv("WORKER_LOG_STREAM", "worker_logs")
WORKER_STATUS_PREFIX = os.getenv("WORKER_STATUS_PREFIX", "worker:status")
WORKER_HEARTBEAT_SEC = int(os.getenv("WORKER_HEARTBEAT_SEC", "5"))
MODEL_SMOKE_TEST_URLS = [
    value.strip()
    for value in os.getenv(
        "MODEL_SMOKE_TEST_URLS",
        "https://huggingface.co/mustafakemal0146/playing-cards-yolov8/resolve/main/1.png,https://huggingface.co/mustafakemal0146/playing-cards-yolov8/resolve/main/2.png,https://huggingface.co/mustafakemal0146/playing-cards-yolov8/resolve/main/3.png",
    ).split(",")
    if value.strip()
]
MODEL_SMOKE_MIN_DETECTIONS = int(os.getenv("MODEL_SMOKE_MIN_DETECTIONS", "1"))


def infer(model: YOLO, image, conf: float):
    return model.predict(
        source=image,
        conf=conf,
        imgsz=1280 if SLOW_ACCURATE_MODE else 960,
        augment=SLOW_ACCURATE_MODE,
        agnostic_nms=SLOW_ACCURATE_MODE,
        max_det=100,
        verbose=False,
    )[0]


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

    if model_path.startswith("yolov") or model_path.startswith("yolo11"):
        return model_path

    if os.path.exists(model_path):
        return model_path

    cache_dir = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "playing_cards_best.pt"

    if target.exists() and target.stat().st_size > 0:
        return str(target)

    url_candidates = [url.strip() for url in MODEL_URL.replace("\n", ",").split(",") if url.strip()]
    if not url_candidates:
        raise RuntimeError(f"Model not found at {model_path} and YOLO_MODEL_URL is empty")

    def smoke_test_model(path_or_url: str) -> tuple[bool, dict]:
        if not MODEL_SMOKE_TEST_URLS:
            return True, {"skipped": True, "reason": "no_smoke_urls"}

        try:
            test_model = YOLO(path_or_url)
            total = 0
            details = []
            for url in MODEL_SMOKE_TEST_URLS[:3]:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                img = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    details.append({"url": url, "count": 0, "error": "decode_failed"})
                    continue
                result = infer(test_model, img, max(CONF_THRES, 0.02))
                count = len(result.boxes)
                total += count
                details.append({"url": url, "count": count})
            return total >= MODEL_SMOKE_MIN_DETECTIONS, {
                "total_detections": total,
                "min_required": MODEL_SMOKE_MIN_DETECTIONS,
                "details": details,
            }
        except Exception as exc:
            return False, {"error": str(exc)}

    last_error = None
    for model_url in url_candidates:
        try:
            parsed = urlparse(model_url)
            filename = Path(parsed.path).name or "playing_cards_best.pt"
            if not filename.endswith(".pt") and not filename.endswith(".onnx"):
                filename = f"{filename}.pt"

            candidate_target = cache_dir / filename
            if candidate_target.exists() and candidate_target.stat().st_size > 0:
                return str(candidate_target)

            print(f"Downloading card model from: {model_url}")
            with requests.get(model_url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with open(candidate_target, "wb") as file_obj:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file_obj.write(chunk)

            if candidate_target.exists() and candidate_target.stat().st_size > 0:
                smoke_ok, smoke = smoke_test_model(str(candidate_target))
                print(f"Model smoke test for {candidate_target.name}: {smoke}")
                if not smoke_ok:
                    raise RuntimeError(f"Smoke test failed for {candidate_target.name}: {smoke}")
                return str(candidate_target)
        except Exception as exc:
            last_error = exc
            print(f"Model download failed for {model_url}: {exc}")

    if last_error:
        raise RuntimeError(f"All model downloads failed. Last error: {last_error}")

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


def extract_topk_probs(result, box, topk_k: int = 5) -> list[list[float | str]]:
    # For detection models, per-box full logits are often unavailable.
    # Use model-native probabilities when exposed; otherwise fallback to top1 only.
    try:
        if getattr(box, "probs", None) is not None:
            probs_tensor = box.probs.data
            if probs_tensor is not None:
                probs = probs_tensor.detach().cpu().numpy().astype(float)
                if probs.size > 0:
                    idx = np.argsort(-probs)[:topk_k]
                    total = float(np.sum(probs[idx])) or 1.0
                    return [[str(result.names.get(int(i), str(i))), float(probs[i] / total)] for i in idx]
    except Exception:
        pass
    return []


def extract_detections(result, angle: int, width: int, height: int, pass_idx: int = 0, pass_weight: float = 1.0):
    detections: list[dict] = []
    topk_k = 5
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        cls_name = result.names.get(cls_id, str(cls_id))
        topk_probs = extract_topk_probs(result, box, topk_k=topk_k)
        if not topk_probs:
            topk_probs = [[cls_name, conf]]
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = map_bbox_to_original(x1, y1, x2, y2, angle, width, height)
        detections.append(
            {
                "cls_id": cls_id,
                "cls_name": cls_name,
                "conf": conf,
                "card_probs_topk": topk_probs,
                "calibrated_card_conf_top1": calibrate_confidence(conf),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "pass_idx": pass_idx,
                "pass_weight": pass_weight,
            }
        )
    return detections


def _cluster_and_rank_detections(raw_detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
    clusters: list[dict] = []

    for det in sorted(raw_detections, key=lambda d: float(d.get("conf", 0.0)), reverse=True):
        bbox = det["bbox"]
        bbox_tuple = (float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"]))
        best_idx = -1
        best_iou = 0.0
        for idx, cluster in enumerate(clusters):
            cbox = cluster["bbox"]
            cbox_tuple = (float(cbox["x1"]), float(cbox["y1"]), float(cbox["x2"]), float(cbox["y2"]))
            overlap = _iou(bbox_tuple, cbox_tuple)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            clusters[best_idx]["members"].append(det)
        else:
            clusters.append({"bbox": bbox, "members": [det]})

    fused: list[dict] = []
    for cluster in clusters:
        members = cluster["members"]
        label_scores: dict[str, float] = {}
        label_ids: dict[str, int] = {}
        det_conf = 0.0
        weighted = {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}
        total_weight = 0.0

        for m in members:
            label = str(m.get("cls_name", ""))
            conf = float(m.get("conf", 0.0))
            w = float(m.get("pass_weight", 1.0))
            det_conf = max(det_conf, conf)
            score = conf * w
            label_scores[label] = label_scores.get(label, 0.0) + score
            label_ids[label] = int(m.get("cls_id", -1))

            for pair in m.get("card_probs_topk", [])[:5]:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                p_label = str(pair[0])
                try:
                    p_score = float(pair[1])
                except Exception:
                    continue
                label_scores[p_label] = label_scores.get(p_label, 0.0) + (p_score * w)

            b = m["bbox"]
            total_weight += max(score, 1e-6)
            weighted["x1"] += float(b["x1"]) * max(score, 1e-6)
            weighted["y1"] += float(b["y1"]) * max(score, 1e-6)
            weighted["x2"] += float(b["x2"]) * max(score, 1e-6)
            weighted["y2"] += float(b["y2"]) * max(score, 1e-6)

        if total_weight <= 0:
            continue

        topk = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        score_total = sum(v for _, v in topk) or 1.0
        topk_probs = [[label, round(score / score_total, 6)] for label, score in topk]
        top1_label = topk_probs[0][0]
        top1_conf = float(topk_probs[0][1])
        calibrated_top1_conf = calibrate_confidence(top1_conf)

        fused.append(
            {
                "cls_id": int(label_ids.get(top1_label, -1)),
                "cls_name": top1_label,
                "conf": top1_conf,
                "card_label_top1": top1_label,
                "card_conf_top1": top1_conf,
                "card_probs_topk": topk_probs,
                "det_conf": round(det_conf, 6),
                "calibrated_card_conf_top1": round(calibrated_top1_conf, 6),
                "bbox": {
                    "x1": weighted["x1"] / total_weight,
                    "y1": weighted["y1"] / total_weight,
                    "x2": weighted["x2"] / total_weight,
                    "y2": weighted["y2"] / total_weight,
                },
            }
        )

    fused.sort(key=lambda d: float(d.get("det_conf", 0.0)), reverse=True)
    return fused


def predict_with_fallback(model: YOLO, image, conf: float):
    height, width = image.shape[:2]
    upscaled = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    pass_stats = []
    passes = [
        (0, image, 1.0),
        (90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 0.92),
        (270, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 0.92),
        (180, cv2.rotate(image, cv2.ROTATE_180), 0.90),
        (0, upscaled, 0.95),
    ]

    aggregated_raw: list[dict] = []

    for pass_idx, (angle, candidate, pass_weight) in enumerate(passes):
        result = infer(model, candidate, conf)
        if candidate.shape[:2] != (height, width):
            ch, cw = candidate.shape[:2]
            scale_x = cw / width
            scale_y = ch / height
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls.item())
                box_conf = float(box.conf.item())
                cls_name = result.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "conf": box_conf,
                        "bbox": {
                            "x1": x1 / scale_x,
                            "y1": y1 / scale_y,
                            "x2": x2 / scale_x,
                            "y2": y2 / scale_y,
                        },
                        "pass_idx": pass_idx,
                        "pass_weight": pass_weight,
                    }
                )
        else:
            detections = extract_detections(result, angle, width, height, pass_idx=pass_idx, pass_weight=pass_weight)

        aggregated_raw.extend(detections)

        pass_stats.append(
            {
                "pass": pass_idx,
                "angle": angle,
                "shape": [int(candidate.shape[1]), int(candidate.shape[0])],
                "count": len(detections),
            }
        )

    if aggregated_raw:
        fused = _cluster_and_rank_detections(aggregated_raw)
        if fused:
            selected_pass = None
            if pass_stats:
                selected_pass = max(pass_stats, key=lambda p: p.get("count", 0)).get("pass")
            return fused, {
                "selected_pass": selected_pass,
                "passes": pass_stats,
                "fused_count": len(fused),
            }

    probe_result = infer(model, image, DEBUG_CONF)
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


def now_ms() -> int:
    return int(time.time() * 1000)


def is_out_of_order_frame(camera_id: str, ts_ms: int, last_processed_ts_by_camera: dict[str, int]) -> bool:
    return int(ts_ms) <= int(last_processed_ts_by_camera.get(camera_id, 0))


def calibrate_confidence(conf: float) -> float:
    temp = max(CALIBRATION_TEMPERATURE, 1e-6)
    c = min(max(float(conf), 1e-6), 1.0 - 1e-6)
    logit = np.log(c / (1.0 - c))
    scaled = 1.0 / (1.0 + np.exp(-(logit / temp)))
    return float(min(max(scaled, 0.0), 1.0))


def summarize_model_names(model: YOLO) -> dict:
    names_map = model.names or {}
    labels = [str(value).upper() for value in names_map.values()]
    card_like = 0
    card_examples = []
    for label in labels:
        normalized = label.replace("_", "").replace("-", "")
        has_rank = any(rank in normalized for rank in ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"])
        has_suit = any(suit in normalized for suit in ["S", "H", "D", "C", "SPADE", "HEART", "DIAMOND", "CLUB"])
        if has_rank and has_suit:
            card_like += 1
            if len(card_examples) < 12:
                card_examples.append(label)
    return {
        "class_count": len(labels),
        "card_like_count": card_like,
        "card_like_examples": card_examples,
        "sample_labels": labels[:12],
    }


def push_worker_log(r: redis.Redis, level: str, worker_name: str, message: str, **fields) -> None:
    payload = {
        "ts_ms": now_ms(),
        "level": level,
        "worker_name": worker_name,
        "message": message,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False))
    try:
        r.xadd(WORKER_LOG_STREAM, {"event": json.dumps(payload)}, maxlen=5000, approximate=True)
    except Exception:
        pass


def write_worker_status(r: redis.Redis, worker_name: str, status: dict) -> None:
    key = f"{WORKER_STATUS_PREFIX}:{worker_name}"
    mapping = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in status.items()}
    mapping["updated_ts_ms"] = str(now_ms())
    r.hset(key, mapping=mapping)
    r.expire(key, 60)


def run_worker(consumer_name: str) -> None:
    resolved_model = ensure_model()
    model = YOLO(resolved_model)

    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    ensure_group(r, FRAME_STREAM, FRAME_GROUP)

    model_summary = summarize_model_names(model)
    push_worker_log(
        r,
        "INFO",
        consumer_name,
        "worker_started",
        model_name=MODEL_NAME,
        resolved_model=resolved_model,
        model_summary=model_summary,
        conf_thres=CONF_THRES,
        slow_accurate_mode=SLOW_ACCURATE_MODE,
    )
    if model_summary["card_like_count"] < 8:
        push_worker_log(
            r,
            "WARN",
            consumer_name,
            "model_labels_not_card_like",
            sample_labels=model_summary["sample_labels"],
            card_like_count=model_summary["card_like_count"],
        )

    stats = {
        "processed_frames": 0,
        "skipped_stale": 0,
        "skipped_out_of_order": 0,
        "error_frames": 0,
        "frames_with_detections": 0,
        "total_detections": 0,
        "avg_latency_ms": 0.0,
        "last_camera_id": "",
        "last_error": "",
        "last_selected_pass": "",
        "last_probe_top": [],
    }
    last_processed_ts_by_camera: dict[str, int] = {}
    last_heartbeat = time.time()

    write_worker_status(
        r,
        consumer_name,
        {
            "model_name": MODEL_NAME,
            "resolved_model": resolved_model,
            "model_summary": model_summary,
            **stats,
        },
    )

    push_worker_log(r, "INFO", consumer_name, "worker_ready")

    while True:
        response = r.xreadgroup(
            groupname=FRAME_GROUP,
            consumername=consumer_name,
            streams={FRAME_STREAM: ">"},
            count=64,
            block=1000,
        )

        if not response:
            if (time.time() - last_heartbeat) >= WORKER_HEARTBEAT_SEC:
                write_worker_status(
                    r,
                    consumer_name,
                    {
                        "model_name": MODEL_NAME,
                        "resolved_model": resolved_model,
                        "model_summary": model_summary,
                        **stats,
                    },
                )
                last_heartbeat = time.time()
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
                    game_id = payload.get("game_id", "default-game")
                    camera_type = payload.get("camera_type", "hand")
                    frame_id = payload.get("frame_id", f"{camera_id}:{payload.get('ts_ms', '')}")
                    ts_ms = int(payload["ts_ms"])
                    frame_b64 = payload["frame_b64"]
                    age_ms = int(time.time() * 1000) - ts_ms

                    if is_out_of_order_frame(camera_id, ts_ms, last_processed_ts_by_camera):
                        stats["skipped_out_of_order"] += 1
                        continue

                    if age_ms > MAX_FRAME_AGE_MS:
                        stats["skipped_stale"] += 1
                        if stats["skipped_stale"] % 20 == 0:
                            push_worker_log(
                                r,
                                "WARN",
                                consumer_name,
                                "stale_frames_skipped",
                                skipped_stale=stats["skipped_stale"],
                                age_ms=age_ms,
                                max_frame_age_ms=MAX_FRAME_AGE_MS,
                            )
                        continue

                    image = decode_b64_to_image(frame_b64)
                    detections, debug_details = predict_with_fallback(model, image, CONF_THRES)
                    quality = image_quality(image)
                    quality_flags = {
                        "is_blurry": float(quality.get("blur_var", 0.0)) < 75.0,
                        "is_dark": float(quality.get("brightness", 0.0)) < 45.0,
                        "is_overexposed": float(quality.get("brightness", 0.0)) > 225.0,
                    }
                    for det in detections:
                        det["is_blurry"] = bool(quality_flags["is_blurry"])
                        det["is_dark"] = bool(quality_flags["is_dark"])
                        det["is_overexposed"] = bool(quality_flags["is_overexposed"])
                    latency_ms = round((time.time() - started) * 1000, 2)

                    stats["processed_frames"] += 1
                    last_processed_ts_by_camera[camera_id] = ts_ms
                    stats["last_camera_id"] = camera_id
                    stats["last_selected_pass"] = debug_details.get("selected_pass")
                    stats["last_probe_top"] = debug_details.get("probe_top", [])
                    stats["total_detections"] += len(detections)
                    if detections:
                        stats["frames_with_detections"] += 1
                    prior_avg = float(stats["avg_latency_ms"])
                    count = stats["processed_frames"]
                    stats["avg_latency_ms"] = round(((prior_avg * (count - 1)) + latency_ms) / count, 2)

                    event = {
                        "game_id": game_id,
                        "frame_id": frame_id,
                        "camera_type": camera_type,
                        "camera_id": camera_id,
                        "ts_ms": ts_ms,
                        "model_name": MODEL_NAME,
                        "resolved_model": resolved_model,
                        "worker_name": consumer_name,
                        "latency_ms": latency_ms,
                        "detections": detections,
                        "debug": {
                            "quality": quality,
                            "conf_thres": CONF_THRES,
                            "debug_conf": DEBUG_CONF,
                            "calibration_temperature": CALIBRATION_TEMPERATURE,
                            "age_ms": age_ms,
                            "max_frame_age_ms": MAX_FRAME_AGE_MS,
                            "model_summary": model_summary,
                            **debug_details,
                        },
                    }

                    r.xadd(
                        DETECTION_STREAM,
                        {"event": json.dumps(event)},
                        maxlen=MAX_QUEUE_LEN,
                        approximate=True,
                    )

                    if not detections and stats["processed_frames"] % 10 == 0:
                        push_worker_log(
                            r,
                            "WARN",
                            consumer_name,
                            "no_detections_recently",
                            processed_frames=stats["processed_frames"],
                            camera_id=camera_id,
                            probe_top=debug_details.get("probe_top", []),
                            quality=quality,
                            selected_pass=debug_details.get("selected_pass"),
                        )

                    if detections and stats["processed_frames"] % 5 == 0:
                        top = sorted(detections, key=lambda d: d["conf"], reverse=True)[:3]
                        push_worker_log(
                            r,
                            "INFO",
                            consumer_name,
                            "detections_observed",
                            camera_id=camera_id,
                            count=len(detections),
                            top=[{"cls_name": d["cls_name"], "conf": round(float(d["conf"]), 4)} for d in top],
                        )
                except Exception as exc:
                    stats["error_frames"] += 1
                    stats["last_error"] = str(exc)
                    push_worker_log(r, "ERROR", consumer_name, "frame_processing_failed", message_id=msg_id, error=str(exc))
                finally:
                    r.xack(stream_name, FRAME_GROUP, msg_id)

                if (time.time() - last_heartbeat) >= WORKER_HEARTBEAT_SEC:
                    write_worker_status(
                        r,
                        consumer_name,
                        {
                            "model_name": MODEL_NAME,
                            "resolved_model": resolved_model,
                            "model_summary": model_summary,
                            **stats,
                        },
                    )
                    last_heartbeat = time.time()


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
