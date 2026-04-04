import asyncio
import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .config import settings
from .belief import belief_diagnostics, build_belief
from .autoscaler import compute_desired_workers
from .feature_flags import FEATURE_FLAGS_METADATA
from .fusion import FusionEngine
from .game_state import GameStateEngine
from .models import (
    ActionScore,
    AdvisorAlternative,
    AdvisorBestAction,
    AdvisorResponse,
    CameraRegister,
    DetectionEvent,
    FrameBatchIn,
    FrameIn,
    GameConfigIn,
    GameStartIn,
    HandAdvice,
    HandCard,
    HandOverrideIn,
    HandState,
    TableCalibrationIn,
)
from .redis_client import redis_client
from .security import require_auth_roles, ws_validate_token
from .simulator import evaluate_actions
from .storage import EventStore
from .thulla import build_thulla_advice


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


class DetectionHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._last_events = defaultdict(lambda: deque(maxlen=50))
        self._event_log = deque(maxlen=2000)
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
        self._event_log.append(event)

    async def latest(self) -> list[dict]:
        out = []
        for camera_events in self._last_events.values():
            if camera_events:
                out.append(camera_events[-1])
        return out

    async def recent_for_game(self, game_id: str, limit: int = 200) -> list[dict]:
        out = [e for e in reversed(self._event_log) if e.get("game_id", "default-game") == game_id]
        return out[: max(1, min(limit, 1000))]

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

    async def client_count(self) -> int:
        async with self._lock:
            return len(self._clients)


class HandStateStore:
    def __init__(self) -> None:
        self._tracks = defaultdict(list)
        self._next_slot_id = defaultdict(int)
        self._latest_ts = defaultdict(int)
        self._debug = defaultdict(dict)
        self._overrides: dict[str, HandState] = {}
        labels = [label.strip() for label in settings.card_labels.split(",") if label.strip()]
        self._card_labels = set(labels)
        self._track_match_iou = 0.45
        self._frame_dedupe_iou = 0.60
        self._track_stale_ms = 3500
        self._min_track_hits = 2
        self._single_hit_conf_override = 0.80

    def _label_candidates(self, track: dict) -> list[tuple[str, float, float]]:
        candidates: list[tuple[str, float, float]] = []
        for label, values in track["labels"].items():
            count = float(values["count"])
            sum_conf = float(values["sum_conf"])
            max_conf = float(values["max_conf"])
            # score combines repeated agreement + strongest worker/frame confidence
            score = (sum_conf + (0.06 * count)) + (0.35 * max_conf)
            conf = max(max_conf, (sum_conf / max(count, 1.0)))
            candidates.append((label, score, conf))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    def _dedupe_frame(self, detections: list[tuple[str, float, tuple[float, float, float, float]]]) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        kept: list[tuple[str, float, tuple[float, float, float, float]]] = []
        for det in sorted(detections, key=lambda d: d[1], reverse=True):
            _, _, bbox = det
            if any(_iou(bbox, existing[2]) >= self._frame_dedupe_iou for existing in kept):
                continue
            kept.append(det)
        return kept

    def update(self, event: DetectionEvent, source_worker: str = "", source_model: str = "") -> HandState:
        camera_id = event.camera_id
        self._latest_ts[camera_id] = max(self._latest_ts[camera_id], event.ts_ms)
        raw_count = len(event.detections)

        frame_detections: list[tuple[str, float, tuple[float, float, float, float]]] = []
        for detection in event.detections:
            label = detection.card_label_top1 or detection.cls_name
            score_conf = float(detection.card_conf_top1 or detection.conf or 0.0)

            if score_conf < settings.min_card_conf:
                continue
            if self._card_labels and label not in self._card_labels:
                continue

            bbox = (
                float(detection.bbox.x1),
                float(detection.bbox.y1),
                float(detection.bbox.x2),
                float(detection.bbox.y2),
            )
            frame_detections.append((label, score_conf, bbox))

        filtered_count = len(frame_detections)
        frame_detections = self._dedupe_frame(frame_detections)
        deduped_count = len(frame_detections)

        active_tracks = [
            track for track in self._tracks[camera_id] if (event.ts_ms - int(track["last_ts"])) <= self._track_stale_ms
        ]

        for label, conf, bbox in frame_detections:
            best_idx = -1
            best_iou = 0.0
            for idx, track in enumerate(active_tracks):
                overlap = _iou(bbox, track["bbox"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx

            if best_idx >= 0 and best_iou >= self._track_match_iou:
                track = active_tracks[best_idx]
                track["bbox"] = bbox
                track["last_ts"] = event.ts_ms
                track["hits"] = int(track.get("hits", 0)) + 1
            else:
                if len(active_tracks) >= settings.hand_size:
                    continue
                self._next_slot_id[camera_id] += 1
                track = {
                    "slot_id": self._next_slot_id[camera_id],
                    "bbox": bbox,
                    "last_ts": event.ts_ms,
                    "hits": 1,
                    "workers": set(),
                    "models": set(),
                    "labels": {},
                }
                active_tracks.append(track)

            if source_worker:
                track.setdefault("workers", set()).add(source_worker)
            if source_model:
                track.setdefault("models", set()).add(source_model)

            label_stats = track["labels"].setdefault(label, {"count": 0.0, "sum_conf": 0.0, "max_conf": 0.0})
            label_stats["count"] += 1.0
            label_stats["sum_conf"] += conf
            label_stats["max_conf"] = max(float(label_stats["max_conf"]), conf)

        self._tracks[camera_id] = active_tracks
        self._debug[camera_id] = {
            "raw_count": raw_count,
            "after_filters_count": filtered_count,
            "after_dedupe_count": deduped_count,
            "active_tracks": len(active_tracks),
            "track_match_iou": self._track_match_iou,
            "frame_dedupe_iou": self._frame_dedupe_iou,
            "track_stale_ms": self._track_stale_ms,
            "ts_ms": event.ts_ms,
        }

        return self._build_state(camera_id)

    def latest(self, camera_id: str) -> HandState:
        if camera_id in self._overrides:
            return self._overrides[camera_id]

        return self._build_state(camera_id)

    def _build_state(self, camera_id: str) -> HandState:
        scored_cards: list[tuple[int, str, float, float]] = []
        duplicate_suppressed = 0
        track_accepted = 0

        for track in self._tracks[camera_id]:
            candidates = self._label_candidates(track)
            if not candidates:
                continue

            hits = int(track.get("hits", 0))
            best_conf = candidates[0][2]
            if hits < self._min_track_hits and best_conf < self._single_hit_conf_override:
                continue

            top_label, top_score, _ = candidates[0]
            label_stats = track["labels"].get(top_label, {})
            if float(label_stats.get("count", 0.0)) < settings.min_card_votes:
                continue

            track["_candidates"] = candidates
            track_accepted += 1

        ranked_tracks = sorted(
            [track for track in self._tracks[camera_id] if track.get("_candidates")],
            key=lambda t: t["_candidates"][0][1],
            reverse=True,
        )[: settings.hand_size]

        used_labels: set[str] = set()
        for track in ranked_tracks:
            chosen = None
            for candidate in track["_candidates"]:
                label = candidate[0]
                if label in used_labels:
                    continue
                chosen = candidate
                break
            if chosen is None:
                duplicate_suppressed += 1
                continue
            track["_chosen"] = chosen
            used_labels.add(chosen[0])

        slot_debug: list[dict] = []
        for track in ranked_tracks:
            if "_chosen" not in track:
                continue
            label, score, conf = track.pop("_chosen")
            workers = sorted(list(track.get("workers", set())))
            models = sorted(list(track.get("models", set())))
            scored_cards.append((int(track.get("slot_id", 0)), label, score, conf))
            slot_debug.append(
                {
                    "slot_id": int(track.get("slot_id", 0)),
                    "label": label,
                    "confidence": round(float(conf), 4),
                    "workers": workers,
                    "models": models,
                    "candidates": [
                        {"label": c_label, "score": round(float(c_score), 4), "confidence": round(float(c_conf), 4)}
                        for c_label, c_score, c_conf in track.get("_candidates", [])[:3]
                    ],
                }
            )

        for track in self._tracks[camera_id]:
            track.pop("_candidates", None)

        scored_cards.sort(key=lambda item: item[2], reverse=True)
        cards = [
            HandCard(label=label, confidence=round(conf, 3), slot_id=slot_id)
            for slot_id, label, _, conf in scored_cards[: settings.hand_size]
        ]

        self._debug[camera_id]["accepted_tracks"] = track_accepted
        self._debug[camera_id]["duplicate_labels_suppressed"] = duplicate_suppressed
        self._debug[camera_id]["slots"] = slot_debug

        return HandState(
            camera_id=camera_id,
            updated_ts_ms=self._latest_ts[camera_id],
            max_cards=settings.hand_size,
            cards=cards,
        )

    def latest_all(self) -> list[HandState]:
        camera_ids = set(self._tracks.keys()) | set(self._overrides.keys())
        return [self.latest(camera_id) for camera_id in camera_ids]

    def debug(self, camera_id: str) -> dict:
        return self._debug.get(camera_id, {})

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
fusion_engine = FusionEngine()
game_state_engine = GameStateEngine()
consume_task: asyncio.Task | None = None
recommendation_metrics = {
    "calls": 0,
    "total_ms": 0.0,
    "max_ms": 0.0,
}
recommendation_latencies_ms: deque[float] = deque(maxlen=4000)
pipeline_metrics = {
    "metrics_deduped_frames": 0,
}
event_store = EventStore()
last_ingest_ts_ms_by_camera: dict[str, int] = {}
storage_metrics = {
    "event_write_ms": 0.0,
    "transition_write_ms": 0.0,
    "recommendation_write_ms": 0.0,
}


def _build_advisor_response(
    game_id: str,
    simulation_seed: int | None = None,
    *,
    persist: bool = True,
    record_metrics: bool = True,
) -> AdvisorResponse:
    started = time.time()
    state = game_state_engine.get(game_id)
    action_scores: list[ActionScore] = evaluate_actions(state, simulation_seed=simulation_seed)
    if not action_scores:
        out = AdvisorResponse(
            best_action=AdvisorBestAction(card="", ev=0.0, confidence=0.0),
            alternatives=[],
            why=["No legal actions available from current state"],
            uncertainty_notes=["Insufficient stable game state"],
            recommendation_mode="cautious",
            freshness_ts_ms=state.updated_ts_ms,
            action_scores=[],
        )
        elapsed = (time.time() - started) * 1000.0
        if record_metrics:
            recommendation_metrics["calls"] += 1
            recommendation_metrics["total_ms"] += elapsed
            recommendation_metrics["max_ms"] = max(recommendation_metrics["max_ms"], elapsed)
            recommendation_latencies_ms.append(float(elapsed))
        if persist and bool(settings.feature_db_event_persistence):
            started = time.time()
            event_store.log_recommendation(game_id, int(time.time() * 1000), out.model_dump())
            storage_metrics["recommendation_write_ms"] = round((time.time() - started) * 1000.0, 3)
        return out

    best = action_scores[0]
    alternatives = [AdvisorAlternative(card=a.card, ev=a.total_ev) for a in action_scores[1:3]]
    belief = build_belief(state)
    belief_mass = sum(len(cards) for cards in belief.values())

    uncertainty_notes = list(state.uncertainty.notes)
    if state.uncertainty.table_conf < 0.55:
        uncertainty_notes.append("Table camera confidence is low")
    if state.uncertainty.hand_conf < 0.55:
        uncertainty_notes.append("Hand camera confidence is low")
    if belief_mass == 0:
        uncertainty_notes.append("Belief state collapsed; hidden-card uncertainty high")

    confidence = max(0.0, min(1.0, (1.0 - best.risk) * state.uncertainty.state_conf))
    mode = "normal" if confidence >= 0.60 else "cautious"

    why = [
        "Follows legal move constraints",
        "Maximizes expected value across sampled hidden-card worlds",
        "Balances immediate trick win chance and future partner support",
    ]
    if state.turn.lead_suit:
        why.insert(0, f"Lead suit is {state.turn.lead_suit}; legal filtering applied")

    out = AdvisorResponse(
        best_action=AdvisorBestAction(card=best.card, ev=best.total_ev, confidence=round(confidence, 4)),
        alternatives=alternatives,
        why=why,
        uncertainty_notes=uncertainty_notes[:8],
        recommendation_mode=mode,
        freshness_ts_ms=state.updated_ts_ms,
        action_scores=action_scores,
    )
    elapsed = (time.time() - started) * 1000.0
    if record_metrics:
        recommendation_metrics["calls"] += 1
        recommendation_metrics["total_ms"] += elapsed
        recommendation_metrics["max_ms"] = max(recommendation_metrics["max_ms"], elapsed)
        recommendation_latencies_ms.append(float(elapsed))
    if persist and bool(settings.feature_db_event_persistence):
        started = time.time()
        event_store.log_recommendation(game_id, int(time.time() * 1000), out.model_dump())
        storage_metrics["recommendation_write_ms"] = round((time.time() - started) * 1000.0, 3)
    return out


async def ensure_group(stream_name: str, group: str) -> None:
    try:
        await redis_client.xgroup_create(stream_name, group, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def _mark_frame_seen(game_id: str, frame_id: str) -> bool:
    if not frame_id:
        return True
    dedupe_key = f"seen_frame:{game_id}:{frame_id}"
    is_new = await redis_client.set(dedupe_key, "1", ex=settings.frame_id_dedupe_ttl_sec, nx=True)
    return bool(is_new)


def _redact_event(event: dict) -> dict:
    if not bool(settings.pii_redaction_enabled):
        return event
    scrubbed = dict(event)
    scrubbed.pop("frame_b64", None)
    scrubbed.pop("raw_frame", None)
    return scrubbed


def _ingest_allowed(camera_id: str, ts_ms: int) -> bool:
    fps = max(0.1, float(settings.frame_ingest_rate_limit_fps))
    min_interval_ms = int(1000.0 / fps)
    last_ts = int(last_ingest_ts_ms_by_camera.get(camera_id, 0))
    if ts_ms - last_ts < min_interval_ms:
        return False
    last_ingest_ts_ms_by_camera[camera_id] = ts_ms
    return True


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
            for msg_id, payload in messages[: max(1, int(settings.reducer_batch_size))]:
                lock_key = ""
                lock_value = ""
                try:
                    raw = payload.get("event", "{}")
                    event = json.loads(raw)
                    try:
                        parsed = DetectionEvent.model_validate(event, strict=True)
                    except ValidationError as exc:
                        event["validation_error"] = str(exc)
                        await hub.remember(event)
                        continue

                    lock_key = f"reduce_lock:{parsed.game_id}"
                    lock_value = f"{settings.backend_consumer_name}:{msg_id}"
                    lock_acquired = await redis_client.set(lock_key, lock_value, ex=settings.reducer_lock_ttl_sec, nx=True)
                    if not lock_acquired:
                        # serialization guard for multi-backend deployment
                        await asyncio.sleep(max(0.001, float(settings.reducer_lock_retry_ms) / 1000.0))
                        lock_acquired = await redis_client.set(lock_key, lock_value, ex=settings.reducer_lock_ttl_sec, nx=True)
                        if not lock_acquired:
                            continue

                    current_state = game_state_engine.get(parsed.game_id)
                    if int(parsed.ts_ms) < int(current_state.updated_ts_ms) - int(settings.reducer_max_event_lag_ms):
                        event["stale_reducer_event"] = True
                        event["stale_reducer_lag_ms"] = int(current_state.updated_ts_ms) - int(parsed.ts_ms)
                        await hub.remember(event)
                        continue

                    is_new = await _mark_frame_seen(parsed.game_id, parsed.frame_id)
                    if not is_new:
                        event["dedupe_skipped"] = True
                        pipeline_metrics["metrics_deduped_frames"] += 1
                        await hub.remember(event)
                        continue

                    fused = fusion_engine.update(
                        parsed,
                        source_worker=str(event.get("worker_name", "")),
                        source_model=str(event.get("model_name", "")),
                    )
                    game_state, transition = game_state_engine.apply_fused_observation(fused)

                    hand_state = hand_state_store.update(
                        parsed,
                        source_worker=str(event.get("worker_name", "")),
                        source_model=str(event.get("model_name", "")),
                    )
                    advice = build_thulla_advice(hand_state)
                    event["hand_state"] = hand_state.model_dump()
                    event["advice"] = advice.model_dump()
                    event["hand_debug"] = hand_state_store.debug(parsed.camera_id)
                    event["fusion_debug"] = fused.get("fusion_debug", {})
                    event["game_state"] = game_state.model_dump()
                    event["transition"] = {
                        "accepted": transition.accepted,
                        "reason": transition.reason,
                        "reason_code": transition.reason_code,
                        "reason_detail": transition.reason_detail,
                    }
                    if bool(settings.feature_db_event_persistence):
                        redacted = _redact_event(event)
                        started = time.time()
                        event_store.log_event(
                            parsed.game_id,
                            parsed.ts_ms,
                            parsed.camera_id,
                            parsed.camera_type,
                            parsed.frame_id,
                            redacted,
                        )
                        storage_metrics["event_write_ms"] = round((time.time() - started) * 1000.0, 3)
                        if bool(settings.persist_recommendation_on_transition):
                            rec = _build_advisor_response(parsed.game_id, persist=False, record_metrics=False)
                            started = time.time()
                            event_store.log_transition_and_recommendation(
                                parsed.game_id,
                                parsed.ts_ms,
                                {
                                    "accepted": transition.accepted,
                                    "reason": transition.reason,
                                    "reason_code": transition.reason_code,
                                    "reason_detail": transition.reason_detail,
                                    "state_version": int(game_state.state_version),
                                    "transition_id": str(game_state.last_transition_id),
                                    "state": game_state.model_dump(),
                                },
                                rec.model_dump(),
                            )
                            storage_metrics["transition_write_ms"] = round((time.time() - started) * 1000.0, 3)
                        else:
                            started = time.time()
                            event_store.log_transition(
                                parsed.game_id,
                                parsed.ts_ms,
                                transition.accepted,
                                transition.reason,
                                transition.reason_code,
                                transition.reason_detail,
                                int(game_state.state_version),
                                str(game_state.last_transition_id),
                                game_state.model_dump(),
                            )
                            storage_metrics["transition_write_ms"] = round((time.time() - started) * 1000.0, 3)
                    await hub.remember(event)
                    await hub.broadcast(event)
                except Exception as exc:
                    await hub.remember({
                        "error": str(exc),
                        "message_id": msg_id,
                        "camera_id": payload.get("camera_id", "unknown"),
                        "game_id": payload.get("game_id", "default-game"),
                    })
                finally:
                    if lock_key:
                        try:
                            current_value = await redis_client.get(lock_key)
                            if current_value == lock_value:
                                await redis_client.delete(lock_key)
                        except Exception:
                            pass
                    if bool(settings.feature_db_event_persistence):
                        try:
                            event_store.upsert_reducer_offset(
                                settings.backend_consumer_group,
                                settings.detection_stream,
                                msg_id,
                                int(time.time() * 1000),
                            )
                        except Exception:
                            pass
                    await redis_client.xack(stream_name, settings.backend_consumer_group, msg_id)


async def _retention_worker() -> None:
    while True:
        await asyncio.sleep(3600)
        if not bool(settings.feature_db_event_persistence):
            continue
        if int(settings.event_retention_days) <= 0 and int(settings.recommendation_retention_days) <= 0:
            continue
        try:
            event_store.purge_older_than_days(
                int(settings.event_retention_days),
                int(settings.recommendation_retention_days),
            )
        except Exception:
            pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    global consume_task
    retention_task: asyncio.Task | None = None
    if settings.app_role in {"all", "reducer"}:
        await ensure_group(settings.detection_stream, settings.backend_consumer_group)
        consume_task = asyncio.create_task(consume_detections())
    if int(settings.event_retention_days) > 0 or int(settings.recommendation_retention_days) > 0:
        retention_task = asyncio.create_task(_retention_worker())
    if bool(settings.feature_db_event_persistence):
        try:
            event_store.log_deploy_audit(
                int(time.time() * 1000),
                settings.deploy_env,
                settings.release_version,
                {"app_role": settings.app_role},
            )
        except Exception:
            pass
    try:
        yield
    finally:
        if consume_task:
            consume_task.cancel()
            with suppress(asyncio.CancelledError):
                await consume_task
        if retention_task:
            retention_task.cancel()
            with suppress(asyncio.CancelledError):
                await retention_task
        await redis_client.close()


app = FastAPI(title="CardSense Distributed Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    pong = await redis_client.ping()
    return {"ok": True, "redis": pong}


@app.get("/api/workers/status")
async def workers_status(_claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    keys = await redis_client.keys(f"{settings.worker_status_prefix}:*")
    workers = []
    for key in sorted(keys):
        data = await redis_client.hgetall(key)
        parsed = {}
        for field, value in data.items():
            try:
                parsed[field] = json.loads(value)
            except Exception:
                parsed[field] = value
        workers.append(parsed)
    return {"ok": True, "count": len(workers), "workers": workers}


async def _worker_status_rows() -> list[dict]:
    keys = await redis_client.keys(f"{settings.worker_status_prefix}:*")
    workers: list[dict] = []
    for key in sorted(keys):
        data = await redis_client.hgetall(key)
        parsed = {}
        for field, value in data.items():
            try:
                parsed[field] = json.loads(value)
            except Exception:
                parsed[field] = value
        workers.append(parsed)
    return workers


@app.get("/api/workers/logs")
async def workers_logs(limit: int = 100, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    clamped = max(1, min(limit, 500))
    entries = await redis_client.xrevrange(settings.worker_log_stream, count=clamped)
    logs = []
    for msg_id, payload in entries:
        raw = payload.get("event", "{}")
        try:
            event = json.loads(raw)
        except Exception:
            event = {"raw": raw}
        event["id"] = msg_id
        logs.append(event)
    return {"ok": True, "count": len(logs), "logs": logs}


@app.get("/api/games/{game_id}/fusion/conflicts")
async def game_fusion_conflicts(game_id: str, limit: int = 100) -> dict:
    rows = fusion_engine.recent_conflicts(game_id, limit=limit)
    return {"game_id": game_id, "count": len(rows), "conflicts": rows}


@app.get("/api/metrics")
async def metrics() -> dict:
    def _percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        idx = int(round((pct / 100.0) * (len(values) - 1)))
        idx = max(0, min(idx, len(values) - 1))
        return float(sorted(values)[idx])

    states = game_state_engine.all_states()
    cautious = 0
    normal = 0
    for state in states:
        rec = _build_advisor_response(state.game_id, persist=False, record_metrics=False)
        if rec.recommendation_mode == "cautious":
            cautious += 1
        else:
            normal += 1
    rec_calls = max(int(recommendation_metrics["calls"]), 1)
    rec_latency_avg = recommendation_metrics["total_ms"] / rec_calls
    avg_ms = float(rec_latency_avg)
    max_ms = float(recommendation_metrics["max_ms"])
    lat_samples = list(recommendation_latencies_ms)
    p95 = _percentile(lat_samples, 95.0)
    p99 = _percentile(lat_samples, 99.0)
    slo_under_300 = avg_ms <= 300.0 and max_ms <= 800.0
    latency_histogram = {
        "le_100": 1 if avg_ms <= 100 else 0,
        "le_300": 1 if avg_ms <= 300 else 0,
        "le_500": 1 if avg_ms <= 500 else 0,
        "gt_500": 1 if avg_ms > 500 else 0,
    }
    worker_rows = await _worker_status_rows()
    return {
        "games_total": len(states),
        "recommendation_mode": {"normal": normal, "cautious": cautious},
        "cautious_mode_rate": round(cautious / max(cautious + normal, 1), 6),
        "state_conf_avg": round(sum(s.uncertainty.state_conf for s in states) / max(len(states), 1), 4),
        "table_conf_avg": round(sum(s.uncertainty.table_conf for s in states) / max(len(states), 1), 4),
        "hand_conf_avg": round(sum(s.uncertainty.hand_conf for s in states) / max(len(states), 1), 4),
        "recommendation_latency_ms": {
            "avg": round(rec_latency_avg, 4),
            "max": round(float(recommendation_metrics["max_ms"]), 4),
            "p95": round(p95, 4),
            "p99": round(p99, 4),
            "calls": int(recommendation_metrics["calls"]),
            "histogram": latency_histogram,
            "slo_under_300ms_avg": slo_under_300,
        },
        "cardsense_recommendation_latency_ms_bucket": latency_histogram,
        "cardsense_transition_reject_total": game_state_engine.metrics().get("transition_reject_reason_code", {}),
        "cardsense_fusion_label_flip_rate": fusion_engine.metrics().get("label_flip_rate", 0.0),
        "cardsense_worker_skipped_out_of_order_total": sum(
            int(w.get("skipped_out_of_order", 0))
            for w in worker_rows
        ),
        "cardsense_cautious_mode_rate": round(cautious / max(cautious + normal, 1), 6),
        "fusion": fusion_engine.metrics(),
        "transitions": game_state_engine.metrics(),
        **pipeline_metrics,
        "metrics_transition_reject_reason_code": game_state_engine.metrics().get("transition_reject_reason_code", {}),
        "metrics_out_of_order_frames_skipped": sum(
            int(w.get("skipped_out_of_order", 0))
            for w in worker_rows
        ),
        "cardsense_reducer_event_lag_ms": max(
            0,
            int(time.time() * 1000)
            - max([int(s.updated_ts_ms or 0) for s in states] or [int(time.time() * 1000)]),
        ),
        "cardsense_ws_connected_clients": await hub.client_count(),
        "cardsense_queue_depth_frames": int(await redis_client.xlen(settings.frame_stream)),
        "cardsense_queue_depth_detections": int(await redis_client.xlen(settings.detection_stream)),
        "cardsense_db_write_latency_ms": storage_metrics,
    }


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    payload = await metrics()
    lines: list[str] = []
    hist = payload.get("cardsense_recommendation_latency_ms_bucket", {})
    lines.append(f"cardsense_recommendation_latency_ms_bucket{{le=\"100\"}} {float(hist.get('le_100', 0))}")
    lines.append(f"cardsense_recommendation_latency_ms_bucket{{le=\"300\"}} {float(hist.get('le_300', 0))}")
    lines.append(f"cardsense_recommendation_latency_ms_bucket{{le=\"500\"}} {float(hist.get('le_500', 0))}")
    lines.append(f"cardsense_recommendation_latency_ms_bucket{{le=\"+Inf\"}} {float(hist.get('gt_500', 0))}")
    lines.append(f"cardsense_reducer_event_lag_ms {float(payload.get('cardsense_reducer_event_lag_ms', 0))}")
    lines.append(f"cardsense_fusion_label_flip_rate {float(payload.get('cardsense_fusion_label_flip_rate', 0))}")
    lines.append(f"cardsense_worker_skipped_out_of_order_total {float(payload.get('cardsense_worker_skipped_out_of_order_total', 0))}")
    lines.append(f"cardsense_ws_connected_clients {float(payload.get('cardsense_ws_connected_clients', 0))}")
    lines.append(f"cardsense_queue_depth_frames {float(payload.get('cardsense_queue_depth_frames', 0))}")
    lines.append(f"cardsense_queue_depth_detections {float(payload.get('cardsense_queue_depth_detections', 0))}")
    lines.append(f"cardsense_cautious_mode_rate {float(payload.get('cardsense_cautious_mode_rate', 0))}")
    transition_reject = payload.get("cardsense_transition_reject_total", {})
    for reason, value in transition_reject.items():
        lines.append(f"cardsense_transition_reject_total{{reason_code=\"{reason}\"}} {float(value)}")
    dbw = payload.get("cardsense_db_write_latency_ms", {})
    for k, v in dbw.items():
        lines.append(f"cardsense_db_write_latency_ms{{phase=\"{k}\"}} {float(v)}")
    body = "\n".join(lines) + "\n"
    return Response(content=body, media_type="text/plain; version=0.0.4")


@app.post("/api/cameras/register")
async def register_camera(body: CameraRegister, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
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
async def ingest_frame(body: FrameIn, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    camera_key = f"camera:{body.camera_id}"
    if not await redis_client.exists(camera_key):
        raise HTTPException(status_code=404, detail="Camera not registered")

    queue_depth = int(await redis_client.xlen(settings.frame_stream))
    if queue_depth >= int(settings.max_queue_len):
        raise HTTPException(status_code=429, detail="Frame queue backpressure active")

    if not _ingest_allowed(body.camera_id, int(body.ts_ms)):
        raise HTTPException(status_code=429, detail="Frame ingest rate limit exceeded for camera")

    payload = {
        "game_id": body.game_id,
        "camera_id": body.camera_id,
        "camera_type": body.camera_type,
        "frame_id": body.frame_id or f"{body.camera_id}:{body.ts_ms}",
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
async def ingest_frame_batch(body: FrameBatchIn, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    queued = []
    for frame in body.frames:
        camera_key = f"camera:{frame.camera_id}"
        if not await redis_client.exists(camera_key):
            raise HTTPException(status_code=404, detail=f"Camera not registered: {frame.camera_id}")

    queue_depth = int(await redis_client.xlen(settings.frame_stream))
    if queue_depth >= int(settings.max_queue_len):
        raise HTTPException(status_code=429, detail="Frame queue backpressure active")

    pipeline = redis_client.pipeline(transaction=False)
    for frame in body.frames:
        if not _ingest_allowed(frame.camera_id, int(frame.ts_ms)):
            continue
        payload = {
            "game_id": frame.game_id,
            "camera_id": frame.camera_id,
            "camera_type": frame.camera_type,
            "frame_id": frame.frame_id or f"{frame.camera_id}:{frame.ts_ms}",
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
    return {
        "hand": hand_state_store.latest(camera_id).model_dump(),
        "debug": hand_state_store.debug(camera_id),
    }


@app.get("/api/hands/{camera_id}/advice")
async def hand_advice(camera_id: str) -> dict:
    hand = hand_state_store.latest(camera_id)
    advice: HandAdvice = build_thulla_advice(hand)
    return {"advice": advice.model_dump(), "hand": hand.model_dump()}


@app.post("/api/games/start")
async def start_game(body: GameStartIn, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    state = game_state_engine.start_game(
        game_id=body.game_id,
        leader=body.leader,
        to_play=body.to_play,
        must_follow_suit=body.must_follow_suit,
        trump_mode=body.trump_mode,
        variant_name=body.variant_name,
        must_trump_if_void=body.must_trump_if_void,
        trump_suit=body.trump_suit,
        rank_order_map=body.rank_order_map,
    )
    if bool(settings.feature_db_event_persistence):
        try:
            event_store.log_game_config(body.game_id, int(time.time() * 1000), body.model_dump())
        except Exception:
            pass
    return {"ok": True, "state": state.model_dump()}


@app.post("/api/games/{game_id}/config")
async def config_game(game_id: str, body: GameConfigIn, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    state = game_state_engine.configure(
        game_id,
        variant_name=body.variant_name,
        leader=body.leader,
        to_play=body.to_play,
        lead_suit=body.lead_suit,
        must_follow_suit=body.must_follow_suit,
        must_trump_if_void=body.must_trump_if_void,
        trump_mode=body.trump_mode,
        trump_suit=body.trump_suit,
        rank_order_map=body.rank_order_map,
    )
    if bool(settings.feature_db_event_persistence):
        try:
            event_store.log_game_config(game_id, int(time.time() * 1000), body.model_dump())
        except Exception:
            pass
    return {"ok": True, "state": state.model_dump()}


@app.post("/api/games/{game_id}/reset")
async def reset_game(game_id: str, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    state = game_state_engine.reset(game_id)
    return {"ok": True, "state": state.model_dump()}


@app.delete("/api/games/{game_id}")
async def end_game(game_id: str, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    ended = game_state_engine.end(game_id)
    return {"ok": ended}


@app.get("/api/games/{game_id}/state")
async def get_game_state(game_id: str) -> dict:
    state = game_state_engine.get(game_id)
    return {"state": state.model_dump()}


@app.get("/api/games/{game_id}/events")
async def get_game_events(game_id: str, limit: int = 200, source: str = "memory") -> dict:
    if source == "db":
        events = event_store.get_events(game_id, limit=limit)
    else:
        events = await hub.recent_for_game(game_id, limit=limit)
    return {"game_id": game_id, "count": len(events), "events": events}


@app.get("/api/games/{game_id}/export")
async def export_game_data(game_id: str, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    return event_store.export_game(game_id)


@app.delete("/api/games/{game_id}/data")
async def delete_game_data(game_id: str, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    if not bool(settings.feature_db_event_persistence):
        raise HTTPException(status_code=404, detail="Persistence disabled")
    deleted = event_store.delete_game(game_id)
    return {"ok": True, "deleted": deleted}


@app.get("/api/games/{game_id}/recommendation")
async def game_recommendation(game_id: str, simulation_seed: int | None = None) -> dict:
    rec = _build_advisor_response(game_id, simulation_seed=simulation_seed)
    return rec.model_dump()


@app.get("/api/games/{game_id}/belief")
async def game_belief(game_id: str) -> dict:
    state = game_state_engine.get(game_id)
    belief = build_belief(state)
    diagnostics = belief_diagnostics(belief)
    top_hidden = {
        seat: sorted(cards.items(), key=lambda kv: kv[1], reverse=True)[:8]
        for seat, cards in belief.items()
    }
    return {
        "game_id": game_id,
        "belief": belief,
        "top_hidden_by_seat": top_hidden,
        "diagnostics": diagnostics,
    }


@app.get("/api/games/{game_id}/simulation/debug")
async def game_simulation_debug(game_id: str, simulation_seed: int = 1337, rollout_count: int = 96) -> dict:
    state = game_state_engine.get(game_id)
    action_scores = evaluate_actions(state, rollout_count=rollout_count, simulation_seed=simulation_seed)
    return {
        "game_id": game_id,
        "simulation_seed": simulation_seed,
        "rollout_count": rollout_count,
        "action_scores": [a.model_dump() for a in action_scores],
    }


@app.post("/api/games/{game_id}/calibration/table")
async def set_table_calibration(game_id: str, body: TableCalibrationIn, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    fusion_engine.set_table_calibration(game_id, body.model_dump())
    return {"ok": True, "game_id": game_id, "calibration": fusion_engine.get_table_calibration(game_id)}


@app.get("/api/games/{game_id}/calibration/table")
async def get_table_calibration(game_id: str) -> dict:
    return {"game_id": game_id, "calibration": fusion_engine.get_table_calibration(game_id)}


@app.get("/api/games/{game_id}/diagnostics")
async def game_diagnostics(game_id: str) -> dict:
    state = game_state_engine.get(game_id)
    belief = build_belief(state)
    return {
        "game_id": game_id,
        "fusion_metrics": fusion_engine.metrics(),
        "transition_metrics": game_state_engine.metrics(),
        "recommendation_metrics": recommendation_metrics,
        "belief_metrics": belief_diagnostics(belief),
        "calibration": fusion_engine.get_table_calibration(game_id),
    }


@app.post("/api/games/{game_id}/replay")
async def replay_game(game_id: str, speed_multiplier: float = 1.0, limit: int = 1000, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    if not bool(settings.feature_replay_api):
        raise HTTPException(status_code=404, detail="Replay API disabled by feature flag")
    events = event_store.replay_game(game_id, speed_multiplier=speed_multiplier, limit=limit)
    replay_rows = events[: min(len(events), 500)]
    prev_ts = None
    rate = max(float(speed_multiplier), 0.1)
    for event in replay_rows:
        current_ts = int(event.get("ts_ms", 0) or 0)
        if prev_ts and current_ts > prev_ts:
            delta_ms = (current_ts - prev_ts) / rate
            await asyncio.sleep(min(delta_ms / 1000.0, 0.5))
        prev_ts = current_ts
        await hub.broadcast(event)
    return {
        "ok": True,
        "game_id": game_id,
        "replayed": len(replay_rows),
        "speed_multiplier": speed_multiplier,
    }


@app.post("/api/hands/{camera_id}/override")
async def set_hand_override(camera_id: str, body: HandOverrideIn, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
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
                "card_label_top1": card.label,
                "card_conf_top1": card.confidence,
                "card_probs_topk": [[card.label, card.confidence]],
                "det_conf": card.confidence,
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
async def clear_hand_override(camera_id: str, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    hand_state_store.clear_override(camera_id)
    return {"ok": True}


@app.websocket("/ws/detections")
async def detections_ws(ws: WebSocket):
    token = ws.query_params.get("token")
    try:
        ws_validate_token(token)
    except HTTPException:
        await ws.close(code=4401)
        return
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


@app.get("/api/features")
async def feature_flags() -> dict:
    return {
        "flags": FEATURE_FLAGS_METADATA,
        "release": {
            "deploy_env": settings.deploy_env,
            "release_version": settings.release_version,
            "canary_percent": settings.canary_percent,
        },
        "roles": {
            "app_role": settings.app_role,
            "worker_enabled": bool(settings.worker_enabled),
        },
    }


@app.get("/api/autoscale/suggest")
async def autoscale_suggest(_claims: dict = Depends(require_auth_roles({settings.admin_role_name}))) -> dict:
    queue_depth = int(await redis_client.xlen(settings.frame_stream))
    return compute_desired_workers(queue_depth)


frontend_path = Path(settings.static_dir)
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
