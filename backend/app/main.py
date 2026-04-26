import asyncio
import json
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .config import settings
from .advice import build_basic_hand_advice, build_hand_advice_from_recommendation
from .belief import belief_diagnostics, build_belief
from .auth_store import AuthStore
from .autoscaler import compute_desired_workers
from .feature_flags import FEATURE_FLAGS_METADATA
from .fusion import FusionEngine
from .game_state import GameStateEngine
from .llm_advisor import llm_advisor_enabled, request_llm_recommendation
from .match_engine import MatchEngine
from .models import (
    ActionScore,
    AdvisorAlternative,
    AdvisorBestAction,
    AdvisorResponse,
    CameraRegister,
    DetectionEvent,
    FrameBatchIn,
    FrameIn,
    AuthLoginIn,
    AuthLogoutIn,
    AuthRefreshIn,
    AuthRegisterIn,
    CameraBindIn,
    CameraSessionState,
    GameConfigIn,
    GameStartIn,
    HandAdvice,
    HandCard,
    HandOverrideIn,
    HandState,
    RoundStartIn,
    TableCalibrationIn,
    VariantSetIn,
)
from .redis_client import redis_client
from .security import issue_access_token, require_auth_roles, revoke_jti, ws_validate_token
from .recommendation_v2 import (
    build_belief_state,
    compute_legal_actions,
    explain_recommendation,
    sample_hidden_worlds_v2,
    score_actions_v2,
    select_recommendation,
    simulate_action_batch,
)
from .simulator import evaluate_actions
from .storage import EventStore


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
        self._consensus_window_frames = max(1, int(settings.hand_consensus_window_frames))
        self._consensus_min_samples = max(1, int(settings.hand_consensus_min_samples))
        self._consensus_min_ratio = max(0.0, min(1.0, float(settings.hand_consensus_min_ratio)))

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

    def _track_consensus(self, track: dict, label: str) -> tuple[int, int, float]:
        if not label:
            return 0, 0, 0.0
        votes = [str(v) for v in track.get("label_votes", []) if str(v)]
        samples = len(votes)
        if samples <= 0:
            return 0, 0, 0.0
        support = sum(1 for vote in votes if vote.upper() == str(label).upper())
        ratio = float(support) / float(samples)
        return support, samples, ratio

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
                    "label_votes": [],
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
            track.setdefault("label_votes", []).append(str(label))
            if len(track["label_votes"]) > self._consensus_window_frames:
                track["label_votes"] = track["label_votes"][-self._consensus_window_frames :]

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
            consensus_support, consensus_samples, consensus_ratio = self._track_consensus(track, top_label)
            if consensus_samples >= self._consensus_min_samples and consensus_ratio < self._consensus_min_ratio:
                continue

            track["_candidates"] = candidates
            track["_consensus"] = {
                "support": int(consensus_support),
                "samples": int(consensus_samples),
                "ratio": float(consensus_ratio),
            }
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
                    "consensus": {
                        "support": int(track.get("_consensus", {}).get("support", 0)),
                        "samples": int(track.get("_consensus", {}).get("samples", 0)),
                        "ratio": round(float(track.get("_consensus", {}).get("ratio", 0.0)), 4),
                    },
                }
            )

        for track in self._tracks[camera_id]:
            track.pop("_candidates", None)
            track.pop("_consensus", None)

        scored_cards.sort(key=lambda item: item[2], reverse=True)
        cards = [
            HandCard(label=label, confidence=round(conf, 3), slot_id=slot_id)
            for slot_id, label, _, conf in scored_cards[: settings.hand_size]
        ]

        self._debug[camera_id]["accepted_tracks"] = track_accepted
        self._debug[camera_id]["duplicate_labels_suppressed"] = duplicate_suppressed
        self._debug[camera_id]["consensus_window_frames"] = int(self._consensus_window_frames)
        self._debug[camera_id]["consensus_min_samples"] = int(self._consensus_min_samples)
        self._debug[camera_id]["consensus_min_ratio"] = round(float(self._consensus_min_ratio), 4)
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
auth_store = AuthStore()
match_engine = MatchEngine()
camera_sessions: dict[str, dict[str, CameraSessionState]] = defaultdict(dict)
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
    if bool(settings.feature_policy_simulator_v2):
        legal = compute_legal_actions(state, "me")
        belief_state = build_belief_state(state, history=[h.model_dump() for h in state.played_history], config={})
        worlds = sample_hidden_worlds_v2(state, belief_state, n=64, seed=simulation_seed)
        simulated = simulate_action_batch(state, legal, worlds, seed=simulation_seed)
        action_scores = score_actions_v2(state, legal, simulated, belief_state)
    else:
        legal = evaluate_actions(state, simulation_seed=simulation_seed)
        action_scores = list(legal)
        belief_state = {"belief": build_belief(state)}

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

    legal_now = compute_legal_actions(state, "me")
    best = select_recommendation(action_scores, legal_now)
    if best is None:
        out = AdvisorResponse(
            best_action=AdvisorBestAction(card="", ev=0.0, confidence=0.0),
            alternatives=[],
            why=["No legal recommendation available"],
            uncertainty_notes=["All candidate actions failed legal filter"],
            recommendation_mode="cautious",
            freshness_ts_ms=state.updated_ts_ms,
            action_scores=[],
        )
        if persist and bool(settings.feature_db_event_persistence):
            started = time.time()
            event_store.log_recommendation(game_id, int(time.time() * 1000), out.model_dump())
            storage_metrics["recommendation_write_ms"] = round((time.time() - started) * 1000.0, 3)
        return out

    action_by_card = {str(a.card).upper(): a for a in action_scores}
    belief = belief_state.get("belief", {})
    belief_mass = sum(len(cards) for cards in belief.values())

    uncertainty_notes = list(state.uncertainty.notes)
    if state.uncertainty.table_conf < 0.55:
        uncertainty_notes.append("Table camera confidence is low")
    if state.uncertainty.hand_conf < 0.55:
        uncertainty_notes.append("Hand camera confidence is low")
    if belief_mass == 0:
        uncertainty_notes.append("Belief state collapsed; hidden-card uncertainty high")

    confidence = max(0.0, min(1.0, float((1.0 - best.risk) * state.uncertainty.state_conf)))
    forced_cautious = False

    # If required camera roles are not bound, force cautious mode.
    sessions = list(camera_sessions.get(game_id, {}).values())
    bound_types = {str(s.camera_type or "").lower() for s in sessions}
    required_types = {"hand", "table"}
    missing = sorted(list(required_types - bound_types))
    if missing:
        forced_cautious = True
        uncertainty_notes.append(f"Missing required camera bindings: {', '.join(missing)}")

    # If a bound camera has gone stale, force cautious mode.
    stale_threshold_ms = int(settings.camera_stale_threshold_ms)
    now_ms = int(time.time() * 1000)
    stale_bound = [
        s for s in sessions if int(s.last_frame_ts_ms or 0) > 0 and (now_ms - int(s.last_frame_ts_ms or 0)) > stale_threshold_ms
    ]
    if stale_bound:
        forced_cautious = True
        uncertainty_notes.append("One or more bound cameras are stale")

    why, v2_notes = explain_recommendation(state, best, belief_state)
    uncertainty_notes.extend(v2_notes)
    if state.turn.lead_suit:
        why.insert(0, f"Lead suit is {state.turn.lead_suit}; legal filtering applied")

    llm_selected = False
    llm_result = request_llm_recommendation(state, legal_now, action_scores, fallback_card=str(best.card))
    if llm_result:
        llm_best = action_by_card.get(str(llm_result.get("best_card", "")).upper())
        if llm_best is not None:
            best = llm_best
            llm_selected = True

            llm_why = [str(row).strip() for row in llm_result.get("why", []) if str(row).strip()]
            if llm_why:
                why = llm_why[:5]

            llm_notes = [str(row).strip() for row in llm_result.get("uncertainty_notes", []) if str(row).strip()]
            uncertainty_notes.extend(llm_notes[:4])

            llm_conf = max(0.0, min(1.0, float(llm_result.get("confidence", 0.0))))
            confidence = max(0.0, min(1.0, (0.65 * confidence) + (0.35 * llm_conf)))

    if llm_advisor_enabled() and not llm_selected:
        uncertainty_notes.append("LLM advisor unavailable; deterministic fallback used")

    if llm_selected:
        alt_cards = list(llm_result.get("alternatives", [])) if isinstance(llm_result, dict) else []
        alternatives = [
            AdvisorAlternative(card=row.card, ev=row.total_ev)
            for row in [action_by_card.get(str(card).upper()) for card in alt_cards]
            if row is not None and str(row.card).upper() != str(best.card).upper()
        ]
    else:
        alternatives = []

    if not alternatives:
        alternatives = [
            AdvisorAlternative(card=a.card, ev=a.total_ev)
            for a in action_scores
            if str(a.card).upper() != str(best.card).upper()
        ][:2]

    mode = "normal" if confidence >= 0.60 else "cautious"
    if forced_cautious:
        mode = "cautious"

    seen_notes: set[str] = set()
    deduped_uncertainty: list[str] = []
    for note in uncertainty_notes:
        item = str(note).strip()
        if not item or item in seen_notes:
            continue
        seen_notes.add(item)
        deduped_uncertainty.append(item)

    out = AdvisorResponse(
        best_action=AdvisorBestAction(card=best.card, ev=best.total_ev, confidence=round(float(confidence), 4)),
        alternatives=alternatives,
        why=why,
        uncertainty_notes=deduped_uncertainty[:8],
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


def select_worker_route(camera_type: str, target_model_type: str | None, worker_capabilities: list[dict] | None = None) -> str:
    model_type = (target_model_type or camera_type or "hand").lower()
    if worker_capabilities:
        for worker in worker_capabilities:
            if model_type == "hand" and bool(worker.get("supports_hand", False)):
                return "route:hand"
            if model_type == "table" and bool(worker.get("supports_table", False)):
                return "route:table"
    return f"route:{model_type if model_type in {'hand', 'table'} else 'hand'}"


def _touch_camera_session(game_id: str, camera_id: str, ts_ms: int) -> None:
    session = camera_sessions.get(game_id, {}).get(camera_id)
    if session is None:
        return
    last_ts = int(session.last_frame_ts_ms or 0)
    session.camera_online = True
    session.last_frame_ts_ms = int(ts_ms)
    if last_ts > 0 and ts_ms > last_ts:
        dt = max(1, int(ts_ms) - last_ts)
        session.frame_rate_actual = round(1000.0 / float(dt), 3)
        expected = max(float(settings.frame_ingest_rate_limit_fps), 0.001)
        observed = max(float(session.frame_rate_actual), 0.0)
        session.drop_rate = round(max(0.0, min(1.0, (expected - observed) / expected)), 4)
    latency_ms = max(0, int(time.time() * 1000) - int(ts_ms))
    # Rolling estimate without unbounded history.
    session.latency_ms_p50 = round((float(session.latency_ms_p50) * 0.85) + (float(latency_ms) * 0.15), 3)
    session.latency_ms_p95 = round(max(float(session.latency_ms_p95) * 0.90, float(latency_ms)), 3)


def _validate_camera_bind_request(game_id: str, camera_id: str, camera_type: str) -> None:
    sessions = camera_sessions.get(game_id, {})
    existing = sessions.get(camera_id)
    requested_type = str(camera_type or "").strip().lower()

    if existing is None and len(sessions) >= 2:
        raise HTTPException(status_code=400, detail="A game can bind at most two cameras")

    for existing_camera_id, session in sessions.items():
        if existing_camera_id == camera_id:
            continue
        if str(session.camera_type or "").strip().lower() == requested_type:
            raise HTTPException(status_code=400, detail=f"Only one {requested_type} camera can be bound per game")


def _require_bound_camera(game_id: str, camera_id: str, camera_type: str) -> None:
    if not bool(settings.feature_enforce_camera_binding):
        return
    session = camera_sessions.get(game_id, {}).get(camera_id)
    if session is None:
        raise HTTPException(status_code=403, detail="Camera must be bound to game before ingest")
    if str(session.camera_type) != str(camera_type):
        raise HTTPException(status_code=403, detail="Bound camera type mismatch")


def _bound_camera_game_and_seat(camera_id: str) -> tuple[str, str]:
    for game_id, sessions in camera_sessions.items():
        session = sessions.get(camera_id)
        if session is None:
            continue
        return game_id, str(session.seat_hint or "")
    return "", ""


def _card_tracking_sets(match_state) -> dict:
    if match_state is None:
        return {
            "seen_cards": [],
            "in_hand_cards": [],
            "on_table_cards": [],
            "won_pile_cards": [],
            "unknown_cards": [],
            "played_history_cards": [],
        }
    in_hand = sorted([card for card, entry in match_state.card_lifecycle.items() if str(entry.location) == "in_hand"])
    on_table = sorted([card for card, entry in match_state.card_lifecycle.items() if str(entry.location) == "on_table"])
    won_pile = sorted([card for card, entry in match_state.card_lifecycle.items() if str(entry.location) == "won_pile"])
    unknown = sorted([card for card, entry in match_state.card_lifecycle.items() if str(entry.location) == "unknown"])
    played = sorted({str(row.card) for row in match_state.played_history if str(row.card)})
    seen = sorted(set(in_hand) | set(on_table) | set(won_pile) | set(unknown) | set(played))
    return {
        "seen_cards": seen,
        "in_hand_cards": in_hand,
        "on_table_cards": on_table,
        "won_pile_cards": won_pile,
        "unknown_cards": unknown,
        "played_history_cards": played,
    }


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

                    observed_hand = [
                        {
                            "card": str(item.get("label", "")),
                            "seat": "me",
                            "confidence": float(item.get("confidence", 0.0)),
                            "ts_ms": int(fused.get("ts_ms", parsed.ts_ms)),
                            "track_id": int(item.get("track_id", 0) or 0),
                        }
                        for item in fused.get("hand_cards", [])
                        if item.get("label")
                    ]
                    observed_table = [
                        {
                            "card": str(item.get("card", "")),
                            "seat": str(item.get("seat", "")),
                            "confidence": float(item.get("confidence", 0.0)),
                            "ts_ms": int(fused.get("ts_ms", parsed.ts_ms)),
                            "track_id": int(item.get("track_id", 0) or 0),
                        }
                        for item in fused.get("table_trick_cards", [])
                        if item.get("card")
                    ]

                    match_state = match_engine.get(parsed.game_id)
                    if match_state is None:
                        match_state = match_engine.start_match(
                            parsed.game_id,
                            variant=VariantConfig(),
                            initial_turn=str(game_state.turn.to_play or "me"),
                        )

                    previous_match_snapshot = match_state.model_copy(deep=True)

                    _, transition_events, reconciliation_debug = match_engine.reconcile_card_locations(
                        game_id=parsed.game_id,
                        ts_ms=int(fused.get("ts_ms", parsed.ts_ms)),
                        hand_cards_observed=observed_hand,
                        table_cards_observed=observed_table,
                        prev_lifecycle=dict(match_state.card_lifecycle),
                        to_play_seat=str(match_state.to_play_seat),
                        trick_context={"replay_mode": bool(settings.allow_won_pile_replay)},
                        variant=match_state.variant,
                    )
                    updated_match_state, invariant_violations = match_engine.apply_transition_events(
                        match_state,
                        transition_events,
                        int(fused.get("ts_ms", parsed.ts_ms)),
                    )
                    if not bool(settings.feature_match_state_authoritative):
                        hand_delta = {
                            seat: {
                                "prev": len(previous_match_snapshot.hand_cards_by_seat.get(seat, [])),
                                "next": len(updated_match_state.hand_cards_by_seat.get(seat, [])),
                            }
                            for seat in updated_match_state.variant.seat_order
                        }
                        match_transition_drift = {
                            "to_play_changed": previous_match_snapshot.to_play_seat != updated_match_state.to_play_seat,
                            "lead_suit_changed": previous_match_snapshot.lead_suit != updated_match_state.lead_suit,
                            "hand_count_delta": hand_delta,
                        }
                        match_engine._states[parsed.game_id] = previous_match_snapshot
                    else:
                        match_transition_drift = {}
                    match_transition = {
                        "accepted": not bool(invariant_violations),
                        "reason": "applied" if not invariant_violations else "invariant_rejected",
                        "reason_code": "ok" if not invariant_violations else "invariant_violation",
                        "violations": invariant_violations,
                        "transition_events": transition_events,
                        "reconciliation_debug": reconciliation_debug,
                        "drift": match_transition_drift,
                        "to_play_seat": updated_match_state.to_play_seat,
                        "lead_suit": updated_match_state.lead_suit,
                    }

                    hand_state = hand_state_store.update(
                        parsed,
                        source_worker=str(event.get("worker_name", "")),
                        source_model=str(event.get("model_name", "")),
                    )
                    rec_for_event = _build_advisor_response(parsed.game_id, persist=False, record_metrics=False)
                    advice = build_hand_advice_from_recommendation(hand_state, rec_for_event)
                    event["hand_state"] = hand_state.model_dump()
                    event["advice"] = advice.model_dump()
                    event["recommendation"] = rec_for_event.model_dump()
                    event["hand_debug"] = hand_state_store.debug(parsed.camera_id)
                    event["fusion_debug"] = fused.get("fusion_debug", {})
                    event["game_state"] = game_state.model_dump()
                    event["transition"] = {
                        "accepted": transition.accepted,
                        "reason": transition.reason,
                        "reason_code": transition.reason_code,
                        "reason_detail": transition.reason_detail,
                    }
                    event["match_transition"] = match_transition
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
                                rec_for_event.model_dump(),
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

    # In prod, do not allow the default symmetric secret.
    if str(settings.deploy_env).lower() in {"prod", "production"} and bool(settings.auth_prod_secret_required):
        if (not settings.jwt_private_key_pem) and str(settings.jwt_hs256_secret).strip() == "dev-secret-change-me":
            raise RuntimeError("Refusing startup in production with default JWT secret")

    if bool(settings.auth_enabled) and bool(settings.auth_strict_iss_aud) and str(settings.deploy_env).lower() in {"prod", "production"}:
        if not str(settings.jwt_issuer or "").strip() or not str(settings.jwt_audience or "").strip():
            raise RuntimeError("Strict auth in production requires JWT_ISSUER and JWT_AUDIENCE")

    advisor_mode = str(settings.advisor_mode or "rules").strip().lower()
    llm_mode_requested = advisor_mode in {"llm", "hybrid"} and bool(settings.llm_enabled)
    if llm_mode_requested and not str(settings.llm_api_key or "").strip():
        raise RuntimeError("LLM mode is enabled but LLM_API_KEY is missing")

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
    advisor_mode = str(settings.advisor_mode or "rules").strip().lower()
    llm_mode_requested = advisor_mode in {"llm", "hybrid"} and bool(settings.llm_enabled)
    return {
        "ok": True,
        "redis": pong,
        "ingest": {
            "frame_ingest_rate_limit_fps": float(settings.frame_ingest_rate_limit_fps),
        },
        "llm": {
            "mode": advisor_mode,
            "enabled": llm_mode_requested,
            "ready": bool(llm_advisor_enabled()),
            "provider_base": str(settings.llm_api_base),
            "model": str(settings.llm_model),
        },
    }


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
    now = int(time.time() * 1000)
    all_sessions = [s for group in camera_sessions.values() for s in group.values()]
    stale_count = sum(
        1
        for s in all_sessions
        if int(s.last_frame_ts_ms or 0) > 0 and (now - int(s.last_frame_ts_ms or 0)) > int(settings.camera_stale_threshold_ms)
    )
    stale_rate = round(float(stale_count) / max(len(all_sessions), 1), 6)
    queue_depth_frames = int(await redis_client.xlen(settings.frame_stream))
    queue_lag_ratio = round(queue_depth_frames / max(int(settings.max_queue_len), 1), 6)
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
        "cardsense_queue_depth_frames": queue_depth_frames,
        "cardsense_queue_depth_detections": int(await redis_client.xlen(settings.detection_stream)),
        "cardsense_stale_camera_rate": stale_rate,
        "cardsense_queue_lag_ratio": queue_lag_ratio,
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
    lines.append(f"cardsense_stale_camera_rate {float(payload.get('cardsense_stale_camera_rate', 0))}")
    lines.append(f"cardsense_queue_lag_ratio {float(payload.get('cardsense_queue_lag_ratio', 0))}")
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


@app.post("/api/auth/register")
async def auth_register(body: AuthRegisterIn) -> dict:
    try:
        created = auth_store.create_user_account(body.email, body.password, body.display_name)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Email already registered")
    return {"ok": True, **created}


@app.post("/api/auth/login")
async def auth_login(body: AuthLoginIn, user_agent: str | None = Query(default=None), ip: str | None = Query(default=None)) -> dict:
    auth = auth_store.authenticate_user(body.email, body.password)
    if not auth.get("auth_ok"):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user = auth["user"]
    roles = list(auth.get("roles", []))
    session_id = f"sess-{user['id']}-{int(time.time())}"
    access = issue_access_token(int(user["id"]), roles, session_id, int(settings.access_token_ttl_sec))
    refresh = auth_store.issue_refresh_token(
        int(user["id"]),
        session_id,
        int(settings.refresh_token_ttl_sec),
        {"ip": ip or "", "user_agent": user_agent or ""},
    )
    return {
        "ok": True,
        "user": user,
        "roles": roles,
        "access_token": access["jwt_access_token"],
        "access_exp": access["exp_ts"],
        "refresh_token": refresh["refresh_token"],
        "refresh_exp": int(refresh["expires_at"]),
        "session_id": session_id,
    }


@app.post("/api/auth/refresh")
async def auth_refresh(body: AuthRefreshIn, user_agent: str | None = Query(default=None), ip: str | None = Query(default=None)) -> dict:
    rotated = auth_store.rotate_refresh_token(
        body.refresh_token,
        int(settings.refresh_token_ttl_sec),
        {"ip": ip or "", "user_agent": user_agent or ""},
    )
    if not rotated.get("ok"):
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    user_id = int(rotated["user_id"])
    user_roles = auth_store.get_user_and_roles(user_id)
    if user_roles is None:
        raise HTTPException(status_code=401, detail="User not found")
    if str(user_roles["user"].get("status", "")).lower() != "active":
        raise HTTPException(status_code=403, detail="User blocked")
    roles = list(user_roles.get("roles", [])) or [settings.operator_role_name]
    access = issue_access_token(user_id, roles, str(rotated["session_id"]), int(settings.access_token_ttl_sec))
    return {
        "ok": True,
        "access_token": access["jwt_access_token"],
        "access_exp": access["exp_ts"],
        "refresh_token": rotated["refresh_token"],
        "refresh_exp": int(rotated["expires_at"]),
        "session_id": rotated["session_id"],
    }


@app.post("/api/auth/logout")
async def auth_logout(body: AuthLogoutIn, claims: dict = Depends(require_auth_roles(set()))) -> dict:
    ok = auth_store.revoke_refresh_token(body.refresh_token)
    session_id = str(claims.get("session_id", "") or "")
    revoked_session_tokens = auth_store.revoke_session_tokens(session_id) if session_id else 0
    jti = str(claims.get("jti", "") or "")
    if jti:
        revoke_jti(jti)
    return {"ok": bool(ok), "revoked_session_tokens": int(revoked_session_tokens)}


@app.get("/api/auth/me")
async def auth_me(claims: dict = Depends(require_auth_roles(set()))) -> dict:
    return {"ok": True, "claims": claims}


@app.post("/api/auth/users/{user_id}/status")
async def auth_set_user_status(
    user_id: int,
    status: str = Query(...),
    _claims: dict = Depends(require_auth_roles({settings.admin_role_name})),
) -> dict:
    ok = auth_store.set_user_status(int(user_id), status)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid user status")
    return {"ok": True, "user_id": int(user_id), "status": status}


@app.post("/api/auth/users/{user_id}/roles")
async def auth_assign_role(
    user_id: int,
    role: str = Query(...),
    _claims: dict = Depends(require_auth_roles({settings.admin_role_name})),
) -> dict:
    ok = auth_store.assign_role(int(user_id), role)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid role")
    return {"ok": True, "user_id": int(user_id), "role": role}


@app.post("/api/games/{game_id}/cameras/bind")
async def bind_camera_session(game_id: str, body: CameraBindIn, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    camera_key = f"camera:{body.camera_id}"
    if not await redis_client.exists(camera_key):
        raise HTTPException(status_code=400, detail="Camera not registered. Call /api/cameras/register first")

    _validate_camera_bind_request(game_id, body.camera_id, body.camera_type)

    session = CameraSessionState(
        game_id=game_id,
        camera_id=body.camera_id,
        camera_type=body.camera_type,
        seat_hint=body.seat_hint,
        device_label=body.device_label,
        stream_profile=body.stream_profile,
        camera_online=False,
    )
    camera_sessions[game_id][body.camera_id] = session
    return {"ok": True, "camera_session": session.model_dump()}


@app.get("/api/games/{game_id}/cameras")
async def list_camera_sessions(game_id: str, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    rows = [s.model_dump() for s in camera_sessions.get(game_id, {}).values()]
    return {"ok": True, "game_id": game_id, "count": len(rows), "camera_sessions": rows}


@app.post("/api/frames")
async def ingest_frame(body: FrameIn, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    camera_key = f"camera:{body.camera_id}"
    if not await redis_client.exists(camera_key):
        raise HTTPException(status_code=404, detail="Camera not registered")

    if not _ingest_allowed(body.camera_id, int(body.ts_ms)):
        raise HTTPException(status_code=429, detail="Frame ingest rate limit exceeded for camera")

    _require_bound_camera(body.game_id, body.camera_id, body.camera_type)

    _touch_camera_session(body.game_id, body.camera_id, int(body.ts_ms))

    payload = {
        "game_id": body.game_id,
        "camera_id": body.camera_id,
        "camera_type": body.camera_type,
        "frame_id": body.frame_id or f"{body.camera_id}:{body.ts_ms}",
        "target_model_type": body.target_model_type or body.camera_type,
        "device_id": body.device_id or "",
        "session_id": body.session_id or "",
        "routing_key": select_worker_route(body.camera_type, body.target_model_type),
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
        _require_bound_camera(frame.game_id, frame.camera_id, frame.camera_type)

    pipeline = redis_client.pipeline(transaction=False)
    for frame in body.frames:
        if not _ingest_allowed(frame.camera_id, int(frame.ts_ms)):
            continue
        _touch_camera_session(frame.game_id, frame.camera_id, int(frame.ts_ms))
        payload = {
            "game_id": frame.game_id,
            "camera_id": frame.camera_id,
            "camera_type": frame.camera_type,
            "frame_id": frame.frame_id or f"{frame.camera_id}:{frame.ts_ms}",
            "target_model_type": frame.target_model_type or frame.camera_type,
            "device_id": frame.device_id or "",
            "session_id": frame.session_id or "",
            "routing_key": select_worker_route(frame.camera_type, frame.target_model_type),
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
    bound_game_id, seat_hint = _bound_camera_game_and_seat(camera_id)

    if bound_game_id and seat_hint:
        projected = match_engine.project_hand_state_for_camera(bound_game_id, camera_id, seat_hint)
        if bool(settings.feature_match_state_authoritative) or projected.cards:
            return {
                "hand": projected.model_dump(),
                "debug": {
                    **hand_state_store.debug(camera_id),
                    "source": "match_state_authoritative",
                    "game_id": bound_game_id,
                    "seat_hint": seat_hint,
                },
            }

    return {
        "hand": hand_state_store.latest(camera_id).model_dump(),
        "debug": hand_state_store.debug(camera_id),
    }


@app.get("/api/hands/{camera_id}/advice")
async def hand_advice(camera_id: str) -> dict:
    hand = hand_state_store.latest(camera_id)
    bound_game_id, _ = _bound_camera_game_and_seat(camera_id)
    if bound_game_id:
        rec = _build_advisor_response(bound_game_id)
        advice: HandAdvice = build_hand_advice_from_recommendation(hand, rec)
        return {"advice": advice.model_dump(), "hand": hand.model_dump(), "game_id": bound_game_id}
    advice = build_basic_hand_advice(hand)
    return {"advice": advice.model_dump(), "hand": hand.model_dump(), "game_id": ""}


@app.post("/api/games/start")
async def start_game(body: GameStartIn, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    state = game_state_engine.start_game(
        game_id=body.game_id,
        leader=body.leader,
        to_play=body.to_play,
        must_follow_suit=body.must_follow_suit,
        trump_mode=body.trump_mode,
        variant_name=body.variant_name,
        seat_order=body.seat_order,
        play_direction=body.play_direction,
        trick_size=body.trick_size,
        cards_per_player=body.cards_per_player,
        must_trump_if_void=body.must_trump_if_void,
        trump_suit=body.trump_suit,
        rank_order_map=body.rank_order_map,
    )
    if bool(settings.feature_db_event_persistence):
        try:
            event_store.log_game_config(body.game_id, int(time.time() * 1000), body.model_dump())
        except Exception:
            pass
    # Initialize v2 match engine state in parallel with canonical reducer state
    try:
        from .models import VariantConfig

        variant = VariantConfig(
            variant_id=body.variant_name,
            player_count=body.player_count,
            seat_order=body.seat_order,
            play_direction=body.play_direction,
            cards_per_player=body.cards_per_player,
            trick_size=body.trick_size,
            rounds_total=body.rounds_total,
            target_points=body.target_points,
            must_follow_suit=body.must_follow_suit,
            must_trump_if_void=body.must_trump_if_void,
            trump_suit=body.trump_suit,
            rank_order_map=body.rank_order_map or state.rules.rank_order_map,
            team_map=body.team_map,
            deck_cards=list(body.deck_cards or []),
        )
        match_state = match_engine.start_match(body.game_id, variant, body.to_play)
    except Exception:
        match_state = None
    return {"ok": True, "state": state.model_dump(), "match_state": match_state.model_dump() if match_state else None}


@app.post("/api/games/{game_id}/variant")
async def set_game_variant(game_id: str, body: VariantSetIn, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    out = match_engine.set_variant_config(game_id, body.variant)
    return {"ok": True, **out}


@app.post("/api/games/{game_id}/round/start")
async def start_game_round(game_id: str, body: RoundStartIn, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    state = match_engine.start_round(game_id, body)
    return {"ok": True, "match_state": state.model_dump()}


@app.post("/api/games/{game_id}/round/end")
async def end_game_round(game_id: str, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    state = match_engine.end_round(game_id)
    return {"ok": True, "match_state": state.model_dump()}


@app.post("/api/games/{game_id}/config")
async def config_game(game_id: str, body: GameConfigIn, _claims: dict = Depends(require_auth_roles({settings.operator_role_name, settings.admin_role_name}))) -> dict:
    state = game_state_engine.configure(
        game_id,
        variant_name=body.variant_name,
        seat_order=body.seat_order,
        play_direction=body.play_direction,
        trick_size=body.trick_size,
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
    match_state = match_engine.get(game_id)
    card_sets = _card_tracking_sets(match_state)
    return {
        "state": state.model_dump(),
        "match_state": match_state.model_dump() if match_state else None,
        "variant": {
            "cards_per_player": int((match_state.variant.cards_per_player if match_state else settings.default_cards_per_player) or settings.default_cards_per_player),
        },
        "hand_cards_by_seat": (match_state.hand_cards_by_seat if match_state else {}),
        "card_lifecycle": ({k: v.model_dump() for k, v in match_state.card_lifecycle.items()} if match_state else {}),
        "card_tracking_sets": card_sets,
    }


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
    match_state = match_engine.get(game_id)
    match_metrics = match_engine.metrics(game_id)
    return {
        "game_id": game_id,
        "fusion_metrics": fusion_engine.metrics(),
        "transition_metrics": game_state_engine.metrics(),
        "recommendation_metrics": recommendation_metrics,
        "belief_metrics": belief_diagnostics(belief),
        "calibration": fusion_engine.get_table_calibration(game_id),
        "camera_sessions": [s.model_dump() for s in camera_sessions.get(game_id, {}).values()],
        "card_tracking_sets": _card_tracking_sets(match_state),
        "card_lifecycle_summary": {
            "tracked_cards": len((match_state.card_lifecycle if match_state else {})),
        },
        "rule_violations_recent": (match_state.rule_violations_recent if match_state else []),
        "invariant_violations_recent": match_metrics.get("invariant_violations_recent", []),
        "card_duplicate_rejected_total": match_metrics.get("card_duplicate_rejected_total", 0),
        "expected_vs_actual_hand_count": match_metrics.get("expected_vs_actual_hand_count", {}),
        "transition_latency_ms": storage_metrics,
    }


@app.get("/api/games/{game_id}/setup/diagnostics")
async def game_setup_diagnostics(game_id: str) -> dict:
    sessions = [s.model_dump() for s in camera_sessions.get(game_id, {}).values()]
    now = int(time.time() * 1000)
    bound_types = sorted({str(row.get("camera_type", "")).strip().lower() for row in sessions if row.get("camera_type")})
    required_types = {"hand", "table"}
    missing_types = sorted(list(required_types - set(bound_types)))
    stale = [
        row
        for row in sessions
        if int(row.get("last_frame_ts_ms", 0) or 0) > 0
        and (now - int(row.get("last_frame_ts_ms", 0))) > int(settings.camera_stale_threshold_ms)
    ]
    readiness = "ready" if len(sessions) == 2 and not missing_types and not stale else "degraded"
    return {
        "ok": True,
        "game_id": game_id,
        "readiness": readiness,
        "required_camera_count": 2,
        "required_camera_types": ["hand", "table"],
        "bound_camera_types": bound_types,
        "missing_camera_types": missing_types,
        "bound_camera_count": len(sessions),
        "stale_camera_count": len(stale),
        "stale_camera_ids": [s.get("camera_id", "") for s in stale],
        "camera_sessions": sessions,
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
    bound_game_id, _ = _bound_camera_game_and_seat(camera_id)
    if bound_game_id:
        rec = _build_advisor_response(bound_game_id, persist=False)
        advice = build_hand_advice_from_recommendation(hand, rec)
    else:
        advice = build_basic_hand_advice(hand)
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
        # Keep UI stream stable even if a stale/invalid token is present.
        pass
    await hub.connect(ws)

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(25)
            await ws.send_json({"type": "ping", "ts_ms": int(time.time() * 1000)})

    heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        warmup = await hub.latest()
        for event in warmup:
            await ws.send_json(event)

        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task
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
