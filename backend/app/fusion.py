from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math

from .models import Detection, DetectionEvent


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


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in dist.values())
    if total <= 0:
        return {}
    return {k: max(v, 0.0) / total for k, v in dist.items()}


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    last_ts: int
    hits: int = 1
    posterior: Dict[str, float] = field(default_factory=dict)
    det_conf_avg: float = 0.0
    source_workers: Dict[str, int] = field(default_factory=dict)
    source_models: Dict[str, int] = field(default_factory=dict)
    miss_count: int = 0
    state: str = "active"

    def top_label(self) -> tuple[str, float]:
        if not self.posterior:
            return "", 0.0
        label, prob = max(self.posterior.items(), key=lambda kv: kv[1])
        return label, float(prob)

    def stability(self) -> float:
        # light stability proxy combining repeat hits + certainty
        _, top_p = self.top_label()
        return min(1.0, 0.15 * self.hits + 0.85 * top_p)


class FusionEngine:
    def __init__(self) -> None:
        self._tracks: Dict[tuple[str, str], List[Track]] = defaultdict(list)  # (game_id, camera_type)
        self._next_id: Dict[tuple[str, str], int] = defaultdict(int)
        self.track_match_iou = 0.45
        self.frame_dedupe_iou = 0.60
        self.track_stale_ms = 5000
        self._metrics = {
            "events_total": 0,
            "label_flip_events": 0,
            "uniqueness_conflicts": 0,
        }
        self._recent_conflicts: Dict[str, List[dict]] = defaultdict(list)
        self.hysteresis_hits_threshold = 2
        self.hysteresis_prob_threshold = 0.58
        self.posterior_entropy_threshold = 1.8
        self.freeze_miss_threshold = 2
        self.expire_miss_threshold = 6
        self.worker_reliability_weight = 0.05
        self.model_reliability_weight = 0.05
        self.source_reliability_worker: Dict[str, float] = defaultdict(lambda: 1.0)
        self.source_reliability_model: Dict[str, float] = defaultdict(lambda: 1.0)
        self._calibration: Dict[str, dict] = defaultdict(dict)
        self._occluded_tracks: Dict[tuple[str, str], Dict[int, int]] = defaultdict(dict)
        self.occlusion_timeout_ms = 2000

    def set_table_calibration(self, game_id: str, calibration: dict) -> None:
        self._calibration[game_id] = dict(calibration)
        if calibration.get("occlusion_timeout_ms") is not None:
            self.occlusion_timeout_ms = int(calibration.get("occlusion_timeout_ms") or self.occlusion_timeout_ms)

    def get_table_calibration(self, game_id: str) -> dict:
        return dict(self._calibration.get(game_id, {}))

    @staticmethod
    def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

    @staticmethod
    def _point_in_rect(point: tuple[float, float], rect_points: list[list[float]]) -> bool:
        if len(rect_points) < 2:
            return False
        xs = [float(p[0]) for p in rect_points]
        ys = [float(p[1]) for p in rect_points]
        x, y = point
        return min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys)

    def _seat_from_calibration(self, game_id: str, bbox: tuple[float, float, float, float], fallback: str) -> str:
        cfg = self._calibration.get(game_id, {})
        seat_regions = cfg.get("seat_regions", {}) if isinstance(cfg, dict) else {}
        if not isinstance(seat_regions, dict) or not seat_regions:
            return fallback
        center = self._center(bbox)
        for seat, region in seat_regions.items():
            if self._point_in_rect(center, region):
                return str(seat)
        return fallback

    @staticmethod
    def _entropy(dist: Dict[str, float]) -> float:
        if not dist:
            return 0.0
        e = 0.0
        for p in dist.values():
            p = max(min(float(p), 1.0), 1e-12)
            e -= p * math.log(p, 2)
        return e

    def _detection_probs(self, det: Detection) -> Dict[str, float]:
        probs: Dict[str, float] = {}
        if det.card_probs_topk:
            for pair in det.card_probs_topk[:5]:
                if len(pair) != 2:
                    continue
                label = str(pair[0])
                try:
                    prob = float(pair[1])
                except Exception:
                    continue
                probs[label] = max(probs.get(label, 0.0), prob)
        if not probs:
            label = det.card_label_top1 or det.cls_name
            if label:
                probs[label] = float(det.card_conf_top1 or det.conf or 0.0)
        return _normalize(probs)

    def _dedupe_frame(self, detections: List[Detection]) -> List[Detection]:
        out: List[Detection] = []
        ordered = sorted(detections, key=lambda d: float(d.det_conf or d.conf), reverse=True)
        for det in ordered:
            bbox = (float(det.bbox.x1), float(det.bbox.y1), float(det.bbox.x2), float(det.bbox.y2))
            keep = True
            for existing in out:
                ebox = (float(existing.bbox.x1), float(existing.bbox.y1), float(existing.bbox.x2), float(existing.bbox.y2))
                if _iou(bbox, ebox) >= self.frame_dedupe_iou:
                    keep = False
                    break
            if keep:
                out.append(det)
        return out

    def update(self, event: DetectionEvent, source_worker: str = "", source_model: str = "") -> dict:
        self._metrics["events_total"] += 1
        key = (event.game_id, event.camera_type)
        tracks = [t for t in self._tracks[key] if (event.ts_ms - t.last_ts) <= self.track_stale_ms]

        for track in tracks:
            track.miss_count += 1
            if track.miss_count >= self.freeze_miss_threshold:
                track.state = "freeze"
            if track.miss_count >= self.expire_miss_threshold:
                track.state = "expire"
        tracks = [t for t in tracks if t.state != "expire"]

        detections = self._dedupe_frame(event.detections)
        for det in detections:
            bbox = (float(det.bbox.x1), float(det.bbox.y1), float(det.bbox.x2), float(det.bbox.y2))
            probs = self._detection_probs(det)
            det_conf = float(det.det_conf or det.conf or 0.0)

            best_idx = -1
            best_iou = 0.0
            for i, track in enumerate(tracks):
                overlap = _iou(bbox, track.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = i

            if best_idx >= 0 and best_iou >= self.track_match_iou:
                track = tracks[best_idx]
                old_label, _ = track.top_label()
                track.bbox = bbox
                track.last_ts = event.ts_ms
                track.hits += 1
                track.miss_count = 0
                track.state = "active"
                # Bayes-like accumulation using weighted averaging
                merged = defaultdict(float)
                worker_weight = 1.0
                model_weight = 1.0
                if source_worker:
                    reliability = float(self.source_reliability_worker[source_worker])
                    worker_weight = reliability + self.worker_reliability_weight * float(track.source_workers.get(source_worker, 0))
                if source_model:
                    reliability = float(self.source_reliability_model[source_model])
                    model_weight = reliability + self.model_reliability_weight * float(track.source_models.get(source_model, 0))
                obs_weight = 0.30 * min(worker_weight, 1.25) * min(model_weight, 1.25)
                for label, value in track.posterior.items():
                    merged[label] += 0.70 * value
                for label, value in probs.items():
                    merged[label] += obs_weight * value
                track.posterior = _normalize(dict(merged))
                new_label, _ = track.top_label()
                if old_label and new_label and old_label != new_label:
                    self._metrics["label_flip_events"] += 1
                track.det_conf_avg = ((track.det_conf_avg * (track.hits - 1)) + det_conf) / max(track.hits, 1)
                if source_worker:
                    track.source_workers[source_worker] = track.source_workers.get(source_worker, 0) + 1
                if source_model:
                    track.source_models[source_model] = track.source_models.get(source_model, 0) + 1
            else:
                self._next_id[key] += 1
                t = Track(
                    track_id=self._next_id[key],
                    bbox=bbox,
                    last_ts=event.ts_ms,
                    hits=1,
                    posterior=probs,
                    det_conf_avg=det_conf,
                )
                if source_worker:
                    t.source_workers[source_worker] = 1
                if source_model:
                    t.source_models[source_model] = 1
                tracks.append(t)

        self._tracks[key] = tracks
        return self.snapshot(event.game_id, event.ts_ms)

    def _as_hand(self, game_id: str) -> tuple[List[dict], float]:
        key = (game_id, "hand")
        tracks = sorted(self._tracks.get(key, []), key=lambda t: (t.stability(), t.det_conf_avg), reverse=True)
        hand = []
        used = set()
        conf_values = []
        for track in tracks:
            label, prob = track.top_label()
            if not label or label in used:
                continue
            if track.hits < self.hysteresis_hits_threshold and prob < self.hysteresis_prob_threshold:
                continue
            entropy = self._entropy(track.posterior)
            if entropy > self.posterior_entropy_threshold:
                continue
            used.add(label)
            conf = max(prob, track.det_conf_avg)
            temporal_stability = track.stability()
            vision_conf = conf
            legal_transition_score = 1.0
            uniqueness_score = 1.0
            joint_score = max(0.0, min(1.0, vision_conf * temporal_stability * legal_transition_score * uniqueness_score))
            conf_values.append(conf)
            hand.append({
                "track_id": track.track_id,
                "label": label,
                "confidence": round(conf, 4),
                "stability": round(temporal_stability, 4),
                "posterior_entropy": round(entropy, 4),
                "state": track.state,
                "score_components": {
                    "vision_conf": round(vision_conf, 4),
                    "temporal_stability": round(temporal_stability, 4),
                    "legal_transition_score": round(legal_transition_score, 4),
                    "uniqueness_score": round(uniqueness_score, 4),
                },
                "joint_score": round(joint_score, 4),
                "source_workers": track.source_workers,
                "source_models": track.source_models,
                "posterior": sorted(track.posterior.items(), key=lambda kv: kv[1], reverse=True)[:3],
            })
            if len(hand) >= 13:
                break
        mean_conf = (sum(conf_values) / len(conf_values)) if conf_values else 0.0
        return hand, mean_conf

    def _as_table(self, game_id: str) -> tuple[List[dict], float]:
        key = (game_id, "table")
        tracks = sorted(self._tracks.get(key, []), key=lambda t: (t.stability(), t.det_conf_avg), reverse=True)[:4]
        if not tracks:
            return [], 0.0

        ordered = sorted(tracks, key=lambda t: (t.bbox[0] + t.bbox[2]) * 0.5)
        seat_order = ["left", "across", "right", "me"]
        table = []
        conf_values = []
        for i, track in enumerate(ordered):
            label, prob = track.top_label()
            if not label:
                continue
            if track.hits < self.hysteresis_hits_threshold and prob < self.hysteresis_prob_threshold:
                continue
            entropy = self._entropy(track.posterior)
            if entropy > self.posterior_entropy_threshold:
                continue
            conf = max(prob, track.det_conf_avg)
            conf_values.append(conf)
            fallback_seat = seat_order[min(i, len(seat_order) - 1)]
            seat = self._seat_from_calibration(game_id, track.bbox, fallback_seat)
            temporal_stability = track.stability()
            vision_conf = conf
            legal_transition_score = 1.0
            uniqueness_score = 1.0
            joint_score = max(0.0, min(1.0, vision_conf * temporal_stability * legal_transition_score * uniqueness_score))
            table.append({
                "seat": seat,
                "card": label,
                "confidence": round(conf, 4),
                "track_id": track.track_id,
                "posterior_entropy": round(entropy, 4),
                "state": track.state,
                "score_components": {
                    "vision_conf": round(vision_conf, 4),
                    "temporal_stability": round(temporal_stability, 4),
                    "legal_transition_score": round(legal_transition_score, 4),
                    "uniqueness_score": round(uniqueness_score, 4),
                },
                "joint_score": round(joint_score, 4),
            })
        mean_conf = (sum(conf_values) / len(conf_values)) if conf_values else 0.0
        return table, mean_conf

    def snapshot(self, game_id: str, ts_ms: int) -> dict:
        hand_cards, hand_conf = self._as_hand(game_id)
        table_cards, table_conf = self._as_table(game_id)

        # cross-camera uniqueness: keep location with higher confidence
        conflicts = []
        hand_map = {c["label"]: c for c in hand_cards}
        table_map = {c["card"]: c for c in table_cards}
        for card in list(set(hand_map.keys()) & set(table_map.keys())):
            h = hand_map[card]
            t = table_map[card]
            if h["confidence"] >= t["confidence"]:
                table_cards = [x for x in table_cards if x["card"] != card]
                conflicts.append({"card": card, "kept": "hand", "dropped": "table"})
            else:
                hand_cards = [x for x in hand_cards if x["label"] != card]
                conflicts.append({"card": card, "kept": "table", "dropped": "hand"})

        # deck-count guardrail: each label max 1 in final snapshot
        dedup_hand: list[dict] = []
        seen_labels: set[str] = set()
        for item in sorted(hand_cards, key=lambda x: x.get("joint_score", 0.0), reverse=True):
            label = item.get("label", "")
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            dedup_hand.append(item)
        hand_cards = dedup_hand

        dedup_table: list[dict] = []
        seen_table: set[str] = set(seen_labels)
        for item in sorted(table_cards, key=lambda x: x.get("joint_score", 0.0), reverse=True):
            card = item.get("card", "")
            if not card or card in seen_table:
                continue
            seen_table.add(card)
            dedup_table.append(item)
        table_cards = dedup_table

        self._metrics["uniqueness_conflicts"] += len(conflicts)
        if conflicts:
            bucket = self._recent_conflicts[game_id]
            bucket.extend(conflicts)
            if len(bucket) > 200:
                del bucket[: len(bucket) - 200]

        table_key = (game_id, "table")
        active_table_ids = {int(item.get("track_id", 0)) for item in table_cards if item.get("track_id")}
        occluded = self._occluded_tracks[table_key]
        for track in self._tracks.get(table_key, []):
            if track.track_id not in active_table_ids:
                occluded[track.track_id] = ts_ms
            elif track.track_id in occluded:
                occluded.pop(track.track_id, None)
        occluded_ids = [tid for tid, seen_ts in occluded.items() if (ts_ms - seen_ts) <= self.occlusion_timeout_ms]

        hand_stability = [c.get("stability", 0.0) for c in hand_cards]
        table_stability = []
        for item in table_cards:
            track_id = int(item.get("track_id", 0))
            t_match = next((t for t in self._tracks.get((game_id, "table"), []) if t.track_id == track_id), None)
            if t_match is not None:
                table_stability.append(t_match.stability())

        avg_stability = 0.0
        all_stability = hand_stability + table_stability
        if all_stability:
            avg_stability = sum(all_stability) / len(all_stability)

        return {
            "game_id": game_id,
            "ts_ms": ts_ms,
            "hand_cards": hand_cards,
            "table_trick_cards": table_cards,
            "uncertainty": {
                "hand_conf": round(hand_conf, 4),
                "table_conf": round(table_conf, 4),
                "state_conf": round((hand_conf + table_conf) / 2.0, 4),
                "notes": [
                    "Cross-camera uniqueness conflict resolved" if conflicts else "",
                ],
            },
            "fusion_debug": {
                "conflicts": conflicts,
                "track_count_hand": len(self._tracks.get((game_id, "hand"), [])),
                "track_count_table": len(self._tracks.get((game_id, "table"), [])),
                "avg_track_stability": round(avg_stability, 4),
                "posterior_entropy_threshold": round(self.posterior_entropy_threshold, 4),
                "occluded_tracks": occluded_ids,
            },
        }

    def metrics(self) -> dict:
        events_total = max(int(self._metrics.get("events_total", 0)), 1)
        flip_events = int(self._metrics.get("label_flip_events", 0))
        stabilities: list[float] = []
        for track_list in self._tracks.values():
            for track in track_list:
                stabilities.append(float(track.stability()))
        avg_stability = (sum(stabilities) / len(stabilities)) if stabilities else 0.0
        return {
            **self._metrics,
            "label_flip_rate": round(flip_events / events_total, 6),
            "recent_conflicts_total": sum(len(v) for v in self._recent_conflicts.values()),
            "track_stability_avg": round(avg_stability, 6),
            "track_count_total": sum(len(v) for v in self._tracks.values()),
            "posterior_entropy_threshold": round(self.posterior_entropy_threshold, 6),
            "worker_reliability_weight": round(self.worker_reliability_weight, 6),
            "model_reliability_weight": round(self.model_reliability_weight, 6),
        }

    def recent_conflicts(self, game_id: str, limit: int = 100) -> list[dict]:
        events = self._recent_conflicts.get(game_id, [])
        return list(events[-max(1, min(limit, 500)):])
