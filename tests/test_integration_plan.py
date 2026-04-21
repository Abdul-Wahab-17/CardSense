import asyncio

from backend.app.fusion import FusionEngine
from backend.app.game_state import GameStateEngine
from backend.app.main import _build_advisor_response, _card_tracking_sets, _mark_frame_seen, game_state_engine, pipeline_metrics
from backend.app.models import BBox, CardLifecycleEntry, Detection, DetectionEvent, MatchState, PlayedHistoryItem
from worker.worker import is_out_of_order_frame


class _FakeRedis:
    def __init__(self) -> None:
        self._seen = set()

    async def set(self, key: str, value: str, ex: int | None = None, nx: bool = False):
        if nx and key in self._seen:
            return False
        self._seen.add(key)
        return True


def _det(label: str, conf: float, x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(
        cls_id=0,
        cls_name=label,
        conf=conf,
        card_label_top1=label,
        card_conf_top1=conf,
        card_probs_topk=[[label, conf]],
        det_conf=conf,
        calibrated_card_conf_top1=conf,
        is_blurry=False,
        is_dark=False,
        is_overexposed=False,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def test_two_camera_ingest_fused_state_convergence():
    fusion = FusionEngine()
    game = GameStateEngine()

    hand_event = DetectionEvent(
        game_id="ig-1",
        frame_id="f-h-1",
        camera_type="hand",
        camera_id="cam-h",
        ts_ms=100,
        model_name="m",
        detections=[_det("AH", 0.95, 10, 10, 50, 80)],
    )
    fused = fusion.update(hand_event, source_worker="w1", source_model="m1")
    state, transition = game.apply_fused_observation(fused)
    assert transition.accepted
    assert "AH" in state.hand_cards

    table_event = DetectionEvent(
        game_id="ig-1",
        frame_id="f-t-1",
        camera_type="table",
        camera_id="cam-t",
        ts_ms=120,
        model_name="m",
        detections=[_det("10S", 0.90, 110, 100, 180, 190)],
    )
    fused = fusion.update(table_event, source_worker="w1", source_model="m1")
    state, transition = game.apply_fused_observation(fused)
    assert transition.accepted
    assert any(t.card == "10S" for t in state.table_trick_cards)


def test_duplicate_frame_id_dropped_exactly_once(monkeypatch):
    from backend.app import main as main_mod

    fake = _FakeRedis()
    monkeypatch.setattr(main_mod, "redis_client", fake)

    pipeline_metrics["metrics_deduped_frames"] = 0
    first = asyncio.run(_mark_frame_seen("ig-2", "frame-1"))
    second = asyncio.run(_mark_frame_seen("ig-2", "frame-1"))
    if not second:
        pipeline_metrics["metrics_deduped_frames"] += 1

    assert first is True
    assert second is False
    assert pipeline_metrics["metrics_deduped_frames"] == 1


def test_out_of_order_frames_skipped_via_monotonic_guard():
    latest = {"cam-a": 2000}
    assert is_out_of_order_frame("cam-a", 1999, latest)
    assert is_out_of_order_frame("cam-a", 2000, latest)
    assert not is_out_of_order_frame("cam-a", 2001, latest)


def test_trick_close_updates_history_won_piles_and_turns():
    engine = GameStateEngine()
    state = engine.start_game("ig-3", "left", "left", True, "variant_config", trump_suit="S")

    fused = {
        "game_id": "ig-3",
        "ts_ms": 200,
        "hand_cards": [{"label": "2H", "confidence": 0.8}],
        "table_trick_cards": [
            {"seat": "left", "card": "10H", "confidence": 0.9},
            {"seat": "across", "card": "KH", "confidence": 0.9},
            {"seat": "right", "card": "2S", "confidence": 0.9},
            {"seat": "me", "card": "AS", "confidence": 0.9},
        ],
        "uncertainty": {"hand_conf": 0.9, "table_conf": 0.9, "state_conf": 0.9, "notes": []},
    }
    state, transition = engine.apply_fused_observation(fused)

    assert transition.accepted
    assert len(state.played_history) == 4
    assert len(state.won_piles["team_me"]) == 4
    assert state.turn.leader == "me"
    assert state.turn.to_play == "me"


def test_recommendation_endpoint_contract_non_empty_explanations():
    g = "ig-4"
    state = game_state_engine.start_game(g, "me", "me", True, "variant_config")
    state.hand_cards = ["AH", "2H", "3D"]
    state.turn.lead_suit = "H"
    state.uncertainty.hand_conf = 0.8
    state.uncertainty.table_conf = 0.8
    state.uncertainty.state_conf = 0.8
    state.uncertainty.notes = ["integration-note"]

    rec = _build_advisor_response(g, simulation_seed=123)
    assert rec.why
    assert rec.uncertainty_notes


def test_fusion_temporal_consensus_prefers_4_of_5_votes():
    fusion = FusionEngine()
    fusion.consensus_window_frames = 5
    fusion.consensus_min_samples = 5
    fusion.consensus_min_ratio = 0.75

    seq = ["AS", "AS", "AS", "AS", "KS"]
    snap = None
    for idx, label in enumerate(seq, start=1):
        ev = DetectionEvent(
            game_id="ig-cons-1",
            frame_id=f"f{idx}",
            camera_type="hand",
            camera_id="cam-h",
            ts_ms=100 + idx,
            model_name="m",
            detections=[_det(label, 0.95, 10, 10, 60, 80)],
        )
        snap = fusion.update(ev, source_worker="w", source_model="m")

    labels = [row["label"] for row in (snap or {}).get("hand_cards", [])]
    assert "AS" in labels
    assert "KS" not in labels


def test_fusion_temporal_consensus_rejects_unstable_votes():
    fusion = FusionEngine()
    fusion.consensus_window_frames = 5
    fusion.consensus_min_samples = 5
    fusion.consensus_min_ratio = 0.8

    seq = ["AS", "KS", "AS", "KS", "QS"]
    snap = None
    for idx, label in enumerate(seq, start=1):
        ev = DetectionEvent(
            game_id="ig-cons-2",
            frame_id=f"f{idx}",
            camera_type="hand",
            camera_id="cam-h",
            ts_ms=200 + idx,
            model_name="m",
            detections=[_det(label, 0.95, 10, 10, 60, 80)],
        )
        snap = fusion.update(ev, source_worker="w", source_model="m")

    assert (snap or {}).get("hand_cards", []) == []


def test_card_tracking_sets_cover_lifecycle_and_history():
    state = MatchState(game_id="ig-sets")
    state.card_lifecycle = {
        "AS": CardLifecycleEntry(location="in_hand", owner_seat="me", seen_ts_ms=100, confidence=0.9),
        "KH": CardLifecycleEntry(location="on_table", owner_seat="left", seen_ts_ms=100, confidence=0.9),
        "2D": CardLifecycleEntry(location="won_pile", owner_team="team_me", seen_ts_ms=100, confidence=1.0),
        "9C": CardLifecycleEntry(location="unknown", seen_ts_ms=100, confidence=0.5),
    }
    state.played_history = [PlayedHistoryItem(trick_no=1, seat="left", card="KH"), PlayedHistoryItem(trick_no=1, seat="me", card="JS")]

    sets = _card_tracking_sets(state)
    assert "AS" in sets["in_hand_cards"]
    assert "KH" in sets["on_table_cards"]
    assert "2D" in sets["won_pile_cards"]
    assert "9C" in sets["unknown_cards"]
    assert "JS" in sets["played_history_cards"]
    assert "JS" in sets["seen_cards"]
