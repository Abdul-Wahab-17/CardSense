from backend.app.belief import build_belief
from backend.app.fusion import FusionEngine
from backend.app.game_state import GameStateEngine
from backend.app.models import (
    BBox,
    CanonicalGameState,
    Detection,
    DetectionEvent,
    PlayedHistoryItem,
    TableTrickCard,
)
from backend.app.rules import legal_actions
from backend.app.simulator import evaluate_actions


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


def test_rules_legal_actions_variant_profiles():
    state = CanonicalGameState(
        game_id="g1",
        hand_cards=["AH", "2S", "9D"],
    )
    state.turn.to_play = "me"
    state.turn.lead_suit = "C"
    state.rules.must_follow_suit = True
    state.rules.must_trump_if_void = True
    state.rules.trump_suit = "S"
    actions = legal_actions(state)
    assert actions == ["2S"]


def test_belief_void_suit_update():
    state = CanonicalGameState(game_id="g2")
    state.played_history = [
        PlayedHistoryItem(trick_no=1, seat="me", card="10H"),
        PlayedHistoryItem(trick_no=1, seat="left", card="2C"),  # off-suit against hearts lead
    ]
    belief = build_belief(state)
    left_hearts = [p for card, p in belief["left"].items() if card.endswith("H")]
    across_hearts = [p for card, p in belief["across"].items() if card.endswith("H")]
    assert sum(left_hearts) < sum(across_hearts)


def test_fusion_uniqueness_conflict_resolution():
    fusion = FusionEngine()
    ev_hand = DetectionEvent(
        game_id="g3",
        frame_id="f1",
        camera_type="hand",
        camera_id="cam-h",
        ts_ms=1,
        model_name="m",
        detections=[_det("AS", 0.9, 10, 10, 60, 80)],
    )
    fusion.update(ev_hand)
    ev_table = DetectionEvent(
        game_id="g3",
        frame_id="f2",
        camera_type="table",
        camera_id="cam-t",
        ts_ms=2,
        model_name="m",
        detections=[_det("AS", 0.5, 110, 110, 180, 190)],
    )
    snap = fusion.update(ev_table)
    table_cards = [c["card"] for c in snap["table_trick_cards"]]
    assert "AS" not in table_cards


def test_transition_reject_table_to_hand():
    engine = GameStateEngine()
    state = engine.start_game("g4", "me", "me", True, "variant_config")
    state.hand_cards = ["AH"]
    state.table_trick_cards = [TableTrickCard(seat="left", card="10S", confidence=1.0)]
    fused = {
        "game_id": "g4",
        "ts_ms": 100,
        "hand_cards": [{"label": "10S", "confidence": 0.8}],
        "table_trick_cards": [],
        "uncertainty": {"hand_conf": 0.8, "table_conf": 0.8, "state_conf": 0.8, "notes": []},
    }
    _, transition = engine.apply_fused_observation(fused)
    assert not transition.accepted
    assert transition.reason_code == "table_to_hand_rejected"


def test_simulator_deterministic_seed():
    state = CanonicalGameState(game_id="g5", hand_cards=["AS", "KH", "2D"])
    state.turn.to_play = "me"
    state.turn.lead_suit = "H"
    out1 = evaluate_actions(state, simulation_seed=777)
    out2 = evaluate_actions(state, simulation_seed=777)
    assert [x.model_dump() for x in out1] == [x.model_dump() for x in out2]
