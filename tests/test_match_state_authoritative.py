from backend.app.match_engine import MatchEngine
from backend.app.models import CardLifecycleEntry, VariantConfig


def _variant(cards_per_player: int = 5, trick_size: int = 4) -> VariantConfig:
    return VariantConfig(
        variant_id="t",
        player_count=4,
        seat_order=["me", "left", "across", "right"],
        play_direction="clockwise",
        cards_per_player=cards_per_player,
        trick_size=trick_size,
        team_map={"me": "team_me", "across": "team_me", "left": "team_opp", "right": "team_opp"},
    )


def test_dynamic_hand_size_5_cards_variant():
    engine = MatchEngine()
    state = engine.start_match("g1", _variant(cards_per_player=5), "me")
    for idx in range(1, 7):
        state.card_lifecycle[f"{idx}S"] = CardLifecycleEntry(location="in_hand", owner_seat="me", seen_ts_ms=100, confidence=0.9)
    state.hand_cards_by_seat = {"me": ["1S", "2S", "3S", "4S", "5S"]}

    projected = engine.project_hand_state_for_camera("g1", "cam-1", "me")
    assert projected.max_cards == 5
    assert len(projected.cards) == 5


def test_play_card_reduces_hand_count_by_one():
    engine = MatchEngine()
    state = engine.start_match("g2", _variant(cards_per_player=5), "me")
    for card in ["AS", "KS", "QS", "JS", "10S"]:
        state.card_lifecycle[card] = CardLifecycleEntry(location="in_hand", owner_seat="me", seen_ts_ms=100, confidence=0.9)

    _, events, _ = engine.reconcile_card_locations(
        game_id="g2",
        ts_ms=200,
        hand_cards_observed=[
            {"card": "KS", "seat": "me", "confidence": 0.9, "ts_ms": 200, "track_id": 1},
            {"card": "QS", "seat": "me", "confidence": 0.9, "ts_ms": 200, "track_id": 2},
            {"card": "JS", "seat": "me", "confidence": 0.9, "ts_ms": 200, "track_id": 3},
            {"card": "10S", "seat": "me", "confidence": 0.9, "ts_ms": 200, "track_id": 4},
        ],
        table_cards_observed=[{"card": "AS", "seat": "me", "confidence": 0.95, "ts_ms": 200, "track_id": 9}],
        prev_lifecycle=state.card_lifecycle,
        to_play_seat="me",
        trick_context={},
        variant=state.variant,
    )
    updated, violations = engine.apply_transition_events(state, events, 200)

    assert not violations
    assert len(updated.hand_cards_by_seat["me"]) == 4
    assert len(updated.table_cards) == 1
    assert any(e["type"] == "PLAY_CARD" and e["card"] == "AS" for e in events)


def test_occluded_card_not_removed_before_ttl():
    engine = MatchEngine()
    state = engine.start_match("g3", _variant(cards_per_player=5), "me")
    state.card_lifecycle["AH"] = CardLifecycleEntry(location="in_hand", owner_seat="me", seen_ts_ms=1000, confidence=0.9)

    _, events, _ = engine.reconcile_card_locations(
        game_id="g3",
        ts_ms=1000 + 200,
        hand_cards_observed=[],
        table_cards_observed=[],
        prev_lifecycle=state.card_lifecycle,
        to_play_seat="me",
        trick_context={},
        variant=state.variant,
    )
    updated, violations = engine.apply_transition_events(state, events, 1200)

    assert not violations
    assert "AH" in updated.hand_cards_by_seat["me"]


def test_no_additive_hand_inflation():
    engine = MatchEngine()
    state = engine.start_match("g4", _variant(cards_per_player=5), "me")
    for card in ["AS", "KS", "QS", "JS", "10S"]:
        state.card_lifecycle[card] = CardLifecycleEntry(location="in_hand", owner_seat="me", seen_ts_ms=100, confidence=0.9)

    _, events, _ = engine.reconcile_card_locations(
        game_id="g4",
        ts_ms=200,
        hand_cards_observed=[
            {"card": "AS", "seat": "me", "confidence": 0.9, "ts_ms": 200, "track_id": 1},
            {"card": "KS", "seat": "me", "confidence": 0.9, "ts_ms": 200, "track_id": 2},
            {"card": "QS", "seat": "me", "confidence": 0.9, "ts_ms": 200, "track_id": 3},
            {"card": "JS", "seat": "me", "confidence": 0.9, "ts_ms": 200, "track_id": 4},
        ],
        table_cards_observed=[],
        prev_lifecycle=state.card_lifecycle,
        to_play_seat="me",
        trick_context={},
        variant=state.variant,
    )
    updated, violations = engine.apply_transition_events(state, events, 200)

    assert not violations
    assert len(updated.hand_cards_by_seat["me"]) in {4, 5}
    assert len(updated.hand_cards_by_seat["me"]) != 9


def test_duplicate_card_across_hand_and_table_resolved_once():
    engine = MatchEngine()
    state = engine.start_match("g5", _variant(cards_per_player=5), "me")

    _, events, _ = engine.reconcile_card_locations(
        game_id="g5",
        ts_ms=200,
        hand_cards_observed=[{"card": "9C", "seat": "me", "confidence": 0.9, "ts_ms": 200, "track_id": 1}],
        table_cards_observed=[{"card": "9C", "seat": "left", "confidence": 0.95, "ts_ms": 200, "track_id": 2}],
        prev_lifecycle=state.card_lifecycle,
        to_play_seat="me",
        trick_context={},
        variant=state.variant,
    )
    updated, violations = engine.apply_transition_events(state, events, 200)

    assert not violations
    assert any(t.card == "9C" for t in updated.table_cards)
    assert "9C" not in updated.hand_cards_by_seat["me"]


def test_trick_close_uses_variant_trick_size():
    engine = MatchEngine()
    state = engine.start_match("g6", _variant(cards_per_player=5, trick_size=3), "left")
    state.leader_seat = "left"
    state.to_play_seat = "left"

    _, events, _ = engine.reconcile_card_locations(
        game_id="g6",
        ts_ms=300,
        hand_cards_observed=[],
        table_cards_observed=[
            {"card": "2H", "seat": "left", "confidence": 0.9, "ts_ms": 300, "track_id": 1},
            {"card": "KH", "seat": "across", "confidence": 0.9, "ts_ms": 300, "track_id": 2},
            {"card": "AH", "seat": "right", "confidence": 0.9, "ts_ms": 300, "track_id": 3},
        ],
        prev_lifecycle=state.card_lifecycle,
        to_play_seat="left",
        trick_context={},
        variant=state.variant,
    )
    updated, violations = engine.apply_transition_events(state, events, 300)

    assert not violations
    assert updated.trick_id == 2
    assert len(updated.table_cards) == 0


def test_won_pile_reappearance_rejected():
    engine = MatchEngine()
    state = engine.start_match("g7", _variant(cards_per_player=5), "me")
    state.card_lifecycle["AD"] = CardLifecycleEntry(
        location="won_pile",
        owner_seat=None,
        owner_team="team_me",
        seen_ts_ms=100,
        confidence=1.0,
        last_transition="table_to_won",
    )

    _, events, _ = engine.reconcile_card_locations(
        game_id="g7",
        ts_ms=200,
        hand_cards_observed=[{"card": "AD", "seat": "me", "confidence": 0.9, "ts_ms": 200, "track_id": 1}],
        table_cards_observed=[],
        prev_lifecycle=state.card_lifecycle,
        to_play_seat="me",
        trick_context={"replay_mode": False},
        variant=state.variant,
    )
    updated, violations = engine.apply_transition_events(state, events, 200)

    assert not violations
    assert updated.card_lifecycle["AD"].location == "won_pile"
    assert any(e["type"] == "CARD_REJECTED" and e["reason"] == "won_pile_reappearance" for e in events)
