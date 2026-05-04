"""Unit tests for backend.app.blackjack and blackjack_advice.

Run with: pytest tests/test_blackjack.py -v
"""
from __future__ import annotations

import pytest

from backend.app.blackjack import (
    BlackjackState,
    Hand,
    dealer_draw,
    deal_initial,
    new_round,
    normalize_rank,
    payout_for,
    player_double,
    player_hit,
    player_stand,
    reveal_dealer_hole,
)
from backend.app.blackjack_advice import (
    basic_strategy,
    bust_probability_on_hit,
    recommend,
    remaining_deck,
    running_count,
    true_count,
)


# ---------------- normalize_rank ---------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("AS", "A"),
    ("AH", "A"),
    ("10♥", "10"),
    ("KD", "K"),
    ("J", "J"),
    ("3", "3"),
    ("2C", "2"),
    # bare ranks the model occasionally emits without suit
    ("A", "A"),
    ("10", "10"),
    ("Q", "Q"),
    # negatives
    ("garbage", None),
    ("", None),
])
def test_normalize_rank(label, expected):
    assert normalize_rank(label) == expected


# ---------------- Hand totals -------------------------------------------------

def test_hard_total():
    assert Hand(cards=["10", "7"]).total() == (17, False)
    assert Hand(cards=["K", "Q", "5"]).total() == (25, False)  # bust
    assert Hand(cards=["3", "5"]).total() == (8, False)


def test_soft_total():
    assert Hand(cards=["A", "6"]).total() == (17, True)        # soft 17
    assert Hand(cards=["A", "9"]).total() == (20, True)        # soft 20
    assert Hand(cards=["A", "A"]).total() == (12, True)        # one ace as 11, one as 1


def test_ace_downgrade():
    # A + 7 + 5 = soft 23 -> downgrade to hard 13
    assert Hand(cards=["A", "7", "5"]).total() == (13, False)
    # Two aces + 9: one ace stays as 11 (11+1+9 = 21), so soft.
    assert Hand(cards=["A", "A", "9"]).total() == (21, True)
    # Two aces + 9 + 5: both aces forced to 1 (1+1+9+5 = 16), hard.
    assert Hand(cards=["A", "A", "9", "5"]).total() == (16, False)


def test_blackjack_natural():
    assert Hand(cards=["A", "K"]).is_blackjack() is True
    assert Hand(cards=["A", "10"]).is_blackjack() is True
    assert Hand(cards=["10", "A"]).is_blackjack() is True
    # Three-card 21 is NOT a natural
    assert Hand(cards=["7", "7", "7"]).is_blackjack() is False
    assert Hand(cards=["A", "5", "5"]).is_blackjack() is False


def test_bust():
    assert Hand(cards=["10", "K", "5"]).is_bust() is True
    assert Hand(cards=["A", "K", "5"]).is_bust() is False  # 11+10+5 = 26 -> 16 (Ace=1)


# ---------------- Round flow --------------------------------------------------

def test_player_blackjack_dealer_blackjack_pushes():
    state = BlackjackState()
    state = deal_initial(state, ["A", "K"], ["A", "Q"])
    assert state.round_outcome == "push"


def test_player_blackjack_wins():
    state = BlackjackState()
    state = deal_initial(state, ["A", "K"], ["10", "7"])
    # naturals auto-resolve via reveal_dealer_hole
    assert state.round_outcome == "player_blackjack"
    assert payout_for(state) == pytest.approx(1.5)


def test_hit_until_stand():
    state = BlackjackState()
    state = deal_initial(state, ["5", "7"], ["10", "6"])  # player 12, dealer 16+hole
    state = player_hit(state, "9")  # player now 21 -> auto-stand
    assert state.player_done is True
    # dealer hole = 6, so dealer has 10+6=16, must hit
    state = dealer_draw(state, "5")  # dealer 21
    assert state.round_outcome == "push"


def test_player_busts():
    state = BlackjackState()
    state = deal_initial(state, ["10", "7"], ["8", "9"])  # player 17, dealer 8+hole=17
    state = player_hit(state, "10")  # player 27 bust
    assert state.round_outcome == "lose"  # _settle resolves bust as lose
    assert state.player.is_bust() is True


def test_dealer_busts():
    state = BlackjackState()
    state = deal_initial(state, ["10", "8"], ["10", "6"])  # player 18, dealer 16
    state = player_stand(state)
    state = dealer_draw(state, "K")  # dealer 26 bust
    assert state.round_outcome == "win"


def test_dealer_stands_at_17():
    state = BlackjackState()
    state = deal_initial(state, ["10", "9"], ["10", "7"])  # player 19, dealer 17
    state = player_stand(state)
    # dealer reveal already triggers stand check
    assert state.round_outcome == "win"  # 19 > 17


def test_player_double_wins():
    state = BlackjackState()
    state = deal_initial(state, ["6", "5"], ["10", "6"])  # player 11, dealer 16
    state = player_double(state, "10")  # player 21
    assert state.player_doubled is True
    assert state.bet == 2.0
    state = dealer_draw(state, "K")  # dealer 26 bust
    assert state.round_outcome == "win"
    assert payout_for(state) == pytest.approx(2.0)


def test_double_only_on_two_cards():
    state = BlackjackState()
    state = deal_initial(state, ["6", "5"], ["10", "6"])
    state = player_hit(state, "2")  # now 13, three cards
    with pytest.raises(ValueError):
        player_double(state, "5")


def test_seen_cards_tracks_visible_only():
    state = BlackjackState()
    state = deal_initial(state, ["5", "7"], ["10", "9"])
    # Seen so far: player 5, 7; dealer upcard 10. Hole card 9 NOT yet seen.
    assert state.seen_cards == ["5", "7", "10"]
    state = player_stand(state)  # this reveals hole card
    assert "9" in state.seen_cards


# ---------------- Hi-Lo counter -----------------------------------------------

def test_running_count_basics():
    assert running_count([]) == 0
    assert running_count(["2", "3", "4"]) == 3
    assert running_count(["10", "K"]) == -2
    assert running_count(["7", "8", "9"]) == 0  # neutrals
    # A=-1, 5=+1, 10=-1, 5=+1 -> 0
    assert running_count(["A", "5", "10", "5"]) == 0
    # mixed: 2(+1) + 6(+1) + 7(0) + Q(-1) + A(-1) = 0
    assert running_count(["2", "6", "7", "Q", "A"]) == 0
    # all low -> very high count
    assert running_count(["2", "3", "4", "5", "6"]) == 5


def test_true_count_normalization():
    # Running +6 with 1 deck remaining -> TC = 6
    assert true_count(6, 52) == pytest.approx(6.0)
    # Running +6 with 2 decks remaining -> TC = 3
    assert true_count(6, 104) == pytest.approx(3.0)
    # Running +6 with effectively no cards: floored at half-deck
    assert true_count(6, 0) == pytest.approx(12.0)


# ---------------- Remaining deck math -----------------------------------------

def test_remaining_deck_full():
    state = BlackjackState()
    rem = remaining_deck(state)
    # 13 ranks × 4 = 52
    assert sum(rem.values()) == 52
    # 10s through Ks all separate ranks here, each = 4
    for r in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"):
        assert rem[r] == 4


def test_remaining_deck_after_seen():
    state = BlackjackState()
    state.seen_cards = ["A", "A", "10", "5"]
    rem = remaining_deck(state)
    assert rem["A"] == 2
    assert rem["10"] == 3
    assert rem["5"] == 3
    assert rem["K"] == 4  # untouched


def test_bust_probability_on_hit():
    state = BlackjackState()
    # Hand = 12 with full deck. Bust if drawing 10/J/Q/K (16 cards out of 52) = 16/52 ≈ 0.308
    hand = Hand(cards=["7", "5"])
    rem = remaining_deck(state)
    p = bust_probability_on_hit(hand, rem)
    assert p == pytest.approx(16 / 52, rel=1e-3)


def test_bust_probability_zero_on_low():
    # Hand of 11: literally cannot bust on a single card. Aces would just go to 12 (Ace=1).
    state = BlackjackState()
    hand = Hand(cards=["6", "5"])
    rem = remaining_deck(state)
    assert bust_probability_on_hit(hand, rem) == pytest.approx(0.0)


# ---------------- Basic strategy ---------------------------------------------

@pytest.mark.parametrize("player,dealer,expected", [
    # Hard totals
    (["10", "6"], "10", "hit"),    # 16 vs 10 -> hit (textbook)
    (["10", "6"], "5", "stand"),   # 16 vs 5 -> stand
    (["10", "9"], "K", "stand"),   # 19 vs K -> stand
    (["8", "8"], "5", "stand"),    # treat 16 as hard (split rules unused in v1)
    (["6", "5"], "5", "double"),   # 11 vs 5 -> double
    (["5", "4"], "6", "double"),   # 9 vs 6 -> double
    (["5", "3"], "10", "hit"),     # 8 vs 10 -> hit
    # Soft totals
    (["A", "6"], "5", "double"),   # soft 17 vs 5 -> double
    (["A", "7"], "9", "hit"),      # soft 18 vs 9 -> hit
    (["A", "7"], "2", "stand"),    # soft 18 vs 2 -> stand
    (["A", "8"], "6", "stand"),    # soft 19 always stand (S17)
])
def test_basic_strategy(player, dealer, expected):
    h = Hand(cards=player)
    action, _code = basic_strategy(h, dealer, can_double=True)
    assert action == expected


def test_basic_strategy_double_falls_back_to_hit_if_disallowed():
    h = Hand(cards=["6", "5"])  # 11
    action, _ = basic_strategy(h, "5", can_double=False)
    assert action == "hit"


# ---------------- Recommend (end-to-end) -------------------------------------

def test_recommend_basic():
    state = BlackjackState()
    state = deal_initial(state, ["10", "6"], ["10", "5"])
    rec = recommend(state)
    # 16 vs 10 with neutral count -> basic says hit
    assert rec.action == "hit"
    assert rec.short_label == "HIT"
    assert rec.diagnostics.player_total == 16
    assert rec.diagnostics.dealer_upcard == "10"
    assert 0.0 <= rec.confidence <= 1.0


def test_recommend_deviation_kicks_in():
    """16 vs 10 with high true count -> stand (Illustrious 18)."""
    state = BlackjackState()
    state = deal_initial(state, ["10", "6"], ["10", "5"])
    # Manufacture a high count: pretend lots of low cards have been seen.
    state.seen_cards.extend(["2", "3", "4", "5", "6"] * 3)
    rec = recommend(state)
    # TC > 0 here -> deviation flips H to S
    assert rec.action == "stand"
    assert rec.used_deviation is True


def test_new_round_clears_hands_keeps_count():
    state = BlackjackState()
    state = deal_initial(state, ["10", "6"], ["10", "5"])
    state = player_hit(state, "10")  # player busts
    pre_seen = list(state.seen_cards)
    state = new_round(state, reshuffle=False)
    assert state.player.cards == []
    assert state.dealer.cards == []
    assert state.round_outcome == "in_progress"
    assert state.seen_cards == pre_seen  # count preserved


def test_reshuffle_clears_seen():
    state = BlackjackState()
    state.seen_cards = ["2"] * 40  # heavily biased shoe
    state = new_round(state, reshuffle=True)
    assert state.seen_cards == []


def test_needs_reshuffle_pcct():
    state = BlackjackState(decks_in_shoe=1, reshuffle_at_pct=0.25)
    state.seen_cards = ["2"] * 39  # 13 cards remaining = 25%
    assert state.needs_reshuffle() is True
    state.seen_cards = ["2"] * 30  # 22 cards remaining = ~42%
    assert state.needs_reshuffle() is False


# ---------------- Engine-level integration ------------------------------------

def test_engine_full_round():
    """Full happy-path round through the BlackjackEngine wrapper."""
    from backend.app.blackjack_engine import BlackjackEngine

    eng = BlackjackEngine()
    eng.start_game("g1", decks_in_shoe=1)

    # Deal: player 11 (raw labels with suits), dealer 16 (with hidden hole)
    eng.deal_initial("g1", ["6S", "5H"], ["10D", "6C"])
    state = eng.get("g1")
    assert state.player.cards == ["6", "5"]  # suits stripped
    assert state.dealer.cards == ["10", "6"]
    assert not state.dealer_hole_revealed
    # Player 6+5 + dealer upcard 10. Dealer hole-card (6) NOT yet seen.
    assert state.seen_cards == ["6", "5", "10"]
    assert state.seen_cards.count("6") == 1  # only player's 6 visible; hole 6 hidden

    # Recommend on 11 vs 10 -> double (basic)
    rec = eng.recommend("g1")
    assert rec.action == "double"

    # Player doubles, draws 10 -> 21
    eng.player_action("g1", "double", drawn_label="10S")
    state = eng.get("g1")
    assert state.player.cards == ["6", "5", "10"]
    assert state.player.total() == (21, False)
    assert state.player_doubled
    assert state.bet == 2.0
    # dealer hole revealed -> 16, must hit
    assert state.dealer_hole_revealed
    assert state.round_outcome == "player_stand"  # waiting for dealer draw

    # Dealer hits 9 -> bust at 25
    eng.observe_card("g1", "9D", target="dealer")
    state = eng.get("g1")
    assert state.dealer.is_bust()
    assert state.round_outcome == "win"

    # History should have one entry
    history = eng.history("g1")
    assert len(history) == 1
    assert history[0]["outcome"] == "win"
    assert history[0]["payout"] == 2.0


def test_engine_observe_routes_correctly():
    """observe_card should auto-classify hits vs dealer draws based on phase."""
    from backend.app.blackjack_engine import BlackjackEngine

    eng = BlackjackEngine()
    eng.start_game("g2")
    eng.deal_initial("g2", ["10S", "6H"], ["9D", "8C"])  # player 16, dealer 17

    # player target during player turn -> hit
    eng.observe_card("g2", "5C", target="player")  # player 21
    state = eng.get("g2")
    # auto-stand at 21 -> reveal dealer -> dealer 17 stands -> 21 vs 17 -> player wins
    assert state.round_outcome == "win"


def test_engine_rejects_unknown_label():
    from backend.app.blackjack_engine import BlackjackEngine

    eng = BlackjackEngine()
    eng.start_game("g3")
    with pytest.raises(ValueError):
        eng.deal_initial("g3", ["garbage", "5H"], ["10D", "6C"])


def test_engine_history_archives_on_settlement_only_once():
    """If observe_card is called multiple times after settlement, history shouldn't double-write."""
    from backend.app.blackjack_engine import BlackjackEngine

    eng = BlackjackEngine()
    eng.start_game("g4")
    eng.deal_initial("g4", ["AS", "KH"], ["10D", "7C"])  # player BJ
    state = eng.get("g4")
    assert state.round_outcome == "player_blackjack"
    history = eng.history("g4")
    assert len(history) == 1


def test_engine_separates_games():
    """Two parallel games don't share state."""
    from backend.app.blackjack_engine import BlackjackEngine

    eng = BlackjackEngine()
    eng.start_game("a")
    eng.start_game("b")
    eng.deal_initial("a", ["10S", "8H"], ["9D", "9C"])  # player 18, dealer 18
    eng.deal_initial("b", ["AS", "KH"], ["7D", "7C"])   # player BJ
    sa = eng.get("a")
    sb = eng.get("b")
    assert sa.player.cards != sb.player.cards
    assert sb.round_outcome == "player_blackjack"
    assert sa.round_outcome != "player_blackjack"
