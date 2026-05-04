"""Blackjack recommender: basic strategy + Hi-Lo card counting.

Decision pipeline:
    1. Look up canonical basic strategy for (player_total, dealer_upcard, soft?)
    2. Apply Hi-Lo count deviations ("the Illustrious 18") if true count crosses
       a known threshold.
    3. Compute auxiliary diagnostics (bust probability, exact remaining-deck
       composition) for the UI.

The recommender is deterministic. The LLM layer (in main.py / a separate
module) is a *narration* layer on top — it never overrides this.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .blackjack import VALID_RANKS, BlackjackState, Hand, _RANK_VALUE


Action = Literal["hit", "stand", "double", "split"]  # split unused in v1, kept for future


# ---------------- Hi-Lo count -------------------------------------------------

_HILO = {
    "2": +1, "3": +1, "4": +1, "5": +1, "6": +1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}


def running_count(seen: List[str]) -> int:
    return sum(_HILO.get(r, 0) for r in seen)


def true_count(running: int, cards_remaining: int) -> float:
    """True count = running count / decks remaining. Half-deck floor to avoid /0 explosion."""
    decks_remaining = max(0.5, cards_remaining / 52.0)
    return running / decks_remaining


# ---------------- Basic strategy tables (S17, single deck, no surrender) ------
#
# Encoded as: player_total -> dealer_upcard -> action.
# For soft totals (Ace counted as 11), use a separate table.
# Action codes:
#   H = Hit
#   S = Stand
#   D = Double if allowed, else Hit
#   Ds = Double if allowed, else Stand (used for soft 18 vs 6 etc.)

_DEALER_UPCARDS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "A")


def _strategy_upcard(rank: str) -> str:
    """Normalize a dealer's actual rank into the basic-strategy upcard space.

    J/Q/K all collapse to '10' since they have identical value.
    """
    if rank in ("J", "Q", "K"):
        return "10"
    return rank

_HARD: Dict[int, Dict[str, str]] = {
    # 5-8 always hit
    5:  {u: "H" for u in _DEALER_UPCARDS},
    6:  {u: "H" for u in _DEALER_UPCARDS},
    7:  {u: "H" for u in _DEALER_UPCARDS},
    8:  {u: "H" for u in _DEALER_UPCARDS},
    9:  {"2": "H", "3": "D", "4": "D", "5": "D", "6": "D",
         "7": "H", "8": "H", "9": "H", "10": "H", "A": "H"},
    10: {"2": "D", "3": "D", "4": "D", "5": "D", "6": "D",
         "7": "D", "8": "D", "9": "D", "10": "H", "A": "H"},
    11: {"2": "D", "3": "D", "4": "D", "5": "D", "6": "D",
         "7": "D", "8": "D", "9": "D", "10": "D", "A": "H"},
    12: {"2": "H", "3": "H", "4": "S", "5": "S", "6": "S",
         "7": "H", "8": "H", "9": "H", "10": "H", "A": "H"},
    13: {"2": "S", "3": "S", "4": "S", "5": "S", "6": "S",
         "7": "H", "8": "H", "9": "H", "10": "H", "A": "H"},
    14: {"2": "S", "3": "S", "4": "S", "5": "S", "6": "S",
         "7": "H", "8": "H", "9": "H", "10": "H", "A": "H"},
    15: {"2": "S", "3": "S", "4": "S", "5": "S", "6": "S",
         "7": "H", "8": "H", "9": "H", "10": "H", "A": "H"},
    16: {"2": "S", "3": "S", "4": "S", "5": "S", "6": "S",
         "7": "H", "8": "H", "9": "H", "10": "H", "A": "H"},
    17: {u: "S" for u in _DEALER_UPCARDS},
    18: {u: "S" for u in _DEALER_UPCARDS},
    19: {u: "S" for u in _DEALER_UPCARDS},
    20: {u: "S" for u in _DEALER_UPCARDS},
    21: {u: "S" for u in _DEALER_UPCARDS},
}

# Soft totals: total here is the optimistic (Ace=11) total.
# Soft 13 = A+2, soft 14 = A+3, ..., soft 21 = A+10 (= blackjack on 2 cards).
_SOFT: Dict[int, Dict[str, str]] = {
    13: {"2": "H", "3": "H", "4": "H", "5": "D", "6": "D",
         "7": "H", "8": "H", "9": "H", "10": "H", "A": "H"},
    14: {"2": "H", "3": "H", "4": "H", "5": "D", "6": "D",
         "7": "H", "8": "H", "9": "H", "10": "H", "A": "H"},
    15: {"2": "H", "3": "H", "4": "D", "5": "D", "6": "D",
         "7": "H", "8": "H", "9": "H", "10": "H", "A": "H"},
    16: {"2": "H", "3": "H", "4": "D", "5": "D", "6": "D",
         "7": "H", "8": "H", "9": "H", "10": "H", "A": "H"},
    17: {"2": "H", "3": "D", "4": "D", "5": "D", "6": "D",
         "7": "H", "8": "H", "9": "H", "10": "H", "A": "H"},
    18: {"2": "S", "3": "Ds", "4": "Ds", "5": "Ds", "6": "Ds",
         "7": "S", "8": "S", "9": "H", "10": "H", "A": "H"},
    19: {u: "S" for u in _DEALER_UPCARDS},
    20: {u: "S" for u in _DEALER_UPCARDS},
    21: {u: "S" for u in _DEALER_UPCARDS},
}

# ---------------- Hi-Lo deviations ("Illustrious 18", subset) ------------------
#
# Each entry: (hand_total, is_soft, dealer_upcard, threshold, action_above_threshold,
#              action_below_threshold).
# Convention: action_above applies when true_count >= threshold.

_DEVIATIONS: List[Tuple[int, bool, str, float, str, str]] = [
    # Most impactful four:
    (16, False, "10", 0.0, "S", "H"),  # 16 vs 10: stand at TC >= 0
    (15, False, "10", 4.0, "S", "H"),  # 15 vs 10
    (12, False, "3",  2.0, "S", "H"),  # 12 vs 3
    (12, False, "2",  3.0, "S", "H"),  # 12 vs 2
    # Doubling deviations:
    (11, False, "A",  1.0, "D", "H"),  # 11 vs A: double at TC >= 1
    (10, False, "10", 4.0, "D", "H"),
    (10, False, "A",  4.0, "D", "H"),
    (9,  False, "2",  1.0, "D", "H"),
    (9,  False, "7",  3.0, "D", "H"),
    # 13 vs 2 already standard 'S', skip
    (12, False, "4", -0.0, "S", "S"),  # already S — kept commented for parity
    (16, False, "9",  4.0, "S", "H"),
    (13, False, "2", -1.0, "S", "H"),  # take insurance below — TC<-1 -> hit
]


# ---------------- bust / outcome math -----------------------------------------

def remaining_deck(state: BlackjackState) -> Counter:
    """Return Counter of ranks remaining in the shoe.

    Assumes uniform 4 of each rank per deck minus seen_cards.
    """
    remaining = Counter()
    for rank in VALID_RANKS:
        remaining[rank] = 4 * state.decks_in_shoe
    for rank in state.seen_cards:
        remaining[rank] -= 1
    # Clamp negatives (shouldn't happen, defensive)
    for r in list(remaining.keys()):
        if remaining[r] < 0:
            remaining[r] = 0
    return remaining


def bust_probability_on_hit(hand: Hand, remaining: Counter) -> float:
    """Probability that the next single hit busts the hand.

    Uses exact remaining-card composition (not infinite-deck approx).
    """
    total = sum(remaining.values())
    if total == 0:
        return 0.0
    busts = 0
    for rank, count in remaining.items():
        if count == 0:
            continue
        new_hand = Hand(cards=hand.cards + [rank])
        if new_hand.is_bust():
            busts += count
    return busts / total


def dealer_bust_probability(dealer_upcard: str, remaining: Counter) -> float:
    """Approximate probability the dealer busts given just the upcard.

    Standard table (infinite-deck approximation, S17). We use a fixed lookup
    rather than a full Monte Carlo for speed. Good enough for UI display.
    """
    # Source: standard blackjack basic-strategy reference tables, S17.
    # Numbers are the dealer-bust % given that upcard.
    BUST_PCT = {
        "2": 0.353, "3": 0.375, "4": 0.402, "5": 0.418, "6": 0.423,
        "7": 0.262, "8": 0.245, "9": 0.230, "10": 0.214, "A": 0.116,
    }
    return BUST_PCT.get(dealer_upcard, 0.0)


# ---------------- recommendation API ------------------------------------------

class Recommendation(BaseModel):
    action: Action
    short_label: str  # 1-2 word HUD-style label, e.g. "HIT", "STAND", "DOUBLE"
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str       # one-line plain-English rationale
    used_deviation: bool = False
    diagnostics: "Diagnostics"


class Diagnostics(BaseModel):
    player_total: int
    player_soft: bool
    dealer_upcard: str
    cards_remaining: int
    running_count: int
    true_count: float
    bust_prob_on_hit: float
    dealer_bust_prob: float
    basic_action: Action
    deviation_threshold: Optional[float] = None


Recommendation.model_rebuild()


def _resolve_action_code(code: str, can_double: bool) -> Action:
    """Translate strategy table codes (H/S/D/Ds) into concrete Actions."""
    if code == "H":
        return "hit"
    if code == "S":
        return "stand"
    if code == "D":
        return "double" if can_double else "hit"
    if code == "Ds":
        return "double" if can_double else "stand"
    raise ValueError(f"unknown strategy code: {code}")


def basic_strategy(hand: Hand, dealer_upcard: str, can_double: bool = True) -> Tuple[Action, str]:
    """Return (action, code) per textbook S17 basic strategy.

    `dealer_upcard` may be any rank — J/Q/K are normalized to '10' internally.
    """
    upcard = _strategy_upcard(dealer_upcard)
    if upcard not in _DEALER_UPCARDS:
        raise ValueError(f"invalid dealer upcard: {dealer_upcard!r}")
    total, is_soft = hand.total()
    table = _SOFT if is_soft else _HARD
    if total < min(table.keys()):
        return "hit", "H"
    if total > max(table.keys()):
        return "stand", "S"
    code = table[total][upcard]
    return _resolve_action_code(code, can_double), code


def _check_deviations(player_total: int, is_soft: bool, dealer_upcard: str, tc: float, can_double: bool) -> Tuple[Optional[Action], Optional[float]]:
    for entry_total, entry_soft, entry_upcard, threshold, above, below in _DEVIATIONS:
        if entry_total != player_total or entry_soft != is_soft or entry_upcard != dealer_upcard:
            continue
        chosen_code = above if tc >= threshold else below
        return _resolve_action_code(chosen_code, can_double), threshold
    return None, None


def recommend(state: BlackjackState) -> Recommendation:
    """Make a recommendation for the player's current decision point."""
    if state.is_round_over():
        # Edge case: caller asked for advice on a finished round.
        return Recommendation(
            action="stand",
            short_label="ROUND OVER",
            confidence=1.0,
            reason=f"Round resolved as {state.round_outcome}.",
            diagnostics=_diagnostics_for(state, basic_action="stand"),
        )

    if not state.dealer.cards:
        raise ValueError("cannot recommend without dealer upcard")

    raw_upcard = state.dealer.cards[0]
    upcard = _strategy_upcard(raw_upcard)
    can_double = len(state.player.cards) == 2 and not state.player_doubled

    basic, _basic_code = basic_strategy(state.player, raw_upcard, can_double=can_double)

    # Hi-Lo true count
    rem = remaining_deck(state)
    rc = running_count(state.seen_cards)
    cr = state.cards_remaining()
    tc = true_count(rc, cr)

    # Deviation lookup
    p_total, p_soft = state.player.total()
    dev_action, dev_threshold = _check_deviations(p_total, p_soft, upcard, tc, can_double)

    used_deviation = dev_action is not None and dev_action != basic
    chosen = dev_action if used_deviation else basic

    # Build narration
    parts = []
    label = "HARD" if not p_soft else "SOFT"
    parts.append(f"{label} {p_total} vs dealer {upcard}")
    if used_deviation:
        parts.append(
            f"deviation: TC {tc:+.1f} >= {dev_threshold:+.1f} flips basic '{basic.upper()}' -> '{chosen.upper()}'"
        )
    else:
        parts.append(f"basic strategy: {basic.upper()}")
        if abs(tc) >= 1.0:
            parts.append(f"TC {tc:+.1f}")
    reason = ". ".join(parts) + "."

    diagnostics = _diagnostics_for(state, basic_action=basic)
    diagnostics.deviation_threshold = dev_threshold

    confidence = _confidence_for(chosen, basic, used_deviation, diagnostics)

    return Recommendation(
        action=chosen,
        short_label=chosen.upper(),
        confidence=confidence,
        reason=reason,
        used_deviation=used_deviation,
        diagnostics=diagnostics,
    )


def _diagnostics_for(state: BlackjackState, basic_action: Action) -> Diagnostics:
    rem = remaining_deck(state)
    rc = running_count(state.seen_cards)
    cr = state.cards_remaining()
    tc = true_count(rc, cr)
    p_total, p_soft = state.player.total()
    raw_upcard = state.dealer.cards[0] if state.dealer.cards else "?"
    upcard = _strategy_upcard(raw_upcard) if raw_upcard != "?" else "?"
    return Diagnostics(
        player_total=p_total,
        player_soft=p_soft,
        dealer_upcard=upcard,
        cards_remaining=cr,
        running_count=rc,
        true_count=round(tc, 2),
        bust_prob_on_hit=round(bust_probability_on_hit(state.player, rem), 3),
        dealer_bust_prob=round(dealer_bust_probability(upcard, rem), 3),
        basic_action=basic_action,
    )


def _confidence_for(chosen: Action, basic: Action, used_deviation: bool, diag: Diagnostics) -> float:
    """Heuristic confidence score for HUD display.

    - Pure basic (no count info) -> 0.85
    - Basic confirmed by neutral count -> 0.90
    - Deviation triggered -> 0.75 (we're departing from canonical, which is right but more contextual)
    - Forced action (bust prob >> 0.6 and we're hitting; or bust prob == 0 and we're hitting) -> bump
    """
    base = 0.85
    if used_deviation:
        base = 0.75
    if chosen == "hit" and diag.bust_prob_on_hit < 0.05:
        base = max(base, 0.95)
    if chosen == "stand" and diag.bust_prob_on_hit > 0.55:
        base = max(base, 0.95)
    if abs(diag.true_count) >= 3.0:
        base = min(base + 0.05, 0.97)
    return round(base, 2)
