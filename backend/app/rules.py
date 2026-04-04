from __future__ import annotations

from typing import Dict, List

from .models import CanonicalGameState


def card_suit(card: str) -> str:
    if not card:
        return ""
    return card[-1].upper()


def variant_profile(state: CanonicalGameState) -> Dict[str, object]:
    rules = state.rules
    trump_mode = (rules.trump_mode or "").lower()
    return {
        "variant_name": rules.variant_name or "default",
        "must_follow_suit": bool(rules.must_follow_suit),
        "must_trump_if_void": bool(rules.must_trump_if_void)
        or trump_mode in {"must_trump_if_void", "force_trump_if_void"},
        "trump_suit": (rules.trump_suit or "").upper(),
        "rank_order_map": dict(rules.rank_order_map or {}),
    }


def legal_actions(state: CanonicalGameState) -> List[str]:
    hand = list(state.hand_cards)
    if not hand:
        return []

    # strict turn guardrail: only act on our turn
    if state.turn.to_play and state.turn.to_play != "me":
        return []

    profile = variant_profile(state)
    must_follow = bool(profile["must_follow_suit"])
    if not must_follow or not state.turn.lead_suit:
        return sorted(set(hand))

    lead = state.turn.lead_suit.upper()
    follow = sorted({c for c in hand if card_suit(c) == lead})
    if follow:
        return follow

    if bool(profile["must_trump_if_void"]):
        trump_suit = str(profile["trump_suit"] or "S").upper()
        trumps = sorted({c for c in hand if card_suit(c) == trump_suit})
        if trumps:
            return trumps

    return sorted(set(hand))
