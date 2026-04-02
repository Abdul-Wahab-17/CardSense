from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

from .models import HandAdvice, HandCard, HandState, HelpTip

_SUIT_ALIASES = {
    "S": "S",
    "SPADE": "S",
    "SPADES": "S",
    "H": "H",
    "HEART": "H",
    "HEARTS": "H",
    "D": "D",
    "DIAMOND": "D",
    "DIAMONDS": "D",
    "C": "C",
    "CLUB": "C",
    "CLUBS": "C",
}

_SUIT_SYMBOL = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}

_RANK_ALIASES = {
    "A": "A",
    "ACE": "A",
    "K": "K",
    "KING": "K",
    "Q": "Q",
    "QUEEN": "Q",
    "J": "J",
    "JACK": "J",
    "T": "10",
    "10": "10",
    "9": "9",
    "8": "8",
    "7": "7",
    "6": "6",
    "5": "5",
    "4": "4",
    "3": "3",
    "2": "2",
}

_RANK_ORDER = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13}


def _clean(label: str) -> str:
    token = label.strip().upper()
    for prefix in ("CARD_", "PLAYINGCARD_", "PLAYING_CARD_"):
        if token.startswith(prefix):
            token = token[len(prefix) :]
    token = token.replace("♠", "S").replace("♥", "H").replace("♦", "D").replace("♣", "C")
    token = re.sub(r"[^A-Z0-9]", "", token)
    return token


def _canonical(rank: str, suit: str) -> str:
    return f"{rank}{_SUIT_SYMBOL[suit]}"


def parse_card(label: str) -> tuple[str, str, str] | None:
    token = _clean(label)
    if not token:
        return None

    candidates: list[tuple[str, str]] = []

    if len(token) >= 2:
        rank_a, suit_a = token[:-1], token[-1]
        if rank_a in _RANK_ALIASES and suit_a in _SUIT_ALIASES:
            candidates.append((_RANK_ALIASES[rank_a], _SUIT_ALIASES[suit_a]))

        suit_b, rank_b = token[0], token[1:]
        if suit_b in _SUIT_ALIASES and rank_b in _RANK_ALIASES:
            candidates.append((_RANK_ALIASES[rank_b], _SUIT_ALIASES[suit_b]))

    for suit_key, suit in _SUIT_ALIASES.items():
        if token.endswith(suit_key):
            maybe_rank = token[: -len(suit_key)]
            if maybe_rank in _RANK_ALIASES:
                candidates.append((_RANK_ALIASES[maybe_rank], suit))

    for rank_key, rank in _RANK_ALIASES.items():
        if token.endswith(rank_key):
            maybe_suit = token[: -len(rank_key)]
            if maybe_suit in _SUIT_ALIASES:
                candidates.append((rank, _SUIT_ALIASES[maybe_suit]))

    if not candidates:
        return None

    rank, suit = candidates[0]
    return rank, suit, _canonical(rank, suit)


def _find_sequences(cards: Iterable[tuple[str, str]]) -> list[str]:
    by_suit: dict[str, set[int]] = defaultdict(set)
    for rank, suit in cards:
        by_suit[suit].add(_RANK_ORDER[rank])

    melds: list[str] = []
    for suit, values in by_suit.items():
        ordered = sorted(values)
        if not ordered:
            continue
        start = ordered[0]
        chain = [start]
        for value in ordered[1:]:
            if value == chain[-1] + 1:
                chain.append(value)
            else:
                if len(chain) >= 3:
                    melds.append(f"{chain[0]}-{chain[-1]}{_SUIT_SYMBOL[suit]}")
                chain = [value]
        if len(chain) >= 3:
            melds.append(f"{chain[0]}-{chain[-1]}{_SUIT_SYMBOL[suit]}")

    return melds


def build_thulla_advice(hand: HandState) -> HandAdvice:
    parsed: list[tuple[str, str, str, float]] = []
    unknown: list[HandCard] = []

    for card in hand.cards:
        parsed_card = parse_card(card.label)
        if not parsed_card:
            unknown.append(card)
            continue
        rank, suit, canonical = parsed_card
        parsed.append((rank, suit, canonical, card.confidence))

    suit_counts = Counter([suit for _, suit, _, _ in parsed])
    rank_counts = Counter([rank for rank, _, _, _ in parsed])

    tips: list[HelpTip] = []

    if len(hand.cards) < hand.max_cards:
        tips.append(
            HelpTip(
                title="Keep scanning",
                detail=f"Only {len(hand.cards)}/{hand.max_cards} cards are stable. Hold your hand steady for better confidence.",
            )
        )

    if suit_counts:
        strongest_suit, strongest_count = suit_counts.most_common(1)[0]
        tips.append(
            HelpTip(
                title="Strong suit",
                detail=f"{strongest_count} cards in {_SUIT_SYMBOL[strongest_suit]} — prioritize runs in this suit.",
            )
        )

        weakest_suit = min(suit_counts, key=lambda s: suit_counts[s])
        weakest_cards = [card for card in parsed if card[1] == weakest_suit]
        weakest_cards = sorted(weakest_cards, key=lambda x: (_RANK_ORDER[x[0]], -x[3]), reverse=True)
        discard_candidates = [card[2] for card in weakest_cards[:3]]
    else:
        discard_candidates = []

    set_ranks = [rank for rank, count in rank_counts.items() if count >= 2]
    if set_ranks:
        tips.append(
            HelpTip(
                title="Pairs/Sets ready",
                detail="Detected matching ranks: " + ", ".join(sorted(set_ranks, key=lambda r: _RANK_ORDER[r])),
            )
        )

    sequence_melds = _find_sequences((rank, suit) for rank, suit, _, _ in parsed)
    if sequence_melds:
        tips.append(
            HelpTip(
                title="Sequence opportunity",
                detail="Detected suit runs: " + ", ".join(sequence_melds),
            )
        )

    if unknown:
        tips.append(
            HelpTip(
                title="Unmapped labels",
                detail="Some detected labels are not parsed as standard cards. Verify class names in your model.",
            )
        )

    summary = "No cards detected yet."
    if parsed:
        summary = f"{len(parsed)} mapped cards. Best discard candidates: {', '.join(discard_candidates) if discard_candidates else 'N/A'}"

    return HandAdvice(
        camera_id=hand.camera_id,
        updated_ts_ms=hand.updated_ts_ms,
        summary=summary,
        discard_candidates=discard_candidates,
        tips=tips,
    )
