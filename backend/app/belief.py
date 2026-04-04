from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List
import math

from .models import CanonicalGameState
from .config import settings

RANKS = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS = ["S", "H", "D", "C"]
FULL_DECK = [f"{r}{s}" for r in RANKS for s in SUITS]
SEATS = ["left", "across", "right"]


def observed_cards(state: CanonicalGameState) -> set[str]:
    seen = set(state.hand_cards)
    seen.update([item.card for item in state.table_trick_cards])
    seen.update([item.card for item in state.played_history])
    seen.update(state.won_piles.get("team_me", []))
    seen.update(state.won_piles.get("team_opp", []))
    return seen


def build_belief(state: CanonicalGameState) -> Dict[str, Dict[str, float]]:
    unseen = [c for c in FULL_DECK if c not in observed_cards(state)]
    if not unseen:
        return {seat: {} for seat in SEATS}

    # Uniform baseline belief for unseen cards across opponents
    belief = {seat: {} for seat in SEATS}
    p = 1.0 / len(SEATS)
    for card in unseen:
        for seat in SEATS:
            belief[seat][card] = p

    if not bool(settings.feature_belief_bayes_update):
        return belief

    # seat-specific evidence from played history ledger
    by_trick: Dict[int, List] = defaultdict(list)
    for item in state.played_history:
        by_trick[item.trick_no].append(item)

    seat_void_suit_penalty: Dict[tuple[str, str], float] = defaultdict(lambda: 1.0)
    seat_strength_estimate: Dict[str, Dict[str, float]] = {
        s: {suit: 1.0 for suit in SUITS} for s in SEATS
    }
    for _, rows in by_trick.items():
        if not rows:
            continue
        lead_card = rows[0].card
        lead_suit = lead_card[-1].upper() if lead_card else ""
        if not lead_suit:
            continue
        for move in rows[1:]:
            card = move.card or ""
            if card and not card.endswith(lead_suit):
                seat_void_suit_penalty[(move.seat, lead_suit)] *= 0.35
                if move.seat in seat_strength_estimate:
                    seat_strength_estimate[move.seat][lead_suit] *= 0.65
            elif card and move.seat in seat_strength_estimate:
                suit = card[-1].upper()
                seat_strength_estimate[move.seat][suit] *= 1.05

    for seat in SEATS:
        for card in list(belief[seat].keys()):
            suit = card[-1].upper()
            belief[seat][card] *= float(seat_void_suit_penalty[(seat, suit)])
            belief[seat][card] *= float(seat_strength_estimate.get(seat, {}).get(suit, 1.0))
        total = sum(belief[seat].values())
        if total > 0:
            for card in list(belief[seat].keys()):
                belief[seat][card] /= total

    return belief


def update_belief_from_play_event(play_event: dict, prior_belief: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    seat = str(play_event.get("seat", "")).lower()
    lead_suit = str(play_event.get("lead_suit", "")).upper()
    card = str(play_event.get("card", ""))
    if seat not in prior_belief:
        return prior_belief

    next_belief = {s: dict(cards) for s, cards in prior_belief.items()}
    if card:
        for s in next_belief:
            next_belief[s].pop(card, None)

    if lead_suit and card and card[-1].upper() != lead_suit:
        # seat likely void in lead suit
        for hidden in list(next_belief.get(seat, {}).keys()):
            if hidden.endswith(lead_suit):
                next_belief[seat][hidden] *= 0.25

    # normalize seat beliefs
    for s in next_belief:
        total = sum(max(v, 0.0) for v in next_belief[s].values())
        if total > 0:
            for h in list(next_belief[s].keys()):
                next_belief[s][h] = max(next_belief[s][h], 0.0) / total
    return next_belief


def belief_diagnostics(belief: Dict[str, Dict[str, float]]) -> dict:
    mass = {}
    entropy = {}
    for seat in SEATS:
        dist = belief.get(seat, {})
        m = float(sum(max(v, 0.0) for v in dist.values()))
        mass[seat] = round(m, 6)
        e = 0.0
        for p in dist.values():
            p = max(min(float(p), 1.0), 1e-12)
            e -= p * math.log(p, 2)
        entropy[seat] = round(e, 6)
    return {
        "belief_mass_per_seat": mass,
        "belief_entropy_per_seat": entropy,
    }


def sample_worlds(state: CanonicalGameState, n: int = 64, rng: random.Random | None = None) -> List[Dict[str, List[str]]]:
    unseen = [c for c in FULL_DECK if c not in observed_cards(state)]
    # rough seat card count constraints: 13 minus already played by seat
    played_by_seat = defaultdict(int)
    for item in state.played_history:
        played_by_seat[item.seat] += 1
    target_counts = {
        seat: max(0, 13 - int(played_by_seat.get(seat, 0)))
        for seat in SEATS
    }

    worlds = []
    randomizer = rng if rng is not None else random
    for _ in range(max(1, n)):
        shuffled = unseen[:]
        randomizer.shuffle(shuffled)
        world = defaultdict(list)
        remaining = target_counts.copy()
        for card in shuffled:
            candidate_seats = [s for s in SEATS if remaining[s] > 0]
            if not candidate_seats:
                candidate_seats = SEATS
            seat = randomizer.choice(candidate_seats)
            world[seat].append(card)
            if remaining.get(seat, 0) > 0:
                remaining[seat] -= 1
        worlds.append(dict(world))
    return worlds
