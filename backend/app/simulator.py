from __future__ import annotations

import math
import random
from statistics import mean, pstdev
from typing import List

from .belief import sample_worlds
from .models import ActionScore, CanonicalGameState
from .rules import card_suit, legal_actions


def _rank_value(card: str) -> int:
    rank = card[:-1]
    order = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }
    return order.get(rank, 0)


def opponent_policy(cards: list[str], lead_suit: str) -> str:
    if not cards:
        return ""
    if lead_suit:
        suited = [c for c in cards if card_suit(c) == lead_suit]
        if suited:
            return max(suited, key=_rank_value)
    return max(cards, key=_rank_value)


def partner_policy(cards: list[str], lead_suit: str) -> str:
    if not cards:
        return ""
    if lead_suit:
        suited = [c for c in cards if card_suit(c) == lead_suit]
        if suited:
            return max(suited, key=_rank_value)
    return max(cards, key=_rank_value)


def self_policy(action: str, _state: CanonicalGameState) -> str:
    return action


def evaluate_actions(state: CanonicalGameState, rollout_count: int = 96, simulation_seed: int | None = None) -> List[ActionScore]:
    actions = legal_actions(state)
    if not actions:
        return []

    rng = random.Random(simulation_seed if simulation_seed is not None else int(state.updated_ts_ms or 0) + 17)
    worlds = sample_worlds(state, n=max(24, rollout_count // 2), rng=rng)
    scores: List[ActionScore] = []

    lead = state.turn.lead_suit.upper() if state.turn.lead_suit else ""
    table_cards = [c.card for c in state.table_trick_cards]
    highest_table = max([_rank_value(c) for c in table_cards], default=0)

    for action in actions:
        trick_outcomes = []
        partner_support_samples = []
        future_samples = []
        ev_now_samples = []
        ev_future_samples = []
        ev_partner_samples = []
        ev_info_samples = []

        for world in worlds:
            rv = _rank_value(action)
            same_suit_bonus = 0.08 if lead and card_suit(action) == lead else 0.0
            beats_table = 1.0 if rv >= highest_table else 0.0

            # simple opponent/partner policy using sampled hidden hands
            opp_right = world.get("right", [])
            partner = world.get("across", [])
            opp_left = world.get("left", [])

            lead_suit_now = lead or card_suit(action)
            right_play = opponent_policy(opp_right, lead_suit_now)
            partner_play = partner_policy(partner, lead_suit_now)
            left_play = opponent_policy(opp_left, lead_suit_now)
            my_play = self_policy(action, state)

            contenders = [my_play]
            if right_play:
                contenders.append(right_play)
            if partner_play:
                contenders.append(partner_play)
            if left_play:
                contenders.append(left_play)

            winning_card = max(contenders, key=_rank_value)
            p_win = 1.0 if winning_card == my_play else 0.0
            # inject small uncertainty band for imperfect information
            p_win = max(0.0, min(1.0, p_win * 0.85 + rng.uniform(0.0, 0.15)))
            trick_outcomes.append(p_win)

            partner_support = 1.0 if partner_play and _rank_value(partner_play) >= _rank_value(action) else 0.45
            partner_support = max(0.0, min(1.0, partner_support + rng.uniform(-0.08, 0.08)))
            partner_support_samples.append(partner_support)

            preserve_bonus = 0.15 if rv < 10 else -0.05
            future_val = (rv / 14.0) * 0.28 + partner_support * 0.30 + preserve_bonus + rng.uniform(-0.10, 0.10)
            future_samples.append(future_val)

            ev_now = 0.55 * p_win
            ev_future = 0.30 * future_val
            ev_partner = 0.15 * partner_support
            info_gain = 0.06 if not beats_table and same_suit_bonus <= 0 else 0.02
            ev_info = info_gain - (0.03 if rv > 11 else 0.0)
            ev_now_samples.append(ev_now)
            ev_future_samples.append(ev_future)
            ev_partner_samples.append(ev_partner)
            ev_info_samples.append(ev_info)

        p_win_current_trick = mean(trick_outcomes)
        expected_future = mean(future_samples)
        partner_support = mean(partner_support_samples)
        risk = pstdev(trick_outcomes) if len(trick_outcomes) > 1 else 0.0
        ev_now = mean(ev_now_samples)
        ev_future = mean(ev_future_samples)
        ev_partner = mean(ev_partner_samples)
        ev_info = mean(ev_info_samples)
        total_ev = ev_now + ev_future + ev_partner + ev_info - 0.20 * risk

        scores.append(
            ActionScore(
                card=action,
                p_win_current_trick=round(p_win_current_trick, 4),
                expected_future_trick_value=round(expected_future, 4),
                partner_support_value=round(partner_support, 4),
                risk=round(risk, 4),
                ev_now=round(ev_now, 4),
                ev_future=round(ev_future, 4),
                ev_partner_synergy=round(ev_partner, 4),
                ev_information_gain=round(ev_info, 4),
                total_ev=round(total_ev, 4),
            )
        )

    scores.sort(key=lambda s: s.total_ev, reverse=True)
    return scores
