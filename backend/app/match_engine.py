from __future__ import annotations

import time
from typing import Dict

from .models import CardLifecycleEntry, MatchState, RoundStartIn, VariantConfig


class MatchEngine:
    def __init__(self) -> None:
        self._states: Dict[str, MatchState] = {}

    def set_variant_config(self, game_id: str, variant: VariantConfig) -> dict:
        state = self._states.get(game_id)
        if state is None:
            state = MatchState(game_id=game_id, variant=variant)
            state.team_points = {team: 0 for team in set(variant.team_map.values())}
            state.team_tricks = {team: 0 for team in set(variant.team_map.values())}
            self._states[game_id] = state
        else:
            state.variant = variant
        return {"variant_version": int(time.time()), "effective_config": state.variant.model_dump()}

    def start_match(self, game_id: str, variant: VariantConfig, initial_turn: str) -> MatchState:
        state = MatchState(
            game_id=game_id,
            phase="in_round",
            round_index=1,
            dealer_seat=variant.seat_order[0] if variant.seat_order else "me",
            leader_seat=initial_turn,
            to_play_seat=initial_turn,
            variant=variant,
            team_points={team: 0 for team in set(variant.team_map.values())},
            team_tricks={team: 0 for team in set(variant.team_map.values())},
        )
        self._states[game_id] = state
        return state

    def start_round(self, game_id: str, body: RoundStartIn) -> MatchState:
        state = self._states.get(game_id)
        if state is None:
            raise ValueError("match not started")
        state.phase = "in_round"
        state.round_index = int(body.round_index)
        state.dealer_seat = body.dealer_seat
        state.table_cards = []
        state.played_history = []
        state.lead_suit = ""
        state.trick_id = 1
        state.round_winner_team = None
        state.to_play_seat = self._next_player(state, body.dealer_seat)
        state.leader_seat = state.to_play_seat
        return state

    def end_round(self, game_id: str) -> MatchState:
        state = self._states.get(game_id)
        if state is None:
            raise ValueError("match not started")
        state.phase = "round_end"
        top_team = None
        if state.team_tricks:
            top_team = max(state.team_tricks.items(), key=lambda kv: kv[1])[0]
        state.round_winner_team = top_team
        if top_team:
            state.team_points[top_team] = int(state.team_points.get(top_team, 0)) + 1
        if state.variant.target_points is not None:
            if any(v >= int(state.variant.target_points) for v in state.team_points.values()):
                state.phase = "match_end"
                state.match_winner_team = max(state.team_points.items(), key=lambda kv: kv[1])[0]
        return state

    def get(self, game_id: str) -> MatchState | None:
        return self._states.get(game_id)

    def reconcile_card_locations(
        self,
        game_id: str,
        hand_cards: list[str],
        table_cards: list[dict],
        won_piles: dict[str, list[str]],
        ts_ms: int,
    ) -> dict[str, CardLifecycleEntry]:
        state = self._states.get(game_id)
        if state is None:
            return {}

        lifecycle: dict[str, CardLifecycleEntry] = dict(state.card_lifecycle)
        for card in hand_cards:
            if not card:
                continue
            lifecycle[card] = CardLifecycleEntry(
                location="in_hand",
                owner_seat="me",
                seen_ts_ms=int(ts_ms),
                last_transition="reconciled_hand",
                confidence=1.0,
            )

        for item in table_cards:
            card = str(item.get("card", ""))
            seat = str(item.get("seat", ""))
            if not card:
                continue
            lifecycle[card] = CardLifecycleEntry(
                location="on_table",
                owner_seat=seat or None,
                seen_ts_ms=int(ts_ms),
                last_transition="reconciled_table",
                confidence=float(item.get("confidence", 0.0) or 0.0),
            )

        for pile_cards in won_piles.values():
            for card in pile_cards or []:
                if not card:
                    continue
                lifecycle[card] = CardLifecycleEntry(
                    location="won_pile",
                    owner_seat=None,
                    seen_ts_ms=int(ts_ms),
                    last_transition="reconciled_won_pile",
                    confidence=1.0,
                )

        state.card_lifecycle = lifecycle
        return lifecycle

    def apply_transition_events(
        self,
        game_id: str,
        transition: dict,
        *,
        allow_won_pile_replay: bool = False,
    ) -> dict:
        state = self._states.get(game_id)
        if state is None:
            return {"accepted": False, "reason": "match_not_started"}

        accepted = bool(transition.get("accepted", False))
        if not accepted:
            return {
                "accepted": False,
                "reason": str(transition.get("reason", "rejected")),
                "reason_code": str(transition.get("reason_code", "rejected")),
            }

        reason_code = str(transition.get("reason_code", "ok"))
        if reason_code == "won_pile_reappeared" and not allow_won_pile_replay:
            return {"accepted": False, "reason": "won_pile_replay_blocked", "reason_code": reason_code}

        trick_size = max(1, int(state.variant.trick_size or 4))
        table_size = int(transition.get("table_size", 0) or 0)
        if table_size >= trick_size:
            state.trick_id = int(state.trick_id) + 1
            state.lead_suit = ""

        state.to_play_seat = self._next_player(state, state.to_play_seat)
        return {"accepted": True, "reason": "applied", "reason_code": "ok", "trick_size": trick_size}

    def _next_player(self, state: MatchState, current: str) -> str:
        seats = [s for s in state.variant.seat_order if s]
        if not seats:
            return current
        if current not in seats:
            return seats[0]
        idx = seats.index(current)
        step = 1 if state.variant.play_direction == "clockwise" else -1
        return seats[(idx + step) % len(seats)]
