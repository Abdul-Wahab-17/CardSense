from __future__ import annotations

import copy
import time
from typing import Dict, List, Tuple

from .config import settings
from .models import CardLifecycleEntry, HandCard, HandState, MatchState, PlayedHistoryItem, RoundStartIn, TableTrickCard, VariantConfig


def _standard_deck_52() -> list[str]:
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    suits = ["S", "H", "D", "C"]
    return [f"{rank}{suit}" for suit in suits for rank in ranks]


def normalize_variant_config(raw_variant: VariantConfig) -> VariantConfig:
    variant = raw_variant.model_copy(deep=True)
    if not variant.deck_cards:
        variant.deck_cards = _standard_deck_52()

    dedup_seats: list[str] = []
    seen_seats: set[str] = set()
    for seat in variant.seat_order:
        seat_key = str(seat).strip()
        if not seat_key or seat_key in seen_seats:
            continue
        seen_seats.add(seat_key)
        dedup_seats.append(seat_key)

    if not dedup_seats:
        dedup_seats = ["me", "left", "across", "right"]

    variant.seat_order = dedup_seats
    variant.player_count = len(variant.seat_order)
    variant.cards_per_player = max(1, int(variant.cards_per_player or settings.default_cards_per_player))

    if variant.cards_per_player * variant.player_count > len(variant.deck_cards):
        raise ValueError("Invalid variant: cards_per_player * player_count exceeds deck size")

    if variant.player_count != len(variant.seat_order):
        raise ValueError("Invalid variant: player_count must match seat_order length")

    return variant


def rebuild_hand_cards_by_seat(match_state: MatchState) -> Dict[str, List[str]]:
    rebuilt: dict[str, list[str]] = {seat: [] for seat in match_state.variant.seat_order}
    cap = max(1, int(match_state.variant.cards_per_player or settings.default_cards_per_player))
    entries_by_seat: dict[str, list[tuple[str, CardLifecycleEntry]]] = {}
    for card, entry in match_state.card_lifecycle.items():
        if entry.location != "in_hand" or not entry.owner_seat:
            continue
        entries_by_seat.setdefault(entry.owner_seat, []).append((card, entry))

    for seat, rows in entries_by_seat.items():
        ordered = sorted(rows, key=lambda x: (float(x[1].confidence), int(x[1].seen_ts_ms)), reverse=True)
        rebuilt.setdefault(seat, [])
        rebuilt[seat] = [card for card, _ in ordered[:cap]]

    return rebuilt


def dedupe_global_cards(match_state: MatchState) -> MatchState:
    seen: set[str] = set()
    priority = {"won_pile": 4, "on_table": 3, "in_hand": 2, "unknown": 1}
    sorted_cards = sorted(
        match_state.card_lifecycle.items(),
        key=lambda x: (
            priority.get(x[1].location, 0),
            float(x[1].confidence),
            int(x[1].seen_ts_ms),
        ),
        reverse=True,
    )

    for card, entry in sorted_cards:
        if card not in seen:
            seen.add(card)
            continue
        entry.location = "unknown"
        entry.owner_seat = None
        entry.owner_team = None
        entry.last_transition = "dedupe_rejected"
        match_state.card_duplicate_rejected_total += 1

    match_state.hand_cards_by_seat = rebuild_hand_cards_by_seat(match_state)
    return match_state


def validate_state_invariants(match_state: MatchState, transition_events: list[dict] | None = None) -> List[str]:
    violations: list[str] = []
    deck = set(match_state.variant.deck_cards or [])

    lifecycle_cards = list(match_state.card_lifecycle.keys())
    if len(set(lifecycle_cards)) > len(deck):
        violations.append("lifecycle exceeds deck size")

    table_cards = [t.card for t in match_state.table_cards if t.card]
    if len(table_cards) != len(set(table_cards)):
        violations.append("duplicate card in current trick table")

    cap = max(1, int(match_state.variant.cards_per_player or settings.default_cards_per_player))
    for seat in match_state.variant.seat_order:
        actual = len(match_state.hand_cards_by_seat.get(seat, []))
        if actual > cap:
            violations.append(f"seat {seat} hand count {actual} exceeds cap {cap}")

    if transition_events:
        if any(e.get("type") == "TABLE_TO_WON" for e in transition_events) and len(table_cards) > 0:
            violations.append("played-to-won transition attempted before trick close")

    return violations


def reconcile_card_locations(
    game_id: str,
    ts_ms: int,
    hand_cards_observed: List[dict],
    table_cards_observed: List[dict],
    prev_lifecycle: Dict[str, CardLifecycleEntry],
    to_play_seat: str,
    trick_context: dict,
    variant: VariantConfig,
) -> Tuple[dict, List[dict], dict]:
    lifecycle: dict[str, CardLifecycleEntry] = {k: v.model_copy(deep=True) for k, v in prev_lifecycle.items()}
    transitions: list[dict] = []
    debug: dict = {"game_id": game_id, "to_play_seat": to_play_seat, "rejected": [], "observed": {}}

    dedup_hand: dict[str, dict] = {}
    dedup_table: dict[str, dict] = {}
    for item in hand_cards_observed:
        card = str(item.get("card", "")).strip()
        if not card:
            continue
        prev = dedup_hand.get(card)
        if prev is None or float(item.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
            dedup_hand[card] = dict(item)

    for item in table_cards_observed:
        card = str(item.get("card", "")).strip()
        if not card:
            continue
        prev = dedup_table.get(card)
        if prev is None or float(item.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
            dedup_table[card] = dict(item)

    debug["observed"]["hand"] = len(dedup_hand)
    debug["observed"]["table"] = len(dedup_table)

    replay_mode = bool(trick_context.get("replay_mode", False))
    hand_ttl = int(settings.hand_occlusion_ttl_ms)
    table_ttl = int(settings.table_occlusion_ttl_ms)

    for card, item in dedup_table.items():
        prev = lifecycle.get(card)
        if prev and prev.location == "won_pile" and not replay_mode:
            transitions.append(
                {
                    "type": "CARD_REJECTED",
                    "card": card,
                    "from_location": "won_pile",
                    "to_location": "on_table",
                    "seat": item.get("seat") or None,
                    "team": None,
                    "confidence": float(item.get("confidence", 0.0)),
                    "ts_ms": ts_ms,
                    "reason": "won_pile_reappearance",
                }
            )
            debug["rejected"].append({"card": card, "reason": "won_pile_reappearance"})
            continue

        event_type = "TABLE_CONFIRM"
        reason = "table_observed"
        if prev and prev.location == "in_hand":
            event_type = "PLAY_CARD"
            reason = "in_hand_to_table"

        lifecycle[card] = CardLifecycleEntry(
            location="on_table",
            owner_seat=str(item.get("seat", "") or ""),
            owner_team=None,
            seen_ts_ms=int(item.get("ts_ms", ts_ms) or ts_ms),
            confidence=float(item.get("confidence", 0.0)),
            last_transition=reason,
        )
        transitions.append(
            {
                "type": event_type,
                "card": card,
                "from_location": prev.location if prev else "unknown",
                "to_location": "on_table",
                "seat": item.get("seat") or None,
                "team": None,
                "confidence": float(item.get("confidence", 0.0)),
                "ts_ms": ts_ms,
                "reason": reason,
            }
        )

    for card, item in dedup_hand.items():
        if card in dedup_table:
            transitions.append(
                {
                    "type": "CARD_REJECTED",
                    "card": card,
                    "from_location": "in_hand",
                    "to_location": "on_table",
                    "seat": item.get("seat") or None,
                    "team": None,
                    "confidence": float(item.get("confidence", 0.0)),
                    "ts_ms": ts_ms,
                    "reason": "duplicate_observed_hand_and_table",
                }
            )
            continue

        prev = lifecycle.get(card)
        if prev and prev.location == "won_pile" and not replay_mode:
            transitions.append(
                {
                    "type": "CARD_REJECTED",
                    "card": card,
                    "from_location": "won_pile",
                    "to_location": "in_hand",
                    "seat": item.get("seat") or None,
                    "team": None,
                    "confidence": float(item.get("confidence", 0.0)),
                    "ts_ms": ts_ms,
                    "reason": "won_pile_reappearance",
                }
            )
            debug["rejected"].append({"card": card, "reason": "won_pile_reappearance"})
            continue

        lifecycle[card] = CardLifecycleEntry(
            location="in_hand",
            owner_seat=str(item.get("seat") or to_play_seat or "me"),
            owner_team=None,
            seen_ts_ms=int(item.get("ts_ms", ts_ms) or ts_ms),
            confidence=float(item.get("confidence", 0.0)),
            last_transition="hand_observed",
        )

    for card, prev in list(lifecycle.items()):
        if card in dedup_hand or card in dedup_table:
            continue

        age_ms = int(ts_ms) - int(prev.seen_ts_ms or 0)
        if prev.location == "in_hand" and age_ms <= hand_ttl:
            transitions.append(
                {
                    "type": "HAND_OCCLUDED",
                    "card": card,
                    "from_location": "in_hand",
                    "to_location": "in_hand",
                    "seat": prev.owner_seat,
                    "team": prev.owner_team,
                    "confidence": float(prev.confidence),
                    "ts_ms": ts_ms,
                    "reason": "within_hand_occlusion_ttl",
                }
            )
            continue

        if prev.location == "on_table" and age_ms <= table_ttl:
            continue

        lifecycle[card] = CardLifecycleEntry(
            location="unknown",
            owner_seat=None,
            owner_team=None,
            seen_ts_ms=int(ts_ms),
            confidence=float(prev.confidence),
            last_transition="occluded_to_unknown",
        )

    seats = list(variant.seat_order)
    max_cards = max(1, int(variant.cards_per_player or settings.default_cards_per_player))
    hand_cards_by_seat: dict[str, list[str]] = {seat: [] for seat in seats}
    for seat in seats:
        in_hand_rows = [
            (card, entry)
            for card, entry in lifecycle.items()
            if entry.location == "in_hand" and entry.owner_seat == seat
        ]
        in_hand_rows.sort(key=lambda x: (float(x[1].confidence), int(x[1].seen_ts_ms)), reverse=True)
        for card, _ in in_hand_rows[:max_cards]:
            hand_cards_by_seat[seat].append(card)
        for card, _ in in_hand_rows[max_cards:]:
            transitions.append(
                {
                    "type": "CARD_REJECTED",
                    "card": card,
                    "from_location": "in_hand",
                    "to_location": "unknown",
                    "seat": seat,
                    "team": None,
                    "confidence": 0.0,
                    "ts_ms": ts_ms,
                    "reason": "seat_hand_cap_exceeded",
                }
            )
            lifecycle[card].location = "unknown"
            lifecycle[card].owner_seat = None
            lifecycle[card].owner_team = None
            lifecycle[card].last_transition = "seat_hand_cap_exceeded"

    table_cards = []
    for card, entry in lifecycle.items():
        if entry.location != "on_table":
            continue
        table_cards.append(
            {
                "seat": entry.owner_seat or "unknown",
                "card": card,
                "confidence": float(entry.confidence),
            }
        )

    state_patch = {
        "hand_cards_by_seat": hand_cards_by_seat,
        "table_cards": table_cards,
        "card_lifecycle": lifecycle,
    }
    return state_patch, transitions, debug


def project_hand_state_for_camera(match_state: MatchState, camera_id: str, seat_hint: str) -> HandState:
    cards = [
        HandCard(label=card, confidence=float(match_state.card_lifecycle.get(card, CardLifecycleEntry()).confidence))
        for card in match_state.hand_cards_by_seat.get(seat_hint, [])
    ]
    return HandState(
        camera_id=camera_id,
        updated_ts_ms=int(time.time() * 1000),
        max_cards=max(1, int(match_state.variant.cards_per_player or settings.default_cards_per_player)),
        cards=cards,
    )


class MatchEngine:
    def __init__(self) -> None:
        self._states: Dict[str, MatchState] = {}
        self._pending_patch: dict[str, dict] = {}
        self._recent_invariant_violations: dict[str, list[str]] = {}

    def set_variant_config(self, game_id: str, variant: VariantConfig) -> dict:
        normalized = normalize_variant_config(variant)
        state = self._states.get(game_id)
        if state is None:
            state = MatchState(game_id=game_id, variant=normalized)
            state.team_points = {team: 0 for team in set(normalized.team_map.values())}
            state.team_tricks = {team: 0 for team in set(normalized.team_map.values())}
            state.hand_cards_by_seat = {seat: [] for seat in normalized.seat_order}
            self._states[game_id] = state
        else:
            state.variant = normalized
            state.hand_cards_by_seat = {seat: state.hand_cards_by_seat.get(seat, []) for seat in normalized.seat_order}
        return {"variant_version": int(time.time()), "effective_config": state.variant.model_dump()}

    def start_match(self, game_id: str, variant: VariantConfig, initial_turn: str) -> MatchState:
        normalized = normalize_variant_config(variant)
        state = MatchState(
            game_id=game_id,
            phase="in_round",
            round_index=1,
            dealer_seat=normalized.seat_order[0] if normalized.seat_order else "me",
            leader_seat=initial_turn,
            to_play_seat=initial_turn,
            variant=normalized,
            team_points={team: 0 for team in set(normalized.team_map.values())},
            team_tricks={team: 0 for team in set(normalized.team_map.values())},
            hand_cards_by_seat={seat: [] for seat in normalized.seat_order},
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
        state.card_lifecycle = {
            card: entry
            for card, entry in state.card_lifecycle.items()
            if entry.location in {"in_hand", "unknown"}
        }
        state.hand_cards_by_seat = rebuild_hand_cards_by_seat(state)
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
        ts_ms: int,
        hand_cards_observed: list[dict],
        table_cards_observed: list[dict],
        prev_lifecycle: dict[str, CardLifecycleEntry],
        to_play_seat: str,
        trick_context: dict,
        variant: VariantConfig,
    ) -> tuple[dict, list[dict], dict]:
        state_patch, transition_events, reconciliation_debug = reconcile_card_locations(
            game_id=game_id,
            ts_ms=ts_ms,
            hand_cards_observed=hand_cards_observed,
            table_cards_observed=table_cards_observed,
            prev_lifecycle=prev_lifecycle,
            to_play_seat=to_play_seat,
            trick_context=trick_context,
            variant=variant,
        )
        self._pending_patch[game_id] = state_patch
        return state_patch, transition_events, reconciliation_debug

    def apply_transition_events(self, match_state: MatchState, transition_events: List[dict], ts_ms: int) -> Tuple[MatchState, List[str]]:
        prev_state = match_state.model_copy(deep=True)
        patch = self._pending_patch.get(match_state.game_id, {})
        patch_lifecycle = patch.get("card_lifecycle", {})

        if patch_lifecycle:
            match_state.card_lifecycle = {
                card: (entry if isinstance(entry, CardLifecycleEntry) else CardLifecycleEntry.model_validate(entry))
                for card, entry in patch_lifecycle.items()
            }
        if patch.get("table_cards") is not None:
            match_state.table_cards = [
                TableTrickCard(
                    seat=str(item.get("seat", "unknown")),
                    card=str(item.get("card", "")),
                    confidence=float(item.get("confidence", 0.0)),
                )
                for item in patch.get("table_cards", [])
                if item.get("card")
            ]

        match_state = dedupe_global_cards(match_state)
        match_state.hand_cards_by_seat = rebuild_hand_cards_by_seat(match_state)

        trick_size = max(1, int(match_state.variant.trick_size or len(match_state.variant.seat_order) or 4))
        if len(match_state.table_cards) >= trick_size:
            trick_cards = list(match_state.table_cards[:trick_size])
            lead = trick_cards[0].card[-1] if trick_cards and trick_cards[0].card else ""
            winner = self._resolve_trick_winner(trick_cards, lead, match_state.variant.trump_suit, match_state.variant.rank_order_map)
            winner_team = str(match_state.variant.team_map.get(winner, "")) or None

            trick_no = (len(match_state.played_history) // trick_size) + 1
            for t in trick_cards:
                match_state.played_history.append(PlayedHistoryItem(trick_no=trick_no, seat=t.seat, card=t.card))
                if t.card in match_state.card_lifecycle:
                    match_state.card_lifecycle[t.card].location = "won_pile"
                    match_state.card_lifecycle[t.card].owner_seat = None
                    match_state.card_lifecycle[t.card].owner_team = winner_team
                    match_state.card_lifecycle[t.card].seen_ts_ms = int(ts_ms)
                    match_state.card_lifecycle[t.card].last_transition = "table_to_won"
                    transition_events.append(
                        {
                            "type": "TABLE_TO_WON",
                            "card": t.card,
                            "from_location": "on_table",
                            "to_location": "won_pile",
                            "seat": None,
                            "team": winner_team,
                            "confidence": float(t.confidence),
                            "ts_ms": int(ts_ms),
                            "reason": "trick_closed",
                        }
                    )

            if winner_team:
                match_state.team_tricks[winner_team] = int(match_state.team_tricks.get(winner_team, 0)) + 1
            match_state.table_cards = []
            match_state.lead_suit = ""
            match_state.trick_id = int(match_state.trick_id) + 1
            match_state.leader_seat = winner
            match_state.to_play_seat = winner
        else:
            if match_state.table_cards:
                card = match_state.table_cards[0].card
                match_state.lead_suit = card[-1] if card else ""
                expected = match_state.leader_seat
                for _ in range(len(match_state.table_cards)):
                    expected = self._next_player(match_state, expected)
                match_state.to_play_seat = expected
            else:
                match_state.lead_suit = ""

        violations = validate_state_invariants(match_state, transition_events)
        if violations:
            match_state.invariant_violations_recent = (match_state.invariant_violations_recent + violations)[-25:]
            match_state.rule_violations_recent = (match_state.rule_violations_recent + violations)[-25:]
            self._recent_invariant_violations[match_state.game_id] = list(match_state.invariant_violations_recent)
            for row in violations:
                if "hand count" in row:
                    match_state.hand_count_violation_total += 1
            self._states[match_state.game_id] = prev_state
            return prev_state, violations

        self._states[match_state.game_id] = match_state
        self._pending_patch.pop(match_state.game_id, None)
        return match_state, []

    def project_hand_state_for_camera(self, game_id: str, camera_id: str, seat_hint: str) -> HandState:
        state = self._states.get(game_id)
        if state is None:
            return HandState(camera_id=camera_id, updated_ts_ms=int(time.time() * 1000), max_cards=settings.default_cards_per_player, cards=[])
        state.hand_cards_by_seat = rebuild_hand_cards_by_seat(state)
        return project_hand_state_for_camera(state, camera_id, seat_hint)

    def metrics(self, game_id: str) -> dict:
        state = self._states.get(game_id)
        if state is None:
            return {
                "invariant_violations_recent": [],
                "card_duplicate_rejected_total": 0,
                "hand_count_violation_total": 0,
                "expected_vs_actual_hand_count": {},
            }

        expected = max(1, int(state.variant.cards_per_player or settings.default_cards_per_player))
        expected_vs_actual: dict[str, dict[str, int]] = {}
        for seat in state.variant.seat_order:
            expected_vs_actual[seat] = {
                "expected": expected,
                "actual": len(state.hand_cards_by_seat.get(seat, [])),
            }

        return {
            "invariant_violations_recent": list(state.invariant_violations_recent),
            "card_duplicate_rejected_total": int(state.card_duplicate_rejected_total),
            "hand_count_violation_total": int(state.hand_count_violation_total),
            "expected_vs_actual_hand_count": expected_vs_actual,
        }

    def _next_player(self, state: MatchState, current: str) -> str:
        seats = [s for s in state.variant.seat_order if s]
        if not seats:
            return current
        if current not in seats:
            return seats[0]
        idx = seats.index(current)
        step = 1 if state.variant.play_direction == "clockwise" else -1
        return seats[(idx + step) % len(seats)]

    @staticmethod
    def _rank_value(card: str, rank_order_map: dict[str, int] | None = None) -> int:
        rank = card[:-1]
        order = rank_order_map or {
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
        return int(order.get(rank, 0))

    def _resolve_trick_winner(self, cards: list[TableTrickCard], lead_suit: str, trump_suit: str, rank_order_map: dict[str, int]) -> str:
        lead = (lead_suit or "").upper()
        trump = (trump_suit or "").upper()
        trump_cards = [c for c in cards if trump and c.card and c.card.endswith(trump)]
        if trump_cards:
            candidates = trump_cards
        else:
            candidates = [c for c in cards if c.card and c.card.endswith(lead)] if lead else list(cards)
        if not candidates:
            candidates = list(cards)
        winner = max(candidates, key=lambda c: self._rank_value(c.card, rank_order_map), default=None)
        return winner.seat if winner else (cards[0].seat if cards else "me")
