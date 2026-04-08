from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import time

from .models import CanonicalGameState, GameRules, PlayedHistoryItem, TableTrickCard, TurnState, UncertaintyBlock
from .config import settings


@dataclass
class TransitionResult:
    accepted: bool
    reason: str
    reason_code: str = "ok"
    reason_detail: str = ""


class GameStateEngine:
    def __init__(self) -> None:
        self._states: Dict[str, CanonicalGameState] = {}
        self._card_lifecycle: Dict[str, Dict[str, dict]] = {}
        self._metrics = {
            "transitions_total": 0,
            "transitions_rejected": 0,
            "illegal_transition_rejects": 0,
            "transition_reject_reason_code": {},
        }

    def _ensure_lifecycle(self, game_id: str) -> Dict[str, dict]:
        if game_id not in self._card_lifecycle:
            self._card_lifecycle[game_id] = {}
        return self._card_lifecycle[game_id]

    def _mark_card_state(self, game_id: str, card: str, state: str, ts_ms: int, seat: str = "") -> None:
        if not card:
            return
        lifecycle = self._ensure_lifecycle(game_id)
        lifecycle[card] = {
            "state": state,
            "updated_ts_ms": int(ts_ms),
            "seat": seat,
        }

    @staticmethod
    def _seat_order_from_state(state: CanonicalGameState) -> list[str]:
        seats = [s for s in (state.rules.seat_order or []) if s]
        return seats or ["me", "left", "across", "right"]

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
        return order.get(rank, 0)

    def _next_player(self, state: CanonicalGameState, current: str) -> str:
        seats = self._seat_order_from_state(state)
        if current not in seats:
            return seats[0]
        idx = seats.index(current)
        step = 1 if str(state.rules.play_direction or "clockwise") == "clockwise" else -1
        return seats[(idx + step) % len(seats)]

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
        return winner.seat if winner else "me"

    def start_game(
        self,
        game_id: str,
        leader: str,
        to_play: str,
        must_follow_suit: bool,
        trump_mode: str,
        variant_name: str = "default",
        seat_order: list[str] | None = None,
        play_direction: str = "clockwise",
        trick_size: int = 4,
        must_trump_if_void: bool = False,
        trump_suit: str = "",
        rank_order_map: dict[str, int] | None = None,
    ) -> CanonicalGameState:
        now = int(time.time() * 1000)
        state = CanonicalGameState(
            game_id=game_id,
            turn=TurnState(leader=leader, to_play=to_play, lead_suit=""),
            rules=GameRules(
                variant_name=variant_name,
                seat_order=seat_order or ["me", "left", "across", "right"],
                play_direction=play_direction if play_direction in {"clockwise", "counterclockwise"} else "clockwise",
                trick_size=max(1, int(trick_size or 4)),
                must_follow_suit=must_follow_suit,
                must_trump_if_void=must_trump_if_void,
                trump_mode=trump_mode,
                trump_suit=trump_suit,
                rank_order_map=rank_order_map or GameRules().rank_order_map,
            ),
            won_piles={"team_me": [], "team_opp": []},
            trick_id=1,
            trick_open_ts_ms=now,
            state_version=1,
            last_transition_id=f"{game_id}:start:{now}",
            updated_ts_ms=now,
        )
        self._states[game_id] = state
        self._card_lifecycle[game_id] = {}
        return state

    def get(self, game_id: str) -> CanonicalGameState:
        if game_id not in self._states:
            self._states[game_id] = CanonicalGameState(game_id=game_id, won_piles={"team_me": [], "team_opp": []})
        return self._states[game_id]

    def reset(self, game_id: str) -> CanonicalGameState:
        now = int(time.time() * 1000)
        state = CanonicalGameState(
            game_id=game_id,
            won_piles={"team_me": [], "team_opp": []},
            trick_id=1,
            trick_open_ts_ms=now,
            state_version=1,
            last_transition_id=f"{game_id}:reset:{now}",
            updated_ts_ms=now,
        )
        self._states[game_id] = state
        self._card_lifecycle[game_id] = {}
        return state

    def end(self, game_id: str) -> bool:
        self._card_lifecycle.pop(game_id, None)
        return self._states.pop(game_id, None) is not None

    def all_states(self) -> list[CanonicalGameState]:
        return list(self._states.values())

    def configure(self, game_id: str, **kwargs) -> CanonicalGameState:
        state = self.get(game_id)
        if kwargs.get("leader") is not None:
            state.turn.leader = str(kwargs["leader"])
        if kwargs.get("to_play") is not None:
            state.turn.to_play = str(kwargs["to_play"])
        if kwargs.get("lead_suit") is not None:
            state.turn.lead_suit = str(kwargs["lead_suit"])
        if kwargs.get("must_follow_suit") is not None:
            state.rules.must_follow_suit = bool(kwargs["must_follow_suit"])
        if kwargs.get("must_trump_if_void") is not None:
            state.rules.must_trump_if_void = bool(kwargs["must_trump_if_void"])
        if kwargs.get("trump_mode") is not None:
            state.rules.trump_mode = str(kwargs["trump_mode"])
        if kwargs.get("trump_suit") is not None:
            state.rules.trump_suit = str(kwargs["trump_suit"])
        if kwargs.get("variant_name") is not None:
            state.rules.variant_name = str(kwargs["variant_name"])
        if kwargs.get("seat_order") is not None and isinstance(kwargs["seat_order"], list):
            state.rules.seat_order = [str(s) for s in kwargs["seat_order"] if str(s)]
        if kwargs.get("play_direction") is not None:
            direction = str(kwargs["play_direction"])
            state.rules.play_direction = direction if direction in {"clockwise", "counterclockwise"} else "clockwise"
        if kwargs.get("trick_size") is not None:
            state.rules.trick_size = max(1, int(kwargs["trick_size"]))
        if kwargs.get("rank_order_map") is not None and isinstance(kwargs["rank_order_map"], dict):
            state.rules.rank_order_map = dict(kwargs["rank_order_map"])
        state.state_version += 1
        return state

    def _transition_score(self, prev: CanonicalGameState, hand_cards: List[str], table_cards: List[dict]) -> tuple[float, list[str], str, str]:
        notes: list[str] = []
        score = 1.0
        reason_code = "ok"
        reason_detail = "accepted"
        prev_hand = set(prev.hand_cards)
        next_hand = set(hand_cards)
        prev_table = {item.card for item in prev.table_trick_cards}
        next_table = {item["card"] for item in table_cards}

        lifecycle = self._card_lifecycle.get(prev.game_id, {})
        won_pile_reappearance = [
            card
            for card in (next_hand | next_table)
            if str(lifecycle.get(card, {}).get("state", "")).startswith("WON_PILE")
        ]
        if won_pile_reappearance and not bool(settings.allow_won_pile_replay):
            score *= 0.2
            notes.append("Rejected/penalized won-pile card reappearance")
            reason_code = "won_pile_reappeared"
            reason_detail = "card from won pile appeared again"

        # legal: hand -> table on turn
        moved_to_table = [card for card in next_table if card in prev_hand and card not in prev_table]
        if moved_to_table and not prev.turn.to_play:
            score *= 0.6
            notes.append("Move to table observed without active turn context")
            reason_code = "no_turn_context"
            reason_detail = "table progressed without to_play seat"

        # invalid: table -> hand (likely CV noise)
        moved_back_to_hand = [card for card in next_hand if card in prev_table and card not in next_table]
        if moved_back_to_hand:
            score *= 0.3
            notes.append("Rejected/penalized table->hand transition candidates")
            reason_code = "table_to_hand_rejected"
            reason_detail = "table card appeared back in hand"

        # progression sanity: table should usually grow by at most one card per step
        if len(next_table) > len(prev_table) + 1:
            score *= 0.7
            notes.append("Table progression jump detected; lowered transition confidence")
            reason_code = "table_jump"
            reason_detail = "table grew by more than one card"

        return score, notes, reason_code, reason_detail

    def apply_fused_observation(self, fused: dict) -> tuple[CanonicalGameState, TransitionResult]:
        self._metrics["transitions_total"] += 1
        game_id = fused["game_id"]
        state = self.get(game_id)
        ts_ms = int(fused.get("ts_ms", state.updated_ts_ms))

        hand_cards = [item["label"] for item in fused.get("hand_cards", [])]
        table_cards_raw = fused.get("table_trick_cards", [])
        transition_score, transition_notes, reason_code, reason_detail = self._transition_score(state, hand_cards, table_cards_raw)

        # keep previous state if very weak transition support
        if transition_score < 0.45 and bool(settings.feature_strict_transition_reject):
            self._metrics["transitions_rejected"] += 1
            self._metrics["illegal_transition_rejects"] += 1
            reject_codes = self._metrics.setdefault("transition_reject_reason_code", {})
            reject_codes[reason_code] = int(reject_codes.get(reason_code, 0)) + 1
            uncertainty_notes = list(state.uncertainty.notes)
            uncertainty_notes.extend(transition_notes)
            state.uncertainty.notes = uncertainty_notes[:8]
            state.updated_ts_ms = ts_ms
            return state, TransitionResult(
                accepted=False,
                reason="low_legal_transition_score",
                reason_code=reason_code,
                reason_detail=reason_detail,
            )

        state.hand_cards = hand_cards[:13]
        state.table_trick_cards = [
            TableTrickCard(seat=item.get("seat", "unknown"), card=item.get("card", ""), confidence=float(item.get("confidence", 0.0)))
            for item in table_cards_raw
            if item.get("card")
        ]
        state.trick_cards_by_seat = {t.seat: t.card for t in state.table_trick_cards if t.seat and t.card}
        if state.table_trick_cards and state.trick_open_ts_ms <= 0:
            state.trick_open_ts_ms = ts_ms

        for card in state.hand_cards:
            self._mark_card_state(game_id, card, "IN_HAND", ts_ms)
        for item in state.table_trick_cards:
            self._mark_card_state(game_id, item.card, "PLAYED_ON_TABLE", ts_ms, seat=item.seat)

        # append history when table has cards and then clears (simple trick-close heuristic)
        if len(state.table_trick_cards) >= max(1, int(state.rules.trick_size or 4)):
            trick_no = (len(state.played_history) // 4) + 1
            for t in state.table_trick_cards:
                state.played_history.append(PlayedHistoryItem(trick_no=trick_no, seat=t.seat, card=t.card))

            lead = state.table_trick_cards[0].card[-1] if state.table_trick_cards and state.table_trick_cards[0].card else ""
            winner = self._resolve_trick_winner(
                state.table_trick_cards,
                lead,
                state.rules.trump_suit,
                state.rules.rank_order_map,
            )
            winner_team = "team_me" if winner in {"me", "across"} else "team_opp"
            state.won_piles[winner_team].extend([t.card for t in state.table_trick_cards])
            for t in state.table_trick_cards:
                self._mark_card_state(game_id, t.card, f"WON_PILE_{winner_team.upper()}", ts_ms, seat=t.seat)
            state.turn.leader = winner
            state.turn.to_play = winner
            state.trick_close_ts_ms = ts_ms
            state.table_trick_cards = []
            state.trick_cards_by_seat = {}
            state.turn.lead_suit = ""
            state.trick_id += 1
            state.trick_open_ts_ms = ts_ms

        # lead suit from first table card
        if state.table_trick_cards:
            c = state.table_trick_cards[0].card
            state.turn.lead_suit = c[-1] if c else ""
            expected_to_play = state.turn.leader
            for _ in range(len(state.table_trick_cards)):
                expected_to_play = self._next_player(state, expected_to_play)
            state.turn.to_play = expected_to_play

        unc = fused.get("uncertainty", {})
        notes = [n for n in unc.get("notes", []) if n]
        notes.extend(transition_notes)
        state.uncertainty = UncertaintyBlock(
            hand_conf=float(unc.get("hand_conf", 0.0)),
            table_conf=float(unc.get("table_conf", 0.0)),
            state_conf=float(unc.get("state_conf", 0.0)) * transition_score,
            notes=notes[:8],
        )
        state.updated_ts_ms = ts_ms
        state.state_version += 1
        state.last_transition_id = f"{game_id}:{state.state_version}:{state.updated_ts_ms}"
        return state, TransitionResult(accepted=True, reason="ok", reason_code="ok", reason_detail="accepted")

    def metrics(self) -> dict:
        total = max(int(self._metrics.get("transitions_total", 0)), 1)
        rejected = int(self._metrics.get("transitions_rejected", 0))
        accepted = max(total - rejected, 0)
        return {
            **self._metrics,
            "illegal_transition_reject_rate": round(rejected / total, 6),
            "transition_accept_rate": round(accepted / total, 6),
        }
