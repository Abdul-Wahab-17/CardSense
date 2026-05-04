"""Blackjack rules engine.

Pure-Python deterministic core for single-player vs dealer Blackjack:
    - Hand evaluation (hard / soft totals, bust, blackjack)
    - Player actions: hit, stand, double
    - Dealer policy: stand on 17 (S17) — hits everything below
    - Settlement: win, lose, push, blackjack 3:2
    - Shoe / discard tracking for downstream card-counting

State is intentionally serializable so it can be persisted to Redis or the
event store by the rest of the backend without translation.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .thulla import parse_card

# ----- card -> blackjack value ------------------------------------------------

# Aces are scored optimistically (11) and downgraded to 1 if the hand busts.
_RANK_VALUE = {
    "A": 11,
    "K": 10, "Q": 10, "J": 10, "10": 10,
    "9": 9, "8": 8, "7": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2,
}

VALID_RANKS = tuple(_RANK_VALUE.keys())


def normalize_rank(label: str) -> Optional[str]:
    """Accept any common card label form and return just the rank ('A', '10', 'K', etc).

    Examples
    --------
    >>> normalize_rank("AS")        # ace of spades
    'A'
    >>> normalize_rank("10♥")
    '10'
    >>> normalize_rank("queen_of_diamonds")
    'Q'
    >>> normalize_rank("garbage")
    None
    """
    if label is None:
        return None
    parsed = parse_card(str(label))
    if parsed is None:
        # last-ditch: maybe it was already a bare rank like "A" or "10"
        bare = str(label).strip().upper()
        if bare in _RANK_VALUE:
            return bare
        return None
    rank, _suit, _canonical = parsed
    return rank if rank in _RANK_VALUE else None


# ----- pydantic state ---------------------------------------------------------

PlayerAction = Literal["hit", "stand", "double"]
RoundOutcome = Literal[
    "in_progress",
    "player_blackjack",
    "player_bust",
    "player_stand",
    "dealer_bust",
    "dealer_stand",
    "win",
    "lose",
    "push",
]


class Hand(BaseModel):
    """A blackjack hand. Stores ranks only (suit irrelevant)."""
    cards: List[str] = Field(default_factory=list)

    def total(self) -> Tuple[int, bool]:
        """Return (best_total, is_soft).

        is_soft = True when at least one Ace is still being counted as 11.
        """
        total = sum(_RANK_VALUE[c] for c in self.cards)
        aces_high = self.cards.count("A")
        # downgrade aces from 11 -> 1 while busting
        while total > 21 and aces_high > 0:
            total -= 10
            aces_high -= 1
        return total, aces_high > 0

    def is_blackjack(self) -> bool:
        """Natural 21 on the first two cards."""
        if len(self.cards) != 2:
            return False
        total, _ = self.total()
        return total == 21

    def is_bust(self) -> bool:
        total, _ = self.total()
        return total > 21


class ROI(BaseModel):
    """Normalized rectangular region-of-interest in [0,1] frame coordinates.

    Coordinates are fractional so they're resolution-independent (works on
    any camera, any orientation). Origin is top-left.
    """
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    def contains_point(self, nx: float, ny: float) -> bool:
        return self.x1 <= nx <= self.x2 and self.y1 <= ny <= self.y2


class BlackjackConfig(BaseModel):
    """ROI + camera config for a blackjack game."""
    player_roi: Optional[ROI] = None
    dealer_roi: Optional[ROI] = None
    # When True: the engine auto-fires deal_initial as soon as 2 player + 2 dealer
    # cards are stable in their ROIs. When False: frontend drives deal explicitly.
    auto_deal: bool = True


class BlackjackState(BaseModel):
    """Authoritative state for a single Blackjack round + persistent shoe state.

    `seen_cards` carries every rank that's been observed across hands in the
    current shoe — that's the input for card-counting in `blackjack_advice.py`.
    """
    game_id: str = "default-game"
    decks_in_shoe: int = Field(default=1, ge=1, le=8)
    reshuffle_at_pct: float = Field(default=0.25, ge=0.05, le=1.0)

    player: Hand = Field(default_factory=Hand)
    dealer: Hand = Field(default_factory=Hand)
    dealer_hole_revealed: bool = False
    player_doubled: bool = False
    player_done: bool = False  # stand / bust / 21 / doubled

    seen_cards: List[str] = Field(default_factory=list)
    hands_played: int = 0
    round_outcome: RoundOutcome = "in_progress"
    bet: float = 1.0

    config: BlackjackConfig = Field(default_factory=BlackjackConfig)

    # ----- derived helpers ----------------------------------------------------

    def deck_total(self) -> int:
        return 52 * self.decks_in_shoe

    def cards_remaining(self) -> int:
        return max(0, self.deck_total() - len(self.seen_cards))

    def needs_reshuffle(self) -> bool:
        if self.deck_total() == 0:
            return False
        return self.cards_remaining() / self.deck_total() <= self.reshuffle_at_pct

    def is_round_over(self) -> bool:
        # Only terminal outcomes are "over". Intermediate states (player_stand,
        # dealer_stand) are still in flux waiting on the next observed card.
        return self.round_outcome in (
            "win", "lose", "push", "player_blackjack",
        )

    def is_player_turn(self) -> bool:
        return self.round_outcome == "in_progress" and not self.player_done


# ----- actions ----------------------------------------------------------------

class _RuleError(ValueError):
    pass


def _record_seen(state: BlackjackState, rank: str) -> None:
    state.seen_cards.append(rank)


def deal_initial(state: BlackjackState, player_cards: List[str], dealer_cards: List[str]) -> BlackjackState:
    """Start a new round with the given dealt cards.

    `player_cards` should be 2 ranks. `dealer_cards` should be 1 or 2 ranks:
    - 2 cards: both cards known (upcard first, hole-card second). Test paths /
      manual mode use this.
    - 1 card: only the upcard is known. The agent will supply the hole later
      via `reveal_dealer_hole(state, hole_rank=...)`. This is the camera flow:
      the hole card is physically face-down on the table.

    Cards are taken at face-value as already-parsed ranks.
    """
    if len(player_cards) != 2:
        raise _RuleError(f"player must be dealt exactly 2 cards, got {len(player_cards)}")
    if len(dealer_cards) not in (1, 2):
        raise _RuleError(f"dealer must be dealt 1 or 2 cards, got {len(dealer_cards)}")
    for c in player_cards + dealer_cards:
        if c not in _RANK_VALUE:
            raise _RuleError(f"invalid rank: {c!r}")

    state.player = Hand(cards=list(player_cards))
    state.dealer = Hand(cards=list(dealer_cards))
    state.dealer_hole_revealed = False
    state.player_doubled = False
    state.player_done = False
    state.round_outcome = "in_progress"

    # Reveal: player's two + dealer's upcard. Hole card (if dealt) stays hidden
    # in seen_cards until reveal_dealer_hole runs.
    _record_seen(state, player_cards[0])
    _record_seen(state, player_cards[1])
    _record_seen(state, dealer_cards[0])

    if state.player.is_blackjack():
        # Auto-resolve naturals: reveal hole to settle.
        return reveal_dealer_hole(state)
    return state


def player_hit(state: BlackjackState, drawn: str) -> BlackjackState:
    if state.player_done or state.is_round_over():
        raise _RuleError("player has already finished their turn")
    if drawn not in _RANK_VALUE:
        raise _RuleError(f"invalid rank: {drawn!r}")
    state.player.cards.append(drawn)
    _record_seen(state, drawn)
    if state.player.is_bust():
        state.player_done = True
        state.round_outcome = "player_bust"
        # round resolved on bust; dealer doesn't play
        return _settle(state)
    total, _ = state.player.total()
    if total == 21:
        # auto-stand on 21 (no point hitting)
        return player_stand(state)
    return state


def player_stand(state: BlackjackState) -> BlackjackState:
    if state.player_done or state.is_round_over():
        raise _RuleError("player has already finished their turn")
    state.player_done = True
    state.round_outcome = "player_stand"
    # Only auto-reveal if the engine already has the hole card. In the camera
    # flow the hole is face-down on the table at deal time, so dealer.cards
    # has only 1 entry — defer the reveal until the agent observes the flip
    # and supplies the actual rank via reveal_dealer_hole(hole_rank=...).
    if len(state.dealer.cards) >= 2:
        return reveal_dealer_hole(state)
    return state


def player_double(state: BlackjackState, drawn: str) -> BlackjackState:
    """Double: bet doubles, exactly one card drawn, then stand."""
    if state.player_done or state.is_round_over():
        raise _RuleError("player has already finished their turn")
    if len(state.player.cards) != 2:
        raise _RuleError("can only double on first action")
    if drawn not in _RANK_VALUE:
        raise _RuleError(f"invalid rank: {drawn!r}")
    state.player.cards.append(drawn)
    _record_seen(state, drawn)
    state.player_doubled = True
    state.bet *= 2
    state.player_done = True
    if state.player.is_bust():
        state.round_outcome = "player_bust"
        return _settle(state)
    state.round_outcome = "player_stand"
    # Only auto-reveal if the hole is already in the engine.
    if len(state.dealer.cards) >= 2:
        return reveal_dealer_hole(state)
    return state


def reveal_dealer_hole(state: BlackjackState, hole_rank: Optional[str] = None) -> BlackjackState:
    """Reveal the dealer's hole card and play out the dealer's hand per S17 policy.

    `hole_rank` is supplied when the engine has only 1 dealer card (camera flow
    where the hole was face-down at deal time). In that case we APPEND the
    revealed rank to dealer.cards. If 2 cards are already present, hole_rank
    is ignored (the hole was supplied at deal_initial time).

    Decision tree after reveal:
      1. Player blackjack → settle now.
      2. Player bust → settle now.
      3. Dealer must draw (total < 17) → leave in 'player_stand' waiting on dealer_draw.
      4. Dealer stands (total >= 17) → settle.
    """
    # Append the actual hole rank if needed
    if not state.dealer_hole_revealed and len(state.dealer.cards) < 2:
        if hole_rank is None:
            # Caller wants to reveal but engine doesn't know the hole yet.
            # Just mark revealed and bail — wait for explicit reveal with rank.
            state.dealer_hole_revealed = True
            return state
        if hole_rank not in _RANK_VALUE:
            raise _RuleError(f"invalid hole rank: {hole_rank!r}")
        state.dealer.cards.append(hole_rank)

    if not state.dealer_hole_revealed:
        state.dealer_hole_revealed = True
        if len(state.dealer.cards) >= 2:
            _record_seen(state, state.dealer.cards[1])

    if state.player.is_blackjack():
        return _settle(state)
    if state.player.is_bust():
        return _settle(state)
    return _play_dealer_to_completion(state)


def _play_dealer_to_completion(state: BlackjackState) -> BlackjackState:
    """Dealer follows S17: stand on hard or soft 17.

    Exposes via `dealer_draw()` for camera-observed external draws. This
    function only checks the *current* dealer total to see whether the dealer's
    turn is already done.
    """
    total, _ = state.dealer.total()
    if state.dealer.is_bust():
        state.round_outcome = "dealer_bust"
        return _settle(state)
    if total >= 17:
        state.round_outcome = "dealer_stand"
        return _settle(state)
    # Dealer must draw — round remains in flux ('player_stand') waiting for
    # the next observed card via dealer_draw().
    return state


def dealer_draw(state: BlackjackState, drawn: str) -> BlackjackState:
    """Add a card to the dealer's hand (camera-observed dealer draw)."""
    if drawn not in _RANK_VALUE:
        raise _RuleError(f"invalid rank: {drawn!r}")
    if not state.dealer_hole_revealed:
        raise _RuleError("dealer must reveal hole card before drawing")
    if state.is_round_over():
        raise _RuleError(f"round is already over ({state.round_outcome})")
    state.dealer.cards.append(drawn)
    _record_seen(state, drawn)
    return _play_dealer_to_completion(state)


# ----- settlement -------------------------------------------------------------

def _settle(state: BlackjackState) -> BlackjackState:
    """Translate the dealer/player end-states into final round_outcome ∈ {win, lose, push}."""
    p_total, _ = state.player.total()
    d_total, _ = state.dealer.total()
    p_bj = state.player.is_blackjack()
    d_bj = state.dealer.is_blackjack()

    # Naturals
    if p_bj and d_bj:
        state.round_outcome = "push"
        state.hands_played += 1
        return state
    if p_bj and state.dealer_hole_revealed and not d_bj:
        state.round_outcome = "player_blackjack"
        # bet payout flagged by callers via outcome
        state.hands_played += 1
        return state
    if p_bj and not state.dealer_hole_revealed:
        # waiting on dealer reveal — not settled yet
        state.round_outcome = "in_progress"
        return state

    # Bust cases
    if state.player.is_bust():
        state.round_outcome = "lose"
        state.hands_played += 1
        return state
    if state.dealer.is_bust():
        state.round_outcome = "win"
        state.hands_played += 1
        return state

    # Compare standing totals
    if not state.dealer_hole_revealed:
        # not enough info to settle yet
        return state
    if p_total > d_total:
        state.round_outcome = "win"
    elif p_total < d_total:
        state.round_outcome = "lose"
    else:
        state.round_outcome = "push"
    state.hands_played += 1
    return state


# ----- new round + reshuffle --------------------------------------------------

def new_round(state: BlackjackState, *, reshuffle: Optional[bool] = None, bet: float = 1.0) -> BlackjackState:
    """Reset hand state for a new round. If reshuffle is None, decides based on penetration.

    `seen_cards` is preserved across rounds so the count keeps running. It's
    cleared on reshuffle.
    """
    do_reshuffle = state.needs_reshuffle() if reshuffle is None else bool(reshuffle)
    if do_reshuffle:
        state.seen_cards = []
    state.player = Hand()
    state.dealer = Hand()
    state.dealer_hole_revealed = False
    state.player_doubled = False
    state.player_done = False
    state.round_outcome = "in_progress"
    state.bet = bet
    return state


def payout_for(state: BlackjackState) -> float:
    """Return net payout on this round (positive = player wins, negative = player loses).

    Blackjack pays 3:2, double-down doubles all gains/losses, push returns 0.
    """
    if state.round_outcome == "win":
        return state.bet
    if state.round_outcome == "lose" or state.round_outcome == "player_bust":
        return -state.bet
    if state.round_outcome == "player_blackjack":
        # Blackjack pays 3:2 of the original (pre-double) bet. We don't allow
        # double on a natural so bet is the original.
        return state.bet * 1.5
    if state.round_outcome == "push":
        return 0.0
    return 0.0  # in_progress or terminal-but-undecided
