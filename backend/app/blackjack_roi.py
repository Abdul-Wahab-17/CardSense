"""ROI-based detection routing for Blackjack.

Solves two problems:

1. **Classification** — a single-camera frame contains both player cards and
   dealer cards. Given two normalized ROI rectangles (player_roi, dealer_roi),
   classify each detected bbox into one of {"player", "dealer", None}.

2. **Dedup / debounce** — the worker re-detects the same physical card every
   frame (10–30 fps). We must call the engine's `observe_card` exactly once
   per *new* physical card. We use position-based tracks: a detected card is
   "the same card we already counted" if its bbox overlaps a known track for
   the same rank and same target. Tracks expire after `STALE_MS`.

Pure functions + a small per-game tracker class. No I/O, no FastAPI deps —
unit-testable on its own.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from .blackjack import ROI

Target = Literal["player", "dealer"]


# ---------------- pure helpers ------------------------------------------------

def normalize_bbox(bbox: Tuple[float, float, float, float], frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
    """Convert pixel bbox (x1,y1,x2,y2) to normalized [0,1] coords."""
    if frame_w <= 0 or frame_h <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, min(1.0, x1 / frame_w)),
        max(0.0, min(1.0, y1 / frame_h)),
        max(0.0, min(1.0, x2 / frame_w)),
        max(0.0, min(1.0, y2 / frame_h)),
    )


def classify_by_roi(
    bbox_norm: Tuple[float, float, float, float],
    player_roi: Optional[ROI],
    dealer_roi: Optional[ROI],
) -> Optional[Target]:
    """Return 'player', 'dealer', or None based on the bbox CENTER position.

    Using center (rather than overlap) makes classification stable when a card
    straddles the boundary — it gets assigned to whichever side its center is in.
    """
    cx = (bbox_norm[0] + bbox_norm[2]) / 2.0
    cy = (bbox_norm[1] + bbox_norm[3]) / 2.0
    if player_roi is not None and player_roi.contains_point(cx, cy):
        return "player"
    if dealer_roi is not None and dealer_roi.contains_point(cx, cy):
        return "dealer"
    return None


def iou_norm(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Standard IoU on two normalized bboxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


# ---------------- per-game tracker --------------------------------------------

@dataclass
class _Track:
    rank: str
    target: Target
    last_bbox_norm: Tuple[float, float, float, float]
    last_seen_ms: int
    confirm_count: int = 0   # total observations at this position (any rank)
    emitted: bool = False    # already committed to engine state
    rank_votes: Dict[str, int] = field(default_factory=dict)


def _bump_vote(track: _Track, rank: str) -> None:
    track.rank_votes[rank] = track.rank_votes.get(rank, 0) + 1


def _winning_rank(track: _Track) -> Optional[str]:
    """Return the rank with the most votes, or None if no votes."""
    if not track.rank_votes:
        return None
    return max(track.rank_votes.items(), key=lambda kv: kv[1])[0]


class CardTracker:
    """Per-game position-based tracker that dedupes detections of the same card.

    Behaviour:
      - First detection of a (rank, target) at a position creates a new track.
      - Subsequent detections within IoU threshold update the existing track.
      - A track is "ready to emit" when confirm_count >= MIN_HITS.
      - Tracks expire (and are removed) after STALE_MS without an update.
      - `clear()` is called between rounds so cards from a previous round
        don't suppress same-rank cards in the next round.

    Thread-safe via internal lock.
    """

    IOU_MATCH = 0.20          # same-rank tracks merge if bboxes overlap by 20%
    SUPPRESS_IOU = 0.20       # cross-rank: ANY emitted card overlapping by 20% suppresses new tracks
    MIN_HITS = 2              # frames before emitting; cross-zone dedup handles phantoms now
    MIN_RANK_RATIO = 0.55     # winning rank must have >55% of votes among recent rank predictions
    STALE_MS = 4000           # drop a track if not seen for N ms

    def __init__(self) -> None:
        self._tracks: Dict[str, List[_Track]] = {}  # game_id -> tracks
        self._lock = threading.Lock()

    @staticmethod
    def _bbox_center(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    @classmethod
    def _close_position(cls, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
        """True if two bboxes likely refer to the same physical card across the
        dealer/player boundary. Uses center-distance with a generous threshold
        because a card's bbox center can shift several percent across frames."""
        ax, ay = cls._bbox_center(a)
        bx, by = cls._bbox_center(b)
        return abs(ax - bx) < 0.20 and abs(ay - by) < 0.20

    def feed(
        self,
        game_id: str,
        rank: str,
        target: Target,
        bbox_norm: Tuple[float, float, float, float],
        now_ms: Optional[int] = None,
    ) -> Optional[_Track]:
        """Record a single observation.

        Returns a track that just crossed the emission threshold (caller should
        now apply it to the engine), or None.

        Three-stage matching to handle the model wobbling between ranks for the
        same physical card:

          1. **Cross-rank suppression of emitted cards.** If the bbox overlaps
             with an ALREADY-EMITTED track on this target by SUPPRESS_IOU, treat
             this as a re-detection of the existing card. Refresh that track's
             last_seen and ignore the rank mismatch — the model is just wobbling.

          2. **Same-rank track update.** If the bbox overlaps with an existing
             unemitted track of the same rank/target, refresh and increment.

          3. **Cross-rank vote for unemitted tracks.** If the bbox overlaps
             with an unemitted track of a DIFFERENT rank, both ranks are voting
             for "a card here". Track all rank votes; emit only when one rank
             has MIN_HITS hits AND beats the alternatives by MIN_RANK_RATIO.

          4. **Otherwise** create a fresh track.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        with self._lock:
            tracks = self._tracks.setdefault(game_id, [])
            self._expire_inplace(tracks, now_ms)

            # ----- (0) Cross-zone same-rank suppression of emitted cards -----
            # A card with this same rank was already emitted, possibly on the
            # OTHER target if its bbox wobbled across the dealer/player
            # boundary. If positions are close, drop this as a duplicate.
            for t in tracks:
                if not t.emitted or t.rank != rank:
                    continue
                if self._close_position(t.last_bbox_norm, bbox_norm):
                    t.last_seen_ms = now_ms
                    return None

            # ----- (1) Same-target cross-rank suppression of emitted cards --
            # A card already committed to state at this position. Any rank
            # detected here now is just model noise; refresh and bail.
            for t in tracks:
                if t.target != target or not t.emitted:
                    continue
                if iou_norm(t.last_bbox_norm, bbox_norm) >= self.SUPPRESS_IOU:
                    t.last_seen_ms = now_ms
                    return None

            # ----- (2) Same-rank, same-target, unemitted track update -----
            best_same = -1
            best_iou_same = 0.0
            for i, t in enumerate(tracks):
                if t.target != target or t.rank != rank or t.emitted:
                    continue
                ov = iou_norm(t.last_bbox_norm, bbox_norm)
                if ov > best_iou_same:
                    best_iou_same = ov; best_same = i
            if best_same >= 0 and best_iou_same >= self.IOU_MATCH:
                t = tracks[best_same]
                t.last_bbox_norm = bbox_norm
                t.last_seen_ms = now_ms
                t.confirm_count += 1
                _bump_vote(t, rank)
                return self._maybe_emit(tracks, target, t)

            # ----- (3) Cross-rank, same-target overlap: vote at the position -----
            # Different rank but same physical position → wobble. Add a vote
            # to the existing track for THIS rank, but don't emit until the
            # winning rank has MIN_HITS and clear majority.
            best_other = -1
            best_iou_other = 0.0
            for i, t in enumerate(tracks):
                if t.target != target or t.emitted:
                    continue
                if t.rank == rank:
                    continue  # already handled in (2)
                ov = iou_norm(t.last_bbox_norm, bbox_norm)
                if ov > best_iou_other:
                    best_iou_other = ov; best_other = i
            if best_other >= 0 and best_iou_other >= self.IOU_MATCH:
                t = tracks[best_other]
                t.last_bbox_norm = bbox_norm  # track the position
                t.last_seen_ms = now_ms
                t.confirm_count += 1  # any vote counts toward MIN_HITS
                _bump_vote(t, rank)
                # Re-anchor the track to the dominant rank if it changed
                winning = _winning_rank(t)
                if winning and winning != t.rank:
                    t.rank = winning
                return self._maybe_emit(tracks, target, t)

            # ----- (3.5) Cross-zone same-rank merge for UNEMITTED tracks -----
            # The same physical card seen in both dealer and player zones
            # across frames. Merge into the existing track (whichever zone
            # has more votes wins).
            for t in tracks:
                if t.emitted or t.rank != rank or t.target == target:
                    continue
                if not self._close_position(t.last_bbox_norm, bbox_norm):
                    continue
                # Add this vote to the existing track. If the new target now
                # has more votes than the existing one's target, keep the new
                # target. Otherwise stick with the existing.
                t.last_seen_ms = now_ms
                t.confirm_count += 1
                _bump_vote(t, rank)
                # Note: we don't flip target — first zone wins. Could be
                # smarter (e.g., compare aggregated count per target), but
                # for the demo, stickiness is more important than nuance.
                return self._maybe_emit(tracks, target, t)

            # ----- (4) Brand new track -----
            new_t = _Track(
                rank=rank,
                target=target,
                last_bbox_norm=bbox_norm,
                last_seen_ms=now_ms,
                confirm_count=1,
                emitted=False,
            )
            _bump_vote(new_t, rank)
            tracks.append(new_t)
            return self._maybe_emit(tracks, target, new_t)

    def _maybe_emit(self, tracks: List["_Track"], target: Target, t: "_Track") -> Optional["_Track"]:
        """Emit if hit threshold + rank dominance reached."""
        if t.emitted:
            return None
        if t.confirm_count < self.MIN_HITS:
            return None
        winning = _winning_rank(t)
        if not winning:
            return None
        votes = getattr(t, "rank_votes", None) or {}
        total = sum(votes.values()) or 1
        ratio = votes.get(winning, 0) / total
        if ratio < self.MIN_RANK_RATIO:
            return None  # not enough consensus yet — keep waiting
        t.rank = winning
        t.emitted = True
        return t

    def emitted_summary(self, game_id: str) -> Dict[Target, List[str]]:
        """Return ranks already emitted per target, ordered by first sighting.

        Used by auto-deal logic to know whether we have 2+2 ready.
        """
        out: Dict[Target, List[str]] = {"player": [], "dealer": []}
        with self._lock:
            for t in self._tracks.get(game_id, []):
                if t.emitted:
                    out[t.target].append(t.rank)
        return out

    def clear(self, game_id: str) -> None:
        with self._lock:
            self._tracks.pop(game_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._tracks.clear()

    @classmethod
    def _expire_inplace(cls, tracks: List[_Track], now_ms: int) -> None:
        cutoff = now_ms - cls.STALE_MS
        if not tracks:
            return
        tracks[:] = [t for t in tracks if t.last_seen_ms >= cutoff]


# Module-level singleton (mirrors the engine pattern).
card_tracker = CardTracker()
