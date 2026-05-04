"""Per-game Blackjack state manager.

Holds BlackjackState for each active blackjack game_id, threading observations
from cameras (or manual UI) into the state machine in `blackjack.py`.

Designed to coexist with the existing trick-taking engines: a game is either
tagged 'blackjack' here or remains a trick-taking game in `match_engine.py`.
The two never overlap on the same game_id.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter, defaultdict, deque
from typing import Deque, Dict, List, Literal, Optional, Tuple

from . import blackjack as bj
from .blackjack import BlackjackConfig, BlackjackState, ROI, normalize_rank
from .blackjack_advice import Recommendation, recommend
from .blackjack_roi import CardTracker, Target, card_tracker, classify_by_roi

logger = logging.getLogger("cardsense.blackjack")

ObservationTarget = Literal["player", "dealer"]


class BlackjackEngine:
    """In-memory authoritative store of Blackjack games.

    Thread-safe via a single coarse lock — Blackjack state mutations are tiny
    so we don't need finer locking. Backed by no persistence in v1; rely on
    caller to log round outcomes to the existing event store if needed.
    """

    def __init__(self, tracker: Optional[CardTracker] = None) -> None:
        self._lock = threading.Lock()
        self._states: Dict[str, BlackjackState] = {}
        # Round-end snapshots per game, used by tipsi (post-round teach mode)
        # and by the UI history panel.
        self._history: Dict[str, List[dict]] = defaultdict(list)
        # Position-based dedup tracker (legacy path, kept for tests/manual feed_detection).
        self._tracker: CardTracker = tracker or card_tracker
        # Pre-deal accumulator (legacy path). New stateless flow uses _frame_snaps.
        self._pending_deal: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"player": [], "dealer": []})
        # Recent per-frame snapshots for stateless deal/hit detection.
        # Each entry: (ts_ms, Counter of (rank, target) tuples).
        self._frame_snaps: Dict[str, Deque[Tuple[int, Counter]]] = defaultdict(lambda: deque(maxlen=20))
        # Cards already applied this round, so we don't double-apply hits.
        # Reset on new_round / start_game.
        self._applied_in_round: Dict[str, Counter] = defaultdict(Counter)
        # Last raw frame's detections + classification verdict — for diagnostics.
        self._last_raw: Dict[str, dict] = {}

    # ---------------- lifecycle -----------------------------------------------

    def start_game(
        self,
        game_id: str,
        *,
        decks_in_shoe: int = 1,
        reshuffle_at_pct: float = 0.25,
        bet: float = 1.0,
    ) -> BlackjackState:
        """Initialize a new Blackjack game (or reset an existing one).

        Also wipes any stale tracker entries / pending-deal accumulator from
        a previous game with this id — otherwise phantom emissions from before
        leak into the new game's deal-initial.
        """
        with self._lock:
            state = BlackjackState(
                game_id=game_id,
                decks_in_shoe=decks_in_shoe,
                reshuffle_at_pct=reshuffle_at_pct,
                bet=bet,
            )
            self._states[game_id] = state
            self._history[game_id] = []
            self._pending_deal[game_id] = {"player": [], "dealer": []}
            self._frame_snaps[game_id] = deque(maxlen=20)
            self._applied_in_round[game_id] = Counter()
            logger.info("blackjack: started game %s (decks=%d)", game_id, decks_in_shoe)
        # Tracker has its own lock — clear outside our lock to avoid nesting.
        self._tracker.clear(game_id)
        # Wipe agent memory + change-detector cache for a clean start
        try:
            from .blackjack_agent import reset_agent_state
            reset_agent_state(game_id)
        except Exception:
            pass
        try:
            from .blackjack_change_detector import reset as cd_reset
            cd_reset(game_id)
        except Exception:
            pass
        return state

    def get(self, game_id: str) -> Optional[BlackjackState]:
        with self._lock:
            return self._states.get(game_id)

    def has(self, game_id: str) -> bool:
        with self._lock:
            return game_id in self._states

    def all_states(self) -> Dict[str, BlackjackState]:
        with self._lock:
            return {k: v.model_copy(deep=True) for k, v in self._states.items()}

    # ---------------- observations --------------------------------------------

    def deal_initial(self, game_id: str, player_cards: List[str], dealer_cards: List[str]) -> BlackjackState:
        """Deal the opening 2+2 cards. Cards are raw labels — normalized here."""
        p = self._normalize_list(player_cards)
        d = self._normalize_list(dealer_cards)
        with self._lock:
            state = self._require_state(game_id)
            state = bj.deal_initial(state, p, d)
            self._states[game_id] = state
            # Player blackjack auto-resolves during deal_initial; archive that
            # outcome so it shows in history without needing a follow-up call.
            self._maybe_archive_round(game_id, state)
            return state

    def observe_card(self, game_id: str, label: str, target: ObservationTarget) -> BlackjackState:
        """Smart helper: routes a single observed card into the appropriate
        state transition based on current phase.

        - During player's turn, target='player' means a hit.
        - target='dealer' (after reveal) means a dealer draw.
        - Before deal_initial: ignored (caller must use deal_initial directly).

        Returns the resulting state.
        """
        rank = normalize_rank(label)
        if rank is None:
            raise ValueError(f"unrecognized card label: {label!r}")

        with self._lock:
            state = self._require_state(game_id)

            # Pre-deal: refuse — caller should explicitly call deal_initial.
            if not state.player.cards or not state.dealer.cards:
                raise ValueError("game has not been dealt — call deal_initial first")

            if target == "player":
                if state.is_round_over() or state.player_done:
                    raise ValueError("cannot add player card: player turn finished")
                state = bj.player_hit(state, rank)
            elif target == "dealer":
                if not state.dealer_hole_revealed:
                    # If we're seeing a third dealer card, the hole must already
                    # be flipped — auto-reveal here for camera convenience.
                    state = bj.reveal_dealer_hole(state)
                if state.is_round_over():
                    # round was settled by reveal (player BJ etc.)
                    self._states[game_id] = state
                    return state
                state = bj.dealer_draw(state, rank)
            else:
                raise ValueError(f"unknown target: {target!r}")

            self._states[game_id] = state
            self._maybe_archive_round(game_id, state)
            return state

    def player_action(self, game_id: str, action: str, drawn_label: Optional[str] = None) -> BlackjackState:
        """Apply an explicit player action (manual UI control).

        - 'hit' requires drawn_label
        - 'stand' takes no card
        - 'double' requires drawn_label
        """
        with self._lock:
            state = self._require_state(game_id)
            action = action.lower().strip()

            if action == "hit":
                if not drawn_label:
                    raise ValueError("hit requires a drawn_label")
                rank = normalize_rank(drawn_label) or self._reject_label(drawn_label)
                state = bj.player_hit(state, rank)
            elif action == "stand":
                state = bj.player_stand(state)
            elif action == "double":
                if not drawn_label:
                    raise ValueError("double requires a drawn_label")
                rank = normalize_rank(drawn_label) or self._reject_label(drawn_label)
                state = bj.player_double(state, rank)
            else:
                raise ValueError(f"unknown action: {action!r}")

            self._states[game_id] = state
            self._maybe_archive_round(game_id, state)
            return state

    def reveal_dealer_hole(self, game_id: str) -> BlackjackState:
        with self._lock:
            state = self._require_state(game_id)
            state = bj.reveal_dealer_hole(state)
            self._states[game_id] = state
            self._maybe_archive_round(game_id, state)
            return state

    def reveal_dealer_hole_with_rank(self, game_id: str, hole_label: str) -> BlackjackState:
        """Reveal the dealer's hole card with the actual observed rank. Used
        in the camera flow where the hole was face-down at deal_initial time
        and the engine has only 1 dealer card. The rank is appended."""
        rank = normalize_rank(hole_label)
        if rank is None:
            raise ValueError(f"unrecognized card label: {hole_label!r}")
        with self._lock:
            state = self._require_state(game_id)
            if state.dealer_hole_revealed and len(state.dealer.cards) >= 2:
                # Already revealed with both cards committed — no-op.
                return state
            state = bj.reveal_dealer_hole(state, hole_rank=rank)
            self._states[game_id] = state
            self._maybe_archive_round(game_id, state)
            return state

    def new_round(self, game_id: str, *, reshuffle: Optional[bool] = None, bet: float = 1.0) -> BlackjackState:
        with self._lock:
            state = self._require_state(game_id)
            state = bj.new_round(state, reshuffle=reshuffle, bet=bet)
            self._states[game_id] = state
            self._tracker.clear(game_id)
            self._pending_deal[game_id] = {"player": [], "dealer": []}
            self._frame_snaps[game_id] = deque(maxlen=20)
            self._applied_in_round[game_id] = Counter()
        # Wipe agent + change-detector caches outside the lock
        try:
            from .blackjack_agent import reset_agent_state
            reset_agent_state(game_id)
        except Exception:
            pass
        try:
            from .blackjack_change_detector import reset as cd_reset
            cd_reset(game_id)
        except Exception:
            pass
        return state

    # ---------------- ROI / camera config -------------------------------------

    def set_roi(self, game_id: str, *, player_roi: Optional[ROI], dealer_roi: Optional[ROI], auto_deal: Optional[bool] = None) -> BlackjackState:
        """Configure ROI rectangles for camera-driven play."""
        with self._lock:
            state = self._require_state(game_id)
            cfg = state.config.model_copy(deep=True)
            cfg.player_roi = player_roi
            cfg.dealer_roi = dealer_roi
            if auto_deal is not None:
                cfg.auto_deal = bool(auto_deal)
            state.config = cfg
            self._states[game_id] = state
            return state

    # ---------------- stateless frame feed (NEW PRIMARY PATH) -----------------

    # Tunables for stateless detection.
    # Min bbox area to count. The model produces tiny bboxes (~0.3%) for the
    # rank-suit corner indicators — that's the SIGNAL we want, not noise. We
    # cluster these into per-card groups in a separate step.
    MIN_BBOX_AREA = 0.0015
    # Snapshots within this many ms participate in the stability check.
    STABILITY_WINDOW_MS = 1200
    # Min consecutive snapshots required (count-based, ~600ms at 5fps).
    MIN_STABLE_SNAPS = 3
    # Per-frame card clustering: detections within this center distance are
    # treated as belonging to the same physical card (top-left + bottom-right
    # corners + center face image). A real card spans ~25-35% of frame height,
    # so the corner-to-corner distance is ~25%. Slightly larger than that to
    # absorb perspective wobble.
    CARD_CLUSTER_DIST = 0.28
    # Cross-frame position clustering for the stability window.
    CROSS_FRAME_CLUSTER_DIST = 0.25

    def feed_frame(
        self,
        game_id: str,
        detections: List[Tuple[str, Tuple[float, float, float, float]]],
        *,
        now_ms: Optional[int] = None,
    ) -> Optional[dict]:
        """Stateless frame ingestion. Each frame is treated as the source of
        truth — no cross-frame voting/tracker accumulation.

        Pre-deal: when a 2+2 player/dealer arrangement is *stably visible* across
        a 1-1.5s window of frames, auto-fire deal_initial.

        Post-deal: when a NEW card (one not in the committed state) is *stably
        visible* in player or dealer ROI, apply player_hit / dealer_draw.

        Returns a "what we did" dict, or None.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        # ---- 1. Filter, classify, dedupe within this frame -------------------
        with self._lock:
            state = self._states.get(game_id)
            if state is None:
                return None
            cfg = state.config

        # ---- 1a. Filter raw detections (rank validity, tiny-bbox cutoff) ----
        valid: List[Tuple[str, Tuple[float, float, float, float]]] = []
        debug_raw: List[dict] = []
        for rank, bbox in detections:
            rank = (rank or "").upper()
            entry = {"rank": rank, "bbox": list(bbox)}
            if rank not in {"A","2","3","4","5","6","7","8","9","10","J","Q","K"}:
                entry["dropped"] = "bad_rank"; debug_raw.append(entry); continue
            w = max(0.0, bbox[2] - bbox[0])
            h = max(0.0, bbox[3] - bbox[1])
            entry["area"] = round(w * h, 4)
            if w * h < self.MIN_BBOX_AREA:
                entry["dropped"] = "tiny_bbox"; debug_raw.append(entry); continue
            entry["bbox_center"] = [round((bbox[0]+bbox[2])/2,3), round((bbox[1]+bbox[3])/2,3)]
            valid.append((rank, bbox))
            debug_raw.append(entry)

        # ---- 1b. Cluster valid detections into "cards" --------------------
        # Multiple detections can fire on the same physical card (top-left
        # corner indicator + bottom-right indicator + center face symbol).
        # Group them by spatial proximity; each cluster is one card.
        cards = self._cluster_detections_into_cards(valid)

        # ---- 1c. Classify each clustered CARD by ROI ----------------------
        by_zone: Dict[Target, List[Tuple[str, Tuple[float, float, float, float]]]] = {
            "player": [], "dealer": []
        }
        cards_debug: List[dict] = []
        for card in cards:
            cx, cy = card["center"]
            # Compose a synthetic bbox covering the cluster
            xs1 = min(b[0] for _, b in card["members"])
            ys1 = min(b[1] for _, b in card["members"])
            xs2 = max(b[2] for _, b in card["members"])
            ys2 = max(b[3] for _, b in card["members"])
            card_bbox = (xs1, ys1, xs2, ys2)
            target = classify_by_roi(card_bbox, cfg.player_roi, cfg.dealer_roi)
            ranks_in_cluster = Counter(r for r, _ in card["members"])
            winner = ranks_in_cluster.most_common(1)[0][0]
            cards_debug.append({
                "rank": winner, "members": len(card["members"]),
                "rank_votes": dict(ranks_in_cluster),
                "center": [round(cx, 3), round(cy, 3)],
                "target": target,
            })
            if target is None:
                continue
            by_zone[target].append((winner, card_bbox))

        # Save raw debug for /diagnostics
        if detections or debug_raw or cards_debug:
            self._last_raw[game_id] = {
                "ts_ms": now_ms,
                "detections_in": len(detections),
                "debug_raw": debug_raw,
                "clustered_cards": cards_debug,
                "after_filter_player": [r for r, _ in by_zone["player"]],
                "after_filter_dealer": [r for r, _ in by_zone["dealer"]],
            }

        # (Within-frame card clustering already handled in 1b; nothing to dedup.)

        # ---- 2. Build a snapshot of this frame -------------------------------
        # Snapshot stores the full per-zone detections (rank, bbox), not just
        # a Counter, so we can do cross-frame position clustering.
        snap = {"player": list(by_zone["player"]), "dealer": list(by_zone["dealer"])}

        with self._lock:
            self._frame_snaps[game_id].append((now_ms, snap))
            # Trim out-of-window snapshots
            cutoff = now_ms - self.STABILITY_WINDOW_MS
            while self._frame_snaps[game_id] and self._frame_snaps[game_id][0][0] < cutoff:
                self._frame_snaps[game_id].popleft()
            recent = list(self._frame_snaps[game_id])

        # ---- 3. Stability check (COUNT-based, majority-tolerant) ------------
        # Old behaviour required EXACT count agreement across all recent snaps;
        # one wobbly frame would block the deal. Now we take the MOST COMMON
        # (player_count, dealer_count) tuple across the window — must appear
        # in >50% of snaps.
        if len(recent) < self.MIN_STABLE_SNAPS:
            return None
        count_pairs = Counter(
            (len(s["player"]), len(s["dealer"])) for _, s in recent
        )
        (best_pc, best_dc), n_agreed = count_pairs.most_common(1)[0]
        if n_agreed * 2 <= len(recent):
            return None  # no majority — too unstable
        latest_player_count = best_pc
        latest_dealer_count = best_dc

        # ---- 4. Resolve ranks via per-position majority voting --------------
        stable_player = self._cluster_and_vote([s for _, s in recent], "player")
        stable_dealer = self._cluster_and_vote([s for _, s in recent], "dealer")

        # Sanity: cluster vote must produce the expected number of cards
        if sum(stable_player.values()) != latest_player_count or sum(stable_dealer.values()) != latest_dealer_count:
            return None

        # ---- 5. Act on the stable snapshot -----------------------------------
        with self._lock:
            state = self._states.get(game_id)
            if state is None:
                return None
            if state.is_round_over():
                return None

            # ---- 4a. PRE-DEAL: auto-deal when 2+2 stable ---------------------
            if not state.player.cards and not state.dealer.cards:
                if sum(stable_player.values()) == 2 and sum(stable_dealer.values()) == 2 and cfg.auto_deal:
                    p_cards = list(stable_player.elements())
                    d_cards = list(stable_dealer.elements())
                    try:
                        new_state = bj.deal_initial(state, p_cards, d_cards)
                        self._states[game_id] = new_state
                        self._maybe_archive_round(game_id, new_state)
                        # Reset applied-in-round counter to match what was just dealt.
                        applied = Counter(p_cards)
                        applied.update(d_cards[:1] if not new_state.dealer_hole_revealed else d_cards)
                        self._applied_in_round[game_id] = applied
                        return {"action": "deal_initial",
                                "player": p_cards, "dealer": d_cards}
                    except (ValueError, bj._RuleError) as exc:
                        logger.warning("auto-deal failed for %s: %s", game_id, exc)
                        return {"action": "deal_pending_error", "error": str(exc)}
                return {"action": "waiting_for_22",
                        "player_count": latest_player_count,
                        "dealer_count": latest_dealer_count}

            # ---- 4b. POST-DEAL: detect new cards via diff --------------------
            committed_player = Counter(state.player.cards)
            # Dealer's hole stays uncommitted to "applied" until it's revealed.
            if state.dealer_hole_revealed:
                committed_dealer = Counter(state.dealer.cards)
            else:
                committed_dealer = Counter(state.dealer.cards[:1])

            # New cards = whatever's currently visible in zone, minus what's
            # already in the committed state. Negative diffs (cards removed)
            # are ignored — state is monotonic during a round.
            new_player = stable_player - committed_player
            new_dealer = stable_dealer - committed_dealer

            actions: List[dict] = []
            # Apply new player cards as hits (player turn only)
            if not state.player_done:
                for rank in new_player.elements():
                    try:
                        new_state = bj.player_hit(state, rank)
                        state = new_state
                        self._states[game_id] = state
                        actions.append({"action": "player_hit", "rank": rank})
                    except (ValueError, bj._RuleError) as exc:
                        logger.warning("player_hit failed for %s rank=%s: %s", game_id, rank, exc)

            # Apply new dealer cards as draws (only after player is done +
            # dealer hole revealed)
            if state.player_done and not state.is_round_over():
                # If hole isn't revealed yet, reveal first
                if not state.dealer_hole_revealed:
                    state = bj.reveal_dealer_hole(state)
                    self._states[game_id] = state
                    if state.is_round_over():
                        self._maybe_archive_round(game_id, state)
                        actions.append({"action": "dealer_reveal", "outcome": state.round_outcome})
                # Apply new visible dealer cards as draws
                if not state.is_round_over():
                    # Re-diff after possible reveal
                    if state.dealer_hole_revealed:
                        committed_dealer = Counter(state.dealer.cards)
                    new_dealer = stable_dealer - committed_dealer
                    for rank in new_dealer.elements():
                        try:
                            state = bj.dealer_draw(state, rank)
                            self._states[game_id] = state
                            actions.append({"action": "dealer_draw", "rank": rank})
                        except (ValueError, bj._RuleError) as exc:
                            logger.warning("dealer_draw failed for %s rank=%s: %s", game_id, rank, exc)

            self._maybe_archive_round(game_id, state)

            # Update applied counter
            self._applied_in_round[game_id] = stable_player + (
                Counter(state.dealer.cards) if state.dealer_hole_revealed else Counter(state.dealer.cards[:1])
            )

            if actions:
                return {"actions": actions}
            return None

    @classmethod
    def _cluster_detections_into_cards(
        cls,
        valid: List[Tuple[str, Tuple[float, float, float, float]]],
    ) -> List[dict]:
        """Group raw detections into per-physical-card clusters.

        The model produces multiple bboxes per physical card (the rank-suit
        indicator at top-left + bottom-right + center face elements). All of
        these should be ONE card. Greedy-cluster by center distance; cluster
        radius ~ card size in frame.
        """
        clusters: List[dict] = []
        for rank, bbox in valid:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            placed = False
            for c in clusters:
                if abs(cx - c["center"][0]) < cls.CARD_CLUSTER_DIST and abs(cy - c["center"][1]) < cls.CARD_CLUSTER_DIST:
                    c["members"].append((rank, bbox))
                    n = len(c["members"])
                    # Update centroid as running mean
                    c["center"] = (
                        ((c["center"][0] * (n - 1)) + cx) / n,
                        ((c["center"][1] * (n - 1)) + cy) / n,
                    )
                    placed = True
                    break
            if not placed:
                clusters.append({"center": (cx, cy), "members": [(rank, bbox)]})
        return clusters

    @classmethod
    def _cluster_and_vote(
        cls,
        snapshots: List[Dict[str, List[Tuple[str, Tuple[float, float, float, float]]]]],
        zone: str,
    ) -> Counter:
        """Cluster all detections from `snapshots[*][zone]` by spatial position
        and return a Counter mapping rank -> count, where each cluster
        contributes ONE rank (the most common rank seen in that cluster).

        This makes the stable result tolerant of model wobble: if a card is
        detected as "7" in 3 frames and "8" in 1 frame, the cluster votes "7".
        """
        # Flatten all detections with their snapshot index
        all_dets: List[Tuple[str, Tuple[float, float, float, float]]] = []
        for snap in snapshots:
            for rank, bbox in snap.get(zone, []):
                all_dets.append((rank, bbox))
        if not all_dets:
            return Counter()

        # Greedy clustering by center distance
        clusters: List[List[Tuple[str, Tuple[float, float, float, float]]]] = []
        for rank, bbox in all_dets:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            placed = False
            for c in clusters:
                # Cluster's centroid (avg)
                ccx = sum((b[0] + b[2]) / 2.0 for _, b in c) / len(c)
                ccy = sum((b[1] + b[3]) / 2.0 for _, b in c) / len(c)
                if abs(cx - ccx) < cls.CROSS_FRAME_CLUSTER_DIST and abs(cy - ccy) < cls.CROSS_FRAME_CLUSTER_DIST:
                    c.append((rank, bbox))
                    placed = True
                    break
            if not placed:
                clusters.append([(rank, bbox)])

        # We expect roughly N clusters where N = max cards seen in zone
        # across snapshots. If we have way more clusters, something is wrong
        # (e.g., card moved a lot). Take the top N by population.
        max_zone_count = max((len(s.get(zone, [])) for s in snapshots), default=0)
        clusters.sort(key=len, reverse=True)
        clusters = clusters[: max(max_zone_count, 1)]

        # Each cluster votes with its majority rank
        out: Counter = Counter()
        for c in clusters:
            ranks_in_cluster = Counter(rank for rank, _ in c)
            winner = ranks_in_cluster.most_common(1)[0][0]
            out[winner] += 1
        return out

    @classmethod
    def _merge_close_in_frame(
        cls,
        items: List[Tuple[str, Tuple[float, float, float, float]]],
    ) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        """If multiple detections in one frame are at very close positions
        (regardless of rank — could be the model wobbling on the same card),
        keep only the largest one."""
        if len(items) <= 1:
            return items
        kept: List[Tuple[str, Tuple[float, float, float, float]]] = []
        for rank, bbox in items:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
            absorbed = False
            for i, (kr, kb) in enumerate(kept):
                kcx = (kb[0] + kb[2]) / 2.0
                kcy = (kb[1] + kb[3]) / 2.0
                karea = max(0.0, kb[2] - kb[0]) * max(0.0, kb[3] - kb[1])
                if abs(cx - kcx) < cls.WITHIN_FRAME_MERGE_DIST and abs(cy - kcy) < cls.WITHIN_FRAME_MERGE_DIST:
                    # Same physical card seen twice — keep the bigger bbox
                    if area > karea:
                        kept[i] = (rank, bbox)
                    absorbed = True
                    break
            if not absorbed:
                kept.append((rank, bbox))
        return kept

    # ---------------- detection feed (auto-route by ROI) ----------------------

    def feed_detection(
        self,
        game_id: str,
        rank: str,
        bbox_norm: tuple,
        *,
        now_ms: Optional[int] = None,
    ) -> Optional[dict]:
        """Feed a single detection event from the worker into the blackjack pipeline.

        Returns a small "what we did" dict for logging, or None if the detection
        was outside any ROI / suppressed as duplicate / no-op.

        - Classifies bbox by ROI.
        - Debounces via CardTracker.
        - When player and dealer pre-deal buffers each have 2 ranks AND
          state.config.auto_deal, auto-fires deal_initial.
        - Otherwise routes to observe_card.
        """
        rank = (rank or "").upper()
        if rank == "":
            return None
        if rank not in {"A","2","3","4","5","6","7","8","9","10","J","Q","K"}:
            return None

        # Filter out tiny bboxes: cards in the demo setup typically occupy
        # 5-30% of frame area. Detections below ~3% area are almost always
        # corner-of-card phantoms (the rank+suit indicator getting detected
        # as a separate card) or suit-pip false positives. Drop them.
        bbox_w = max(0.0, bbox_norm[2] - bbox_norm[0])
        bbox_h = max(0.0, bbox_norm[3] - bbox_norm[1])
        if bbox_w * bbox_h < 0.03:
            return None

        with self._lock:
            state = self._states.get(game_id)
            if state is None:
                return None
            cfg = state.config

            target: Optional[Target] = classify_by_roi(bbox_norm, cfg.player_roi, cfg.dealer_roi)
            if target is None:
                return None

            # Bypass tracker if state already settled — nothing useful to do.
            if state.is_round_over():
                return None

            # Outside the lock for this single tracker call (the tracker has its own lock).
        # ^ split lock segments so we don't nest engine.lock + tracker.lock.

        track_to_emit = self._tracker.feed(game_id, rank, target, bbox_norm, now_ms=now_ms)
        if track_to_emit is None:
            return None  # not yet confirmed or already emitted

        # Now apply the observation into game state.
        with self._lock:
            state = self._states.get(game_id)
            if state is None:
                return None

            # Pre-deal phase: accumulate, fire deal_initial when ready
            if not state.player.cards and not state.dealer.cards:
                pending = self._pending_deal[game_id]
                pending[target].append(rank)
                # Avoid going over 2 (we cap & ignore extras until deal fires)
                pending["player"] = pending["player"][:2]
                pending["dealer"] = pending["dealer"][:2]
                if cfg.auto_deal and len(pending["player"]) == 2 and len(pending["dealer"]) == 2:
                    try:
                        new_state = bj.deal_initial(state, pending["player"], pending["dealer"])
                        self._states[game_id] = new_state
                        self._maybe_archive_round(game_id, new_state)
                        self._pending_deal[game_id] = {"player": [], "dealer": []}
                        return {"action": "deal_initial", "rank": rank, "target": target}
                    except (ValueError, bj._RuleError) as exc:
                        logger.warning("auto-deal failed for %s: %s", game_id, exc)
                        return {"action": "deal_pending_error", "rank": rank, "target": target, "error": str(exc)}
                return {"action": "deal_pending", "rank": rank, "target": target}

            # In-round phase: route as hit / dealer-draw via observe_card semantics.
            try:
                if target == "player":
                    if state.is_round_over() or state.player_done:
                        return None
                    new_state = bj.player_hit(state, rank)
                else:
                    if not state.dealer_hole_revealed:
                        new_state = bj.reveal_dealer_hole(state)
                        if new_state.is_round_over():
                            self._states[game_id] = new_state
                            self._maybe_archive_round(game_id, new_state)
                            return {"action": "dealer_reveal", "rank": rank, "target": target}
                        state = new_state
                    new_state = bj.dealer_draw(state, rank)
                self._states[game_id] = new_state
                self._maybe_archive_round(game_id, new_state)
                return {"action": "observe", "rank": rank, "target": target}
            except (ValueError, bj._RuleError) as exc:
                logger.warning("feed_detection failed for %s (%s -> %s): %s", game_id, rank, target, exc)
                return {"action": "error", "rank": rank, "target": target, "error": str(exc)}

    # ---------------- recommendation + history --------------------------------

    def recommend(self, game_id: str) -> Recommendation:
        with self._lock:
            state = self._require_state(game_id)
            # recommend() copies state internally; safe to call with lock held
            return recommend(state)

    def history(self, game_id: str, limit: int = 50) -> List[dict]:
        with self._lock:
            rows = self._history.get(game_id, [])
            return list(rows[-limit:])

    def diagnostics(self, game_id: str) -> dict:
        """Return what the engine currently sees: recent snapshots, ROI config,
        applied-this-round counter. Use to debug 'why is it not committing'."""
        with self._lock:
            snaps = list(self._frame_snaps.get(game_id, []))
            applied = dict(self._applied_in_round.get(game_id, Counter()))
            pending = dict(self._pending_deal.get(game_id, {"player": [], "dealer": []}))
            state = self._states.get(game_id)
        snap_summary = []
        for ts, snap in snaps:
            snap_summary.append({
                "ts_ms": ts,
                "player_count": len(snap.get("player", [])),
                "dealer_count": len(snap.get("dealer", [])),
                "player_ranks": [r for r, _ in snap.get("player", [])],
                "dealer_ranks": [r for r, _ in snap.get("dealer", [])],
            })
        with self._lock:
            last_raw = self._last_raw.get(game_id)
        return {
            "game_id": game_id,
            "snapshot_window_size": len(snaps),
            "snapshots": snap_summary,
            "last_raw_frame": last_raw,
            "min_stable_snaps": self.MIN_STABLE_SNAPS,
            "stability_window_ms": self.STABILITY_WINDOW_MS,
            "min_bbox_area": self.MIN_BBOX_AREA,
            "applied_in_round": applied,
            "pending_deal": pending,
            "rois": {
                "player": state.config.player_roi.model_dump() if state and state.config.player_roi else None,
                "dealer": state.config.dealer_roi.model_dump() if state and state.config.dealer_roi else None,
            } if state else None,
        }

    # ---------------- internals -----------------------------------------------

    def _require_state(self, game_id: str) -> BlackjackState:
        state = self._states.get(game_id)
        if state is None:
            raise KeyError(f"no blackjack game {game_id!r} (call start_game first)")
        return state

    def _normalize_list(self, labels: List[str]) -> List[str]:
        out = []
        for label in labels:
            r = normalize_rank(label)
            if r is None:
                raise ValueError(f"unrecognized card label: {label!r}")
            out.append(r)
        return out

    @staticmethod
    def _reject_label(label: str) -> str:
        raise ValueError(f"unrecognized card label: {label!r}")

    def _maybe_archive_round(self, game_id: str, state: BlackjackState) -> None:
        """If round just finished, snapshot it for history + tipsi."""
        if not state.is_round_over():
            return
        snap = {
            "ts_ms": int(time.time() * 1000),
            "player_cards": list(state.player.cards),
            "player_total": state.player.total()[0],
            "dealer_cards": list(state.dealer.cards),
            "dealer_total": state.dealer.total()[0],
            "outcome": state.round_outcome,
            "doubled": state.player_doubled,
            "bet": state.bet,
            "payout": bj.payout_for(state),
            "running_count_after": sum(1 if r in ("2","3","4","5","6") else (-1 if r in ("10","J","Q","K","A") else 0) for r in state.seen_cards),
            "cards_remaining": state.cards_remaining(),
            "hands_played": state.hands_played,
        }
        # Avoid double-archiving if observe_card is called multiple times after settlement
        prev = self._history[game_id]
        if prev and prev[-1].get("hands_played") == snap["hands_played"]:
            prev[-1] = snap  # update in place
        else:
            prev.append(snap)


# Module-level singleton, mirrors `game_state_engine` / `match_engine` pattern.
blackjack_engine = BlackjackEngine()
