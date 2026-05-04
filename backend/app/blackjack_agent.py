"""Blackjack agent: Claude-with-tools that owns the camera→state pipeline.

Each call to `process_frame()` does one of:
    1. Skip — frame hasn't changed enough to be worth examining (cheap pHash diff
       against the last classified frame).
    2. Call Claude Sonnet 4.6 vision with the frame + current engine state +
       a tool schema. Claude returns 0+ tool calls. We dispatch each call
       directly into the BlackjackEngine. Engine state IS the source of truth.
    3. On error / no relevant tool calls: state stays unchanged.

The agent does NOT carry its own copy of the game. It re-reads engine state on
every call. Tool calls are the only way state gets mutated.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from . import blackjack as bj
from .blackjack_engine import blackjack_engine
from .config import settings

logger = logging.getLogger("cardsense.blackjack.agent")


# ============================================================================
# Tool schema (OpenAI/OpenRouter format — works with Claude via OpenRouter)
# ============================================================================

TOOL_SCHEMA: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "deal_initial",
            "description": (
                "Deal the opening cards for a fresh round. "
                "Use ONLY when: (a) the engine state shows no committed cards, "
                "AND (b) the image clearly shows 2 player cards (bottom half) "
                "AND 2 dealer cards (top half — one face-up, one face-down OR "
                "both face-up). Pass face-up cards by rank+suit; for the "
                "dealer's hole card, pass 'HOLE' if face-down, or its rank+suit "
                "if visible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_cards": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exactly 2 cards (e.g. ['AS','KH']).",
                    },
                    "dealer_cards": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exactly 2 entries: dealer upcard first, then 'HOLE' if face-down or the rank+suit if visible.",
                    },
                },
                "required": ["player_cards", "dealer_cards"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "player_hit",
            "description": "Add a new card to the player's hand. Use when player_cards in state has FEWER cards than the player's hand visible in the image, AND state.player_done is false.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rank": {"type": "string", "description": "The new card's rank+suit (e.g. '5C')."},
                },
                "required": ["rank"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "player_stand",
            "description": "Player has finished taking cards (no more visible additions to player area). Trigger dealer reveal phase.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dealer_reveal",
            "description": "Reveal the dealer's previously-hidden hole card. Use when player_done is true and the image now shows BOTH dealer cards face-up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hole_rank": {"type": "string", "description": "Rank+suit of the hole card now visible (e.g. '7D')."},
                },
                "required": ["hole_rank"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dealer_draw",
            "description": "Dealer drew a new card. Use only after dealer_hole_revealed is true and dealer total in state is < 17.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rank": {"type": "string", "description": "New dealer card rank+suit."},
                },
                "required": ["rank"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "new_round",
            "description": "Clear the table and start a new round. Use when the previous round is over (round_outcome ≠ in_progress) AND the image shows no cards or a freshly cleared table.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "no_change",
            "description": "Nothing on the table has changed since the state was last updated. Use this if the cards visible in the image already match the engine state. Always preferable to calling a tool with wrong arguments.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


SYSTEM_PROMPT = """You are an automated Blackjack vision agent.

You will receive: (1) UP TO 2 overhead camera images in temporal order — the
PREVIOUS classified frame (if any) and the CURRENT frame (always last),
(2) the current engine state as JSON, and (3) a list of your recent tool
calls. The TOP HALF of each image is the dealer's area; the BOTTOM HALF is
the player's area.

Use the temporal context: a card visible in the current frame but not in
the previous frame was JUST placed. A card present in earlier frames that's
gone from the current is being moved/removed — do nothing, state is
monotonic mid-round. If a card is blurry in one frame, cross-check with the
other. If the two frames disagree on what a card is, call no_change.

Card codes are uppercase rank+suit: A,2,3,4,5,6,7,8,9,10,J,Q,K and S/H/D/C.
Examples: AS, 10H, KD, 7C.

CONSERVATIVE BIAS — when in doubt, call no_change.

Hard rules:
- Only call a tool when the change is OBVIOUS, PERSISTENT, and CLEARLY VISIBLE.
  If the image is blurry, the camera is moving, or a card is partially in the
  wrong zone due to drift — call no_change. The next frame will give you
  another chance.
- A card is "in" a zone only if its CENTER is clearly inside that zone's half
  of the image. Cards spilling across the boundary do NOT count.
- Do NOT call deal_initial if state.player_cards or state.dealer_cards already
  has cards. Re-dealing is forbidden.
- Do NOT call player_hit if state.player_done is true.
- Do NOT call dealer_draw unless state.dealer_hole_revealed is true AND a
  card is visible in the dealer area beyond what's already in state.
- Tool calls execute IMMEDIATELY. There is no second-chance gate. If you're
  not sure about a card, call no_change — you'll see another frame in ~5s.
- Use the "recent_actions" history to avoid redundant calls. If you JUST
  called player_hit("5C") and the state already reflects it, do NOT call it
  again — call no_change.
- The 4 frames you see are your cross-check. A card visible in only the
  current frame and not the previous 1-3 frames is suspicious — verify it's
  really there before calling a tool.

When in doubt → no_change. False negatives recover automatically; false
positives corrupt the game state.
"""


# ============================================================================
# Tool dispatch — maps tool names to engine method calls
# ============================================================================

def _normalize_card(s: str) -> str:
    """Uppercase + strip. The engine's parse_card handles most variants."""
    return (s or "").strip().upper()


def dispatch_tool(game_id: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single tool call against the engine. Returns a small result
    dict for logging / agent feedback. Engine errors are caught and reported."""
    try:
        if name == "deal_initial":
            p = [_normalize_card(c) for c in args.get("player_cards", [])]
            d_raw = [_normalize_card(c) for c in args.get("dealer_cards", [])]
            # Drop face-down placeholders. Engine now supports 1-card dealer
            # at deal time; we'll learn the hole rank later via dealer_reveal.
            d_visible = [c for c in d_raw if c not in ("HOLE", "?", "X", "FACEDOWN", "")]
            if not d_visible:
                return {"ok": False, "tool": name, "error": "no visible dealer card"}
            blackjack_engine.deal_initial(game_id, p, d_visible)
            return {"ok": True, "tool": name, "player": p, "dealer": d_raw, "dealer_dealt": d_visible}

        if name == "player_hit":
            rank = _normalize_card(args.get("rank", ""))
            blackjack_engine.player_action(game_id, "hit", drawn_label=rank)
            return {"ok": True, "tool": name, "rank": rank}

        if name == "player_stand":
            blackjack_engine.player_action(game_id, "stand")
            return {"ok": True, "tool": name}

        if name == "dealer_reveal":
            rank_label = _normalize_card(args.get("hole_rank", ""))
            # Engine has 1 dealer card (just upcard); APPEND the revealed rank
            # and run dealer_to_completion. No placeholder substitution.
            blackjack_engine.reveal_dealer_hole_with_rank(game_id, rank_label)
            return {"ok": True, "tool": name, "hole": rank_label}

        if name == "dealer_draw":
            rank = _normalize_card(args.get("rank", ""))
            blackjack_engine.observe_card(game_id, rank, target="dealer")
            return {"ok": True, "tool": name, "rank": rank}

        if name == "new_round":
            blackjack_engine.new_round(game_id, bet=1.0)
            return {"ok": True, "tool": name}

        if name == "no_change":
            return {"ok": True, "tool": name, "noop": True}

        return {"ok": False, "tool": name, "error": f"unknown tool: {name}"}
    except (ValueError, KeyError) as exc:
        return {"ok": False, "tool": name, "error": str(exc), "args": args}


# ============================================================================
# Agent process_frame — top-level call invoked by the API endpoint
# ============================================================================

def _summarize_state_for_agent(state) -> dict:
    """Pick the fields Claude needs to decide. Avoids dumping the whole pydantic."""
    if state is None:
        return {"missing": True}
    return {
        "game_id": state.game_id,
        "player_cards": list(state.player.cards),
        "dealer_cards_visible": (
            list(state.dealer.cards) if state.dealer_hole_revealed
            else (state.dealer.cards[:1] + ["HOLE"] if state.dealer.cards else [])
        ),
        "dealer_hole_revealed": bool(state.dealer_hole_revealed),
        "player_done": bool(state.player_done),
        "round_outcome": state.round_outcome,
        "hands_played": state.hands_played,
    }


# Thread-safe per-game lock so concurrent /process_frame calls for the same
# game serialize (avoids two Claude calls racing to mutate state).
_game_locks: Dict[str, threading.Lock] = {}
_game_locks_lock = threading.Lock()


def _get_game_lock(game_id: str) -> threading.Lock:
    with _game_locks_lock:
        if game_id not in _game_locks:
            _game_locks[game_id] = threading.Lock()
        return _game_locks[game_id]


# ---- Per-game history (recent committed tool calls) -----------------------
# Used as memory in the agent's context so it doesn't redundantly re-call
# tools for state it just committed.
_history: Dict[str, List[dict]] = {}
_HISTORY_MAX = 10


def _push_history(game_id: str, entry: dict) -> None:
    h = _history.setdefault(game_id, [])
    h.append(entry)
    if len(h) > _HISTORY_MAX:
        del h[: len(h) - _HISTORY_MAX]


def _get_history(game_id: str) -> List[dict]:
    return list(_history.get(game_id, []))


# ---- Pending action confirmations (2-frame rule) --------------------------
# The agent's first proposal of a state-mutating tool call is held as
# "pending". Only when the SAME proposal is repeated on the next frame does
# it actually execute. This kills single-frame hallucinations (camera shake,
# blurry frame, glare).
#
# Format per game: {"tool": str, "args": dict, "ts_ms": int}
_pending: Dict[str, dict] = {}
# How long a pending proposal lives before being abandoned.
# Claude vision calls take 5-10s each, so consecutive frames are 6-12s apart.
# Need a TTL longer than two frame intervals to give the confirmation a chance.
_PENDING_TTL_MS = 30_000


# All tools execute immediately. The 2-frame confirmation gate was removed:
# 4-frame temporal context already gives Claude cross-frame corroboration
# within a single call, making single-frame hallucinations unlikely.
_TOOLS_NEED_CONFIRM: set = set()
_TOOLS_IMMEDIATE = {
    "deal_initial", "player_hit", "player_stand", "dealer_reveal",
    "dealer_draw", "new_round", "no_change",
}


def _matches_pending(pending: Optional[dict], name: str, args: Dict[str, Any]) -> bool:
    """Return True if the new proposal (name, args) matches the pending one."""
    if not pending:
        return False
    if pending.get("tool") != name:
        return False
    # Compare args by JSON-serialised form (handles list/dict equality)
    try:
        return json.dumps(pending.get("args", {}), sort_keys=True) == json.dumps(args, sort_keys=True)
    except Exception:
        return pending.get("args") == args


def reset_agent_state(game_id: str) -> None:
    """Wipe per-game agent memory + pending. Called on start_game / new_round."""
    _history.pop(game_id, None)
    _pending.pop(game_id, None)


def process_frame(game_id: str, image_b64: str) -> dict:
    """Run the agent on one frame. Returns a dict describing what happened."""
    logger.info("[%s] process_frame called, image_b64_len=%d", game_id, len(image_b64 or ""))
    if not blackjack_engine.has(game_id):
        logger.warning("[%s] game not found in engine", game_id)
        return {"ok": False, "error": "game not found", "game_id": game_id}

    if not image_b64:
        return {"ok": False, "error": "empty frame"}

    # Per-game lock — only one agent call mutates a given game's state at a time
    lock = _get_game_lock(game_id)
    if not lock.acquire(blocking=False):
        return {"ok": True, "skipped": "concurrent_call_in_progress"}

    try:
        # Lazy imports to avoid circular dependencies
        from .blackjack_change_detector import frame_changed
        from .llm_blackjack import classify_with_tools, llm_enabled

        if not llm_enabled():
            logger.error("[%s] LLM not enabled", game_id)
            return {"ok": False, "error": "LLM not enabled (set LLM_API_KEY in .env)"}

        # ---- 1. Change detection (also caches this frame to recent buffer)
        changed = frame_changed(game_id, image_b64)
        logger.info("[%s] frame_changed=%s", game_id, changed)
        if not changed:
            return {"ok": True, "skipped": "frame_unchanged"}

        # Gather recent classified frames (excluding the just-cached current)
        # for temporal context. Up to 3 prior frames + current = 4 total.
        from .blackjack_change_detector import get_recent_frames
        prior = get_recent_frames(game_id)  # [(ts, b64), ...] oldest first
        prior_b64s = [b for _, b in prior]
        all_images = prior_b64s + [image_b64]  # current always last

        # ---- 2. Build context for Claude -----------------------------------
        state = blackjack_engine.get(game_id)
        state_summary = _summarize_state_for_agent(state)

        now_ms = int(time.time() * 1000)

        # Expire stale pending if older than TTL
        pending = _pending.get(game_id)
        if pending and (now_ms - int(pending.get("ts_ms", 0))) > _PENDING_TTL_MS:
            _pending.pop(game_id, None)
            pending = None

        recent = _get_history(game_id)
        # Pass the agent its own memory: recent committed actions + the pending
        # proposal (if any) it's awaiting confirmation on.
        agent_context = {
            "engine_state": state_summary,
            "recent_actions": recent[-6:],
            "pending_proposal": pending if pending else None,
        }

        # ---- 3. Call Claude with tools -------------------------------------
        logger.info("[%s] calling LLM with state=%s pending=%s history_len=%d images=%d",
                    game_id, json.dumps(state_summary), bool(pending), len(recent),
                    len(all_images))
        t0 = time.time()
        response = classify_with_tools(
            system_prompt=SYSTEM_PROMPT,
            images_b64=all_images,
            state_summary=agent_context,
            tools=TOOL_SCHEMA,
        )
        duration_ms = int((time.time() - t0) * 1000)
        if response is None:
            logger.error("[%s] LLM call returned None after %dms", game_id, duration_ms)
            return {"ok": False, "error": "LLM call failed", "duration_ms": duration_ms}
        logger.info("[%s] LLM responded in %dms, tool_calls=%d, text=%r",
                    game_id, duration_ms, len(response.get("tool_calls", [])),
                    (response.get("text") or "")[:200])

        # ---- 4. Dispatch tool calls (with 2-frame confirmation gate) -------
        actions: List[dict] = []
        for tool_call in response.get("tool_calls", []):
            name = tool_call.get("name", "")
            args = tool_call.get("arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            # ---- Immediate-execute tools (no confirmation needed) ----------
            if name in _TOOLS_IMMEDIATE:
                result = dispatch_tool(game_id, name, args)
                if result.get("ok") and name != "no_change":
                    _push_history(game_id, {
                        "ts_ms": now_ms, "tool": name, "args": args, "committed": True,
                    })
                actions.append({"tool": name, "args": args, "result": result, "phase": "immediate"})
                continue

            # ---- Confirmation-required tools -------------------------------
            if name not in _TOOLS_NEED_CONFIRM:
                actions.append({"tool": name, "args": args, "result": {"ok": False, "error": "unknown tool"}})
                continue

            current = _pending.get(game_id)
            if _matches_pending(current, name, args):
                # 2nd consecutive identical proposal → COMMIT
                result = dispatch_tool(game_id, name, args)
                _pending.pop(game_id, None)
                if result.get("ok"):
                    _push_history(game_id, {
                        "ts_ms": now_ms, "tool": name, "args": args, "committed": True,
                    })
                logger.info("[%s] COMMIT %s args=%s result=%s", game_id, name, args, result)
                actions.append({"tool": name, "args": args, "result": result, "phase": "committed"})
            else:
                # First proposal → PEND (don't execute yet)
                _pending[game_id] = {"tool": name, "args": args, "ts_ms": now_ms}
                logger.info("[%s] PEND %s args=%s (was: %s)",
                            game_id, name, args,
                            current.get("tool") if current else "(none)")
                actions.append({
                    "tool": name, "args": args,
                    "result": {"ok": True, "pending": True},
                    "phase": "pending",
                })

        # If no tool was called at all, expire any pending proposal — the
        # agent walked back from its previous suggestion.
        if not response.get("tool_calls"):
            _pending.pop(game_id, None)

        # ---- 5. Return everything -----------------------------------------
        new_state = blackjack_engine.get(game_id)
        return {
            "ok": True,
            "duration_ms": duration_ms,
            "actions": actions,
            "agent_text": response.get("text", ""),
            "pending": _pending.get(game_id),
            "state": new_state.model_dump() if new_state else None,
        }
    finally:
        lock.release()
