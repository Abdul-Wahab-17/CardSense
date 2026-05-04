"""LLM advisor for Blackjack — explanations + tipsi (post-round teach mode).

Wraps OpenRouter (or any OpenAI-compatible endpoint) to:

1. `explain_recommendation()` — turn a deterministic recommendation into a
   short, glanceable HUD-style sentence. Used live during play.

2. `tipsi_review()` — given a finished round (and optionally surrounding
   context), produce a 2-4 sentence "what could have been better" review.

Both functions are best-effort: if the LLM is unavailable, misconfigured,
times out, or returns garbage, we fall back to the deterministic reason
already provided by the recommender. The deterministic core never depends
on this module being healthy.

Configure via existing settings (see backend/app/config.py):
    advisor_mode = 'hybrid' or 'llm'
    llm_enabled = 1
    llm_api_base = 'https://openrouter.ai/api/v1'
    llm_api_key = '<openrouter sk-or-...>'
    llm_model = 'anthropic/claude-sonnet-4.5' (or 4.6 when available)
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .blackjack import BlackjackState
from .blackjack_advice import Recommendation
from .config import settings

logger = logging.getLogger("cardsense.blackjack.llm")


_EXPLAIN_SYSTEM_PROMPT = (
    "You are an in-game AI assistant displayed in a player's smart glasses while "
    "they play Blackjack. The deterministic recommender has already chosen the "
    "correct action via basic strategy + Hi-Lo card counting. Your job is to "
    "narrate WHY in <=12 words, using language a casual player understands. "
    "Be concrete. Reference the dealer upcard or count when it's the deciding "
    "factor. NEVER override the recommendation."
)

_TIPSI_SYSTEM_PROMPT = (
    "You are 'Tipsi', a friendly Blackjack coach reviewing a finished hand. "
    "Given the hand outcome, the player's actions, the dealer's actions, and "
    "the running card count, produce a 2-4 sentence review. Be specific about "
    "what was correct and what (if anything) could have been better. Do not "
    "lecture about basic principles — focus on this hand. End with one concrete "
    "tip for the next hand."
)


def llm_enabled() -> bool:
    if not bool(settings.llm_enabled):
        return False
    if not str(settings.llm_api_key or "").strip():
        return False
    return True


def explain_recommendation(state: BlackjackState, rec: Recommendation) -> Optional[str]:
    """Return a short LLM narration, or None if LLM unavailable.

    Caller should fall back to `rec.reason` (deterministic) if this returns None.
    """
    if not llm_enabled():
        return None

    user_payload: Dict[str, Any] = {
        "player_cards": state.player.cards,
        "player_total": state.player.total()[0],
        "player_soft": state.player.total()[1],
        "dealer_upcard": state.dealer.cards[0] if state.dealer.cards else None,
        "running_count": rec.diagnostics.running_count,
        "true_count": rec.diagnostics.true_count,
        "cards_remaining": rec.diagnostics.cards_remaining,
        "bust_prob_on_hit": rec.diagnostics.bust_prob_on_hit,
        "dealer_bust_prob": rec.diagnostics.dealer_bust_prob,
        "recommended_action": rec.action,
        "used_count_deviation": rec.used_deviation,
        "deterministic_reason": rec.reason,
    }
    user_text = (
        "State:\n"
        + json.dumps(user_payload, separators=(",", ":"))
        + "\n\nWrite the narration now (<=12 words, concrete, single sentence):"
    )
    try:
        return _chat_complete(_EXPLAIN_SYSTEM_PROMPT, user_text, max_tokens=60, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("explain_recommendation failed: %s", exc)
        return None


def classify_with_tools(
    *,
    system_prompt: str,
    images_b64: List[str],
    state_summary: dict,
    tools: List[dict],
) -> Optional[dict]:
    """Call Claude (via OpenRouter, OpenAI-compat) with one or more images +
    state + a tool schema. Images are passed in temporal order (oldest first);
    the LAST image is the "current" frame.

    Returns: {"text": str, "tool_calls": [...]} or None on failure.
    """
    if not llm_enabled():
        return None

    api_base = str(settings.llm_api_base or "").rstrip("/")
    api_key = str(settings.llm_api_key or "").strip()
    model = str(settings.llm_vision_model or settings.llm_model or "").strip()
    timeout = max(15.0, float(settings.llm_timeout_sec or 8.0) * 2)

    if not api_base or not api_key or not model:
        return None
    if not images_b64:
        return None

    n = len(images_b64)
    if n == 1:
        frames_intro = "You see ONE frame: the current state of the table."
    else:
        frames_intro = (
            f"You see {n} frames in temporal order, oldest first. "
            f"Frames 1..{n-1} are recent past; frame {n} (the LAST image) is "
            "the CURRENT state of the table. Use the earlier frames to verify "
            "card identities (cross-check if a frame is blurry) and to detect "
            "what JUST changed (a card visible only in the current frame was "
            "just placed)."
        )

    user_text = (
        frames_intro
        + "\n\nCurrent engine state:\n"
        + json.dumps(state_summary, separators=(",", ":"))
        + "\n\nIf the cards visible in the CURRENT frame differ from the "
        "engine state, call tools to sync. Use earlier frames as confirmation. "
        "If state already matches the current frame, call no_change."
    )

    # Build content: image1, image2, ..., imageN, text
    user_content: List[dict] = []
    for b64 in images_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    user_content.append({"type": "text", "text": user_text})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "tools": tools,
        # "required" forces Claude to call exactly one tool. Eliminates the
        # paragraph-of-prose preamble that was costing ~300-500ms per call.
        # The "no_change" tool exists specifically as a valid choice when
        # nothing needs to mutate.
        "tool_choice": "required",
        # Defensive: extended thinking is OFF by default on Claude 4.x but
        # this guarantees it stays off via OpenRouter's unified param.
        "reasoning": {"enabled": False},
        "temperature": 0.0,
        # Lower than before (was 800). Tool calls are short — 200 is plenty
        # and any prose Claude tries to emit gets capped.
        "max_tokens": 300,
    }
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://cardsense.local",
            "X-Title": "CardSense Blackjack Agent",
        },
    )
    try:
        total_kb = sum(len(b) for b in images_b64) // 1024
        logger.info("vision call → %s model=%s tools=%d images=%d total_kb=%d",
                    api_base, model, len(tools), len(images_b64), total_kb)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        logger.warning("agent HTTP %s: %s", exc.code, exc.reason)
        try:
            err_body = exc.read().decode("utf-8", errors="replace")[:400]
            logger.warning("agent HTTP body: %s", err_body)
        except Exception:
            pass
        return None
    except urllib.error.URLError as exc:
        logger.warning("agent URL error: %s", exc.reason)
        return None
    except TimeoutError:
        logger.warning("agent timeout after %.1fs", timeout)
        return None

    logger.info("vision raw response (first 500 chars): %r", raw[:500])
    try:
        data = json.loads(raw)
        msg = data["choices"][0]["message"]
        text = msg.get("content") or ""
        # OpenAI-compat: tool_calls is a list of {id, type, function: {name, arguments}}
        tool_calls = []
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            tool_calls.append({"name": fn.get("name", ""), "arguments": args})
        return {"text": text if isinstance(text, str) else "", "tool_calls": tool_calls}
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("agent bad response: %s | raw=%r", exc, raw[:300])
        return None


def classify_cards_in_frame(image_b64: str) -> Optional[dict]:
    """Send a frame to Claude vision via OpenRouter, return identified cards
    grouped by zone.

    Returns: {"player": ["7H", "KS", ...], "dealer": ["10D", "5C", ...]} or None
    """
    if not llm_enabled():
        return None

    system_prompt = (
        "You identify standard playing cards from a single overhead camera "
        "image of a Blackjack table. The TOP HALF of the image is the dealer's "
        "area; the BOTTOM HALF is the player's area. List all visible card "
        "ranks and suits. Return ONLY valid JSON in this exact form, no other "
        "text:\n"
        '{"dealer": ["10S", "7H"], "player": ["AC", "KD"]}\n'
        "Use uppercase rank+suit codes: A,2,3,4,5,6,7,8,9,10,J,Q,K and "
        "S(spades),H(hearts),D(diamonds),C(clubs). Examples: AS=ace of spades, "
        "10H=ten of hearts, KD=king of diamonds. If a card is face-down or "
        "unreadable, omit it. If no cards, return empty arrays."
    )

    try:
        raw = _chat_complete_vision(
            system_prompt,
            image_b64,
            "Identify the cards. Respond with JSON only.",
            max_tokens=400,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("classify_cards_in_frame failed: %s", exc)
        return None
    if not raw:
        return None

    # Strip code fences if Claude wraps in ```json ... ```
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```")).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("classify_cards_in_frame: bad JSON: %s | raw=%r", exc, raw[:200])
        return None

    # Validate / coerce
    out = {"player": [], "dealer": []}
    for zone in ("player", "dealer"):
        cards = result.get(zone, [])
        if not isinstance(cards, list):
            continue
        for c in cards:
            if isinstance(c, str):
                out[zone].append(c.strip().upper())
    return out


def tipsi_review(state: BlackjackState, recent_history: List[dict]) -> Optional[str]:
    """Post-round coach review. Returns short paragraph or None on failure."""
    if not llm_enabled():
        return None
    if not recent_history:
        return None

    user_payload = {
        "last_round": recent_history[-1],
        "previous_rounds": recent_history[-6:-1],  # context, last 5 prior
        "current_count": sum(1 if r in ("2","3","4","5","6") else (-1 if r in ("10","J","Q","K","A") else 0) for r in state.seen_cards),
        "cards_remaining": state.cards_remaining(),
    }
    user_text = (
        "Hand context:\n"
        + json.dumps(user_payload, separators=(",", ":"), default=str)
        + "\n\nReview the last hand now (2-4 sentences, end with one tip):"
    )
    try:
        return _chat_complete(_TIPSI_SYSTEM_PROMPT, user_text, max_tokens=200, temperature=0.4)
    except Exception as exc:  # noqa: BLE001
        logger.warning("tipsi_review failed: %s", exc)
        return None


# ---------------- transport ---------------------------------------------------

def _chat_complete_vision(
    system: str,
    image_b64: str,
    user_text: str,
    *,
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    """OpenAI/OpenRouter-compatible vision chat completion.

    `image_b64` is raw base64 (no data:image/... prefix). We add the prefix
    here using image/jpeg as the assumed MIME (matches the frame uploads from
    our frontend's canvas.toDataURL("image/jpeg", ...)).
    """
    api_base = str(settings.llm_api_base or "").rstrip("/")
    api_key = str(settings.llm_api_key or "").strip()
    model = str(settings.llm_model or "").strip()
    # Vision calls take longer than text — give them more room.
    timeout = max(20.0, float(settings.llm_timeout_sec or 8.0) * 3)

    if not api_base or not api_key or not model:
        return None

    url = f"{api_base}/chat/completions"
    image_url = f"data:image/jpeg;base64,{image_b64}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://cardsense.local",
            "X-Title": "CardSense Blackjack Vision",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        logger.warning("vision HTTP %s: %s", exc.code, exc.reason)
        return None
    except urllib.error.URLError as exc:
        logger.warning("vision URL error: %s", exc.reason)
        return None
    except TimeoutError:
        logger.warning("vision timeout after %.1fs", timeout)
        return None

    try:
        data = json.loads(raw)
        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            return None
        return text.strip()
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("vision bad response: %s", exc)
        return None


def _chat_complete(system: str, user: str, *, max_tokens: int, temperature: float) -> Optional[str]:
    api_base = str(settings.llm_api_base or "").rstrip("/")
    api_key = str(settings.llm_api_key or "").strip()
    model = str(settings.llm_model or "").strip()
    timeout = float(settings.llm_timeout_sec or 8.0)

    if not api_base or not api_key or not model:
        return None

    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # OpenRouter recommends these headers for attribution. Harmless on
            # plain OpenAI / other compatible backends.
            "HTTP-Referer": "https://cardsense.local",
            "X-Title": "CardSense Blackjack Advisor",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        logger.warning("LLM HTTP %s: %s", exc.code, exc.reason)
        return None
    except urllib.error.URLError as exc:
        logger.warning("LLM URL error: %s", exc.reason)
        return None
    except TimeoutError:
        logger.warning("LLM timeout after %.1fs", timeout)
        return None

    try:
        data = json.loads(raw)
        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            return None
        return text.strip()
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("LLM bad response: %s", exc)
        return None
