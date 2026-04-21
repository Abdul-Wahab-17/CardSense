from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from .config import settings
from .models import ActionScore, CanonicalGameState

logger = logging.getLogger("cardsense.llm")

_DEFAULT_RULES_TEXT = (
    "You are an expert card-game move advisor for trick-taking play.\n"
    "Rules:\n"
    "- Recommend exactly one best move from the legal actions list.\n"
    "- Never invent cards; only choose legal cards provided in the prompt.\n"
    "- Prefer higher expected value and lower tactical risk.\n"
    "- Respect lead-suit, must-follow-suit, trump, and turn constraints.\n"
    "- Return concise reasoning and uncertainty notes.\n"
)


def llm_advisor_enabled() -> bool:
    mode = str(settings.advisor_mode or "rules").strip().lower()
    if mode not in {"llm", "hybrid"}:
        return False
    if not bool(settings.llm_enabled):
        return False
    if not str(settings.llm_api_key or "").strip():
        return False
    return True


def _clean_json_payload(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _normalize_card(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def request_llm_recommendation(
    state: CanonicalGameState,
    legal_actions: list[str],
    action_scores: list[ActionScore],
    fallback_card: str,
) -> dict[str, Any] | None:
    if not llm_advisor_enabled():
        return None
    if not legal_actions:
        return None

    rules_text = str(settings.llm_rules_text or "").strip() or _DEFAULT_RULES_TEXT
    legal_upper = [a.upper() for a in legal_actions]
    scores_payload = [
        {
            "card": row.card,
            "ev": round(float(row.total_ev), 4),
            "risk": round(float(row.risk), 4),
            "rule_validity": round(float(row.rule_validity_score), 4),
            "state_stability": round(float(row.state_stability_score), 4),
            "info_value": round(float(row.info_value_score), 4),
        }
        for row in action_scores[:12]
    ]
    prompt_payload = {
        "game_id": state.game_id,
        "turn": {
            "leader": state.turn.leader,
            "to_play": state.turn.to_play,
            "lead_suit": state.turn.lead_suit,
        },
        "rules": {
            "variant_name": state.rules.variant_name,
            "must_follow_suit": state.rules.must_follow_suit,
            "must_trump_if_void": state.rules.must_trump_if_void,
            "trump_mode": state.rules.trump_mode,
            "trump_suit": state.rules.trump_suit,
        },
        "hand_cards": state.hand_cards[:16],
        "table_cards": [row.model_dump() for row in state.table_trick_cards],
        "legal_actions": legal_upper,
        "candidate_scores": scores_payload,
        "uncertainty": state.uncertainty.model_dump(),
        "fallback_best_card": fallback_card,
        "output_contract": {
            "best_card": "string from legal_actions",
            "confidence": "number between 0 and 1",
            "why": ["max 4 short bullets"],
            "uncertainty_notes": ["max 4 short notes"],
            "alternatives": ["up to 3 cards from legal_actions, excluding best_card"],
        },
    }
    messages = [
        {"role": "system", "content": rules_text},
        {
            "role": "user",
            "content": (
                "Choose the best move and output valid JSON only. "
                "No markdown, no prose outside JSON.\n\n"
                f"{json.dumps(prompt_payload, ensure_ascii=True)}"
            ),
        },
    ]
    payload = {
        "model": str(settings.llm_model),
        "temperature": float(settings.llm_temperature),
        "max_tokens": int(settings.llm_max_tokens),
        "response_format": {"type": "json_object"},
        "messages": messages,
    }

    request = urllib.request.Request(
        url=f"{str(settings.llm_api_base).rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {str(settings.llm_api_key).strip()}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=max(1.0, float(settings.llm_timeout_sec))) as response:
            body = response.read().decode("utf-8")
    except TimeoutError:
        logger.warning("LLM timeout for game_id=%s model=%s", state.game_id, settings.llm_model)
        return None
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("LLM transport failure for game_id=%s error=%s", state.game_id, exc)
        return None

    try:
        envelope = json.loads(body)
        content = str(envelope["choices"][0]["message"]["content"])
        model_out = json.loads(_clean_json_payload(content))
    except Exception as exc:
        logger.warning("LLM parse failure for game_id=%s error=%s", state.game_id, exc)
        return None

    best_card = _normalize_card(model_out.get("best_card"))
    if best_card not in legal_upper:
        logger.warning(
            "LLM illegal best_card for game_id=%s best_card=%s legal_count=%s",
            state.game_id,
            best_card,
            len(legal_upper),
        )
        return None

    alternatives_raw = model_out.get("alternatives", [])
    alternatives: list[str] = []
    if isinstance(alternatives_raw, list):
        for item in alternatives_raw:
            card = _normalize_card(item)
            if not card or card == best_card or card not in legal_upper or card in alternatives:
                continue
            alternatives.append(card)
            if len(alternatives) >= 3:
                break

    why_raw = model_out.get("why", [])
    why = [str(item).strip() for item in why_raw if str(item).strip()][:4] if isinstance(why_raw, list) else []

    notes_raw = model_out.get("uncertainty_notes", [])
    notes = [str(item).strip() for item in notes_raw if str(item).strip()][:4] if isinstance(notes_raw, list) else []

    confidence = max(0.0, min(1.0, _safe_float(model_out.get("confidence"), fallback=0.0)))
    return {
        "best_card": best_card,
        "confidence": confidence,
        "why": why,
        "uncertainty_notes": notes,
        "alternatives": alternatives,
    }
