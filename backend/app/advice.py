from __future__ import annotations

from .models import AdvisorResponse, HandAdvice, HandState, HelpTip


def build_basic_hand_advice(hand: HandState) -> HandAdvice:
    card_count = len(hand.cards)
    if card_count == 0:
        summary = "No cards detected yet. Keep both cameras steady."
    else:
        summary = f"{card_count}/{hand.max_cards} cards detected. Waiting for stable game context."
    tips = [
        HelpTip(
            title="Stability",
            detail="Keep hand and table in frame for 2-3 seconds to improve tracking confidence.",
        ),
    ]
    if card_count < hand.max_cards:
        tips.append(
            HelpTip(
                title="Coverage",
                detail=f"Only {card_count}/{hand.max_cards} cards are stable. Adjust angle and lighting.",
            )
        )
    return HandAdvice(
        camera_id=hand.camera_id,
        updated_ts_ms=hand.updated_ts_ms,
        summary=summary,
        discard_candidates=[card.label for card in hand.cards[:3]],
        tips=tips,
    )


def build_hand_advice_from_recommendation(hand: HandState, recommendation: AdvisorResponse) -> HandAdvice:
    best_card = str(recommendation.best_action.card or "").strip()
    confidence = max(0.0, min(1.0, float(recommendation.best_action.confidence or 0.0)))
    move_list = [best_card] if best_card else []
    move_list.extend([row.card for row in recommendation.alternatives if row.card and row.card not in move_list])
    discard_candidates = move_list[:3]

    summary = "No legal move recommendation available yet."
    if best_card:
        summary = (
            f"Best move: {best_card} "
            f"(confidence {(confidence * 100.0):.0f}%, mode={recommendation.recommendation_mode})."
        )

    tips: list[HelpTip] = []
    for idx, reason in enumerate((recommendation.why or [])[:3], start=1):
        tips.append(HelpTip(title=f"Reason {idx}", detail=str(reason)))
    for idx, note in enumerate((recommendation.uncertainty_notes or [])[:2], start=1):
        tips.append(HelpTip(title=f"Uncertainty {idx}", detail=str(note)))

    return HandAdvice(
        camera_id=hand.camera_id,
        updated_ts_ms=hand.updated_ts_ms,
        summary=summary,
        discard_candidates=discard_candidates,
        tips=tips,
    )
