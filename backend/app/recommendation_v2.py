from __future__ import annotations

from typing import Any

from .belief import belief_diagnostics, build_belief, sample_worlds
from .models import ActionScore, CanonicalGameState
from .rules import legal_actions
from .simulator import evaluate_actions


def compute_legal_actions(match_state: CanonicalGameState, seat: str) -> list[str]:
    if seat != "me":
        return []
    return legal_actions(match_state)


def build_belief_state(match_state: CanonicalGameState, history: list[dict] | None, config: dict[str, Any] | None) -> dict[str, Any]:
    _ = history
    _ = config
    belief = build_belief(match_state)
    return {
        "belief": belief,
        "diagnostics": belief_diagnostics(belief),
    }


def sample_hidden_worlds_v2(match_state: CanonicalGameState, belief_state: dict[str, Any], n: int = 64, seed: int | None = None) -> list[dict[str, list[str]]]:
    _ = belief_state
    import random

    rng = random.Random(seed) if seed is not None else None
    return sample_worlds(match_state, n=n, rng=rng)


def simulate_action_batch(match_state: CanonicalGameState, legal: list[str], worlds: list[dict[str, list[str]]], seed: int | None = None) -> list[ActionScore]:
    # Keep a single simulation backend for determinism and to avoid divergence.
    _ = worlds
    raw = evaluate_actions(match_state, rollout_count=max(32, len(legal) * 16), simulation_seed=seed)
    legal_set = set(legal)
    return [score for score in raw if score.card in legal_set]


def score_actions_v2(match_state: CanonicalGameState, legal: list[str], action_scores: list[ActionScore], belief_state: dict[str, Any]) -> list[ActionScore]:
    legal_set = set(legal)
    state_stability = float(match_state.uncertainty.state_conf)
    entropy = belief_state.get("diagnostics", {}).get("belief_entropy_per_seat", {})
    info_entropy = float(sum(float(v) for v in entropy.values()))
    info_norm = max(0.0, min(1.0, 1.0 - (info_entropy / 15.0)))

    scored: list[ActionScore] = []
    for row in action_scores:
        row.rule_validity_score = 1.0 if row.card in legal_set else 0.0
        row.state_stability_score = round(state_stability, 4)
        row.info_value_score = round(max(0.0, min(1.0, (0.5 * info_norm) + (0.5 * max(0.0, row.ev_information_gain + 0.2)))), 4)
        # Bias ranking to always prefer legal and stable outputs.
        row.total_ev = round(
            float(row.total_ev)
            + (0.15 * float(row.rule_validity_score))
            + (0.10 * float(row.state_stability_score))
            + (0.10 * float(row.info_value_score)),
            4,
        )
        scored.append(row)

    scored.sort(key=lambda item: item.total_ev, reverse=True)
    return scored


def select_recommendation(scored_actions: list[ActionScore], legal: list[str]) -> ActionScore | None:
    legal_set = set(legal)
    for action in scored_actions:
        if action.card in legal_set:
            return action
    return None


def explain_recommendation(match_state: CanonicalGameState, best: ActionScore | None, belief_state: dict[str, Any]) -> tuple[list[str], list[str]]:
    why: list[str] = []
    notes: list[str] = list(match_state.uncertainty.notes)

    if best is None:
        return ["No legal recommendation available"], ["Insufficient state quality"]

    why.append("Action is legal under current variant constraints")
    why.append("Selected by expected value under sampled hidden worlds")
    why.append("Scored with rule validity, state stability, and information value")

    if match_state.uncertainty.state_conf < 0.6:
        notes.append("State stability is low; confidence reduced")
    if float(best.info_value_score) < 0.35:
        notes.append("Information value is limited for this state")
    if not belief_state.get("belief"):
        notes.append("Belief state unavailable; uncertainty elevated")

    return why[:5], notes[:8]
