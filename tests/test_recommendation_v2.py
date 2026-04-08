from backend.app.models import CanonicalGameState
from backend.app.recommendation_v2 import (
    build_belief_state,
    compute_legal_actions,
    sample_hidden_worlds_v2,
    score_actions_v2,
    select_recommendation,
    simulate_action_batch,
)


def test_v2_recommendation_is_always_legal():
    state = CanonicalGameState(game_id="g-v2", hand_cards=["AH", "2S", "9D", "KC"])
    state.turn.to_play = "me"
    state.turn.lead_suit = "H"
    state.rules.must_follow_suit = True

    legal = compute_legal_actions(state, "me")
    belief = build_belief_state(state, history=[], config={})
    worlds = sample_hidden_worlds_v2(state, belief, n=48, seed=42)
    simulated = simulate_action_batch(state, legal, worlds, seed=42)
    scored = score_actions_v2(state, legal, simulated, belief)
    best = select_recommendation(scored, legal)

    assert best is not None
    assert best.card in legal


def test_v2_scoring_deterministic_with_seed():
    state = CanonicalGameState(game_id="g-v2-d", hand_cards=["AS", "KH", "2D", "10C"])
    state.turn.to_play = "me"

    legal = compute_legal_actions(state, "me")
    belief = build_belief_state(state, history=[], config={})

    worlds1 = sample_hidden_worlds_v2(state, belief, n=32, seed=777)
    sim1 = simulate_action_batch(state, legal, worlds1, seed=777)
    score1 = score_actions_v2(state, legal, sim1, belief)

    worlds2 = sample_hidden_worlds_v2(state, belief, n=32, seed=777)
    sim2 = simulate_action_batch(state, legal, worlds2, seed=777)
    score2 = score_actions_v2(state, legal, sim2, belief)

    assert [row.model_dump() for row in score1] == [row.model_dump() for row in score2]
