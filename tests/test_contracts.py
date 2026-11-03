from backend.app.models import FrameIn, Detection, CanonicalGameState


def test_contract_framein_required_fields():
    m = FrameIn(
        game_id="g",
        camera_id="cam",
        camera_type="hand",
        frame_id="f1",
        frame_b64="a" * 20,
        ts_ms=1,
    )
    d = m.model_dump()
    assert d["camera_id"] == "cam"
    assert d["camera_type"] == "hand"


def test_contract_detection_has_extended_fields():
    fields = set(Detection.model_fields.keys())
    assert "card_probs_topk" in fields
    assert "calibrated_card_conf_top1" in fields
    assert "is_blurry" in fields


def test_contract_canonical_state_has_trick_metadata():
    state = CanonicalGameState(game_id="g")
    dumped = state.model_dump()
    assert "trick_id" in dumped
    assert "state_version" in dumped
