from backend.app.main import _redact_event


def test_chaos_redaction_handles_missing_keys():
    event = {"camera_id": "cam-1", "x": 1}
    redacted = _redact_event(event)
    assert redacted["camera_id"] == "cam-1"


def test_chaos_redaction_drops_frame_payload():
    event = {"frame_b64": "abc", "raw_frame": "def", "game_id": "g"}
    redacted = _redact_event(event)
    assert "frame_b64" not in redacted
    assert "raw_frame" not in redacted
