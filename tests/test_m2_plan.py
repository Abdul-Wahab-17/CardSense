import asyncio

import pytest
from fastapi import HTTPException

from backend.app.auth_store import AuthStore
from backend.app.main import (
    _validate_camera_bind_request,
    auth_register,
    camera_sessions,
    select_worker_route,
)
from backend.app.match_engine import MatchEngine
from backend.app.models import AuthRegisterIn, CameraBindIn, CameraSessionState, FrameIn, RoundStartIn, VariantConfig


def test_auth_store_register_login_rotate(tmp_path):
    db = tmp_path / "auth.db"
    store = AuthStore(str(db))
    created = store.create_user_account("a@example.com", "password123", "A")
    assert created["user_id"] > 0

    auth = store.authenticate_user("a@example.com", "password123")
    assert auth["auth_ok"]

    issued = store.issue_refresh_token(int(auth["user"]["id"]), "sess-1", 600, {"ip": "1.1.1.1", "user_agent": "pytest"})
    rotated = store.rotate_refresh_token(issued["refresh_token"], 600, {"ip": "1.1.1.1", "user_agent": "pytest"})
    assert rotated["ok"]


def test_variant_and_round_flow():
    engine = MatchEngine()
    variant = VariantConfig(
        variant_id="v1",
        player_count=4,
        seat_order=["me", "left", "across", "right"],
        play_direction="clockwise",
        cards_per_player=13,
        trick_size=4,
    )
    state = engine.start_match("g1", variant, "me")
    assert state.phase == "in_round"
    state2 = engine.start_round("g1", RoundStartIn(round_index=2, dealer_seat="right"))
    assert state2.round_index == 2


def test_frame_contract_has_v2_fields():
    frame = FrameIn(
        game_id="g",
        camera_id="cam",
        camera_type="hand",
        frame_id="f",
        target_model_type="hand",
        device_id="dev1",
        session_id="sess1",
        frame_b64="a" * 20,
        ts_ms=1,
    )
    dumped = frame.model_dump()
    assert dumped["target_model_type"] == "hand"
    assert dumped["device_id"] == "dev1"
    assert dumped["session_id"] == "sess1"


def test_camera_bind_schema_and_route_selector():
    bind = CameraBindIn(camera_id="cam-1", camera_type="table", seat_hint="left", device_label="laptop", stream_profile="balanced")
    assert bind.camera_type == "table"
    route = select_worker_route("table", "table", [{"supports_table": True}])
    assert route == "route:table"


def test_auth_register_duplicate_email_returns_409(tmp_path, monkeypatch):
    from backend.app import main as main_mod

    store = AuthStore(str(tmp_path / "auth-main.db"))
    monkeypatch.setattr(main_mod, "auth_store", store)

    result = asyncio.run(auth_register(AuthRegisterIn(email="dup@example.com", password="password123", display_name="Dup")))
    assert result["ok"]

    with pytest.raises(HTTPException) as exc:
        asyncio.run(auth_register(AuthRegisterIn(email="dup@example.com", password="password123", display_name="Dup2")))
    assert exc.value.status_code == 409


def test_camera_binding_enforces_max_two_and_unique_type():
    game_id = "g-bind"
    camera_sessions.pop(game_id, None)
    try:
        camera_sessions[game_id]["cam-hand"] = CameraSessionState(game_id=game_id, camera_id="cam-hand", camera_type="hand")
        _validate_camera_bind_request(game_id, "cam-table", "table")

        camera_sessions[game_id]["cam-table"] = CameraSessionState(game_id=game_id, camera_id="cam-table", camera_type="table")

        with pytest.raises(HTTPException):
            _validate_camera_bind_request(game_id, "cam-extra", "hand")
    finally:
        camera_sessions.pop(game_id, None)
