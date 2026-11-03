import pytest

from backend.app.security import _extract_roles, ws_validate_token


def test_security_extract_roles_from_claims():
    claims = {"roles": ["operator"], "scope": "read write"}
    roles = _extract_roles(claims)
    assert "operator" in roles
    assert "read" in roles


def test_security_ws_missing_token_allowed_as_anonymous_when_auth_enabled(monkeypatch):
    from backend.app import security as sec

    monkeypatch.setattr(sec.settings, "auth_enabled", 1)
    claims = ws_validate_token(None)
    assert claims.get("sub") == "anonymous"
