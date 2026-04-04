from __future__ import annotations

import hashlib
from typing import Any

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings

try:
    import jwt
except Exception:  # pragma: no cover
    jwt = None


bearer = HTTPBearer(auto_error=False)


def _api_hashes() -> set[str]:
    return {h.strip() for h in settings.api_key_hashes.split(",") if h.strip()}


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _decode_jwt(token: str) -> dict[str, Any]:
    if jwt is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="JWT library unavailable")
    if not settings.jwt_public_key_pem:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT public key not configured")
    try:
        claims = jwt.decode(
            token,
            settings.jwt_public_key_pem,
            algorithms=["RS256", "ES256"],
            audience=settings.jwt_audience or None,
            issuer=settings.jwt_issuer or None,
            options={"verify_aud": bool(settings.jwt_audience)},
        )
        if not isinstance(claims, dict):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        return claims
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {exc}")


def _extract_roles(claims: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    roles = claims.get("roles") or claims.get("role") or []
    if isinstance(roles, str):
        out.add(roles)
    elif isinstance(roles, (list, tuple)):
        out |= {str(r) for r in roles}
    scope = claims.get("scope")
    if isinstance(scope, str):
        out |= {s for s in scope.split(" ") if s}
    return out


def require_auth_roles(allowed_roles: set[str]):
    async def _dep(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if not bool(settings.auth_enabled):
            return {"sub": "anonymous", "roles": ["anonymous"]}

        hashes = _api_hashes()
        if x_api_key and hashes and _hash_token(x_api_key) in hashes:
            return {"sub": "api-key", "roles": [settings.operator_role_name, settings.admin_role_name]}

        if credentials is None or not credentials.credentials:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

        claims = _decode_jwt(credentials.credentials)
        roles = _extract_roles(claims)
        if allowed_roles and not (roles & allowed_roles):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return claims

    return _dep


def ws_validate_token(token: str | None) -> dict[str, Any]:
    if not bool(settings.auth_enabled):
        return {"sub": "anonymous", "roles": ["anonymous"]}
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing websocket token")
    return _decode_jwt(token)
