from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings

try:
    import jwt
except Exception:  # pragma: no cover
    jwt = None


bearer = HTTPBearer(auto_error=False)
_REVOKED_JTI: set[str] = set()


def revoke_jti(jti: str) -> None:
    if jti:
        _REVOKED_JTI.add(str(jti))


def is_jti_revoked(jti: str | None) -> bool:
    return bool(jti and str(jti) in _REVOKED_JTI)


def _api_hashes() -> set[str]:
    return {h.strip() for h in settings.api_key_hashes.split(",") if h.strip()}


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _decode_jwt(token: str) -> dict[str, Any]:
    if jwt is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="JWT library unavailable")
    try:
        if settings.jwt_public_key_pem:
            claims = jwt.decode(
                token,
                settings.jwt_public_key_pem,
                algorithms=["RS256", "ES256"],
                audience=settings.jwt_audience or None,
                issuer=settings.jwt_issuer or None,
                options={"verify_aud": bool(settings.jwt_audience)},
            )
        else:
            claims = jwt.decode(
                token,
                settings.jwt_hs256_secret,
                algorithms=["HS256"],
                audience=settings.jwt_audience or None,
                issuer=settings.jwt_issuer or None,
                options={"verify_aud": bool(settings.jwt_audience)},
            )
        if not isinstance(claims, dict):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

        now = int(time.time())
        iat = int(claims.get("iat", 0) or 0)
        exp = int(claims.get("exp", 0) or 0)
        if iat <= 0 or exp <= 0:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing iat/exp claims")
        if exp <= now:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        if iat > (now + 60):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid iat claim")

        if bool(settings.auth_strict_iss_aud) and str(settings.deploy_env).lower() in {"prod", "production"}:
            if not settings.jwt_issuer or not settings.jwt_audience:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Issuer/audience not configured")
            if str(claims.get("iss", "")) != str(settings.jwt_issuer):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Issuer mismatch")
            aud = claims.get("aud")
            aud_values = [aud] if isinstance(aud, str) else list(aud or [])
            if str(settings.jwt_audience) not in {str(v) for v in aud_values}:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Audience mismatch")

        if bool(settings.auth_enforce_jti):
            jti = str(claims.get("jti", ""))
            if not jti:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing jti claim")
            if is_jti_revoked(jti):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")

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
    # Permit anonymous websocket subscribers for UI stability pre-login.
    if not token:
        return {"sub": "anonymous", "roles": ["anonymous"]}
    return _decode_jwt(token)


def issue_access_token(user_id: int, roles: list[str], session_id: str, ttl_sec: int) -> dict[str, Any]:
    if jwt is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="JWT library unavailable")
    now = int(time.time())
    exp = now + max(int(ttl_sec), 60)
    claims = {
        "sub": str(user_id),
        "roles": roles,
        "session_id": session_id,
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": exp,
    }
    if settings.jwt_issuer:
        claims["iss"] = settings.jwt_issuer
    if settings.jwt_audience:
        claims["aud"] = settings.jwt_audience

    if settings.jwt_private_key_pem:
        token = jwt.encode(claims, settings.jwt_private_key_pem, algorithm="RS256")
    else:
        token = jwt.encode(claims, settings.jwt_hs256_secret, algorithm="HS256")
    return {"jwt_access_token": token, "exp_ts": exp, "session_id": session_id or str(uuid.uuid4())}
