from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gearmate.config import Settings

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class CurrentUser:
    user_id: str
    nickname: str | None
    timezone: str
    roles: tuple[str, ...]
    access_token: str


def _settings(request: Request) -> Settings:
    return request.app.state.settings


async def current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ],
    settings: Annotated[Settings, Depends(_settings)],
) -> CurrentUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
        )
    if settings.jwt_public_key_path is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT public key is not configured",
        )
    try:
        public_key = settings.jwt_public_key_path.read_text(encoding="utf-8")
        claims = jwt.decode(
            credentials.credentials,
            public_key,
            algorithms=["RS256"],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except (OSError, jwt.PyJWTError) as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
        ) from error
    roles_claim = claims.get("roles")
    roles = tuple(str(role) for role in roles_claim) if isinstance(roles_claim, list) else ()
    return CurrentUser(
        user_id=str(claims["sub"]),
        nickname=str(claims["nickname"]) if claims.get("nickname") is not None else None,
        timezone=str(claims.get("timezone") or "UTC"),
        roles=roles,
        access_token=credentials.credentials,
    )
