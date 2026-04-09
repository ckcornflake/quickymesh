"""
API key authentication for quickymesh.

Users are configured in users.yaml (or pointed to by the QUICKYMESH_USERS_FILE
env var).  A fallback single-user mode is supported via the API_KEY env var.

users.yaml format
-----------------
users:
  alice:
    api_key: "super-secret-key-abc"
    role: admin
  bob:
    api_key: "another-key-xyz"
    role: user
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

import yaml
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)

# api_key → {"username": ..., "role": ...}
_users: dict[str, dict] = {}

# When False, _get_current_user bypasses all checks and returns a synthetic
# admin user.  This is the default for OSS / single-user / localhost setups.
# Enable by passing `auth_enabled=True` (or a `users_file`) to create_app, or
# via the api_server.py --auth-file CLI flag.
_auth_enabled: bool = False


_log = logging.getLogger(__name__)


def set_auth_enabled(enabled: bool) -> None:
    """Toggle whether _get_current_user enforces credentials."""
    global _auth_enabled
    _auth_enabled = enabled


def is_auth_enabled() -> bool:
    return _auth_enabled


def load_users(users_file: str | Path | None = None) -> None:
    """Load users from YAML into memory. Call once at server startup."""
    global _users
    _users = {}

    path = Path(users_file or os.environ.get("QUICKYMESH_USERS_FILE", "users.yaml"))
    if not path.exists():
        # Fallback: single-user mode via API_KEY env var
        api_key = os.environ.get("API_KEY")
        if api_key:
            _users[api_key] = {"username": "admin", "role": "admin"}
            _log.info("Auth: single-user mode (API_KEY env var)")
        else:
            _log.warning(
                "Auth: no users configured — all requests will be rejected. "
                "Set API_KEY env var or create users.yaml."
            )
        return

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for username, info in (data.get("users") or {}).items():
        key = info.get("api_key", "")
        if key:
            _users[key] = {"username": username, "role": info.get("role", "user")}
    _log.info("Auth: loaded %d user(s) from %s", len(_users), path)


class User:
    __slots__ = ("username", "role")

    def __init__(self, username: str, role: str) -> None:
        self.username = username
        self.role = role

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def __repr__(self) -> str:
        return f"User({self.username!r}, role={self.role!r})"


def _get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    if not _auth_enabled:
        return User("local", "admin")
    if credentials is None:
        _log.warning("Auth failure: missing Authorization header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header (Bearer <api_key>)",
        )
    info = _users.get(credentials.credentials)
    if not info:
        # Log only a prefix of the key — never the full value
        prefix = credentials.credentials[:6] if credentials.credentials else "(empty)"
        _log.warning("Auth failure: invalid API key (prefix=%s...)", prefix)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return User(info["username"], info["role"])


# Annotated type for use in route signatures:
#   async def my_route(user: CurrentUser): ...
CurrentUser = Annotated[User, Depends(_get_current_user)]
