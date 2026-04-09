"""
Tests for src/api/auth.py
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from src.api.auth import User, load_users, _users


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_users_yaml(tmp_path: Path, content: str) -> Path:
    f = tmp_path / "users.yaml"
    f.write_text(textwrap.dedent(content))
    return f


# ---------------------------------------------------------------------------
# load_users
# ---------------------------------------------------------------------------


class TestLoadUsers:
    def setup_method(self):
        """Reset the module-level _users dict before each test."""
        import src.api.auth as auth_mod
        auth_mod._users.clear()

    def test_loads_users_from_yaml(self, tmp_path):
        f = _write_users_yaml(tmp_path, """
            users:
              alice:
                api_key: "key-alice"
                role: admin
              bob:
                api_key: "key-bob"
                role: user
        """)
        load_users(f)
        import src.api.auth as auth_mod
        assert "key-alice" in auth_mod._users
        assert "key-bob" in auth_mod._users
        assert auth_mod._users["key-alice"]["username"] == "alice"
        assert auth_mod._users["key-alice"]["role"] == "admin"
        assert auth_mod._users["key-bob"]["role"] == "user"

    def test_missing_yaml_falls_back_to_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("API_KEY", "env-test-key")
        load_users(tmp_path / "nonexistent.yaml")
        import src.api.auth as auth_mod
        assert "env-test-key" in auth_mod._users
        assert auth_mod._users["env-test-key"]["username"] == "admin"
        assert auth_mod._users["env-test-key"]["role"] == "admin"

    def test_no_yaml_no_env_var_gives_empty_users(self, tmp_path, monkeypatch):
        monkeypatch.delenv("API_KEY", raising=False)
        load_users(tmp_path / "nonexistent.yaml")
        import src.api.auth as auth_mod
        assert auth_mod._users == {}

    def test_empty_yaml_gives_empty_users(self, tmp_path):
        f = tmp_path / "users.yaml"
        f.write_text("")
        load_users(f)
        import src.api.auth as auth_mod
        assert auth_mod._users == {}

    def test_user_without_api_key_is_skipped(self, tmp_path):
        f = _write_users_yaml(tmp_path, """
            users:
              ghost:
                role: user
        """)
        load_users(f)
        import src.api.auth as auth_mod
        assert auth_mod._users == {}

    def test_calling_load_users_twice_replaces_previous(self, tmp_path):
        f1 = _write_users_yaml(tmp_path, """
            users:
              alice:
                api_key: "key-alice"
                role: admin
        """)
        f2 = tmp_path / "users2.yaml"
        f2.write_text("users:\n  bob:\n    api_key: key-bob\n    role: user\n")
        load_users(f1)
        load_users(f2)
        import src.api.auth as auth_mod
        assert "key-alice" not in auth_mod._users
        assert "key-bob" in auth_mod._users


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------


class TestAuthLogging:
    """These tests exercise the auth-enabled code paths.  Auth is OFF by
    default (OSS / localhost) so we explicitly enable it here."""

    def test_missing_auth_logs_warning(self, tmp_path, caplog):
        import logging
        from src.api.auth import _get_current_user, set_auth_enabled
        import src.api.auth as auth_mod
        auth_mod._users.clear()
        set_auth_enabled(True)
        try:
            with caplog.at_level(logging.WARNING, logger="src.api.auth"):
                try:
                    _get_current_user(None)
                except Exception:
                    pass
            assert any("missing" in r.message.lower() for r in caplog.records)
        finally:
            set_auth_enabled(False)

    def test_invalid_key_logs_warning_with_prefix(self, tmp_path, caplog):
        import logging
        from fastapi.security import HTTPAuthorizationCredentials
        from src.api.auth import _get_current_user, set_auth_enabled
        import src.api.auth as auth_mod
        auth_mod._users.clear()
        set_auth_enabled(True)
        try:
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="badkey123")
            with caplog.at_level(logging.WARNING, logger="src.api.auth"):
                try:
                    _get_current_user(creds)
                except Exception:
                    pass
            assert any("badkey" in r.message for r in caplog.records)
            # Full key must not appear in log
            assert not any("badkey123" in r.message for r in caplog.records)
        finally:
            set_auth_enabled(False)


class TestUser:
    def test_is_admin_true_for_admin_role(self):
        u = User("alice", "admin")
        assert u.is_admin is True

    def test_is_admin_false_for_user_role(self):
        u = User("bob", "user")
        assert u.is_admin is False

    def test_repr_contains_username(self):
        u = User("charlie", "user")
        assert "charlie" in repr(u)
