"""
Tests for src/cli/client.py — QuickymeshClient HTTP wrapper.

The client is exercised against an in-memory ``httpx.MockTransport`` so the
real network and the real API server are never touched.  Each test installs
a small handler that asserts the outgoing request shape (method, path, body)
and returns a canned response.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.cli import client as client_mod
from src.cli.client import (
    AuthError,
    ConflictError,
    ConnectionError as QmConnectionError,
    NotFoundError,
    QuickymeshAPIError,
    QuickymeshClient,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(handler) -> QuickymeshClient:
    """Build a QuickymeshClient whose underlying httpx.Client uses MockTransport."""
    c = QuickymeshClient(base_url="http://test.local:8000", api_key="testkey")
    # Replace the real http client with a mock-transport one that preserves
    # the same base_url + headers contract.
    c._http.close()
    c._http = httpx.Client(
        base_url="http://test.local:8000" + QuickymeshClient.API_PREFIX,
        headers={"Authorization": "Bearer testkey"},
        transport=httpx.MockTransport(handler),
    )
    return c


def _json_response(payload, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


# ---------------------------------------------------------------------------
# Local-state helpers (preferences, token)
# ---------------------------------------------------------------------------


class TestClientStateDir:
    def test_dir_under_appdata_on_windows(self, monkeypatch):
        monkeypatch.setattr(client_mod.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", "C:/fake/appdata")
        d = client_mod._client_state_dir()
        assert d == Path("C:/fake/appdata") / "quickymesh"

    def test_dir_under_xdg_on_linux(self, monkeypatch, tmp_path):
        monkeypatch.setattr(client_mod.sys, "platform", "linux")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        d = client_mod._client_state_dir()
        assert d == tmp_path / "quickymesh"


class TestPreferences:
    def test_load_missing_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(client_mod, "_preferences_path", lambda: tmp_path / "p.json")
        assert client_mod.load_preferences() == {}

    def test_save_then_load_roundtrip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(client_mod, "_preferences_path", lambda: tmp_path / "p.json")
        client_mod.save_preferences({"backend": "gemini", "n": 4})
        assert client_mod.load_preferences() == {"backend": "gemini", "n": 4}

    def test_load_handles_corrupt_json(self, monkeypatch, tmp_path):
        p = tmp_path / "p.json"
        p.write_text("{not json")
        monkeypatch.setattr(client_mod, "_preferences_path", lambda: p)
        assert client_mod.load_preferences() == {}


class TestToken:
    def test_load_missing_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(client_mod, "_token_path", lambda: tmp_path / "tok")
        assert client_mod.load_token() is None

    def test_save_then_load(self, monkeypatch, tmp_path):
        monkeypatch.setattr(client_mod, "_token_path", lambda: tmp_path / "tok")
        client_mod.save_token("abc123")
        assert client_mod.load_token() == "abc123"

    def test_clear_removes_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(client_mod, "_token_path", lambda: tmp_path / "tok")
        client_mod.save_token("abc")
        client_mod.clear_token()
        assert client_mod.load_token() is None

    def test_clear_missing_is_silent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(client_mod, "_token_path", lambda: tmp_path / "nope")
        client_mod.clear_token()  # must not raise


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_uses_explicit_api_key(self, monkeypatch):
        monkeypatch.setattr(client_mod, "load_token", lambda: None)
        c = QuickymeshClient(base_url="http://x", api_key="mykey")
        try:
            assert c._http.headers["Authorization"] == "Bearer mykey"
        finally:
            c.close()

    def test_falls_back_to_load_token(self, monkeypatch):
        monkeypatch.setattr(client_mod, "load_token", lambda: "from-disk")
        c = QuickymeshClient(base_url="http://x")
        try:
            assert c._http.headers["Authorization"] == "Bearer from-disk"
        finally:
            c.close()

    def test_no_token_no_auth_header(self, monkeypatch):
        monkeypatch.setattr(client_mod, "load_token", lambda: None)
        c = QuickymeshClient(base_url="http://x")
        try:
            assert "Authorization" not in c._http.headers
        finally:
            c.close()

    def test_api_prefix_appended(self, monkeypatch):
        monkeypatch.setattr(client_mod, "load_token", lambda: None)
        c = QuickymeshClient(base_url="http://x:1234/")
        try:
            # base_url is normalized, trailing slash stripped
            assert str(c._http.base_url).rstrip("/") == "http://x:1234/api/v1"
        finally:
            c.close()

    def test_set_api_key_updates_header(self, monkeypatch):
        monkeypatch.setattr(client_mod, "load_token", lambda: None)
        c = QuickymeshClient(base_url="http://x")
        try:
            c.set_api_key("new")
            assert c._http.headers["Authorization"] == "Bearer new"
        finally:
            c.close()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    def test_401_raises_auth_error(self):
        def handler(req):
            return _json_response({"detail": "bad key"}, status_code=401)
        c = _make_client(handler)
        try:
            with pytest.raises(AuthError) as exc:
                c.get_status()
            assert exc.value.status_code == 401
            assert "bad key" in exc.value.detail
        finally:
            c.close()

    def test_404_raises_not_found(self):
        def handler(req):
            return _json_response({"detail": "missing"}, status_code=404)
        c = _make_client(handler)
        try:
            with pytest.raises(NotFoundError):
                c.get_pipeline("nope")
        finally:
            c.close()

    def test_409_raises_conflict(self):
        def handler(req):
            return _json_response({"detail": "exists"}, status_code=409)
        c = _make_client(handler)
        try:
            with pytest.raises(ConflictError):
                c.create_pipeline("dup", "desc")
        finally:
            c.close()

    def test_500_raises_generic_api_error(self):
        def handler(req):
            return httpx.Response(500, text="boom")
        c = _make_client(handler)
        try:
            with pytest.raises(QuickymeshAPIError) as exc:
                c.get_status()
            assert exc.value.status_code == 500
        finally:
            c.close()

    def test_connect_error_wrapped(self):
        def handler(req):
            raise httpx.ConnectError("refused", request=req)
        c = _make_client(handler)
        try:
            with pytest.raises(QmConnectionError):
                c.get_status()
        finally:
            c.close()

    def test_get_pipeline_or_none_swallows_404(self):
        def handler(req):
            return _json_response({"detail": "missing"}, status_code=404)
        c = _make_client(handler)
        try:
            assert c.get_pipeline_or_none("nope") is None
        finally:
            c.close()

    def test_get_3d_pipeline_or_none_swallows_404(self):
        def handler(req):
            return _json_response({"detail": "missing"}, status_code=404)
        c = _make_client(handler)
        try:
            assert c.get_3d_pipeline_or_none("nope") is None
        finally:
            c.close()


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------


class TestSystemEndpoints:
    def test_get_status(self):
        seen = {}
        def handler(req):
            seen["url"] = str(req.url)
            seen["method"] = req.method
            return _json_response({"status": "ok"})
        c = _make_client(handler)
        try:
            assert c.get_status() == {"status": "ok"}
            assert seen["url"].endswith("/api/v1/status")
            assert seen["method"] == "GET"
        finally:
            c.close()

    def test_get_config(self):
        def handler(req):
            assert req.url.path == "/api/v1/config"
            return _json_response({"output_root": "/tmp", "gemini_api_key_present": True})
        c = _make_client(handler)
        try:
            cfg = c.get_config()
            assert cfg["output_root"] == "/tmp"
            assert cfg["gemini_api_key_present"] is True
        finally:
            c.close()

    def test_get_pipelines_with_failures(self):
        def handler(req):
            assert req.url.path == "/api/v1/pipelines-with-failures"
            return _json_response({"pipelines": ["a", "b"]})
        c = _make_client(handler)
        try:
            assert c.get_pipelines_with_failures() == ["a", "b"]
        finally:
            c.close()


# ---------------------------------------------------------------------------
# 2D pipeline endpoints
# ---------------------------------------------------------------------------


class TestPipelinesCRUD:
    def test_list_pipelines(self):
        def handler(req):
            assert req.method == "GET"
            assert req.url.path == "/api/v1/pipelines"
            return _json_response([{"name": "p1"}])
        c = _make_client(handler)
        try:
            assert c.list_pipelines() == [{"name": "p1"}]
        finally:
            c.close()

    def test_get_pipeline(self):
        def handler(req):
            assert req.url.path == "/api/v1/pipelines/foo"
            return _json_response({"name": "foo", "status": "initializing"})
        c = _make_client(handler)
        try:
            assert c.get_pipeline("foo")["name"] == "foo"
        finally:
            c.close()

    def test_get_pipeline_tasks(self):
        def handler(req):
            assert req.url.path == "/api/v1/pipelines/foo/tasks"
            return _json_response([{"id": 1, "status": "done"}])
        c = _make_client(handler)
        try:
            assert c.get_pipeline_tasks("foo") == [{"id": 1, "status": "done"}]
        finally:
            c.close()

    def test_create_pipeline_sends_full_body(self):
        seen = {}
        def handler(req):
            assert req.method == "POST"
            assert req.url.path == "/api/v1/pipelines"
            seen["body"] = json.loads(req.content)
            return _json_response({"name": "foo"}, status_code=201)
        c = _make_client(handler)
        try:
            c.create_pipeline("foo", "a robot", num_polys=5000,
                              symmetrize=True, symmetry_axis="y-",
                              concept_art_backend="flux")
            assert seen["body"] == {
                "name": "foo",
                "description": "a robot",
                "num_polys": 5000,
                "symmetrize": True,
                "symmetry_axis": "y-",
                "concept_art_backend": "flux",
            }
        finally:
            c.close()

    def test_create_pipeline_from_upload(self, tmp_path):
        img = tmp_path / "base.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        seen = {}
        def handler(req):
            assert req.method == "POST"
            assert req.url.path == "/api/v1/pipelines/from-upload"
            seen["content_type"] = req.headers.get("content-type", "")
            seen["body"] = req.content
            return _json_response({"name": "foo"}, status_code=201)
        c = _make_client(handler)
        try:
            c.create_pipeline_from_upload(
                "foo", "a robot", img, num_polys=5000,
                symmetrize=True, symmetry_axis="y-",
                concept_art_backend="flux",
            )
            assert "multipart/form-data" in seen["content_type"]
            # form fields and the file bytes both end up in the body
            assert b"a robot" in seen["body"]
            assert b"\x89PNG" in seen["body"]
            assert b"base.png" in seen["body"]
        finally:
            c.close()

    def test_create_pipeline_from_upload_missing_image_raises(self, tmp_path):
        def handler(req):
            return _json_response({}, status_code=201)
        c = _make_client(handler)
        try:
            with pytest.raises(FileNotFoundError):
                c.create_pipeline_from_upload("foo", "x", tmp_path / "nope.png")
        finally:
            c.close()

    def test_patch_pipeline_drops_none_fields(self):
        seen = {}
        def handler(req):
            seen["body"] = json.loads(req.content)
            return _json_response({"name": "foo"})
        c = _make_client(handler)
        try:
            c.patch_pipeline("foo", description="new", num_polys=None, hidden=True)
            assert seen["body"] == {"description": "new", "hidden": True}
        finally:
            c.close()

    def test_cancel_pipeline(self):
        seen = {}
        def handler(req):
            seen["method"] = req.method
            seen["url"] = req.url.path
            return httpx.Response(200, json={"status": "ok"})
        c = _make_client(handler)
        try:
            c.cancel_pipeline("foo")
            assert seen["method"] == "DELETE"
            assert seen["url"] == "/api/v1/pipelines/foo"
        finally:
            c.close()

    def test_retry_pipeline_returns_count(self):
        def handler(req):
            assert req.url.path == "/api/v1/pipelines/foo/retry"
            return _json_response({"status": "ok", "tasks_reset": 3})
        c = _make_client(handler)
        try:
            assert c.retry_pipeline("foo") == 3
        finally:
            c.close()


class TestConceptArtReview:
    def test_get_concept_art_sheet_downloads_to_temp(self):
        def handler(req):
            assert req.url.path == "/api/v1/pipelines/foo/concept_art/sheet"
            return httpx.Response(200, content=b"PNGBYTES")
        c = _make_client(handler)
        try:
            p = c.get_concept_art_sheet("foo")
            assert p.exists()
            assert p.read_bytes() == b"PNGBYTES"
            assert p.name == "foo_concept_art_sheet.png"
        finally:
            c.close()

    def test_get_concept_art_image(self):
        def handler(req):
            assert req.url.path == "/api/v1/pipelines/foo/concept_art/2"
            return httpx.Response(200, content=b"img")
        c = _make_client(handler)
        try:
            p = c.get_concept_art_image("foo", 2)
            assert p.read_bytes() == b"img"
            assert p.name == "foo_ca_2.png"
        finally:
            c.close()

    def test_regenerate_with_indices_and_override(self):
        seen = {}
        def handler(req):
            seen["body"] = json.loads(req.content)
            return _json_response({"status": "ok"})
        c = _make_client(handler)
        try:
            c.regenerate_concept_art("foo", indices=[0, 2], description_override="new desc")
            assert seen["body"] == {"indices": [0, 2], "description_override": "new desc"}
        finally:
            c.close()

    def test_regenerate_without_args_sends_empty_body(self):
        seen = {}
        def handler(req):
            seen["body"] = json.loads(req.content)
            return _json_response({"status": "ok"})
        c = _make_client(handler)
        try:
            c.regenerate_concept_art("foo")
            assert seen["body"] == {}
        finally:
            c.close()

    def test_modify_concept_art(self):
        seen = {}
        def handler(req):
            seen["body"] = json.loads(req.content)
            # Server returns AcceptedResponse — the request enqueues a broker
            # task and returns immediately; caller polls the pipeline state.
            return _json_response({"status": "accepted", "message": "Modifying image 2"})
        c = _make_client(handler)
        try:
            result = c.modify_concept_art("foo", 1, "make it red")
            assert seen["body"] == {"index": 1, "instruction": "make it red"}
            assert result["status"] == "accepted"
        finally:
            c.close()

    def test_restyle_concept_art_passes_defaults(self):
        seen = {}
        def handler(req):
            seen["body"] = json.loads(req.content)
            return _json_response({"status": "accepted", "message": "Restyling image 1"})
        c = _make_client(handler)
        try:
            result = c.restyle_concept_art("foo", 0, "anime style")
            assert seen["body"]["index"] == 0
            assert seen["body"]["positive"] == "anime style"
            assert seen["body"]["denoise"] == 0.75
            assert "blurry" in seen["body"]["negative"]
            assert result["status"] == "accepted"
        finally:
            c.close()


# ---------------------------------------------------------------------------
# 3D pipelines
# ---------------------------------------------------------------------------


class Test3DPipelines:
    def test_list_and_get(self):
        def handler(req):
            if req.url.path == "/api/v1/3d-pipelines":
                return _json_response([{"name": "p1"}])
            if req.url.path == "/api/v1/3d-pipelines/p1":
                return _json_response({"name": "p1"})
            return httpx.Response(404)
        c = _make_client(handler)
        try:
            assert c.list_3d_pipelines() == [{"name": "p1"}]
            assert c.get_3d_pipeline("p1") == {"name": "p1"}
        finally:
            c.close()

    def test_get_3d_tasks(self):
        def handler(req):
            assert req.url.path == "/api/v1/3d-pipelines/p1/tasks"
            return _json_response([])
        c = _make_client(handler)
        try:
            assert c.get_3d_pipeline_tasks("p1") == []
        finally:
            c.close()

    def test_create_3d_from_ref(self):
        seen = {}
        def handler(req):
            assert req.url.path == "/api/v1/3d-pipelines/from-ref"
            seen["body"] = json.loads(req.content)
            return _json_response({"name": "p1"}, status_code=201)
        c = _make_client(handler)
        try:
            c.create_3d_pipeline_from_ref(
                "src2d", 1, concept_art_version=2, num_polys=8000,
                symmetrize=True, symmetry_axis="x+",
            )
            assert seen["body"] == {
                "pipeline_name": "src2d",
                "concept_art_index": 1,
                "concept_art_version": 2,
                "num_polys": 8000,
                "symmetrize": True,
                "symmetry_axis": "x+",
            }
        finally:
            c.close()

    def test_create_3d_from_upload(self, tmp_path):
        img = tmp_path / "i.png"
        img.write_bytes(b"\x89PNGfake")
        seen = {}
        def handler(req):
            seen["url"] = str(req.url)
            seen["body"] = req.content
            seen["content_type"] = req.headers.get("content-type", "")
            return _json_response({"name": "foo"}, status_code=201)
        c = _make_client(handler)
        try:
            c.create_3d_pipeline_from_upload(
                "foo", img, num_polys=4000, symmetrize=False,
            )
            assert "/api/v1/3d-pipelines/from-upload" in seen["url"]
            assert "name=foo" in seen["url"]
            assert "num_polys=4000" in seen["url"]
            assert "multipart/form-data" in seen["content_type"]
            assert b"\x89PNG" in seen["body"]
        finally:
            c.close()

    def test_patch_3d_drops_nones(self):
        seen = {}
        def handler(req):
            seen["body"] = json.loads(req.content)
            return _json_response({"name": "p1"})
        c = _make_client(handler)
        try:
            c.patch_3d_pipeline("p1", hidden=True, num_polys=None)
            assert seen["body"] == {"hidden": True}
        finally:
            c.close()

    def test_cancel_3d(self):
        seen = {}
        def handler(req):
            seen["method"] = req.method
            return httpx.Response(200, json={"status": "ok"})
        c = _make_client(handler)
        try:
            c.cancel_3d_pipeline("p1")
            assert seen["method"] == "DELETE"
        finally:
            c.close()

    def test_retry_3d(self):
        def handler(req):
            return _json_response({"status": "ok", "tasks_reset": 7})
        c = _make_client(handler)
        try:
            assert c.retry_3d_pipeline("p1") == 7
        finally:
            c.close()


class Test3DReview:
    def test_review_sheet_download(self):
        def handler(req):
            assert req.url.path == "/api/v1/3d-pipelines/p1/sheet"
            return httpx.Response(200, content=b"sheet")
        c = _make_client(handler)
        try:
            p = c.get_3d_review_sheet("p1")
            assert p.read_bytes() == b"sheet"
            assert p.name == "p1_review_sheet.png"
        finally:
            c.close()

    def test_screenshot_download(self):
        def handler(req):
            assert req.url.path == "/api/v1/3d-pipelines/p1/screenshot/front.png"
            return httpx.Response(200, content=b"shot")
        c = _make_client(handler)
        try:
            p = c.get_3d_screenshot("p1", "front.png")
            assert p.read_bytes() == b"shot"
            assert p.name == "p1_front.png"
        finally:
            c.close()

    def test_preview_download(self):
        def handler(req):
            assert req.url.path == "/api/v1/3d-pipelines/p1/preview"
            return httpx.Response(200, content=b"<html/>")
        c = _make_client(handler)
        try:
            p = c.get_3d_preview("p1")
            assert p.suffix == ".html"
        finally:
            c.close()

    def test_mesh_download(self):
        def handler(req):
            assert req.url.path == "/api/v1/3d-pipelines/p1/mesh"
            return httpx.Response(200, content=b"glb-bytes")
        c = _make_client(handler)
        try:
            p = c.get_3d_mesh("p1")
            assert p.suffix == ".glb"
            assert p.read_bytes() == b"glb-bytes"
        finally:
            c.close()

    def test_approve_with_defaults(self):
        seen = {}
        def handler(req):
            seen["body"] = json.loads(req.content)
            return _json_response({"status": "ok"})
        c = _make_client(handler)
        try:
            c.approve_3d_mesh("p1")
            assert seen["body"] == {"asset_name": "p1"}
        finally:
            c.close()

    def test_approve_with_explicit_asset_and_format(self):
        seen = {}
        def handler(req):
            seen["body"] = json.loads(req.content)
            return _json_response({"status": "ok"})
        c = _make_client(handler)
        try:
            c.approve_3d_mesh("p1", asset_name="hero", export_format="fbx")
            assert seen["body"] == {"asset_name": "hero", "export_format": "fbx"}
        finally:
            c.close()

    def test_reject_with_overrides(self):
        seen = {}
        def handler(req):
            seen["body"] = json.loads(req.content)
            return _json_response({"status": "ok"})
        c = _make_client(handler)
        try:
            c.reject_3d_mesh("p1", num_polys=2000, symmetrize=True, symmetry_axis="z-")
            assert seen["body"] == {
                "num_polys": 2000,
                "symmetrize": True,
                "symmetry_axis": "z-",
            }
        finally:
            c.close()

    def test_reject_empty_body(self):
        seen = {}
        def handler(req):
            seen["body"] = json.loads(req.content) if req.content else {}
            return _json_response({"status": "ok"})
        c = _make_client(handler)
        try:
            c.reject_3d_mesh("p1")
            assert seen["body"] == {}
        finally:
            c.close()


# ---------------------------------------------------------------------------
# Asset tempdir cleanup
# ---------------------------------------------------------------------------


class TestAssetTempdir:
    def test_close_removes_asset_dir(self):
        def handler(req):
            return httpx.Response(200, content=b"x")
        c = _make_client(handler)
        c.get_concept_art_sheet("foo")
        d = c._asset_tmpdir
        assert d is not None and d.exists()
        c.close()
        assert not d.exists()

    def test_assets_share_one_tempdir(self):
        def handler(req):
            return httpx.Response(200, content=b"x")
        c = _make_client(handler)
        try:
            p1 = c.get_concept_art_sheet("foo")
            p2 = c.get_concept_art_image("foo", 0)
            assert p1.parent == p2.parent
        finally:
            c.close()
