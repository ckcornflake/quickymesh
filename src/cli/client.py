"""
QuickymeshClient — HTTP client wrapping the quickymesh API.

The CLI (and any future web frontend) talks to the server through this class.
Method names mirror the in-process ``PipelineAgent`` so call-sites translate
one-to-one when porting menu code.

Design notes
------------
- Synchronous (``httpx.Client``).  The CLI is interactive and serial; async
  buys nothing here and would force every menu function to become a coroutine.
- Auth is opt-in.  When the server runs without ``--auth-file`` (the OSS
  default) no token is required and ``api_key`` can be left as None.  When
  the server requires auth and the client gets a 401, it raises ``AuthError``
  so the menu can prompt for a key and retry.
- Asset downloads (concept art images, mesh review sheets, GLBs) are saved
  to a per-process temp dir and a ``Path`` is returned, so existing
  ``ui.show_image(path)`` plumbing keeps working without modification.
- Preferences and the API token live under
  ``~/.config/quickymesh/`` (XDG) on Linux/macOS and
  ``%APPDATA%/quickymesh/`` on Windows.

This module deliberately avoids importing from ``src.api``, ``src.agent``,
or ``src.broker`` — the client must remain runnable in environments where
none of the server-side dependencies are installed.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class QuickymeshAPIError(Exception):
    """Base error raised when the API returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str, url: str = "") -> None:
        super().__init__(f"HTTP {status_code} from {url}: {detail}")
        self.status_code = status_code
        self.detail = detail
        self.url = url


class AuthError(QuickymeshAPIError):
    """401 Unauthorized — server requires a valid API key."""


class NotFoundError(QuickymeshAPIError):
    """404 — pipeline / asset / endpoint not found."""


class ConflictError(QuickymeshAPIError):
    """409 — duplicate name, illegal transition, etc."""


class ConnectionError(Exception):
    """Server is unreachable."""


# ---------------------------------------------------------------------------
# Local-state paths (preferences, token)
# ---------------------------------------------------------------------------


def _client_state_dir() -> Path:
    """
    Return the per-user directory for client-side state (preferences, token).

    Linux/macOS: ``$XDG_CONFIG_HOME/quickymesh`` or ``~/.config/quickymesh``
    Windows:     ``%APPDATA%/quickymesh``
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "quickymesh"
        return Path.home() / "AppData" / "Roaming" / "quickymesh"
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "quickymesh"
    return Path.home() / ".config" / "quickymesh"


def _preferences_path() -> Path:
    return _client_state_dir() / "preferences.json"


def _token_path() -> Path:
    return _client_state_dir() / "token"


def load_preferences() -> dict:
    """Load client preferences (e.g. preferred concept art backend)."""
    p = _preferences_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_preferences(prefs: dict) -> None:
    p = _preferences_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prefs, indent=2), encoding="utf-8")


def load_token() -> str | None:
    p = _token_path()
    try:
        return p.read_text(encoding="utf-8").strip() or None
    except (FileNotFoundError, OSError):
        return None


def save_token(token: str) -> None:
    p = _token_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token, encoding="utf-8")
    # Best-effort restrictive permissions on POSIX.
    if sys.platform != "win32":
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass


def clear_token() -> None:
    p = _token_path()
    try:
        p.unlink()
    except (FileNotFoundError, OSError):
        pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class QuickymeshClient:
    """
    HTTP wrapper for the quickymesh API.

    Parameters
    ----------
    base_url:
        Server root URL, e.g. ``http://localhost:8000``.  ``/api/v1`` is
        appended automatically — pass only the host/port.
    api_key:
        Optional bearer token.  If None, the client will look at
        ``load_token()`` first, then fall back to no auth header.  When the
        server is running without ``--auth-file`` (the OSS default) no token
        is needed at all.
    timeout:
        Default per-request timeout in seconds.  Most endpoints enqueue work
        on the broker and return immediately, so a small timeout is fine.
        Endpoints that block the HTTP request until a GPU workflow finishes
        (ControlNet restyle, Gemini modify) pass an explicit larger timeout
        at the call site.
    """

    # Endpoints that run synchronously on the server and block the HTTP
    # request for the full ComfyUI / Gemini workflow duration.  ~5 minutes
    # is enough headroom for a Trellis-priority GPU restyle.
    LONG_SYNC_TIMEOUT = 300.0

    API_PREFIX = "/api/v1"

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key if api_key is not None else load_token()
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._http = httpx.Client(
            base_url=self._base + self.API_PREFIX,
            headers=headers,
            timeout=timeout,
        )
        self._asset_tmpdir: Path | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._http.close()
        if self._asset_tmpdir is not None:
            import shutil
            shutil.rmtree(self._asset_tmpdir, ignore_errors=True)
            self._asset_tmpdir = None

    def __enter__(self) -> "QuickymeshClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def set_api_key(self, api_key: str) -> None:
        """Update the bearer token in-place (used after an interactive login)."""
        self._api_key = api_key
        self._http.headers["Authorization"] = f"Bearer {api_key}"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        data: dict | None = None,
        files: dict | None = None,
        params: dict | None = None,
        stream: bool = False,
        timeout: float | None = None,
    ) -> httpx.Response:
        kwargs: dict[str, Any] = {
            "json": json, "data": data, "files": files, "params": params,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot reach quickymesh server at {self._base}: {e}"
            ) from e
        except httpx.RequestError as e:
            raise ConnectionError(f"HTTP request failed: {e}") from e
        self._raise_for_status(resp)
        return resp

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.is_success:
            return
        # Try to extract a JSON detail; fall back to text.
        detail: str
        try:
            payload = resp.json()
            detail = (
                payload.get("detail")
                if isinstance(payload, dict) and "detail" in payload
                else json.dumps(payload)
            )
        except (ValueError, TypeError):
            detail = resp.text or ""
        url = str(resp.request.url) if resp.request else ""
        sc = resp.status_code
        if sc == 401:
            raise AuthError(sc, detail or "Unauthorized", url)
        if sc == 404:
            raise NotFoundError(sc, detail or "Not found", url)
        if sc == 409:
            raise ConflictError(sc, detail or "Conflict", url)
        raise QuickymeshAPIError(sc, detail or resp.reason_phrase, url)

    def _asset_dir(self) -> Path:
        if self._asset_tmpdir is None:
            self._asset_tmpdir = Path(
                tempfile.mkdtemp(prefix="quickymesh-cli-")
            )
        return self._asset_tmpdir

    def _download(self, path: str, filename: str) -> Path:
        """GET an asset and save to a temp file.  Returns the local path."""
        resp = self._request("GET", path)
        target = self._asset_dir() / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resp.content)
        return target

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return self._request("GET", "/status").json()

    def get_config(self) -> dict:
        return self._request("GET", "/config").json()

    def get_pipelines_with_failures(self) -> list[str]:
        return self._request("GET", "/pipelines-with-failures").json()["pipelines"]

    # ------------------------------------------------------------------
    # 2D pipelines
    # ------------------------------------------------------------------

    def list_pipelines(self) -> list[dict]:
        return self._request("GET", "/pipelines").json()

    def get_pipeline(self, name: str) -> dict:
        return self._request("GET", f"/pipelines/{name}").json()

    def get_pipeline_or_none(self, name: str) -> dict | None:
        try:
            return self.get_pipeline(name)
        except NotFoundError:
            return None

    def get_pipeline_tasks(self, name: str) -> list[dict]:
        return self._request("GET", f"/pipelines/{name}/tasks").json()

    def create_pipeline(
        self,
        name: str,
        description: str,
        num_polys: int | None = None,
        *,
        symmetrize: bool = False,
        symmetry_axis: str = "x-",
        concept_art_backend: str = "gemini",
    ) -> dict:
        """Create a 2D pipeline without a base image (JSON variant)."""
        payload = {
            "name": name,
            "description": description,
            "num_polys": num_polys,
            "symmetrize": symmetrize,
            "symmetry_axis": symmetry_axis,
            "concept_art_backend": concept_art_backend,
        }
        return self._request("POST", "/pipelines", json=payload).json()

    def create_pipeline_from_upload(
        self,
        name: str,
        description: str,
        image_path: str | Path,
        num_polys: int | None = None,
        *,
        symmetrize: bool = False,
        symmetry_axis: str = "x-",
        concept_art_backend: str = "gemini",
    ) -> dict:
        """Create a 2D pipeline with an uploaded base image (multipart variant)."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Base image not found: {image_path}")
        data: dict[str, Any] = {
            "name": name,
            "description": description,
            "symmetrize": "true" if symmetrize else "false",
            "symmetry_axis": symmetry_axis,
            "concept_art_backend": concept_art_backend,
        }
        if num_polys is not None:
            data["num_polys"] = str(num_polys)
        with image_path.open("rb") as f:
            files = {"image": (image_path.name, f, "application/octet-stream")}
            return self._request(
                "POST", "/pipelines/from-upload", data=data, files=files
            ).json()

    def patch_pipeline(self, name: str, **fields: Any) -> dict:
        """PATCH a 2D pipeline.  Pass any of: description, num_polys,
        symmetrize, symmetry_axis, hidden."""
        body = {k: v for k, v in fields.items() if v is not None}
        return self._request("PATCH", f"/pipelines/{name}", json=body).json()

    def cancel_pipeline(self, name: str) -> None:
        self._request("DELETE", f"/pipelines/{name}")

    def retry_pipeline(self, name: str) -> int:
        resp = self._request("POST", f"/pipelines/{name}/retry").json()
        return int(resp.get("tasks_reset", 0))

    # 2D concept art review

    def get_concept_art_sheet(self, name: str) -> Path:
        return self._download(
            f"/pipelines/{name}/concept_art/sheet",
            f"{name}_concept_art_sheet.png",
        )

    def get_concept_art_image(self, name: str, idx: int) -> Path:
        return self._download(
            f"/pipelines/{name}/concept_art/{idx}",
            f"{name}_ca_{idx}.png",
        )

    def regenerate_concept_art(
        self,
        name: str,
        indices: list[int] | None = None,
        description_override: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if indices is not None:
            body["indices"] = indices
        if description_override is not None:
            body["description_override"] = description_override
        return self._request(
            "POST", f"/pipelines/{name}/concept_art/regenerate", json=body
        ).json()

    def modify_concept_art(
        self,
        name: str,
        index: int,
        instruction: str,
        *,
        source_version: int | None = None,
    ) -> dict:
        # Enqueues a broker task and returns immediately.  Poll the pipeline
        # state (concept art status flips to "regenerating") to wait for completion.
        body: dict[str, Any] = {"index": index, "instruction": instruction}
        if source_version is not None:
            body["source_version"] = source_version
        return self._request(
            "POST", f"/pipelines/{name}/concept_art/modify", json=body,
        ).json()

    def restyle_concept_art(
        self,
        name: str,
        index: int,
        positive: str,
        negative: str = "blurry, low quality, text, watermark, deformed",
        denoise: float = 0.75,
        *,
        source_version: int | None = None,
    ) -> dict:
        # Enqueues a broker task and returns immediately.  Poll the pipeline
        # state (concept art status flips to "regenerating") to wait for completion.
        body: dict[str, Any] = {
            "index": index,
            "positive": positive,
            "negative": negative,
            "denoise": denoise,
        }
        if source_version is not None:
            body["source_version"] = source_version
        return self._request(
            "POST", f"/pipelines/{name}/concept_art/restyle", json=body,
        ).json()

    # ------------------------------------------------------------------
    # 3D pipelines
    # ------------------------------------------------------------------

    def list_3d_pipelines(self) -> list[dict]:
        return self._request("GET", "/3d-pipelines").json()

    def get_3d_pipeline(self, name: str) -> dict:
        return self._request("GET", f"/3d-pipelines/{name}").json()

    def get_3d_pipeline_or_none(self, name: str) -> dict | None:
        try:
            return self.get_3d_pipeline(name)
        except NotFoundError:
            return None

    def get_3d_pipeline_tasks(self, name: str) -> list[dict]:
        return self._request("GET", f"/3d-pipelines/{name}/tasks").json()

    def create_3d_pipeline_from_ref(
        self,
        source_2d_pipeline: str,
        concept_art_index: int,
        *,
        concept_art_version: int | None = None,
        num_polys: int | None = None,
        symmetrize: bool = False,
        symmetry_axis: str = "x-",
    ) -> dict:
        body: dict[str, Any] = {
            "pipeline_name": source_2d_pipeline,
            "concept_art_index": concept_art_index,
            "symmetrize": symmetrize,
            "symmetry_axis": symmetry_axis,
        }
        if concept_art_version is not None:
            body["concept_art_version"] = concept_art_version
        if num_polys is not None:
            body["num_polys"] = num_polys
        return self._request("POST", "/3d-pipelines/from-ref", json=body).json()

    def create_3d_pipeline_from_upload(
        self,
        name: str,
        image_path: str | Path,
        num_polys: int | None = None,
        *,
        symmetrize: bool = False,
        symmetry_axis: str = "x-",
    ) -> dict:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        params: dict[str, Any] = {
            "name": name,
            "symmetrize": "true" if symmetrize else "false",
            "symmetry_axis": symmetry_axis,
        }
        if num_polys is not None:
            params["num_polys"] = num_polys
        with image_path.open("rb") as f:
            files = {"image": (image_path.name, f, "application/octet-stream")}
            return self._request(
                "POST", "/3d-pipelines/from-upload",
                files=files, params=params,
            ).json()

    def patch_3d_pipeline(self, name: str, **fields: Any) -> dict:
        body = {k: v for k, v in fields.items() if v is not None}
        return self._request("PATCH", f"/3d-pipelines/{name}", json=body).json()

    def cancel_3d_pipeline(self, name: str) -> None:
        self._request("DELETE", f"/3d-pipelines/{name}")

    def retry_3d_pipeline(self, name: str) -> int:
        resp = self._request("POST", f"/3d-pipelines/{name}/retry").json()
        return int(resp.get("tasks_reset", 0))

    # 3D mesh review

    def get_3d_review_sheet(self, name: str) -> Path:
        return self._download(
            f"/3d-pipelines/{name}/sheet",
            f"{name}_review_sheet.png",
        )

    def get_3d_screenshot(self, name: str, filename: str) -> Path:
        return self._download(
            f"/3d-pipelines/{name}/screenshot/{filename}",
            f"{name}_{filename}",
        )

    def get_3d_preview(self, name: str) -> Path:
        return self._download(
            f"/3d-pipelines/{name}/preview",
            f"{name}_preview.html",
        )

    def get_3d_mesh(self, name: str) -> Path:
        return self._download(
            f"/3d-pipelines/{name}/mesh",
            f"{name}.glb",
        )

    def approve_3d_mesh(
        self, name: str, asset_name: str | None = None,
        export_format: str | None = None,
    ) -> None:
        body: dict[str, Any] = {"asset_name": asset_name or name}
        if export_format is not None:
            body["export_format"] = export_format
        # Server runs Blender export synchronously — typically 10–30s.
        self._request(
            "POST", f"/3d-pipelines/{name}/approve", json=body,
            timeout=self.LONG_SYNC_TIMEOUT,
        )

    def reject_3d_mesh(
        self,
        name: str,
        *,
        num_polys: int | None = None,
        symmetrize: bool | None = None,
        symmetry_axis: str | None = None,
    ) -> None:
        body: dict[str, Any] = {}
        if num_polys is not None:
            body["num_polys"] = num_polys
        if symmetrize is not None:
            body["symmetrize"] = symmetrize
        if symmetry_axis is not None:
            body["symmetry_axis"] = symmetry_axis
        self._request("POST", f"/3d-pipelines/{name}/reject", json=body)
