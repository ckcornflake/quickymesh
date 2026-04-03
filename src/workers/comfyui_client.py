"""
ComfyUI REST API client.

Handles workflow submission, polling, and image uploads.
Trellis2ExportMesh writes output directly to disk (not in ComfyUI history),
so callers are responsible for scanning the filesystem for output files.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

log = logging.getLogger(__name__)


class ComfyUIClient:

    def __init__(
        self,
        base_url: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_alive(self) -> bool:
        try:
            self._get("/system_stats", timeout=5.0)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    def queue_workflow(self, workflow: dict) -> str:
        """Submit a workflow and return the prompt_id."""
        # Strip any non-node entries (e.g. _comment keys)
        nodes = {
            k: v
            for k, v in workflow.items()
            if isinstance(v, dict) and "class_type" in v
        }
        payload = {"prompt": nodes, "client_id": self.client_id}
        result = self._post("/prompt", payload)
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI rejected the workflow: {result}")
        log.info(f"Queued workflow → prompt_id={prompt_id}")
        return prompt_id

    def wait_for_completion(self, prompt_id: str) -> dict:
        """
        Block until the workflow is finished.  Returns the history entry.

        Uses a generous per-request timeout (120 s) because history responses
        for large Trellis meshes can be slow to read.
        """
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                history = self._get(f"/history/{prompt_id}", timeout=120.0)
            except (urllib.error.HTTPError, TimeoutError, OSError):
                time.sleep(self.poll_interval)
                continue

            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("completed") or status.get("status_str") == "success":
                    log.info(f"Workflow {prompt_id} completed.")
                    return entry
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, list) and len(msg) > 1:
                            info = msg[1] if isinstance(msg[1], dict) else {}
                            exc = info.get("exception_message", "")
                            if "[Errno 22]" in exc:
                                raise OSError(f"ComfyUI [Errno 22] transient error: {exc}")
                    raise RuntimeError(f"Workflow {prompt_id} failed: {status}")

            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"Workflow {prompt_id} did not complete within {self.timeout}s"
        )

    def run_workflow(self, workflow: dict) -> None:
        """Queue and wait.  Output discovery is caller's responsibility."""
        prompt_id = self.queue_workflow(workflow)
        self.wait_for_completion(prompt_id)

    def run_workflow_and_get_history(self, workflow: dict) -> dict:
        """Queue, wait, and return the history entry (contains node outputs)."""
        prompt_id = self.queue_workflow(workflow)
        return self.wait_for_completion(prompt_id)

    # ------------------------------------------------------------------
    # Image download
    # ------------------------------------------------------------------

    def free_memory(self) -> None:
        """
        Ask ComfyUI to unload all models and free VRAM.

        Call this after a workflow completes so the next (possibly different)
        model can load without running out of VRAM.  Safe to call even if
        ComfyUI is not running — errors are logged and swallowed.
        """
        try:
            self._post("/free", {"unload_models": True, "free_memory": True})
            log.info("ComfyUI: models unloaded from VRAM")
        except Exception as exc:
            log.warning("ComfyUI free_memory() failed (non-fatal): %s", exc)

    def get_image(self, filename: str, subfolder: str = "", img_type: str = "output") -> bytes:
        """
        Download an image from ComfyUI via the /view endpoint.
        Used to retrieve SaveImage node outputs after a workflow completes.
        """
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": img_type,
        })
        url = f"{self.base_url}/view?{params}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to download '{filename}' from ComfyUI: {exc}"
            ) from None

    # ------------------------------------------------------------------
    # Image upload
    # ------------------------------------------------------------------

    def upload_image(self, image_path: Path | str) -> str:
        """
        Upload a local image to ComfyUI's input directory.
        Returns the server-side filename (used in workflow node inputs).
        """
        image_path = Path(image_path)
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        boundary = uuid.uuid4().hex

        with open(image_path, "rb") as f:
            file_data = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
            f"true\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            f"{self.base_url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
        except urllib.error.URLError as exc:
            _raise_connection_error(exc, self.base_url)
            raise  # unreachable — satisfies type checkers

        server_name = result.get("name", image_path.name)
        log.info(f"Uploaded {image_path.name} → server name: {server_name}")
        return server_name

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, timeout: float = 30.0) -> dict:
        url = f"{self.base_url}{path}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            _raise_connection_error(exc, self.base_url)
            raise  # unreachable — satisfies type checkers

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ComfyUI HTTP {e.code} at {path}: {body}") from None
        except urllib.error.URLError as exc:
            _raise_connection_error(exc, self.base_url)
            raise  # unreachable — satisfies type checkers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raise_connection_error(exc: urllib.error.URLError, base_url: str) -> None:
    """
    If *exc* wraps a connection-refused / network-unreachable error, raise a
    friendlier RuntimeError in its place.  Otherwise do nothing (let the
    caller re-raise the original).
    """
    reason = exc.reason
    if isinstance(reason, OSError) and reason.errno in (
        61,      # ECONNREFUSED (macOS/BSD)
        111,     # ECONNREFUSED (Linux)
        10061,   # WSAECONNREFUSED (Windows)
    ):
        raise RuntimeError(
            f"Cannot connect to ComfyUI at {base_url} — is it running?"
        ) from None
