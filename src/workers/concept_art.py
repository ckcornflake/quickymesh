"""
Concept art workers.

ConceptArtWorker   — abstract interface (generate images, modify an image)
GeminiConceptArtWorker — real implementation using the google-genai SDK
MockConceptArtWorker   — deterministic stub for tests (returns solid-colour PNGs)
"""

from __future__ import annotations

import copy
import io
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

# Transient HTTP status codes worth retrying
_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_INITIAL_DELAY = 5.0   # seconds


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ConceptArtWorker(ABC):
    """
    Generates or modifies concept-art images.

    Workers are stateless — they know nothing about pipelines.  All path and
    naming decisions live in the orchestration layer (concept_art_pipeline.py).
    """

    #: Set to False on workers that cannot edit existing images (e.g. FLUX).
    supports_modify: bool = True

    @abstractmethod
    def generate_image(self, prompt: str) -> bytes:
        """
        Generate one image for `prompt`.
        Returns raw image bytes (PNG or JPEG — caller should not assume format).
        Raises RuntimeError if the API returns no image.
        """

    @abstractmethod
    def modify_image(self, image_bytes: bytes, instruction: str) -> bytes:
        """
        Edit `image_bytes` according to `instruction`.
        Returns raw image bytes for the modified image.
        Raises RuntimeError if the API returns no image.
        """



# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------


class GeminiConceptArtWorker(ConceptArtWorker):
    """
    Uses the google-genai SDK to generate and modify images.

    The genai.Client is created lazily on first use so that startup doesn't
    fail when no API key is configured (e.g. user intends to use FLUX only).
    If `api_key` is None or empty, the worker reads GEMINI_API_KEY from the
    environment at call time — allowing the CLI to set it interactively.
    """

    def __init__(self, api_key: str | None, model: str):
        self._api_key = api_key or None   # None → read from env at call time
        self._model = model
        self._client: Any = None
        self._types: Any = None

    def _ensure_client(self) -> None:
        if self._client is None:
            import os
            key = self._api_key or os.environ.get("GEMINI_API_KEY", "")
            if not key:
                raise EnvironmentError(
                    "GEMINI_API_KEY is not set. Add it to .env, export it, "
                    "or enter it when prompted."
                )
            from google import genai
            from google.genai import types as _types
            self._client = genai.Client(api_key=key)
            self._types = _types

    def generate_image(self, prompt: str) -> bytes:
        self._ensure_client()
        def _call():
            return self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    responseModalities=["TEXT", "IMAGE"],
                ),
            )
        response = _call_with_backoff(_call, context="generate_image")
        return self._extract_image_bytes(response)

    def modify_image(self, image_bytes: bytes, instruction: str) -> bytes:
        self._ensure_client()
        contents = [
            self._types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            self._types.Part.from_text(text=instruction),
        ]
        def _call():
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    responseModalities=["TEXT", "IMAGE"],
                ),
            )
        response = _call_with_backoff(_call, context="modify_image")
        return self._extract_image_bytes(response)

    def _extract_image_bytes(self, response) -> bytes:
        import base64

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data is not None:
                    data = part.inline_data.data
                    # SDK may return base64 string or raw bytes
                    if isinstance(data, str):
                        return base64.b64decode(data)
                    return data

        # No image found — collect diagnostic info to help the user understand why
        diag_parts = []
        for i, candidate in enumerate(response.candidates):
            finish = getattr(candidate, "finish_reason", None)
            if finish:
                diag_parts.append(f"candidate[{i}] finish_reason={finish}")
            safety = getattr(candidate, "safety_ratings", None)
            if safety:
                blocked = [str(r) for r in safety if getattr(r, "blocked", False)]
                if blocked:
                    diag_parts.append(f"candidate[{i}] blocked_safety={blocked}")
            # Collect any text the model returned (often a refusal explanation)
            try:
                texts = [
                    p.text for p in candidate.content.parts
                    if getattr(p, "text", None)
                ]
                if texts:
                    diag_parts.append(f"candidate[{i}] text={' | '.join(texts)!r}")
            except Exception:
                pass

        diag = "; ".join(diag_parts) if diag_parts else "no diagnostic info available"
        raise RuntimeError(
            f"Gemini ({self._model}) returned no image.\n"
            f"Diagnostic: {diag}\n"
            "Check that the model supports image generation and the prompt is not blocked."
        )


# ---------------------------------------------------------------------------
# FLUX.1 [dev] implementation via ComfyUI
# ---------------------------------------------------------------------------


class FluxComfyUIConceptArtWorker(ConceptArtWorker):
    """
    Generates concept art using FLUX.1 [dev] FP8 running in ComfyUI.

    Requires the flux1-dev-fp8.safetensors checkpoint to be installed in
    ComfyUI's models/checkpoints directory and ComfyUI to be running.

    Image modification is not supported — modify_image() raises NotImplementedError.

    Pass `arbiter` (a VRAMArbiter) to serialize this worker with Trellis/Blender
    and prevent VRAM exhaustion when multiple local-GPU workflows run concurrently.
    """

    supports_modify: bool = False

    def __init__(
        self,
        client: "Any",          # ComfyUIClient
        comfyui_output_dir: Path,
        workflow_path: Path,
        image_size: int = 1024,
        arbiter: "Any | None" = None,   # VRAMArbiter
        vram_lock_timeout: float = 1800.0,
    ):
        import json
        self._client = client
        self._output_dir = Path(comfyui_output_dir)
        self._base_workflow: dict = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
        self._image_size = image_size
        self._arbiter = arbiter
        self._vram_lock_timeout = vram_lock_timeout

    def generate_image(self, prompt: str) -> bytes:
        from contextlib import nullcontext
        job_id = f"qm_flux_{random.randint(0, 0xFFFFFFFF):08x}"
        seed = random.randint(0, 0xFFFFFFFF)
        workflow = _inject_flux_params(
            self._base_workflow,
            prompt=prompt,
            job_id=job_id,
            image_size=self._image_size,
            seed=seed,
        )
        ctx = (
            self._arbiter.acquire(timeout=self._vram_lock_timeout)
            if self._arbiter else nullcontext()
        )
        with ctx:
            self._client.free_memory()  # evict any previously loaded model (e.g. Trellis)
            history = self._client.run_workflow_and_get_history(workflow)
        return _extract_comfyui_image(history, self._client)

    def modify_image(self, image_bytes: bytes, instruction: str) -> bytes:
        raise NotImplementedError(
            "FLUX.1 [dev] does not support image modification. "
            "Use 'regenerate' to create a new image from scratch."
        )


class ControlNetRestyleWorker:
    """
    Restyles an existing concept art image using ControlNet Canny + SDXL.

    Preserves the silhouette of the input image while applying a new visual
    style described by a positive/negative prompt pair and a denoise strength.

    This worker is NOT a ConceptArtWorker subclass — it is passed separately
    as `restyle_worker` and used only during the interactive concept-art review.
    It is never used for generation or text-based modification.

    Pass `arbiter` (a VRAMArbiter) to serialize this with Trellis/Blender/FLUX
    and prevent VRAM exhaustion when multiple local-GPU workflows run concurrently.
    """

    def __init__(
        self,
        client: "Any",          # ComfyUIClient
        workflow_path: Path,
        arbiter: "Any | None" = None,   # VRAMArbiter
        vram_lock_timeout: float = 1800.0,
    ):
        import json
        self._client = client
        self._base_workflow: dict = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
        self._arbiter = arbiter
        self._vram_lock_timeout = vram_lock_timeout

    def restyle_image(
        self,
        image_bytes: bytes,
        positive: str,
        negative: str,
        denoise: float,
    ) -> bytes:
        """
        Restyle `image_bytes` using ControlNet Canny.

        Parameters
        ----------
        image_bytes:  Raw PNG/JPEG bytes of the source image.
        positive:     Comma-separated style keywords (e.g. "zerg, biomechanical, dark chitin").
        negative:     Comma-separated unwanted elements (e.g. "weapons, people, blurry").
        denoise:      0.1 (very faithful to original shape) → 1.0 (maximum creativity).
        """
        import io as _io
        import tempfile
        from contextlib import nullcontext

        # Upload source image to ComfyUI input directory
        img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_path = Path(tf.name)
            img.save(tmp_path, format="PNG")
        try:
            server_image = self._client.upload_image(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        job_id = f"qm_restyle_{random.randint(0, 0xFFFFFFFF):08x}"
        seed = random.randint(0, 0xFFFFFFFF)
        workflow = _inject_restyle_params(
            self._base_workflow,
            image_name=server_image,
            positive=positive,
            negative=negative,
            denoise=denoise,
            job_id=job_id,
            seed=seed,
        )

        ctx = (
            self._arbiter.acquire(timeout=self._vram_lock_timeout)
            if self._arbiter else nullcontext()
        )
        with ctx:
            history = self._client.run_workflow_and_get_history(workflow)
            self._client.free_memory()

        return _extract_comfyui_image(history, self._client)


def _inject_restyle_params(
    base_workflow: dict,
    *,
    image_name: str,
    positive: str,
    negative: str,
    denoise: float,
    job_id: str,
    seed: int,
) -> dict:
    """Deep-copy the ControlNet restyle workflow and inject runtime values."""
    wf = copy.deepcopy(base_workflow)
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inp = node.get("inputs", {})
        if ct == "LoadImage":
            inp["image"] = image_name
        elif ct == "CLIPTextEncode" and inp.get("text") == "__POSITIVE__":
            inp["text"] = positive
        elif ct == "CLIPTextEncode" and inp.get("text") == "__NEGATIVE__":
            inp["text"] = negative
        elif ct == "KSampler":
            inp["seed"] = seed
            inp["denoise"] = denoise
            inp["control_after_generate"] = "fixed"
        elif ct == "SaveImage":
            inp["filename_prefix"] = job_id
    return wf


def _inject_flux_params(
    base_workflow: dict,
    *,
    prompt: str,
    job_id: str,
    image_size: int,
    seed: int,
) -> dict:
    """Deep-copy the workflow and inject runtime values."""
    wf = copy.deepcopy(base_workflow)
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inp = node.get("inputs", {})
        if ct == "CLIPTextEncode" and inp.get("text") == "__PROMPT__":
            inp["text"] = prompt
        elif ct == "EmptySD3LatentImage":
            inp["width"] = image_size
            inp["height"] = image_size
        elif ct == "KSampler":
            inp["seed"] = seed
            inp["control_after_generate"] = "fixed"
        elif ct == "SaveImage":
            inp["filename_prefix"] = job_id
    return wf


def _extract_comfyui_image(history: dict, client: "Any") -> bytes:
    """Pull the first image from a completed FLUX workflow history entry."""
    for node_output in history.get("outputs", {}).values():
        images = node_output.get("images", [])
        if images:
            img = images[0]
            return client.get_image(
                img["filename"],
                img.get("subfolder", ""),
                img.get("type", "output"),
            )
    raise RuntimeError(
        "FLUX workflow completed but produced no images. "
        "Check that the workflow ran correctly in ComfyUI."
    )


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _call_with_backoff(fn, *, context: str = "Gemini API call"):
    """
    Call `fn()` with exponential backoff on transient errors.

    Retries up to _MAX_RETRIES times.  Delay starts at _INITIAL_DELAY seconds
    and doubles on each retry.  Logs a warning before each wait so the user
    can see what's happening in the console.

    Raises the last exception if all retries are exhausted.
    """
    delay = _INITIAL_DELAY
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 2):  # +2 so last attempt can still run
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt > _MAX_RETRIES:
                break
            # Determine if it's worth retrying
            code = _http_code(exc)
            if code is not None and code not in _RETRYABLE_CODES:
                raise  # non-transient — fail immediately
            log.warning(
                "[%s] attempt %d/%d failed: %s — retrying in %.0fs ...",
                context, attempt, _MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
            delay *= 2
    raise last_exc  # type: ignore[misc]


def _http_code(exc: Exception) -> int | None:
    """Extract an HTTP status code from a google-genai exception, if present."""
    # google-genai raises APIError with a .code attribute
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    # Also check status attribute (some versions use .status)
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    return None


# ---------------------------------------------------------------------------
# Mock implementation (for tests)
# ---------------------------------------------------------------------------


class MockConceptArtWorker(ConceptArtWorker):
    """
    Deterministic stub that returns solid-colour PNG images instantly.

    Parameters
    ----------
    colors:
        Cycle of RGB colours to use for successive generate_image() calls.
        Defaults to a fixed palette of 8 distinct colours.
    modify_color:
        RGB colour used for images returned by modify_image().
    fail_on_generate:
        If True, generate_image() raises RuntimeError (simulates API failure).
    fail_on_modify:
        If True, modify_image() raises RuntimeError.
    image_size:
        Side length (pixels) of the generated square PNG.
    """

    _DEFAULT_COLORS: list[tuple[int, int, int]] = [
        (220, 80,  80),   # red
        (80,  180, 80),   # green
        (80,  80,  220),  # blue
        (220, 180, 80),   # yellow
        (180, 80,  220),  # purple
        (80,  200, 200),  # cyan
        (220, 130, 80),   # orange
        (130, 220, 130),  # light green
    ]

    def __init__(
        self,
        colors: list[tuple[int, int, int]] | None = None,
        modify_color: tuple[int, int, int] = (180, 180, 50),
        fail_on_generate: bool = False,
        fail_on_modify: bool = False,
        image_size: int = 64,
    ):
        self._colors = colors or self._DEFAULT_COLORS
        self._modify_color = modify_color
        self._fail_on_generate = fail_on_generate
        self._fail_on_modify = fail_on_modify
        self._image_size = image_size
        self._generate_call_count = 0
        self.generate_prompts: list[str] = []
        self.modify_instructions: list[str] = []

    def generate_image(self, prompt: str) -> bytes:
        if self._fail_on_generate:
            raise RuntimeError("MockConceptArtWorker: simulated generate failure")
        self.generate_prompts.append(prompt)
        color = self._colors[self._generate_call_count % len(self._colors)]
        self._generate_call_count += 1
        return self._make_png(color)

    def modify_image(self, image_bytes: bytes, instruction: str) -> bytes:
        if self._fail_on_modify:
            raise RuntimeError("MockConceptArtWorker: simulated modify failure")
        self.modify_instructions.append(instruction)
        return self._make_png(self._modify_color)

    def _make_png(self, color: tuple[int, int, int]) -> bytes:
        img = Image.new("RGB", (self._image_size, self._image_size), color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
