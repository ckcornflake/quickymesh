"""
Concept art workers.

ConceptArtWorker   — abstract interface (generate images, modify an image)
GeminiConceptArtWorker — real implementation using the google-genai SDK
MockConceptArtWorker   — deterministic stub for tests (returns solid-colour PNGs)
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ConceptArtWorker(ABC):
    """
    Generates or modifies concept-art images.

    Workers are stateless — they know nothing about pipelines.  All path and
    naming decisions live in the orchestration layer (concept_art_pipeline.py).
    """

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
    """Uses the google-genai SDK to generate and modify images."""

    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types as _types

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._types = _types

    def generate_image(self, prompt: str) -> bytes:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                responseModalities=["TEXT", "IMAGE"],
            ),
        )
        return self._extract_image_bytes(response)

    def modify_image(self, image_bytes: bytes, instruction: str) -> bytes:
        contents = [
            self._types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            self._types.Part.from_text(text=instruction),
        ]
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=self._types.GenerateContentConfig(
                responseModalities=["TEXT", "IMAGE"],
            ),
        )
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
        raise RuntimeError(
            f"Gemini ({self._model}) returned no image. "
            "Check that the model supports image generation and the prompt is valid."
        )


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
