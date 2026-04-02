"""
Tests for src/workers/concept_art.py — MockConceptArtWorker and
GeminiConceptArtWorker (via a patched client).
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.workers.concept_art import GeminiConceptArtWorker, MockConceptArtWorker


# ---------------------------------------------------------------------------
# MockConceptArtWorker
# ---------------------------------------------------------------------------


class TestMockConceptArtWorker:
    def test_generate_returns_png_bytes(self):
        worker = MockConceptArtWorker()
        data = worker.generate_image("a dragon")
        img = Image.open(io.BytesIO(data))
        assert img.format == "PNG"

    def test_generate_cycles_colors(self):
        colors = [(255, 0, 0), (0, 255, 0)]
        worker = MockConceptArtWorker(colors=colors)
        d1 = worker.generate_image("a")
        d2 = worker.generate_image("b")
        d3 = worker.generate_image("c")  # wraps back to colors[0]
        c1 = Image.open(io.BytesIO(d1)).getpixel((0, 0))
        c2 = Image.open(io.BytesIO(d2)).getpixel((0, 0))
        c3 = Image.open(io.BytesIO(d3)).getpixel((0, 0))
        assert c1 == (255, 0, 0)
        assert c2 == (0, 255, 0)
        assert c3 == (255, 0, 0)

    def test_generate_records_prompt(self):
        worker = MockConceptArtWorker()
        worker.generate_image("a dragon on a white background")
        assert worker.generate_prompts == ["a dragon on a white background"]

    def test_generate_count_increments(self):
        worker = MockConceptArtWorker()
        for _ in range(3):
            worker.generate_image("prompt")
        assert worker._generate_call_count == 3

    def test_modify_returns_png_bytes(self):
        worker = MockConceptArtWorker()
        original = worker.generate_image("dragon")
        modified = worker.modify_image(original, "make it red")
        img = Image.open(io.BytesIO(modified))
        assert img.format == "PNG"

    def test_modify_records_instruction(self):
        worker = MockConceptArtWorker()
        original = worker.generate_image("dragon")
        worker.modify_image(original, "remove the wings")
        assert worker.modify_instructions == ["remove the wings"]

    def test_modify_uses_modify_color(self):
        worker = MockConceptArtWorker(modify_color=(10, 20, 30))
        original = worker.generate_image("x")
        modified = worker.modify_image(original, "change it")
        color = Image.open(io.BytesIO(modified)).getpixel((0, 0))
        assert color == (10, 20, 30)

    def test_fail_on_generate(self):
        worker = MockConceptArtWorker(fail_on_generate=True)
        with pytest.raises(RuntimeError, match="simulated generate failure"):
            worker.generate_image("test")

    def test_fail_on_modify(self):
        worker = MockConceptArtWorker()
        original = worker.generate_image("x")
        worker._fail_on_modify = True
        with pytest.raises(RuntimeError, match="simulated modify failure"):
            worker.modify_image(original, "change")


# ---------------------------------------------------------------------------
# GeminiConceptArtWorker — unit tests with mocked client
# ---------------------------------------------------------------------------


def _make_fake_response(image_bytes: bytes):
    """Build a fake google-genai response containing one image part."""
    part = MagicMock()
    part.inline_data = MagicMock()
    part.inline_data.data = image_bytes
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    return response


def _dummy_png(color=(100, 150, 200)) -> bytes:
    img = Image.new("RGB", (32, 32), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestGeminiConceptArtWorker:
    def _make_worker(self, mock_client):
        with patch("src.workers.concept_art.GeminiConceptArtWorker.__init__",
                   lambda self, api_key, model: None):
            worker = GeminiConceptArtWorker.__new__(GeminiConceptArtWorker)
        worker._client = mock_client
        worker._model = "test-model"
        from google.genai import types
        worker._types = types
        return worker

    def test_generate_image_calls_api_with_prompt(self):
        png = _dummy_png()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_fake_response(png)
        worker = self._make_worker(mock_client)

        result = worker.generate_image("a dragon")

        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["contents"] == "a dragon"
        assert call_kwargs.kwargs["model"] == "test-model"

    def test_generate_image_returns_bytes(self):
        png = _dummy_png()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_fake_response(png)
        worker = self._make_worker(mock_client)

        result = worker.generate_image("prompt")
        assert result == png

    def test_generate_image_raises_if_no_image_in_response(self):
        # Response with no inline_data
        part = MagicMock()
        part.inline_data = None
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        worker = self._make_worker(mock_client)

        with pytest.raises(RuntimeError, match="no image"):
            worker.generate_image("prompt")

    def test_generate_image_decodes_base64_string(self):
        import base64
        png = _dummy_png()
        b64 = base64.b64encode(png).decode()

        part = MagicMock()
        part.inline_data = MagicMock()
        part.inline_data.data = b64  # string, not bytes
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        worker = self._make_worker(mock_client)

        result = worker.generate_image("prompt")
        assert result == png

    def test_modify_image_includes_image_part(self):
        png = _dummy_png()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_fake_response(png)
        worker = self._make_worker(mock_client)

        worker.modify_image(png, "make it blue")

        call_kwargs = mock_client.models.generate_content.call_args
        contents = call_kwargs.kwargs["contents"]
        assert isinstance(contents, list)
        assert len(contents) == 2
