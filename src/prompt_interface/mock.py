"""
Mock prompt interface for automated testing and agent-driven pipelines.

Pre-program responses via the `responses` queue.  Each call to `ask()`
pops the next response; if the queue is empty it raises `StopIteration`.
The `inform()` and `show_image()` calls are recorded for later assertion.
"""

from pathlib import Path
from collections import deque

from .base import PromptInterface


class MockPromptInterface(PromptInterface):
    """
    Deterministic prompt interface for tests.

    Usage:
        ui = MockPromptInterface(["approve 1 2", "quit"])
        response = ui.ask("Choose an action")   # returns "approve 1 2"
    """

    def __init__(self, responses: list[str] | None = None):
        self._responses: deque[str] = deque(responses or [])
        self.messages: list[str] = []      # recorded inform() calls
        self.shown_images: list[Path] = [] # recorded show_image() calls
        self.asked: list[str] = []         # recorded ask() prompts

    def queue(self, *responses: str) -> None:
        """Add more canned responses at the end of the queue."""
        self._responses.extend(responses)

    def ask(self, message: str, *, options: list[str] | None = None) -> str:
        self.asked.append(message)
        if not self._responses:
            raise StopIteration(
                f"MockPromptInterface ran out of responses. Last prompt was: {message!r}"
            )
        response = self._responses.popleft()
        # Validate against options if provided (helps catch bad test data early)
        if options and response not in options:
            # Allow numeric shortcuts so tests can pass "1" instead of options[0]
            if response.isdigit() and 1 <= int(response) <= len(options):
                return options[int(response) - 1]
        return response

    def inform(self, message: str) -> None:
        self.messages.append(message)

    def show_image(self, path: Path) -> None:
        self.shown_images.append(Path(path))
