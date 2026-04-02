"""
Abstract prompt interface.

All user interaction in the pipeline goes through a PromptInterface so that:
  - The CLI, email, and agent frontends share the same pipeline logic.
  - Tests can inject a MockPromptInterface with pre-programmed answers.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class PromptInterface(ABC):

    @abstractmethod
    def ask(self, message: str, *, options: list[str] | None = None) -> str:
        """
        Display `message` and return the user's response as a string.

        If `options` is provided, the implementation should present them as a
        numbered list and only accept a valid choice (returning the chosen
        string, not its number).
        """

    @abstractmethod
    def inform(self, message: str) -> None:
        """Display a message that requires no response."""

    @abstractmethod
    def show_image(self, path: Path) -> None:
        """Display or open an image file for the user to review."""
