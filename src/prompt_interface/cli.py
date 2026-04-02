"""
Command-line implementation of PromptInterface.
"""

import os
import subprocess
import sys
from pathlib import Path

from .base import PromptInterface


class CLIPromptInterface(PromptInterface):

    def ask(self, message: str, *, options: list[str] | None = None) -> str:
        if options:
            self.inform(message)
            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt}")
            while True:
                raw = input("Enter number or text: ").strip()
                # Accept numeric shortcut
                if raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(options):
                        return options[idx]
                # Accept exact text match (case-insensitive)
                for opt in options:
                    if raw.lower() == opt.lower():
                        return opt
                print(f"  Please enter a number 1–{len(options)} or one of the listed options.")
        else:
            return input(f"{message}\n> ").strip()

    def inform(self, message: str) -> None:
        print(message)

    def show_image(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            print(f"[image not found: {path}]")
            return
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
