"""
quickymesh CLI — client entry point.

This script launches the interactive CLI, which talks to a running
quickymesh API server over HTTP.  To run quickymesh end-to-end you need
two processes:

    1.  The API server (owns the workers, broker, and pipeline state):
            python api_server.py

    2.  This CLI (the user-facing menu):
            python main.py

Both can run on the same machine (the default) or on different hosts — the
CLI talks to the server over ``http(s)``.

Usage
-----
    python main.py
    python main.py --server http://10.0.0.2:8000
    python main.py --server http://my-host:8000 --api-key my-token

Configuration precedence (first wins):
    --server / --api-key command-line flags
    QUICKYMESH_SERVER / QUICKYMESH_API_KEY environment variables
    token file at ~/.config/quickymesh/token   (XDG)
                   %APPDATA%/quickymesh/token  (Windows)
    built-in default server http://localhost:8000
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from src.logging_config import configure_logging

configure_logging()

from src.cli.client import QuickymeshClient
from src.cli.main import run_cli
from src.prompt_interface.cli import CLIPromptInterface

log = logging.getLogger(__name__)

_DEFAULT_SERVER = "http://localhost:8000"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="quickymesh CLI — connects to a running API server.",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("QUICKYMESH_SERVER", _DEFAULT_SERVER),
        help=(
            "Base URL of the quickymesh API server "
            f"(default: {_DEFAULT_SERVER}, env: QUICKYMESH_SERVER)"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("QUICKYMESH_API_KEY"),
        help=(
            "Bearer token for server auth.  Only needed if the server was "
            "started with --auth-file.  Falls back to the saved token file "
            "if unset.  (env: QUICKYMESH_API_KEY)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request HTTP timeout in seconds (default: 30)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ui = CLIPromptInterface()
    client = QuickymeshClient(
        base_url=args.server,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    try:
        run_cli(client, ui)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
