from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from mathmodel_ai.core.config import get_settings
from mathmodel_ai.startup import (
    StartupPreflightError,
    print_startup_preflight,
    run_startup_preflight,
)

APP_FACTORY = "mathmodel_ai.main:create_app"


@dataclass(frozen=True)
class ServeConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mathmodel_ai",
        description="MathModel AI local developer commands.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser("serve", help="start the FastAPI backend")
    serve_parser.add_argument("--host", default="127.0.0.1", help="bind host")
    serve_parser.add_argument("--port", default=8000, type=_valid_port, help="bind port")
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="reload when Python source files change",
    )
    return parser


def serve(config: ServeConfig) -> int:
    settings = get_settings()
    try:
        report = run_startup_preflight(settings)
    except StartupPreflightError as exc:
        print(f"Startup preflight failed: {exc}", file=sys.stderr)
        return 2
    print_startup_preflight(report)
    if config.reload:
        uvicorn.run(
            APP_FACTORY,
            factory=True,
            host=config.host,
            port=config.port,
            reload=True,
            reload_dirs=[str(Path(__file__).resolve().parents[1])],
        )
    else:
        uvicorn.run(
            APP_FACTORY,
            factory=True,
            host=config.host,
            port=config.port,
            reload=False,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    if namespace.command == "serve":
        return serve(
            ServeConfig(
                host=str(namespace.host),
                port=int(namespace.port),
                reload=bool(namespace.reload),
            )
        )
    raise AssertionError("argparse accepted an unknown command")  # pragma: no cover


def _valid_port(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= value <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return value
