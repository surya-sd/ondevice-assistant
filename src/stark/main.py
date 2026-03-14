"""Stark CLI entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from stark import __version__
from stark.config import Settings
from stark.pipeline import Pipeline

app = typer.Typer(
    name="stark",
    help="Stark — on-device voice assistant for macOS (Apple Silicon).",
    add_completion=False,
)
console = Console()


def _setup_logging(dev: bool) -> None:
    level = logging.DEBUG if dev else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=dev, markup=True)],
    )
    # Silence noisy third-party loggers unless in dev mode
    if not dev:
        for name in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
            logging.getLogger(name).setLevel(logging.WARNING)


@app.command()
def run(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a custom YAML config file.",
        show_default=False,
    ),
    dev: bool = typer.Option(
        False,
        "--dev",
        help="Enable verbose debug output and per-stage timing.",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Print version and exit.",
        is_eager=True,
    ),
) -> None:
    """Start Stark and begin listening."""
    if version:
        console.print(f"stark {__version__}")
        raise typer.Exit()

    _setup_logging(dev)
    log = logging.getLogger(__name__)

    try:
        settings = Settings.load(config_path=config)
    except Exception as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    console.rule(f"[bold cyan]Stark v{__version__}[/bold cyan]")
    if dev:
        console.print("[dim]Dev mode — timing and debug logs enabled.[/dim]")

    pipeline = Pipeline(settings, dev=dev)

    try:
        pipeline.load()
        pipeline.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down.[/yellow]")
        raise typer.Exit()
    except Exception as exc:
        log.exception("Fatal error: %s", exc)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
