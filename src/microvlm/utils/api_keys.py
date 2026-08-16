"""Load API keys from a ``.env`` file and pass them explicitly to backends."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from microvlm.utils.paths import project_root


def load_api_keys(env_file: Path | None = None) -> dict[str, str | None]:
    """Return API keys from ``.env`` (never read inside teacher/judge classes).

    Args:
        env_file: Optional path to a dotenv file. Defaults to ``<root>/.env``.

    Returns:
        Mapping of key names to values (value may be None if unset).
    """

    path = env_file or (project_root() / ".env")
    if path.is_file():
        load_dotenv(path, override=False)
    return {
        "openai": os.environ.get("OPENAI_API_KEY"),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "google": os.environ.get("GOOGLE_API_KEY"),
    }
